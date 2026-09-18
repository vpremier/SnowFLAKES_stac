# /// script
# dependencies = [
#   "threadpoolctl",
#   "scipy",
# ]
# ///



import xarray
import numpy as np
from openeo.metadata import CubeMetadata
from openeo.udf import inspect
from training.collection import calculate_training_samples

REPRESENTATIVE_PIXEL_BAND_NAME = "representative"

def apply_metadata(metadata: CubeMetadata, context: dict) -> CubeMetadata:
    """Rename the bands by using apply metadata
    :param metadata: Metadata of the input data
    :param context: Context of the UDF
    :return: Renamed labels
    """
    # rename band labels
    return metadata.rename_labels(dimension="bands", target=[REPRESENTATIVE_PIXEL_BAND_NAME])



def apply_datacube(cube :xarray.DataArray, context) -> xarray.DataArray:

    inspect(data = cube.bands, message="Running representative pixel UDF")
    
    total_samples = context.get("total_samples", 500)

    solar_incidence_angle = cube.sel(bands="local_solar_incidence_angle").values.astype(float)
    
    # define solar incidence angle ranges
    max_SIA = np.nanmax(solar_incidence_angle)
    if max_SIA <= 90: 
        ranges = ((0, 20), (20, 45), (45, 70), (70, max_SIA))
    else:
        ranges = ((0, 20), (20, 45), (45, 70), (70, 90), (90, 180))

    # get a number of training proportional to the area belonging to that range 
    range_samples = calculate_training_samples(solar_incidence_angle, ranges, total_samples)

    curr_bands = cube.sel(bands=["B02", "B03", "B04", "B08", "B11"])
    curr_NDVI = cube.sel(bands="NDVI")

    curr_scene_valid = np.isnan(curr_NDVI.values.astype(float)) | np.isnan(solar_incidence_angle)

    # Calculate distance from snow_sure
    # snow_sure = (cube.sel(bands="NDSI").values.astype(float) > 0.6) & (
    #             cube.sel(bands="B08").values.astype(float) > 0.45 * 10000)
    # distance_from_snow = np.full_like(snow_sure, np.nan, dtype=np.float32)
    #
    # distance_from_snow[curr_scene_valid] = distance_transform_edt(~snow_sure)[curr_scene_valid]
    # distance_from_snow = np.nan_to_num(distance_from_snow, nan=np.nanmax(distance_from_snow))
    # distance_from_snow_normalized = (distance_from_snow - np.nanmin(distance_from_snow)) / (
    #         np.nanmax(distance_from_snow) - np.nanmin(distance_from_snow)
    # )
    #TODO following original code: actual distance score is not used at all, only the elevation mask has effect???

    values =  np.full_like(curr_NDVI, 0, dtype=np.float32)
    for curr_range, sample_count in range_samples.items():

        valid_angles = np.logical_and(~np.isnan(solar_incidence_angle),np.logical_and(solar_incidence_angle >= curr_range[0], solar_incidence_angle < curr_range[1]))
        mask = np.logical_or(curr_scene_valid, ~valid_angles)

        def apply_mask(values):
            if len(values.shape) == 3:
                masked = values.values[:, ~mask]
            else:
                masked = [values.values[~mask].astype(float)]

            return np.stack(masked, axis=-1)

        curr_NDVI_masked = apply_mask(curr_NDVI)

        #find representative pixels, if there are any
        if curr_NDVI_masked.shape[0] > 0:
            try:

                if curr_range[0] >= 90:
                    representative_pixels_mask_snow, representative_pixels_mask_noSnow = compute_representative_snow_pixels_high_range(
                        apply_mask(cube.sel(bands="NDSI")), apply_mask(curr_bands), apply_mask(cube.sel(bands="diff_B_NIR")),None,
                        apply_mask(cube.sel(bands="SI")), 1000)
                else:
                    representative_pixels_mask_snow, representative_pixels_mask_noSnow  = compute_representative_snow_pixels(
                                    apply_mask(cube.sel(bands="NDSI")), curr_NDVI_masked, apply_mask(curr_bands), None, apply_mask(cube.sel(bands="B03")), 1000)


                if representative_pixels_mask_snow.shape[0] > 0 or representative_pixels_mask_noSnow.shape[0] > 0:

                    if representative_pixels_mask_snow.shape[0] > 0:
                            values[~mask] += representative_pixels_mask_snow * 10

                    if representative_pixels_mask_noSnow.shape[0] > 0:
                            values[~mask] +=  representative_pixels_mask_noSnow * 4
            except Exception as e:
                inspect(data=str(e), message=f"Error while finding representative pixels for {curr_range} and count {sample_count} {curr_NDVI_masked.shape} {e} ")

    if(context.get("classify", False)):
        def reshape(v):
            return np.stack(v, axis=-1)
        try:
            snow_representative_pixel_bands = curr_bands.values[:, values == 10]
            #classify if there are snow pixels to begin with

            if snow_representative_pixel_bands.shape[1] > 0:
                class_array, normalizer, svm, training_array = model_training_svm(reshape(snow_representative_pixel_bands), reshape(curr_bands.values[:, values == 8]))

                SCF_map = classify(curr_bands.values, ~curr_scene_valid, normalizer, svm)
                values = SCF_map
        except Exception as e:
            inspect(data=str(e), message=f"Error during classification, not classifying pixels {e} {curr_bands.values[:,values==10].shape}  {curr_bands.values[:,values==8].shape}  ")

    cube.loc['B03'] = 0

    if values is not None and values.shape[0] > 0:
        cube.loc['B03'] = values

    bands_to_drop = list(cube.bands.values)
    bands_to_drop.remove("B03")

    cube = cube.drop_sel(bands=bands_to_drop)
    return cube



def hyp_disatance(svmModel, svmMatrix):
    return svmModel.decision_function(svmMatrix)

def classify(bands, valid_mask, normalizer, svm, Nprocesses = 8):

    min_score_ns = -1
    max_score_s = 1
    Image_array_to_classify = bands[:, valid_mask].transpose()
    Samples_to_classify = normalizer.transform(Image_array_to_classify)
    # Divide Samples_to_classify into blocks for parallel processing
    samplesBlocks = np.array_split(Samples_to_classify, Nprocesses, axis=0)
    # Calculate the score
    from concurrent.futures import ThreadPoolExecutor
    #use threadpoolexecutor, scikit learn will release the GIL
    with ThreadPoolExecutor(max_workers=Nprocesses) as executor:
        scoreImage_arrayBlocks = list(executor.map(lambda block: hyp_disatance(svm, block), samplesBlocks))

    scoreImage_array = np.concatenate(scoreImage_arrayBlocks, axis=0)
    Score_map = 255 * np.ones(np.shape(valid_mask)).astype(float)
    Score_map[valid_mask] = scoreImage_array.flatten()
    scoreImage_array[scoreImage_array < min_score_ns] = min_score_ns
    scoreImage_array[scoreImage_array > max_score_s] = max_score_s
    SCF_Image_array = (scoreImage_array * 50 + 50).astype('uint8')
    # Create the SCF map
    SCF_map = 255 * np.ones(np.shape(valid_mask))
    SCF_map[valid_mask] = SCF_Image_array.flatten()

    return SCF_map

def model_training_svm(snow_training, no_snow_training, gamma=None):
    from sklearn import preprocessing
    from sklearn.svm import SVC
    from sklearn.metrics.pairwise import rbf_kernel

    training_array = np.concatenate((snow_training, no_snow_training), axis=0)
    class_array = np.concatenate((np.ones(snow_training.shape[0]), np.zeros(no_snow_training.shape[0])), axis=0)
    # Rescale: standardization between 0 and 1
    normalizer = preprocessing.StandardScaler().fit(training_array)
    Samples_train_normalized = normalizer.transform(training_array)
    if gamma == None:
        # Gamma selection by examining the kernel
        gamma_range = np.logspace(-2, 2, 100)
        std_list = []
        for curr_gamma in gamma_range:
            rbf = rbf_kernel(Samples_train_normalized, Samples_train_normalized, gamma=curr_gamma)
            std_list.append(rbf.std())

        idx_max_std = np.argmax(std_list)
        best_gamma = gamma_range[idx_max_std]

        print('The best Gamma is: ' + str(best_gamma))

    else:
        best_gamma = gamma
    svm = SVC(C=2000000, kernel='rbf', gamma=best_gamma, probability=False,
              decision_function_shape='ovo', cache_size=8000)
    svm.fit(Samples_train_normalized, class_array)
    #pred = svm.predict(Samples_train_normalized)
    return class_array, normalizer, svm, training_array

def compute_representative_snow_pixels_high_range(curr_NDSI, curr_bands, curr_diff_B_NIR, curr_distance_idx, curr_shad_idx,
                                                  sample_count):
    """
    Original code in: https://github.com/bare92/SnowFLAKES/blob/main/training_collection_kmeans.py
    """
    representative_pixels_mask_snow = np.array([])
    representative_pixels_mask_noSnow = np.array([])
    diff_B_NIR_low_perc, diff_B_NIR_high_perc = np.percentile(curr_diff_B_NIR, [2, 95])
    shad_idx_low_perc, shad_idx_high_perc = np.percentile(curr_shad_idx, [2, 95])
    curr_diff_B_NIR_norm = np.clip(
        (curr_diff_B_NIR - diff_B_NIR_low_perc) / (diff_B_NIR_high_perc - diff_B_NIR_low_perc), 0, 1)
    curr_shad_idx_norm = np.clip((curr_shad_idx - shad_idx_low_perc) / (shad_idx_high_perc - shad_idx_low_perc),
                                 0, 1)
    curr_score_snow_shadow = curr_diff_B_NIR_norm - curr_shad_idx_norm
    threshold_shadow = np.percentile(curr_score_snow_shadow, 95)
    curr_valid_snow_mask_shadow = np.logical_and.reduce(
        (curr_score_snow_shadow >= threshold_shadow, curr_NDSI > 0.7)).flatten() #TODO add again , curr_distance_idx != 255
    if np.sum(curr_valid_snow_mask_shadow) > 10:
        representative_pixels_mask_snow = get_representative_pixels(curr_bands, curr_valid_snow_mask_shadow,
                                                                    sample_count=int(sample_count / 2), k=5,
                                                                    n_closest='auto')

    threshold_shadow_no_snow = np.percentile(curr_score_snow_shadow, 5)
    curr_valid_no_snow_mask_shadow = (curr_score_snow_shadow <= threshold_shadow_no_snow).flatten()

    if np.sum(curr_valid_no_snow_mask_shadow) > 10:
        representative_pixels_mask_noSnow = get_representative_pixels(curr_bands,
                                                                      curr_valid_no_snow_mask_shadow,
                                                                      sample_count=int(sample_count / 2), k=5,
                                                                      n_closest='auto') * 2

    return representative_pixels_mask_snow, representative_pixels_mask_noSnow

def compute_representative_snow_pixels(curr_NDSI, curr_NDVI, curr_bands, curr_distance_idx, curr_green, sample_count):
    """
    Original code: https://github.com/bare92/SnowFLAKES/blob/main/training_collection_kmeans.py
    """
    # Normalize indices and compute sun metric
    representative_pixels_mask_snow = np.array([])
    representative_pixels_mask_noSnow = np.array([])

    NDSI_low_perc, NDSI_high_perc = np.percentile(curr_NDSI[np.logical_not(np.isnan(curr_NDSI))], [1, 99])
    NDVI_low_perc, NDVI_high_perc = np.percentile(curr_NDVI[np.logical_not(np.isnan(curr_NDVI))], [1, 99])
    green_low_perc, green_high_perc = np.percentile(curr_green, [1, 99])
    curr_NDSI_norm = np.clip((curr_NDSI - NDSI_low_perc) / (NDSI_high_perc - NDVI_low_perc), 0, 1)
    curr_NDVI_norm = np.clip((curr_NDVI - NDVI_low_perc) / (NDVI_high_perc - NDVI_low_perc), 0, 1)
    curr_green_norm = np.clip((curr_green - green_low_perc) / (green_high_perc - green_low_perc), 0, 1)
    curr_score_snow_sun = curr_NDSI_norm - curr_NDVI_norm + curr_green_norm
    threshold = np.percentile(curr_score_snow_sun, 95)
    curr_valid_snow_mask = np.logical_and.reduce(
        (curr_score_snow_sun >= threshold, curr_NDSI > 0.7)).flatten() #TODO add again , curr_distance_idx != 255
    if np.sum(curr_valid_snow_mask) > 10:
        representative_pixels_mask_snow = get_representative_pixels(curr_bands, curr_valid_snow_mask,
                                                                    sample_count=int(sample_count / 2), k=5,
                                                                    n_closest='auto')
    curr_valid_no_snow_mask = (curr_NDSI < 0).flatten()
    if np.sum(curr_valid_no_snow_mask) > 10:
        representative_pixels_mask_noSnow = get_representative_pixels(curr_bands, curr_valid_no_snow_mask,
                                                                      sample_count=int(sample_count / 2), k=10,
                                                                      n_closest='auto') * 2
    return representative_pixels_mask_snow, representative_pixels_mask_noSnow



def get_representative_pixels(bands_data, valid_mask, sample_count=50, k='auto', n_closest='auto'):
    """
    Selects representative "no snow" pixels by clustering and distance to cluster centroids.
    Saves the output as a raster.

    See: https://github.com/bare92/SnowFLAKES/blob/main/training_collection_kmeans.py

    Parameters
    ----------
    bands_data : numpy.ndarray
        3D array (bands, height, width) containing spectral data for each band.
    valid_mask : numpy.ndarray
        2D mask of valid pixels for selection.
    k : int, optional
        Number of clusters for K-means, by default 5.
    n_closest : int, optional
        Number of closest pixels to each centroid to select, by default 5.

    Returns
    -------
    representative_pixels_mask : numpy.ndarray
        2D mask with representative pixels marked as 1.
    """
    from sklearn.cluster import KMeans
    from scipy.spatial import distance
    from sklearn.preprocessing import StandardScaler

    # Extract "valid" pixels for clustering
    valid_pixels = bands_data[valid_mask, :]  # Shape (pixels, bands)

    # Normalize the valid pixels
    scaler = StandardScaler()
    normalized_pixels = scaler.fit_transform(valid_pixels)

    # find optimal K
    if k == 'auto':
        k = find_optimal_k(normalized_pixels, max_k=10, method="elbow")
    if n_closest == 'auto':
        n_closest = int(sample_count / k)

    # Perform K-means clustering on "no snow" pixels
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(normalized_pixels)

    # Get cluster centroids and labels
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Initialize an empty mask for representative pixels
    representative_pixels_mask = np.zeros(valid_mask.shape, dtype='uint8')

    # Find the n_closest pixels to each centroid
    for cluster_idx in range(k):
        # Select pixels in the current cluster
        cluster_indices = np.where(labels == cluster_idx)[0]
        cluster_pixels = normalized_pixels[cluster_indices]

        # Compute distances to the centroid for these pixels
        distances = distance.cdist(cluster_pixels, [centroids[cluster_idx]], 'euclidean').flatten()

        # Get the indices of the n_closest pixels in the cluster
        closest_indices = np.argsort(distances)[:n_closest]

        # Map the closest indices back to the original image coordinates
        original_indices = np.argwhere(valid_mask)[cluster_indices]
        selected_pixels = original_indices[closest_indices]

        # Set these pixels in the representative mask
        representative_pixels_mask[selected_pixels] = 1

    return representative_pixels_mask


def find_optimal_k(data, max_k=10, method="elbow", random_state=42):
    """
    Find the optimal number of clusters using the Elbow or Silhouette method.

    See: https://github.com/bare92/SnowFLAKES/blob/main/training_collection_kmeans.py

    Parameters:
    - data (array-like): The dataset to cluster.
    - max_k (int): The maximum number of clusters to evaluate.
    - method (str): "elbow" for WCSS-based elbow method or "silhouette" for silhouette score.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - int: The optimal number of clusters.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    wcss = []  # Within-Cluster Sum of Squares
    silhouette_scores = []  # Silhouette Scores
    k_values = range(2, max_k + 1)  # Start from 2 clusters for silhouette

    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(data, kmeans.labels_))

    if method == "elbow":
        # Calculate second derivative to find the "elbow"
        wcss_diff = np.diff(wcss)
        wcss_diff2 = np.diff(wcss_diff)
        optimal_k = k_values[np.argmin(wcss_diff2) + 1]  # Offset for the diff
    elif method == "silhouette":
        # Choose k with the highest silhouette score
        optimal_k = k_values[np.argmax(silhouette_scores)]
    else:
        raise ValueError("Invalid method. Choose 'elbow' or 'silhouette'.")

    return optimal_k