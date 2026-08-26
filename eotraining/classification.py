# Simplified SVM classifier.
import numpy as np
from rasterio.transform import rowcol
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def SVM(data_stack, training_samples, bands=None):
    """Train an SVM using only reflectance bands and return SCF (%)."""
    data_stack = data_stack.squeeze(drop=True)
    if set(data_stack.dims) != {"band", "y", "x"}:
        raise ValueError(f"Expected dimensions band/y/x, got {data_stack.dims}")
    if bands is None:
        bands = [str(band) for band in data_stack.band.values]
    data = np.asarray(data_stack.sel(band=bands).values, dtype="float32")
    height, width = data.shape[-2:]
    image_features = np.moveaxis(data, 0, -1).reshape(-1, len(bands))

    training_samples = training_samples.to_crs(data_stack.rio.crs)
    rows, cols = rowcol(
        data_stack.rio.transform(),
        training_samples.geometry.x.to_numpy(),
        training_samples.geometry.y.to_numpy(),
    )
    rows, cols = np.asarray(rows), np.asarray(cols)
    inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    labels = training_samples["class"].to_numpy()[inside]
    sample_features = data[:, rows[inside], cols[inside]].T
    valid_samples = np.isfinite(sample_features).all(axis=1)
    if set(np.unique(labels[valid_samples])) != {1, 2}:
        raise ValueError("Training points must contain both classes 1 (snow) and 2 (snow-free).")

    classifier = make_pipeline(
        StandardScaler(),
        SVC(C=2_000_000, kernel="rbf", probability=True, random_state=0),
    )
    classifier.fit(sample_features[valid_samples], labels[valid_samples])

    valid_pixels = np.isfinite(image_features).all(axis=1)
    scf = np.full(image_features.shape[0], np.nan, dtype="float32")
    snow_index = list(classifier.classes_).index(1)
    scf[valid_pixels] = classifier.predict_proba(image_features[valid_pixels])[:, snow_index] * 100
    scf = scf.reshape(height, width)

    return scf, classifier