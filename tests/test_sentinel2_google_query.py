from data_download import sentinel2_google_query


def test_normalize_tile():
    assert sentinel2_google_query.normalize_tile('19hde') == 'T19HDE'
    assert sentinel2_google_query.normalize_tile('MGRS-19HDE') == 'T19HDE'


def test_google_product_urls():
    l1c = 'S2B_MSIL1C_20230324T143729_N0509_R096_T19HDE_20230324T193724'
    l2a = 'S2A_MSIL2A_20220111T021351_N0301_R060_T50HLK_20220111T041611'

    l1c_url, l1c_metadata = sentinel2_google_query.get_google_product_url(l1c)
    l2a_url, l2a_metadata = sentinel2_google_query.get_google_product_url(l2a)

    assert '/tiles/19/H/DE/' in l1c_url
    assert l1c_metadata == 'MTD_MSIL1C.xml'
    assert '/L2/tiles/50/H/LK/' in l2a_url
    assert l2a_metadata == 'MTD_MSIL2A.xml'


def test_query_returns_only_matching_google_products(monkeypatch):
    matching = 'S2B_MSIL1C_20230324T143729_N0509_R096_T19HDE_20230324T193724'
    outside_date = 'S2B_MSIL1C_20230325T143729_N0509_R096_T19HDE_20230325T193724'

    monkeypatch.setattr(
        sentinel2_google_query,
        'list_tile_products',
        lambda tile, collection, session: [matching, outside_date],
    )
    monkeypatch.setattr(
        sentinel2_google_query,
        'get_cloud_cover',
        lambda product_id, session: 3.5,
    )

    products = sentinel2_google_query.query_google_sentinel2(
        '2023-03-24',
        '2023-03-25',
        tile='T19HDE',
        max_cc=10,
    )

    assert products['Name'].tolist() == [matching + '.SAFE']
    assert products['cloudCover'].tolist() == [3.5]
    assert products['tile'].tolist() == ['T19HDE']
