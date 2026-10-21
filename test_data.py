from diagml.data import data_fingerprint, feature_names, load_data, split


def test_load_data_shape():
    X, y = load_data()
    assert X.shape == (569, 30)
    assert set(y.unique()) == {0, 1}
    assert not X.isna().any().any()


def test_split_is_stratified_and_deterministic():
    X, y = load_data()
    a = split(X, y, 0.2, seed=1)
    b = split(X, y, 0.2, seed=1)
    assert a[0].index.equals(b[0].index)
    assert abs(a[2].mean() - a[3].mean()) < 0.03  # class balance preserved


def test_fingerprint_stable():
    X, y = load_data()
    assert data_fingerprint(X, y) == data_fingerprint(X, y)
    assert len(feature_names()) == 30
