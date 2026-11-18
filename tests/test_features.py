import numpy as np
import pytest

from sysrisk.clearing import k_step_payments
from sysrisk.dataset import data_to_dense, generate_dataset
from sysrisk.features import FEATURES_A, FEATURES_B, dataset_features, graph_features, gini


def test_feature_values():
    d = generate_dataset({"topology": "cp", "n": 40}, 1, seed=0)[0]
    L, x = data_to_dense(d)
    fa = dict(zip(FEATURES_A, graph_features(d, "A")))
    assert fa["N"] == 40 and fa["n_edges"] == (L > 0).sum()
    assert fa["pbar_sum"] == pytest.approx(L.sum(), rel=1e-6)
    assert fa["x_sum"] == pytest.approx(x.sum(), rel=1e-6)
    fb = dict(zip(FEATURES_B, graph_features(d, "B")))
    S1 = (L.sum(1) - k_step_payments(L, x, 1)).sum()
    assert fb["S1"] == pytest.approx(S1, rel=1e-5, abs=1e-6)


def test_shapes_and_finite():
    ds = generate_dataset("er", 4, seed=1)
    for fs, names in (("A", FEATURES_A), ("B", FEATURES_B)):
        X = dataset_features(ds, fs)
        assert X.shape == (4, len(names)) and np.isfinite(X).all()


def test_gini():
    assert gini(np.ones(10)) == pytest.approx(0.0)
    assert gini(np.r_[np.zeros(9), 1.0]) == pytest.approx(0.9)
