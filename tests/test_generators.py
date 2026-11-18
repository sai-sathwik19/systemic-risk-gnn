import numpy as np
import pytest

from sysrisk.dataset import data_to_dense, generate_dataset, network_to_data
from sysrisk.generators import er_edge_prob, make_config, sample_network


@pytest.mark.parametrize("topo", ["er", "cp", "cpf"])
def test_basic_shapes(topo):
    L, x, meta = sample_network({"topology": topo}, np.random.default_rng(0))
    assert L.shape == (100, 100) and x.shape == (100,)
    assert np.all(np.diag(L) == 0) and (L >= 0).all() and (x > 0).all()


def test_er_density_and_unit_weights():
    L, x, _ = sample_network("er", np.random.default_rng(1))
    assert set(np.unique(L)) <= {0.0, 1.0}
    assert abs((L > 0).sum() / (100 * 99) - 0.4) < 0.02
    assert x.max() <= 10.0


def test_fixed_degree_mode_keeps_degree():
    cfg = make_config({"topology": "er", "size_mode": "fixed_degree"})
    for n in (50, 200, 1000):
        assert er_edge_prob(cfg, n) * (n - 1) == pytest.approx(0.4 * 99)
    L, _, _ = sample_network({"topology": "er", "n": 400, "size_mode": "fixed_degree"}, np.random.default_rng(2))
    assert abs((L > 0).sum(1).mean() - 39.6) < 2.0


def test_cp_structure_and_shuffle():
    L, x, meta = sample_network({"topology": "cp", "shuffle": False}, np.random.default_rng(3))
    g = meta["group"]
    assert (g[:10] == 0).all() and (g[10:] == 1).all()
    assert set(np.unique(L[:10, :10])) <= {0.0, 10.0}
    assert set(np.unique(L[10:, 10:])) <= {0.0, 1.0}
    assert x[:10].mean() > x[10:].mean()  # core assets scaled by 50 vs 10


def test_cpf_shares_structure():
    L1, x1, _ = sample_network("cpf", np.random.default_rng(10))
    L2, x2, _ = sample_network("cpf", np.random.default_rng(11))
    np.testing.assert_array_equal(L1, L2)
    assert not np.allclose(x1, x2)


def test_determinism_and_roundtrip():
    a = generate_dataset({"topology": "cp", "n": 30}, 3, seed=5)
    b = generate_dataset({"topology": "cp", "n": 30}, 3, seed=5)
    for da, db in zip(a, b):
        assert float(da.y) == float(db.y)
        L, x = data_to_dense(da)
        d2 = network_to_data(L, x)
        assert float(d2.y) == pytest.approx(float(da.y), rel=1e-5)


def test_mixed_sizes():
    ds = generate_dataset({"topology": "er", "n": [20, 40], "size_mode": "fixed_degree", "mean_degree": 8}, 20, seed=0)
    assert {int(d.n) for d in ds} == {20, 40}
