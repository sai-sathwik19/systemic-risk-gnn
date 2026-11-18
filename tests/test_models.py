import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from sysrisk.dataset import data_to_dense, generate_dataset, network_to_data
from sysrisk.models import build_model, model_spec
from sysrisk.train import predict_gnn, train_gnn

GNNS = ["g1_paper", "g2_en", "g2_en_mean", "g2_en_raw", "g2_en_untied"]


def permuted(d, perm):
    L, x = data_to_dense(d)
    return network_to_data(L[np.ix_(perm, perm)], x[perm])


@pytest.mark.parametrize("name", GNNS)
def test_equivariance_and_invariance(name):
    torch.manual_seed(0)
    model = build_model(model_spec(name)).eval()
    d = generate_dataset({"topology": "cp", "n": 30}, 1, seed=0)[0]
    perm = np.random.default_rng(0).permutation(30)
    dp = permuted(d, perm)
    with torch.no_grad():
        o, op = model(d, T=10), model(dp, T=10)
    assert float(o["S"]) == pytest.approx(float(op["S"]), rel=1e-4, abs=1e-4)
    node_key = "r" if "r" in o else "node"
    np.testing.assert_allclose(op[node_key].numpy(), o[node_key].numpy()[perm], atol=1e-4)


@pytest.mark.parametrize("name", GNNS)
def test_variable_size_batches(name):
    model = build_model(model_spec(name)).eval()
    ds = generate_dataset({"topology": "er", "n": [15, 35], "size_mode": "fixed_degree", "mean_degree": 6}, 6, seed=1)
    b = Batch.from_data_list(ds)
    with torch.no_grad():
        out = model(b)
        single = [float(model(d)["S"]) for d in ds]
    assert out["S"].shape == (6,) and torch.isfinite(out["S"]).all()
    np.testing.assert_allclose(out["S"].numpy(), single, rtol=1e-4, atol=1e-4)


def test_g2_shortfall_bounds():
    model = build_model(model_spec("g2_en")).eval()
    d = generate_dataset("er", 1, seed=2)[0]
    with torch.no_grad():
        out = model(d, T=32)
    assert 0.0 <= float(out["S_rel"]) <= 1.0
    assert ((out["r"] >= 0) & (out["r"] <= 1)).all()


@pytest.mark.parametrize("name", ["g1_paper", "g2_en"])
def test_training_reduces_loss(name):
    torch.manual_seed(0)
    train = generate_dataset({"topology": "er", "n": 30, "size_mode": "fixed_degree", "mean_degree": 8}, 64, seed=3)
    model = build_model(model_spec(name))
    model.fit_normalizer(train)
    hist = train_gnn(model, train, train, lr=3e-3, epochs=15, batch_size=16, patience=100, verbose=False)
    assert hist["val"][-1] < hist["val"][0]
    pred = predict_gnn(model, train)
    assert pred["S"].shape == (64,)
