import numpy as np
import pytest
import torch

from sysrisk.clearing import (clearing_fda, en_map, k_step_payments, picard_dense, picard_sparse,
                              relative_liabilities)
from sysrisk.generators import sample_network


def random_networks(k=6, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(k):
        topo = ["er", "cp", "cpf"][i % 3]
        spec = {"topology": topo, "n": int(rng.choice([20, 50, 100]))}
        if topo == "er" and i % 2:
            spec["weights"] = "lognormal"
        L, x, _ = sample_network(spec, rng)
        out.append((L, x))
    return out


def test_hand_computed_chain():
    # A owes B 10, B owes C 10; x = (4, 3, 0.5).  A pays 4, B pays 3 + 4 = 7 -> S = 6 + 3 = 9, two rounds.
    L = np.array([[0, 10, 0], [0, 0, 10], [0, 0, 0]], dtype=float)
    res = clearing_fda(L, np.array([4.0, 3.0, 0.5]))
    np.testing.assert_allclose(res.p, [4.0, 7.0, 0.0])
    assert res.S == pytest.approx(9.0)
    assert res.rounds == 2
    assert res.default.tolist() == [True, True, False]


def cascade(N, start):
    """Paper 2, Sec. 6.1: each node j != start-1 owes N to j+1 (cyclically); all assets 1."""
    L = np.zeros((N, N))
    for j in range(N):
        if j != (start - 1) % N:
            L[j, (j + 1) % N] = N
    return L, np.ones(N)


def star(N, center):
    L = np.zeros((N, N))
    for j in range(N):
        if j != center:
            L[j, center] = 2
    return L, np.ones(N)


@pytest.mark.parametrize("N", [10, 20, 50, 100])
def test_stylised_networks_from_paper(N):
    S_cascade = [clearing_fda(*cascade(N, i)).S for i in range(N)]
    S_star = [clearing_fda(*star(N, i)).S for i in range(N)]
    np.testing.assert_allclose(S_cascade, N * (N - 1) / 2)
    np.testing.assert_allclose(S_star, N - 1)
    # paper 2 reports the mean "no bailout" shortfall [N(N-1)/2 + (N-1)] / 2, e.g. 27 at N = 10
    assert np.mean(S_cascade + S_star) == pytest.approx((N * (N - 1) / 2 + (N - 1)) / 2)


def test_cascade_depth():
    res = clearing_fda(*cascade(10, 0))
    assert res.rounds == 9  # contagion travels down the whole chain


@pytest.mark.parametrize("L,x", random_networks())
def test_fda_properties(L, x):
    res = clearing_fda(L, x)
    Pi, pbar = relative_liabilities(L)
    p = res.p
    # fixed point, bounds
    assert np.abs(p - en_map(L, x, p)).max() < 1e-9 * max(1.0, pbar.max())
    assert (p >= -1e-12).all() and (p <= pbar + 1e-12).all()
    # equity conservation (no external liabilities): sum_i (x_i + (Pi^T p)_i - p_i) = sum_i x_i
    assert (x + Pi.T @ p - p).sum() == pytest.approx(x.sum(), rel=1e-9)
    # Picard (dense, from pbar and from 0) agrees
    pd_ = picard_dense(torch.tensor(L), torch.tensor(x), start="pbar", tol=1e-13).p.numpy()
    p0 = picard_dense(torch.tensor(L), torch.tensor(x), start="zero", tol=1e-13).p.numpy()
    np.testing.assert_allclose(pd_, p, atol=1e-7 * max(1.0, pbar.max()))
    np.testing.assert_allclose(p0, p, atol=1e-7 * max(1.0, pbar.max()))


@pytest.mark.parametrize("L,x", random_networks(seed=1))
def test_invariances(L, x):
    S = clearing_fda(L, x).S
    # homogeneity
    assert clearing_fda(3.7 * L, 3.7 * x).S == pytest.approx(3.7 * S, rel=1e-9, abs=1e-9)
    # permutation invariance of S, equivariance of p
    perm = np.random.default_rng(0).permutation(len(x))
    res_p = clearing_fda(L[np.ix_(perm, perm)], x[perm])
    assert res_p.S == pytest.approx(S, rel=1e-9, abs=1e-9)
    np.testing.assert_allclose(res_p.p, clearing_fda(L, x).p[perm], atol=1e-8)
    # monotone in external assets
    assert clearing_fda(L, x + 0.5).S <= S + 1e-9


@pytest.mark.parametrize("L,x", random_networks(seed=2))
def test_k_step_lower_bounds(L, x):
    _, pbar = relative_liabilities(L)
    S = clearing_fda(L, x).S
    SK = [float((pbar - k_step_payments(L, x, K)).sum()) for K in range(0, 8)]
    assert SK[0] == 0.0
    assert all(b >= a - 1e-9 for a, b in zip(SK, SK[1:]))
    assert SK[-1] <= S + 1e-9
    a = L.sum(0)
    assert SK[1] == pytest.approx(np.maximum(0, pbar - x - a).sum())


def test_sparse_matches_dense():
    for L, x in random_networks(k=3, seed=3):
        src, dst = np.nonzero(L)
        ei = torch.tensor(np.stack([src, dst]))
        sp = picard_sparse(ei, torch.tensor(L[src, dst]), torch.tensor(x), tol=1e-13).p.numpy()
        np.testing.assert_allclose(sp, clearing_fda(L, x).p, atol=1e-7 * max(1.0, L.sum(1).max()))
        for K in (1, 3):
            spk = picard_sparse(ei, torch.tensor(L[src, dst]), torch.tensor(x), n_iter=K).p.numpy()
            np.testing.assert_allclose(spk, k_step_payments(L, x, K), atol=1e-10)
