import numpy as np

from geoai_prospector.features.geochem import close, helmert_basis, ilr, ilr_inv, multiplicative_replacement, pivot


def test_helmert_basis_is_orthonormal_and_in_clr_plane():
    for D in (3, 5, 10):
        V = helmert_basis(D)
        assert np.allclose(V.T @ V, np.eye(D - 1))
        assert np.allclose(V.sum(0), 0)


def test_ilr_round_trip():
    rng = np.random.default_rng(0)
    C = close(rng.lognormal(3, 1, (50, 6)))
    assert np.allclose(ilr_inv(ilr(C)), C)


def test_ilr_is_scale_invariant():
    rng = np.random.default_rng(1)
    C = rng.lognormal(0, 1, (20, 5))
    a = ilr(C / C.sum(1, keepdims=True))
    b = ilr(7.3 * C / (7.3 * C).sum(1, keepdims=True))
    assert np.allclose(a, b)


def test_ilr_is_an_isometry():
    """Aitchison distance between compositions == Euclidean distance between ILR coordinates."""
    rng = np.random.default_rng(2)
    x, y = close(rng.lognormal(0, 1, (1, 6))), close(rng.lognormal(0, 1, (1, 6)))
    lx, ly = np.log(x), np.log(y)
    clr_x, clr_y = lx - lx.mean(), ly - ly.mean()
    aitchison = np.linalg.norm(clr_x - clr_y)
    assert np.isclose(aitchison, np.linalg.norm(ilr(x) - ilr(y)))


def test_multiplicative_replacement_keeps_ratios_of_detected_parts():
    X = np.array([[100.0, np.nan, 50.0]])
    cens = np.array([[False, True, False]])
    dl = np.array([[0.0, 2.0, 0.0]])
    R = multiplicative_replacement(X, cens, dl, factor=0.65)
    assert np.isclose(R[0, 1], 1.3)
    assert np.isclose(R[0, 0] / R[0, 2], 2.0)


def test_pivot_positive_when_enriched():
    C = close(np.array([[1000.0, 10, 10], [1.0, 10, 10]]))
    p = pivot(C, 0)
    assert p[0] > 0 > p[1]
