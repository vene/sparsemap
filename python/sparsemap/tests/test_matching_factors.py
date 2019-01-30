import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.optimize import linear_sum_assignment
from scipy import sparse as sp

from .._factors import PFactorMatching, PFactorMatchingSparse
from .._sparsemap import sparsemap_fwd


def generate_array(n_rows, n_cols, random_state):

    X = random_state.uniform(size=(n_rows, n_cols))

    # zero out one element in each row to ensure sparsity
    ind = random_state.randint(0, n_cols, size=n_rows)
    X[np.arange(n_rows), ind] = 0

    return X


@pytest.mark.parametrize('n_rows', (4, 3, 5))
def test_map_dense(n_rows, n_cols=4, seed=42):
    rng = np.random.RandomState(seed)
    X = generate_array(n_rows, n_cols, rng)

    S = X.copy()
    S[S == 0] = 9999
    row_ix, col_ix = linear_sum_assignment(S)

    u_gold = np.zeros((n_rows, n_cols))
    u_gold[row_ix, col_ix] = 1

    factor_dense = PFactorMatching()
    factor_dense.initialize(n_rows, n_cols)
    logp = -S.ravel()
    val, u, _ = factor_dense.solve_map(logp, [])
    u = np.array(u).reshape(n_rows, n_cols)

    assert_allclose(u_gold, u)


@pytest.mark.parametrize('n_rows', (4,))
def test_map_sparse(n_rows, n_cols=4, seed=42):
    rng = np.random.RandomState(seed)

    for _ in range(20):
        X = generate_array(n_rows, n_cols, rng)

        S = X.copy()
        S[S == 0] = 9999
        row_ix, col_ix = linear_sum_assignment(S)

        u_gold = np.zeros((n_rows, n_cols))
        u_gold[row_ix, col_ix] = 1

        val_gold = -np.dot(X.ravel(), u_gold.ravel())

        Xsp = sp.csr_matrix(X)
        factor_sparse = PFactorMatchingSparse()
        factor_sparse.initialize(Xsp)
        logp = -Xsp.data
        val, u, _ = factor_sparse.solve_map(logp, [])

        assert_allclose(val_gold, val)

        U = Xsp.copy()
        U.data[:] = u
        assert_allclose(u_gold, U.toarray())


@pytest.mark.parametrize('n_rows', (4,))
def test_sparsemap(n_rows, n_cols=4, seed=42):
    rng = np.random.RandomState(seed)

    for _ in range(20):
        X = generate_array(n_rows, n_cols, rng)
        S = X.copy()
        S[S == 0] = 9999
        Xsp = sp.csr_matrix(X)

        factor_dense = PFactorMatching()
        factor_dense.initialize(*Xsp.shape)
        u_dense, _, _ = sparsemap_fwd(factor_dense, -S.ravel(), [])
        U_dense = u_dense.reshape(*X.shape)

        factor_sparse = PFactorMatchingSparse()
        factor_sparse.initialize(Xsp)
        u_sparse, _, _ = sparsemap_fwd(factor_sparse, -Xsp.data, [])
        U_sparse = Xsp.copy()
        U_sparse.data[:] = u_sparse

        assert_allclose(U_dense, U_sparse.toarray())
