# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3-clause


import numpy as np
from numpy.testing import assert_allclose
import pytest

from ..fw_solver import SparseMAPFW


@pytest.mark.parametrize('variant', ('vanilla', 'pairwise', 'away-step'))
def test_pairwise_factor(variant):

    class PairwiseFactor(object):
        """A factor with two binary variables and a coupling between them."""

        def vertex(self, y):

            # y is a tuple (0, 0), (0, 1), (1, 0) or (1, 1)
            u = np.array(y, dtype=np.float)
            v = np.atleast_1d(np.prod(u))
            return u, v

        def map_oracle(self, eta_u, eta_v):

            best_score = -np.inf
            best_y = None
            for x1 in (0, 1):
                for x2 in (0, 1):
                    y = (x1, x2)
                    u, v = self.vertex(y)

                    score = np.dot(u, eta_u) + np.dot(v, eta_v)
                    if score > best_score:
                        best_score = score
                        best_y = y
            return best_y

        def qp(self, eta_u, eta_v):
            """Prop 6.5 in Andre Martins' thesis"""

            c1, c2, c12 = eta_u[0], eta_u[1], eta_v[0]

            flip_sign = False
            if c12 < 0:
                flip_sign = True
                c1, c2, c12 = c1 + c12, 1 - c2, -c12

            if c1 > c2 + c12:
                u = [c1, c2 + c12]
            elif c2 > c1 + c12:
                u = [c1 + c12, c2]
            else:
                uu = (c1 + c2 + c12) / 2
                u = [uu, uu]

            u = np.clip(np.array(u), 0, 1)
            v = np.atleast_1d(np.min(u))

            if flip_sign:
                u[1] = 1 - u[1]
                v[0] = u[0] - v[0]

            return u, v

    pw = PairwiseFactor()
    fw = SparseMAPFW(pw, max_iter=10000, tol=1e-12, variant=variant)

    params = [
        (np.array([0, 0]), np.array([0])),
        (np.array([100, 0]), np.array([0])),
        (np.array([0, 100]), np.array([0])),
        (np.array([100, 0]), np.array([-100])),
        (np.array([0, 100]), np.array([-100]))
    ]

    rng = np.random.RandomState(0)
    for _ in range(20):
        eta_u = rng.randn(2)
        eta_v = rng.randn(1)
        params.append((eta_u, eta_v))

    for eta_u, eta_v in params:

        u, v, active_set = fw.solve(eta_u, eta_v)
        ustar, vstar = pw.qp(eta_u, eta_v)

        uv = np.concatenate([u, v])
        uvstar = np.concatenate([ustar, vstar])

        assert_allclose(uv, uvstar, atol=1e-10)


@pytest.mark.parametrize('variant', ('vanilla', 'pairwise', 'away-step'))
@pytest.mark.parametrize('k', (1, 4, 20))
def test_xor(variant, k):
    class XORFactor(object):
        """A one-of-K factor"""

        def __init__(self, k):
            self.k = k

        def vertex(self, y):
            # y is an integer between 0 and k-1
            u = np.zeros(k)
            u[y] = 1
            v = np.array(())

            return u, v

        def map_oracle(self, eta_u, eta_v):
            return np.argmax(eta_u)

        def qp(self, eta_u, eta_v):
            """Projection onto the simplex"""
            z = 1
            v = np.array(eta_u)
            n_features = v.shape[0]
            u = np.sort(v)[::-1]
            cssv = np.cumsum(u) - z
            ind = np.arange(n_features) + 1
            cond = u - cssv / ind > 0
            rho = ind[cond][-1]
            theta = cssv[cond][-1] / float(rho)
            uu = np.maximum(v - theta, 0)
            vv = np.array(())
            return uu, vv

    xor = XORFactor(k)
    fw = SparseMAPFW(xor, max_iter=10000, tol=1e-12, variant=variant)

    params = [np.zeros(k), np.ones(k), np.full(k, -1)]

    rng = np.random.RandomState(0)
    for _ in range(20):
        eta_u = rng.randn(k)
        params.append(eta_u)

    for eta_u in params:

        # try different ways of supplying empty eta_v
        for eta_v in (np.array(()), [], 0, None):

            u, v, active_set = fw.solve(eta_u, eta_v)
            ustar, vstar = xor.qp(eta_u, eta_v)

            uv = np.concatenate([u, v])
            uvstar = np.concatenate([ustar, vstar])

            assert_allclose(uv, uvstar, atol=1e-10)
