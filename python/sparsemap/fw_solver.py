"""Generic SparseMAP solver via Frank-Wolfe (Conditional Gradient).

This file is not used in this repository for any pytorch hidden layers: the
Active Set implementation from AD3 is used there. This is just for reference
and research purposes."""

# Author: Vlad Niculae <vlad@vene.ro>
# License: BSD 3-clause

from collections import defaultdict

import numpy as np


class SparseMAPFW(object):

    def __init__(self, polytope, max_iter=100, tol=1e-6, variant="vanilla"):
        """Generic SparseMAP solver via Frank-Wolfe (Conditional Gradient).

        This solver is not as good as Active Set, but it is much easier to read
        and to implement.  We are providing this implementation to help
        researchers who want to experiment with SparseMAP solvers without having
        to implement their factors in C++ using AD3.

        Parameters
        ----------

        polytope: object,
            A user-supplied object implementing the following methods:
             - `polytope.vertex(y)`, given a hashable structure representation
               `y`, must return a tuple [m_y, n_y] of vectors encoding the
               unaries and additionals of structure y. (n_y can be empty.).
               This is the `y`th column of the matrices M and N in our paper.
             - `polytope.map(eta_u, eta_v)` returns the y that solves
               `argmax_y <m_y, eta_u> + <n_y, eta_v>`.

        max_iter: int,
            The number of FW iterations to run.

        variant: {'vanilla' | 'away-step' | 'pairwise'}
            FW variant to run. Pairwise seems to perform the best.

        tol: float,
            Tolerance in the Wolfe gap, for convergence.
        """
        self.polytope = polytope
        self.max_iter = max_iter
        self.variant = variant
        self.tol = tol

    def _reconstruct_guess(self, active_set):
        """Compute the current guess from the weights over the vertices:

            [u, v] = sum_{y in active_set} alpha[y] * [m_y, n_y]

        """
        u, v = [], []

        for y, alpha_y in active_set.items():
            m_y, n_y = self.polytope.vertex(y)
            u.append(alpha_y * m_y)
            v.append(alpha_y * n_y)

        return sum(u), sum(v)

    def obj(self, u, v, eta_u, eta_v):
        """SparseMAP objective: -<u, eta_u> - <v, eta_v> + .5 || u ||²
        """
        obj = (.5 * np.sum(u ** 2)
                - np.sum(u * eta_u)
                - np.sum(v * eta_v))

        return obj

    def grad(self, u, v, eta_u, eta_v):
        """Gradient of self.obj"""
        g_u = u - eta_u
        g_v = -eta_v

        return [g_u, g_v]

    def worst_atom(self, g_u, g_v, active_set):
        """Find argmax_{w in active_set} <g, a_w> """

        max_w = None
        max_m_w = None
        max_n_w = None
        max_score = -float('inf')

        for w in active_set:
            m_w, n_w = self.polytope.vertex(w)
            score_w = np.sum(g_u * m_w) + np.sum(g_v * n_w)

            if score_w > max_score:
                max_w = w
                max_m_w = m_w
                max_n_w = n_w
                max_score = score_w

        return max_w, max_m_w, max_n_w

    def solve(self, eta_u, eta_v, full_path=False):

        eta_u = np.asarray(eta_u, dtype=np.float)
        eta_v = np.asarray(eta_v, dtype=np.float)

        y0 = self.polytope.map_oracle(eta_u, eta_v)
        active_set = defaultdict(float)
        active_set[y0] = 1

        objs = []
        size = [1]

        for it in range(1, self.max_iter):

            u, v = self._reconstruct_guess(active_set)
            obj = self.obj(u, v, eta_u, eta_v)
            objs.append(obj)

            # find forward direction
            g_u, g_v = self.grad(u, v, eta_u, eta_v)
            y = self.polytope.map_oracle(-g_u, -g_v)
            m_y, n_y = self.polytope.vertex(y)

            d_f_u = m_y - u
            d_f_v = n_y - v

            # compute forward gap
            gap_f = np.sum(-g_u * d_f_u) + np.sum(-g_v * d_f_v)

            if gap_f < self.tol:  # check convergence
                break

            if self.variant == "vanilla":
                # use forward direction
                d_u, d_v = d_f_u, d_f_v
                max_step = 1

            else:
                # for away-step and pairwise we need the "away" direction
                w, m_w, n_w = self.worst_atom(g_u, g_v, active_set)

                d_w_u = u - m_w
                d_w_v = v - n_w

                p_w = active_set[w]

                if self.variant == "pairwise":
                    d_u = d_f_u + d_w_u
                    d_v = d_f_v + d_w_v
                    max_step = p_w

                elif self.variant == "away-step":
                    gap_w = np.sum(-g_u * d_w_u) + np.sum(-g_v * d_w_v)

                    if gap_f >= gap_w:
                        d_u = d_f_u
                        d_v = d_f_v
                        max_step = 1
                    else:
                        d_u = d_w_u
                        d_v = d_w_v

                        p = active_set[w]
                        max_step = p_w / (1 - p_w)

                else:
                    raise ValueError("invalid variant")

            # compute step size by line search
            gamma = np.sum(-g_u * d_u) + np.sum(-g_v * d_v)
            gamma /= np.sum(d_u ** 2)
            gamma = max(0, min(gamma, max_step))

            # update convex combinaton coefficients
            if self.variant == "pairwise":
                active_set[w] -= gamma
                active_set[y] += gamma

            else:  # forward or away_step
                which = y

                if self.variant == "away-step" and gap_f < gap_w:
                    # if we took an away step, flip the update
                    gamma *= -1
                    which = w

                for y_ in active_set:
                    active_set[y_] *= (1 - gamma)
                active_set[which] += gamma

            # clean up zeros to speed up away-step searches
            zeros = [y_ for y_, p in active_set.items() if p == 0]
            for y_ in zeros:
                active_set.pop(y_)

            # sanity checks
            assert all(p > 0 for p in active_set.values())
            assert np.abs(1 - sum(active_set.values())) <= 1e-6

            size.append(len(active_set))

        u, v = self._reconstruct_guess(active_set)
        obj = self.obj(u, v, eta_u, eta_v)
        objs.append(obj)

        # assert objective always decreases
        assert np.all(np.diff(objs) <= 1e-6)

        if full_path:
            return u, v, active_set, objs, size
        else:
            return u, v, active_set
