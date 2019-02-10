import numpy as np

import torch

from .. import sparsemap


def _Z_from_inv(inv):
    """

    active set maintains the inverse :

    inv = [0, 1.T; 1, MtM] ^ -1

    we recover Z = (MtM)^{-1} by Sherman-Morrison-Woodbury
    """

    Z = inv[1:, 1:]
    k = inv[0, 0]
    b = inv[0, 1:].unsqueeze(0)

    Z -= (1 / k) * (b * b.t())
    return Z


def _d_vbar(M, dy, inv):

    Z = _Z_from_inv(inv)

    # B = S11t / 1S1t
    # dvbar = (I - B) S M dy

    # we first compute S M dy
    first_term = Z @ (M @ dy)
    # then, BSMt dy = B * first_term. Optimized:
    # 1S1t = S.sum()
    # S11tx = (S1) (1t * x)
    second_term = (first_term.sum() * Z.sum(0)) / Z.sum()
    d_vbar = first_term - second_term
    return d_vbar


def _from_np_like(X_np, Y_pt):
    X = torch.from_numpy(X_np)
    return torch.as_tensor(X, dtype=Y_pt.dtype, device=Y_pt.device)


class _BaseSparseMAP(torch.nn.Module):

    def __init__(self, max_iter=20, verbose=0):
        self.max_iter = max_iter
        self.verbose = verbose
        super(_BaseSparseMAP, self).__init__()

    def sparsemap(self, unaries, factor, additionals=None):
        if additionals is not None:
            return _SparseMAPAdd.apply(
                unaries,
                additionals,
                factor,
                self,
                self.max_iter,
                self.verbose)

        else:
            return _SparseMAP.apply(
                unaries,
                factor,
                self,
                self.max_iter,
                self.verbose)


class _SparseMAP(torch.autograd.Function):

    @staticmethod
    def forward(ctx, unaries, factor, caller=None, max_iter=20, verbose=0):

        u, v, status = sparsemap(factor, unaries.cpu(), [],
                                 max_iter=max_iter,
                                 verbose=verbose)

        u = _from_np_like(u, unaries)
        inv = _from_np_like(status['inverse'], unaries)
        M = _from_np_like(status['M'], unaries)

        if caller is not None:
            caller.distribution = status['distribution']
            caller.configurations = status['active_set']

        ctx.save_for_backward(inv, M)
        return u

    @staticmethod
    def backward(ctx, dy):

        inv, M = ctx.saved_tensors
        d_vbar = _d_vbar(M, dy, inv)
        d_unary = M.t() @ d_vbar

        return d_unary, None, None, None, None


class _SparseMAPAdd(torch.autograd.Function):
    """SparseMAP with additional inputs/outputs, as for a linear chain CRF"""

    @staticmethod
    def forward(ctx, unaries, additionals, factor, caller=None, max_iter=20,
                verbose=0):

        u, _, status = sparsemap(factor,
                                 unaries.cpu(),
                                 additionals.cpu(),
                                 max_iter=max_iter,
                                 verbose=verbose)

        u = _from_np_like(u, unaries)
        inv = _from_np_like(status['inverse'], unaries)
        M = _from_np_like(status['M'], unaries)
        N = _from_np_like(status['N'], unaries)

        if caller is not None:
            caller.distribution = status['distribution']
            caller.configurations = status['active_set']

        ctx.save_for_backward(inv, M, N)
        return u

    @staticmethod
    def backward(ctx, dy):

        inv, M, N = ctx.saved_tensors
        d_vbar = _d_vbar(M, dy, inv)

        d_u = M.t() @ d_vbar
        d_v = N.t() @ d_vbar
        return d_u, d_v, None, None, None, None
