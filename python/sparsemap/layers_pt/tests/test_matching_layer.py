from .. import matching_layer

import torch
from torch.autograd import gradcheck, Variable


def test_matching_sparse_decode():
    torch.manual_seed(0)
    n_rows = 3
    n_cols = 4

    for _ in range(20):
        matcher = matching_layer.MatchingSparseMarginals(max_iter=100)
        W = torch.randn(n_rows, n_cols)
        W = Variable(W, requires_grad=True)
        res = gradcheck(matcher, (W,), eps=1e-3,
                        atol=1e-3)
        assert res
