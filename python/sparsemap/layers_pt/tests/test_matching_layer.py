import torch

from .custom_gradcheck import gradcheck
from .. import matching_layer


def test_matching_sparse_decode():
    torch.manual_seed(0)
    n_rows = 3
    n_cols = 4

    for _ in range(20):
        matcher = matching_layer.Matching(max_iter=100)
        W = torch.randn(n_rows, n_cols, dtype=torch.double, requires_grad=True)
        res = gradcheck(matcher, (W,), eps=1e-3, atol=1e-5)
        assert res
