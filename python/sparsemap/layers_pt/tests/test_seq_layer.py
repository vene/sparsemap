import torch

from .custom_gradcheck import gradcheck
from .. import seq_layer


def test_seq_sparse_decode():
    torch.manual_seed(2)
    n_vars = 4
    n_states = 3
    for _ in range(20):
        seq = seq_layer.Sequence(max_iter=1000)
        unary = torch.randn(n_vars, n_states, dtype=torch.double, requires_grad=True)
        additionals = torch.randn(2 * n_states + (n_vars - 1) * n_states ** 2,
                                  dtype=torch.double,
                                  requires_grad=True)
        res = gradcheck(seq, (unary, additionals), eps=1e-4, atol=1e-4)
        assert res


def test_seq_dist_sparse_decode():
    torch.manual_seed(42)
    n_vars = 4
    n_states = 3
    bandwidth = 3
    for _ in range(20):
        seq = seq_layer.SequenceDistance(bandwidth, max_iter=1000)
        unary = torch.randn(n_vars, n_states,
                            dtype=torch.double,
                            requires_grad=True)
        additionals = torch.randn(1 + 4 * bandwidth,
                                  dtype=torch.double,
                                  requires_grad=True)
        res = gradcheck(seq, (unary, additionals), eps=1e-4, atol=1e-3)
        assert res
