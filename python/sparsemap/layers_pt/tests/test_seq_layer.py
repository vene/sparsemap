from .. import seq_layer

import torch
from torch.autograd import gradcheck, Variable


def test_seq_sparse_decode():
    torch.manual_seed(2)
    n_vars = 4
    n_states = 3
    for _ in range(20):
        sequence_smap = seq_layer.SequenceSparseMarginals(max_iter=1000)
        unary = Variable(torch.randn(n_vars, n_states), requires_grad=True)
        additionals = Variable(torch.randn(2 * n_states +
                                           (n_vars - 1) * n_states ** 2),
                               requires_grad=True)
        res = gradcheck(sequence_smap, (unary, additionals), eps=1e-4, atol=1e-3)
        print(res)
        assert res


def test_seq_dist_sparse_decode():
    torch.manual_seed(42)
    n_vars = 4
    n_states = 3
    bandwidth = 3
    for _ in range(20):
        seq_dist_smap = seq_layer.SequenceDistanceSparseMarginals(bandwidth)
        unary = Variable(torch.randn(n_vars, n_states), requires_grad=True)
        additionals = Variable(torch.randn(1 + 4 * bandwidth),
                               requires_grad=True)
        res = gradcheck(seq_dist_smap, (unary, additionals), eps=1e-4, atol=1e-3)
        print(res)
        assert res
