from ..tree_layer import TreeSparseMarginalsFast

import torch
from torch.autograd import gradcheck, Variable



def test_fasttree_sparse_decode():
    torch.manual_seed(42)
    n_nodes = 5
    tsm = TreeSparseMarginalsFast(n_nodes, max_iter=1000)
    for _ in range(20):
        W = torch.randn(n_nodes, n_nodes + 1).view(-1)
        W = Variable(W, requires_grad=True)
        res = gradcheck(tsm, (W,), eps=1e-4,
                        atol=1e-3)
        print(res)
        assert res


def test_meaning_sparse_decode():
    n_nodes = 4
    w = torch.zeros(n_nodes, n_nodes + 1)
    w[2, 1] = 100
    w = Variable(w)
    tsm = TreeSparseMarginalsFast(n_nodes, verbose=3)
    u = tsm(w.view(-1))
    for config in tsm.status['active_set']:
        assert config[1 + 2] == 1


def test_fast_tree_ignores_diag():
    n_nodes = 4
    # w = torch.zeros(n_nodes, n_nodes + 1)
    w_init = torch.randn(n_nodes * (n_nodes + 1))

    w = Variable(w_init)
    tsm = TreeSparseMarginalsFast(n_nodes)
    u = tsm(w.view(-1))

    k = 0
    for m in range(1, n_nodes + 1):
        for h in range(0, n_nodes + 1):
            if h == m:
                w_init[k] = 0
            k += 1

    w = Variable(w_init)
    tsm = TreeSparseMarginalsFast(n_nodes)
    u_zeroed = tsm(w.view(-1))

    assert (u_zeroed - u).data.norm() < 1e-12
