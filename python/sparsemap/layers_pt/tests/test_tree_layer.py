import torch

from .custom_gradcheck import gradcheck
from ..tree_layer import DependencyTreeFast


def test_fasttree_sparse_decode():
    torch.manual_seed(42)
    n_nodes = 6
    tree = DependencyTreeFast(n_nodes, max_iter=100)
    for _ in range(20):
        W = torch.randn(n_nodes * (n_nodes + 1),
                        dtype=torch.double,
                        requires_grad=True)
        res = gradcheck(tree, (W,), eps=1e-5, atol=1e-2)
        assert res


def test_meaning_sparse_decode():
    n_nodes = 4
    w = torch.zeros(n_nodes, n_nodes + 1, dtype=torch.double)
    w[2, 1] = 100
    tree = DependencyTreeFast(n_nodes)
    u = tree(w.view(-1))
    for config in tree.configurations:
        assert config[1 + 2] == 1


def test_fast_tree_ignores_diag():
    n_nodes = 4
    # w = torch.zeros(n_nodes, n_nodes + 1)
    w_init = torch.randn(n_nodes * (n_nodes + 1))

    tree = DependencyTreeFast(n_nodes)
    u = tree(w_init.view(-1))

    k = 0
    for m in range(1, n_nodes + 1):
        for h in range(0, n_nodes + 1):
            if h == m:
                w_init[k] = 0
            k += 1

    tree = DependencyTreeFast(n_nodes)
    u_zeroed = tree(w_init.view(-1))

    assert (u_zeroed - u).data.norm() < 1e-12
