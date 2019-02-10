# pytorch layer that applies a TreeFactor

import numpy as np
import torch

from ad3 import PFactorGraph
from ad3.extensions import PFactorTree

from .base import _BaseSparseMAP
from .._factors import PFactorTreeFast


class DependencyTree(_BaseSparseMAP):

    def __init__(self, n_nodes, max_iter=20, verbose=0):
        self.n_nodes = n_nodes
        super(DependencyTree, self).__init__(max_iter, verbose)

    def forward(self, unaries):

        n_nodes = self.n_nodes
        arcs = [(h, m) for m in range(1, n_nodes + 1)
                       for h in range(n_nodes + 1)
                       if h != m]
        g = PFactorGraph()
        arc_vars = [g.create_binary_variable() for _ in arcs]
        tree = PFactorTree()
        g.declare_factor(tree, arc_vars)
        tree.initialize(n_nodes + 1, arcs)

        return self.sparsemap(unaries.view(-1), tree)


class DependencyTreeFast(DependencyTree):

    def forward(self, unaries):
        n_nodes = self.n_nodes
        arcs = [(h, m) for m in range(1, n_nodes + 1)
                       for h in range(n_nodes + 1)
                       if h != m]
        g = PFactorGraph()
        arc_vars = [g.create_binary_variable() for _ in arcs]
        tree = PFactorTreeFast()
        g.declare_factor(tree, arc_vars)
        tree.initialize(self.n_nodes + 1)
        return self.sparsemap(unaries.view(-1), tree)


if __name__ == '__main__':
    n_nodes = 3
    W = torch.randn((n_nodes + 1) * n_nodes, requires_grad=True)

    Wskip_a = []
    k = 0
    for m in range(1, n_nodes + 1):
        for h in range(n_nodes + 1):
            if h != m:
                Wskip_a.append(W.data[k])
            k += 1
    Wskip_a = np.array(Wskip_a, dtype=np.double)

    Wskip = torch.from_numpy(Wskip_a).requires_grad_()

    tree_slow = DependencyTree(n_nodes)
    posteriors = tree_slow(Wskip)
    print("posteriors slow", posteriors)

    tree_fast = DependencyTreeFast(n_nodes)
    posteriors = tree_fast(W)
    print("posteriors fast", posteriors)

    posteriors[0].backward()
    print("dposteriors_dW", W.grad)
