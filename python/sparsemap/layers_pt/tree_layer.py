# pytorch layer that applies a TreeFactor

import numpy as np
import torch
from torch.autograd import Variable, Function
from torch import nn

from ad3 import PFactorGraph
from ad3.extensions import PFactorTree

from .base import _BaseSparseMarginals
from .._factors import PFactorTreeFast


class TreeSparseMarginals(_BaseSparseMarginals):

    def __init__(self, n_nodes=None, max_iter=10, verbose=0):
        self.n_nodes = n_nodes
        self.max_iter = max_iter
        self.verbose = verbose

    def build_factor(self):
        n_nodes = self.n_nodes
        g = PFactorGraph()
        self.arcs = [(h, m)
                     for m in range(1, n_nodes + 1)
                     for h in range(n_nodes + 1)
                     if h != m]
        arc_vars = [g.create_binary_variable() for _ in self.arcs]
        tree = PFactorTree()
        g.declare_factor(tree, arc_vars)
        tree.initialize(n_nodes + 1, self.arcs)
        return tree


class TreeSparseMarginalsFast(_BaseSparseMarginals):

    def __init__(self, n_nodes=None, max_iter=10, verbose=0):
        self.n_nodes = n_nodes
        self.max_iter = max_iter
        self.verbose = verbose

    def build_factor(self):
        n_nodes = self.n_nodes
        g = PFactorGraph()
        self.arcs = [(h, m)
                     for m in range(1, n_nodes + 1)
                     for h in range(n_nodes + 1)
                     if h != m]
        arc_vars = [g.create_binary_variable() for _ in self.arcs]
        tree = PFactorTreeFast()
        g.declare_factor(tree, arc_vars)
        tree.initialize(n_nodes + 1)
        return tree


if __name__ == '__main__':
    n_nodes = 3
    Wt = torch.randn((n_nodes + 1) * n_nodes)
    W = Variable(Wt, requires_grad=True)

    Wskip_a = []
    k = 0
    for m in range(1, n_nodes + 1):
        for h in range(n_nodes + 1):
            if h != m:
                Wskip_a.append(Wt[k])
            k += 1
    Wskip_a = np.array(Wskip_a)

    Wskip_t = torch.from_numpy(Wskip_a)
    Wskip = Variable(Wskip_t, requires_grad=True)

    tsm_slow = TreeSparseMarginals(n_nodes)
    posteriors = tsm_slow(Wskip)
    print("posteriors slow", posteriors)

    tsm = TreeSparseMarginalsFast(n_nodes)
    posteriors = tsm(W)
    print("posteriors fast", posteriors)
    posteriors.sum().backward()
    print("dposteriors_dW", W.grad)
