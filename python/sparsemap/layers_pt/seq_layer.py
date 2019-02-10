from itertools import product
from ad3 import PFactorGraph
from ad3.extensions import PFactorSequence

import torch

from .base import _BaseSparseMAP
from .._factors import PFactorSequenceDistance


class StationarySequencePotentials(torch.nn.Module):
    def forward(self, transition, n_variables, start=None, end=None):
        n_states, n_states_ = transition.size()
        assert n_states == n_states_

        if start is None:
            start = transition.zero_(n_states)
        else:
            assert start.dim() == 1 and start.size()[0] == n_states

        if end is None:
            end = transition.zero_(n_states)
        else:
            assert end.dim() == 1 and end.size()[0] == n_states

        return torch.cat([start,
                          transition.view(-1).repeat(n_variables - 1),
                          end])


class Sequence(_BaseSparseMAP):

    def forward(self, unaries, additionals):
        """Returns a weighted sum of the most likely posterior assignments.

        Inputs:

        unaries: tensor, size: (n_variables, n_states)
            Scores for each variable to be in each of its valid states.
            (For simplicity we assume equal number of states for each variable,
            but it's possible to relax this.)

        additionals: tensor, size (n_states + (n_variables - 1) * n_states ** 2
                                   + n_states)
            Initial, transitional and final log-potentials, in this order. For
            simple stationary models, use StationarySequencePotentials.

        Returns:

            marginals: tensor, size (n_variables, n_states)
            Sparse posterior unary marginals: a-posteriori probabilities for
            each variable to be in each of its valid states.
        """

        self.n_variables, self.n_states = unaries.size()
        seq = PFactorSequence()
        seq.initialize([self.n_states] * self.n_variables)
        u = self.sparsemap(unaries.view(-1), seq, additionals)
        return u.view_as(unaries)


class SequenceDistance(Sequence):
    def __init__(self, bandwidth, max_iter=20, verbose=False):
        self.bandwidth = bandwidth
        super(SequenceDistance, self).__init__(max_iter, verbose)

    def forward(self, unaries, additionals):
        self.n_variables, self.n_states = unaries.size()
        seq = PFactorSequenceDistance()
        seq.initialize(self.n_variables, self.n_states, self.bandwidth)
        u = self.sparsemap(unaries.view(-1), seq, additionals)
        return u.view_as(unaries)


if __name__ == '__main__':

    n_variables = 4
    n_states = 3
    torch.manual_seed(12)

    unary = torch.randn(n_variables, n_states, requires_grad=True)
    start = torch.randn(n_states, requires_grad=True)
    end = torch.randn(n_states, requires_grad=True)
    transition = torch.randn(n_states, n_states, requires_grad=True)

    stationary_seq = StationarySequencePotentials()
    seq_marginals = Sequence()

    additionals = stationary_seq(transition,
                                 n_variables,
                                 start,
                                 end)
    posterior = seq_marginals(unary, additionals)

    print(posterior)
    posterior[0, 0].backward()

    print("dpost_dunary", unary.grad)
    print("dstart", start.grad)
    print("dend", end.grad)
    print("dtrans", transition.grad)

    print("With distance-based parametrization")

    bw = 3
    dist_additional = torch.randn(1 + 4 * bw, requires_grad=True)
    seq_dist_marg = SequenceDistance(bw)
    posterior = seq_dist_marg(unary, dist_additional)
    print(posterior)
    posterior[0, 0].backward()
    print("dpost_dunary", unary.grad)
    print("dpost_dadd", dist_additional.grad)
