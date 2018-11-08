from ad3 import PFactorGraph

import torch
from torch.autograd import Variable, Function

from .base import _BaseSparseMarginals
from .._factors import PFactorMatching


class MatchingSparseMarginals(_BaseSparseMarginals):

    def build_factor(self):
        match = PFactorMatching()
        match.initialize(self.n_rows, self.n_cols)
        return match

    def forward(self, unaries):
        self.n_rows, self.n_cols = unaries.size()
        u = super().forward(unaries.view(-1))
        return u.view_as(unaries)

    def backward(self, dy):
        dy = dy.contiguous().view(-1)
        da = super().backward(dy)
        return da.view(self.n_rows, self.n_cols)


if __name__ == '__main__':

    n_rows = 5
    n_cols = 3
    scores = torch.randn(n_rows, n_cols)
    scores = Variable(scores, requires_grad=True)

    matcher = MatchingSparseMarginals()
    matching = matcher(scores)

    print(matching)
    matching.sum().backward()

    print("dpost_dunary", scores.grad)
