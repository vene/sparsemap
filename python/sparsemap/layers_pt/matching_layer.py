from ad3 import PFactorGraph

import torch
from torch.autograd import Function

from .base import _SparseMAP, _BaseSparseMAP
from .._factors import PFactorMatching


class Matching(_BaseSparseMAP):

    def forward(self, unaries):
        self.n_rows, self.n_cols = unaries.size()

        match = PFactorMatching()
        match.initialize(self.n_rows, self.n_cols)

        u = self.sparsemap(unaries.view(-1), match)

        return u.view_as(unaries)


if __name__ == '__main__':

    n_rows = 5
    n_cols = 3
    scores = torch.randn(n_rows, n_cols, dtype=torch.double, requires_grad=True)

    matcher = Matching(max_iter=1000)
    matching = matcher(scores)

    print(torch.autograd.grad(matching[0, 0], scores, retain_graph=True))
    print(torch.autograd.grad(matching[0, 0], scores))

    from torch.autograd import gradcheck
    print(gradcheck(matcher, scores, eps=1e-4, atol=1e-3))
