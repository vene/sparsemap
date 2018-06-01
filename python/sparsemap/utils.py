from __future__ import division

import math
import numpy as np

import torch
from torch.autograd import Variable


def batch_slices(n_samples, batch_size=32):
    n_batches = math.ceil(n_samples / batch_size)
    batches = [slice(ix * batch_size, (ix + 1) * batch_size)
               for ix in range(n_batches)]
    return batches


def S_from_Ainv(Ainv):
    """See footnote in notes.pdf"""

    # Ainv = torch.FloatTensor(Ainv).view(1 + n_active, 1 + n_active)
    S = Ainv[1:, 1:]
    k = Ainv[0, 0]
    b = Ainv[0, 1:].unsqueeze(0)

    S -= (1 / k) * (b * b.t())
    return S


def expand_with_zeros(x, rows, cols):
    orig_rows, orig_cols = x.size()

    ret = x
    if orig_cols < cols:
        horiz = Variable(x.data.new(orig_rows, cols - orig_cols).zero_())
        ret = torch.cat([ret, horiz], dim=-1)

    if orig_rows < rows:
        vert = Variable(x.data.new(rows - orig_rows, cols).zero_())
        ret = torch.cat([ret, vert], dim=0)

    return ret


def zeros_like(torch_var):
    data = torch_var.data.new(torch_var.size()).zero_()
    return Variable(data)
