# SparseMAP: Differentiable Sparse Structure Inference

![SparseMAP cartoon](sparsemap.png?raw=true "SparseMAP cartoon")

SparseMAP is a new method for **sparse structured inference,**
able to automatically select only a few global structures:
it is  situated between MAP inference, which picks a single structure, 
and marginal inference, which assigns probability mass to all structures, 
including implausible ones. 

SparseMAP is **differentiable** and can work with any structure for which a MAP
oracle is available.

More info in our paper,

> [SparseMAP: Differentiable Sparse Structured Inference](https://arxiv.org/abs/1802.04223).
> Vlad Niculae, Andre F.T. Martins, Mathieu Blondel, Claire Cardie.
> In: Proc. of ICML, 2018. (Camera ready version coming soon.)


## Current state of the codebase

We are working to slowly provide useful implementations. At the moment,
the codebase provides a generic pytorch layer supporting version 0.2,
as well as particular instantiations for sequence, matching, and tree layers.

Dynet custom layers, as well as the SparseMAP loss, are on the way.


## Setup

Requirements: numpy, scipy, Cython, pytorch=0.2, and ad3 >= 2.2

1. Set the `AD3_DIR` environment variable to point to the
   [AD3](https://github.com/andre-martins/ad3) source directory.

2. Inside the `python` dir, run  `python setup.py build_ext --inplace`.


## Notes on testing

The implemented layers pass numerical tests. However, the pytorch
gradcheck (as of version 0.2) has a very strict "reentrant" test, which we fail
due to tiny numerical differences. To reliably check gradients, please comment
out the `if not reentrant: ...` part of pytorch's gradcheck.py.
