# SparseMAP: Differentiable Sparse Structure Inference

![SparseMAP cartoon](sparsemap.png?raw=true "SparseMAP cartoon")

[![Build Status](https://travis-ci.org/vene/sparsemap.svg?branch=master)](https://travis-ci.org/vene/sparsemap)<Paste>

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
> In: Proc. of ICML, 2018.

SparseMAP may be used to dynamically infer the computation graph structure,
marginalizing over a sparse distribution over all possible structures. Navigate 
to the `cpp` folder for an implementation, and see our paper,

> [Towards Dynamic Computation Graphs via Sparse Latent
> Structure.](https://arxiv.org/abs/1809.00653)
> Vlad Niculae, André F.T. Martins, Claire Cardie.
> In: Proc. of EMNLP, 2018.

## Current state of the codebase

We are working to slowly provide useful implementations. At the moment,
the codebase provides a generic pytorch 1.0 layer, as well as particular 
instantiations for sequence, matching, and tree layers.

Dynet custom layers, as well as the SparseMAP loss, are on the way.


## Python Setup

Requirements: numpy, scipy, Cython, pytorch>=1.0, and ad3 >= 2.2

1. Set the `AD3_DIR` environment variable to point to the
   [AD3](https://github.com/andre-martins/ad3) source directory,
   where you have compiled AD3.

2. Run `pip install .` (optionally with the `-e` flag).


### Notes on testing

Because of slight numerical differences, we had to relax the reentrancy
test from pytorch's gradcheck.

## Dynet (c++) setup:

See the instructions in the `cpp` folder.
