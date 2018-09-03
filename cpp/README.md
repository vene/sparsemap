# SparseMAP for dynet

![SparseMAP cartoon](sparsemap-cg.png?raw=true "SparseMAP for Dynamic Computation Graphs")

This is an implementation of *Latent Dependency TreeLSTMs* for classification,
NLI and reverse dictionary. For conceptual details, see our paper

> Towards Dynamic Computation Graphs via Sparse Latent Structure
> Vlad Niculae, André F.T. Martins, Claire Cardie
> In: Proc. of EMNLP, 2018. (preprint coming soon.)

Coming soon to this folder: dynet modules for the SparseMAP loss and SparseMAP
structured attention.

## Build & Run

**Requirements:**
[AD3](https://github.com/andre-martins/ad3),
[dynet](https://github.com/clab/dynet) (tested with version
[2.0.2](https://github.com/clab/dynet/releases/tag/2.0.2)),
[Eigen](http://eigen.tuxfamily.org/) (as required by dynet).

**Optional:** MKL, CUDA (via dynet).

**Environment setup:** point `AD3_DIR`, `DYNET_DIR` and `EIGEN_DIR` to the
corresponding source folders. By default, set to `~/code/{ad3|dynet|eigen}`.

**Compiling dynet:**. Make sure to compile dynet such that `libdynet.so` is in
`DYNET_DIR/build/dynet` (for CPU support), or `DYNET_DIR/build-cuda/dynet` if
using CUDA. For instructions on building dynet, see [their
documentation](https://dynet.readthedocs.io/en/latest/install.html). 

After compilation, make sure the correct `libdynet.so` is in your `LD_LIBRARY_PATH`.

**Compiling AD3:** should be as easy as `$ cd AD3_DIR; make`.

**Compiling SparseMAP code and classifiers.**

The provided programs are

  - `test-sparsemst`: test program which runs gradient checks
and demonstrates the usage of the sparse latent MST parser module.
  - `sentclf-{cpu|gpu}`: Sentence classifier, e.g. for the SST dataset.
  - `nliclf-{cpu|gpu}`: Natural Language Inference classifier (sentence pairs)
  - `revdict-{cpu|gpu}`: Reverse Dictionary (given a definition, output
embedding of defined word.)

Example: to compile and run the test file, you may use

```
cd dyncg
make test-sparsemst
./test-sparsemst
```

To compile and run the CPU version of the NLI classifier on the subjectivity data, type

```
cd dyncg
make sentclf-cpu
./sentclf-cpu --dynet-seed 42 --dynet-autobatch 1 --dataset subj --strategy latent
```

## Data

Preprocessed data can be downloaded
[here](https://www.dropbox.com/s/1chr6ur2swrfypv/niculae18-sparsemap-cg-data.tar.xz?dl=0).
This archive contains a `data` folder that should be placed in the `dyncg` folder. 
Scripts used to recreate the sentence classification data are provided in
`dyncg/data/sentclf` (in this repository).
