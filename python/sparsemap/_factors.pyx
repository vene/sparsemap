from libcpp cimport bool
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np

from ad3.base cimport Factor, GenericFactor, PGenericFactor, Configuration

ctypedef unsigned int uint


cdef extern from "FactorMatching.h" namespace "sparsemap":
    cdef cppclass FactorMatching(Factor):
        FactorMatching()
        void Initialize(int, int)


cdef extern from "FactorMatchingSparse.h" namespace "sparsemap":
    cpdef void test_factor()
    cdef cppclass FactorMatchingSparse(Factor):
        void Initialize(uint, uint, vector[uint], vector[uint])


cdef extern from "FactorSequenceDistance.h" namespace "sparsemap":
    cdef cppclass FactorSequenceDistance(Factor):
        FactorSequenceDistance()
        void Initialize(int n_nodes, int n_states, int bandwidth)


cdef extern from "FactorTreeFast.h" namespace "sparsemap":
    cdef cppclass FactorTreeFast(GenericFactor):
        FactorTreeFast()
        void Initialize(int length)


cdef class PFactorMatching(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
            self.thisptr = new FactorMatching()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int rows, int cols):
        (<FactorMatching*>self.thisptr).Initialize(rows, cols)


cdef class PFactorMatchingSparse(PGenericFactor):

    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
            self.thisptr = new FactorMatchingSparse()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, X):

        cdef vector[uint] indices, indptr
        for j in X.indices:
            indices.push_back(j)
        for j in X.indptr:
            indptr.push_back(j)

        (<FactorMatchingSparse*>self.thisptr).Initialize(
            X.shape[0],
            X.shape[1],
            indptr,
            indices)


cdef class PFactorSequenceDistance(PGenericFactor):
    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
           self.thisptr = new FactorSequenceDistance()

    def __dealloc__(self):
        if self.allocate:
            del self.thisptr

    def initialize(self, int n_nodes, int n_states, int bandwidth):
        (<FactorSequenceDistance*>self.thisptr).Initialize(n_nodes,
                                                           n_states,
                                                           bandwidth)


cdef class PFactorTreeFast(PGenericFactor):

    def __cinit__(self, allocate=True):
        self.allocate = allocate
        if allocate:
            self.thisptr = <Factor*> new FactorTreeFast()

    def initialize(self, int length):
            (<FactorTreeFast*>self.thisptr).Initialize(length)
