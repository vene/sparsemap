from libcpp.vector cimport vector

from cython.view cimport array as cvarray
from cython.view cimport array_cwrapper

from ad3.base cimport PGenericFactor, GenericFactor, Configuration
from ad3.base cimport FactorGraph, BinaryVariable, Factor

import numpy as np
cimport numpy as np
np.import_array()


# MAKES A COPY but should not leak
cdef asfloatvec(void* vec, int n):
    cdef np.npy_intp shape[1]

    shape[0] = n
    arr = np.PyArray_SimpleNewFromData(1, shape, np.NPY_DOUBLE, vec)
    farr = np.PyArray_Cast(arr, np.NPY_FLOAT32)
    return farr


cdef asfloatarray(void* vec, int rows, int cols):
    cdef np.npy_intp shape[2]
    shape[0] = <np.npy_intp> rows
    shape[1] = <np.npy_intp> cols
    arr = np.PyArray_SimpleNewFromData(2, shape, np.NPY_DOUBLE, vec)
    farr = np.PyArray_Cast(arr, np.NPY_FLOAT32)
    return farr


cpdef sparsemap(PGenericFactor f,
                vector[double] unaries,
                vector[double] additionals,
                int max_iter=10,
                int verbose=0):

    cdef:
        int i, n_active, n_var, n_add

        vector[BinaryVariable*] variables

        vector[double] post_unaries
        vector[double] post_additionals
        vector[Configuration] active_set_c
        vector[double] distribution
        vector[double] inverse_A
        vector[double] M, Madd

        GenericFactor* gf

    n_var = unaries.size()

    cdef FactorGraph fg
    fg.SetVerbosity(verbose)

    variables.resize(n_var)
    for i in range(n_var):
        variables[i] = fg.CreateBinaryVariable();

    fg.DeclareFactor(<Factor*> f.thisptr, variables, False)

    gf = <GenericFactor*?> f.thisptr
    gf.SetQPMaxIter(max_iter)
    gf.SetClearCache(False)  # because we need the cache
    f.thisptr.SetAdditionalLogPotentials(additionals)
    f.thisptr.SolveQP(unaries, additionals, &post_unaries, &post_additionals)

    active_set_c = gf.GetQPActiveSet()
    distribution = gf.GetQPDistribution()
    inverse_A = gf.GetQPInvA()
    gf.GetCorrespondence(&M, &Madd)

    n_active = active_set_c.size()
    n_add = post_additionals.size()

    post_unaries_np = asfloatvec(post_unaries.data(), n_var)
    post_additionals_np = asfloatvec(post_additionals.data(), n_add)
    distribution_np = asfloatvec(distribution.data(), n_active)
    invA_np = asfloatarray(inverse_A.data(), 1 + n_active, 1 + n_active)
    M_np = asfloatarray(M.data(), n_active, n_var)
    Madd_np = asfloatarray(Madd.data(), n_active, n_add)

    active_set_py = [f._cast_configuration(x) for x in active_set_c]

    solver_data = {
        'active_set': active_set_py,
        'distribution': distribution_np,
        'inverse_A': invA_np,
        'M': M_np,
        'Madd': Madd_np
    }

    return post_unaries_np, post_additionals_np, solver_data
