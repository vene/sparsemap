/* SparseMAP in configuration-space with an MST dependency tree factor.
 * Author: vlad@vene.ro
 * License: Apache 2.0
 */

#pragma once

#include <Eigen/Eigen>
#include <dynet/dynet.h>
#include <dynet/nodes-def-macros.h>
#include <dynet/expr.h>
#include <dynet/tensor-eigen.h>

#include <map>

using namespace dynet;
using std::string;
using std::vector;

typedef Eigen::RowVectorXf eivector;


struct SparseMST : public Node {

    size_t length_;
    size_t max_iter_;

    explicit SparseMST(const std::initializer_list<VariableIndex>& a,
                       size_t length, size_t max_iter)
        : Node(a), length_(length), max_iter_(max_iter)
    {
        this->has_cuda_implemented = false;
    }
    DYNET_NODE_DEFINE_DEV_IMPL()
    size_t aux_storage_size() const override;

    int get_n_active() const;
    vector<int> get_config(size_t i) const;

    protected:
    void sparsemap_decode(const Tensor*, int*, int*, float*, float*) const;
    void backward_restricted(const eivector& dEdfbar, Tensor& dEdx) const;

    float* get_inv_A_ptr() const;
};

struct SparseMSTFull : public SparseMST {
    explicit SparseMSTFull(const std::initializer_list<VariableIndex>& a,
                           size_t length, size_t max_iter)
        : SparseMST(a, length, max_iter) { }

    DYNET_NODE_DEFINE_DEV_IMPL()
};


/**
 * \brief Compute SparseMAP sparse posterior over trees.
 * \param x Input arc scores.
 * \param length Number of words in sentence
 * \param max_iter Number of iterations of Active Set to perform.
 * \return sparse vector of posteriors (fixed size = max_iter)
 *
 * To correctly identify which tree each output index corresponds to,
 * use SparseMST::get_config.
 */
Expression sparse_mst(const Expression& x, size_t length, size_t max_iter=10);

/**
 * \brief Debug: compute SparseMAP posterior over trees with dense output.
 * \param x Input arc scores.
 * \param length Number of words in sentence
 * \param max_iter Number of iterations of Active Set to perform.
 * \return Exponentially-sized vectors with posterior of every tree.
 */
Expression sparse_mst_full(const Expression& x, size_t length, size_t max_iter=10);


/**
 * \brief Compute MAP inference (maximum scoring tree)
 * \param x Input arc scores.
 * \param length Number of words in sentence
 * \return Parent vector of MAP tree.
 */
vector<int> mst(const Tensor& x, size_t length);
