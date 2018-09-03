/* SparseMAP in configuration-space with an MST dependency tree factor.
 * Author: vlad@vene.ro
 * License: Apache 2.0
 */

#include <cmath>
#include <dynet/nodes-impl-macros.h>
#include <ad3/FactorGraph.h>

#include "sparsemst.h"
#include "../../src/FactorTreeFast.h"


using namespace dynet;
using std::string;
using std::vector;
using sparsemap::FactorTreeFast;

namespace ei = Eigen;


size_t SparseMST::aux_storage_size() const
{
    return (sizeof(int) * ((length_ * max_iter_) + 1) +
            sizeof(float) * (max_iter_ + 1) * (max_iter_ + 1));
}

/* very evil pointer arithmetic. look away for 3 rows */
float* SparseMST::get_inv_A_ptr() const {
    int* aux = static_cast<int*>(aux_mem);
    return static_cast<float*>(
        static_cast<void*>(1 + aux + (length_ * max_iter_)));
}


int SparseMST::get_n_active() const {
    int* aux = static_cast<int*>(aux_mem);
    return aux[0];
}


/* TODO: can we return the vector by reference here and preserve it */
vector<int> SparseMST::get_config(size_t i) const
{
    int* active_set_ptr = 1 + static_cast<int*>(aux_mem);
    return vector<int>(active_set_ptr + length_ * i,
                       active_set_ptr + length_ * (i + 1));
}

/**
 * \brief Solves the SparseMAP optimization problem for tree factor
 * \param x Arc potentials
 * \param n_active Location where support size will be stored
 * \param active_set_ptr Location where the active set will be stored
 * \param distribution_ptr Location where the posterior weights will be stored
 * \param inverse_A_ptr Location where (MtM)^-1 will be stored
 */

void SparseMST::sparsemap_decode(const Tensor* x,
                                 int* n_active,
                                 int* active_set_ptr,
                                 float* distribution_ptr,
                                 float* inverse_A_ptr) const
{
    auto xvec = vec(*x);

    // run ad3
    AD3::FactorGraph factor_graph;
    // factor_graph.SetVerbosity(3);
    vector<AD3::BinaryVariable*> vars;

    for (int m = 1; m < length_; ++m)
        for (int h = 0; h < length_; ++h)
            if (h != m)
                vars.push_back(factor_graph.CreateBinaryVariable());

    // input potentials
    vector<double> unaries_in(xvec.data(), xvec.data() + xvec.size());

    // output variables (additionals will be unused)
    vector<double> unaries_post;
    vector<double> additionals;
    FactorTreeFast tree_factor;

    factor_graph.DeclareFactor(&tree_factor, vars, false);
    tree_factor.SetQPMaxIter(max_iter_);
    tree_factor.SetClearCache(false);
    tree_factor.Initialize(length_);
    tree_factor.SolveQP(unaries_in, additionals, &unaries_post, &additionals);

    auto active_set = tree_factor.GetQPActiveSet();
    *n_active = active_set.size();

    // write distribution as floats
    auto distribution = tree_factor.GetQPDistribution();
    assert(active_set.size() <= distribution.size());
    std::copy(distribution.begin(),
              distribution.begin() + *n_active,
              distribution_ptr);

    // std::cout << "Distribution ";
    // for(auto &&v : distribution) std::cout << v << " ";
    // std::cout << std::endl;

    int* active_set_moving_ptr = active_set_ptr;
    for (auto&& cfg_ptr : active_set)
    {
        auto cfg = static_cast<vector<int>*>(cfg_ptr);
        std::copy(cfg->begin(), cfg->end(), active_set_moving_ptr);
        active_set_moving_ptr += cfg->size();
    }

    auto inverse_A = tree_factor.GetQPInvA();
    std::copy(inverse_A.begin(), inverse_A.end(), inverse_A_ptr);
}


/**
 * \brief Backward pass restricted to the active set:
 *
 * \param dEdf_bar Gradient dE wrt layer output
 * \param dEdx Gradient wrt potentials x; incremented in place
 *
 * Computes dE/db_bar as intermediate quantity
 * where:
 *
 * f(x) is the sparse distribution over all possible trees
 * f_bar(x) is f restricted to its support
 * b_bar(x) is the vector of total scores for the configurations in the support
 *
 */
void SparseMST::backward_restricted(const eivector& dEdf_bar, Tensor& dEdx)
const
{

    int* aux = static_cast<int*>(aux_mem);
    int* active_set_ptr = 1 + aux;
    float* inv_A_ptr = get_inv_A_ptr();

    int n_active = *aux;

    ei::Map<ei::Matrix<float, ei::Dynamic, ei::Dynamic, ei::RowMajor> >
        e_inva(inv_A_ptr, 1 + n_active, 1 + n_active);

    /* A^-1    = [ k   b^T]
     *           [ b    S ]
     *
     * (MtM)^-1 = S - (1/k) outer(b, b)
     */

    float k = e_inva(0, 0);
    auto b = e_inva.row(0).tail(n_active);
    auto S = e_inva.bottomRightCorner(n_active, n_active);
    S.noalias() -= (1 / k) * (b.transpose() * b);
    auto first_term = dEdf_bar * S;
    auto second_term = (first_term.sum() * S.rowwise().sum()) / S.sum();
    auto dEdb_bar = first_term - second_term.transpose();

    // map gradients back to the unaries that contribute to them
    // (sparse multiplication by M)
    for (int i = 0; i < n_active ; ++i)
    {
        for (int mod = 1; mod < length_; ++mod)
        {
            size_t mod_address = (i * length_) + mod;
            int head = active_set_ptr[mod_address];
            mat(dEdx)(head, mod - 1) += dEdb_bar(i);
        }
    }
}


/* Sparse version
 * **************
 *
 * Cannot perform gradient checks because output is dynamic
 */

string SparseMST::as_string(const vector<string>& arg_names) const
{
    std::ostringstream s;
    s << "sparse_mst(" << arg_names[0] << ")";
    return s.str();
}


Dim SparseMST::dim_forward(const vector<Dim> &xs) const
{
    DYNET_ARG_CHECK(xs.size() == 1, "SparseMST takes a single input");
    DYNET_ARG_CHECK(xs[0][0] == length_, "input has wrong first dim");
    DYNET_ARG_CHECK(xs[0][1] == length_ - 1, "input has wrong second dim");
    unsigned int d = max_iter_;
    return Dim({d});
}


template<class MyDevice>
void SparseMST::forward_dev_impl(const MyDevice& dev,
                                     const vector<const Tensor*>& xs,
                                     Tensor& fx) const
{
    const Tensor* x = xs[0];

    auto out = vec(fx);
    out.setZero();

    int* aux = static_cast<int*>(aux_mem);
    int* active_set_ptr = aux + 1;
    float* inv_A_ptr = get_inv_A_ptr();

    vector<float> distribution(max_iter_);
    sparsemap_decode(x, aux, active_set_ptr, out.data(), inv_A_ptr);
}

template <class MyDevice>
void SparseMST::backward_dev_impl(const MyDevice& dev,
                                  const vector<const Tensor*>& xs,
                                  const Tensor& fx,
                                  const Tensor& dEdf,
                                  unsigned i,
                                  Tensor& dEdxi) const
{
    int n_active = *(static_cast<int*>(aux_mem));
    ei::Map<const eivector> dEdf_bar(mat(dEdf).data(), n_active);
    backward_restricted(dEdf_bar, dEdxi);
}

DYNET_NODE_INST_DEV_IMPL(SparseMST)

Expression sparse_mst(const Expression& x, size_t length, size_t max_iter) {
    return Expression(x.pg,
                      x.pg->add_function<SparseMST>({x.i}, length, max_iter));
}


/*
 * Debug version with full output
 * ******************************
 */

string SparseMSTFull::as_string(const vector<string>& arg_names) const
{
    std::ostringstream s;
    s << "sparse_mst_full(" << arg_names[0] << ")";
    return s.str();
}


Dim SparseMSTFull::dim_forward(const vector<Dim> &xs) const
{
    DYNET_ARG_CHECK(xs.size() == 1, "SparseMST takes a single input");
    DYNET_ARG_CHECK(xs[0][0] == length_, "input has wrong first dim");
    DYNET_ARG_CHECK(xs[0][1] == length_ - 1, "input has wrong second dim");
    unsigned out_space = pow(length_, length_ - 1);
    return Dim({out_space});
}


template<class MyDevice>
void SparseMSTFull::forward_dev_impl(const MyDevice& dev,
                                     const vector<const Tensor*>& xs,
                                     Tensor& fx) const
{
    const Tensor* x = xs[0];

    auto out = vec(fx);
    out.setZero();

    int* aux = static_cast<int*>(aux_mem);

    int* active_set_ptr = aux + 1;
    float* inv_A_ptr = get_inv_A_ptr();


    vector<float> distribution(max_iter_);
    sparsemap_decode(x, aux, active_set_ptr, distribution.data(), inv_A_ptr);

    for (size_t i = 0; i < *aux; ++i)
    {
        unsigned flat_ix = 0;  // debug: index into full config space

        // skip the initial -1
        //
        for (int mod = 1; mod < length_; ++mod)
        {
            size_t mod_address = (i * length_) + mod;
            int head = active_set_ptr[mod_address];
            flat_ix += head * pow(length_, mod - 1);
        }
        out(flat_ix) = distribution[i];
    }

}


template <class MyDevice>
void SparseMSTFull::backward_dev_impl(const MyDevice& dev,
                                      const vector<const Tensor*>& xs,
                                      const Tensor& fx,
                                      const Tensor& dEdf,
                                      unsigned i,
                                      Tensor& dEdxi) const
{

    int* aux = static_cast<int*>(aux_mem);
    int* active_set_ptr = aux + 1;
    int n_active = aux[0];

    // this is specific to this debugging case
    eivector dEdf_bar(n_active);
    for (int i = 0; i < n_active; ++i)
    {
        size_t flat_ix = 0;
        for (int mod = 1; mod < length_; ++mod)
        {
            size_t mod_address = (i * length_) + mod;
            int head = active_set_ptr[mod_address];
            flat_ix += head * pow(length_, mod - 1);
        }
        dEdf_bar(i) = vec(dEdf)(flat_ix);
    }

    backward_restricted(dEdf_bar, dEdxi);
}


DYNET_NODE_INST_DEV_IMPL(SparseMSTFull)


Expression sparse_mst_full(const Expression& x, size_t length, size_t max_iter) {
    return Expression(x.pg,
                      x.pg->add_function<SparseMSTFull>({x.i}, length, max_iter));
}

vector<int> mst(const Tensor& x, size_t length)
{
    auto xvec = vec(x);

    vector<double> unaries_in(xvec.data(), xvec.data() + xvec.size());
    vector<double> additionals;
    double val;

    vector<int> out(length, -1);
    AD3::Configuration cfg = static_cast<AD3::Configuration>(&out);

    FactorTreeFast tree_factor;
    tree_factor.Initialize(length);
    tree_factor.Maximize(unaries_in, additionals, cfg, &val);
    return out;
}

