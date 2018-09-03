/* Latent TreeLSTM encoders powered by SparseMAP.
 * Author: vlad@vene.ro
 * License: Apache 2.0
 */

# pragma once

#include <dynet/nodes.h>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/devices.h>

#include "../common/treernn.h"
#include "../common/dependency_scorers.h"

#include "sparsemst.h"

namespace dy = dynet;

using dy::Expression;
using dy::Parameter;
using dy::LookupParameter;
using dy::parameter;

using std::vector;


/**
 * \brief Perform dy::"incremental forward" on multiple expressions efficiently.
 */
void incremental_forward_all(ComputationGraph& cg,
                             vector<Expression> exprs)
{
    Expression* max_expr = &(exprs[0]);
    for (auto && expr : exprs)
        if (expr.i > max_expr->i)
            max_expr = &expr;
    cg.incremental_forward(*max_expr);
}


struct BaseTreeEncoder
{

    unsigned vocab_size_;
    unsigned hidden_dim_;

    LookupParameter p_emb;
    dy::ParameterCollection p;

    explicit
    BaseTreeEncoder(
        dy::ParameterCollection& params,
        unsigned hidden_dim)
        : hidden_dim_(hidden_dim)
    {
        p = params.add_subcollection("tree-encoder");
    }
};

struct FixedTreeEncoder : public BaseTreeEncoder
{

    std::unique_ptr<ChildSumTreeLSTMBuilder> tree_rnn;

    explicit
    FixedTreeEncoder(
        dy::ParameterCollection& params,
        unsigned hidden_dim,
        ChildSumTreeLSTMBuilder::Strategy strategy)
        : BaseTreeEncoder(params, hidden_dim)
    {
        tree_rnn = std::unique_ptr<ChildSumTreeLSTMBuilder>(
            new ChildSumTreeLSTMBuilder(p, hidden_dim, hidden_dim, strategy));
    }

    /* Automatically determined tree */
    virtual
    vector<Expression>
    encode_batch(
        dy::ComputationGraph& cg,
        vector<vector<Expression> >& sequences)
    {
        assert(tree_rnn->strategy_ != ChildSumTreeLSTMBuilder::Strategy::CUSTOM);

        tree_rnn->new_graph(cg);
        vector<Expression> out;
        for (int k = 0; k < sequences.size(); ++k)
        {
            tree_rnn->new_sequence(sequences[k]);
            auto repr = tree_rnn->transform();
            out.push_back(repr);
        }
        return out;
    }

    /* Custom trees: indices must be passed */
    virtual
    vector<Expression>
    encode_batch(
        dy::ComputationGraph& cg,
        vector<vector<Expression> >& sequences,
        vector<vector<int> >& trees)
    {
        assert(tree_rnn->strategy_ == ChildSumTreeLSTMBuilder::Strategy::CUSTOM);

        tree_rnn->new_graph(cg);
        vector<Expression> out;
        for (int k = 0; k < sequences.size(); ++k)
        {
            tree_rnn->new_sequence(sequences[k]);
            auto repr = tree_rnn->transform(trees[k]);
            out.push_back(repr);
        }
        return out;
    }

};


struct LatentTreeEncoder : public BaseTreeEncoder
{

    size_t max_iter_;

    bool print_trees = false;
    std::shared_ptr<std::ostream> out_trees;
    std::unique_ptr<EdgeScorer> scorer;
    std::unique_ptr<ChildSumTreeLSTMBuilder> tree_rnn;

    #if USING_CUDA
    dy::Device* cpu_device = dy::get_device_manager()->get_global_device("CPU");
    dy::Device* gpu_device = dy::get_device_manager()->get_global_device("GPU:0");
    #endif

    explicit
    LatentTreeEncoder(
        dy::ParameterCollection& params,
        unsigned hidden_dim,
        size_t max_iter=10,
        std::string scorer_type="mlp")

        : BaseTreeEncoder(params, hidden_dim)
        , max_iter_(max_iter)
    {
        auto strategy = ChildSumTreeLSTMBuilder::Strategy::CUSTOM;
        tree_rnn = std::unique_ptr<ChildSumTreeLSTMBuilder>(
            new ChildSumTreeLSTMBuilder(p, hidden_dim, hidden_dim, strategy));

        if (scorer_type == "mlp")
            scorer.reset(new MLPScorer(p, hidden_dim, hidden_dim));
        else if (scorer_type == "bilinear")
            scorer.reset(new BilinearScorer(p, hidden_dim, hidden_dim));
        else
            throw std::invalid_argument("Invalid scorer, need mlp|bilinear");
    }

    std::tuple<Expression, vector<vector<Expression> > >
    encode_batch(
        dy::ComputationGraph& cg,
        vector<vector<Expression> >& sequences,
        float temperature=1.0f)
    {

        using dy::concatenate_cols;
        using dy::transpose;

        tree_rnn->new_graph(cg);
        scorer->new_graph(cg);

        vector<Expression> mst_nodes;

        for (auto && seq : sequences)
        {
            // std::cout << temperature << std::endl;
            auto edge_scores = scorer->make_potentials(seq) / temperature;
            // std::cout << edge_scores.value() << std::endl;
            // std::abort();

            #if USING_CUDA
            edge_scores = dy::to_device(edge_scores, cpu_device);
            #endif

            auto post = sparse_mst(edge_scores, 1 + seq.size(), max_iter_);
            mst_nodes.push_back(post);
        }

        auto posteriors = dy::concatenate_cols(mst_nodes);
        cg.incremental_forward(posteriors);

        #if USING_CUDA
        posteriors = dy::to_device(posteriors, gpu_device);
        #endif

        vector<vector<Expression> > out(sequences.size());

        for(int i = 0; i < sequences.size(); ++i)
        {
            tree_rnn->new_sequence(sequences[i]);

            auto node = cg.nodes[mst_nodes[i].i];
            auto mst_node = static_cast<SparseMST*>(node);
            int n_active = mst_node->get_n_active();

            if (print_trees) *out_trees << n_active << '\t';

            out[i].reserve(n_active);
            for (int k = 0; k < n_active; ++k)
            {
                auto parents = mst_node->get_config(k);
                auto repr = tree_rnn->transform(parents);
                out[i].push_back(repr);

                if (print_trees)
                {
                    *out_trees << t<2>(posteriors.value())(k, i) << ' ';
                    for(auto && val: parents)
                        *out_trees << val << ' ';
                    *out_trees << '\t';
                }
            }
            if (print_trees) *out_trees << '\n';
        }
        return std::make_tuple(posteriors, out);
    }

    vector<Expression>
    encode_batch_map(
        dy::ComputationGraph& cg,
        vector<vector<Expression> >& sequences)
    {

        using dy::concatenate_cols;
        using dy::transpose;

        tree_rnn->new_graph(cg);
        scorer->new_graph(cg);

        vector<Expression> all_potentials;

        for (auto && seq : sequences)
        {
            auto edge_scores = scorer->make_potentials(seq);

            #if USING_CUDA
            edge_scores = dy::to_device(edge_scores, cpu_device);
            #endif
            all_potentials.push_back(edge_scores);
            // auto post = sparse_mst(edge_scores, 1 + seq.size(), max_iter_);
        }

        incremental_forward_all(cg, all_potentials);

        vector<Expression> out(sequences.size());

        for (int i = 0; i < sequences.size(); ++i)
        {
            tree_rnn->new_sequence(sequences[i]);

            auto parents = mst(all_potentials[i].value(),
                               1 + sequences[i].size());
            out[i] = tree_rnn->transform(parents);
        }
        return out;
    }
};
