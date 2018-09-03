/* TreeLSTM sentiment classifier with latent trees
 * Author: vlad@vene.ro
 * License: Apache 2.0
 */

# pragma once


#include <string>
#include <dynet/rnn.h>
#include <dynet/lstm.h>
#include <dynet/tensor.h>
#include <dynet/index-tensor.h>

#include "../common/basemodel.h"
#include "tree_encoders.h"

namespace dy = dynet;

using dy::Expression;
using dy::Parameter;
using dy::LookupParameter;
using dy::ComputationGraph;
using dy::ParameterCollection;
using dy::parameter;
using dy::RNNBuilder;

using std::vector;
using std::unique_ptr;


struct BaseSentClf : public BaseEmbedBiLSTMModel
{
    Parameter p_out_W;
    Parameter p_out_b;

    unsigned n_classes_;
    bool training_ = false;

    explicit
    BaseSentClf(
        ParameterCollection& params,
        unsigned vocab_size,
        unsigned embed_dim,
        unsigned hidden_dim,
        unsigned n_classes,
        bool update_embed)
        : BaseEmbedBiLSTMModel(
            params,
            vocab_size,
            embed_dim,
            hidden_dim,
            update_embed,
            "sentclf")
        , n_classes_(n_classes)
    {
        p_out_W = p.add_parameters({n_classes_, hidden_dim});
        p_out_b = p.add_parameters({n_classes_});
    }

    /**
     * \brief Initialize stream for printing latent trees during prediction.
     *
     * Only used by the latent tree model.
     */
    virtual
    void
    set_print_trees(std::shared_ptr<std::ostream> out)
    { }

    /**
     * \brief Set temperature used by latent parser.
     *
     * Only used by the latent tree model.
     */
    virtual
    void
    set_temperature(float t)
    { }

    /**
     * \brief Count number of correct predictions in minibatch.
     * \param cg Current computation graph
     * \param batch Minibatch to run inference on
     * \return Number of correct predictions.
     */
    virtual
    int
    n_correct(
        ComputationGraph& cg,
        const SentBatch& batch)
    {
        training_ = false;
        auto out_v = predict_batch(cg, batch);
        auto out_b = dy::concatenate_to_batch(out_v);

        cg.incremental_forward(out_b);
        auto out = out_b.value();
        auto pred = dy::as_vector(dy::TensorTools::argmax(out));

        int n_correct = 0;
        for (int i = 0; i < batch.size(); ++i)
            if (batch[i].target == pred[i])
                n_correct += 1;

        return n_correct;
    }

    /**
     * \brief Compute loss on minibatch.
     * \param cg Current computation graph
     * \param batch Minibatch to run inference on
     * \return Dynet expression of total loss.
     */
    virtual
    Expression
    batch_loss(
        ComputationGraph& cg,
        const SentBatch& batch)
    = 0;

    /**
     * \brief Compute predicted values on minibatch.
     * \param cg Current computation graph
     * \param batch Minibatch to run inference on
     * \return Dynet expressions of predicted probabilities for each sample.
     */
    virtual
    vector<Expression>
    predict_batch(
        ComputationGraph& cg,
        const SentBatch& batch)
    = 0;
};


/**
 * \brief TreeLSTM sentence classifier with fixed tree.
 */
struct FixedDepSentClf : public BaseSentClf
{
    FixedTreeEncoder encoder;
    bool is_custom_;

    explicit
    FixedDepSentClf(
        ParameterCollection& params,
        unsigned vocab_size,
        unsigned embed_dim,
        unsigned hidden_dim,
        unsigned n_classes,
        ChildSumTreeLSTMBuilder::Strategy strategy,
        bool update_embed)
    : BaseSentClf(
        params,
        vocab_size,
        embed_dim,
        hidden_dim,
        n_classes,
        update_embed)
    , encoder(p, hidden_dim, strategy)
    , is_custom_(strategy == ChildSumTreeLSTMBuilder::Strategy::CUSTOM)
    { }

    virtual
    vector<Expression>
    predict_batch(
        ComputationGraph& cg,
        const SentBatch& batch)
    override
    {
        enc_rnn_fw->new_graph(cg);
        enc_rnn_bw->new_graph(cg);
        auto out_b = parameter(cg, p_out_b);
        auto out_W = parameter(cg, p_out_W);

        vector<vector<Expression> > embedded_sents;
        vector<vector<int> > trees;

        for (auto && sample : batch)
        {
            embedded_sents.push_back(embed_ctx_sent(cg, sample.sentence));
            if (is_custom_)
                trees.push_back(sample.sentence.heads);
        }

        auto hid = is_custom_ ? encoder.encode_batch(cg, embedded_sents, trees)
                              : encoder.encode_batch(cg, embedded_sents);

        vector<Expression> out;

        for (auto && h : hid)
        {
            h = dy::affine_transform({out_b, out_W, h});
            out.push_back(h);
        }

        // return non-normalized probabilities, it's ok
        return out;
    }

    virtual
    Expression
    batch_loss(
        ComputationGraph& cg,
        const SentBatch& batch)
    override
    {
        auto out = predict_batch(cg, batch);

        vector<Expression> losses;
        for(int i = 0; i < batch.size(); ++i)
        {
            auto loss = dy::pickneglogsoftmax(out[i], batch[i].target);
            losses.push_back(loss);
        }

        return dy::sum(losses);
    }
};


/**
 * \brief TreeLSTM sentence classifier with latent (inferred) tree.
 */
struct LatentDepSentClf : public BaseSentClf
{
    LatentTreeEncoder encoder;
    float test_temperature_;

    explicit LatentDepSentClf(
        ParameterCollection& params,
        unsigned vocab_size,
        unsigned embed_dim,
        unsigned hidden_dim,
        unsigned n_classes,
        bool update_embed,
        size_t max_iter,
        std::string scorer_type)
    : BaseSentClf(
        params,
        vocab_size,
        embed_dim,
        hidden_dim,
        n_classes,
        update_embed)
    , encoder(p, hidden_dim, max_iter, scorer_type)
    { }

    virtual
    void
    set_print_trees(
        std::shared_ptr<std::ostream> out)
    override
    {
        encoder.print_trees = true;
        encoder.out_trees = out;
    }

    virtual
    void
    set_temperature(float t)
    {
        test_temperature_ = t;
    }


    virtual
    vector<Expression>
    predict_batch(
        dy::ComputationGraph& cg,
        const SentBatch& batch)
    override
    {
        enc_rnn_fw->new_graph(cg);
        enc_rnn_bw->new_graph(cg);

        auto out_b = parameter(cg, p_out_b);
        auto out_W = parameter(cg, p_out_W);

        vector<vector<Expression> > embedded_sents;
        for (auto && sample : batch)
            embedded_sents.push_back(embed_ctx_sent(cg, sample.sentence));

        vector<vector<Expression> > encoded;
        Expression y_post;
        std::tie(y_post, encoded) = encoder.encode_batch(cg,
                                                         embedded_sents,
                                                         test_temperature_);

        vector<Expression> out;

        for(int i = 0; i < batch.size(); ++i)
        {
            for(auto && enc : encoded[i])
            {
                enc = dy::affine_transform({out_b, out_W, enc});
                enc = dy::softmax(enc);
            }

            auto weights = dy::pick(y_post, i, 1);
            weights = dy::pick_range(weights, 0, encoded[i].size());
            auto comb = dy::concatenate_cols(encoded[i]) * weights;

            out.push_back(comb);
        }

        // this time they are normalized
        return out;
    }

    virtual
    vector<Expression>
    predict_batch_map(
        dy::ComputationGraph& cg,
        const SentBatch& batch)
    {
        enc_rnn_fw->new_graph(cg);
        enc_rnn_bw->new_graph(cg);

        auto out_b = parameter(cg, p_out_b);
        auto out_W = parameter(cg, p_out_W);

        vector<vector<Expression> > embedded_sents;
        for (auto && sample : batch)
            embedded_sents.push_back(embed_ctx_sent(cg, sample.sentence));

        auto encoded = encoder.encode_batch_map(cg, embedded_sents);

        vector<Expression> out;
        for (auto && h : encoded)
        {
            h = dy::affine_transform({out_b, out_W, h});
            out.push_back(h);
        }

        return out;
    }

    virtual
    Expression
    batch_loss(
        dy::ComputationGraph& cg,
        const SentBatch& batch)
    override
    {
        using dy::concatenate;
        using dy::dot_product;
        using dy::affine_transform;
        using dy::pickneglogsoftmax;
        using dy::pick;
        using dy::pickrange;

        enc_rnn_fw->new_graph(cg);
        enc_rnn_bw->new_graph(cg);

        auto out_b = parameter(cg, p_out_b);
        auto out_W = parameter(cg, p_out_W);

        vector<vector<Expression> > embedded_sents;
        for (auto && sample : batch)
            embedded_sents.push_back(embed_ctx_sent(cg, sample.sentence));

        vector<vector<Expression> > encoded;
        Expression y_post;
        std::tie(y_post, encoded) = encoder.encode_batch(cg, embedded_sents);

        vector<Expression> losses;

        for(int i = 0; i < batch.size(); ++i)
        {
            vector<Expression> latent_losses;
            for(auto&& enc : encoded[i])
            {
                enc = affine_transform({out_b, out_W, enc});
                auto loss = pickneglogsoftmax(enc, batch[i].target);
                latent_losses.push_back(loss);
            }

            auto weights = pick(y_post, i, 1);
            weights = pick_range(weights, 0, encoded[i].size());

            auto loss = dot_product(concatenate(latent_losses), weights);
            losses.push_back(loss);
        }

        return dy::sum(losses);
    }
};
