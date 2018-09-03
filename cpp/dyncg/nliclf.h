# pragma once

/*
 * RepEval-style NLI classifiers.
 * Independently encode the premise and hypothesis with a shared encoder
 * Then use a 2-layer MLP predictive head.
 */

#include "../common/basenli.h"
#include "tree_encoders.h"


struct BaseRepNLI : public BaseNLI
{
    Parameter p_hid_W, p_hid_b;
    Parameter p_out_W, p_out_b;

    explicit
    BaseRepNLI(
        ParameterCollection& params,
        unsigned vocab_size,
        unsigned embed_dim,
        unsigned hidden_dim,
        bool update_embed,
        unsigned n_classes)
    : BaseNLI(
        params,
        vocab_size,
        embed_dim,
        hidden_dim,
        update_embed)
    {
        p_hid_W = p.add_parameters({hidden_dim, 4 * hidden_dim});
        p_hid_b = p.add_parameters({hidden_dim});
        p_out_W = p.add_parameters({n_classes, hidden_dim});
        p_out_b = p.add_parameters({n_classes});
    }
};


struct FixedDepNLI : public BaseRepNLI
{
    FixedTreeEncoder encoder;
    bool is_custom_;

    explicit
    FixedDepNLI(
        ParameterCollection& params,
        unsigned vocab_size,
        unsigned embed_dim,
        unsigned hidden_dim,
        unsigned n_classes,
        ChildSumTreeLSTMBuilder::Strategy strategy,
        bool update_embed=true)
    : BaseRepNLI(
        params,
        vocab_size,
        embed_dim,
        hidden_dim,
        update_embed,
        n_classes)
    , encoder(p, hidden_dim, strategy)
    , is_custom_(strategy == ChildSumTreeLSTMBuilder::Strategy::CUSTOM)
    { }

    virtual
    vector<Expression>
    predict_batch(
        ComputationGraph& cg,
        const NLIBatch& batch)
    override
    {
        enc_rnn_fw->new_graph(cg);
        enc_rnn_bw->new_graph(cg);
        auto hid_b = parameter(cg, p_hid_b);
        auto hid_W = parameter(cg, p_hid_W);
        auto out_b = parameter(cg, p_out_b);
        auto out_W = parameter(cg, p_out_W);

        vector<vector<Expression> > embedded_sents;
        vector<vector<int> > trees;

        for (auto && sample : batch)
        {
            embedded_sents.push_back(embed_ctx_sent(cg, sample.prem));
            embedded_sents.push_back(embed_ctx_sent(cg, sample.hypo));

            if (is_custom_)
            {
                trees.push_back(sample.prem.heads);
                trees.push_back(sample.hypo.heads);
            }
        }

        auto hid = is_custom_ ? encoder.encode_batch(cg, embedded_sents, trees)
                              : encoder.encode_batch(cg, embedded_sents);

        vector<Expression> out;
        for (int k = 0; k < batch.size(); ++k)
        {
            auto prem_v = hid[2 * k];
            auto hypo_v = hid[1 + 2 * k];

            auto pair = dy::concatenate(
                {prem_v,
                 hypo_v,
                 prem_v - hypo_v,
                 dy::cmult(prem_v, hypo_v)});

            auto hid = dy::affine_transform({hid_b, hid_W, pair});
            hid = dy::tanh(hid);
            out.push_back(dy::affine_transform({out_b, out_W, hid}));
        }

        // return non-normalized probabilities
        return out;
    }

};


struct LatentDepNLI : public BaseRepNLI
{
    LatentTreeEncoder encoder;

    explicit
    LatentDepNLI(
        ParameterCollection& params,
        unsigned vocab_size,
        unsigned embed_dim,
        unsigned hidden_dim,
        unsigned n_classes,
        bool update_embed,
        size_t max_iter,
        std::string scorer_type)
    : BaseRepNLI(
        params,
        vocab_size,
        embed_dim,
        hidden_dim,
        update_embed,
        n_classes)
    , encoder(p, hidden_dim, max_iter, scorer_type)
    { }

    virtual
    vector<Expression>
    predict_batch(
        dy::ComputationGraph& cg,
        const NLIBatch& batch)
    override
    {
        enc_rnn_fw->new_graph(cg);
        enc_rnn_bw->new_graph(cg);
        auto hid_b = parameter(cg, p_hid_b);
        auto hid_W = parameter(cg, p_hid_W);
        auto out_b = parameter(cg, p_out_b);
        auto out_W = parameter(cg, p_out_W);

        vector<vector<Expression> > embedded_sents;

        for (auto && sample : batch)
        {
            embedded_sents.push_back(embed_ctx_sent(cg, sample.prem));
            embedded_sents.push_back(embed_ctx_sent(cg, sample.hypo));
        }

        vector<vector<Expression> > enc;
        Expression y_post;
        std::tie(y_post, enc) = encoder.encode_batch(cg, embedded_sents, test_temperature_);

        vector<Expression> out;
        for (int k = 0; k < batch.size(); ++k)
        {
            vector<Expression> col;
            for (auto&& prem_v : enc[2 * k])
            {
                vector<Expression> row;
                for (auto&& hypo_v : enc[1 + 2 * k])
                {
                    auto pair = dy::concatenate(
                        {prem_v,
                         hypo_v,
                         prem_v - hypo_v,
                         dy::cmult(prem_v, hypo_v)});

                    auto hid = dy::affine_transform({hid_b, hid_W, pair});
                    hid = dy::tanh(hid);
                    hid = dy::affine_transform({out_b, out_W, hid});
                    hid = dy::softmax(hid);
                    row.push_back(hid);
                }
                auto weights = dy::pick(y_post, 1 + 2 * k, 1);
                weights = dy::pick_range(weights, 0, enc[1 + 2 * k].size());
                col.push_back(dy::concatenate_cols(row) * weights);
            }
            auto weights = dy::pick(y_post, 2 * k, 1);
            weights = dy::pick_range(weights, 0, enc[2 * k].size());
            out.push_back(dy::concatenate_cols(col) * weights);
        }

        // this time they are normalized
        return out;
    }

    virtual
    Expression
    batch_loss(
        dy::ComputationGraph& cg,
        const NLIBatch& batch)
    override
    {
        enc_rnn_fw->new_graph(cg);
        enc_rnn_bw->new_graph(cg);
        auto hid_b = parameter(cg, p_hid_b);
        auto hid_W = parameter(cg, p_hid_W);
        auto out_b = parameter(cg, p_out_b);
        auto out_W = parameter(cg, p_out_W);

        vector<vector<Expression> > embedded_sents;

        for (auto && sample : batch)
        {
            embedded_sents.push_back(embed_ctx_sent(cg, sample.prem));
            embedded_sents.push_back(embed_ctx_sent(cg, sample.hypo));
        }

        vector<vector<Expression> > enc;
        Expression y_post;
        std::tie(y_post, enc) = encoder.encode_batch(cg, embedded_sents);

        vector<Expression> losses;

        for(int k = 0; k < batch.size(); ++k)
        {
            vector<Expression> col;
            for (auto&& prem_v : enc[2 * k])
            {
                vector<Expression> row;
                for (auto&& hypo_v : enc[1 + 2 * k])
                {
                    auto pair = dy::concatenate(
                        {prem_v,
                         hypo_v,
                         prem_v - hypo_v,
                         dy::cmult(prem_v, hypo_v)});

                    auto hid = dy::affine_transform({hid_b, hid_W, pair});
                    hid = dy::tanh(hid);
                    hid = dy::affine_transform({out_b, out_W, hid});
                    auto loss = dy::pickneglogsoftmax(hid, batch[k].target);
                    row.push_back(loss);
                }
                auto weights = dy::pick(y_post, 1 + 2 * k, 1);
                weights = dy::pick_range(weights, 0, enc[1 + 2 * k].size());
                col.push_back(dy::dot_product(dy::concatenate(row), weights));
            }
            auto weights = dy::pick(y_post, 2 * k, 1);
            weights = dy::pick_range(weights, 0, enc[2 * k].size());
            losses.push_back(
                dy::dot_product(dy::concatenate(col), weights));
        }

        return dy::sum(losses);
    }
};
