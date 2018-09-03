# pragma once

#include <dynet/devices.h>
#include "../common/basemodel.h"
#include "tree_encoders.h"

namespace dy = dynet;

using dy::Expression;
using dy::Parameter;
using dy::LookupParameter;
using dy::parameter;

using std::vector;


struct BaseRevdict : public BaseEmbedBiLSTMModel
{
    vector<unsigned> shortlist_;
    dy::Tensor shortlist_emb_;
    Parameter p_out_W;

    #if USING_CUDA
    dy::Device* cpu_device = dy::get_device_manager()->get_global_device("CPU");
    #endif

    explicit BaseRevdict(
        ParameterCollection& params,
        unsigned vocab_size,
        unsigned embed_dim,
        unsigned hidden_dim)
        : BaseEmbedBiLSTMModel(
            params,
            vocab_size,
            embed_dim,
            hidden_dim,
            false,
            "revdict")
    {
        p_out_W = p.add_parameters({embed_dim, hidden_dim});
    }

    void load_shortlist(const std::string filename)
    {
        std::cerr << "~ ~ loading shortlist... ~ ~ ";
        std::ifstream in(filename);
        assert(in);
        unsigned k;
        while (in) {
            in >> k;
            if (!in) continue;
            shortlist_.push_back(k);
        }
        std::cerr << "size=" << shortlist_.size() << std::endl;

        /*
         *  TODO: here I would like to be able to cache the shortlist TENSOR
         *  (on whatever device is selected)
         *  and simply load it into a new computation graph when rank() is
         *  called, ideally with **zero copy**.
         */
        /*
        {
            dy::ComputationGraph cg;
            vector<Expression> shortlist_emb_v;
            for (auto& k : shortlist_)
                shortlist_emb_v.push_back(dy::const_lookup(cg, p_emb, k));
            auto shortlist_emb = dy::concatenate_cols(shortlist_emb_v);
            shortlist_emb_ = shortlist_emb.value();
        }
        */
    }

    void rank(dy::ComputationGraph& cg,
              const DefBatch& batch,
              vector<int>& ranks)
    {
        vector<Expression> shortlist_emb_v;
        for (auto& k : shortlist_)
            shortlist_emb_v.push_back(dy::const_lookup(cg, p_emb, k));
        auto shortlist_emb = dy::concatenate_cols(shortlist_emb_v);

        /* SEE COMMENT ABOVE
        auto shortlist_emb = dy::input(cg, shortlist_emb_.d, shortlist_emb_.v, shortlist_emb_.device);
        */
        vector<Expression> out_v = predict_batch(cg, batch);

        vector<unsigned> tgt_ix;
        for (auto&& sent : batch)
            tgt_ix.push_back(sent.target);

        auto out = dy::concatenate_to_batch(out_v);
        auto tgt = dy::const_lookup(cg, p_emb, tgt_ix);
        auto expected_scores = dy::dot_product(out, tgt);
        auto shortlist_scores = dy::transpose(out) * shortlist_emb;

        #if USING_CUDA
        expected_scores = dy::to_device(expected_scores, cpu_device);
        shortlist_scores = dy::to_device(shortlist_scores, cpu_device);
        #endif

        cg.incremental_forward(expected_scores);
        cg.incremental_forward(shortlist_scores);

        auto exp_s = dy::vec(expected_scores.value());
        auto sls_s = dy::tbvec(shortlist_scores.value()); // shortlist x batch_sz

        for (int i = 0; i < batch.size(); ++i)
        {
            int rank = 0;
            for (int j = 0; j < shortlist_.size(); ++j)
                if (sls_s(j, i) > exp_s(i))
                    rank += 1;
            ranks.push_back(rank);
        }
    }

    Expression batch_loss(dy::ComputationGraph& cg,
                          const DefBatch& batch)
    {
        vector<Expression> out_v = predict_batch(cg, batch);

        vector<unsigned> tgt_ix;
        for (auto&& sent : batch)
            tgt_ix.push_back(sent.target);

        auto out = dy::concatenate_to_batch(out_v);
        auto tgt = dy::const_lookup(cg, p_emb, tgt_ix);
        auto loss = -dy::sum_batches(dy::dot_product(out, tgt));
        return loss;
    }

    void
    load_embeddings(
        const std::string filename,
        bool normalize=true)
    {
        std::cerr << "~ ~ loading embeddings... ~ ~ ";
        std::ifstream in(filename);
        assert(in);
        std::string line;
        vector<float> embed(embed_dim_);
        unsigned ix = 0;

        while (getline(in, line))
        {
            std::istringstream lin(line);

            for (int i = 0; i < embed_dim_; ++i)
                lin >> embed[i];

            if (normalize) normalize_vector(embed);

            p_emb.initialize(ix, embed);
            ix += 1;
        }
        std::cerr << "done." << std::endl;
    }

    void set_temperature(float t) { test_temperature_ = t; }

    protected:
    virtual vector<Expression> predict_batch(dy::ComputationGraph& cg,
                                             const DefBatch& batch) = 0;
    float test_temperature_ = 1;

};


struct FixedDepRevdict : public BaseRevdict
{

    FixedTreeEncoder encoder;
    bool is_custom_;

    explicit FixedDepRevdict(dy::ParameterCollection& params,
                             unsigned vocab_size,
                             unsigned embed_dim,
                             unsigned hidden_dim,
                             ChildSumTreeLSTMBuilder::Strategy strategy)
        : BaseRevdict(params, vocab_size, embed_dim, hidden_dim)
        , encoder(p, hidden_dim, strategy)
        , is_custom_(strategy == ChildSumTreeLSTMBuilder::Strategy::CUSTOM)
    { }



    virtual
    vector<Expression>
    predict_batch(
        dy::ComputationGraph& cg,
        const DefBatch& batch)
    {
        enc_rnn_fw->new_graph(cg);
        enc_rnn_bw->new_graph(cg);
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

        vector<Expression> out_v;

        for (auto && h : hid)
        {
            auto out_one = out_W * h;
            out_one = dy::cdiv(out_one, dy::l2_norm(out_one));
            out_v.push_back(out_one);
        }
        return out_v;
    }
};


struct LatentDepRevdict : public BaseRevdict
{
    LatentTreeEncoder encoder;

    explicit
    LatentDepRevdict(
        dy::ParameterCollection& params,
        unsigned vocab_size,
        unsigned embed_dim,
        unsigned hidden_dim,
        size_t max_iter,
        std::string scorer_type)
        : BaseRevdict(params, vocab_size, embed_dim, hidden_dim)
        , encoder(p, hidden_dim, max_iter, scorer_type)
    { }

    protected:

    vector<Expression>
    predict_batch(
        dy::ComputationGraph& cg,
        const DefBatch& batch)
    override
    {
        enc_rnn_fw->new_graph(cg);
        enc_rnn_bw->new_graph(cg);
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
        for (int i = 0; i < batch.size(); ++i)
        {
            for (auto&& enc : encoded[i])
            {
                enc = out_W * enc;
                enc = dy::cdiv(enc, dy::l2_norm(enc));
            }
            auto weights = dy::pick(y_post, i, 1);
            weights = dy::pick_range(weights, 0, encoded[i].size());
            out.push_back(dy::concatenate_cols(encoded[i]) * weights);
        }
        return out;
    }
};
