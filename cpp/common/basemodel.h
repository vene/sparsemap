/* Common classes for dynet NLP models.
 * Author: vlad@vene.ro
 * License: Apache 2.0
 */

#include <dynet/io.h>
#include <dynet/rnn.h>
#include <dynet/lstm.h>

#include <string>
#include <fstream>

namespace dy = dynet;

/**
 * \brief Basic dynet model with save and load functionality.
 */
struct BaseModel
{
    dy::ParameterCollection p;

    void save(const std::string filename)
    {
        dy::TextFileSaver s(filename);
        s.save(p);
    }

    void load(const std::string filename)
    {
        dy::TextFileLoader l(filename);
        l.populate(p);
    }
};


/**
 * \brief Basic model for NLP: includes embeddings and bi-LSTM
 */
struct BaseEmbedBiLSTMModel : BaseModel
{
    dy::LookupParameter p_emb;
    std::unique_ptr<dy::RNNBuilder> enc_rnn_fw;
    std::unique_ptr<dy::RNNBuilder> enc_rnn_bw;

    unsigned vocab_size_;
    unsigned embed_dim_;
    unsigned hidden_dim_;

    bool update_embed_;

    explicit
    BaseEmbedBiLSTMModel(
        dy::ParameterCollection& params,
        unsigned vocab_size,
        unsigned embed_dim,
        unsigned hidden_dim,
        bool update_embed,
        std::string model_name="model")
    : vocab_size_(vocab_size)
    , embed_dim_(embed_dim)
    , hidden_dim_(hidden_dim)
    , update_embed_(update_embed)
    {
        p = params.add_subcollection(model_name);

        if (update_embed)
            p_emb = p.add_lookup_parameters(vocab_size_, {embed_dim_});
        else
            p_emb = params.add_lookup_parameters(vocab_size_, {embed_dim_});

        assert(hidden_dim_ % 2 == 0);

        const int ly = 1; // layers
        enc_rnn_fw.reset(
            new dy::VanillaLSTMBuilder(ly, embed_dim_, hidden_dim_ / 2, p));
        enc_rnn_bw.reset(
            new dy::VanillaLSTMBuilder(ly, embed_dim_, hidden_dim_ / 2, p));
    }

    /**
     * \brief Load pretrained embeddings from disk
     * \param filename Text file with space-separated embeddings, one per line.
     * \param normalize Whether to normalize (L2) after loading.
     */
    void
    load_embeddings(
        const std::string filename,
        bool normalize=false)
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

    /**
     * \brief Encode a sentence through embeddings and bi-LSTM.
     * \param cg Current computation graph.
     * \param sent Sentence to encode
     * \return Encoded vector expressions, one per word.
     */
    std::vector<dy::Expression>
    embed_ctx_sent(
        dy::ComputationGraph& cg,
        const Sentence& sent)
    {
        std::vector<dy::Expression> embeds(sent.size());
        std::vector<dy::Expression> enc_fw(sent.size());
        std::vector<dy::Expression> enc_bw(sent.size());
        std::vector<dy::Expression> enc(sent.size());

        enc_rnn_fw->start_new_sequence();
        enc_rnn_bw->start_new_sequence();

        auto sent_sz = sent.size();

        for (size_t i = 0; i < sent_sz; ++i)
        {
            auto w = sent.word_ixs[i];
            embeds[i] = update_embed_ ? dy::lookup(cg, p_emb, w)
                                      : dy::const_lookup(cg, p_emb, w);
        }

        for (size_t i = 0; i < sent_sz; ++i)
        {
            size_t j = sent_sz - i - 1;
            enc_fw[i] = enc_rnn_fw->add_input(embeds[i]);
            enc_bw[j] = enc_rnn_bw->add_input(embeds[j]);
        }

        for (size_t i = 0; i < sent_sz; ++i)
            enc[i] = dy::concatenate({enc_fw[i], enc_bw[i]});

        return enc;
    }
};
