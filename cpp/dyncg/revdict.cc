#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/dict.h"
#include "dynet/expr.h"
#include "dynet/globals.h"
#include "dynet/io.h"
#include "dynet/timing.h"
#include "dynet/training.h"

#include <iostream>
#include <algorithm>

#include "../common/utils.h"
#include "../common/args.h"

#include "revdict.h"

namespace dy = dynet;

using std::vector;
using std::cout;
using std::endl;


void validate(std::unique_ptr<BaseRevdict>& revdict,
              const vector<DefBatch>& data,
              float& median_rank,
              float& acc_at_10,
              float& acc_at_100)
{
    vector<int> ranks;
    for (auto&& valid_batch : data)
    {
        dy::ComputationGraph cg;
        revdict->rank(cg, valid_batch, ranks);
    }

    acc_at_10 = 0;
    acc_at_100 = 0;

    for (auto&& r : ranks)
    {
        if (r < 10)
            acc_at_10 += 1;
        if (r < 100)
            acc_at_100 += 1;
    }

    acc_at_10 /= ranks.size();
    acc_at_100 /= ranks.size();

    auto mid_iter = ranks.begin() + ranks.size() / 2;
    std::nth_element(ranks.begin(), mid_iter, ranks.end());
    median_rank = *mid_iter;
}


void test(std::unique_ptr<BaseRevdict>& revdict,
          const RevdictArgs& args)
{
    revdict->load(args.saved_model);

    auto test_filenames = { "data/revdict/WN_seen_correct.txt",
                            "data/revdict/WN_unseen_correct.txt",
                            "data/revdict/concept_descriptions.txt" };

    for (auto&& fn : test_filenames)
    {
        std::cout << fn << std::endl;
        auto test_data = read_batches<Definition>(fn, args.batch_size);

        float rank, acc_at_10, acc_at_100;
        validate(revdict, test_data, rank, acc_at_10, acc_at_100);
        std::cout << "      Median rank: " << rank << std::endl
                  << "   Accuracy at 10: " << acc_at_10 << std::endl
                  << "  Accuracy at 100: " << acc_at_100 << std::endl
                  << std::endl;
    }
}


void test_temp(std::unique_ptr<BaseRevdict>& revdict,
               const RevdictArgs& args)
{
    revdict->load(args.saved_model);

    float best_rank = 9999;
    float best_temp = 0;

    auto valid_data = read_batches<Definition>("data/revdict/valid.txt",
                                               args.batch_size);

    for (int t = -8; t <= 8; ++t)
    {
        float temperature = std::pow(2.0f, t);
        revdict->set_temperature(temperature);
        float rank, acc_at_10, acc_at_100;
        validate(revdict, valid_data, rank, acc_at_10, acc_at_100);
        cout << "Temperature " << temperature
             << ":\t valid rank " << rank
             << "acc@10 " << acc_at_10
             << "acc@100 " << acc_at_100 << endl;

        if (rank <= best_rank)
        {
            best_rank = rank;
            best_temp = temperature;
        }
    }

    revdict->set_temperature(best_temp);
    auto test_filenames = { "data/revdict/WN_seen_correct.txt",
                            "data/revdict/WN_unseen_correct.txt",
                            "data/revdict/concept_descriptions.txt" };

    for (auto&& fn : test_filenames)
    {
        std::cout << fn << std::endl;
        auto test_data = read_batches<Definition>(fn, args.batch_size);

        float rank, acc_at_10, acc_at_100;
        validate(revdict, test_data, rank, acc_at_10, acc_at_100);
        std::cout << "      Median rank: " << rank << std::endl
                  << "   Accuracy at 10: " << acc_at_10 << std::endl
                  << "  Accuracy at 100: " << acc_at_100 << std::endl
                  << std::endl;
    }
}


void train(std::unique_ptr<BaseRevdict>& revdict,
           const RevdictArgs& args)
{
    auto train_data = read_batches<Definition>("data/revdict/train.txt",
                                               args.batch_size);
    auto valid_data = read_batches<Definition>("data/revdict/valid.txt",
                                               args.batch_size);
    auto n_batches = train_data.size();

    unsigned n_train_sents = 0;
    for (auto&& batch : train_data)
        n_train_sents += batch.size();

    unsigned n_valid_sents = 0;
    for (auto&& batch : valid_data)
        n_valid_sents += batch.size();

    // make an identity permutation vector of pointers into the batches
    vector<vector<DefBatch>::iterator> train_iter(n_batches);
    std::iota(train_iter.begin(), train_iter.end(), train_data.begin());

    dy::SimpleSGDTrainer trainer(revdict->p, args.lr);
    int best_rank = 99999;
    int patience = 0;

    for (unsigned it = 0; it < args.max_iter; ++it)
    {

        // shuffle the permutation vector
        std::shuffle(train_iter.begin(), train_iter.end(), *dy::rndeng);

        float total_loss = 0;

        {
            std::unique_ptr<dy::Timer> timer(new dy::Timer("train took"));
            for (auto&& batch : train_iter)
            {
                dy::ComputationGraph cg;
                auto loss = revdict->batch_loss(cg, *batch);
                total_loss += dy::as_scalar(cg.incremental_forward(loss));
                cg.backward(loss);
                trainer.update();
            }
        }

        std::cout << "training loss "
                  << total_loss / n_train_sents
                  << std::endl;

        float rank, acc_10, acc_100;

        {
            std::unique_ptr<dy::Timer> timer(new dy::Timer("valid took"));
            validate(revdict, valid_data, rank, acc_10, acc_100);
        }

        std::cout << " median rank " << rank << std::endl
                  << "      acc@10 " << acc_10 << std::endl
                  << "     acc@100 " << acc_100 << std::endl << std::endl;

        if (rank <= best_rank)
        {
            patience = 0;
            best_rank = rank;

            std::ostringstream fn;
            fn << args.save_prefix
               << "revdict"
               << args.get_filename()
               << "_rank_" << rank
               << "_iter_" << it
               << ".dy";
            revdict->save(fn.str());
        }
        else
        {
            trainer.learning_rate *= args.decay;
            std::cout << "Decay to " << trainer.learning_rate << std::endl;
            patience += 1;
        }
        if (patience > args.patience)
        {
            std::cout << "5 epochs without improvement, stopping." << std::endl;
            return;
        }
    }
}


int main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);
    RevdictArgs args;
    args.parse(argc, argv);
    std::cout << args << std::endl;

    if (args.override_dy)
    {
        dyparams.random_seed = 42;
        dyparams.autobatch = true;
    }
    dy::initialize(dyparams);

    // the only thing fixed in this model
    unsigned EMBED_DIM = 500;


    unsigned vocab_size = line_count("data/revdict/vocab.txt");
    cout << "vocabulary size " << vocab_size << endl;

    dy::ParameterCollection params;

    std::unique_ptr<BaseRevdict> revdict = nullptr;
    if (args.latent)
        revdict.reset(new LatentDepRevdict(params,
                                           vocab_size,
                                           EMBED_DIM,
                                           args.hidden_dim,
                                           args.max_decode_iter,
                                           args.scorer_str));
    else
        revdict.reset(new FixedDepRevdict(params,
                                          vocab_size,
                                          EMBED_DIM,
                                          args.hidden_dim,
                                          args.get_strategy()));
    revdict->load_shortlist("data/revdict/shortlist.txt");
    revdict->load_embeddings("data/revdict/embed.txt", true);

    if (args.test)
        if (args.latent)
            test_temp(revdict, args);
        else
            test(revdict, args);
    else
        train(revdict, args);

    return 0;
}
