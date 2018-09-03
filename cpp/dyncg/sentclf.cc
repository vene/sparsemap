/* TreeLSTM sentiment classifier with latent trees
 * Author: vlad@vene.ro
 * License: Apache 2.0
 */

#include <iostream>
#include <algorithm>

#include <dynet/nodes.h>
#include <dynet/dynet.h>
#include <dynet/dict.h>
#include <dynet/expr.h>
#include <dynet/globals.h>
#include <dynet/io.h>
#include <dynet/timing.h>
#include <dynet/training.h>

#include "../common/utils.h"
#include "../common/args.h"

#include "sentclf.h"

namespace dy = dynet;

using std::vector;
using std::cout;
using std::endl;

/**
 * \brief Compute validation accuracy of model on passed batches.
 * \param clf A trained instance fo BaseSentClf
 * \param data A batched validation dataset
 * \return Accuracy score.
 */
float
validate(
    std::unique_ptr<BaseSentClf>& clf,
    const vector<SentBatch>& data)
{
    int n_correct = 0;
    int n_total = 0;
    for (auto&& valid_batch : data)
    {
        dy::ComputationGraph cg;
        n_correct += clf->n_correct(cg, valid_batch);
        n_total += valid_batch.size();
    }

    return float(n_correct) / n_total;
}


/**
 * \brief Print test accuracy of model.
 * \param clf An appropriate instance of BaseSentClf (untrained)
 * \param args Command-line args (used to find saved model)
 * \param test_fn Path to test data file
 */
void
test(
    std::unique_ptr<BaseSentClf>& clf,
    const SentClfArgs& args,
    const std::string& test_fn)
{
    clf->load(args.saved_model);

    auto test_data = read_batches<LabeledSentence>(test_fn, args.batch_size);
    float acc = validate(clf, test_data);
    cout << "Test accuracy: " << acc << endl;
}

/**
 * \brief Print test accuracy of latent tree model, after tuning temperature.
 * \param clf An appropriate instance of BaseSentClf (untrained)
 * \param args Command-line args (used to find saved model)
 * \param valid_fn Path to validation data file
 * \param test_fn Path to test data file
 */
void
test_temp(
    std::unique_ptr<BaseSentClf>& clf,
    const SentClfArgs& args,
    const std::string& valid_fn,
    const std::string& test_fn)
{
    clf->load(args.saved_model);

    // pick best temperature on a log grid

    float best_acc = 0;
    float best_temp = 0;

    auto valid_data = read_batches<LabeledSentence>(valid_fn, args.batch_size);
    for (int t = -8; t <= 8; ++t)
    {
        float temperature = std::pow(2.0f, t);
        clf->set_temperature(temperature);
        float acc = validate(clf, valid_data);
        cout << "Temperature " << temperature << ":\t valid " << acc << endl;
        if (acc > best_acc)
        {
            best_acc = acc;
            best_temp = temperature;
        }
    }

    clf->set_temperature(best_temp);

    std::shared_ptr<std::ostream> trees_out;
    if (args.latent && args.print_trees)
    {
        trees_out.reset(new std::ofstream(args.out_trees_fn));
        clf->set_print_trees(trees_out);
    }

    // predict again but with printing
    validate(clf, valid_data);

    (*trees_out) << "***TEST***" << endl;

    auto test_data = read_batches<LabeledSentence>(test_fn, args.batch_size);
    float acc = validate(clf, test_data);
    cout << "Test accuracy: " << acc << endl;

    trees_out->flush();
}


/**
 * \brief Train a model using supplied configuration.
 * \param clf An appropriate instance of BaseSentClf (untrained)
 * \param args Command-line args specifying training configuration.
 * \param valid_fn Path to validation data file
 * \param test_fn Path to test data file
 */
void
train(
    std::unique_ptr<BaseSentClf>& clf,
    const SentClfArgs& args,
    const std::string& train_fn,
    const std::string& valid_fn)
{
    auto train_data = read_batches<LabeledSentence>(train_fn, args.batch_size);
    auto valid_data = read_batches<LabeledSentence>(valid_fn, args.batch_size);
    auto n_batches = train_data.size();

    unsigned n_train_sents = 0;
    for (auto&& batch : train_data)
        n_train_sents += batch.size();

    std::cout << "Training on " << n_train_sents << " sentences." << std::endl;

    // make an identity permutation vector of pointers into the batches
    vector<vector<SentBatch>::iterator> train_iter(n_batches);
    std::iota(train_iter.begin(), train_iter.end(), train_data.begin());

    dy::SimpleSGDTrainer trainer(clf->p, args.lr);

    float best_valid_acc = 0;
    unsigned impatience = 0;

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
                auto loss = clf->batch_loss(cg, *batch);
                total_loss += dy::as_scalar(cg.incremental_forward(loss));
                cg.backward(loss);
                trainer.update();
            }
        }

        float valid_acc;
        {
            std::unique_ptr<dy::Timer> timer(new dy::Timer("valid took"));
            valid_acc = validate(clf, valid_data);
        }

        std::cout << "training loss "
                  << total_loss / n_train_sents
                  << " valid accuracy " << valid_acc << std::endl;

        if ((valid_acc + 0.0001) > best_valid_acc)
        {
            impatience = 0;
            best_valid_acc = valid_acc;

            std::ostringstream fn;
            fn << args.save_prefix
               << "sentclf_"
               << args.get_filename()
               << "_acc_"
               << std::internal << std::setfill('0')
               << std::fixed << std::setprecision(2) << std::setw(5)
               << valid_acc * 100.0
               << "_iter_" << std::setw(3) << it
               << ".dy";
            clf->save(fn.str());
        }
        else
        {
            trainer.learning_rate *= args.decay;
            cout << "Decaying LR to " << trainer.learning_rate << endl;
            impatience += 1;
        }

        if (impatience > args.patience)
        {
            cout << args.patience << " epochs without improvement." << endl;
            return;
        }
    }
}


int main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);

    SentClfArgs args;
    args.parse(argc, argv);
    std::cout << args << std::endl;

    if (args.override_dy)
    {
        dyparams.random_seed = 42;
        dyparams.autobatch = true;
    }
    dy::initialize(dyparams);

    unsigned EMBED_DIM = 300;

    std::stringstream vocab_fn, train_fn, valid_fn, test_fn, embed_fn, class_fn;
    vocab_fn << "data/sentclf/" << args.dataset << ".vocab";
    class_fn << "data/sentclf/" << args.dataset << ".classes";
    embed_fn << "data/sentclf/" << args.dataset << ".embed";

    if (args.override_train_fn)
        train_fn << args.train_fn;
    else
        train_fn << "data/sentclf/" << args.dataset << ".train.txt";

    if (args.override_valid_fn)
        valid_fn << args.valid_fn;
    else
        valid_fn << "data/sentclf/" << args.dataset << ".valid.txt";

    if (args.override_test_fn)
        test_fn << args.test_fn;
    else
        test_fn  << "data/sentclf/" << args.dataset << ".test.txt";

    unsigned vocab_size = line_count(vocab_fn.str());
    cout << "vocabulary size: " << vocab_size << endl;

    unsigned n_classes = line_count(class_fn.str());
    cout << "number of classes: " << n_classes << endl;

    dy::ParameterCollection params;

    std::unique_ptr<BaseSentClf> clf = nullptr;

    if (args.latent)
        clf.reset(new LatentDepSentClf(params,
                                       vocab_size,
                                       EMBED_DIM,
                                       args.hidden_dim,
                                       n_classes,
                                       args.update_embed,
                                       args.max_decode_iter,
                                       args.scorer_str));
    else
        clf.reset(new FixedDepSentClf(params,
                                      vocab_size,
                                      EMBED_DIM,
                                      args.hidden_dim,
                                      n_classes,
                                      args.get_strategy(),
                                      args.update_embed));
    clf->load_embeddings(embed_fn.str());

    if (args.test)
    {
        if (args.latent)
            test_temp(clf, args, valid_fn.str(), test_fn.str());
        else
            test(clf, args, test_fn.str());
    }
    else
        train(clf, args, train_fn.str(), valid_fn.str());

    return 0;
}
