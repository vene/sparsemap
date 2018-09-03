# pragma once

#include <dynet/training.h>
#include <dynet/timing.h>
#include <dynet/tensor.h>
#include <dynet/index-tensor.h>
#include <dynet/devices.h>

#include <string>

#include "utils.h"
#include "args.h"
#include "basemodel.h"

namespace dy = dynet;

using dy::Expression;
using dy::Parameter;
using dy::LookupParameter;
using dy::ParameterCollection;
using dy::ComputationGraph;

using std::vector;
using std::cout;
using std::endl;

struct BaseNLI : public BaseEmbedBiLSTMModel
{
    #if USING_CUDA
    dy::Device* cpu_device = dy::get_device_manager()->get_global_device("CPU");
    dy::Device* gpu_device = dy::get_device_manager()->get_global_device("GPU:0");
    #endif

    bool training_ = false;

    // print latent structures at test time
    bool print_ = false;
    std::shared_ptr<std::ostream> out_st_;

    explicit BaseNLI(
        ParameterCollection& params,
        unsigned vocab_size,
        unsigned embed_dim,
        unsigned hidden_dim,
        bool update_embed)
    : BaseEmbedBiLSTMModel(
        params,
        vocab_size,
        embed_dim,
        hidden_dim,
        update_embed,
        "nliclf")
    { }

    void
    set_print(
        const std::string& fn)
    {
        out_st_.reset(new std::ofstream(fn));
        print_ = true;
    }

    void
    print_param(
        Expression expr)
    {
        auto v = dy::as_vector(expr.value());
        for (auto && val : v)
            (*out_st_) << val << ' ';
    }

    virtual
    int
    n_correct(
        dy::ComputationGraph& cg,
        const NLIBatch& batch)
    {
        training_ = false;
        auto out_v = predict_batch(cg, batch);
        auto out_b = dy::concatenate_to_batch(out_v);

        #if USING_CUDA
        out_b = dy::to_device(out_b, cpu_device);
        #endif

        cg.incremental_forward(out_b);
        auto out = out_b.value();
        auto pred = dy::as_vector(dy::TensorTools::argmax(out));

        int n_correct = 0;
        for (int i = 0; i < batch.size(); ++i)
        {
            if (batch[i].target == pred[i])
                n_correct += 1;
        }
        return n_correct;
    }

    virtual
    Expression
    batch_loss(
        dy::ComputationGraph& cg,
        const NLIBatch& batch)
    {
        training_ = true;
        auto out = predict_batch(cg, batch);

        vector<Expression> losses;
        for(int i = 0; i < batch.size(); ++i)
        {
            auto loss = dy::pickneglogsoftmax(out[i], batch[i].target);
            losses.push_back(loss);
        }

        return dy::sum(losses);
    }

    virtual
    vector<Expression>
    predict_batch(
        ComputationGraph& cg,
        const NLIBatch& batch)
    = 0;


    void set_temperature(float t) { test_temperature_ = t; }
    protected:
    float test_temperature_ = 1;
};


// shared training code
float
validate(
    std::unique_ptr<BaseNLI>& clf,
    const vector<NLIBatch>& data)
{
    int n_correct = 0;
    int n_total = 0;
    for (auto && valid_batch : data)
    {
        dy::ComputationGraph cg;
        n_correct += clf->n_correct(cg, valid_batch);
        n_total += valid_batch.size();
    }

    return float(n_correct) / n_total;
}


void
test(
    std::unique_ptr<BaseNLI>& clf,
    const BaseArgs& args,
    const std::string& valid_fn,
    const std::string& test_fn)
{
    clf->load(args.saved_model);

    std::ostringstream valid_print_fn(args.save_prefix);
    valid_print_fn << "PRED_valid_"
                   << args.get_filename()
                   << ".txt";

    clf->set_print(valid_print_fn.str());
    auto valid_data = read_batches<NLIPair>(valid_fn, args.batch_size);
    float acc = validate(clf, valid_data);
    std::cout << "Validation accuracy: " << acc << std::endl;

    std::ostringstream test_print_fn(args.save_prefix);
    test_print_fn << "PRED_test_"
                  << args.get_filename()
                  << ".txt";

    clf->set_print(test_print_fn.str());
    auto test_data = read_batches<NLIPair>(test_fn, args.batch_size);
    acc = validate(clf, test_data);
    std::cout << "Test accuracy: " << acc << std::endl;
}


void
test_temp(
    std::unique_ptr<BaseNLI>& clf,
    const BaseArgs& args,
    const std::string& valid_fn,
    const std::string& test_fn)
{
    clf->load(args.saved_model);

    float best_acc = 0;
    float best_temp = 0;
    auto valid_data = read_batches<NLIPair>(valid_fn, args.batch_size);
    for (int t = -8; t <= 8; ++t)
    {
        float temperature = std::pow(2.0f, t);
        clf->set_temperature(temperature);
        float acc = validate(clf, valid_data);
        cout << "Temperature " << temperature << ":\t valid " << acc << endl;
        if (acc >= best_acc)
        {
            best_acc = acc;
            best_temp = temperature;
        }
    }

    auto test_data = read_batches<NLIPair>(test_fn, args.batch_size);
    clf->set_temperature(best_temp);
    float acc = validate(clf, test_data);
    std::cout << "Test accuracy: " << acc << std::endl;
}


void
train(
    std::unique_ptr<BaseNLI>& clf,
    const BaseArgs& args,
    const std::string& train_fn,
    const std::string& valid_fn)
{
    auto train_data = read_batches<NLIPair>(train_fn, args.batch_size);
    auto valid_data = read_batches<NLIPair>(valid_fn, args.batch_size);
    auto n_batches = train_data.size();

    unsigned n_train_sents = 0;
    for (auto&& batch : train_data)
        n_train_sents += batch.size();

    std::cout << "Training on " << n_train_sents << " sentences." << std::endl;

    // make an identity permutation vector of pointers into the batches
    vector<vector<NLIBatch>::iterator> train_iter(n_batches);
    std::iota(train_iter.begin(), train_iter.end(), train_data.begin());

    dy::SimpleSGDTrainer trainer(clf->p, args.lr);
    float best_valid_acc = 0;
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

        std::cout << "Completed epoch " << it
                  << " training loss "
                  << total_loss / n_train_sents
                  << " valid accuracy " << valid_acc << std::endl;

        if ((valid_acc + 0.0001) > best_valid_acc)
        {
            patience = 0;
            best_valid_acc = valid_acc;

            std::ostringstream fn;
            fn << args.save_prefix
               << "nli_" << args.get_filename()
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
