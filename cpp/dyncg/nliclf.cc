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

#include "nliclf.h"

namespace dy = dynet;

using std::vector;
using std::cout;
using std::endl;


int main(int argc, char** argv)
{
    auto dyparams = dy::extract_dynet_params(argc, argv);
    NLIClfArgs args;
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
    vocab_fn << "data/nli/" << args.dataset << ".vocab";
    class_fn << "data/nli/" << args.dataset << ".classes";
    train_fn << "data/nli/" << args.dataset << ".train.txt";
    // train_fn << "data/nli/" << args.dataset << ".train-tiny.txt";
    valid_fn << "data/nli/" << args.dataset << ".valid.txt";
    // valid_fn << "data/nli/" << args.dataset << ".train-tiny.txt";
    test_fn  << "data/nli/" << args.dataset << ".test.txt";
    // test_fn  << "data/nli/" << args.dataset << ".train-tiny.txt";
    embed_fn << "data/nli/" << args.dataset << ".embed";

    unsigned vocab_size = line_count(vocab_fn.str());
    cout << "vocabulary size: " << vocab_size << endl;

    size_t n_classes = 3;

    dy::ParameterCollection params;

    std::unique_ptr<BaseNLI> clf = nullptr;

    if (args.latent)
        clf.reset(new LatentDepNLI(params,
                                   vocab_size,
                                   EMBED_DIM,
                                   args.hidden_dim,
                                   n_classes,
                                   args.update_embed,
                                   args.max_decode_iter,
                                   args.scorer_str));
    else
        clf.reset(new FixedDepNLI(params,
                                  vocab_size,
                                  EMBED_DIM,
                                  args.hidden_dim,
                                  n_classes,
                                  args.get_strategy(),
                                  args.update_embed));
    clf->load_embeddings(embed_fn.str());

    if (args.test)
        if (args.latent)
            test_temp(clf, args, valid_fn.str(), test_fn.str());
        else
            test(clf, args, valid_fn.str(), test_fn.str());
    else
        train(clf, args, train_fn.str(), valid_fn.str());

    return 0;
}
