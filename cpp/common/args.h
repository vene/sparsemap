#pragma once
#include <iostream>
#include "treernn.h"


struct BaseArgs
{
    unsigned max_iter = 20;
    unsigned batch_size = 16;
    unsigned hidden_dim = 300;
    unsigned patience = 5;
    float lr = 1;
    float decay = 0.9;

    std::string saved_model;
    std::string save_prefix = "./";
    bool test = false;
    bool override_dy = true;

    virtual std::string get_filename() const = 0;

    virtual void parse(int argc, char** argv)
    {
        int i = 1;
        while (i < argc)
        {
            std::string arg = argv[i];
            if (arg == "--test")
            {
                test = true;
                i += 1;
            }
            else if (arg == "--no-override-dy")
            {
                override_dy = false;
                i += 1;
            }
            else if (arg == "--lr")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> lr;
                i += 2;
            }
            else if (arg == "--decay")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> decay;
                i += 2;
            }
            else if (arg == "--patience")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> patience;
                i += 2;
            }
            else if (arg == "--save-prefix")
            {
                assert(i + 1 < argc);
                save_prefix = argv[i + 1];
                i += 2;
            }
            else if (arg == "--max-iter")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> max_iter;
                i += 2;
            }
            else if (arg == "--saved-model")
            {
                assert(i + 1 < argc);
                saved_model = argv[i + 1];
                i += 2;
            }
            else if (arg == "--hidden-dim")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> hidden_dim;
                i += 2;
            }
            else if (arg == "--batch-size")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> batch_size;
                i += 2;
            }
            else
            {
                i += 1;
            }
        }
    }

    virtual std::ostream& print(std::ostream& o)
    const
    {
        o <<  (test ? "Test" : "Train") << " mode. "
          << "Arguments:\n"
          << " Save prefix: " << save_prefix     << '\n'
          << "   Max. iter: " << max_iter        << '\n'
          << "  Batch size: " << batch_size      << '\n'
          << "  Hidden dim: " << hidden_dim      << '\n'
          << "          LR: " << lr              << '\n'
          << "       Decay: " << decay           << '\n'
          << "  Model file: " << saved_model     << std::endl;

        return o;
    }

};


struct RevdictArgs : BaseArgs
{
    bool latent = false;
    size_t max_decode_iter = 10;
    bool print_trees = false;
    std::string out_trees_fn;
    std::string strategy_str = "flat";
    std::string scorer_str = "mlp";

    ChildSumTreeLSTMBuilder::Strategy get_strategy() const
    {
        if (strategy_str == "flat")
            return ChildSumTreeLSTMBuilder::Strategy::FLAT;
        else if (strategy_str == "none")
            return ChildSumTreeLSTMBuilder::Strategy::NONE;
        else if (strategy_str == "ltr")
            return ChildSumTreeLSTMBuilder::Strategy::LTR;
        else if (strategy_str == "rtl")
            return ChildSumTreeLSTMBuilder::Strategy::RTL;
        else if (strategy_str == "custom")
            return ChildSumTreeLSTMBuilder::Strategy::CUSTOM;
        else if (strategy_str == "latent")
            return ChildSumTreeLSTMBuilder::Strategy::CUSTOM;
        else
        {
            std::cerr << "invalid strategy argument" << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    virtual void parse(int argc, char** argv)
    {
        BaseArgs::parse(argc, argv);
        int i = 1;
        while (i < argc)
        {
            std::string arg = argv[i];
            if (arg == "--strategy")
            {
                assert(i + 1 < argc);
                strategy_str = argv[i + 1];
                if (strategy_str == "latent")
                    latent = true;
                i += 2;
            }
            else if (arg == "--arc-scorer")
            {
                assert(i + 1 < argc);
                scorer_str = argv[i + 1];
                i += 2;
            }
            else if (arg == "--max-decode-iter")
            {
                assert(i + 1 < argc);
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> max_decode_iter;
                i += 2;
            }
            else if (arg == "--print-trees")
            {
                assert(i + 1 < argc);
                print_trees = true;
                std::string val = argv[i + 1];
                std::istringstream vals(val);
                vals >> out_trees_fn;
                i += 2;
            }
            else
            {
                i += 1;
            }
        }
    }

    virtual std::ostream& print(std::ostream& o)
    const
    override
    {
        BaseArgs::print(o);
        o << " Decode iter: " << max_decode_iter << '\n'
          << "    Strategy: " << strategy_str    << '\n'
          << "  Arc scorer: " << scorer_str      << std::endl;
        if (print_trees)
            o << "Print trees:" << out_trees_fn << std::endl;
        return o;
    }

    virtual std::string get_filename()
    const
    override
    {
        std::ostringstream fn;
        fn << "_strat_" << strategy_str
           << "_lr_" << lr << "_decay_" << decay;
        if (latent)
        {
            fn << "_scorer_" << scorer_str
               << "_" << max_decode_iter;
        }
        return fn.str();
    }
};

struct SentClfArgs : RevdictArgs
{
    std::string dataset;
    bool update_embed = false;

    bool override_train_fn = false;
    bool override_valid_fn = false;
    bool override_test_fn = false;
    std::string train_fn;
    std::string valid_fn;
    std::string test_fn;

    virtual void parse(int argc, char** argv)
    override
    {
        RevdictArgs::parse(argc, argv);
        int i = 1;
        while (i < argc)
        {
            std::string arg = argv[i];
            if (arg == "--dataset")
            {
                assert(i + 1 < argc);
                dataset = argv[i + 1];
                i += 2;
            }
            if (arg == "--train-fn")
            {
                assert(i + 1 < argc);
                override_train_fn = true;
                train_fn = argv[i + 1];
                i += 2;
            }
            if (arg == "--valid-fn")
            {
                assert(i + 1 < argc);
                override_valid_fn = true;
                valid_fn = argv[i + 1];
                i += 2;
            }
            if (arg == "--test-fn")
            {
                assert(i + 1 < argc);
                override_test_fn = true;
                test_fn = argv[i + 1];
                i += 2;
            }
            else if (arg == "--update-embed")
            {
                update_embed = true;
                i += 1;
            }
            else
            {
                i += 1;
            }
        }
    }

    virtual std::ostream& print(std::ostream& o) const
    override
    {
        RevdictArgs::print(o);
        o << "     Dataset: " << dataset << '\n';
        if (override_train_fn)
            o << "Override train: " << train_fn << '\n';
        if (override_valid_fn)
            o << "Override valid: " << valid_fn << '\n';
        if (override_test_fn)
            o << " Override test: " << test_fn << '\n';
        o << "Update embed: " << update_embed << '\n';
        return o;
    }

    virtual std::string get_filename()
    const
    override
    {
        std::ostringstream fn;
        fn << dataset
           << "_lr_" << lr << "_decay_" << decay
           << (update_embed ? "_elearn" : "_efixed")
           << "_strat_" << strategy_str;
        if (latent)
        {
            fn << "_scorer_" << scorer_str
               << "_" << max_decode_iter;
        }
        return fn.str();
    }
};


struct NLIClfArgs : public SentClfArgs
{
};


std::ostream& operator << (std::ostream &o, const BaseArgs &args)
{
    return args.print(o);
}

