/* Child-Sum TreeLSTM in dynet
 * Reference: https://www.aclweb.org/anthology/P15-1150
 * Author: vlad@vene.ro
 * License: Apache 2.0
 */

#pragma once

#include <dynet/nodes.h>
#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/param-init.h>

namespace dy = dynet;

using dy::Parameter;
using dy::parameter;
using dy::Expression;
using dy::affine_transform;
using std::vector;


/**
 * \brief Convenience structure to collect all parameter matrices.
 */
struct LSTMParamExprs {
    Expression Wi, Ui, bi;
    Expression Wf, Uf, bf;
    Expression Wo, Uo, bo;
    Expression Wu, Uu, bu;

    LSTMParamExprs() = default;

    LSTMParamExprs(dy::ComputationGraph& cg,
                   std::vector<Parameter> vec_params)
    {
        Wi = parameter(cg, vec_params[0]);
        Ui = parameter(cg, vec_params[1]);
        bi = parameter(cg, vec_params[2]);

        Wf = parameter(cg, vec_params[3]);
        Uf = parameter(cg, vec_params[4]);
        bf = parameter(cg, vec_params[5]);

        Wo = parameter(cg, vec_params[6]);
        Uo = parameter(cg, vec_params[7]);
        bo = parameter(cg, vec_params[8]);

        Wu = parameter(cg, vec_params[9]);
        Uu = parameter(cg, vec_params[10]);
        bu = parameter(cg, vec_params[11]);
    }
};


struct ChildSumTreeLSTMBuilder
{

    enum class Strategy {
        CUSTOM, ///< Use the tree passed by the user.
        FLAT,   ///< Use a flat tree: all words attached to root.
        LTR,    ///< Use left-to-right tree: each word attached to previous.
        RTL,    ///< Use right-to-left tree: each word attached to next.
        NONE    ///< No tree: just average vectors.
    };

    Strategy strategy_;
    unsigned input_dim_;
    unsigned hidden_dim_;

    dy::ParameterCollection p;
    vector<dy::Parameter> vec_p;
    LSTMParamExprs expr_p;

    Expression zero_;
    vector<Expression> input_;
    vector<int> parents_;
    vector<vector<int> > children_;

    explicit ChildSumTreeLSTMBuilder(
            dy::ParameterCollection& params,
            unsigned input_dim,
            unsigned hidden_dim,
            Strategy strategy = Strategy::CUSTOM)
        : strategy_(strategy)
        , input_dim_(input_dim)
        , hidden_dim_(hidden_dim)
    {
        p = params.add_subcollection("child-sum-tree-lstm-builder");

        unsigned hid = hidden_dim_;
        unsigned inp = input_dim_;

        // recommended init forget bias to 1
        auto bfinit = dy::ParameterInitConst(1.0f);

        vec_p = {
            // input gate
            p.add_parameters({hid, inp}, 0, "Wi"),
            p.add_parameters({hid, hid}, 0, "Ui"),
            p.add_parameters({hid},      0, "bi"),

            // forget gate
            p.add_parameters({hid, inp}, 0, "Wf"),
            p.add_parameters({hid, hid}, 0, "Uf"),
            p.add_parameters({hid}, bfinit, "bf"),

            // output gate
            p.add_parameters({hid, inp}, 0, "Wo"),
            p.add_parameters({hid, hid}, 0, "Uo"),
            p.add_parameters({hid},      0, "bo"),

            // candidate transform
            p.add_parameters({hid, inp}, 0, "Wu"),
            p.add_parameters({hid, hid}, 0, "Uu"),
            p.add_parameters({hid},      0, "bu")
        };
    }

    /**
     * \brief Initialize builder to use a new computation graph.
     * \param cg The current computation graph.
     */
    void new_graph(dy::ComputationGraph& cg)
    {
        expr_p = LSTMParamExprs(cg, vec_p);
        zero_ = dy::zeros(cg, {input_dim_});
    }

    /**
     * \brief Initialize builder to process a new sentence.
     * \param input Encoded vectors, one for each word in sentence.
     */
    void new_sequence(const vector<Expression>& input)
    {
        input_ = input;
    }

    /**
     * \brief Transform the current sentence.
     * \return Single vector summarizing the sentence.
     */
    Expression transform()
    {
        switch (strategy_) {
            case Strategy::RTL:
                return transform_rtl();
                break;
            case Strategy::LTR:
                return transform_ltr();
                break;
            case Strategy::FLAT:
                return transform_flat();
                break;
            case Strategy::CUSTOM:
                throw "requires parents";
            case Strategy::NONE:
                return dy::average(input_);
            default:
                throw "not implemented";
            }
    }

    /**
     * \brief Transform the current sentence.
     * \param parents Integer pointing to the parent of each word; root at 0.
     * \return Single vector summarizing the sentence.
     */
    Expression transform(const vector<int> parents)
    {
        if (strategy_ != Strategy::CUSTOM)
            throw "parents passed but not used";

        return transform_custom(parents);
    }

    Expression transform_rtl()
    {
        vector<int> parents(input_.size() + 1);

        // example: [-1, 0, 1, 2, 3]
        parents[0] = -1;
        for (int i = 0; i < input_.size(); ++i)
            parents[i + 1] = i;

        return transform_custom(parents);
    }

    Expression transform_ltr()
    {
        vector<int> parents(input_.size() + 1);

        // example: [-1, 2, 3, 4, 0]
        parents[0] = -1;
        parents[input_.size()] = 0;
        for(int i = 1; i < input_.size(); ++i)
            parents[i] = i + 1;
        return transform_custom(parents);
    }

    Expression transform_flat()
    {
        // example: [-1, 0, 0, 0, 0]
        vector<int> parents(input_.size() + 1, 0);
        parents[0] = -1;
        return transform_custom(parents);
    }

    Expression transform_custom(vector<int> parents)
    {
        parents_ = parents;

        /*
        std::cout << "encoding using parent vector ";
        for (auto& i : parents_) std::cout << i << " ";
        std::cout << std::endl;
        */

        // make children vector

        children_ = vector<vector<int> >(parents_.size());
        for (int i = 1; i < parents_.size(); ++i)
            children_[parents_[i]].push_back(i);

        /*
        for (auto& c : children_) {
            std::cout << "[";
            for (auto& i : c)
                std::cout << i << " ";
            std::cout << "]" << std::endl;
        }
        std::cout << std::endl;
        */

        Expression h, c;
        std::tie(h, c) = encode(0);
        return h;
    }

    std::tuple<Expression, Expression> encode(int ix) {

        Expression x;
        if (ix > 0)
            x = input_[ix - 1];
        else
            x = zero_;

        Expression i, o, u, c, h;
        if (children_[ix].size() == 0) {

            i = affine_transform({expr_p.bi, expr_p.Wi, x});
            o = affine_transform({expr_p.bo, expr_p.Wo, x});
            u = affine_transform({expr_p.bu, expr_p.Wu, x});

            i = dy::logistic(i);
            o = dy::logistic(o);
            u = dy::tanh(u);

            c = dy::cmult(i, u);

        } else {

            vector<Expression> H, F;
            Expression h_k, c_k;
            Expression f;

            for (auto& k : children_[ix])
            {
                std::tie(h_k, c_k) = encode(k);
                H.push_back(h_k);

                f = affine_transform({expr_p.bf, expr_p.Wf, x, expr_p.Uf, h_k});
                f = dy::logistic(f);
                F.push_back(dy::cmult(c_k, f));
            }

            Expression h_bar = dy::sum(H);
            i = affine_transform({expr_p.bi, expr_p.Wi, x, expr_p.Ui, h_bar});
            o = affine_transform({expr_p.bo, expr_p.Wo, x, expr_p.Uo, h_bar});
            u = affine_transform({expr_p.bu, expr_p.Wu, x, expr_p.Uu, h_bar});

            i = dy::logistic(i);
            o = dy::logistic(o);
            u = dy::tanh(u);

            c = dy::cmult(i, u) + dy::sum(F);
        }

        h = dy::cmult(o, dy::tanh(c));

        return std::tie(h, c);
    }
};
