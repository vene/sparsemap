# pragma once

namespace dy = dynet;

struct EdgeScorer
{
    virtual void new_graph(dy::ComputationGraph& cg) = 0;
    virtual dy::Expression make_potentials(std::vector<dy::Expression> enc) = 0;
};


struct BilinearScorer : public EdgeScorer
{
    dy::Parameter p_U_hed, p_b_hed;
    dy::Parameter p_U_mod, p_b_mod;
    dy::Parameter p_root;

    // re-instantiated for each CG
    dy::Expression U_hed, b_hed;
    dy::Expression U_mod, b_mod;
    dy::Expression root;

    explicit BilinearScorer(dy::ParameterCollection params,
                            unsigned hidden_dim,
                            unsigned input_dim)
    {
        auto p = params.add_subcollection("bilinear-scorer");
        p_U_hed = p.add_parameters({hidden_dim, input_dim});
        p_U_mod = p.add_parameters({hidden_dim, input_dim});
        p_b_hed = p.add_parameters({hidden_dim});
        p_b_mod = p.add_parameters({hidden_dim});
        p_root = p.add_parameters({hidden_dim});
    }

    virtual void new_graph(dy::ComputationGraph& cg)
    override
    {
        U_hed = dy::parameter(cg, p_U_hed);
        U_mod = dy::parameter(cg, p_U_mod);
        b_hed = dy::parameter(cg, p_b_hed);
        b_mod = dy::parameter(cg, p_b_mod);
        root = dy::parameter(cg, p_root);
    }

    virtual dy::Expression make_potentials(std::vector<dy::Expression> enc)
    override
    {
        auto T = dy::concatenate_cols(enc);
        auto T_hed = dy::affine_transform({b_hed, U_hed, T});
        auto T_mod = dy::affine_transform({b_mod, U_mod, T});
        T_hed = dy::concatenate_cols({root, T_hed});
        auto S = dy::transpose(T_hed) * T_mod;
        return S;
    }
};


struct MLPScorer : public EdgeScorer
{
    dy::Parameter p_U_hed;
    dy::Parameter p_U_mod;
    dy::Parameter p_v;
    dy::Parameter p_b;
    dy::Parameter p_root;

    dy::Expression U_hed;
    dy::Expression U_mod;
    dy::Expression v;
    dy::Expression b;
    dy::Expression root;

    explicit MLPScorer(dy::ParameterCollection params,
                       unsigned hidden_dim,
                       unsigned input_dim)
    {
        auto p = params.add_subcollection("mlp-scorer");
        p_U_hed = p.add_parameters({hidden_dim, input_dim});
        p_U_mod = p.add_parameters({hidden_dim, input_dim});
        p_b = p.add_parameters({hidden_dim});
        p_v = p.add_parameters({1, hidden_dim});
        p_root = p.add_parameters({hidden_dim});
    }

    virtual void new_graph(dy::ComputationGraph& cg)
    override
    {
        U_hed = dy::parameter(cg, p_U_hed);
        U_mod = dy::parameter(cg, p_U_mod);
        b = dy::parameter(cg, p_b);
        v = dy::parameter(cg, p_v);
        root = dy::parameter(cg, p_root);
    }

    virtual dy::Expression make_potentials(std::vector<dy::Expression> enc)
    override
    {
        size_t sz = enc.size();
        std::vector<dy::Expression> T_hed, T_mod;

        T_hed.push_back(root);
        for (auto && w : enc)
        {
            T_hed.push_back(U_hed * w);
            T_mod.push_back(U_mod * w);
        }

        auto TH = dy::concatenate_cols(T_hed);
        TH = TH + b;
        std::vector<dy::Expression> columns;
        for (size_t m = 0; m < sz; ++m)
            columns.push_back(v * dy::tanh(TH + T_mod[m]));

        // alternative implementation:
        // for m: for h: s_flat.push_back(...)
        // S = s.reshape(...)

        auto S = dy::transpose(dy::concatenate(columns, 0));
        return S;
    }
};
