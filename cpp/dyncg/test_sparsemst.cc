#include <dynet/dynet.h>
#include <dynet/expr.h>
#include <dynet/grad-check.h>

#include <iostream>

#include "sparsemst.h"

namespace dy = dynet;


int main(int argc, char** argv)
{
    dy::initialize(argc, argv);

    const unsigned DIM = 3;

    dy::ParameterCollection m;
    auto x_par = m.add_parameters({1 + DIM, DIM}, 0, "x");

    for(int k = 0; k < 64; ++k)
    {
        // std::cout << "wrt dimension " << k << std::endl;

        ComputationGraph cg;
        Expression x = dy::parameter(cg, x_par);
        Expression y = sparse_mst_full(x, 1 + DIM, 200);
        Expression yk = dy::pick(y, k);
        cg.forward(yk);
        cg.backward(yk);
        // std::cout << x.gradient() << std::endl << std::endl;
        check_grad(m, yk, 1);
    }

    std::cout << "Done with gradient checks." << std::endl;


    {
        ComputationGraph cg;
        Expression x = dy::parameter(cg, x_par);
        Expression y = sparse_mst_full(x, 1 + DIM);
        std::cout << y.value() << std::endl;

        cg.backward(dy::pick(y, 3));
        std::cout << x.gradient() << std::endl << std::endl;
    }

    std::cout << std::endl << "===" << std::endl;

    {
        ComputationGraph cg;
        Expression x = dy::parameter(cg, x_par);
        Expression y = sparse_mst(x, 1 + DIM);
        std::cout << y.value() << std::endl;
        cg.backward(dy::pick(y, 2));
        std::cout << x.gradient() << std::endl << std::endl;

        auto sparse_mst_node = static_cast<SparseMST*>(cg.nodes[y.i]);
        int n_active = sparse_mst_node->get_n_active();
        std::cout << n_active << std::endl;
        for (int k = 0; k < n_active; ++k) {
            for(auto&& hd : sparse_mst_node->get_config(k))
                std::cout << hd << " ";
            std::cout << std::endl;
        }
    }
}

