#ifndef FACTOR_MATCHING_SPARSE
#define FACTOR_MATCHING_SPARSE

#include <limits>
#include <algorithm>
#include <iterator>

#include <ad3/GenericFactor.h>

#include "lap/lapjv.h"

using AD3::GenericFactor;
using AD3::Configuration;
using std::vector;


namespace sparsemap {

    double inf = std::numeric_limits<double>::infinity();


    class FactorMatchingSparse : public GenericFactor {

        protected:
        vector<int>* cfg_cast(Configuration cfg) {
            return static_cast<vector<int> *>(cfg);
        }

        public:
        FactorMatchingSparse() = default;
        //FactorMatchingSparse() {}
        virtual ~FactorMatchingSparse() { ClearActiveSet(); }

        void
        Evaluate(
            const vector<double> &variable_log_potentials,
            const vector<double> &additional_log_potentials,
            const Configuration configuration,
            double *value)
        {
            const vector<int>* assigned = cfg_cast(configuration);
            int_t j;
            *value = 0;
            for (uint_t i = 0; i < rows_; ++i)
            {
                j = (*assigned)[i];

                if (j < 0)
                    continue; // not assigned

                bool found = false;
                for (uint_t jj = indptr_[i]; jj < indptr_[i + 1]; ++jj)
                {
                    if (indices_[jj] == j)
                    {
                        *value += variable_log_potentials[jj];
                        found = true;
                        break;
                    }
                }

                if (!found)
                    *value = -inf;
            }
        }

        void
        Maximize(
            const vector<double> &variable_log_potentials,
            const vector<double> &additional_log_potentials,
            Configuration &configuration,
            double *value)
        {
            uint_t nnz = variable_log_potentials.size();
            vector<int_t> x, y;

            vector<cost_t> cost;

            cost.reserve(nnz);
            for (uint_t j = 0; j < nnz; ++j)
                cost[j] = -variable_log_potentials[j];

            /* below is wip attempt at padding for non-square
                vector<uint_t> indices, indptr;
                uint_t rows, cols;
                rows = cols_ + rows_;
                cols = rows;

                //uint_t n_append = rows_ - cols_;
                // append diag with max value
                auto min_elem = std::min_element(std::begin(variable_log_potentials),
                                                 std::end(variable_log_potentials));
                double pad = -(*min_elem) + 99;
                for (int i = 0; i < indptr_.size(); ++i)
                {
                    indptr.push_back(indptr_[i] + i);
                }

                for (int i = 0; i < rows_; ++i)
                {
                    for (int jj = indptr_[i]; jj < indptr_[i + 1]; ++jj)
                    {
                        indices.push_back(indices_[jj]);
                        cost.push_back(-variable_log_potentials[jj]);
                    }
                    indices.push_back(cols_ + i);
                    cost.push_back(pad);
                }

                for (int i = rows_; i <= rows_ + cols_; ++i)
                    indptr.push_back(indptr[i] + 2);
                for (int i = rows_; i < rows_ + cols_; ++i)
                {
                    indices.push_back(i - rows_);
                    indices.push_back(i);
                    cost.push_back(pad);
                    cost.push_back(0);
                }


            for (int jj = 0; jj < indptr.size(); ++jj)
                std::cout << indptr[jj] << " ";
            std::cout << std::endl;
            for (int jj = 0; jj < indices.size(); ++jj)
                std::cout << indices[jj] << " ";
            std::cout << std::endl;
            for (int jj = 0; jj < cost.size(); ++jj)
                std::cout << cost[jj] << " ";
            std::cout << std::endl;

            */
            x.assign(rows_, 0);
            y.assign(cols_, 0);

            lapmod_internal(rows_, cost.data(), indptr_.data(), indices_.data(),
                            x.data(), y.data(), FP_DYNAMIC);

            /*
            std::cout << "x and y" << std::endl;
            for (uint_t i = 0; i < rows_; ++i)
                std::cout << x[i] << " ";
            std::cout << std::endl;
            for (uint_t i = 0; i < cols_; ++i)
                std::cout << y[i] << " ";
            std::cout << std::endl;
            */


            vector<int> *cfg_vec = cfg_cast(configuration);
            for (uint_t i = 0; i < rows_; ++i) {
                cfg_vec->push_back(x[i] < cols_ ? x[i] : -1);
            }

            /*
            std::cout << "Maximized: ";
            for (uint_t i = 0; i < rows_; ++i) {
                std::cout << (*cfg_vec)[i] << " ";
            }
            std::cout << std::endl;
            */

            Evaluate(variable_log_potentials,
                     additional_log_potentials,
                     configuration,
                     value);
        }

        void
        UpdateMarginalsFromConfiguration(
            const Configuration &configuration,
            double weight,
            vector<double> *variable_posteriors,
            vector<double> *additional_posteriors)
        {
            const vector<int>* assigned = cfg_cast(configuration);
            int_t j;
            for (uint_t i = 0; i < rows_; ++i) {
                j = (*assigned)[i];

                if (j < 0)
                    continue; // not assigned

                for (uint_t jj = indptr_[i]; jj < indptr_[i + 1]; ++jj)
                {
                    if (indices_[jj] == j)
                    {
                        (*variable_posteriors)[jj] += weight;
                        break;
                    }
                }
            }
        }

        int
        CountCommonValues(
            const Configuration &configuration1,
            const Configuration &configuration2)
        {
            const vector<int>* assigned1 = cfg_cast(configuration1);
            const vector<int>* assigned2 = cfg_cast(configuration2);

            int common = 0;
            int j1, j2;
            for (uint_t i = 0; i < rows_; ++i) {
                j1 = (*assigned1)[i];
                j2 = (*assigned2)[i];
                if (j1 == j2 && j1 >= 0)
                    common += 1;
            }

            return common;
        }

        bool
        SameConfiguration(
            const Configuration &configuration1,
            const Configuration &configuration2)
        {
            const vector<int>* assigned1 = cfg_cast(configuration1);
            const vector<int>* assigned2 = cfg_cast(configuration2);

            for (uint_t i = 0; i < rows_; ++i)
                if (! ((*assigned1)[i] == (*assigned2)[i]))
                    return false;
            return true;
        }

        void
        DeleteConfiguration(
            Configuration configuration)
        {
            vector<int>* assigned = cfg_cast(configuration);
            delete assigned;
        }

        Configuration
        CreateConfiguration()
        {
            vector<int>* config = new vector<int>;
            return static_cast<Configuration>(config);
        }

        void
        Initialize(
            uint_t rows,
            uint_t cols,
            vector<uint_t> &indptr,
            vector<uint_t> &indices)
        {
            rows_ = rows;
            cols_ = cols;
            indptr_ = indptr;
            indices_ = indices;
        }

        private:
        uint_t rows_, cols_;
        vector<uint_t> indptr_;
        vector<uint_t> indices_;

    };

    void test_factor()
    {
        uint_t n = 4;
        vector<uint_t> indptr = {0, 4, 6, 9, 13};
        vector<uint_t> indices = {0, 1, 2, 3, 0, 2, 1, 2, 3, 0, 1, 2, 3};
        vector<double> data = {-1.76, -0.40, -0.98, -2.24, -1.87, -0.95,
                               -0.41, -0.14, -1.45, -0.76, -0.12, -0.44, -0.33};

        GenericFactor* f = new FactorMatchingSparse;
        static_cast<FactorMatchingSparse* >(f)->Initialize(n, n, indptr, indices);
        vector<double> u, v, z;
        double val;
        f->SolveMAP(data, z, &u, &v, &val);
        std::cout << val << std::endl;
        for (auto && j : u) std::cout << j << " ";
        std::cout << std::endl;
    }
} // namespace sparsemap

#endif
