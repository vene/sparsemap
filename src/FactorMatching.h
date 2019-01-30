#ifndef FACTOR_MATCHING
#define FACTOR_MATCHING

#include <limits>
#include <algorithm>
#include <iterator>

#include <ad3/GenericFactor.h>

#include "lap/lapjv.h"

using AD3::GenericFactor;
using AD3::Configuration;
using std::vector;


namespace sparsemap {

    class FactorMatching : public GenericFactor {

        protected:

        int ix(int i, int j) { return cols_ * i + j; }

        vector<int>* cfg_cast(Configuration cfg) {
            return static_cast<vector<int> *>(cfg);
        }

        public:
        FactorMatching () {}
        virtual ~FactorMatching() { ClearActiveSet(); }

        void Evaluate(const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials,
                      const Configuration configuration,
                      double *value) {

            const vector<int>* assigned = cfg_cast(configuration);
            int j;
            *value = 0;
            for (int i = 0; i < rows_; ++i) {
                j = (*assigned)[i];
                if (j >= 0) // -1 denotes not assigned if n > m
                    *value += variable_log_potentials[ix(i, j)];
            }
        }

        void Maximize(const vector<double> &variable_log_potentials,
                      const vector<double> &additional_log_potentials,
                      Configuration &configuration,
                      double *value) {

            int n = rows_ > cols_ ? rows_ : cols_;
            vector<vector<double> > byrow;
            vector<double*> cost_ptr;
            byrow.resize(n);

            /* if needed, will pad up to square matrix with highest cost */
            double pad = 0;
            if (cols_ != rows_) {
                auto min_elem = std::min_element(std::begin(variable_log_potentials),
                                                 std::end(variable_log_potentials));
                pad = -(*min_elem) + 1;
            }

            for (int i = 0; i < rows_; ++i) {
                byrow[i].resize(n);
                for (int j = 0; j < cols_; ++j)
                    byrow[i][j] = -variable_log_potentials[ix(i, j)];

                /* fill remaining columns */
                if (cols_ < rows_) {
                    for (int j = cols_; j < rows_; ++j)
                        byrow[i][j] = pad;
                }

                cost_ptr.push_back(byrow[i].data());
            }

            /* fill remaining rows */
            if (rows_ < cols_)
                for (int i = rows_; i < cols_; ++i) {
                    byrow[i].assign(n, pad);
                    cost_ptr.push_back(byrow[i].data());
                }

            vector<int> x_c, y_c;
            x_c.reserve(n);
            y_c.reserve(n);

            lapjv_internal(n, cost_ptr.data(), x_c.data(), y_c.data());

            vector<int> *cfg_vec = cfg_cast(configuration);
            for (int i = 0; i < rows_; ++i) {
                cfg_vec->push_back(x_c[i] < cols_ ? x_c[i] : -1);
            }

            Evaluate(variable_log_potentials,
                     additional_log_potentials,
                     configuration,
                     value);
        }

        void UpdateMarginalsFromConfiguration(
                const Configuration &configuration,
                double weight,
                vector<double> *variable_posteriors,
                vector<double> *additional_posteriors) {

            const vector<int>* assigned = cfg_cast(configuration);
            int j;
            for (int i = 0; i < rows_; ++i) {
                j = (*assigned)[i];
                if (j >= 0) // -1 denotes not assigned if n > m
                    (*variable_posteriors)[ix(i, j)] += weight;
            }
        }

        int CountCommonValues(const Configuration &configuration1,
                              const Configuration &configuration2) {
            const vector<int>* assigned1 = cfg_cast(configuration1);
            const vector<int>* assigned2 = cfg_cast(configuration2);

            int common = 0;
            int j1, j2;
            for (int i = 0; i < rows_; ++i) {
                j1 = (*assigned1)[i];
                j2 = (*assigned2)[i];
                if (j1 == j2 && j1 >= 0)
                    common += 1;
            }

            return common;
        }

        bool SameConfiguration(const Configuration &configuration1,
                               const Configuration &configuration2) {
            const vector<int>* assigned1 = cfg_cast(configuration1);
            const vector<int>* assigned2 = cfg_cast(configuration2);

            for (int i = 0; i < rows_; ++i)
                if (! ((*assigned1)[i] == (*assigned2)[i]))
                    return false;
            return true;
        }

        void DeleteConfiguration(Configuration configuration) {
            vector<int>* assigned = cfg_cast(configuration);
            delete assigned;
        }

        Configuration CreateConfiguration() {
            vector<int>* config = new vector<int>;
            return static_cast<Configuration>(config);
        }

        void Initialize(int rows, int cols) {
            rows_ = rows;
            cols_ = cols;
        }

        private:
        int rows_, cols_;

    };
} // namespace sparsemap

#endif
