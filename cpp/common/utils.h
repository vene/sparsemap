/*
 * Generic utils
 * Author: vlad@vene.ro
 * License: Apache 2.0
 */

#pragma once

#include <Eigen/Eigen>
#include <fstream>
#include <iostream>

using std::vector;
using std::tuple;
using std::make_tuple;

void normalize_vector(vector<float>& v)
{
    Eigen::Map<Eigen::RowVectorXf> v_map(v.data(), v.size());
    v_map.normalize();
}


unsigned line_count(const std::string filename)
{
    std::ifstream in(filename);
    assert(in);
    std::string line;

    unsigned lines = 0;
    while (getline(in, line))
        ++lines;

    return lines;
}


/*
 * Data loaders
 */


/**
 * \brief Sentence represented by word indices and parse tree
 */
struct Sentence
{

    std::vector<unsigned> word_ixs;
    std::vector<int> heads;

    size_t size() const { return word_ixs.size(); }
};


/**
 * \brief Sentence with a label
 */
struct LabeledSentence
{
    Sentence sentence;
    unsigned target;

    size_t size() const { return sentence.size(); }
};


/**
 * \brief Reverse Dictionary definition
 */
struct Definition : public LabeledSentence { };


/**
 * \brief Pair of sentences (prem, hypo) with label
 */
struct NLIPair
{
    Sentence prem;
    Sentence hypo;
    unsigned target;

    size_t size()
    const
    { return prem.word_ixs.size() + hypo.word_ixs.size(); }
};


typedef vector<LabeledSentence> SentBatch;
typedef vector<Definition> DefBatch;
typedef vector<NLIPair> NLIBatch;


/**
 * \brief Read dataset from disk in batches
 * \param filename File to read from
 * \param batch_size Number of samples per batch
 * \return Batches (vector of vectors of data points)
 */
template<typename Datum>
vector<vector<Datum> >
read_batches(const std::string& filename, unsigned batch_size)
{
    vector<vector<Datum> > batches;

    std::ifstream in(filename);
    assert(in);

    std::string line;
    vector<Datum> curr_batch;

    while(in)
    {
        Datum s;
        in >> s;
        if (!in) break;

        if (curr_batch.size() == batch_size)
        {
            batches.push_back(curr_batch);
            curr_batch.clear();
        }
        curr_batch.push_back(s);
    }

    // leftover batch
    if (curr_batch.size() > 0)
        batches.push_back(curr_batch);

    // test
    unsigned total_samples = 0;
    unsigned total_words = 0;
    for (auto& batch : batches)
    {
        total_samples += batch.size();
        for (auto& s : batch)
            total_words += s.size();
    }
    std::cerr << batches.size() << " batches, "
              << total_samples << " samples, "
              << total_words << " words\n";

    return batches;
}


std::istream& operator>>(std::istream& in, Definition& data)
{
    unsigned tmp;

    std::string line;
    getline(in, line);

    std::istringstream sin(line);
    sin >> data.target;
    while (sin)
    {
        sin >> tmp;
        if (!sin) break;
        data.sentence.word_ixs.push_back(tmp);
    }

    return in;
}


std::istream& operator>>(std::istream& in, LabeledSentence& data)
{
    std::string target_buf, ixs_buf, heads_buf;
    std::getline(in, target_buf, '\t');
    std::getline(in, ixs_buf, '\t');
    std::getline(in, heads_buf);
    if (!in)  // failed
        return in;

    {
        std::stringstream target_ss(target_buf);
        target_ss >> data.target;
    }

    {
        std::stringstream ixs(ixs_buf);
        unsigned tmp;
        while(ixs >> tmp)
            data.sentence.word_ixs.push_back(tmp);
    }

    {
        std::stringstream heads(heads_buf);
        int tmp;
        while(heads >> tmp)
            data.sentence.heads.push_back(tmp);
    }

    return in;
}


std::istream& operator>>(std::istream& in, NLIPair& data)
{
    std::string target_buf;
    std::string prem_ixs_buf, prem_heads_buf;
    std::string hypo_ixs_buf, hypo_heads_buf;
    std::getline(in, target_buf, '\t');
    std::getline(in, prem_ixs_buf, '\t');
    std::getline(in, prem_heads_buf, '\t');
    std::getline(in, hypo_ixs_buf, '\t');
    std::getline(in, hypo_heads_buf, '\n');

    if (!in)  // failed
        return in;

    {
        std::stringstream target_ss(target_buf);
        target_ss >> data.target;
    }

    {
        std::stringstream ixs(prem_ixs_buf);
        unsigned tmp;
        while(ixs >> tmp)
            data.prem.word_ixs.push_back(tmp);
    }

    {
        std::stringstream heads(prem_heads_buf);
        int tmp;
        while(heads >> tmp)
            data.prem.heads.push_back(tmp);
    }

    {
        std::stringstream ixs(hypo_ixs_buf);
        unsigned tmp;
        while(ixs >> tmp)
            data.hypo.word_ixs.push_back(tmp);
    }

    {
        std::stringstream heads(hypo_heads_buf);
        int tmp;
        while(heads >> tmp)
            data.hypo.heads.push_back(tmp);
    }

    return in;
}
