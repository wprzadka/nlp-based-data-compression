//
// Created by viking on 18.09.22.
//

#ifndef NLP_BASED_COMPRESSION_VOCABULARY_FILTER_H
#define NLP_BASED_COMPRESSION_VOCABULARY_FILTER_H

#include <torch/script.h>
#include "tokenizer.h"


class VocabularyFilter {

    Tokenizer tokenizer;
    torch::Tensor filter;

    const uint8_t VOCAB_INDEX_BYTES = 2;

public:
    explicit VocabularyFilter(const Tokenizer& tokenizer);

    torch::Tensor forward(const torch::Tensor& input) const;
    inline torch::Tensor operator()(const torch::Tensor& input) const { return forward(input); }

    void createFilter(const torch::Tensor& tokens);
    void saveFilter(std::ofstream &file);
    void readFilter(std::ifstream &file);
};


#endif //NLP_BASED_COMPRESSION_VOCABULARY_FILTER_H
