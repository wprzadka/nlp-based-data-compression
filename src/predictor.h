//
// Created by viking on 27.06.22.
//

#ifndef ANS_PREDICTOR_H
#define ANS_PREDICTOR_H

#include <torch/script.h>
#include "vocabulary_filter.h"


class Predictor {

    torch::jit::script::Module model;
    const VocabularyFilter& filter;
    bool use_filter;

public:
    Predictor(const std::string& path, const VocabularyFilter& filter, bool use_filter = false);

    torch::Tensor forward(const torch::Tensor& tokens);
    inline torch::Tensor operator()(const torch::Tensor& tokens) {return forward(tokens);}
    static int argmax(const torch::Tensor& probs);
};


#endif //ANS_PREDICTOR_H
