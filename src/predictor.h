//
// Created by viking on 27.06.22.
//

#ifndef ANS_PREDICTOR_H
#define ANS_PREDICTOR_H

#include <torch/script.h>


class Predictor {

    torch::jit::script::Module model;

public:
    explicit Predictor(const std::string& path);
    torch::Tensor forward(const torch::Tensor& tokens);
    inline torch::Tensor operator()(const torch::Tensor& tokens) {return forward(tokens);}
    static int argmax(const torch::Tensor& probs);
};


#endif //ANS_PREDICTOR_H
