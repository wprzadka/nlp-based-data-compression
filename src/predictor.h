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
    int forward(const std::vector<int>& tokens);
    inline int operator()(const std::vector<int>& tokens) {return forward(tokens);}
};


#endif //ANS_PREDICTOR_H
