//
// Created by viking on 27.06.22.
//

#include "predictor.h"

Predictor::Predictor(const std::string& path) {
    try {
        model = torch::jit::load(path);
    }
    catch (const c10::Error& e) {
        throw std::runtime_error("Can not load the model from \"" + path + "\"");
    }
}

torch::Tensor Predictor::forward(const torch::Tensor& tokens) {
    std::vector<c10::IValue> input{tokens};

    torch::Tensor output = model.forward(input).toTuple()->elements()[0].toTensor();
    torch::Tensor logits = output[0][output.size(1) - 1];
    torch::Tensor probs = torch::softmax(logits, 0);
    return probs;
}

int Predictor::argmax(const torch::Tensor& probs){
    torch::Tensor argmax = torch::argmax(probs);
    return argmax.item().toInt();
}
