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

int Predictor::forward(const std::vector<int>& tokens) {
    auto options = torch::TensorOptions().dtype(torch::kInt32);
    torch::Tensor tensor = torch::from_blob((void *) tokens.data(), {1, static_cast<long>(tokens.size())}, options)
            .to(torch::kInt32)
            .clone();
    std::vector<c10::IValue> input{tensor};

    torch::Tensor output = model.forward(input).toTuple()->elements()[0].toTensor();
    torch::Tensor logits = output[0][output.size(1) - 1];
    torch::Tensor probs = torch::softmax(logits, 0);

    torch::Tensor argmax = torch::argmax(probs);
    return argmax.item().toInt();
}
