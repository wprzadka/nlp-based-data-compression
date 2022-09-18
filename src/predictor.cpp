//
// Created by viking on 27.06.22.
//

#include "predictor.h"

Predictor::Predictor(const std::string& path, const VocabularyFilter& filter, bool use_filter)
: filter(filter), use_filter(use_filter) {
    try {
        model = torch::jit::load(path);
        model.eval();
    }
    catch (const c10::Error& e) {
        throw std::runtime_error("Can not load the model from \"" + path + "\"");
    }
    torch::manual_seed(0);
}

torch::Tensor Predictor::forward(const torch::Tensor& tokens) {
//    loading model every time solves problem with numerical errors on probabilities
//    model = torch::jit::load("./data/gpt2-lm.pt");
    std::vector<c10::IValue> input{tokens};

    torch::Tensor output = model.forward(input).toTuple()->elements()[0].toTensor();
    torch::Tensor logits = output[0][output.size(1) - 1];
    if(use_filter){
        logits = filter(logits);
    }
    torch::Tensor probs = torch::softmax(logits, 0);
    return probs;
}

int Predictor::argmax(const torch::Tensor& probs){
    torch::Tensor argmax = torch::argmax(probs);
    return argmax.item().toInt();
}
