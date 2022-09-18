//
// Created by viking on 18.09.22.
//

#include "vocabulary_filter.h"
#include "debug_macros.h"

VocabularyFilter::VocabularyFilter(const Tokenizer& tokenizer): tokenizer(tokenizer) {
    auto options = torch::TensorOptions().dtype(torch::kBool);
    filter = torch::full(Tokenizer::VOCAB_SIZE, false, options);
}

torch::Tensor VocabularyFilter::forward(const torch::Tensor &input) const {
    return torch::where(filter, input, 0);
}

void VocabularyFilter::createFilter(const torch::Tensor& tokens) {
    fill(filter, false);
    for(int i = 0; i < tokens.size(1); ++i){
        filter[tokens.index({0, i}).item<int>()] = true;
    }
}

void VocabularyFilter::saveFilter(std::ofstream& file){
    torch::Tensor indices = filter.nonzero();
    char* mem_buff = new char[VOCAB_INDEX_BYTES];

    // save number of tokens that are present in vocabulary
    int num_of_tokens = indices.size(1);
    DEBUG_LOG("number of present tokens = ", num_of_tokens);
    for (int i = VOCAB_INDEX_BYTES - 1; i >= 0; --i) {
        mem_buff[i] = static_cast<char>(num_of_tokens & 255);
        num_of_tokens >>= 8;
    }
    file.write(mem_buff, VOCAB_INDEX_BYTES);

    // save indices of tokens
    // TODO encode differences
    for(int k = 0; k < indices.size(1); ++k){
        int token_index = indices.index({0, k}).item<int>();
        for (int i = VOCAB_INDEX_BYTES - 1; i >= 0; --i) {
            mem_buff[i] = static_cast<char>(token_index & 255);
            token_index >>= 8;
        }
        file.write(mem_buff, VOCAB_INDEX_BYTES);
    }
    delete[] mem_buff;
}

void VocabularyFilter::readFilter(std::ifstream& file){
    fill(filter, false);

    char* mem_buff = new char[VOCAB_INDEX_BYTES];

    // read number of tokens present in vocabulary
    file.read(mem_buff, VOCAB_INDEX_BYTES);
    uint32_t num_of_tokens = 0;
    for (int i = 0; i < VOCAB_INDEX_BYTES; ++i) {
        num_of_tokens <<= 8;
        num_of_tokens += mem_buff[i] & 255;
    }
    DEBUG_LOG("number of present tokens = ", num_of_tokens);

    // populate filter with indices of present tokens
    for(int k = 0; k < num_of_tokens; ++k){
        file.read(mem_buff, VOCAB_INDEX_BYTES);

        uint32_t token_index = 0;
        for (int i = 0; i < VOCAB_INDEX_BYTES; ++i) {
            token_index <<= 8;
            token_index += mem_buff[i] & 255;
        }
        filter[token_index] = true;
    }

    delete[] mem_buff;
}