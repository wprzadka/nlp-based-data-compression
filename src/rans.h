//
// Created by viking on 18.04.22.
//

#ifndef ANS_RANS_H
#define ANS_RANS_H


#include <cstdint>
#include <map>
#include <climits>
#include "gtest/gtest.h"
#include "predictor.h"
#include "tokenizer.h"

#define USE_LOOKUP_TABLE

class RANS {
    FRIEND_TEST(RANS_Test, get_symbol);
    FRIEND_TEST(RANS_Test, prepare_frequencies);

    Tokenizer tokenizer;
    Predictor predictor;
    const int prediction_window = 32;
public:
    const static uint8_t N_VALUE = 22;
    const static uint32_t MASK = (1 << N_VALUE) - 1;

    const static uint8_t STATE_BITS = 48;
    const static uint8_t HALF_STATE_BITS = STATE_BITS >> 1;

    const static uint16_t MAX_SYMBOL = 50256;
    const static uint16_t BLOCK_SIZE = 8192;

    std::array<uint32_t, MAX_SYMBOL> frequencies{};
    std::array<uint32_t, MAX_SYMBOL> accumulated{};
    bool use_predictor = false;

    RANS(Tokenizer tokenizer, Predictor predictor);

    inline uint32_t get_frequency(uint32_t symbol) {return frequencies[static_cast<uint32_t>(symbol)];};
    inline uint32_t get_accumulated(uint32_t symbol) {return accumulated[static_cast<uint32_t>(symbol)];};

    std::string encode(const char* data, uint16_t size);
    std::string decode(const char* state, uint16_t size);

protected:
#ifdef USE_LOOKUP_TABLE
    std::array<char, 1 << N_VALUE> symbols_lookup{};
#endif
    std::array<uint32_t, MAX_SYMBOL> compute_cumulative_freq();
    void normalize_symbol_frequencies();
    char get_symbol(uint32_t value);

    void compute_frequencies_from_probas(const torch::Tensor &probabilities);
};


#endif //ANS_RANS_H
