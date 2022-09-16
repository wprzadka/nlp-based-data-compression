//
// Created by viking on 18.04.22.
//

#include "rans.h"
#include "debug_macros.h"
#include <bits/stdc++.h>
#include <utility>

RANS::RANS(Tokenizer tokenizer, Predictor predictor): tokenizer(std::move(tokenizer)), predictor(std::move(predictor)) {}

std::array<uint32_t, RANS::MAX_SYMBOL> RANS::compute_cumulative_freq(){
    std::array<uint32_t, RANS::MAX_SYMBOL> acc{};
    acc[0] = 0;
    std::partial_sum(frequencies.begin(), frequencies.end() - 1, acc.begin() + 1);
    return acc;
}


#ifdef USE_LOOKUP_TABLE
char RANS::get_symbol(uint32_t value){
    return symbols_lookup[value];
}
#else
RANS::SYMBOL RANS::get_symbol(uint32_t value){
    auto ptr = std::upper_bound(accumulated.begin(), accumulated.end(), value);
    SYMBOL symbol = static_cast<SYMBOL>(ptr - accumulated.begin() - 1);
    return symbol;
}
#endif

std::string RANS::encode(const char* data, uint16_t size) {
    uint32_t state = (1 << HALF_STATE_BITS);
    std::string encoded;
    torch::Tensor tokens = tokenizer(std::string(data, size));

    // Encode data
    for (long i = tokens.size(1) - 1; i >= 1; --i) {

        long beg = std::max(0l, i - prediction_window);
        torch::Tensor input = torch::unsqueeze(tokens.index(
                {0,torch::indexing::Slice(beg, i, torch::indexing::None)}
                ), 0);
        DEBUG_LOG("context: ", input);

        torch::Tensor probas = predictor(input);
        compute_frequencies_from_probas(probas);

        uint32_t symbol = tokens.index({0, i}).item<int>();
        DEBUG_LOG("symbol: (" + tokenizer.decode({(int)symbol}) + ")| ", symbol);

        uint32_t freq = get_frequency(symbol);
        DEBUG_LOG("freq: ", freq);
        DEBUG_LOG("accum: ", get_accumulated(symbol));

        assert(freq > 0);
        while (state >= (freq << (STATE_BITS - N_VALUE))){
            encoded += static_cast<char>(state & 255);
            state >>= 8;
            encoded += static_cast<char>(state & 255);
            state >>= 8;
        }
        state = ((state / freq) << N_VALUE) + (state % freq) + get_accumulated(symbol);
        DEBUG_LOG("state: ", state);
    }

    DEBUG_LOG("state at end: ", state);
    // Write state at the end of encoding
    uint8_t state_bits = STATE_BITS;
    while (state_bits > 0) {
        encoded += static_cast<char>(state & 255);
        state >>= 8;
        state_bits -= 8;
    }

    // save last symbol without encoding;
    uint16_t first_symbol = tokens.index({0, 0}).item().toShort();
    DEBUG_LOG("first symbol: ", first_symbol);

    encoded += static_cast<char>(first_symbol & 255);
    encoded += static_cast<char>((first_symbol >> 8) & 255);

    std::reverse(encoded.begin(), encoded.end());
    return encoded;
}

std::string RANS::decode(const char* code, uint16_t size) {
    std::deque<SYMBOL> decoded;
    int idx = 0;
    uint32_t state = 0;

    // read first symbol (not encoded)
    uint16_t first_symbol = 0;
    first_symbol += code[idx++] & 255;
    first_symbol <<= 8;
    first_symbol += code[idx++] & 255;

    DEBUG_LOG("first symbol: (" + tokenizer.decode({first_symbol}) + ")| ", first_symbol);
    decoded.push_back(first_symbol);

    // Reconstruct state of rANS at end of encoding
    uint8_t state_bits = STATE_BITS;
    while (state_bits > 0) {
        state <<= 8;
        state += code[idx++] & 255;
        state_bits -= 8;
    }
    DEBUG_LOG("reconstructed state: ", state);

    auto options = torch::TensorOptions().dtype(torch::kInt32);
    torch::Tensor context = torch::empty(prediction_window, options);

    int sym_idx = 0;
    // Decode data
    while(state > (1 << HALF_STATE_BITS)){
        ++sym_idx;
        int beg = std::max(0, sym_idx - prediction_window);
        int window_size = std::min(sym_idx, prediction_window);
        std::deque<RANS::SYMBOL>::iterator iter = decoded.begin() + beg;
        for(int i = 0; i < window_size; ++i){
            context[i] = *iter;
            ++iter;
        }

        torch::Tensor input = torch::unsqueeze(context.index({
            torch::indexing::Slice(0, window_size, torch::indexing::None)
        }), 0);
        DEBUG_LOG("context: ", input);

        torch::Tensor probas = predictor(input);
        compute_frequencies_from_probas(probas);

        SYMBOL s = get_symbol(state & MASK);
        decoded.push_back(s);

        DEBUG_LOG("symbol: (" + tokenizer.decode({s})  + ")| ", s);
        DEBUG_LOG("freq: ", get_frequency(s));
        DEBUG_LOG("accum: ", get_accumulated(s));

        state = get_frequency(s) * (state >> N_VALUE) + (state & MASK) - get_accumulated(s);
        while (state < (1 << HALF_STATE_BITS) && idx < size) {
            state <<= 8;
            state += code[idx++] & 255;
            state <<= 8;
            state += code[idx++] & 255;
        }
        DEBUG_LOG("state: ", state);
    }

    std::vector<int> tokens({decoded.begin(), decoded.end()});
    std::string res = tokenizer.decode(tokens);

    return res;
}

void RANS::normalize_symbol_frequencies(){
    // Find probabilities of symbols occurrences
    uint32_t sum_freq = 0;
    for (uint32_t val : frequencies) {
        sum_freq += val;
    }
    std::map<unsigned char, double> probabilities{};
    for (unsigned char unsigned_symbol = 0; unsigned_symbol < MAX_SYMBOL; ++unsigned_symbol){
        if (frequencies[unsigned_symbol] != 0) {
            probabilities[unsigned_symbol] = static_cast<double>(frequencies[unsigned_symbol]) / sum_freq;
        }
    }
    // Normalize occurrence probabilities to fractions of 2^N_VALUE
    sum_freq = 0;
    for (auto& pair: probabilities){
        uint32_t new_freq = static_cast<uint32_t>(pair.second * (1 << N_VALUE));
        new_freq = new_freq == 0 ? 1 : new_freq;
        frequencies[pair.first] = new_freq;
        sum_freq += new_freq;
    }
    // Ensure that frequencies sums to 2^N
    auto iter = std::find_if(
            frequencies.begin(),
            frequencies.end(),
            [](uint32_t x){return x > 0;}
            );
    *iter += (1 << N_VALUE) - sum_freq;
    // Check if all frequencies are in valid range
    for(auto val : frequencies){
        assert(val <= (1 << N_VALUE));
    }
}

void RANS::compute_frequencies_from_probas(const torch::Tensor& probabilities) {
    int32_t range_limit = 1 << N_VALUE;

    torch::Tensor freqs = (probabilities * (range_limit - MAX_SYMBOL) + 1).floor().toType(torch::kInt32);
    for(int idx = 0; idx < MAX_SYMBOL; ++idx){
        frequencies[idx] = freqs[idx].item<int>();
    }

    // reduce some probabilities to hold the sum limit
    int diff = range_limit - static_cast<int>(std::accumulate(frequencies.begin(), frequencies.end(), 0u));
    int argmax = freqs.argmax().item<int>();
    frequencies[argmax] += diff;

    /* TODO slower but more accurate solution
    assert(diff > 0);
    for(int idx = 0; diff > 0; idx = (idx + 1) % MAX_SYMBOL) {
        ++frequencies[idx];
        --diff;
    }
    assert(diff == 0);
    */

    accumulated[0] = 0;
    for (int idx = 1; idx < MAX_SYMBOL; ++idx) {
        accumulated[idx] = accumulated[idx - 1] + frequencies[idx - 1];
    }

    for (int idx = 1; idx < MAX_SYMBOL; ++idx) {
        assert(accumulated[idx] - accumulated[idx - 1] != 0);
    }
    assert(accumulated[0] == 0);
    assert(accumulated[MAX_SYMBOL - 1] + frequencies[MAX_SYMBOL - 1] == static_cast<uint32_t>(range_limit));
}