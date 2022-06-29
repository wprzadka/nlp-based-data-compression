//
// Created by viking on 18.04.22.
//

#include "rans.h"
#include <bits/stdc++.h>
#include <utility>

RANS::RANS(Tokenizer tokenizer, Predictor predictor): tokenizer(std::move(tokenizer)), predictor(std::move(predictor)) {
    use_predictor = true;
}

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
char RANS::get_symbol(uint32_t value){
    auto ptr = std::upper_bound(accumulated.begin(), accumulated.end(), value);
    char symbol = static_cast<char>((ptr - accumulated.begin() - 1) - NEGATIVE_SYMBOLS_NUM);
    return symbol;
}
#endif

std::string RANS::encode(const char* data, uint16_t size) {
    uint32_t state = (1 << HALF_STATE_BITS);
    std::string encoded;

    torch::Tensor tokens = tokenizer(std::string(data));
    std::cout << tokens.sizes() << " | " << tokens.size(1) << "\n";

    // Encode data
    for (long i = tokens.size(1) - 1; i >= 1; --i) {

        long beg = std::max(0l, i - prediction_window);
        torch::Tensor input = torch::unsqueeze(tokens.index({0, torch::indexing::Slice(beg, i, torch::indexing::None)}), 0);

        torch::Tensor probas = predictor(input);
        compute_frequencies_from_probas(probas);

        uint32_t freq = get_frequency(tokens.index({0, i}).item().toInt());
        while (state >= freq * (1 << (STATE_BITS - N_VALUE))){
            encoded += static_cast<char>(state & 255);
            state >>= 8;
            encoded += static_cast<char>(state & 255);
            state >>= 8;
        }
        state = ((state / freq) << N_VALUE) + (state % freq) + get_accumulated(data[i]);
    }

    // Write state at the end of encoding
    uint8_t state_bits = STATE_BITS;
    while (state_bits > 0) {
        encoded += static_cast<char>(state & 255);
        state >>= 8;
        state_bits -= 8;
    }
    std::reverse(encoded.begin(), encoded.end());
    return encoded;
}

std::string RANS::decode(const char* code, uint16_t size) {
    std::string decoded;
    int idx = 0;
    uint32_t state = 0;

    // Reconstruct state of rANS at end of encoding
    uint8_t state_bits = STATE_BITS;
    while (state_bits > 0) {
        state <<= 8;
        state += code[idx++] & 255;
        state_bits -= 8;
    }
    // Decode data
    while(state > (1 << HALF_STATE_BITS)){
        char s = get_symbol(state & MASK);

        decoded += s;
        state = get_frequency(s) * (state >> N_VALUE) + (state & MASK) - get_accumulated(s);

        while (state < (1 << HALF_STATE_BITS) && idx < size) {
            state <<= 8;
            state += code[idx++] & 255;
            state <<= 8;
            state += code[idx++] & 255;
        }
    }
    return decoded;
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
    torch::Tensor freqs = (probabilities * (1 << N_VALUE)).toType(torch::kInt32);

    // TODO normalize sum of frequencies to (1 << N_VAL)
    long diff = (1 << N_VALUE) - freqs.sum().item().toLong();
    while (diff < 0){
        torch::Tensor mask = freqs > 0;
        torch::Tensor nd = diff / mask.sum();
        mask = freqs > nd;
        freqs += mask * nd;
        diff -= (mask.sum() * nd).item().toLong();
    }

    for (int idx = 0; idx < frequencies.size(); ++idx) {
        frequencies[idx] = freqs.index({idx}).item().toInt();
    }

    // Check if all frequencies are in valid range
    for(auto val : frequencies){
        assert(val <= (1 << N_VALUE));
    }
    accumulated = compute_cumulative_freq();
}
