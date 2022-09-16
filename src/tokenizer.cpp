//
// Created by viking on 25.06.22.
//

#include <string>
#include <map>
#include <regex>
#include <iostream>
#include "tokenizer.h"
#include "../nlohmann/json.hpp"

Tokenizer::Tokenizer(
        const std::string& vocabulary_path,
        const std::string& pair_merges_path,
        const std::string& unicodes_path
        ) {
    vocabulary = read_vocabulary(vocabulary_path);
    pair_merges_ranks = read_pair_merges(pair_merges_path);
    unicode = read_unicode_mapping(unicodes_path);

    for (const auto& entry: vocabulary){
        decoder[entry.second] = entry.first;
    }
    for (const auto& entry: unicode){
        unicode_decoder[entry.second] = entry.first;
    }
}

std::map<std::string, int> Tokenizer::read_vocabulary(const std::string & path) {
    std::ifstream file(path, std::ios_base::in);
    if (!file.is_open()) {
        throw std::runtime_error("can't open the file");
    }

    nlohmann::json json_vocab;
    file >> json_vocab;
    file.close();

    std::map<std::string, int> vocab{};
    for (nlohmann::json::iterator iter = json_vocab.begin(); iter != json_vocab.end(); ++iter){
        vocab[iter.key()] = iter.value();
    }
    return vocab;
}

std::map<Tokenizer::StringPair, int> Tokenizer::read_pair_merges(const std::string& path){
    std::ifstream file(path, std::ios_base::in);
    if (!file.is_open()) {
        throw std::runtime_error("can't open the file");
    }

    std::map<StringPair, int> merges;
    int rank = 0;
    for(std::string line; getline(file, line); ){
        unsigned int pos = line.find(' ');
        StringPair pair = std::make_pair(line.substr(0, pos), line.substr(pos + 1));
        merges[pair] = rank;
        ++rank;
    }
    return merges;
}

std::map<char, std::string> Tokenizer::read_unicode_mapping(const std::string & path) {
    std::ifstream file(path, std::ios_base::in);
    if (!file.is_open()) {
        throw std::runtime_error("can't open the file");
    }

    nlohmann::json json_unicodes;
    file >> json_unicodes;
    file.close();

    std::map<char, std::string> unicodes{};
    for (nlohmann::json::iterator iter = json_unicodes.begin(); iter != json_unicodes.end(); ++iter){
        unicodes[std::stoi(iter.key())] = iter.value();
    }
    return unicodes;
}

torch::Tensor Tokenizer::tokenize(const std::string& text) {
    std::vector<int> tokens{};

    std::regex re(R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?\d+| ?[^a-zA-Z\s\d]+|\s+(?!\S)|\s+)");
    std::smatch result;
    auto iter = text.begin();
    while(std::regex_search(iter, text.end(), result, re)){
        std::string word = result.begin()->str();
        std::vector<std::string> subwords = divide_to_subwords(word);
        for (const std::string& v: subwords){
            tokens.push_back(vocabulary[v]);
        }

        iter += result.prefix().str().size();
        iter += result.str().size();
    }

    auto options = torch::TensorOptions().dtype(torch::kInt32);
    torch::Tensor tensor = torch::from_blob((void *) tokens.data(), {1, static_cast<long>(tokens.size())}, options)
            .to(torch::kInt32)
            .clone();

    return tensor;
}

std::vector<std::string> Tokenizer::divide_to_subwords(const std::string &word) {
    std::vector<std::string> tokens = bytes_to_unicode(word);
    std::vector<StringPair> pairs = get_character_pairs(tokens);
    if (pairs.empty()){
        std::string res{};
        for (const auto& w: tokens){
            res += w;
        }
        return std::vector<std::string>{res};
    }
    while (true) {
        StringPair bigram = *std::min_element(
                pairs.begin(),
                pairs.end(),
                [this](const StringPair& a, const StringPair& b){return compare_by_rank(a, b);}
                );
        if (pair_merges_ranks.find(bigram) == pair_merges_ranks.end()) {
            break;
        }
        std::string first = bigram.first;
        std::string second = bigram.second;

        std::vector<std::string> new_tokens{};
        auto iter = tokens.begin();
        while (iter != tokens.end()){
            auto next_occurrence = std::find(iter, tokens.end(), first);
            new_tokens.insert(new_tokens.end(), iter, next_occurrence);
            iter = next_occurrence;
            if (iter == tokens.end()){
                break;
            }

            auto next = iter + 1;
            if(*iter == first && next != tokens.end() && *next == second){
                new_tokens.push_back(first + second);
                iter = next + 1;
            }else{
                new_tokens.push_back(*iter);
                ++iter;
            }
        }

        tokens = new_tokens;
        if (tokens.size() == 1){
            break;
        }
        pairs = get_character_pairs(tokens);
    }
    return tokens;
}

//std::vector<Tokenizer::StringPair> Tokenizer::get_character_pairs(const std::string& word){
//    std::vector<StringPair> pairs(word.size() - 1);
//    for (int i = 0; i < word.size() - 1; ++i){
//        pairs[i] = std::make_pair(word[i], word[i + 1]);
//    }
//    return pairs;
//}

std::vector<Tokenizer::StringPair> Tokenizer::get_character_pairs(const std::vector<std::string>& word){
    std::vector<StringPair> pairs(word.size() - 1);
    for (int i = 0; i < word.size() - 1; ++i){
        pairs[i] = std::make_pair(word[i], word[i + 1]);
    }
    return pairs;
}

bool Tokenizer::compare_by_rank(const Tokenizer::StringPair &a, const Tokenizer::StringPair &b) {
    auto fst = pair_merges_ranks.find(a);
    auto snd = pair_merges_ranks.find(b);
    if (fst == pair_merges_ranks.end()){
        return false;
    }
    if (snd == pair_merges_ranks.end()){
        return true;
    }
    return fst->second < snd->second;
}

std::vector<std::string> Tokenizer::bytes_to_unicode(const std::string& word){
    std::vector<std::string> new_word;
    for (char w: word){
        new_word.emplace_back(unicode[w]);
    }
    return new_word;
}

std::string Tokenizer::decode(const std::vector<int>& tokens){
    std::string result;
    for (auto v: tokens){
        std::string word = decoder[v];
        result += decode_unicodes(word);
    }
    return result;
}

std::string Tokenizer::decode_unicodes(const std::string& word){
    std::string result = "";
    int beg = 0;
    int end = 1;
    while(end < word.size()){
        std::string subword = word.substr(beg, end - beg);
        auto iter = unicode_decoder.find(subword);
        if(iter != unicode_decoder.end()){
            result += iter->second;
            beg = end;
        }
        ++end;
    }
    if(beg < end){
        std::string subword = word.substr(beg, end);
        auto iter = unicode_decoder.find(subword);
        result += iter->second;
    }

    return result;
}
