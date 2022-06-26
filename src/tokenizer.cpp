//
// Created by viking on 25.06.22.
//

#include <string>
#include <map>
#include <fstream>
#include <regex>
#include "tokenizer.h"
#include "../nlohmann/json.hpp"

Tokenizer::Tokenizer(const std::string& vocabulary_path, const std::string& pair_merges_path){
    vocabulary = read_vocabulary(vocabulary_path);
    pair_merges_ranks = read_pair_merges(pair_merges_path);
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
//        std::cout << iter.key() << ": " << iter.value() << "\n";
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

std::vector<int> Tokenizer::tokenize(const std::string& text) {
    std::vector<int> tokens{};

    std::regex re(R"('s|'t|'re|'ve|'m|'ll|'d| ?\w+|[,.!?])");
    std::smatch result;
    auto iter = text.begin();
    while(std::regex_search(iter, text.end(), result, re)){
        std::string word = result.begin()->str();
        if (word[0] == ' ') {
            word = word.substr(1);
        }
        std::vector<std::string> subwords = divide_to_subwords(word);
        // TODO work with leading spaces properly
        for (const std::string& v: subwords){
            tokens.push_back(vocabulary[v]);
        }

        iter += (result.begin()->str().size());
    }

    return tokens;
}

std::vector<std::string> Tokenizer::divide_to_subwords(const std::string &word) {
    std::vector<std::string> tokens(word.size());
    for (int i = 0; i < word.size(); ++i){
        tokens[i] = word[i];
    }
    std::vector<StringPair> pairs = get_character_pairs(tokens);
    if (pairs.empty()){
        return std::vector<std::string>{word};
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
