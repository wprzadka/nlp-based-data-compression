//
// Created by viking on 25.06.22.
//

#include <string>
#include <map>
#include <fstream>
#include "tokenizer.h"
#include "../nlohmann/json.hpp"

Tokenizer::Tokenizer(const std::string& vocabulary_path, const std::string& pair_merges_path){
    vocabulary = read_vocabulary(vocabulary_path);
    pair_merges = read_pair_merges(pair_merges_path);
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

std::map<std::pair<std::string, std::string>, int> Tokenizer::read_pair_merges(const std::string& path){
    std::ifstream file(path, std::ios_base::in);
    if (!file.is_open()) {
        throw std::runtime_error("can't open the file");
    }

    std::map<std::pair<std::string, std::string>, int> merges;
    int rank = 0;
    for(std::string line; getline(file, line); ){
        unsigned int pos = line.find(' ');
        std::pair<std::string, std::string> pair = std::make_pair(line.substr(0, pos), line.substr(pos));
        merges[pair] = rank;
        ++rank;
    }
    return merges;
}

std::vector<std::string> Tokenizer::divide_to_subwords(const std::string &word) {
    // TODO implement subdivision with merge ranks
    return std::vector<std::string>();
}
