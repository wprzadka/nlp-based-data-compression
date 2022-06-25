//
// Created by viking on 25.06.22.
//

#ifndef NLP_BASED_COMPRESSION_TOKENIZER_H
#define NLP_BASED_COMPRESSION_TOKENIZER_H


#include <vector>

class Tokenizer {

    std::map<std::string, int> vocabulary;
    std::map<std::pair<std::string, std::string>, int> pair_merges;

public:
    Tokenizer(const std::string& vocabulary_path, const std::string& pair_merges_path);

private:
    static std::map<std::string, int> read_vocabulary(const std::string&);
    static std::map<std::pair<std::string, std::string>, int> read_pair_merges(const std::string& path);
    std::vector<std::string> divide_to_subwords(const std::string& word);
};


#endif //NLP_BASED_COMPRESSION_TOKENIZER_H
