//
// Created by viking on 25.06.22.
//

#ifndef NLP_BASED_COMPRESSION_TOKENIZER_H
#define NLP_BASED_COMPRESSION_TOKENIZER_H


#include <vector>
#include <torch/script.h>

class Tokenizer {

    typedef std::pair<std::string, std::string> StringPair;

    std::map<std::string, int> vocabulary;
    std::map<int, std::string> decoder;
    std::map<StringPair, int> pair_merges_ranks;
    std::map<char, std::string> unicode;
    std::map<std::string, char> unicode_decoder;

public:

    const static uint16_t VOCAB_SIZE = 50257;

    Tokenizer(const std::string& vocabulary_path, const std::string& pair_merges_path, const std::string& unicodes_path);
    torch::Tensor tokenize(const std::string& text);
    inline torch::Tensor operator()(const std::string& text){return tokenize(text);}
    std::string decode(const std::vector<int> &tokens);

private:
    static std::map<std::string, int> read_vocabulary(const std::string&);
    static std::map<StringPair, int> read_pair_merges(const std::string& path);
    static std::map<char, std::string> read_unicode_mapping(const std::string &path);
    std::vector<std::string> divide_to_subwords(const std::string& word);
    //    static std::vector<StringPair> get_character_pairs(const std::string& word);
    static std::vector<StringPair> get_character_pairs(const std::vector<std::string>& word);
    bool compare_by_rank(const StringPair &a, const StringPair &b);
    std::vector<std::string> bytes_to_unicode(const std::string &word);
    std::string decode_unicodes(const std::string &word);
};


#endif //NLP_BASED_COMPRESSION_TOKENIZER_H
