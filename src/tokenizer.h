//
// Created by viking on 25.06.22.
//

#ifndef NLP_BASED_COMPRESSION_TOKENIZER_H
#define NLP_BASED_COMPRESSION_TOKENIZER_H


#include <vector>

class Tokenizer {

    typedef std::pair<std::string, std::string> StringPair;

    std::map<std::string, int> vocabulary;
    std::map<StringPair, int> pair_merges_ranks;
    std::map<char, std::string> unicode;

public:
    Tokenizer(const std::string& vocabulary_path, const std::string& pair_merges_path, const std::string& unicodes_path);
    std::vector<int> tokenize(const std::string& text);
    inline std::vector<int> operator()(const std::string& text){return tokenize(text);}

private:
    static std::map<std::string, int> read_vocabulary(const std::string&);
    static std::map<StringPair, int> read_pair_merges(const std::string& path);
    static std::map<char, std::string> read_unicode_mapping(const std::string &path);
    std::vector<std::string> divide_to_subwords(const std::string& word);
    //    static std::vector<StringPair> get_character_pairs(const std::string& word);
    static std::vector<StringPair> get_character_pairs(const std::vector<std::string>& word);
    bool compare_by_rank(const StringPair &a, const StringPair &b);
    std::string bytes_to_unicode(const std::string &word);
};


#endif //NLP_BASED_COMPRESSION_TOKENIZER_H
