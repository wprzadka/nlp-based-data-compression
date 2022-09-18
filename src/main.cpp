//
// Created by viking on 17.04.22.
//

#include <string>
#include <fstream>
#include <getopt.h>
#include <cassert>
#include "rans.h"
#include "tokenizer.h"
#include "predictor.h"
#include "debug_macros.h"
#include "vocabulary_filter.h"

static const uint8_t BLOCK_SIZE_BYTES = 2;
static const bool USE_FILTER = false;

static std::map<std::string, std::string> config {
        {"vocab", "./data/gpt2-vocab/vocab.json"},
        {"merges", "./data/gpt2-vocab/merges.txt"},
        {"unicode", "./data/gpt2-vocab/unicode.json"},
        {"predictor", "./data/gpt2-lm.pt"}
};

void write_size_of_block(std::ofstream& file, uint32_t size){
    assert(size < (1 << (BLOCK_SIZE_BYTES * 8)));
    char* mem_buff = new char[BLOCK_SIZE_BYTES];

    for (int i = BLOCK_SIZE_BYTES - 1; i >= 0; --i) {
        mem_buff[i] = static_cast<char>(size & 255);
        size >>= 8;
    }
    file.write(mem_buff, BLOCK_SIZE_BYTES);

    delete[] mem_buff;
}

uint32_t read_size_of_block(std::ifstream& file){
    char* mem_buff = new char[BLOCK_SIZE_BYTES];
    file.read(mem_buff, BLOCK_SIZE_BYTES);

    uint32_t size = 0;
    for (int i = 0; i < BLOCK_SIZE_BYTES; ++i) {
        size <<= 8;
        size += mem_buff[i] & 255;
    }
    return size;
}

int encode_file(const std::string& input_file, const std::string& output_file = "out.bin"){
    std::ifstream file_reader(input_file, std::ios::binary | std::ios::in);
    std::ofstream file_writer(output_file, std::ios::binary | std::ios::out);
    if(!file_reader.is_open() || !file_writer.is_open()){
        return 1;
    }
    Tokenizer tokenizer(config["vocab"], config["merges"], config["unicode"]);
    VocabularyFilter filter(tokenizer);
    Predictor predictor(config["predictor"], filter);
    RANS rans(tokenizer, predictor);

    char* mem_buff = new char[RANS::BLOCK_SIZE];
    while(file_reader){
        // read next block
        file_reader.read(mem_buff, RANS::BLOCK_SIZE);
        uint32_t bits_read = file_reader.gcount();
        torch::Tensor tokens = tokenizer(std::string(mem_buff, bits_read));
        // prepare and save vocabulary filter
        if (USE_FILTER) {
            filter.createFilter(tokens);
            filter.saveFilter(file_writer);
        }
        // encode block
        std::string enc = rans.encode(tokens);
        // save block to file
        write_size_of_block(file_writer, enc.size());
        DEBUG_LOG("size of block: ", enc.size());
        file_writer.write(enc.c_str(), static_cast<long>(enc.size()));
    }
    delete[] mem_buff;
    file_reader.close();
    file_writer.close();
    return 0;
}

int decode_file(const std::string& input_file, const std::string& output_file = "decoded.bin"){
    std::ifstream file_reader(input_file, std::ios::binary | std::ios::in);
    std::ofstream file_writer(output_file, std::ios::binary | std::ios::out);
    if(!file_reader.is_open() || !file_writer.is_open()){
        return 1;
    }
    Tokenizer tokenizer(config["vocab"], config["merges"], config["unicode"]);
    VocabularyFilter filter(tokenizer);
    Predictor predictor(config["predictor"], filter);
    RANS rans(tokenizer, predictor);

    char* mem_buff = new char[RANS::BLOCK_SIZE];
    // read end of file position
    file_reader.seekg(0, std::ifstream::end);
    long file_length = file_reader.tellg();
    file_reader.seekg(0, std::ifstream::beg);

    while(file_reader && file_reader.tellg() != file_length){
        // read vocabulary filter
        if (USE_FILTER) {
            filter.readFilter(file_reader);
        }
        // read number of bytes in block
        uint32_t bytes_num = read_size_of_block(file_reader);
        DEBUG_LOG("size of block: ", bytes_num);
        // read next block
        file_reader.read(mem_buff, bytes_num);
        uint32_t bits_read = file_reader.gcount();
        // decode block
        std::string dec = rans.decode(mem_buff, bits_read);
        DEBUG_LOG("decoded: ", dec);
        // save decoded block to file
        file_writer.write(dec.c_str(), static_cast<long>(dec.size()));
    }
    delete[] mem_buff;
    file_reader.close();
    file_writer.close();
    return 0;
}

int main(int argc, char** argv){

    option option_names[] = {
            {"version", no_argument, nullptr, 'v'},
            {"help", no_argument, nullptr, 'h'},
            {"encode", required_argument, nullptr, 'e'},
            {"decode", required_argument, nullptr, 'd'}
    };

    int opt;
    opt = getopt_long(argc, argv, "vh:e:d", option_names, nullptr);
    switch (opt) {
        case 'v':
            printf("0.1");
            return 0;
        case 'h':
            printf("Possible arguments:\n"
                   "-v --version\n"
                   "    Prints current program version\n"
                   "-h --help\n"
                   "    Prints arguments informations\n"
                   "-e --encode file_path\n"
                   "    Encodes file indicated by \"file_path\"\n"
                   "-d --decode file_path\n"
                   "    Decodes file indicated by \"file_path\"");
            return 0;
        case 'e':
            return encode_file(optarg);
        case 'd':
            return decode_file(optarg);
        case ':':
            printf("Option requires an argument.\n");
            return 1;
        case '?':
        default:
            printf("Unknown argument \"%c\" provided.\n", optopt);
            return 1;
    }
}
