//
// Created by viking on 17.04.22.
//

#include <string>
#include <fstream>
#include <getopt.h>
#include <cassert>
#include <numeric>
#include "rans.h"
#include "tokenizer.h"
#include "predictor.h"

static const uint8_t BLOCK_SIZE_BYTES = 2;
static const uint8_t SYMBOL_FREQ_BYTES = 3;

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
    RANS rans(
        Tokenizer(config["vocab"], config["merges"], config["unicode"]),
        Predictor(config["predictor"])
    );
    char* mem_buff = new char[rans.BLOCK_SIZE];
    while(file_reader){
        // Read next block
        file_reader.read(mem_buff, rans.BLOCK_SIZE);
        uint32_t bits_read = file_reader.gcount();

        /*
        // Prepare and frequencies of symbol occurrence
        rans.prepare_frequencies(mem_buff, bits_read);
        */

        // encode block
        std::string enc = rans.encode(mem_buff, bits_read);
        // save block with frequencies to file
        /*
        write_symbol_freqencies(rans.frequencies, file_writer);
        */
         write_size_of_block(file_writer, enc.size());
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
    RANS rans(
            Tokenizer(config["vocab"], config["merges"], config["unicode"]),
            Predictor(config["predictor"])
    );
    char* mem_buff = new char[rans.BLOCK_SIZE];
    // read end of file position
    file_reader.seekg(0, std::ifstream::end);
    long file_length = file_reader.tellg();
    file_reader.seekg(0, std::ifstream::beg);

    while(file_reader && file_reader.tellg() != file_length){
        /*
        // Read frequencies
        std::array<uint32_t, RANS::MAX_SYMBOL> freqs{};
        freqs = read_symbol_frequencies(file_reader);
        rans.init_frequencies(freqs);
        */
         // Read number of bytes in block
        uint32_t bytes_num = read_size_of_block(file_reader);
        // Read next block
        file_reader.read(mem_buff, bytes_num);
        uint32_t bits_read = file_reader.gcount();
        // decode block
        std::string dec = rans.decode(mem_buff, bits_read);
        // save decoded block to file
        file_writer.write(dec.c_str(), static_cast<long>(dec.size()));
    }
    delete[] mem_buff;
    file_reader.close();
    file_writer.close();
    return 0;
}

int main(int argc, char** argv){
    /*
    Tokenizer tokenizer = Tokenizer(
            "./data/gpt2-vocab/vocab.json",
            "./data/gpt2-vocab/merges.txt",
            "./data/gpt2-vocab/unicode.json"
            );
    Predictor pred("./data/gpt2-lm.pt");

    std::string text = "I am";
    for (int i = 0; i < 5; ++i) {
        torch::Tensor tokens = tokenizer(text);

        std::cout << "sizes: " << tokens.sizes() << "\n";
        std::cout << "sizes: " << tokens.index({0, torch::indexing::Slice(0, i + 1, torch::indexing::None)}).sizes() << "\n";

//        tokens = tokens.index(
//                {torch::indexing::Slice(i - 2, i - 1, torch::indexing::None)}
//        );
        int arg = pred(tokens).argmax().item().toInt();
        std::string next = tokenizer.decode({arg});
        std::cout << ">" << next << "\n";
        text += next;
    }
    std::cout << "\n--\n" << text << "\n";

    return 0;
    */
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
