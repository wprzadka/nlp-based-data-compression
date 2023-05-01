//
// Created by viking on 26.04.22.
//

#include <fstream>
#include "gtest/gtest.h"
#include "../src/rans.h"


static std::map<std::string, std::string> config {
        {"vocab", "./data/gpt2-vocab/vocab.json"},
        {"merges", "./data/gpt2-vocab/merges.txt"},
        {"unicode", "./data/gpt2-vocab/unicode.json"},
        {"predictor", "./data/gpt2-lm.pt"}
};


class RANS_Coding_Test: public ::testing::Test{
public:
    const uint16_t plain_text_size = 1080;
    const char* plain_text =
            "One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin.\n"
            "He lay on his armour-like back, and if he lifted his head a little he could see his brown belly, slightly domed and divided by arches into stiff sections.\n"
            "The bedding was hardly able to cover it and seemed ready to slide off any moment.\n"
            "His many legs, pitifully thin compared with the size of the rest of him, waved about helplessly as he looked.\n"
            "\"What's happened to me? \" he thought. It wasn't a dream.\n"
            "His room, a proper human room although a little too small, lay peacefully between its four familiar walls.\n"
            "A collection of textile samples lay spread out on the table - Samsa was a travelling salesman - and above it there hung a picture that he had recently cut out of an illustrated magazine and housed in a nice, gilded frame.\n"
            "It showed a lady fitted out with a fur hat and fur boa who sat upright, raising a heavy fur muff that covered the whole of her lower arm towards the viewer.\n"
            "Gregor then turned to look out the window at the dull weather. Drops";
};


TEST_F(RANS_Coding_Test, encode_decode_consistency){
    RANS rans(
            Tokenizer(config["vocab"], config["merges"], config["unicode"]),
            Predictor(config["predictor"])
    );

    // encode data
    std::string encoded = rans.encode(plain_text, plain_text_size);

    // decode data
    std::string decoded = rans.decode(encoded.c_str(), encoded.size());

    ASSERT_EQ(plain_text_size, decoded.size());
    for (int i = 0; i < plain_text_size; ++i){
        ASSERT_EQ(plain_text[i], decoded[i]);
    }
}


int main(int argc, char** argv){
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
