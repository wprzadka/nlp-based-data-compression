//
// Created by viking on 14.09.22.
//

#ifndef NLP_BASED_COMPRESSION_DEBUG_MACROS_H
#define NLP_BASED_COMPRESSION_DEBUG_MACROS_H

//#define DEBUG

#ifdef DEBUG
#define DEBUG_LOG( text, data ) std::cout << (text) << (data) << '\n'
#else
// empty macro
#define DEBUG_LOG( text, data )
#endif

#endif //NLP_BASED_COMPRESSION_DEBUG_MACROS_H
