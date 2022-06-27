import numpy as np
import torch
import argparse
import json
import logging

from transformers import GPT2LMHeadModel, GPT2Tokenizer


M_VAL = 2 ** 12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='msg')
    parser.add_argument('output', type=str, help='msg')
    parser.add_argument('-encode', action='store_true', help='encode')
    parser.add_argument('-log_level', type=int, default=1)
    # TODO: configurable model
    return parser.parse_args()


def C_rANS(s, state, symbol_counts, M):
    cumul_counts = np.insert(np.cumsum(symbol_counts),0,0)  # the cumulative frequencies

    s_count = symbol_counts[s]  # current symbol count/frequency
    next_state = (state // s_count) * M + cumul_counts[s] + (state % s_count) 
    return next_state


#The Cumulative frequency inverse function
def cumul_inverse(y, cumul_sum): 
    for i,_s in enumerate(cumul_sum): 
        if y < _s: return i-1


def D_rANS(state, symbol_counts, M):
    cumul_counts = np.insert(np.cumsum(symbol_counts),0,0) #the cumulative frequencies

    slot = state % M #compute the slot     
    s = cumul_inverse(slot, cumul_counts) #decode the symbol
    prev_state = (state // M) * symbol_counts[s] + slot - cumul_counts[s] #update the state
    return s,prev_state


def get_frequencies(probabilities: torch.tensor, M) -> np.ndarray:
    freqs = (probabilities > 0)
    M_red = M - freqs.sum()
    freqs += np.floor(np.array(probabilities * M_red)).astype(int)

    freqs[freqs.argmax()] += (M - freqs.sum())

    return freqs


def Streaming_rANS_encoder(tokens, predictor, tokenizer, filter, M=2**12, verbose=False):
    enc_states = []
    enc_tokens = []
    enc_probs = []
    enc_counts = []

    range_factor = (2 ** (32-12))
    bitstream = [] #initialize stream
    state = 2 ** 16 #state initialized to lM

    prediction_window = 12
    # // WARNING! NO ATTENTION

    number_of_tokens = tokens.size(1)

    for idx, current in reversed(list(enumerate(tokens[0, 1:], 1))): #iterate over the input   

        if verbose:
            print(f'current, idx -> {current, idx}')

        # predict symbol probability
        beg = max(0, idx - prediction_window)
        # // WARNING LOGITS
        prepared_tokens = torch.unsqueeze(tokens[0, beg: idx], 0)
        with torch.no_grad():
            if verbose:
                for c in prepared_tokens[0]:
                    print(tokenizer.decode(c), end='|')
                print()

            enc_tokens.append(prepared_tokens)

            output = predictor(prepared_tokens).logits[:, -1, :]
            logits = output[0]
        # filter words out of vocabulary
        logits[~filter] = -torch.inf
        prob = torch.softmax(logits, dim=0)

        enc_probs.append(prob[filter])

        symbol_counts = get_frequencies(prob, M)

        enc_counts.append(symbol_counts[filter])

        # Output bits to the stream to bring the state in the range for the next encoding
        while state >= range_factor * symbol_counts[current]:
            bitstream.append(state % (2**16))
            state = state // (2**16)

        state = C_rANS(current, state, symbol_counts, M) # The rANS encoding step
        enc_states.append((state, bitstream.copy()))
    return bitstream, enc_states, enc_tokens, enc_probs, enc_counts


def Streaming_rANS_decoder(state, bitstream, symbol_counts, M):
    range_factor = (2 ** (32-12))

    #perform the rANS decoding
    s_decoded, state = D_rANS(state, symbol_counts, M) 

    # remap the state into the acceptable range
#     while state < range_factor * M and len(bitstream) > 0:
    while state < (2 ** 16) and len(bitstream) > 0:
        bits = bitstream.pop()
        state = state * (2 ** 16) + bits
    return s_decoded, state


def decode(tokenizer, model, state, fst_token, length, bitstream, filter, prediction_window=12, verbose=False, M=2**12):
    dec_tokens, dec_probs, dec_counts = [], [], []
    decode_result = tokenizer.decode(fst_token[0])
    tokens = fst_token
    idx = 1
    while state > 0 and length > 0:

        beg = max(0, idx - prediction_window)
        with torch.no_grad():

            prepared_tokens = torch.unsqueeze(tokens[0, beg: idx], 0)
            dec_tokens.append(prepared_tokens)

            if verbose:
                for c in prepared_tokens[0]:
                    print(tokenizer.decode(c), end='|')
                print()

            output = model(prepared_tokens).logits[:, -1, :]
            logits = output[0]

        # filter words out of vocabulary
        logits[~filter] = -torch.inf
        prob = torch.softmax(logits, dim=0)
        dec_probs.append(prob[filter])
        symbol_counts = get_frequencies(prob, M_VAL)
        dec_counts.append(symbol_counts[filter])

        symbol, state = Streaming_rANS_decoder(state, bitstream, symbol_counts, M)

        idx += 1
        length -= 1
        tokens = torch.cat((tokens, torch.tensor([[symbol]])), 1)
        decode_result += tokenizer.decode(symbol)
    print(state, length)
    return decode_result, dec_tokens


def bitstring_to_bytes(s):
    length = (len(s) + 7) // 8
    return int(s, 2).to_bytes(length, byteorder='big'), length


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel([logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR][args.log_level])

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    if args.encode:
        with open(args.input, 'r') as f:
            msg = f.read()

        tokens = tokenizer(msg, return_tensors="pt")['input_ids']
        filter_ = torch.zeros(tokenizer.vocab_size, dtype=bool)
        filter_[tokens] = 1

        total = int(filter_.sum())

        use_bitmask = tokenizer.vocab_size < total * 2 + int(np.ceil(np.log2(total)))

        bitstream, enc_states, enc_tokens, enc_probs, enc_counts = Streaming_rANS_encoder(tokens, model, tokenizer, filter_, M_VAL)

        state = enc_states[-1][0]
        fst_token = tokens[:,0]
        length = len(enc_counts)

        seq = ''.join(map(str,filter_.int().numpy()))
        logger.info('len(seq) %s', len(seq))

        with open(args.output, 'wb') as f:
            f.write(length.to_bytes(4, byteorder='big'))
            f.write(int(fst_token).to_bytes(4, byteorder='big'))

            flags = 0 | use_bitmask

            f.write(flags.to_bytes(1, byteorder='big'))
            if use_bitmask:
                s, ll = bitstring_to_bytes(seq)
                f.write(ll.to_bytes(2, byteorder='big'))
                logger.info('s %s', s)
                logger.info('ll %s', ll)
                f.write(s)
            else:
                # save indices instead
                f.write(total.to_bytes(2, byteorder='big'))
                for x in filter_.nonzero():
                    f.write(int(x).to_bytes(2, byteorder='big'))

            # TODO: try to optimize these sizes
            f.write(len(bitstream).to_bytes(4, byteorder='big'))
            for b in bitstream:
                f.write(int(b).to_bytes(4, byteorder='big'))

            f.write(int(state).to_bytes(16, byteorder='big'))
    else:
        with open(args.input, 'rb') as f:
            length = int.from_bytes(f.read(4), byteorder='big')
            fst_token = torch.tensor(int.from_bytes(f.read(4), byteorder='big'), dtype=torch.int).unsqueeze(0).unsqueeze(0)

            flags = int.from_bytes(f.read(1), byteorder='big')

            use_bitmask = bool(flags & 1)

            if use_bitmask:
                num_filter_bytes = int.from_bytes(f.read(2), byteorder='big')
                filter_bytes = f.read(num_filter_bytes)
                filter_ = int.from_bytes(filter_bytes, byteorder='big')
                logger.info('num_filter_bytes: %s, filter_bytes: %s', num_filter_bytes, filter_bytes)
                filter_bin = bin(filter_)
                filter_bin = filter_bin[0] + filter_bin[2:]
                logger.info('filter mask length: %s', len(filter_bin))
                filter_ = torch.tensor([int(x) for x in filter_bin], dtype=torch.bool)
            else:
                total = int.from_bytes(f.read(2), byteorder='big')
                indices = [0] * total
                for i in range(total):
                    indices[i] = int.from_bytes(f.read(2), byteorder='big')
                filter_ = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
                filter_[indices] = 1


            bs_length = int.from_bytes(f.read(4), byteorder='big')

            bitstream = []
            for i in range(bs_length):
                bitstream.append(int.from_bytes(f.read(4), byteorder='big'))

            state = int.from_bytes(f.read(16), byteorder='big')

        logger.info('length: %s', length)
        logger.info('fst_token: %s', fst_token)
        logger.info('flags: %s', flags)
        logger.info('filter: %s', filter_.shape)
        logger.info('bitstream: %s', bitstream)
        logger.info('state: %s', state)

        result, dec_tokens = decode(tokenizer=tokenizer,
                model=model,
                state=state,
                fst_token=fst_token,
                length=length,
                bitstream=bitstream,
                filter=filter_,
                M=M_VAL)
        with open(args.output, 'w') as f:
            f.write(result)

