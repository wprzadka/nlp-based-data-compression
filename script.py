import os
import numpy as np
import torch
import argparse
import json
import logging
import math

from transformers import GPT2LMHeadModel, GPT2Tokenizer


M_VAL = 2 ** 12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str, help='msg')
    parser.add_argument('output', type=str, help='msg')
    parser.add_argument('-encode', action='store_true', help='encode')
    parser.add_argument('-log_level', type=int, default=1)
    parser.add_argument('-verbose', action='store_true', help='additional verbosity')
    parser.add_argument('-batch_size', type=int, default=8192, help='how many bytes of text per block')
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
    range_factor = (2 ** (32-12))
    bitstream = []
    state = 2 ** 16 #state initialized to lM

    prediction_window = 12
    # // WARNING! NO ATTENTION

    # list() will pull everything into memory right?
    #for idx, current in reversed(list(enumerate(tokens[0, 1:], 1))):
    for idx, current in zip(range(len(tokens[0]) - 1, 0, -1), reversed(tokens[0, 1:])):

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

            output = predictor(prepared_tokens).logits[:, -1, :]
            logits = output[0]
        # filter words out of vocabulary
        logits[~filter] = -torch.inf
        prob = torch.softmax(logits, dim=0)

        symbol_counts = get_frequencies(prob, M)

        # Output bits to the stream to bring the state in the range for the next encoding
        while state >= range_factor * symbol_counts[current]:
            bitstream.append(state % (2**16))
            state = state // (2**16)

        state = C_rANS(current, state, symbol_counts, M) # The rANS encoding step
    return bitstream, state, int(tokens.shape[1])


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
    decode_result = tokenizer.decode(fst_token[0])
    tokens = fst_token
    idx = 1
    while state > 0 and length > 0:

        beg = max(0, idx - prediction_window)
        with torch.no_grad():

            prepared_tokens = torch.unsqueeze(tokens[0, beg: idx], 0)

            if verbose:
                for c in prepared_tokens[0]:
                    print(tokenizer.decode(c), end='|')
                print()

            output = model(prepared_tokens).logits[:, -1, :]
            logits = output[0]

        # filter words out of vocabulary
        logits[~filter] = -torch.inf
        prob = torch.softmax(logits, dim=0)
        symbol_counts = get_frequencies(prob, M_VAL)

        symbol, state = Streaming_rANS_decoder(state, bitstream, symbol_counts, M)

        idx += 1
        length -= 1
        tokens = torch.cat((tokens, torch.tensor([[symbol]])), 1)
        decode_result += tokenizer.decode(symbol)
    logger.debug('state: %s, length: %s', state, length)
    return decode_result


def bitstring_to_bytes(s, byteorder='big'):
    length = (len(s) + 7) // 8
    return int(s, 2).to_bytes(length, byteorder=byteorder), length


#https://stackoverflow.com/questions/2301789/how-to-read-a-file-in-reverse-order#23646049
def reverse_read(fh, buf_size=8192):
    """A generator that returns characters from a file in `buf_size` chunks in reverse order"""
    offset = 0
    fh.seek(0, os.SEEK_END)
    file_size = remaining_size = fh.tell()
    while remaining_size > 0:
        offset = min(file_size, offset + buf_size)
        fh.seek(file_size - offset)
        buffer = fh.read(min(remaining_size, buf_size))
        remaining_size -= buf_size
        yield buffer


def save_block(f, fst_token, filter_, vocab_size, bitstream, state, length, byteorder='big'):
    total = int(filter_.sum())
    use_bitmask = vocab_size < total * 2 + int(np.ceil(np.log2(total)))

    seq = ''.join(map(str,filter_.int().numpy()))
    logger.debug('len(seq) %s', len(seq))

    f.write(length.to_bytes(4, byteorder=byteorder))
    f.write(int(fst_token).to_bytes(4, byteorder=byteorder))

    flags = 0 | use_bitmask

    f.write(flags.to_bytes(1, byteorder=byteorder))
    if use_bitmask:
        filter_mask_as_bytes, filter_mask_as_bytes_length = bitstring_to_bytes(seq, byteorder=byteorder)
        f.write(filter_mask_as_bytes_length.to_bytes(2, byteorder=byteorder))
        logger.debug('filter_mask_as_bytes_length: %s', filter_mask_as_bytes_length)
        logger.debug('filter_mask_as_bytes: %s', filter_mask_as_bytes)
        f.write(s)
    else:
        # save indices instead
        f.write(total.to_bytes(2, byteorder=byteorder))
        for x in filter_.nonzero():
            f.write(int(x).to_bytes(2, byteorder=byteorder))

    # TODO: try to optimize these sizes
    f.write(len(bitstream).to_bytes(4, byteorder=byteorder))
    for b in bitstream:
        f.write(int(b).to_bytes(4, byteorder=byteorder))

    # numpy throws an error if the number is too large
    num_state_bytes = math.ceil(math.log2(int(state)))
    f.write(int(num_state_bytes).to_bytes(4, byteorder=byteorder))
    f.write(int(state).to_bytes(num_state_bytes, byteorder=byteorder))

class EOFError(Exception):
    pass

def read_block(f):
    # TODO: minimize number of f.read()s? (use larger buffers when possible)
    length = int.from_bytes(f.read(4), byteorder=byteorder)
    if not length:  # next block doesn't exist
        raise EOFError()
    fst_token = torch.tensor(int.from_bytes(f.read(4), byteorder=byteorder), dtype=torch.int).unsqueeze(0).unsqueeze(0)
    flags = int.from_bytes(f.read(1), byteorder=byteorder)
    use_bitmask = bool(flags & 1)

    if use_bitmask:
        num_filter_bytes = int.from_bytes(f.read(2), byteorder=byteorder)
        filter_bytes = f.read(num_filter_bytes)
        filter_ = int.from_bytes(filter_bytes, byteorder=byteorder)
        logger.debug('num_filter_bytes: %s, filter_bytes: %s', num_filter_bytes, filter_bytes)
        filter_bin = bin(filter_)
        # FIXME: binary string was 1 char shorter, missing a '0' at the beginning
        #        not sure why though, it was saved properly
        logger.debug('filter mask length: %s', len(filter_bin))
        filter_bin = filter_bin[0] + filter_bin[2:]
        logger.debug('filter mask length after prepending 0: %s', len(filter_bin))
        filter_ = torch.tensor([int(x) for x in filter_bin], dtype=torch.bool)
    else:
        total = int.from_bytes(f.read(2), byteorder=byteorder)
        indices = [0] * total
        for i in range(total):
            indices[i] = int.from_bytes(f.read(2), byteorder=byteorder)
        filter_ = torch.zeros(tokenizer.vocab_size, dtype=torch.bool)
        filter_[indices] = 1

    bs_length = int.from_bytes(f.read(4), byteorder=byteorder)
    bitstream = []
    for i in range(bs_length):
        bitstream.append(int.from_bytes(f.read(4), byteorder=byteorder))

    num_state_bytes = int.from_bytes(f.read(4), byteorder=byteorder)
    state = int.from_bytes(f.read(num_state_bytes), byteorder=byteorder)

    logger.debug('length: %s', length)
    logger.debug('fst_token: %s', fst_token)
    logger.debug('flags: %s', flags)
    logger.debug('filter: %s', filter_.shape)
    logger.debug('bitstream: %s', bitstream)
    logger.debug('state: %s', state)
    return fst_token, length, state, bitstream, filter_


if __name__ == '__main__':
    args = parse_args()

    logging.basicConfig()
    logger = logging.getLogger(__name__)
    logger.setLevel([logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR][args.log_level])

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()

    byteorder = 'big'

    if args.encode:
        with open(args.input, 'r') as fin, open(args.output, 'wb') as fout:
            filter_ = torch.zeros(tokenizer.vocab_size, dtype=bool)
            for batch in reverse_read(fin, buf_size=args.batch_size):
                filter_.zero_()
                tokens = tokenizer(batch, return_tensors="pt")['input_ids']
                filter_[tokens] = 1
                bitstream, state, length = Streaming_rANS_encoder(tokens, model, tokenizer, filter_, M=M_VAL, verbose=args.verbose)

                save_block(fout, tokens[:,0], filter_, tokenizer.vocab_size, bitstream, state, length, byteorder=byteorder)
    else:
        with open(args.input, 'rb') as fin, open(args.output, 'w') as fout:
            while True:
                try:
                    fst_token, length, state, bitstream, filter_ = read_block(fin)

                    result = decode(tokenizer=tokenizer,
                            model=model,
                            state=state,
                            fst_token=fst_token,
                            length=length,
                            bitstream=bitstream,
                            filter=filter_,
                            M=M_VAL,
                            verbose=args.verbose)
                    fout.write(result)
                except EOFError as error:
                    break
