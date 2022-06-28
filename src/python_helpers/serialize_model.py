import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from transformers import DistilBERT
import os


def save_model(tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel, path: str):
    inp = 'How are you?'

    example_input = tokenizer(inp, return_tensors='pt')
    traced_model = torch.jit.trace(model, [example_input['input_ids']])
    torch.jit.save(traced_model, path)

if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", torchscript=True)
    tokenizer.save_vocabulary('data/gpt2-vocab')
    save_model(tokenizer, model, 'data/gpt2-lm.pt')
