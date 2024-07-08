import torch
import os
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import Dataset
from configs import Config
from .yield_tokens import yield_tokens
from .sequential_transforms import sequential_transforms
from .token_transform import token_transform
from .add_special_tokens import add_special_tokens
from .tensor_transform import tensor_transform


"""
Build the vocab based on the train dataset
    args: (
        train_data: Dataset, the train dataset;
        cfg; Config, config of the model
    )
    Return two text_transform functions for processing data from string to list of integers
"""
def build_vocab(train_data: Dataset, cfg: Config):
    if os.path.exists(cfg.save_vocab_path):
        vocab = torch.load(cfg.save_vocab_path)
        vocab_transform = vocab['vocab_transform']
        SRC_VOCAB_SIZE = vocab['SRC_VOCAB_SIZE']
        TGT_VOCAB_SIZE = vocab['TGT_VOCAB_SIZE']
    else:
        special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
        vocab_transform = {}
        cfg.verbose and print("Building vocabs...")
        for ln in [cfg.src_lang, cfg.tgt_lang]:
            # Training data Iterator
            # Create torchtext's Vocab object
            vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_data, ln, cfg),
                                                            min_freq=2,
                                                            specials=special_symbols,
                                                            special_first=True)

        for ln in [cfg.src_lang, cfg.tgt_lang]:
            vocab_transform[ln].set_default_index(cfg.special_tokens['UNK_IDX'])

        SRC_VOCAB_SIZE = len(vocab_transform[cfg.src_lang])
        TGT_VOCAB_SIZE = len(vocab_transform[cfg.tgt_lang])

        torch.save({
            'vocab_transform': vocab_transform,
            "SRC_VOCAB_SIZE": SRC_VOCAB_SIZE,
            "TGT_VOCAB_SIZE": TGT_VOCAB_SIZE,
        }, cfg.save_vocab_path)

    text_transform = {}
    # ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices

    for ln in [cfg.src_lang, cfg.tgt_lang]:
        text_transform[ln] = sequential_transforms(token_transform[ln], # Tokenization
                                                add_special_tokens,
                                                vocab_transform[ln], # Numericalization
                                                tensor_transform)    # Add BOS/EOS and create tensor

    if (cfg.verbose):
        print("source language vocab size:", SRC_VOCAB_SIZE)
        print("target language vocab size:", TGT_VOCAB_SIZE)

    return vocab_transform, text_transform, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE