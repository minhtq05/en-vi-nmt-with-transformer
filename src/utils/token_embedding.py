import torch
import math


"""
Token embedding layer
    args: (
        vocab_size: int, the vocab size of the language;
        emb_size: int, the embedding size of the model;
        cfg: Config, The config of the model;
    )
    Return a token embedding function
"""
class TokenEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, emb_size: int, cfg):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_size, padding_idx=cfg.special_tokens['PAD_IDX'])
        self.n_embed = emb_size

    def forward(self, tokens): # (T, B, C)
        return self.embedding(tokens.long()) * math.sqrt(self.n_embed) # (T, B, C)