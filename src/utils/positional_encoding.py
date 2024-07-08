import torch
from configs import Config
import math

"""
Positional encoding layer
    args: (
        emb_size: int, the embedding dimension of the model;
        cfg: Config, the config of the model
    )
    Return a positional encoding function
"""
class PositionalEncoding(torch.nn.Module):
    def __init__(self, cfg: Config):
        max_len = 3000
        super().__init__()
        den = torch.exp(-torch.arange(0, cfg.emb_size, 2) * math.log(10000) / cfg.emb_size)
        pos = torch.arange(0, max_len).reshape(int(max_len), 1)
        pos_embedding = torch.zeros(max_len, cfg.emb_size)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = torch.nn.Dropout(cfg.pos_enc_dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])