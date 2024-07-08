import torch
from .generate_square_subsequent_mask import generate_square_subsequent_mask
from configs import Config

"""
create mask for training / evaluating.
    args: (
        src: List[int], source language tokens;
        tgt: List[int], target language tokens;
        cfg: Config, the config of the model
    )
    Return all the necessary items for the transformer model to generate new tokens 
"""
def create_mask(src, tgt, cfg: Config):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, cfg) # We are predicting the future so past tokens cannot communicate with future tokens
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=cfg.device).type(torch.bool) # We already know every tokens about the src => no need to mask the future

    src_padding_mask = (src == cfg.special_tokens['PAD_IDX']).transpose(0, 1) # boolean tensor (B, T, C) -> (T, B, C)
    tgt_padding_mask = (tgt == cfg.special_tokens['PAD_IDX']).transpose(0, 1) # boolean tensor (B, T, C) -> (T, B, C)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask