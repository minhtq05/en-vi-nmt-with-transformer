import torch
from configs import Config

"""
create square subsequent mask when training and validating for tokens to communicate with each other (only the present token with past tokens, not with future tokens).
    args: (
        sz: int, the size of the tokens;
        device: str or device, the device of the model (cpu or cuda);
    )
    Return the square mask  
"""
def generate_square_subsequent_mask(sz: int, cfg: Config):
    mask = (torch.triu(torch.ones((sz, sz), device=cfg.device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask