import torch
from configs import Config
from utils.generate_square_subsequent_mask import generate_square_subsequent_mask


"""
greddy decode function
    args: (
        model: nn.Module, the main model;
        ...other parameters for generating tokens from the model;
        max_len: maximum length of the generated translation;
        start_symbol: the starting symbol, usually is BOS_IDX;
        cfg: Config, the config of the model;
    )
    return a list of integer tokens as a translation of the model;
"""
def greedy_decode(model: torch.nn.Module, src, src_mask, max_len, start_symbol, cfg: Config):
    # Certified
    # function to generate output sequence using greedy algorithm
    model.eval()
    src = src.to(cfg.device)
    src_mask = src_mask.to(cfg.device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(cfg.device) # ys is gonna have shape (T, B) with B = 1 when generating
    with torch.no_grad():
        for i in range(max_len-1):
            memory = memory.to(cfg.device)
            tgt_mask = generate_square_subsequent_mask(ys.size(0), cfg=cfg).type(torch.bool).to(cfg.device)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1) # (T, B) -> (B, T)
            prob = model.generator(out[:, -1])
            _, next_word_idx = torch.max(prob, dim=1)
            next_word_idx = next_word_idx.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word_idx)], dim=0)
            if next_word_idx == cfg.special_tokens['EOS_IDX']:
                break
    return ys


def greedy_decode_multiple(model: torch.nn.Module, src, src_mask, max_len, start_symbol, cfg: Config):
    # Certified
    # function to generate output sequence using greedy algorithm
    model.eval()
    src = src.to(cfg.device)
    src_mask = src_mask.to(cfg.device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, src.shape[1]).fill_(start_symbol).type(torch.long).to(cfg.device) # ys is gonna have shape (T, B) when generating
    with torch.no_grad():
        for i in range(max_len-1):
            memory = memory.to(cfg.device)
            tgt_mask = generate_square_subsequent_mask(ys.size(0), cfg=cfg).type(torch.bool).to(cfg.device)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1) # (T, B) -> (B, T)
            prob = model.generator(out[:, -1]) # -> (B, 1, vocab_size)
            print(prob.shape)
            _, next_word_idx = torch.max(prob, dim=1)
            ys = torch.cat([ys, next_word_idx], dim=0)
    return ys