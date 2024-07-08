from torch.nn.utils.rnn import pad_sequence
from configs import Config

"""
collate function to pre process data during training
"""
def collate_fn(batch, text_transform, cfg: Config):
    # Certified
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[cfg.src_lang](src_sample))
        tgt_batch.append(text_transform[cfg.tgt_lang](tgt_sample))

    src_batch = pad_sequence(src_batch, padding_value=cfg.special_tokens['PAD_IDX'])
    tgt_batch = pad_sequence(tgt_batch, padding_value=cfg.special_tokens['PAD_IDX'])
    return src_batch, tgt_batch
