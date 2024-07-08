from dataclasses import dataclass, field
from torch import device
from typing import Dict, Literal
from rich import print

SPECIAL_TOKENS = Literal['UNK_IDX', 'PAD_IDX', 'BOS_IDX', 'EOS_IDX']

@dataclass
class Config:
    data: str
    src_lang: str
    tgt_lang: str
    emb_size: int
    n_head: int
    ffn_hid_dim: int
    max_length: int
    batch_size: int
    n_encoder: int
    n_decoder: int
    lr: float
    n_epoch: int
    model_dropout: float
    pos_enc_dropout: float
    activation: str
    seed: int
    save_model_path: str
    save_vocab_path: str
    verbose: bool
    device: str | device
    special_tokens: Dict[SPECIAL_TOKENS, int] = field(default_factory=lambda: {
        'UNK_IDX': 0,
        'PAD_IDX': 1,
        'BOS_IDX': 2,
        'EOS_IDX': 3,
    })

    def show_config(self):
        print('Model configs:')
        for k, v in vars(self).items():
            print(f'- {k:<20}: {v}')
        