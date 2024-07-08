from torch.utils.data import Dataset
from configs import Config
from .token_transform import token_transform


"""
Yield tokens for building vocab
    args: (
        data: Dataset, the dataset;
        langauge: current langauge of the dataset;
        cfg: Config, the config of the model;
    )
    return a list of tokens for building vocab,

"""
def yield_tokens(data: Dataset, language: str, cfg: Config):
    language_index = {cfg.src_lang: 0, cfg.tgt_lang: 1}

    for data_sample in data:
        # data_sample: (en_sent, vi_sent)
        yield token_transform[language](data_sample[language_index[language]])