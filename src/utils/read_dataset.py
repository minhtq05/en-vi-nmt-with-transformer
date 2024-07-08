from typing import Literal
from configs import Config

"""
dataset reader
    args: (
        dataset: str, name of the dataset. Either './data/IWSLT-2015' or './data/PhoMT';
        cfg: Config, the config of the model;
    )
    Return four datasets respectively:
        train and test for source language
        train and test for target language
"""
def read_dataset(dataset: Literal['./data/IWSLT-2015/', './data/PhoMT/'], cfg: Config):
    cfg.verbose and print(f"Loading datasets from {dataset}...")

    if (dataset == './data/IWSLT-2015/'):
        with open(f'{dataset}/test/tst2012.en', 'r') as f:
            en_test = f.readlines()
        with open(f'{dataset}/test/tst2012.vi', 'r') as f:
            vi_test = f.readlines()
    elif (dataset == './data/PhoMT/'):
        with open(f'{dataset}/test/test.en', 'r') as f:
            en_test = f.readlines()
        with open(f'{dataset}/test/test.vi', 'r') as f:
            vi_test = f.readlines()
    else:
        raise FileNotFoundError(f'No such dataset found! Expect \'./data/IWSLT-2015/\' or \'./data/PhoMT/\', got \'{dataset}\' instead.')

    with open(f'{dataset}/train/train.en', 'r') as f:
        en_train = f.readlines()
    with open(f'{dataset}/train/train.vi', 'r') as f:
        vi_train = f.readlines()

    return en_train, en_test, vi_train, vi_test 