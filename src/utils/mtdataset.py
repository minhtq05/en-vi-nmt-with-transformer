from torch.utils.data import Dataset
from typing import List, Literal

"""
Dataset class
    args: (
        src: List[str], source language list of sentences;
        tgt: List[str], target language list of sentences;
        split: Literal['train', 'test'], just a label for the dataset;
    )
    Return a torch Dataset object for creating Dataloader

"""
class MTDataset(Dataset):
    def __init__(self, src: List[str], tgt: List[str], split: Literal['train', 'test']):
        super().__init__()
        self.split = split
        self.X = src
        self.y = tgt

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

    def __len__(self):
        return len(self.X)