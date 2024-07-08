import torch
from typing import List 


"""
Helper transform function for build_vocab
    args: (tokend_ids: List[int], the list of tokens)
    Return a token list as tensor
"""
def tensor_transform(token_ids: List[int]):
    return torch.as_tensor(token_ids)