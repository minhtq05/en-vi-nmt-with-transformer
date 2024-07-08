from typing import List


"""
Helper transform function for build_vocab
    args: (text: List[str], list of string tokens)
    Return a new string with added 'Begin of sequence' and 'End of sequence' tokens
"""
def add_special_tokens(text: List[str]):
        return ['<bos>'] + text + ['<eos>']
