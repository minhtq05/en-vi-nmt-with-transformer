from typing import Literal
from underthesea import word_tokenize
import re

"""
Tokenizer
    args: (lang: str, Literal['en', 'vi'])
    Return a BPE tokenizer implemented from tiktoken
"""
class Tokenizer:
    def __init__(self, lang: Literal['en', 'vi']):
        self.lang = lang
        if (lang == 'en'):
            self.pattern = re.compile(r"\b\w+'\w+|\b\w+|'\b|[^\w\s]")
        # self.enc = tiktoken.get_encoding('o200k_base')

    def __call__(self, sent):
        if (len(sent) == 0):
            return []
        if (self.lang == 'en'):
            return self.pattern.findall(sent)
        elif (self.lang == 'vi'):
            return word_tokenize(sent)
        # return [self.enc.decode([c]) for c in self.enc.encode(sent)]