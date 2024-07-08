import re


re_clean_patterns = [
    (re.compile(r"&amp; lt ;.*?&amp; gt ;"), ""),
    (re.compile(r"&amp; lt ;"), "<"),
    (re.compile(r"&amp; gt ;"), ">"),
    (re.compile(r"&amp; amp ; quot ;"), "\""),
    (re.compile(r"&amp; amp ; amp ;"), "&"),
    (re.compile(r"&amp; amp ;"), "&"),
    (re.compile(r"&apos; "), ""),
    (re.compile(r"&apos;"), "'"),
    (re.compile(r"&quot;"), "\""),
    (re.compile(r"&#91;"), ""),
    (re.compile(r"&#93;"), ""),
]


"""
Preprocess the train and test datasets with cleaning patterns
    args: (
        en_train: List[str], English train set
        en_test: List[str], English test set
        vi_train: List[str], Vietnamese train set
        vi_test: List[str], Vietnamese test set
    )
    Return all the train and test set just like the read_dataset function
"""
def preprocess(en_train, en_test, vi_train, vi_test):
    def clean(sent):
        sent = sent.rstrip("\n")
        for pattern, repl in re_clean_patterns:
            sent = re.sub(pattern, repl, sent)
        return sent
    
    en_train = [clean(sent) for sent in en_train]
    vi_train = [clean(sent) for sent in vi_train]
    en_test = [clean(sent) for sent in en_test]
    vi_test = [clean(sent) for sent in vi_test]

    return en_train, en_test, vi_train, vi_test