import torch
from tqdm import tqdm
from sacrebleu.metrics import BLEU
from translate import translate_greedy, translate_greedy_multiple
from typing import List
from configs import Config


def create_batch(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


# def calculate_bleu_greedy(model: torch.nn.Module, epoch, vocab_transform, text_transform, mt, md, en_test: List[str], vi_test: List[str], cfg: Config):
#     print("Calculating BLEU score with greedy translate...")
#     model.eval()
#     pred_greedy = []

#     for sent in tqdm(create_batch(en_test), dynamic_ncols=True):
#         text = translate_greedy_multiple(model, sent, mt, vocab_transform, text_transform, cfg)
#         text = [sent.split(' ') for sent in text]
#         text = [md.detokenize(sent) for sent in text]
#         pred_greedy.update(text)

#     ref = [[md.detokenize(s.split(' ')) for s in vi_test]]

#     with open(f"bleu/greedy/{epoch}_translation.txt", "w") as f:
#         for sent in pred_greedy:
#             f.write(sent + "\n\n")

#     bleu = BLEU()
#     res = bleu.corpus_score(pred_greedy, ref)
#     return res


def calculate_bleu_greedy(model: torch.nn.Module, epoch, vocab_transform, text_transform, mt, md, en_test: List[str], vi_test: List[str], cfg: Config):
    print("Calculating BLEU score with greedy translate...")
    model.eval()
    pred_greedy = []
    for sent in tqdm(en_test, dynamic_ncols=True):
        text = translate_greedy(model, sent, mt, vocab_transform, text_transform, cfg)
        text = text.split(' ')
        text = md.detokenize(text)
        pred_greedy.append(text)
    ref = [[md.detokenize(s.split(' ')) for s in vi_test]]

    with open(f"bleu/greedy/{epoch}_translation.txt", "w") as f:
        for sent in pred_greedy:
            f.write(sent + "\n\n")

    bleu = BLEU()
    res = bleu.corpus_score(pred_greedy, ref)
    return res