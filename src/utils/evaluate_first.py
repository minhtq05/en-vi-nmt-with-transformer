import torch
from torch.utils.data import Dataset
from configs import Config
from trainer import evaluate_model
from bleu import calculate_bleu_greedy


"""
Evaluation functions for calculating bleu score before training
    args: (
        model: nn.Module, the main model;
        verbose: whether to print the logs or not;
    )
"""
def evaluate_first(model, verbose, loss_fn: torch.nn.CrossEntropyLoss, val_data: Dataset, vocab_transform, text_transform, mt, md, en_test, vi_test, cfg: Config):
    print("First eval loss:", evaluate_model(model=model, loss_fn=loss_fn, val_data=val_data, text_transform=text_transform, cfg=cfg))
    bleu_score_greedy = calculate_bleu_greedy(model, "init", vocab_transform=vocab_transform, text_transform=text_transform, mt=mt, md=md, en_test=en_test, vi_test=vi_test, cfg=cfg)
    verbose and print(f"First BLEU score (greedy): {bleu_score_greedy}")
    bleu_score_greedy = float(str(bleu_score_greedy)[6:12])
    return bleu_score_greedy