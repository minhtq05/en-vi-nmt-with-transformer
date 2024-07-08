from bleu import calculate_bleu_greedy


"""
Evaluation functions for calculating bleu score during training
    args: (
        model: nn.Module, the main model;
        verbose: whether to print the logs or not;
    )
"""
def evaluate_during_training(model, epoch, train_loss, val_loss, vocab_transform, text_transform, mt, md, en_test, vi_test, cfg, verbose: bool):
    bleu_score_greedy = calculate_bleu_greedy(model, epoch, vocab_transform, text_transform, mt, md, en_test, vi_test, cfg)
    verbose and print("greedy:", bleu_score_greedy)
    bleu_score_greedy = float(str(bleu_score_greedy)[6:12])
    verbose and print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
    verbose and print(f"BLEU Score of current model (greedy): {bleu_score_greedy}")
    return bleu_score_greedy