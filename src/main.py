from configs import Config
import torch
import argparse
from rich import print
from utils import read_dataset, preprocess, create_dataset, total_params, write_to_log, write_reference_to_bleu_folder, evaluate_first, evaluate_during_training
from vocab import build_vocab
from model import Seq2SeqTransformer
from configs import Config
from trainer import train_epoch, evaluate_model
import wandb
import copy
import gc
import sys
import os
from sacremoses import MosesTokenizer, MosesDetokenizer
import warnings
warnings.filterwarnings("ignore")

"""
Arguments Parser

Use the train.sh scrip to run the main trainer instead of calling main.py
Easy to train models
"""
parser = argparse.ArgumentParser(description='PyTorch "English -> Vietnamese" Neural Machine Translation Transformer Model')
parser.add_argument('--data', type=str, choices=['./data/IWSLT-2015/', './data/PhoMT/'], help='dataset location') 
parser.add_argument('--src_lang', type=str, default='en', help='source language')
parser.add_argument('--tgt_lang', type=str, default='vi', help='target language')
parser.add_argument('--emb_size', type=int, help='embedding dimension')
parser.add_argument('--n_head', type=int, help='number of heads')
parser.add_argument('--ffn_hid_dim', type=int, help='feed forward hidden dimension')
parser.add_argument('--max_length', type=int, help='maximum length of the tokens list')
parser.add_argument('--batch_size', type=int, help='size of each training / validating batch')
parser.add_argument('--n_encoder', type=int, help='number of encoder layers')
parser.add_argument('--n_decoder', type=int, help='number of decoder layers')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--n_epoch', type=int, help='number of epochs')
parser.add_argument('--model_dropout', type=float, help='transformer model dropout')
parser.add_argument('--pos_enc_dropout', type=float, help='positional encoding dropout')
parser.add_argument('--activation', type=str, help='activation function')
# parser.add_argument('--beam_size', type=int, help='beam search total beam size')
parser.add_argument('--seed', type=int, help='randomization seed')
parser.add_argument('--save_model_path', type=str, help='location to save the model')
parser.add_argument('--save_vocab_path', type=str, help='location to save the model vocab')
parser.add_argument('--verbose', action='store_true', default=True)


"""
Local run-time configurations
"""
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(DEVICE)
cfg = Config(**vars(parser.parse_args()), device=DEVICE)
cfg.show_config()
cfg.verbose and print(f'Training on device: {DEVICE}')
mt, md = MosesTokenizer(lang='en'), MosesDetokenizer(lang='en')

if (cfg.src_lang != 'en' and cfg.tgt_lang != 'vi'):
    raise ValueError('These source and target languages is currently not supported for this trainer.')


"""
Main trainer of the program
    args: model: nn.Module
"""
def train(model: torch.nn.Module, optimizer: torch.optim.Optimizer, cfg: Config):
    write_to_log('first')
    write_reference_to_bleu_folder(en_test, vi_test, md)

    patient = 10
    best_model_weight = None
    best_bleu_score = evaluate_first(model, verbose=cfg.verbose, loss_fn=loss_fn, val_data=val_data, vocab_transform=vocab_transform, text_transform=text_transform, mt=mt, md=md, en_test=en_test, vi_test=vi_test, cfg=cfg)

    """
    Start training
    """
    wandb.init(
        project='machine-translation-v4',
        config={
            "learning_rate": cfg.lr,
            "architecture": "Transformer",
            "dataset": "IWSLT",
            "epochs": cfg.n_epoch,
        }
    )

    try:
        for epoch in range(1, cfg.n_epoch+1):
            try:
                train_loss = train_epoch(model=model, optimizer=optimizer, loss_fn=loss_fn, train_data=train_data, text_transform=text_transform, cfg=cfg)
            except KeyboardInterrupt:
                print('Canceled by user.')
                try:
                    sys.exit(130)
                except SystemExit:
                    os._exit(130)

            val_loss = evaluate_model(model, loss_fn=loss_fn, val_data=val_data, text_transform=text_transform, cfg=cfg)
            bleu_score = evaluate_during_training(model, epoch, train_loss, val_loss, vocab_transform, text_transform, mt, md, en_test, vi_test, cfg, cfg.verbose)

            wandb.log({
                "val_loss:": val_loss,
                "max_bleu_score (greedy)": bleu_score,
            })

            if bleu_score > best_bleu_score:
                best_bleu_score = bleu_score
                best_model_weight = copy.deepcopy(model.state_dict())
                best_optimizer_weight = copy.deepcopy(optimizer.state_dict())
                patient = 10
                write_to_log('demo', model=model, mt=mt, vocab_transform=vocab_transform, text_transform=text_transform, cfg=cfg)
            else:
                patient -= 1
                print("Patient reduced to", patient)
                if patient == 0:
                    print("Early stopping due to increasing BLEU score.")
                    break
            write_to_log('training', epoch=epoch, train_loss=train_loss, val_loss=val_loss, bleu_score=bleu_score)

            torch.cuda.empty_cache()
            gc.collect()
    except KeyboardInterrupt:
        print("Canceled by user.")

    model.load_state_dict(best_model_weight)
    optimizer.load_state_dict(best_optimizer_weight)

    print("Saving model...")
    # save_checkpoint(transformer, hyperparameters=hyperparameters, path=PATH)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, cfg.save_model_path)
    print("Model completed! Saved at", cfg.save_model_path)
    torch.cuda.empty_cache()
    gc.collect()




en_train, en_test, vi_train, vi_test = preprocess(*read_dataset(cfg.data, cfg))

assert(len(en_train) == len(vi_train) and len(en_test) == len(vi_test))

train_data, val_data = create_dataset(en_train, en_test, vi_train, vi_test)
vocab_transform, text_transform, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE = build_vocab(train_data, cfg)
cfg.verbose and print('Creating model...')
transformer = Seq2SeqTransformer(SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, cfg)
optimizer = torch.optim.AdamW(params=transformer.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=cfg.special_tokens['PAD_IDX']).cuda()


transformer = transformer.to(DEVICE)
for p in transformer.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)


cfg.verbose and total_params(transformer)
cfg.verbose and print("Start training...")

train(model=transformer, optimizer=optimizer, cfg=cfg)
cfg.verbose and print("Final loss:", evaluate_model(transformer, loss_fn=loss_fn, val_data=val_data,text_transform=text_transform, cfg=cfg))
wandb.finish()

