import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from underthesea import word_tokenize
from typing import List, Literal
import math
from tqdm import tqdm
import sys
import os
import gc
import copy
from rich.console import Console
import warnings
import random
from datetime import datetime
import wandb
from pytz import timezone
from sacremoses import MosesTokenizer, MosesDetokenizer
import re

torch.manual_seed(1337)
random.seed(1337)

warnings.filterwarnings("ignore")
# Hyperparameters
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'vi'
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 2048
MAX_LENGTH = 1024
BATCH_SIZE = 16
NUM_ENCODER_LAYERS = 5
NUM_DECODER_LAYERS = 5
LEARNING_RATE = 1e-5
NUM_EPOCHS = 40
DROPOUT = 0.1
ACTIVATION = 'gelu'
TRAIN_SIZE = 133318
TEST_SIZE = 1270
BEAM_SIZE = 3
PATH = "model_experiment.pth"
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
# puncs = ['.', '!', '?', ';']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(DEVICE)
print('Device:', DEVICE)

"""
All special tokens
&amp; amp ; amp ;
&amp; amp ; quot ;
&apos;
&amp; amp ;
&quot;
&amp; lt ;
&amp; gt ;
&#91;
&#93;
"""

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
    
# Test sents during training. Just for validating
demo_sents = [s.lower() for s in [
    "' We all will die '. They said",
    "Vulnerability is not winning or losing; it's having the courage to show up and be seen when we have no control over the outcome. Vulnerability is not weakness; it's our greatest measure of courage.",
    "People don't buy what you do; they buy why you do it. And what you do simply proves what you believe.",
    "Our bodies change our minds, and our minds can change our behavior, and our behavior can change our outcomes. Tiny tweaks can lead to big changes.",
    "We don't grow into creativity, we grow out of it. Or rather, we get educated out of it.",
    "Show a people as one thing, as only one thing, over and over again, and that is what they become. The single story creates stereotypes, and the problem with stereotypes is not that they are untrue, but that they are incomplete.",
    "In the future, there will be no female leaders. There will just be leaders.",
    "We all have a genius inside of us. Creativity is a process, not a destination.",
    "Natural happiness is what we get when we get what we wanted, and synthetic happiness is what we make when we don't get what we wanted. In our society, we have a strong belief that synthetic happiness is of an inferior kind.",
    "We are losing our listening. We spend roughly 60 percent of our communication time listening, but we're not very good at it.",
    "We have the solutions we need to solve this crisis. All we need is the political will, but political will is a renewable resource.",
    "Believe you can and you're halfway there.",
    "The only way to do great work is to love what you do.",
    "Life is what happens when you're busy making other plans.",
    "Keep your face always toward the sunshine, and shadows will fall behind you.",
    "The best way to predict your future is to create it.",
    "You miss 100 percent of the shots you don't take.",
    "Don't watch the clock; do what it does. Keep going.",
    "The best time to plant a tree was 20 years ago. The second best time is now.",
    "It always seems impossible until it's done.",
    "Success is not the key to happiness. Happiness is the key to success.",
    "You are never too old to set another goal or to dream a new dream.",
    "Hardships often prepare ordinary people for an extraordinary destiny.",
    "The only limit to our realization of tomorrow is our doubts of today.",
    "Dream big and dare to fail.",
    "Success is not in what you have, but who you are.",
    "Act as if what you do makes a difference. It does.",
    "What we think, we become.",
    "Happiness is not something ready-made. It comes from your own actions.",
    "Turn your wounds into wisdom.",
    "Your time is limited, so don't waste it living someone else's life."
]]


console = Console()
mt, md = MosesTokenizer(lang='en'), MosesDetokenizer(lang='en')
    
def en_tokenizer(sent: str):
    if len(sent) == 0:
        return []
    return [x for x in sent.split(' ') if x != '']

def vi_tokenizer(sent: str):
    if (len(sent) == 0):
        return []
    return [x for x in word_tokenize(sent) if x != '']
    
token_transform = {
    'en': en_tokenizer,
    'vi': vi_tokenizer,
}
        
# Certified
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

# Certified
def yield_tokens(data: Dataset, language: str):
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data:
        # data_sample: (en_sent, vi_sent)
        yield token_transform[language](data_sample[language_index[language]])

def add_special_tokens(text: List[str]):
    return ['<bos>'] + text + ['<eos>']
    # res = []
    # for token in text:
    #     res.append(token)
    #     try:
    #         if token in puncs:
    #             res.append('<sep>')
    #     except:
    #         raise ValueError("Empty string", text)
    # if res[-1] == '<sep>':
    #     res.pop()
    # res = res[:MAX_LENGTH - 2]
    # res = ['<bos>'] + res + ['<eos>']
    # return res
        
# Certified. Default positional encoding function
class PositionalEncoding(nn.Module):
    def __init__(self, n_embed: int, dropout: int | float):
        max_len = 3000
        super().__init__()
        den = torch.exp(-torch.arange(0, n_embed, 2) * math.log(10000) / n_embed)
        pos = torch.arange(0, max_len).reshape(int(max_len), 1)
        pos_embedding = torch.zeros(max_len, n_embed)
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# Certified
class TokenEmbedding(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 n_embed: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embed, padding_idx=PAD_IDX)
        self.n_embed = n_embed

    def forward(self, tokens): # (T, B, C)
        return self.embedding(tokens.long()) * math.sqrt(self.n_embed) # (T, B, C)

# Certified
class Seq2SeqTransformer(nn.Module):
    def __init__(
            self,
            n_encoder_layer: int,
            n_decoder_layer: int,
            n_embed: int,
            n_head: int,
            src_vocab_size: int,
            tgt_vocab_size: int,
            dff: int,
            dropout: float,
            activation: str,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = nn.Transformer(
            d_model=n_embed,
            nhead=n_head,
            num_encoder_layers=n_encoder_layer,
            num_decoder_layers=n_decoder_layer,
            dim_feedforward=dff,
            dropout=0,
            activation=activation,   
        )
        self._inner_layer = nn.Linear(n_embed, n_embed * 4)
        self._generator = nn.Linear(n_embed * 4, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, n_embed)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, n_embed)
        self.positional_encoding = PositionalEncoding(n_embed=n_embed, dropout=dropout)

    def generator(self, x):
        return self._generator(self._inner_layer(x))
            
    def forward(
            self,
            src,
            tgt,
            src_mask,
            tgt_mask,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
    ):
        src_emb = self.positional_encoding(self.src_tok_emb(src)) # (T, B, C)
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt)) # (T, B, C)
        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            memory_key_padding_mask,
        )
        return self.generator(outs)

    def encode(self, src, src_mask):
        # Listen
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        # Say
        return self.transformer.decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

# Certified
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# Certified
def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len) # We are predicting the future so past tokens cannot communicate with future tokens
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool) # We already know every tokens about the src => no need to mask the future

    src_padding_mask = (src == PAD_IDX).transpose(0, 1) # boolean tensor (B, T, C) -> (T, B, C)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1) # boolean tensor (B, T, C) -> (T, B, C)

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

# Certified
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func
        
# Certified
def tensor_transform(token_ids: List[int]):
    return torch.as_tensor(token_ids)

def collate_fn(batch):
    # Certified
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch
            
def greedy_decode(model: nn.Module, src, src_mask, max_len, start_symbol, **kwargs):
    # Certified
    # function to generate output sequence using greedy algorithm
    model.eval()
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE) # ys is gonna have shape (T, B) with B = 1 when generating
    with torch.no_grad():
        for i in range(max_len-1):
            memory = memory.to(DEVICE)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                        .type(torch.bool)).to(DEVICE)
            out = model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1) # (T, B) -> (B, T)
            prob = model.generator(out[:, -1])
            _, next_word_idx = torch.max(prob, dim=1)
            next_word_idx = next_word_idx.item()

            ys = torch.cat([ys,
                            torch.ones(1, 1).type_as(src.data).fill_(next_word_idx)], dim=0)
            if next_word_idx == EOS_IDX:
                break
    return ys
                
# def beam_search_decode(model: nn.Module, src, src_mask, max_len, start_symbol, beam_size: int):
#     # Certified
#     alpha = 0.75
#     model.eval()
#     src = src.to(DEVICE)
#     src_mask = src_mask.to(DEVICE)
    
#     memory = model.encode(src, src_mask)
#     beams = [([[start_symbol]], 0.0)] # (T, beam_size)
#     with torch.no_grad():
#         for it in range(max_len-1):
#             if it <= 3:
#                 k = 1
#             else:
#                 k = beam_size
#             new_beams = []
#             for ys, score in beams:
#                 # ys (T), prob (1)
#                 if ys[-1][0] == EOS_IDX:
#                     new_beams.append((ys, score))
#                     continue
#                 memory = memory.to(DEVICE)
#                 ys = torch.as_tensor(ys, device=DEVICE) # (T, B)
#                 tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
#                 out = model.decode(ys, memory, tgt_mask) # (1, C)
#                 out = out.transpose(0, 1)
#                 prob = torch.log_softmax(model.generator(out[:, -1]), dim=-1) # (C, 1) -> # (vocab_size, 1)
#                 # assert abs(torch.sum(prob) - 1) < 1e-3
#                 prob, next_word_idxs = torch.topk(prob, k, dim=1) # -> ((beam_size), (beam_size))
#                 prob, next_word_idxs = prob[0].tolist(), next_word_idxs[0].tolist()
#                 for p, i in zip(prob, next_word_idxs):
#                     new_beams.append((ys.tolist() + [[i]], (score / max(1, it ** alpha) - p) * (it + 1) ** alpha))
#             beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:k]
#     # return [torch.as_tensor(beams[i][0]) for i in range(beam_size)]
#     return torch.as_tensor(beams[0][0])
        
def translate_greedy(model: nn.Module, src_sentence: str):
    # Certified
    # actual function to translate input sentence into target language
    model.eval()
    
    src_sentence = mt.tokenize(src_sentence, return_str=True)
    with torch.no_grad():
        src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens, device=DEVICE)).type(torch.bool)
        tgt_tokens = greedy_decode(model=model, src=src, src_mask=src_mask, max_len=int(1.6 * num_tokens), start_symbol=BOS_IDX).flatten()
        
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "").replace("<pad>", "").replace("<unk>", "")

# def translate_beam_search(model: nn.Module, src_sentence: str):
#     # Certified
#     model.eval()
    
#     src_sentence = mt.tokenize(src_sentence, return_str=True)
#     with torch.no_grad():
#         src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
#         num_tokens = src.shape[0]
#         src_mask = (torch.zeros(num_tokens, num_tokens, device=DEVICE)).type(torch.bool)
#         tgt_tokens = beam_search_decode(model=model, src=src, src_mask=src_mask, max_len=int(1.6 * num_tokens), start_symbol=BOS_IDX, beam_size=BEAM_SIZE).flatten()
        
#     return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "").replace("<pad>", "").replace("<unk>", "")

def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer):
    # Certified
    model.train()

    scaler = torch.cuda.amp.GradScaler()

    losses = 0
    train_dataloader = DataLoader(
        train_data, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=True,
        generator=torch.Generator(device='cuda')
    )
    total = math.ceil(len(train_data) / BATCH_SIZE)

    for i, (src, tgt) in tqdm(enumerate(train_dataloader), total=total, dynamic_ncols=True):
        with torch.autocast(device_type=str(DEVICE), dtype=torch.float16):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)
            
            tgt_input = tgt[:-1, :] # (T, B)

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask) # (T, B, tgt_vocab_size)


            tgt_out = tgt[1:, :].type(torch.long) # (T, B)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))

        scaler.scale(loss).backward()
        # loss.backward()
        scaler.step(optimizer)
        # optimizer.step()
        losses += loss.item()

        scaler.update()

        optimizer.zero_grad(set_to_none=True)

        if i % 500 == 0:
            wandb.log({"train_loss:": loss.item()})

        # if i % 50 == 0:
            # torch.cuda.empty_cache()
        #     gc.collect()

    torch.cuda.empty_cache()
    gc.collect()

    return losses / len(list(train_dataloader))

def evaluate_model(model: nn.Module):
# Certified
    model.eval()
    losses = 0

    val_dataloader = DataLoader(
        val_data, 
        batch_size=BATCH_SIZE, 
        collate_fn=collate_fn, 
        shuffle=True,
        generator=torch.Generator(device='cuda')
    )
    total = math.ceil(len(val_data) / BATCH_SIZE)

    with torch.no_grad():
        for i, (src, tgt) in tqdm(enumerate(val_dataloader), total=total, dynamic_ncols=True):
            with torch.autocast(device_type=str(DEVICE), dtype=torch.float16):
                src = src.to(DEVICE)
                tgt = tgt.to(DEVICE)

                tgt_input = tgt[:-1, :] # (T, B)

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

                logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

                tgt_out = tgt[1:, :].type(torch.long) # (T, B)
                loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
                losses += loss.item()

            # if i % 50 == 0:
            #     torch.cuda.empty_cache()
            #     gc.collect()

    torch.cuda.empty_cache()
    gc.collect()

    return losses / len(list(val_dataloader))
                

from sacrebleu.metrics import BLEU

def calculate_bleu_greedy(model, epoch):
    print("Calculating BLEU score with greedy translate...")
    model.eval()
    pred_greedy = []
    for sent in tqdm(en_test, dynamic_ncols=True):
        text = translate_greedy(model, sent)
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

# def calculate_bleu_beam_search(model, epoch):
#     print("Calculating BLEU score with beam search translate...")
#     model.eval()
#     pred_beam_search = []
#     for sent in tqdm(en_test, dynamic_ncols=True):
#         text = translate_beam_search(model, sent)
#         text = text.split(' ')
#         text = md.detokenize(text)
#         pred_beam_search.append(text)

#     with open(f"bleu/beam_search/{epoch}_translation.txt", "w") as f:
#         for sent in pred_beam_search:
#             f.write(sent + "\n\n")

#     ref = [[md.detokenize(s.split(' ')) for s in vi_test]]
#     bleu = BLEU()
#     res = bleu.corpus_score(pred_beam_search, ref)
#     return res

# Certified
def train(model: nn.Module, optimizer: torch.optim.Optimizer, num_epochs = NUM_EPOCHS):
    with open("training.log", "a") as log:
        t = datetime.now(tz=timezone('Asia/Ho_Chi_Minh'))
        log.write(f"[{t.month:0>2}/{t.day:0>2}/{t.year} - {t.hour:0>2}:{t.minute:0>2}:{t.second:0>2}] IWSLT - Start new training session!\n\n")

    with open("bleu/en_test.txt", "w") as f:
        for e in en_test:
            f.write(md.detokenize(e.split(' ')) + "\n\n")

    with open("bleu/vi_test.txt", "w") as f:
        for v in vi_test:
            f.write(md.detokenize(v.split(' ')) + "\n\n")

    patient = 10
    best_model_weight = None
    best_bleu_score = 0.0
    print("First eval loss:", evaluate_model(model=model))
    bleu_score_greedy = calculate_bleu_greedy(model, "init")
    # bleu_score_beam_search = calculate_bleu_beam_search(model, "init")
    print("First BLEU score:")
    print("bleu_score_greedy:", bleu_score_greedy)
    # print("bleu_score_beam_search:", bleu_score_beam_search)
    bleu_score_greedy = float(str(bleu_score_greedy)[6:12])
    # bleu_score_beam_search = float(str(bleu_score_beam_search)[6:12])
    # print("bleu: ", bleu_score_greedy, bleu_score_beam_search)
    print("First")

    """
    Start training
    """
    wandb.init(
        project='machine-translation-v4',
        config={
            "learning_rate": LEARNING_RATE,
            "architecture": "Transformer",
            "dataset": "IWSLT",
            "epochs": NUM_EPOCHS,
        }
    )

    try:
        for epoch in range(1, num_epochs+1):
            try:
                train_loss = train_epoch(model=model, optimizer=optimizer)
            except KeyboardInterrupt:
                print('Interrupted')
                try:
                    sys.exit(130)
                except SystemExit:
                    os._exit(130)

            val_loss = evaluate_model(model)
            bleu_score_greedy = calculate_bleu_greedy(model, epoch)
            # bleu_score_beam_search = calculate_bleu_beam_search(model, epoch)
            print("greedy:", bleu_score_greedy)
            # print("beam_search:", bleu_score_beam_search)
            bleu_score_greedy = float(str(bleu_score_greedy)[6:12])
            # bleu_score_beam_search = float(str(bleu_score_beam_search)[6:12])
            # bleu_score = max(bleu_score_greedy, bleu_score_beam_search)
            bleu_score = bleu_score_greedy
            print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
            print(f"BLEU Score of current model: {bleu_score}")

            wandb.log({
                "val_loss:": val_loss,
                "bleu_score_greedy": bleu_score_greedy,
                # "bleu_score_beam_search": bleu_score_beam_search,
                "max_bleu_score": bleu_score,
            })

            if bleu_score > best_bleu_score:
                best_bleu_score = bleu_score
                best_model_weight = copy.deepcopy(model.state_dict())
                best_optimizer_weight = copy.deepcopy(optimizer.state_dict())
                patient = 10
                with open('training.log', 'a') as log:
                    log.write("translate:\n")
                    # if bleu_score_greedy > bleu_score_beam_search:
                    #     log.write("greedy!\n")
                    #     for i, sent in enumerate(demo_sents):
                    #         log.write(f"{i}.{translate_greedy(model, sent)}\n")
                    #     log.write("\n")
                    # else: 
                    #     log.write("beam search!\n")
                    #     for i, sent in enumerate(demo_sents):
                    #         log.write(f"{i}.{translate_beam_search(model, sent)}\n")
                        # log.write("\n")
                    log.write("greedy!\n")
                    for i, sent in enumerate(demo_sents):
                        log.write(f"{i}.{translate_greedy(model, sent)}\n")
                    log.write("\n")
            else:
                patient -= 1
                print("Patient reduced to", patient)
                if patient == 0:
                    print("Early stopping due to increasing BLEU score.")
                    break
            t = datetime.now()
            with open("training.log", "a") as log:
                print("Writing to log...")
                log.write(f"[{t.month:0>2}/{t.day:0>2}/{t.year} - {t.hour:0>2}:{t.minute:0>2}:{t.second:0>2}] Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, BLEU: {bleu_score}\n\n")

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
    }, PATH)
    print("Model completed! Saved at", PATH)
    torch.cuda.empty_cache()
    gc.collect()

def read_dataset():
    print("Loading datasets...")

    with open('./data/en.txt', 'r') as f:
        en = f.readlines()

    with open('./data/vi.txt', 'r') as f:
        vi = f.readlines()

    # with open('./data/train.vi', 'r') as f:
    #     vi_only = f.readlines()

    # vi_only = random.sample(vi_only, TRAIN_SIZE)
    # vi_only = [s.lower() for s in vi_only]


    # assert(len(en) == len(vi), f"Expected equal number of sentences, got {len(en)} and {len(vi)} instead.")

    en = [s.lower() for s in en]
    vi = [s.lower() for s in vi]

    en_train, en_test, vi_train, vi_test = en[:TRAIN_SIZE], en[TRAIN_SIZE:], vi[:TRAIN_SIZE], vi[TRAIN_SIZE:]

    return en_train, en_test, vi_train, vi_test 


__process = 'train'
# __process = 'test'

if __process == 'train':
    # print("Start training...")
    # SAVE_STATE = len(sys.argv) > 1 and sys.argv[1] == '--continue'
    """
    Load the dataset, and split it into train and test sets
    """

    en_train, en_test, vi_train, vi_test = read_dataset()

    def clean(sent):
        sent = sent.rstrip("\n")
        for pattern, repl in re_clean_patterns:
            sent = re.sub(pattern, repl, sent)
        return sent
    
    en_train = [clean(sent) for sent in en_train]
    vi_train = [clean(sent) for sent in vi_train]
    en_test = [clean(sent) for sent in en_test]
    vi_test = [clean(sent) for sent in vi_test]

    # en_train += vi_only
    # vi_train += vi_only

    print(f"Train size: {len(en_train)}, test size: {len(en_test)}")

    train_data = MTDataset(en_train, vi_train, split='train')
    val_data = MTDataset(en_test, vi_test, split='test')
    """
    Build vocabs for src and tgt languages
    """
    if os.path.exists(f'vocab_of_{PATH}.pth'):
        vocab = torch.load(f'vocab_of_{PATH}.pth')
        vocab_transform = vocab['vocab_transform']
        SRC_VOCAB_SIZE = vocab['SRC_VOCAB_SIZE']
        TGT_VOCAB_SIZE = vocab['TGT_VOCAB_SIZE']
    else:
        vocab_transform = {}
        print("Building vocabs...")
        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            # Training data Iterator
            # Create torchtext's Vocab object
            vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_data, ln),
                                                            min_freq=2,
                                                            specials=special_symbols,
                                                            special_first=True,
                                                            max_tokens=30000)

        for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
            vocab_transform[ln].set_default_index(UNK_IDX)

        SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
        TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

        torch.save({
            'vocab_transform': vocab_transform,
            "SRC_VOCAB_SIZE": SRC_VOCAB_SIZE,
            "TGT_VOCAB_SIZE": TGT_VOCAB_SIZE,
        }, f'vocab_of_{PATH}.pth')



    text_transform = {}
    # ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequential_transforms(token_transform[ln], # Tokenization
                                                add_special_tokens,
                                                vocab_transform[ln], # Numericalization
                                                tensor_transform)    # Add BOS/EOS and create tensor

    print("src vocab size:", SRC_VOCAB_SIZE)
    print("tgt vocab size:", TGT_VOCAB_SIZE)
    # print("Sample tokenization")
    # for i in range(20):
    #     e_sent = random.choice(en_train)
    #     print(f"en: {e_sent} --> {en_tokenizer(e_sent)}")
    #     x = input()

    # for i in range(20):
    #     v_sent = random.choice(vi_train)
    #     print(f"vi: {v_sent} --> {vi_tokenizer(v_sent)}")
    #     x = input()

    # hyperparams = {
    #     "NUM_EPOCHS": NUM_EPOCHS,
    #     "SRC_LANGUAGE": SRC_LANGUAGE,
    #     "TGT_LANGUAGE": TGT_LANGUAGE,
    #     "SRC_VOCAB_SIZE": SRC_VOCAB_SIZE,
    #     "TGT_VOCAB_SIZE": TGT_VOCAB_SIZE,
    #     "vocab_transform": vocab_transform,
    #     "EMB_SIZE": EMB_SIZE,
    #     "NHEAD": NHEAD,
    #     "FFN_HID_DIM": FFN_HID_DIM,
    #     "BATCH_SIZE": BATCH_SIZE,
    #     "NUM_ENCODER_LAYERS": NUM_ENCODER_LAYERS,
    #     "NUM_DECODER_LAYERS": NUM_DECODER_LAYERS,
    # }

    """
    Create the main transformer model.
    """
    print("Creating model...")

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, DROPOUT, ACTIVATION)

    optimizer = torch.optim.AdamW(params=transformer.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)

    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX).cuda()


    # if SAVE_STATE:
    #     try:
    #         checkpoint = torch.load(PATH, map_location=DEVICE)
    #         print("Checkpoint loaded successfully!")
    #         transformer.load_state_dict(checkpoint["model_state_dict"])
    #         transformer = transformer.to(DEVICE)
    #         transformer.train()
    #         print("Loading transformer state dict successfully!")
    #         optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    #         print("Loading optimizer state dict successfully!")
    #     except FileNotFoundError:
    #         print("No checkpoint found. Starting training from scratch.")
    #     except KeyError as e:
    #         print(f"Missing key in checkpoint: {e}")
    #     except Exception as e:
    #         print(f"An error occurred while loading the checkpoint: {e}")
    # else:

    transformer = transformer.to(DEVICE)
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            

    total_params = sum(p.numel() for p in transformer.parameters())
    total_trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f} M, Trainable parameters: {total_trainable_params / 1e6:.2f} M")


    print("Start training...")
    train(model=transformer, optimizer=optimizer, num_epochs=NUM_EPOCHS)
    print("Final loss:", evaluate_model(transformer))
    wandb.finish()



    """

    The code below are only for testing
    Your code for training model should be above this text

    """

if __process == 'test':
    from rich import print

    def calc_bleu(en, vi):
        vi = [md.detokenize(vi)]
        ref = [[md.detokenize(vi)]]
        print(md.detokenize(en))
        print(vi, ref)
        bleu = BLEU()
        res = bleu.corpus_score(vi, ref)
        return res
    
    def en_tokenizer(sent: str):
        if len(sent) == 0:
            return []
        sent = sent.rstrip("\n")
        for pattern, repl in re_clean_patterns:
            sent = re.sub(pattern, repl, sent)
        return [x for x in sent.split(' ') if x != '']

    def vi_tokenizer(sent: str):
        if (len(sent) == 0):
            return []
        sent = sent.rstrip("\n")
        for pattern, repl in re_clean_patterns:
            sent = re.sub(pattern, repl, sent)
        return [x for x in sent.split(' ') if x != '']

    def clean(text):
        amp = -1
        comma = -1
        wierd_tokens = []
        for i, c in enumerate(text):
            if (c == '&'):
                if (amp == -1):
                    amp = i
                elif (comma != -1):
                    wierd_tokens.append(text[amp:comma + 1])
                    amp = -1
                    comma = -1
            elif (c == ';'):
                comma = i
        if (amp != -1 and comma == -1):
            pass
            # raise ValueError(f"amp not closed with data point : \"{text}\"")
        else:
            wierd_tokens.append(text[amp:comma + 1])
        return wierd_tokens
    
    # en_train, en_test, vi_train, vi_test = read_dataset()        
    # wierd_tokens = set()

    # for sent in en_train:
    #     sent = ' '.join(en_tokenizer(sent))
    #     w = clean(sent)
        # wierd_tokens.update(w)

    # for sent in vi_test:
    #     sent = vi_tokenizer(sent)
    #     w = clean(sent)
    #     if len(w) > 0:
    #         wierd_tokens.update(w)

    # print(len(wierd_tokens))
    # print(wierd_tokens)
    # matches = re.findall(pattern, "Như vậy , tỷ lệ chết trẻ em &amp; lt ; 5t đã giảm 2 lần .")
    # print(matches)

    en = en_tokenizer("I &apos;ll give you one last illustration of variability , and that is -- oh , I &apos;m sorry .")
    vi = vi_tokenizer("Tôi sẽ nêu ra cho các bạn một ví dụ cuối cùng về sự đa dạng , đó là -- à , tôi xin lỗi .")

    print(calc_bleu(en, vi))

