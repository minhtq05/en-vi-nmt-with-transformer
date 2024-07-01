import torch
import torchtext; torchtext.disable_torchtext_deprecation_warning()
import torch.nn as nn
import torch.nn.functional as F
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import tiktoken
import numpy as np
from typing import List, Iterable, Literal, Tuple
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
import html
import re

torch.manual_seed(1337)
random.seed(1337)

warnings.filterwarnings("ignore")

# Hyperparameters
SRC_LANGUAGE = 'en'
TGT_LANGUAGE = 'vi'
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 1024
MAX_LENGTH = 1024
BATCH_SIZE = 16
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
LEARNING_RATE = 5e-5
NUM_EPOCHS = 40
DROPOUT = 0.1
ACTIVATION = 'gelu'
TRAIN_SIZE = 133318
TEST_SIZE = 1268
BEAM_SIZE = 5
PATH = "model_experiment_4.pth"
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX, SEP_IDX = 0, 1, 2, 3, 4
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>', '<sep>']
puncs = ['.', '!', '?', ';']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
re_clean_patterns = [
    (re.compile(r"&amp; lt ;.*?&amp; gt ;"), ""),
    (re.compile(r"&amp amp ;"), "&"),
]

# Test sents during training. Just for validating
demo_sents = [
    "The local bakery unveiled its newest pastry , a lavender - infused croissant , to rave reviews . Customers lined up early in the morning to get a taste of the unique creation . The bakery owner said the inspiration came from a recent trip to Provence , France .",
    "A new study has shown that regular meditation can significantly reduce stress levels . Participants who meditated daily for eight weeks reported feeling calmer and more focused . The researchers believe that meditation could be a valuable tool for mental health .",
    "The city 's public library has launched a new digital lending program . Residents can now borrow e - books and audiobooks directly from the library 's website . The program aims to make reading more accessible to everyone , especially during the pandemic .",
    "An amateur astronomer discovered a new comet that will be visible from Earth next month . The comet , named after the discoverer , will be most visible in the Northern Hemisphere . Experts suggest using binoculars for the best view .",
    "The annual marathon took place under perfect weather conditions , drawing thousands of participants from all over the country . The event included runners of all ages and abilities , from elite athletes to first - time marathoners . Organizers praised the community for their support and enthusiasm .",
    "A local artist has transformed an abandoned warehouse into a vibrant art gallery . The space now features murals , sculptures , and interactive installations . The artist hopes the gallery will become a hub for creativity and community engagement .",
    "Researchers have developed a new biodegradable plastic that could help reduce pollution.  The plastic is made from plant materials and breaks down naturally in the environment . This innovation is seen as a major step forward in the fight against plastic waste .",
    "The latest smartphone app is designed to help users manage their time more effectively . The app includes features like task lists , reminders , and productivity analytics . Users have reported increased efficiency and less stress since using the app .",
    "A historic theater in the downtown area is undergoing major renovations and upgrades. The project aims to restore the theater to its former glory while updating it for modern audiences . The theater is expected to reopen next summer with a lineup of classic and contemporary performances .",
    "A new coffee shop opened in the neighborhood in the summer, offering a cozy atmosphere and locally sourced beans . The shop 's owner , a former barista , wanted to create a space where people could relax and enjoy high - quality coffee. The shop has quickly become a favorite spot for locals and visitors .",
]

console = Console()
# enc = tiktoken.get_encoding('o200k_base')
mt, md = MosesTokenizer(lang='en'), MosesDetokenizer(lang='en')

def tokenizer(sent: str):
    if len(sent) == 0:
        return []
    for pattern, repl in re_clean_patterns:
        sent = re.sub(pattern, repl, sent)
    return sent.split(' ')

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
        yield tokenizer(data_sample[language_index[language]])

def add_special_tokens(text: List[str]):
    res = ['<bos>']
    for token in text:
        res.append(token)
        try:
            if token in puncs:
                res.append('<sep>')
        except:
            raise ValueError("Empty string", text)
    if res[-1] == '<sep>':
        res[-1] = '<eos>'
    else:
        res.append('<eos>')
    return res

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
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=n_embed,
            nhead=n_head,
            num_encoder_layers=n_encoder_layer,
            num_decoder_layers=n_decoder_layer,
            dim_feedforward=dff,
            dropout=0,
            activation=activation,   
        )
        self.generator = nn.Linear(n_embed, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, n_embed)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, n_embed)
        self.positional_encoding = PositionalEncoding(n_embed=n_embed, dropout=dropout)

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
        return self.generator(outs) # (T, B, new_C)

    def encode(self, src, src_mask):
        return self.transformer.encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
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
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

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

def beam_search_decode(model: nn.Module, src, src_mask, max_len, start_symbol, beam_size: int):
    # Certified
    model.eval()
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    
    memory = model.encode(src, src_mask)
    beams = [([[start_symbol]], 0.0)] # (T, beam_size)
    with torch.no_grad():
        for it in range(max_len-1):
            if it <= 3:
                k = 1
            else:
                k = beam_size
            new_beams = []
            for (ys, score) in beams:
                # ys (T), prob (1)
                if ys[-1][0] == EOS_IDX:
                    new_beams.append(tuple((ys, score)))
                    continue
                memory = memory.to(DEVICE)
                ys = torch.as_tensor(ys, device=DEVICE) # (T, B)
                tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(DEVICE)
                out = model.decode(ys, memory, tgt_mask) # (1, C)
                out = out.transpose(0, 1)
                prob = model.generator(out[:, -1]) # (C, 1) -> # (vocab_size, 1)
                prob = F.softmax(prob)
                assert abs(torch.sum(prob) - 1) < 1e-3
                prob, next_word_idxs = torch.topk(prob, k, dim=1) # -> ((beam_size), (beam_size))
                prob, next_word_idxs = prob[0].tolist(), next_word_idxs[0].tolist()
                for p, i in zip(prob, next_word_idxs):
                    new_beams.append((ys.tolist() + [[i]], (score + p)))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:k]
    # return [torch.as_tensor(beams[i][0]) for i in range(beam_size)]
    return torch.as_tensor(beams[0][0])

def translate_greedy(model: nn.Module, src_sentence: str):
    # Certified
    # actual function to translate input sentence into target language
    model.eval()
    
    src_sentence = mt.tokenize(src_sentence, return_str=True)
    with torch.no_grad():
        src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens, device=DEVICE)).type(torch.bool)
        tgt_tokens = greedy_decode(model=model, src=src, src_mask=src_mask, max_len=int(num_tokens * 1.3), start_symbol=BOS_IDX).flatten()
        
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "").replace("<pad>", "").replace("<sep>", "")

def translate_beam_search(model: nn.Module, src_sentence: str):
    # Certified
    model.eval()
    
    src_sentence = mt.tokenize(src_sentence, return_str=True)
    with torch.no_grad():
        src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens, device=DEVICE)).type(torch.bool)
        tgt_tokens = beam_search_decode(model=model, src=src, src_mask=src_mask, max_len=int(num_tokens * 1.3), start_symbol=BOS_IDX, beam_size=BEAM_SIZE).flatten()
        
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "").replace("<pad>", "").replace("<sep>", "")

def train_epoch(model: nn.Module, optimizer: torch.optim.Optimizer):
    # Certified
    model.train()

    losses = 0
    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    total = math.ceil(len(train_data) / BATCH_SIZE)

    for i, (src, tgt) in tqdm(enumerate(train_dataloader), total=total, dynamic_ncols=True):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :] # (T, B)

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask) # (T, B, tgt_vocab_size)

        optimizer.zero_grad(set_to_none=True)

        tgt_out = tgt[1:, :].type(torch.long) # (T, B)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

        if i % 500 == 0:
            wandb.log({"train_loss:": loss.item()})

        if i % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    torch.cuda.empty_cache()
    gc.collect()

    return losses / len(list(train_dataloader))
    
def evaluate_model(model: nn.Module):
# Certified
    model.eval()
    losses = 0

    val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True)
    total = math.ceil(len(val_data) / BATCH_SIZE)

    with torch.no_grad():
        for i, (src, tgt) in tqdm(enumerate(val_dataloader), total=total, dynamic_ncols=True):
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :] # (T, B)

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :].type(torch.long) # (T, B)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

            if i % 20 == 0:
                torch.cuda.empty_cache()
                gc.collect()

    torch.cuda.empty_cache()
    gc.collect()

    return losses / len(list(val_dataloader))
from sacrebleu.metrics import BLEU


def calculate_bleu_greedy(model):
    print("Calculating BLEU score with greedy translate...")
    model.eval()
    pred_greedy = []
    for sent in tqdm(en_test, dynamic_ncols=True):
        text = translate_greedy(model, sent)
        text = text.split(' ')
        text = md.detokenize(text)
        pred_greedy.append(text)
    ref = [[md.detokenize(s.split(' ')) for s in vi_test]]
    bleu = BLEU()
    res = bleu.corpus_score(pred_greedy, ref)
    return res

def calculate_bleu_beam_search(model):
    print("Calculating BLEU score with beam search translate...")
    model.eval()
    pred_beam_search = []
    for sent in tqdm(en_test, dynamic_ncols=True):
        text = translate_beam_search(model, sent)
        text = text.split(' ')
        text = md.detokenize(text)
        pred_beam_search.append(text)
    ref = [[md.detokenize(s.split(' ')) for s in vi_test]]
    bleu = BLEU()
    res = bleu.corpus_score(pred_beam_search, ref)
    return res

# Certified
def train(model: nn.Module, optimizer: torch.optim.Optimizer, num_epochs = NUM_EPOCHS):
    with open("training.log", "a") as log:
        t = datetime.now(tz=timezone('Asia/Ho_Chi_Minh'))
        log.write(f"[{t.month:0>2}/{t.day:0>2}/{t.year} - {t.hour:0>2}:{t.minute:0>2}:{t.second:0>2}] IWSLT - Start new training session!\n\n")

    patient = 10
    best_model_weight = None
    best_bleu_score = 0.0
    print("First eval loss:", evaluate_model(model=model))
    bleu_score_greedy = calculate_bleu_greedy(model)
    bleu_score_beam_search = calculate_bleu_beam_search(model)
    print("First BLEU score:")
    print("bleu_score_greedy:", bleu_score_greedy)
    print("bleu_score_beam_search:", bleu_score_beam_search)
    bleu_score_greedy = float(str(bleu_score_greedy)[6:12])
    bleu_score_beam_search = float(str(bleu_score_beam_search)[6:12])
    print("bleu: ", bleu_score_greedy, bleu_score_beam_search)
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
            bleu_score_greedy = calculate_bleu_greedy(model)
            bleu_score_beam_search = calculate_bleu_beam_search(model)
            print("greedy:", bleu_score_greedy)
            print("beam_search:", bleu_score_beam_search)
            bleu_score_greedy = float(str(bleu_score_greedy)[6:12])
            bleu_score_beam_search = float(str(bleu_score_beam_search)[6:12])
            bleu_score = max(bleu_score_greedy, bleu_score_beam_search)
            print(f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}")
            print(f"BLEU Score of current model: {bleu_score}")

            wandb.log({
                "val_loss:": val_loss,
                "bleu_score_greedy": bleu_score_greedy,
                "bleu_score_beam_search": bleu_score_beam_search,
                "max_bleu_score": bleu_score,
            })

            if bleu_score > best_bleu_score:
                best_bleu_score = bleu_score
                best_model_weight = copy.deepcopy(model.state_dict())
                best_optimizer_weight = copy.deepcopy(optimizer.state_dict())
                patient = 10
                with open('training.log', 'a') as log:
                    log.write("translate:\n")
                    if bleu_score_greedy > bleu_score_beam_search:
                        log.write("greedy!")
                        for i, sent in enumerate(demo_sents):
                            log.write(f"{i}.{translate_greedy(model, sent)}\n")
                        log.write("\n")
                    else: 
                        log.write("beam search!")
                        for i, sent in enumerate(demo_sents):
                            log.write(f"{i}.{translate_beam_search(model, sent)}\n")
                        log.write("\n")
            else:
                patient -= 1
                print("Patient reduced to", patient)
                if patient == 0:
                    print("Early stopping due to increasing BLEU score.")
                    break
            t = datetime.now(tz=timezone('Asia/Ho_Chi_Minh'))
            with open("training.log", "a") as log:
                print("Writing to log...")
                log.write(f"[{t.month:0>2}/{t.day:0>2}/{t.year} - {t.hour:0>2}:{t.minute:0>2}:{t.second:0>2}] Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, BLEU: {bleu_score}\n\n")

            torch.cuda.empty_cache()
            gc.collect()
    except KeyboardInterrupt as e:
        print("Keyboard Interrupted! Now saving model with the best state")

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
# SAVE_STATE = len(sys.argv) > 1 and sys.argv[1] == '--continue'
"""
Load the dataset, and split it into train and test sets
"""
print("Loading datasets...")

with open('./data/en.txt', 'r') as f:
    en = f.readlines()

with open('./data/vi.txt', 'r') as f:
    vi = f.readlines()

with open('./data/train.vi', 'r') as f:
    vi_only = f.readlines()

vi_only = random.sample(vi_only, TRAIN_SIZE)
vi_only = [s.lower() for s in vi_only]


assert(len(en) == len(vi), f"Expected equal number of sentences, got {len(en)} and {len(vi)} instead.")

en = [s.lower() for s in en]
vi = [s.lower() for s in vi]

en_train, en_test, vi_train, vi_test = en[:TRAIN_SIZE], en[TRAIN_SIZE:], vi[:TRAIN_SIZE], vi[TRAIN_SIZE:]

en_train += vi_only
vi_train += vi_only

data = random.sample(zip(en_train, vi_train), TRAIN_SIZE)

e, v = [], []
for e_sent, v_sent in data:
    e.append(e_sent)
    v.append(v_sent)

en_train = e
vi_train = v

print(f"Train size: {len(en_train)}, test size: {len(en_test)}")

train_data = MTDataset(en_train, vi_train, split='train')
val_data = MTDataset(en_test, vi_test, split='test')
"""
Build vocabs for src and tgt languages
"""
vocab_transform = {}
print("Building vocabs...")
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # Training data Iterator
    # Create torchtext's Vocab object
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_data, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    vocab_transform[ln].set_default_index(UNK_IDX)

text_transform = {}
# ``src`` and ``tgt`` language text transforms to convert raw strings into tensors indices
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(tokenizer, # Tokenization
                                               add_special_tokens,
                                               vocab_transform[ln], # Numericalization
                                               tensor_transform)    # Add BOS/EOS and create tensor

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])

print("src vocab size:", SRC_VOCAB_SIZE)
print("tgt vocab size:", TGT_VOCAB_SIZE)

torch.save({
    'vocab_transform': vocab_transform,
    "SRC_VOCAB_SIZE": SRC_VOCAB_SIZE,
    "TGT_VOCAB_SIZE": TGT_VOCAB_SIZE,
}, 'vocab_of_{PATH}.pth')

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

optimizer = torch.optim.AdamW(params=transformer.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)


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