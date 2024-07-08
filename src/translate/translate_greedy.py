import torch
from translate import greedy_decode, greedy_decode_multiple
from configs import Config
from typing import List


"""
translate function to translate a source language sentence to a target language sentence using greedy decode
    args: (
        model: nn.Module, the main model;
        src_sent: str, the source language sentence;
        mt: MosesTokenizer, to tokenize the sentence to a more human readable sentence for calculating bleu score.
        vocab_transform: function, the function to convert words tokens to integer tokens;
        text_transform: function, the function to transform a normal sentence into a list of integer tokens;
        cfg: Config, the config of the model; 
    )
    Return a string translation
""" 
def translate_greedy(model: torch.nn.Module, src_sent: str, mt, vocab_transform, text_transform, cfg: Config):
    # Certified
    # actual function to translate input sentence into target language
    model.eval()
    
    src_sent = mt.tokenize(src_sent, return_str=True)
    with torch.no_grad():
        src = text_transform[cfg.src_lang](src_sent).view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens, device=cfg.device)).type(torch.bool)
        tgt_tokens = greedy_decode(model=model, src=src, src_mask=src_mask, max_len=int(1.6 * num_tokens), start_symbol=cfg.special_tokens['BOS_IDX'], cfg=cfg).flatten()
        
    return " ".join(vocab_transform[cfg.tgt_lang].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "").replace("<pad>", "").replace("<unk>", "")


def translate_greedy_multiple(model: torch.nn.Module, src_sents: List[str], mt, vocab_transform, text_transform, cfg: Config):
    # Certified
    # actual function to translate input sentence into target language
    model.eval()
    
    src_sents = [mt.tokenize(sent, return_str=True) for sent in src_sents]
    with torch.no_grad():
        src = torch.as_tensor([text_transform[cfg.src_lang](sent) for sent in src_sents])
        num_tokens = max([len(sent) for sent in src])
        src = src.t()
        src_mask = torch.as_tensor([torch.zeros(num_tokens, num_tokens).type(torch.bool)], device=cfg.device).t()
        tgt_tokens_batch = greedy_decode_multiple(model=model, src=src, src_mask=src_mask, max_len=int(1.6 * num_tokens), start_symbol=cfg.special_tokens['BOS_IDX'], cfg=cfg).flatten()
        
    return [" ".join(vocab_transform[cfg.tgt_lang].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "").replace("<pad>", "").replace("<unk>", "") for tgt_tokens in tgt_tokens_batch]