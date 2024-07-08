import torch
from utils import TokenEmbedding, PositionalEncoding
from configs import Config


"""
Seq2Seq Transformer model structure
    args: (
        src_vocab_size: int, the source language vocab size;
        tgt_vocab_size: int, the target language vocab size;
        cfg: Config, the config of the model
    )
    Return the model for training and testing
"""
class Seq2SeqTransformer(torch.nn.Module):
    # def __init__(
    #     self,
    #     n_encoder_layer: int,
    #     n_decoder_layer: int,
    #     n_embed: int,
    #     n_head: int,
    #     src_vocab_size: int,
    #     tgt_vocab_size: int,
    #     dff: int,
    #     dropout: float,
    #     activation: str,
    # ):
    def __init__(self, src_vocab_size, tgt_vocab_size, cfg: Config):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = torch.nn.Transformer(
            d_model=cfg.emb_size,
            nhead=cfg.n_head,
            num_encoder_layers=cfg.n_encoder,
            num_decoder_layers=cfg.n_decoder,
            dim_feedforward=cfg.ffn_hid_dim,
            dropout=cfg.model_dropout,
            activation=cfg.activation,   
        )
        self._inner_layer = torch.nn.Linear(cfg.emb_size, cfg.emb_size * 4)
        self._generator = torch.nn.Linear(cfg.emb_size * 4, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, cfg.emb_size, cfg)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, cfg.emb_size, cfg)
        self.positional_encoding = PositionalEncoding(cfg)

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