python3 main.py \
    --data              './data/IWSLT-2015/' \
    --src_lang          'en' \
    --tgt_lang          'vi' \
    --emb_size          512 \
    --n_head            8 \
    --ffn_hid_dim       2048 \
    --max_length        2048 \
    --batch_size        16 \
    --n_encoder         5 \
    --n_decoder         5 \
    --lr                2e-5 \
    --n_epoch           50 \
    --model_dropout     0.3 \
    --pos_enc_dropout   0.1 \
    --activation        'gelu' \
    --seed              1337 \
    --save_model_path   'models/nmt_en_vi.pth' \
    --save_vocab_path   'models/nmt_en_vi_vocab.pth' \
    --verbose \