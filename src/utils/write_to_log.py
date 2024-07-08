from typing import Literal
from datetime import datetime
from pytz import timezone
from translate import translate_greedy


"""
Log writer
    args: (status: Literal['first', 'demo', 'training'], status of the current log process)
"""
def write_to_log(status: Literal['first', 'demo', 'training'], **params):
    if status == 'first':
        with open("training.log", "a") as log:
            t = datetime.now(tz=timezone('Asia/Ho_Chi_Minh'))
            log.write(f"[{t.month:0>2}/{t.day:0>2}/{t.year} - {t.hour:0>2}:{t.minute:0>2}:{t.second:0>2}] IWSLT - Start new training session!\n")
    elif status == 'demo':
        from constants import demo_sents
        with open('training.log', 'a') as log:
            log.write("translation:\n")
            for i, sent in enumerate(demo_sents):
                log.write(f"{i}.{translate_greedy(params['model'], sent, params['mt'], params['vocab_transform'], params['text_transform'], params['cfg'])}\n")
            log.write("\n")
    elif status == 'training':
        t = datetime.now()
        with open("training.log", "a") as log:
            print("Writing to log...")
            log.write(f"[{t.month:0>2}/{t.day:0>2}/{t.year} - {t.hour:0>2}:{t.minute:0>2}:{t.second:0>2}] Epoch: {params['epoch']}, Train loss: {params['train_loss']:.3f}, Val loss: {params['val_loss']:.3f}, BLEU: {params['bleu_score']}\n\n")