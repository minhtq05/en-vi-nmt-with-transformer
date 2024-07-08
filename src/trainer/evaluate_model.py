import torch
from torch.utils.data import Dataset, DataLoader
from utils import collate_fn, create_mask
from configs import Config
import math
from tqdm import tqdm
import gc


def evaluate_model(model: torch.nn.Module, loss_fn: torch.nn.CrossEntropyLoss, val_data: Dataset, text_transform, cfg: Config):
# Certified
    model.eval()
    losses = 0

    val_dataloader = DataLoader(
        val_data, 
        batch_size=cfg.batch_size, 
        collate_fn=lambda data: collate_fn(data, text_transform=text_transform, cfg=cfg), 
        shuffle=True,
        generator=torch.Generator(device='cuda')
    )
    total = math.ceil(len(val_data) / cfg.batch_size)

    with torch.no_grad():
        for i, (src, tgt) in tqdm(enumerate(val_dataloader), total=total, dynamic_ncols=True):
            with torch.autocast(device_type=str(cfg.device), dtype=torch.float16):
                src = src.to(cfg.device)
                tgt = tgt.to(cfg.device)

                tgt_input = tgt[:-1, :] # (T, B)

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input,cfg)

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