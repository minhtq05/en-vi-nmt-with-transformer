import torch


"""
Print the total number of trainable params of the model
    args: model: nn.Module, the model to get the result from
    Return an integer as the number of params
"""
def total_params(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f} M, Trainable parameters: {total_trainable_params / 1e6:.2f} M")