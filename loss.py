import torch.nn as nn

def get_loss(model, x, config):
    loss = nn.MSELoss()
    x = x.to(config.model.device)
    return loss(model(x), x)
