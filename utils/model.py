import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save(model, optim, path):
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict()
    }, path)


def load(path, model, optim=None):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    if optim:
        optim.load_state_dict(checkpoint['optim'])