import numpy as np
import torch
import torch.nn.functional as F
from utils.model import *
from utils.data import *

def infer(model, input, device=None):
    """
        Inference given the model and the input 
        returns:
            - outputs_p list[7]: output probability of all the classes
            - pred_p (float): output probability of the max class
            - pred (int): label of the max class
    """
    model.eval()
    if device:
        input = input.to(device)

    with torch.no_grad():
        outputs = model(input)
        outputs_p = F.softmax(outputs, dim=-1)
        pred_p, pred = torch.max(outputs_p, dim=1)

        return outputs_p, pred_p, pred


def _ensemble_voting(models, loader, device):
    pass


def _ensemble_average(models, loader, device):
    preds = []
    with torch.no_grad():
        for input, _ in loader:
            preds_p = torch.zeros(input.shape[0], 7).to(device)
            for model in models:
                preds_p += infer(model, input, device=device)[0]
            preds.extend(torch.argmax(preds_p, dim=1).tolist())
            
            del preds_p
            torch.cuda.empty_cache()

    return preds
            

def ensemble(method, models, loader, device):
    for model in models:
        model.eval()

    if method == 0:
        return _ensemble_average(models, loader, device=device)
    else:
        return _ensemble_voting(models, loader, device=device)
