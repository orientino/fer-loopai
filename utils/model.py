"""
This module contains utilities to save, load and test the models.
"""

import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix, ConfusionMatrixDisplay


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save(model, optim, path):
    torch.save({
        'model': model.state_dict(),
        'optim': optim.state_dict()
    }, path)


def load(path, model, optim=None, device=None):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    if optim:
        optim.load_state_dict(checkpoint['optim'])


def test(model):
    x = torch.rand([1,1, 48, 48])
    y = model(x)
    return y


def evaluate(model, data_loader, criterion, class_names, device):
    running_loss = 0
    correct, total = 0, 0
    model.eval()

    y, y_out = [], []
    with torch.no_grad():
        for input, label in (data_loader):
            input, label = input.to(device), label.to(device)
            outputs = model(input)
            loss = criterion(outputs, label)

            running_loss += loss.item()
            total += label.size(0)
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==label).item()
            
            y_out.extend(pred)
            y.extend(label)

    print(f"Accuracy: {accuracy_score(y, y_out):.3f}")
    print(f"Precision: {precision_score(y, y_out, average='micro'):.3f}")
    print(f"Recall: {recall_score(y, y_out, average='micro'):.3f}")
    print(f"F1 Score: {f1_score(y, y_out, average='micro'):.3f}")

    plt.rc('font', size=7)
    plt.rc('axes', grid=False)
    cm = confusion_matrix(y, y_out, normalize='true')
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot()