import numpy as np
import torch
import time
import torch.nn.functional as F
from torch import optim
from utils.model import *


def fit(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, device, name='model', save_after=10, save_model=False):
    valid_acc_max = 0
    valid_loss, valid_acc = [], []
    train_loss, train_acc = [], []

    for epoch in range(1, epochs+1):
        start = time.time()
        train(model, train_loader, criterion, optimizer, device, train_acc, train_loss)
        valid(model, valid_loader, criterion, device, valid_acc, valid_loss)
        end = time.time()

        print(f'Epoch {epoch} ================================')
        print(f'Train Loss: {train_loss[-1]:.3f}, Accuracy: {train_acc[-1]:.3f}')
        print(f'Valid Loss: {valid_loss[-1]:.3f}, Accuracy: {valid_acc[-1]:.3f}, Time Epoch: {end-start}')

        # scheduler step
        if (isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)):
            scheduler.step(valid_acc[-1])
        else:
            scheduler.step()

        # checkpoint based on validation accuracy
        if valid_acc[-1] > valid_acc_max:
            valid_acc_max = valid_acc[-1]
            if save_model and epoch > save_after:
                save(model, optimizer, f'./{name}_{epoch}_{valid_acc_max:.3f}.tar')
                print(f'Save model epoch {epoch}, valid accuracy {valid_acc[-1]}...')

    return train_loss, train_acc, valid_loss, valid_acc


def train(model, train_loader, criterion, optimizer, device, train_acc, train_loss):
    running_loss = 0
    correct, total = 0, 0
    model.train()

    for input, label in train_loader:
        input, label = input.to(device), label.to(device)
        optimizer.zero_grad()
        outputs = model(input)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total += label.size(0)
        _, pred = torch.max(outputs, dim=1)
        correct += torch.sum(pred==label).item()

    train_acc.append(correct / total)
    train_loss.append(running_loss / len(train_loader))


def valid(model, valid_loader, criterion, device, valid_acc, valid_loss):
    running_loss = 0
    correct, total = 0, 0
    model.eval()

    with torch.no_grad():
        for input, label in valid_loader:
            input, label = input.to(device), label.to(device)
            outputs = model(input)
            loss = criterion(outputs, label)

            running_loss += loss.item()
            total += label.size(0)
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==label).item()
            
        valid_acc.append(correct / total)
        valid_loss.append(running_loss / len(valid_loader))
