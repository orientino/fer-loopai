import numpy as np
from numpy.lib.arraysetops import isin
import torch
import matplotlib.pyplot as plt
from torch import optim
from torch.optim import lr_scheduler

def train(model, train_loader, valid_loader, criterion, optimizer, scheduler, epochs, device, save_model=False, verbose=0):
    n_epochs = epochs
    valid_loss_min = np.Inf
    valid_loss, valid_acc = [], []
    train_loss, train_acc = [], []

    for epoch in range(1, n_epochs+1):
        # training phase
        running_loss = 0.0
        correct, total = 0, 0
        model.train()

        for batch_idx, (input, label) in enumerate(train_loader):
            input, label = input.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(input)
            # _, label = torch.max(label, dim=1)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            total += label.size(0)
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==label).item()
            
            if (batch_idx) % 100 == 0 and verbose>2:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}' 
                    .format(epoch, n_epochs, batch_idx, len(train_loader), loss.item()))

        train_acc.append(correct / total)
        train_loss.append(running_loss / len(train_loader))
        print('==================================')
        print(f'Epoch [{epoch}/{n_epochs}] Train Loss: {train_loss[-1]:.3f}, Accuracy: {train_acc[-1]:.3f}')
        
        # validation phase
        running_loss = 0.0
        correct, total = 0, 0
        model.eval()

        with torch.no_grad():
            for input, label in (valid_loader):
                input, label = input.to(device), label.to(device)
                outputs = model(input)
                loss = criterion(outputs, label)

                running_loss += loss.item()
                total += label.size(0)
                _,pred = torch.max(outputs, dim=1)
                correct += torch.sum(pred==label).item()
                
            valid_acc.append(correct / total)
            valid_loss.append(running_loss / len(valid_loader))
            print(f'Epoch [{epoch}/{n_epochs}] Valid Loss: {valid_loss[-1]:.3f}, Accuracy: {valid_acc[-1]:.3f}')

            if running_loss < valid_loss_min and save_model:
                valid_loss_min = running_loss
                torch.save(model.state_dict(), f'./model_base_patch16_224_ep{epoch}_lr1e-3.pt')
                print('Saving model...')

        # take the scheduler step
        if (isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)):
            scheduler.step(valid_acc[-1])
        else:
            scheduler.step()

    return train_loss, train_acc, valid_loss, valid_acc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_loss_accuracy(train_loss, valid_loss, train_acc, valid_acc):
    fig = plt.figure(figsize=(20, 5)) 
    fig_dims = (1, 2)

    # plot loss curve
    plt.subplot2grid(fig_dims, (0, 0))
    plot_loss(train_loss, valid_loss)
    
    # plot accuracy curve
    plt.subplot2grid(fig_dims, (0, 1))
    plot_accuracy(train_acc, valid_acc)


def plot_loss(train_loss, valid_loss):
    # loss mean and std
    train_loss_mean = np.mean(train_loss, axis=0)
    train_loss_std = np.std(train_loss, axis=0)
    valid_loss_mean = np.mean(valid_loss, axis=0)
    valid_loss_std = np.std(valid_loss, axis=0)
    train_sizes = range(len(train_loss[0]))

    # clamp values in 0, inf
    train_up = train_loss_mean + train_loss_std
    train_down = train_loss_mean - train_loss_std
    train_down = [loss if loss > 0 else 0 for loss in train_down]
    
    test_up = valid_loss_mean + valid_loss_std
    test_down = valid_loss_mean - valid_loss_std
    test_down = [loss if loss > 0 else 0 for loss in test_down]
    
    plt.plot(train_sizes, train_loss_mean, label='train loss')
    plt.fill_between(train_sizes, train_up, train_down, alpha=0.2)
    plt.plot(train_sizes, valid_loss_mean, label='validation loss')
    plt.fill_between(train_sizes, test_up, test_down, alpha=0.2)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    print(f'Train Loss mean: {train_loss_mean[-1]:.3f}, std: {train_loss_std[-1]:.3f}')
    print(f'Valid Loss mean: {valid_loss_mean[-1]:.3f}, std: {valid_loss_std[-1]:.3f}')


def plot_accuracy(train_acc, valid_acc):
    # accuracy mean and std
    train_acc_mean = np.mean(train_acc, axis=0)
    train_acc_std = np.std(train_acc, axis=0)
    valid_acc_mean = np.mean(valid_acc, axis=0)
    valid_acc_std = np.std(valid_acc, axis=0)
    train_sizes = range(len(train_acc[0]))

    # clamp values in 0, 1
    train_up = train_acc_mean + train_acc_std
    train_up = [acc if acc < 1 else 1 for acc in train_up]
    train_down = train_acc_mean - train_acc_std
    train_down = [acc if acc > 0 else 0 for acc in train_down]
    
    test_up = valid_acc_mean + valid_acc_std
    test_up = [acc if acc < 1 else 1 for acc in test_up]
    test_down = valid_acc_mean - valid_acc_std
    test_down = [acc if acc > 0 else 0 for acc in test_down]
    
    # plot the learning curve
    plt.plot(train_sizes, train_acc_mean, label='train acc')
    plt.fill_between(train_sizes, train_up, train_down, alpha=0.2)
    plt.plot(train_sizes, valid_acc_mean, label='validation acc')
    plt.fill_between(train_sizes, test_up, test_down, alpha=0.2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    print(f'Train Accuracy mean: {train_acc_mean[-1]:.3f}, std: {train_acc_std[-1]:.3f}')
    print(f'Valid Accuracy mean: {valid_acc_mean[-1]:.3f}, std: {valid_acc_std[-1]:.3f}')