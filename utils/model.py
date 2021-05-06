import numpy as np
import torch

def train(model, train_loader, valid_loader, criterion, optimizer, epochs, device, save_model=False):
    n_epochs = epochs
    valid_loss_min = np.Inf
    valid_loss, valid_acc = [], []
    train_loss, train_acc = [], []

    for epoch in range(1, n_epochs+1):
        loss_epoch = 0.0
        correct, total = 0, 0
        model.train()
        print(f'Epoch {epoch}')

        for batch_idx, (input, label) in enumerate(train_loader):
            input, label = input.to(device), label.to(device)
            optimizer.zero_grad()
            
            outputs = model(input)
            # _, label = torch.max(label, dim=1)

            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            loss_epoch += loss.item()
            total += label.size(0)
            _, pred = torch.max(outputs, dim=1)
            correct += torch.sum(pred==label).item()
            
            if (batch_idx) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch, n_epochs, batch_idx, len(train_loader), loss.item()))

        train_acc.append(correct / total)
        train_loss.append(loss_epoch / len(train_loader))
        print(f'train-loss: {np.mean(train_loss):.4f}, train-acc: {(correct / total):.4f}')
        
        # validation phase
        loss_epoch = 0.0
        correct, total = 0, 0
        model.eval()

        with torch.no_grad():
            for input, label in (valid_loader):
                input, label = input.to(device), label.to(device)
                outputs = model(input)
                _, label = torch.max(label, dim=1)

                loss = criterion(outputs, label)

                loss_epoch += loss.item()
                total += label.size(0)
                _,pred = torch.max(outputs, dim=1)
                correct += torch.sum(pred==label).item()
                
            valid_acc.append(correct / total)
            valid_loss.append(loss_epoch / len(valid_loader))
            network_learned = loss_epoch < valid_loss_min
            print(f'valididation loss: {np.mean(valid_loss):.4f}, valididation acc: {(correct / total):.4f}')
            
            if network_learned and save_model:
                valid_loss_min = loss_epoch
                torch.save(model.state_dict(), f'./model_base_patch16_224_ep{epoch}_lr1e-3.pt')
                print('Saving model...')

    return train_loss, train_acc, valid_loss, valid_acc