import torch
from sklearn.metrics import precision_score, f1_score, recall_score, confusion_matrix

def evaluate(model, data_loader, criterion, device):
    running_loss, total = 0, 0
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

            y_out.append(pred)
            y.append(label)

    y_out = [item for batch in y_out for item in batch] 
    y = [item for batch in y for item in batch] 
    print(f"Precision: {precision_score(y, y_out, average='micro'):.3f}")
    print(f"Recall: {recall_score(y, y_out, average='micro'):.3f}")
    print(f"F1 Score: {f1_score(y, y_out, average='micro'):.3f}")
    print(f"Confusion Matrix:\n {confusion_matrix(y, y_out)}, '\n'")


def saliency_map(input, model):
    image = torch.squeeze(input)
    input.requires_grad_()

    output = model(input)
    output_max = output[0, output.argmax()]
    output_max.backward()

    saliency = torch.max(input.grad.data.abs(), dim=1)[0][0]

    # rescale between 0 and 1
    saliency -= saliency.min(1, keepdim=True)[0]
    saliency /= saliency.max(1, keepdim=True)[0]

    return image, saliency