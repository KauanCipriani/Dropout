import torch
import torch.nn.functional as F

def evaluate(model, dataloader, device):
    """
    Avalia o modelo no conjunto de teste e retorna (loss, accuracy).
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()

            _, pred = output.max(1)  # <- pega apenas os índices das classes preditas
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy
