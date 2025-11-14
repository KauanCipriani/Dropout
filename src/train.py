import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import get_mnist_loaders
from src.model import MLPDropout
from src.eval import evaluate
from src.utils import set_seed, save_checkpoint, plot_losses


def train_model(dropout_rate=0.5, epochs=10, lr=0.001, seed=42):
    """
    Treina o modelo MLP com taxa de Dropout configurável.
    """
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Usando dispositivo: {device}")

    train_loader, test_loader = get_mnist_loaders(batch_size=64)
    model = MLPDropout(dropout_rate).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    train_losses, val_losses, val_accs = [], [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_loss, val_acc = evaluate(model, test_loader, device)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"📘 Época {epoch+1}/{epochs} | Dropout={dropout_rate:.2f} | "
              f"Treino Loss={avg_train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.2f}%")

    plot_losses(train_losses, val_losses, title=f"Dropout {dropout_rate}")
    save_checkpoint(model, f"results/models/model_dropout{int(dropout_rate*100)}.pt")

    print(f"✅ Treinamento finalizado! Acurácia final: {val_accs[-1]:.2f}%")
    return val_accs[-1]


if __name__ == "__main__":
    print("Treinando modelo SEM dropout...")
    acc_no_dropout = train_model(dropout_rate=0.0, epochs=10)

    print("\nTreinando modelo COM dropout (0.5)...")
    acc_with_dropout = train_model(dropout_rate=0.5, epochs=10)

    print("\n📊 Comparação final:")
    print(f"Sem dropout: {acc_no_dropout:.2f}%")
    print(f"Com dropout (0.5): {acc_with_dropout:.2f}%")
