import matplotlib
matplotlib.use('Agg')  # usa backend sem interface gráfica
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import os

def set_seed(seed=42):
    """
    Define semente global para resultados reprodutíveis.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, path="results/models/model.pt"):
    """
    Salva o modelo treinado.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"✅ Modelo salvo em {path}")


def plot_losses(train_losses, val_losses, title="Treinamento"):
    """
    Gera e salva gráfico de perdas.
    """
    os.makedirs("results/figures", exist_ok=True)
    plt.figure()
    plt.plot(train_losses, label="Treino")
    plt.plot(val_losses, label="Validação")
    plt.title(title)
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.legend()
    path = f"results/figures/{title.replace(' ', '_').lower()}.png"
    plt.savefig(path)
    plt.close()
    print(f"📊 Gráfico salvo em {path}")
