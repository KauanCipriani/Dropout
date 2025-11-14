import torch.nn as nn
import torch.nn.functional as F

class MLPDropout(nn.Module):
    """
    MLP simples com Dropout, inspirado no artigo de Srivastava et al. (2014).
    """

    def __init__(self, dropout_rate=0.5):
        super(MLPDropout, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
