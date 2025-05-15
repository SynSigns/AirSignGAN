import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.spectral_norm import spectral_norm

__all__ = ['d_basiclstm', 'd_basicgru', 'd_basiccnn1d', 'd_attentionlstm', 'd_attentioncnn1d']

class GRUDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, normalization=False):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,batch_first=True,dropout=dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        if normalization:
            self.classifier = spectral_norm(self.classifier)

    def forward(self, x):
        out, hn = self.gru(x)           
        last_hidden = hn[-1]             
        logits = self.classifier(last_hidden)
        return logits                     


class LSTMDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0, normalization=False):
        super(LSTMDiscriminator, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.classifier = nn.Linear(hidden_size, 1)
        if normalization:
            self.classifier = spectral_norm(self.classifier)
        

    def forward(self, x):
        out, (hn, _) = self.lstm(x)
        last_hidden = hn[-1]  # shape: [batch, hidden_size]
        logits = self.classifier(last_hidden)
        return logits
    
    
class AttentionLSTMDiscriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,dropout=0.0, normalization=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=1,
            dropout=dropout,
            batch_first=True
        )
        self.classifier = nn.Linear(hidden_size, 1)
        if normalization:
            self.classifier = spectral_norm(self.classifier)

    def forward(self, x):
        # x is (b,t,f)
        out, _ = self.lstm(x)               # (b,t,h)
        attn_out, _ = self.attn(out, out, out)  
        pooled = attn_out.mean(dim=1)       # (b,h)
        logits = self.classifier(pooled)
        return logits 
    

########### for the 1D CNN discriminators, we have patch based discriminators. ###########


class CNN1DDiscriminator(nn.Module):
    def __init__(self,input_size,hidden_size,kernel_size=3,dropout=0.0, normalization=False):
        super().__init__()
        # conv stack
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size, padding='same')
        # if normalization:
        #     self.conv1 = spectral_norm(self.conv1)

        self.relu  = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

        # classifier on pooled features
        self.classifier = nn.Linear(hidden_size, 1)
        if normalization:
            self.classifier = spectral_norm(self.classifier)

    def forward(self, x):
        x = x.permute(0, 2, 1)             # (b,t,f) → (b,f,t)
        x = self.relu(self.conv1(x))
        x = self.drop(x)
        pooled = x.mean(dim=2)                  # global average pool over time → (b, h)
        return self.classifier(pooled)  #single classifier score.



class AttentionCNN1DDiscriminator(nn.Module):
    def __init__(self,input_size,hidden_size,kernel_size=3,dropout=0.0, normalization=False):
        super().__init__()
        # conv stack
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size, padding='same')
        # if normalization:
        #     self.conv1 = spectral_norm(self.conv1)

        self.relu   = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout)

        # attention over the T timesteps
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=1,
            dropout=dropout,
            batch_first=True
        )

        # final classifier after pooling
        self.classifier = nn.Linear(hidden_size, 1)
        if normalization:
            self.classifier = spectral_norm(self.classifier)


    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x)) # from f features to h features
        x = self.drop(x)
        x = x.permute(0, 2, 1)
        x, _ = self.attn(x, x, x)
        x = x.mean(dim=1) #(b,h)
        return self.classifier(x)


def d_basiclstm(**config):
    config.setdefault('input_size', 6)
    config.setdefault('hidden_size', 32)
    config.setdefault('num_layers', 3)
    config.setdefault('dropout', 0.0)
    config.setdefault('normalization', True)
    return LSTMDiscriminator(
        input_size  = config['input_size'],
        hidden_size = config['hidden_size'],
        num_layers  = config['num_layers'],
        dropout     = config['dropout'],
        normalization=config['normalization']
    )

def d_basicgru(**config):
    config.setdefault('input_size', 6)
    config.setdefault('hidden_size', 32)
    config.setdefault('num_layers', 3)
    config.setdefault('dropout', 0.0)
    config.setdefault('normalization', True)
    return GRUDiscriminator(
        input_size  = config['input_size'],
        hidden_size = config['hidden_size'],
        num_layers  = config['num_layers'],
        dropout     = config['dropout'],
        normalization=config['normalization']
    )

def d_attentionlstm(**config):
    config.setdefault('input_size', 6)
    config.setdefault('hidden_size', 32)
    config.setdefault('num_layers', 3)
    config.setdefault('dropout', 0.0)
    config.setdefault('normalization', True)
    return AttentionLSTMDiscriminator(
        input_size  = config['input_size'],
        hidden_size = config['hidden_size'],
        num_layers  = config['num_layers'],
        dropout     = config['dropout'],
        normalization=config['normalization']
    )

def d_basiccnn1d(**config):
    config.setdefault('input_size', 6)
    config.setdefault('hidden_size', 32)
    config.setdefault('kernel_size', 3)
    config.setdefault('dropout', 0.0)
    config.setdefault('normalization', True)
    return CNN1DDiscriminator(
        input_size  = config['input_size'],
        hidden_size = config['hidden_size'],
        kernel_size  = config['kernel_size'],
        dropout     = config['dropout'],
        normalization=config['normalization']
    )

def d_attentioncnn1d(**config):
    config.setdefault('input_size', 6)
    config.setdefault('hidden_size', 32)
    config.setdefault('kernel_size', 3)
    config.setdefault('dropout', 0.0)
    config.setdefault('normalization', True)
    return AttentionCNN1DDiscriminator(
        input_size  = config['input_size'],
        hidden_size = config['hidden_size'],
        kernel_size  = config['kernel_size'],
        dropout     = config['dropout'],
        normalization=config['normalization']
    )

