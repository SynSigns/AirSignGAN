import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from torch.nn import functional as F
from utils.core import timeseries_resize

__all__ = ['g_basiclstm', 'g_basicgru', 'g_basiccnn1d', 'g_attentionlstm', 'g_attentioncnn1d']

class BasicGRUBlock(nn.Module):
    """
    It takes as input a tensor x (e.g. noise+previous output) with shape [batch, timesteps, features],
    runs it through a GRU and a linear layer, and returns a residual output.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, input_size)
    
    def forward(self, z, y):
        x = y + z # z,y have (b,t,f)
        out, _ = self.gru(x)
        out = self.linear(out)
        return y + out  # residual connection

class BasicLSTMBlock(nn.Module):
    """
    It takes as input a tensor x (e.g. noise+previous output) with shape [batch, timesteps, features],
    runs it through an LSTM and a linear layer, and returns a residual output.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        # Project back to the original feature dimension.
        self.linear = nn.Linear(hidden_size, input_size)
    
    def forward(self, z, y):
        # z and y are expected to have shape [batch, timesteps, features]
        # z is the previous output with noise added, y is the output of compute_previous.
        x = y + z
        out, _ = self.lstm(x)
        out = self.linear(out)
        return y + out  # residual connection

class AttentionLSTMBlock(nn.Module):
    """
    The basic lstm block with attention.
    """
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=1, dropout=dropout ,batch_first=True)
        # Project back to the original feature dimension.
        self.linear = nn.Linear(hidden_size, input_size)
    
    def forward(self, z, y):
        # z and y are expected to have shape [batch, timesteps, features]
        # z is the previous output with noise added, y is the output of compute_previous.
        x = y + z
        out, _ = self.lstm(x)
        attn_out,_ = self.attn(out, out, out)
        out = self.linear(attn_out)
        return y + out  # residual connection

class BasicCNN1DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size, padding='same')
        self.relu  = nn.ReLU(inplace=True)
        self.linear = nn.Linear(hidden_size, input_size)
        self.drop  = nn.Dropout(dropout)

    def forward(self, z, y):
        x = y + z                      
        x = x.permute(0, 2, 1)          # (b,f,t) for conv1d
        x = self.conv1(x)              
        x = self.relu(x)
        x = self.drop(x)
        x = x.permute(0, 2, 1)          # back to (b,t,h)
        x = self.linear(x)              # (b,t,f)              
                 
        return y + x                    # residual
    
class AttentionCNN1DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size=3,dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size, padding='same')
        self.relu  = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.attn = nn.MultiheadAttention(embed_dim=hidden_size,num_heads=1,dropout=dropout,batch_first=True)

        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, z, y):
        x = y + z                         
        x = x.permute(0, 2, 1)             # (b,f,t) for conv1d
        x = self.conv1(x)                  
        x = self.relu(x)
        x = self.drop1(x)
        x = x.permute(0, 2, 1)             # back to (b,t,h) for attention

        attn_out, _ = self.attn(x, x, x)

        update = self.linear(attn_out)     # (b,t,f)
        return y + update       

class MultiLSTMGenerator(nn.Module):
    """
    Multi-scale generator for time series using LSTM blocks.
    Similar in spirit to MultiVanilla, but working on sequences.
    """
    def __init__(self, block_factory, config):
        super().__init__()
        self.block_factory = block_factory
        self.config = config
        self.scale = 0
        self.key = 's0'
        self.scale_factor = 0  # Will be set during initialization (_init_models)

        # the current generator block.
        self.curr = self.block_factory(**self.config)

        # for previous (frozen) generator blocks.
        self.prev = nn.Module()
    
    def add_scale(self, device):
        self.scale += 1

        # previous
        self.prev.add_module(self.key, deepcopy(self.curr))
        self._reset_grad(self.prev, False)
        self.key = f's{self.scale}'

        self.curr = self.block_factory(**self.config).to(device)

    
    def _compute_previous(self, reals, amps, noises=None):
        """
        Returns: a tensor y of shape [1, timesteps, features] (upsampled to match the next scale)
        """
        keys = list(reals.keys())
        # Start from a zero tensor matching the coarsest scale.
        y = torch.zeros_like(reals[keys[0]])
        
        # Loop over scales. Assume keys are ordered from coarsest (s0) to finer scales.
        for key, single_scale in self.prev.named_children():
            # Determine next key (assumes keys are in order: s0, s1, s2, ...)
            next_key = keys[keys.index(key) + 1]
            # fixed z
            if noises and key in noises:
                z = y + amps[key].view(-1, 1, 1) * noises[key]
            else:
                n = self._generate_noise(reals[key], repeat=(key == 's0'))
                z = y + amps[key].view(-1, 1, 1) * n
            
            y = single_scale(z, y)
            y = timeseries_resize(y, 1./self.scale_factor)
            target_length = reals[next_key].size(1)
            y = y[:, :target_length, :]  # Truncate to match the next scale
        return y


    def _generate_noise(self, tensor_like, repeat=False):
        """
        Generates noise for a time series.
        tensor_like: expected shape [1, timesteps, features]
        If repeat is True, generate noise with a single feature and repeat it across channels.
        """
        if not repeat:
            noise = torch.randn(tensor_like.size(), device=tensor_like.device)
        else:
            batch, timesteps, features = tensor_like.size()
            noise = torch.randn((batch, 1, features), device=tensor_like.device)
            noise = noise.repeat(1, timesteps, 1)
        return noise

    def forward(self, reals, amps, noises=None):
        # compute previous layers (without gradients).
        with torch.no_grad():
            y = self._compute_previous(reals, amps, noises).detach()
        # For the current scale, get noise (or use provided one).
        if noises:
            # print(f"shape of y is {y.shape}",flush=True)
            # print(f"shape of noises is {noises[self.key].shape}",flush=True)
            # print(f"shape of amps is {amps[self.key].shape}",flush=True)
            # print(f"shape of reals is {reals[self.key].shape}",flush=True)
            z = y + amps[self.key].view(-1, 1, 1) * noises[self.key]
        else:
            n = self._generate_noise(reals[self.key], repeat=(not self.scale))
            z = y + amps[self.key].view(-1, 1, 1) * n
        o = self.curr(z.detach(), y.detach())
        return o

    def _reset_grad(self, model, require_grad=False):
        for p in model.parameters():
            p.requires_grad_(require_grad)

    def train(self, mode=True):
        # Override train so that previous blocks stay in eval mode.
        self.training = mode
        for module in self.curr.children():
            module.train(mode)
        for module in self.prev.children():
            module.train(False)
        return self

    def eval(self):
        return self.train(False)
    


def g_basiclstm(**config):
    config.setdefault('input_size', 6)
    config.setdefault('hidden_size', 32)
    config.setdefault('num_layers', 3)
    config.setdefault('dropout', 0.0)
    return MultiLSTMGenerator(
        block_factory=lambda **cfg: BasicLSTMBlock(
            input_size  = cfg['input_size'],
            hidden_size = cfg['hidden_size'],
            num_layers  = cfg['num_layers'],
            dropout     = cfg['dropout']
        ),
        config=config
    )

def g_basicgru(**config):
    config.setdefault('input_size', 6)
    config.setdefault('hidden_size', 32)
    config.setdefault('num_layers', 3)
    config.setdefault('dropout', 0.0)
    return MultiLSTMGenerator(
        block_factory=lambda **cfg: BasicGRUBlock(
            input_size  = cfg['input_size'],
            hidden_size = cfg['hidden_size'],
            num_layers  = cfg['num_layers'],
            dropout     = cfg['dropout']
        ),
        config=config
    )

def g_attentionlstm(**config):
    config.setdefault('input_size', 6)
    config.setdefault('hidden_size', 32)
    config.setdefault('num_layers', 3)
    config.setdefault('dropout', 0.0)
    return MultiLSTMGenerator(
        block_factory=lambda **cfg: AttentionLSTMBlock(
            input_size  = cfg['input_size'],
            hidden_size = cfg['hidden_size'],
            num_layers  = cfg['num_layers'],
            dropout     = cfg['dropout']
        ),
        config=config
    )

def g_basiccnn1d(**config):
    config.setdefault('input_size', 6)
    config.setdefault('hidden_size', 32)
    config.setdefault('kernel_size', 3)
    config.setdefault('dropout', 0.0)
    return MultiLSTMGenerator(
        block_factory=lambda **cfg: BasicCNN1DBlock(
            input_size  = cfg['input_size'],
            hidden_size = cfg['hidden_size'],
            kernel_size = cfg['kernel_size'],
            dropout     = cfg['dropout']
        ),
        config=config
    )

def g_attentioncnn1d(**config):
    config.setdefault('input_size', 6)
    config.setdefault('hidden_size', 32)
    config.setdefault('kernel_size', 3)
    config.setdefault('dropout', 0.0)
    return MultiLSTMGenerator(
        block_factory=lambda **cfg: AttentionCNN1DBlock(
            input_size  = cfg['input_size'],
            hidden_size = cfg['hidden_size'],
            kernel_size = cfg['kernel_size'],
            dropout     = cfg['dropout']
        ),
        config=config
    )
