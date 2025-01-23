import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight  = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        output = (x / output) * self.weight 

        return output


class GELU(nn.Module):
    def forward(self, x):
        output = x * (1 + torch.erf(x / (2 ** 0.5))) / 2
        return output


class FFN(nn.Module):
    def __init__(self, d_model, d_fnn):
        super(FFN, self).__init__()
        self.d_model = d_model
        self.d_fnn = d_fnn
        self.w1 = nn.Linear(d_model, d_fnn, bias=False)
        self.w2 = nn.Linear(d_fnn, d_model, bias=False)

        self.activate = GELU()
    
    def forward(self, x):
        # x: (batch_size, seq_length, d_model), w1: (d_dff, d_model) -> (batch_size, seq_length, d_ff)
        output = self.w2(self.activate(self.w1(x)))
        return output
