import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


def softmax(x, dim):
    x = x - torch.max(x, axis=dim, keepdim=True).values
    exp_x = torch.exp(x)
    output = exp_x / torch.sum(exp_x, axis=dim, keepdim=True)

    return output


def scaled_dot_product_attention(K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    prob: Optional[float] = None,):
    """
    K: (bs, ..., seq_len, key_dim)
    Q: (bs, ..., seq_len, key_dim)
    V: (bs, ..., seq_len, value_dim)
    mask: (seq_len, seq_len)
    prob: float
    """
    qk = Q @ K.transpose(-1, -2) / np.sqrt(K.size()[-1])    # (bs, ..., seq_len, seq_len)

    if mask is not None:
        mask_values = torch.where(mask, float('-inf'), 0.0)
        qk = qk + mask_values

    attention_weights = softmax(qk, dim=-1)

    if prob is not None:
        attention_weights = F.dropout(attention_weights, p=prob)


    output = attention_weights @ V  #(bs, ..., seq_len, value_dim)

    return output


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps: float=1e-5):
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


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_pdrop):
        super(MultiHeadSelfAttention, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.d_key = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)


    def forward(self, x):
        B, T, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, self.num_heads, self.d_key).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.d_key).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.d_key).transpose(1, 2)

        # mask
        # casual_mask = torch.triu(torch.ones(T, T)).bool()
        casual_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        
        output = scaled_dot_product_attention(k, q, v, casual_mask, self.attn_pdrop)    # (bs, heads, seq_len, d_key)
        output = output.transpose(1,2)  # (bs, seq_len, heads, d_key)
        output = output.contiguous().view(B, T, self.d_model)
        output = self.output_proj(output)   
        
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_fnn, num_heads, attn_pdrop, residual_pdrop):
        super(TransformerBlock, self).__init__()

        self.d_model = d_model
        self.d_fnn = d_fnn
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.residual_pdrop = residual_pdrop

        self.ln1 = RMSNorm(self.d_model)
        self.ln2 = RMSNorm(self.d_model)
        self.attn = MultiHeadSelfAttention(self.d_model, self.num_heads, self.attn_pdrop)
        self.ffn = FFN(self.d_model, self.d_fnn)

    def forward(self, x):
 
        tmp_x = self.attn(self.ln1(x))
        if self.residual_pdrop:
            tmp_x = F.dropout(tmp_x, p=self.residual_pdrop)
        sub_layer1_output = x + tmp_x

        output = self.ffn(self.ln2(sub_layer1_output))
        if self.residual_pdrop:
            output = F.dropout(output, p=self.residual_pdrop)
        output = output + sub_layer1_output

        return output


class TransformerLM(nn.Module):
    def __init__(self, vocab_size, 
                 context_length, 
                 d_model, 
                 residual_pdrop, 
                 num_layers, 
                 d_fnn, 
                 num_heads, 
                 attn_pdrop):
        super(TransformerLM, self).__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.residual_pdrop = residual_pdrop

        self.drop = nn.Dropout(self.residual_pdrop)

        self.token_embeddings = nn.Embedding(self.vocab_size, self.d_model)
        self.position_embeddings = nn.Embedding(self.context_length, self.d_model)

        self.num_layers = num_layers
        self.d_fnn = d_fnn
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.layers = nn.ModuleList([TransformerBlock(self.d_model, 
                                                        self.d_fnn, 
                                                        self.num_heads, 
                                                        self.attn_pdrop, 
                                                        self.residual_pdrop) for _ in range(self.num_layers)])
                    
        self.ln_final = RMSNorm(self.d_model)

        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)
            

    def forward(self, x):
        B, T = x.size()
        positions = torch.arange(T).expand(B, T)

        tokens_emb = self.token_embeddings(x)
        position_emb = self.position_embeddings(positions)

        input_emb = tokens_emb + position_emb
        input_emb = self.drop(input_emb)

        output = input_emb
        for layer in self.layers:
            output = layer(output)

        output = self.ln_final(output)
        output = self.lm_head(output)

        return output

