import math
import torch
from torch import nn

class SelfAttention(nn.Module):
  def __init__(self, input_dim):
    super(SelfAttention, self).__init__()
    self.query = nn.Linear(input_dim, input_dim)
    self.key = nn.Linear(input_dim, input_dim)
    self.value = nn.Linear(input_dim, input_dim)
    # Add a Layer Normalisation layer
    self.layer_norm = nn.LayerNorm(input_dim)
    # Add a learnable weight
    self.learnable_weight = nn.Parameter(torch.ones((input_dim,)))
    
  def forward(self, x):
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    d_k = k.size(-1)
    weights = torch.nn.functional.softmax((q @ k.transpose(-2, -1)) / math.sqrt(d_k), dim=-1)
    output = weights @ v
    # Apply the learnable weight
    output = output * self.learnable_weight
    # Apply Layer Normalisation to the output
    output = self.layer_norm(output)
    return output
  
class CrossAttention(nn.Module):
  def __init__(self, query_dim, key_dim):
    super(CrossAttention, self).__init__()
    self.query = nn.Linear(query_dim, key_dim)
    self.key = nn.Linear(key_dim, key_dim)
    self.value = nn.Linear(key_dim, key_dim)

  def forward(self, query, key):
    q = self.query(query)
    k = self.key(key)
    v = self.value(key)
    d_k = k.size(-1)
    weights = torch.nn.functional.softmax((q @ k.transpose(-2, -1)) / math.sqrt(d_k), dim=-1)
    output = weights @ v
    return output