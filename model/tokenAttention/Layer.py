import torch
import torch.nn as nn
import torch.nn.functional as F

import os 
import numpy as np

from model.tokenAttention import FeedForwardNetwork
from model.tokenAttention import MultiHeadAttention

class Layer(nn.Module):

	def __init__(self, d_model, n_head):

		super(Layer,self).__init__()
		self.d_model = d_model
		self.n_head = n_head
		self.attn_network = MultiHeadAttention.MultiHeadAttention(d_model, n_head)
		self.ffn = FeedForwardNetwork.FeedForwardNetwork()

	def forward(self, query, key, val, key_structure = None, val_structure = None, attention_mask = None):

		self_atten_features, atten_values = self.attn_network(query, key, val, key_structure = key_structure, val_structure = val_structure, attention_mask = attention_mask)
		enc_output = self.ffn(self_atten_features)

		del self_atten_features
		torch.cuda.empty_cache()

		return enc_output, atten_values