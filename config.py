import torch
import torchtext.vocab as vocab
import torch.nn as nn

# print(vocab.pretrained_aliases.keys())
# glove向量
glove_directory = "H:/pythonProject/rumor detection/myModel/data/glove"
glove = vocab.GloVe(name='twitter.27B', dim=200, cache=glove_directory)
glove.unk_init = torch.Tensor.normal_
# print("一共包含%d个词。" % len(glove.stoi))
# print(glove.stoi['beautiful'], glove.itos[3366])
# print(glove.vectors[0])
# input = torch.LongTensor([glove.stoi["the"]]) #词汇表中的index
# print(input)
# input = torch.LongTensor([glove.stoi["The"]]) #词汇表中的index
# print(input)
embedding = nn.Embedding(glove.vectors.size(0), glove.vectors.size(1))
embedding.weight.data.copy_(glove.vectors)  # 权重使用预训练的glove
# print(embedding(input))
# print(glove.vectors[13])
embedding_dim = 200
source_length = 15
d_model = 200
dropout_rate = 0.3
train_pos_emb = False
user_embedding_dim = 12
# <------------------------ WORD LEVEL ------------------------>
ff_word = True
num_emb_layers_word = 2 # Model parameters settings (To encode query, key and val)
n_mha_layers_word = 2 # Number of Multihead Attention layers
n_head_word = 2 # Number of MHA heads
word_module_version = 4  # {0: max_pooling, 1: average_pooling, 2: max_pooling_w_attention, 3: average_pooling_w_attention, 4: attention}
n_head = 2  # Number of MHA heads
# Model parameters settings (For feedforward network)
d_feed_forward = 600
gpu = False
# list = [1,2,3]
# input = torch.LongTensor([list])
# print(embedding(input))
import numpy as np
# rootfeat = np.zeros([source_length, 200])
# rootfeat[1:3,:] = glove.vectors[1:3,:]
# print(rootfeat)



