import torch
import torchtext.vocab as vocab
import torch.nn as nn
# print(vocab.pretrained_aliases.keys())
# glove向量
glove_directory = "./data/glove"
glove_file = "glove.6B.300d.txt"
glove = vocab.GloVe(name='twitter.27B', dim=200, cache=glove_directory)
# print("一共包含%d个词。" % len(glove.stoi))
# print(glove.stoi['beautiful'], glove.itos[3366])
# print(glove.vectors[0])
# input = torch.LongTensor([glove.stoi["the"]]) #词汇表中的index
# print(input)
embedding = nn.Embedding(glove.vectors.size(0), glove.vectors.size(1))
embedding.weight.data.copy_(glove.vectors) #权重使用预训练的glove
# print(embedding(input))
# print(glove.vectors[13])
