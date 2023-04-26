import numpy as np
import torch
import config
from PositionEncoder import PositionEncoder
from WordModule import WordModule
import config
position = PositionEncoder(15)
word_position = torch.tensor([np.arange(0,15)])
# print(word_position.shape)
encode_pos = position(word_position)
# print(encode_pos.shape)
encode_pos.reshape(-1,1,15,200)
# print(encode_pos.shape)
# print(encode_pos)
# word = torch.tensor(config.glove.vectors[np.arange(0,15),:])

word = torch.FloatTensor(np.zeros([1,1,15,200]))
word[0,0,:,:] = torch.as_tensor(config.glove.vectors[0,:])
# print(word.shape)
# print(word)
word_module = WordModule()
X_word, self_atten_weights_dict_word = word_module(word,encode_pos)
print(X_word.shape)
print(X_word)
print(self_atten_weights_dict_word)
