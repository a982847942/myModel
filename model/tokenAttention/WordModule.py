import torch 
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from model.tokenAttention import Transformer
import config

class WordModule(nn.Module):

	@staticmethod
	def init_weights(layer):
		if type(layer) == nn.Linear:
			nn.init.xavier_normal_(layer.weight)

	def __init__(self):

		super(WordModule, self).__init__()

		# <----------- Config ----------->

		if config.word_module_version in [2,3,4]:

			# <----------- Word Level Transformer ----------->
			self.transformer_word = Transformer.Transformer( config.n_mha_layers_word, config.d_model, config.n_head_word)

		# <----------- Embedding the words through n FC layers (word level) ----------->
		if config.ff_word:
			self.emb_layer_word = nn.ModuleList([nn.Linear(config.embedding_dim, config.embedding_dim) for _ in range(config.num_emb_layers_word)])

		if config.word_module_version == 4:

			# <----------- To map each post vector to a scalar ----------->
			self.condense_layer_word = nn.Linear(config.d_model, 1)

		# <----------- Droutput for regularization ----------->
		# self.dropout = nn.Dropout(p = config.dropout_rate, inplace = True)
		self.dropout = nn.Dropout(p = config.dropout_rate, inplace = False)

		# <----------- Initialization of weights ----------->
		if config.ff_word:
			self.emb_layer_word.apply(WordModule.init_weights)

		if config.word_module_version == 4:
			self.condense_layer_word.apply(WordModule.init_weights)


	def forward(self, X, word_pos, attention_mask = None):

		# <----------- Getting the dimensions ----------->
		# batch_size, num_posts, num_words, emb_dim = X.shape
		num_posts, num_words, emb_dim = X.shape

		# <----------- Passing X with n number of FC layers (word level), set on/off in config ----------->
		#为什么要经过全连接层？
		if config.ff_word:
			for i in range(config.num_emb_layers_word):
				X = self.emb_layer_word[i](X)

		# <----------- Adding in the position information ----------->
		X += word_pos

		# <----------- Setting the query, key and val ----------->
		query_word = X 
		key_word = X
		val_word = X

		# <----------- Dropout ----------->
		self.dropout(query_word)
		self.dropout(key_word)
		self.dropout(val_word)

		# <----------- Clearing some memory ----------->
		del X
		torch.cuda.empty_cache()

		if config.word_module_version in [2,3,4]:

			# <----------- Passing through word level transformer (Not keeping the attention values for now) ----------->
			X_word, self_atten_weights_dict_word = self.transformer_word(query_word, key_word, val_word, attention_mask = attention_mask)
			
		else:

			X_word = query_word
			self_atten_weights_dict_word = {}

		# # <----------- Adding dropout to X_word ----------->
		# self.dropout(X_word)

		# <----------- Baseline (Without self attention)- 1: Max pooling of X (No self attention) ----------->

		if config.word_module_version == 0:

			X_word = query_word.view(-1, num_words, emb_dim)
			X_word = X_word.permute(0, 2, 1).contiguous()
			X_word = F.adaptive_max_pool1d(X_word, 1).squeeze(-1)
			X_word = X_word.view(num_posts, emb_dim)

		# <----------- Baseline (Without self attention) - 2: Average pooling of X (No self attention) ----------->

		if config.word_module_version == 1:

			X_word = query_word.view(-1, num_words, emb_dim)
			X_word = X_word.permute(0, 2, 1).contiguous()
			X_word = F.adaptive_avg_pool1d(X_word, 1).squeeze(-1)
			X_word = X_word.view( num_posts, emb_dim)

		# <----------- Improvement made: TO PERFORM SELF ATTENTION FOR WORDS! ----------->
		
		# <----------- Baseline (With self attention): Max pooling to get the most important words per post ----------->
		if config.word_module_version == 2:

			X_word = X_word.view(-1, num_words, emb_dim)
			X_word = X_word.permute(0,2,1).contiguous()
			X_word = F.adaptive_max_pool1d(X_word, 1).squeeze(-1)
			X_word = X_word.view( num_posts, emb_dim)

		# <----------- Baseline (With self attention): Average pooling to get the most important words per post ----------->
		if config.word_module_version == 3:

			X_word = X_word.view(-1, num_words, emb_dim)
			X_word = X_word.permute(0,2,1).contiguous()
			X_word = F.adaptive_avg_pool1d(X_word, 1).squeeze(-1)
			X_word = X_word.view( num_posts, emb_dim)

		# <----------- Improvement 1 (With self attention): Attention to get important word embedding ----------->
		if config.word_module_version == 4:

			# attention_mask += -1.0
			# attention_mask *= 100000.0

			words_attention_values = self.condense_layer_word(X_word)
			# words_attention_weights = F.softmax(words_attention_values.permute(0,1,3,2) + attention_mask.unsqueeze(-2), dim = -1)
			# words_attention_weights = F.softmax(words_attention_values.permute(0,1,3,2) , dim = -1)
			words_attention_weights = F.softmax(words_attention_values.permute(0,2,1) , dim = -1)

			del attention_mask
			del words_attention_values
			torch.cuda.empty_cache()
			
			X_word = torch.matmul(words_attention_weights, X_word).squeeze(-2)

		return X_word, self_atten_weights_dict_word
