import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data

import config
from model.tokenAttention.PositionEncoder import PositionEncoder

class GraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('../..', '..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.droprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        rootFeat = torch.FloatTensor(data['rootUserFeat'])
        wordFeat = torch.FloatTensor(data['x'])
        return Data(x=wordFeat,
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=rootFeat,
             rootindex=torch.LongTensor([int(data['rootindex'])]))

def collate_fn(data):
    return data

class BiGraphDataset(Dataset):
    # fold_x_train 存的是eid  treeDic {eid:{index:{parent:,vec:}},...}
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, tddroprate=0, budroprate=0,
                 data_path=os.path.join('../..', '..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate
    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id = self.fold_x[index]
        data = np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']#边集
        #那比例进行DropEdge(GSL)
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        #bu是边集转置
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        rootPosition = torch.tensor(data['rootPosition'])
        word_position = torch.tensor(data['word_position'])
        rootNoEnhancementFeat = torch.FloatTensor(data['root']).reshape([1,config.source_length,config.embedding_dim])
        wordNoEnhancementFeat = torch.FloatTensor(data['x'])

        #进行positionEmbedding  后续用于Transformer操作  这里不需要池化
        rootTransformerFeat = rootNoEnhancementFeat
        wordTransformerFeat = wordNoEnhancementFeat
        #后续用于GCN处理 需要进行max-pooling或avg-pooling等操作
        value,_ = rootNoEnhancementFeat.max(axis=1)
        rootNoEnhancementFeat = value
        value, _ = wordNoEnhancementFeat.max(axis=1)
        wordNoEnhancementFeat = value
        userFeat = torch.FloatTensor(data['x_user'])
        return Data(x=wordNoEnhancementFeat,userFeat = userFeat,
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=rootNoEnhancementFeat,transformerRoot = rootTransformerFeat,
                    transformerWord = wordTransformerFeat,positionRoot = rootPosition,positionWord = word_position,
                    rootindex=torch.LongTensor([int(data['rootindex'])]), time = torch.LongTensor(data['x_time']))


class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic, lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('../..', '..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate
        self.posEncoder = PositionEncoder(config.source_length)

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]

        rootPosition = torch.tensor(data['rootPosition'])
        word_position = torch.tensor(data['word_position'])
        rootFeat = torch.LongTensor(data['root'])
        rootPosEmd = self.posEncoder(rootPosition)
        wordFeat = torch.LongTensor(data['x'])
        wordPosEmd = self.posEncoder(word_position)
        rootFeat += rootPosEmd
        wordFeat += wordPosEmd

        return Data(x=wordFeat,
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=rootFeat,
             rootindex=torch.LongTensor([int(data['rootindex'])]))
