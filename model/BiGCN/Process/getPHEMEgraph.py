# -*- coding: utf-8 -*-
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import os
import config
import torch
cwd = "H:/pythonProject/rumor detection/myModel/"
class Node_tweet(object):
    def __init__(self, idx=None):
        self.children = []
        self.idx = idx
        self.word = []
        self.index = []
        #用户向量
        self.vector = []
        #时间信息
        self.time = None
        self.parent = None

def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordIndex =  []
    for pair in Str.split(' '):
        index=int(pair)
        if index != -1:
            wordIndex.append(index)
    return  wordIndex
def userStr2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordIndex =  []
    for pair in Str.split(' '):
        index=int(float(pair))
        wordIndex.append(index)
    return  wordIndex

def constructMat(id,tree):
    index2node = {}
    for i in tree:
        # 遍历事件树 i 即事件树中当前结点的index
        node = Node_tweet(idx=i)
        #600s 作为一个时间段
        time = int(float(tree[i]['time'])) // config.time_interval
        time = 10 if time > 10 else time
        node.time = time
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]

        # wordFreq, wordIndex = str2matrix(tree[j]['vec'])
        wordIndex  = tree[j]['vec']
        userVector  = tree[j]['userVec']
        nodeC.index = str2matrix(wordIndex)
        nodeC.vector = userStr2matrix(userVector)
        # nodeC.word = wordFreq
        ## not root node ##
        if not indexP == 'None':
            # print("index2node:",index2node)
            # print("indexP:",indexP)
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            root = nodeC
            rootindex=indexC-1
            root_index=nodeC.index
            # root_word=nodeC.word
    # rootfeat = np.zeros([1,config.embedding_dim])
    rootfeat = np.zeros([config.source_length,config.embedding_dim])
    rootPosition = np.zeros([1,config.source_length])
    if len(root_index)>0:
        token_feat = np.zeros([config.source_length, config.embedding_dim])
        # rootfeat[0, np.array(root_index)] = np.array(root_word)
        if len(root_index) >= config.source_length:
            # token_feat = config.glove.vectors[root_index[0:config.source_length], :]
            rootPosition[0,:] = np.arange(0,config.source_length)
            # print(root_index)
            rootfeat[0:config.source_length, :] = config.glove.vectors[root_index[0:config.source_length],:]
        else:
            # token_feat[0:len(root_index), :] = config.glove.vectors[root_index, :]
            rootPosition[0,:len(root_index)] = np.arange(0,len(root_index))
            rootPosition[0,len(root_index):] = config.source_length - 1
            rootfeat[0:len(root_index), :] = config.glove.vectors[root_index,:]

        # if(type(token_feat) == torch.Tensor):
        #     value,index = token_feat.max(axis=0)
        # else:
        #     value = token_feat.max(axis=0)
        # rootfeat = value

    ## 3. convert tree to matrix and edgematrix
    matrix=np.zeros([len(index2node),len(index2node)])
    raw=[]
    col=[]
    # x_word=[]
    x_index=[]
    x_user = []
    edgematrix=[]
    x_time = []
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if (index_i + 1) not in index2node or (index_j + 1) not  in index2node:
                print("index2node:", index2node)
                print("index_i  index_j", index_i, index_j)
                print("id:",id)
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                matrix[index_i][index_j]=1
                raw.append(index_i)
                col.append(index_j)
        #1....len(传入的每一个事件树中结点数)
        # x_word.append(index2node[index_i+1].word)#[[],[],[],[]] 词频矩阵
        x_index.append(index2node[index_i+1].index)#[[],[],[],[]] 词典中的序号
        x_user.append(index2node[index_i+1].vector)
        x_time.append(index2node[index_i + 1].time)
    '''
    边集 i 是 j 的父亲
    [i..]
    [j..] 
    '''
    edgematrix.append(raw)
    edgematrix.append(col)
    #x_word, x_index 相当于（但不是）每个回复的词向量 edgematrix边集  rootfeat,rootindex根结点词向量 、根结点编号
    #x_index 文本特征
    return  x_time,x_index,x_user, edgematrix,rootfeat,rootPosition,rootindex

def getfeature(x_index):
    xfeat = np.zeros([len(x_index),config.source_length, config.embedding_dim])
    # xfeat = np.zeros([len(x_index), config.embedding_dim])
    word_position = np.zeros([len(x_index), config.source_length])
    for i,e_index in  enumerate(x_index):
        # if(e_index is None):continue
        # token_feat = np.zeros([config.source_length,config.embedding_dim])
        if len(e_index) > 0:
            # rootfeat[0, np.array(root_index)] = np.array(root_word)
            if len(e_index) >= config.source_length:
                # token_feat = config.glove.vectors[e_index[0:config.source_length], :]
                word_position[i, :] = np.arange(0, config.source_length)
                xfeat[i,0:config.source_length, :] = config.glove.vectors[e_index[0:config.source_length], :]
            else:
                # token_feat[0:len(e_index),:] = config.glove.vectors[e_index, :]
                xfeat[i,0:len(e_index), :] = config.glove.vectors[e_index, :]
                word_position[i, :len(e_index)] = np.arange(0, len(e_index))
                # word_position[i,len(e_index):] = config.source_length - 1 #直接用0填充

            # xfeat[i, :] = token_feat.max(axis=0)
    return xfeat,word_position


def main():
    treePath = os.path.join(cwd, 'process/pheme/PHEMEtree.txt')
    print("reading PHEME tree:")
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC,time,userVec,Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]), line.split('\t')[3].split(' ')[-1],line.split('\t')[3],line.split('\t')[4]
        # print(Vec)
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        #事件eid在当前事件图中的第j个结点的父亲节点（当前事件图中）是indexP，当前节点的词向量是Vec
        '''
        {'eid':{'indexC':{}},eid:{{}}}
        '''
        treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec,'userVec':userVec,'time':time}
    print('tree no:', len(treeDic))

    labelPath = os.path.join(cwd, "process/pheme/PHEME_id_label.txt")
    print("loading PHEME label:")
    event,y= [],[]
    l1 = l2 = l3 = l4 = 0
    labelDic = {}
    for line in open(labelPath):
        line = line.rstrip()
        eid,label = line.split(' ')[0], line.split(' ')[1]
        labelDic[eid] = int(label)
        y.append(labelDic[eid])
        event.append(eid)
        if labelDic[eid]==0:
            l1 += 1
        if labelDic[eid]==1:
            l2 += 1
        if labelDic[eid]==2:
            l3 += 1
        if labelDic[eid]==3:
            l4 += 1

    print(len(labelDic),len(event),len(y))
    print(l1, l2, l3, l4)
    '''
    {'eid':{'indexC':{}},eid:{{}}}
    treeDic[eid][indexC] = {'parent': indexP, 'vec': Vec}
    tree[eid] 事件字典
    id = eid
    y label
    '''
    def loadEid(event,id,y):
        if event is None:
            return None
        #事件树小于2 不要
        if len(event) < 2:
            return None
        if len(event)>1:
            # x_word, x_index 相当于（但不是）每个回复的词向量 edgematrix边集  rootfeat,rootindex根结点词向量 、根结点编号
            x_time,x_index,x_user, tree, rootfeat,rootPosition, rootindex = constructMat(id,event)
            # 输入词向量
            x_x,word_position = getfeature(x_index)
            x_time,rootfeat, tree, x_x, rootindex, y ,rootPosition,word_position,x_user= np.array(x_time),np.array(rootfeat), np.array(tree), np.array(x_x), np.array(
                rootindex), np.array(y),np.array(rootPosition),np.array(word_position),np.array(x_user)
            np.savez(os.path.join(cwd,'process/pheme/PHEMEgraph/'+id+'.npz'), x_time = x_time,x=x_x,x_user=x_user,root=rootfeat,rootPosition = rootPosition,word_position = word_position,edgeindex=tree,rootindex=rootindex,y=y)
            return None

        x_time,x_index,x_user, tree, rootfeat,rootPosition, rootindex = constructMat(event)
        x_x, word_position = getfeature(x_index)
        return rootfeat, tree, x_x, [rootindex]

    print("loading dataset", )
    results = Parallel(n_jobs=30, backend='threading')(delayed(loadEid)(treeDic[eid] if eid in treeDic else None,eid,labelDic[eid]) for eid in tqdm(event))
    # for eid in event:
    #     ent = treeDic[eid] if eid in treeDic else None
    #     tru = labelDic[eid]
    #     loadEid(ent,eid,tru)
    #     break
    return

if __name__ == '__main__':
    main()
