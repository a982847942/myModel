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
        self.vector = []
        self.parent = None

def str2matrix(Str):  # str = index:wordfreq index:wordfreq
    wordIndex =  []
    for pair in Str.split(' '):
        index=int(float(pair))
        wordIndex.append(index)
    return  wordIndex

def constructMat(tree):
    index2node = {}
    for i in tree:
        # 遍历事件树 i 即事件树中当前结点的index
        node = Node_tweet(idx=i)
        index2node[i] = node
    for j in tree:
        indexC = j
        indexP = tree[j]['parent']
        nodeC = index2node[indexC]

        userVector  = tree[j]['userVec']
        nodeC.vector = str2matrix(userVector)
        # nodeC.word = wordFreq
        ## not root node ##
        if not indexP == 'None':
            nodeP = index2node[int(indexP)]
            nodeC.parent = nodeP
            nodeP.children.append(nodeC)
        ## root node ##
        else:
            root = nodeC
            rootindex=indexC-1
            rootUserFeat=nodeC.vector
            textVector = tree[j]['textVec']
            root_textVec = str2matrix(textVector)

    rootTextfeat = np.zeros([config.source_length,config.embedding_dim])
    if len(root_textVec)>0:
        token_feat = np.zeros([config.source_length, config.embedding_dim])
        if len(root_textVec) >= config.source_length:
            token_feat = config.glove.vectors[root_textVec[0:config.source_length], :]
        else:
            token_feat[0:len(root_textVec), :] = config.glove.vectors[root_textVec, :]

        if(type(token_feat) == torch.Tensor):
            value,index = token_feat.max(axis=0)
        else:
            value = token_feat.max(axis=0)
        rootTextfeat = value

    ## 3. convert tree to matrix and edgematrix
    matrix=np.zeros([len(index2node),len(index2node)])
    raw=[]
    col=[]
    x_index=[]
    edgematrix=[]
    for index_i in range(len(index2node)):
        for index_j in range(len(index2node)):
            if index2node[index_i+1].children != None and index2node[index_j+1] in index2node[index_i+1].children:
                matrix[index_i][index_j]=1
                raw.append(index_i)
                col.append(index_j)
        #1....len(传入的每一个事件树中结点数)
        # x_word.append(index2node[index_i+1].word)#[[],[],[],[]] 词频矩阵
        x_index.append(index2node[index_i+1].vector)#[[],[],[],[]] 词典中的序号
    '''
    边集 i 是 j 的父亲
    [i..]
    [j..] 
    '''
    edgematrix.append(raw)
    edgematrix.append(col)
    return  x_index, edgematrix,rootUserFeat,rootTextfeat,rootindex

def main():
    treePath = os.path.join(cwd, 'process/pheme/PHEMEtree.txt')
    print("reading PHEME tree:")
    treeDic = {}
    for line in open(treePath):
        line = line.rstrip()
        eid, indexP, indexC,userVec,Vec = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2]), line.split('\t')[3],line.split('\t')[4]
        # print(Vec)
        if not treeDic.__contains__(eid):
            treeDic[eid] = {}
        #事件eid在当前事件图中的第j个结点的父亲节点（当前事件图中）是indexP，当前节点的词向量是Vec
        '''
        {'eid':{'indexC':{}},eid:{{}}}
        '''
        treeDic[eid][indexC] = {'parent': indexP, 'userVec': userVec,'textVec':Vec}
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
           #x_index 用户特征 edgematrix 边集 rootUserFeat,rootTextfeat 根结点的用户特征 文本特征 rootindex根结点下标
            x_index, edgematrix,rootUserFeat,rootTextfeat,rootindex = constructMat(event)
            # 输入词向量
            # x_x = getfeature(x_index)
            rootUserFeat,rootTextfeat,x_index, tree,  rootindex, y = np.array(rootUserFeat),np.array(rootTextfeat),np.array(x_index),np.array(edgematrix), np.array(
                rootindex), np.array(y)
            np.savez(os.path.join(cwd,'process/pheme/PHEMEUsergraph/'+id+'.npz'), x=x_index,rootUserFeat=rootUserFeat,rootTextfeat = rootTextfeat,edgeindex=tree,rootindex=rootindex,y=y)
            return None

        x_index, edgematrix,rootUserFeat,rootTextfeat,rootindex = constructMat(event)
        # x_x = getfeature(x_index)
        return rootTextfeat,rootUserFeat, edgematrix, x_index, [rootindex]

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
