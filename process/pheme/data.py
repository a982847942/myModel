import json
import os
import time
from copy import copy
from convert_veracity_annotations import *
from util.vocabProcess import tokenize_text

DATA_PATH = 'D:/计算机书籍/研究生/谣言监测/Ma原始数据集/pheme-rnr-dataset18/all-rnr-annotated-threads/'
EVENT_NAME = ['charliehebdo',
              'ebola-essien',
              'ferguson',
              'germanwings-crash',
              'ottawashooting',
              'prince-toronto',
              'putinmissing',
              'sydneysiege']
# 标签转化为 数字
label2Index = {'non-rumor': 0, 'true': 1, 'false': 2, 'unverified': 3}
# 标签逆向字典
index2Label = {'0': 'non-rumor', '1': 'true', '2': 'false', '3': 'unverified'}

def getThread():
    threadId = [] #只记录每一个事件发布者ID
    postId = [] #所有事件涉及的帖子ID集合
    threadTag = {} #事件标签
    structures = {} #每一个事件对应的回复的结构
    users = {} #每一个用户对应的特征
    '''
    posts{'事件ID':{'每一个帖子id':{time,text}}}
    '''
    posts = {} #每一个推文集合（内容和time）
    eventName = os.listdir(DATA_PATH)
    index = 0
    for event in eventName:
        print("curEvent:",event)
        if '.' not in event:
            #'./all-rnr-annotated-threads/charliehebdo/non-rumours/'
            ids = os.listdir(DATA_PATH + '/' + event + '/non-rumours/')
            for id in ids:
                if '.' not in id:
                   # './all-rnr-annotated-threads/charliehebdo/non-rumours/552784600502915072/'
                    path = DATA_PATH + '/' + event + '/non-rumours/' + str(id) + '/'
                    # id  eg:552784600502915072
                    threadId.append(id)
                    postId.append(id)
                    # tag
                    threadTag[id] = 0
                    # structure
                    with open(path + 'structure.json', 'r',encoding='utf-8') as f:
                        content = f.read()
                    structures[id] = json.loads(content)
                    # post
                    post = {}
                    user = {}
                   #'./all-rnr-annotated-threads/charliehebdo/non-rumours/552784600502915072/source-tweets/552784600502915072.json'
                    with open(path + 'source-tweets/' + str(id) + '.json', 'r') as f:
                        content = json.loads(f.read())
                    time = strTime2Timestamp(content['created_at'])
                    post[id] = {
                        'time': time,
                        'text': tokenize_text(content['text'])
                    }
                    user_dict = content['user']
                    user[id] = {
                        'created_at':time,
                        'user':user_dict
                   }
                   # './all-rnr-annotated-threads/charliehebdo/non-rumours/552784600502915072/reactions/'
                    pids = os.listdir(path + 'reactions/')
                    for pfname in pids:
                        if '._' not in pfname and '.json' in pfname:
                            pid = pfname[0:-5]# 去掉.json 保留下每一个事件的响应事件的id
                            postId.append(pid)
                            with open(path + 'reactions/' + str(pfname), 'r') as f:
                                content = json.loads(f.read())
                            time = strTime2Timestamp(content['created_at'])
                            post[pid] = {
                                'time': time,
                                'text': tokenize_text(content['text'])
                            }
                            user_dict = content['user']
                            user[pid] = {
                            'created_at': time,
                            'user': user_dict
                            }
                    posts[id] = post
                    users[id] = user

            ids = os.listdir(DATA_PATH + '/' + event + '/rumours/')
            for id in ids:
                if '.' not in id:
                    path = DATA_PATH + '/' + event + '/rumours/' + str(id) + '/'
                    # id
                    threadId.append(id)
                    postId.append(id)
                    # tag
                    with open(path + 'annotation.json', 'r') as f:
                        annotation = json.loads(f.read())
                        label = convert_annotations(annotation)
                    threadTag[id] = label2Index[label]
                    # structure
                    with open(path + 'structure.json', 'r',encoding='utf-8') as f:
                        content = f.read()
                    structures[id] = json.loads(content)
                    # post
                    post = {}
                    user = {}
                    with open(path + 'source-tweets/' + str(id) + '.json', 'r') as f:
                        content = json.loads(f.read())
                        time = strTime2Timestamp(content['created_at'])
                        post[id] = {
                            'time': time,
                            'text': tokenize_text(content['text'])
                        }
                        user_dict = content['user']
                        user[id] = {
                        'created_at': time,
                        'user': user_dict
                        }
                    pids = os.listdir(path + 'reactions/')
                    for pfname in pids:
                        if '._' not in pfname and '.json' in pfname:
                            pid = pfname[0:-5]
                            postId.append(pid)
                            with open(path + 'reactions/' + str(pfname), 'r', encoding='utf8') as f:
                                content = json.loads(f.read())
                            time = strTime2Timestamp(content['created_at'])
                            post[pid] = {
                                'time': time,
                                'text': tokenize_text(content['text'])
                            }
                            user_dict = content['user']
                            user[pid] = {
                                'created_at': time,
                                'user': user_dict
                            }
                    posts[id] = post
                    users[id] = user
                    if(index == 0):
                        print(post[pid]['text'])
                        print(post[pid]['time'])
                        print(user[pid]['user'])
                        print(user[pid]['created_at'])
                        index += 1
    return threadId, postId, threadTag, structures, posts,users

# 转换字符串时间为时间戳
def strTime2Timestamp(strTime: str):
    #"created_at": "Tue Feb 01 07:40:04 +0000 2011
    temp = strTime.split(' ')
    temp.pop(-2) # 放弃掉不能读入的时区字段
    strTime = ' '.join(temp)
    structureTime = time.strptime(strTime, '%a %b %d %H:%M:%S %Y')
    return time.mktime(structureTime) # 把结构化时间转化成时间戳