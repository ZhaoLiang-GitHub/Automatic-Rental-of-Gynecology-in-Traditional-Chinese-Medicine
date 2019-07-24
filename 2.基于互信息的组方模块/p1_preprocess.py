# -*- coding: utf-8 -*-
# from treat import *
import re
import pandas as pd
import numpy as np
import itertools
import operator
import utils
import gol

# 2.2 按value排列字典，这里指频率降序排列药物，并将药物名称和频率保存各自保存到list
def dic_list(dic):
    list_name, list_frequecy = [], []
    reversed_list = sorted(dic.items(), key=lambda x: x[1], reverse=True)
    for i in reversed_list:
        list_name.append(i[0])
        list_frequecy.append(i[-1])
    return list_name, list_frequecy


# 4. 药物两两组合，计算组合频率，返回字典格式
def combinations_dic_2(one_hot_data):
    """
    两两组合的频数，(0,1):5
    :param one_hot_data: type;dataFrame
    :return: type:dict
    """
    # row_len = one_hot_data.iloc[:,0].size
    row_len = len(one_hot_data)
    cols = list(one_hot_data.columns)
    wordMap = dict(zip(cols, range(len(cols))))
    combinations_fre = {}
    for i in range(row_len):
        row = one_hot_data.iloc[i, :]
        list_word = list(row[row == 1].index)
        combinations = list(itertools.combinations(list_word, 2))
        for item in combinations:
            pre, suf = item
            item = wordMap[pre], wordMap[suf]
            combinations_fre[item] = (combinations_fre[item] if item in combinations_fre else 0) + 1
    key_list = sorted(combinations_fre.items(), key=operator.itemgetter(0))
    combinations_fre = dict(key_list)
    return combinations_fre

def create_list():
    """"
    由同义词集返回所有同义词列表
    :return: word_all：所有同义词；word_map：{词：首义词}
    """
    with open(gol.get_value('SYNONYM_FILE') ,'r',encoding=gol.get_value('CODE')) as f:
        file_list = f.readlines()
        word_num = len(file_list)
        word_list = []#首义词list
        word_map = {}#词：首义词
        word_all = []#所有词list
        for line in file_list:
            line_list = line.strip().split()
            first_word = line_list[0]#首义词
            word_list.append(first_word)
            for word in line_list:
                word_map[word] = first_word
                if(word in word_all):
                    print('重复词：'+word)
                else:
                    word_all.append(word)
    return word_all,word_map


def write_symptom(word_all,word_map):
    """

    :param word_all:
    :param word_map:
    :return:
    """
    word_all = sorted(word_all, key=len, reverse=True)#按照字符数从大到小排序
    df = pd.read_csv('data/prescription_10000.csv',encoding=gol.get_value('CODE'))
    df.rename(columns={'id': '序号', 'title': '方名', '标准药物名称': '处方'}, inplace=True)  #['id','title','主治','','药物组成'] ['序号','方名','主治','symptom','处方']
    #新加一列
    col_name = df.columns.tolist()
    col_name.insert(8, 'symptom')
    df = df.reindex(columns=col_name)
    for index, row in df.iterrows():
        function = row['主治']
        small_list = max_match2(function,word_all)
        small_list = [word_map[x] for x in small_list]
        df.loc[index,'symptom'] = ' '.join(small_list)
    df.to_csv(gol.get_value('ENTITY_FILE'),encoding=gol.get_value('CODE'),index=0,columns=['name'])

def write_symptom2(word_map2):
    """
    使用similar_words.csv第一次匹配，替换，再用同义词进行第二次替换
    :return:
    """
    # 创建替换词典
    with open('similar_word/split.txt','r',encoding='utf8') as f:
        line_list = f.readlines()
    word_map = {}
    for line in line_list:
        line = line.strip()
        line_sep = line.split(',')
        word_map[line_sep[0]]=line_sep[1].split()
    #使用all_words.txt第一次匹配
    with open('similar_word/all_words.txt','r',encoding='utf8') as f:
        word_all = f.readlines()
    word_all = [x.strip() for x in word_all]
    #处理原始文件
    df = pd.read_csv('data/prescription_10000.csv', encoding=gol.get_value('CODE'),usecols=['id','title','处方来源','药物组成','主治','标准药物名称'])
    df.rename(columns={'id': '序号', 'title': '方名', '标准药物名称': '处方','处方来源':'出处'},inplace=True)
    # 新加一列
    col_name = df.columns.tolist()
    col_name.insert(6, 'symptom_1')
    col_name.insert(6, 'symptom_2')
    col_name.insert(6, 'symptom')
    df = df.reindex(columns=col_name)
    for index, row in df.iterrows():
        function = row['主治']
        small_list1 = max_match2(function, word_all)
        df.loc[index, 'symptom_1'] = ' '.join(small_list1)
        small_list2 = []
        for word in small_list1:
            if(word in word_map):
                small_list2.extend(word_map[word])
            else:
                small_list2.append(word)
        df.loc[index, 'symptom_2'] = ' '.join(small_list2)

        small_list3 = [word_map2[x] if x in word_map2 else '' for x in small_list2]
        df.loc[index, 'symptom'] = ' '.join(small_list3)
    df.to_csv(gol.get_value('ENTITY_FILE'), encoding=gol.get_value('CODE'), index=0,columns=['序号','方名','主治','symptom_1','symptom_2','symptom','药物组成','处方','出处'])

def max_match(str,word_all):
    '''
    使用最大正相匹配的方法来标记词典，即有限匹配长的字典,不考虑重复词，不考虑词重叠
    :param str:
    :param word_all:
    :return:
    '''
    word_dic = {}
    for word in word_all:
        index = str.find(word)
        if(index!=-1):
            word_dic[index] = word
    keys = list(word_dic.keys())
    keys.sort()
    return [word_dic[key] for key in keys]


def max_match2(file_str,word_all):
    '''
    使用最大正相匹配的方法来标记词典，即有限匹配长的字典,考虑词重叠
    :param str:
    :param word_all:
    :return:
    '''
    word_dic = {}
    tag_list = [0 for x in file_str]
    char_list = list(file_str)
    for word in word_all:
        start = 0 #字符串索引
        first = 1 #是否是第一次查找
        word_len = len(word)
        while (True):
            if (first):
                start = file_str.find(word)
                first = 0
            else:
                start = file_str.find(word, start + word_len)
            if (start == -1):
                break
            # 检查是否有本词典的其他标记
            flag = 0
            for i in range(word_len):
                if(tag_list[start+i]==1):#有标记
                    flag = 1
            if (flag):
                break
            for i in range(0, word_len):
                tag_list[start + i]=1
            word_dic[start] = word

    keys = list(word_dic.keys())
    keys.sort()
    result = utils.delete_duplicate([word_dic[key] for key in keys])
    return result

if __name__ == "__main__":
    #初始化全局变量
    gol._init()
    ENTITY_FILE = gol.set_value('ENTITY_FILE','data/symptom_entity.csv')    #添加实体的方剂文件
    CODE = gol.set_value('CODE','UTF-8')    #code
    SYNONYM_FILE = gol.set_value('SYNONYM_FILE','data/clean2.txt') #同义词

    #读取原方剂文件，抽取实体，重新保存一份文件：ENTITY_FILE
    a,word_map = create_list()
    # write_symptom(word_all,word_map)
    #抽取实体，保存方剂文件的简化信息
    # extract_entity()
    write_symptom2(word_map)
