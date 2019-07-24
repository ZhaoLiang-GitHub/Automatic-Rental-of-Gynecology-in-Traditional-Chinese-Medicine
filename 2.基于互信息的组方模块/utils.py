# -*- coding: utf-8 -*-
import pickle
import sys
import pandas as pd
from functools import reduce
import copy

def save_pickle(file_name, input_data):
    with open(file_name, 'wb') as f:
        pickle.dump(input_data, f)


def load_pickle(file_name):
    with open(file_name, 'rb') as f:
        output_data = pickle.load(f)
    return output_data


def word_2_num(list_name, list_word):
    wordMap = dict(zip(list_name, range(len(list_name))))
    # list_num = [[wordMap[word] if word in wordMap else word for word in i] for i in list_word]
    # 使用迭代方法来写
    if isinstance(list_word, (list, tuple, set)):
        newList = list()
        for item in list_word:
            newList.append(word_2_num(list_name, item))
        return newList
    else:
        return wordMap[list_word]

def num_2_word(list_name, list_num):
    """
    通过list_name生成（0：当归）这样的map，把list_num转成对应的中文
    :param list_name: 所有中药名的list
    :param list_num:以数字组成的list/set/tuple（任意层）
    :return:list_num对应的中文
    """
    # wordMap = dict(zip(range(len(list_name)),list_name))
    wordMap = dict(enumerate(list_name))
    # 使用迭代方法来写
    if isinstance(list_num, (list, tuple, set)):
        newList = list()
        for item in list_num:
            newList.append(num_2_word(list_name, item))
        return newList
    else:
        return wordMap[list_num]


def cut_by_num(list_double, max_num):
    """
    对list每个子list进行删除操作，保留前max_num项
    :param list_double:
    :param max_num:
    :return:
    """
    new_list = list()
    for i, row in enumerate(list_double):
        if len(row) > max_num:
            new_list.append(row[:max_num])
        else:
            new_list.append(row)
    return new_list


def write_csv(name_list, file_path, *args):
    if len(name_list) != len(args):
        print('list长度不对应！')
        sys.exit(1)
    series_list = []
    for i, name in enumerate(name_list):
        column = pd.Series(args[i], name=name)
        series_list.append(column)
    data = pd.concat(series_list, axis=1)
    # data = data.sort_values(by=name_list[1], ascending=False)
    data.to_csv(file_path, index=False, encoding='utf-8')
    return data

def is_in(sub_list,parent_list):
    """
    判断parent_list是否包含sub_list
    :param sub_list:
    :param parent_list:
    :return:
    """
    d = [False for item in sub_list if item not in parent_list]
    return not d

def is_in2(sub_list,parent_list):
    """
    判断parent_list的元素是否包含sub_list
    :param sub_list:
    :param parent_list:
    :return:
    """
    for item_list in parent_list:
        if(is_in(sub_list,item_list)):
            return True
    return False
def gene_dic(path):
    """
    {'词根0':['词','词','词']}
    根据同义词表创建dic
    :return:
    """
    combine_dict = {}
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            seperate_word = line.strip().split()
            combine_dict[seperate_word[0]] = seperate_word
    return combine_dict


def gene_dic_2(path='data/tongyici_3.txt'):
    """
    {'词':'词根0','词':'词根0','词':'词根1'}
    :param path:
    :return:
    """
    combine_dict = {}
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            words = lines[i].strip().split()
            for word in words:
                # combine_dict[word]=i
                combine_dict[word] = words[0]
    return combine_dict

def gene_dic_3(path):
    """
    作废：词的序号与主程序不对应
    {'词':0,'词':0,'词':1}
    :param path:
    :return:
    """
    combine_dict = {}
    with open(path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for i in range(len(lines)):
            words = lines[i].strip().split()
            for word in words:
                combine_dict[word]=i
    return combine_dict
def  delete_duplicate(old_list):
    """
    列表去重
    """
    func = lambda x,y:x if y in x else x + [y]
    result = reduce(func, [[], ] + old_list)
    return result
def group_clean(pkl_file):
    """
    把不同成员数的亲友团整理在一起
    :param pkl_file:
    :return:
    """
    group = load_pickle(pkl_file)
    all_list = []
    for i in range(len(group) - 1, 0, -1):
        item = list(group[i])
        member_num = len(item[0])
        new_item = copy.deepcopy(item)#用来进行删除操作的复制项
        for item_li in item:
            for li in all_list:
                if(is_in(item_li,li) and item_li in new_item):
                    new_item.remove(item_li)
                    break
        all_list.extend(new_item)
    return all_list

def group_clean2(pkl_file):
    """
    不做删除
    :param pkl_file:
    :return:
    """
    group = load_pickle(pkl_file)
    all_list = []
    for item in group:
        all_list.extend(item)
    return all_list


if __name__ == '__main__':
    print(is_in([1,2],[1,2,3]))
