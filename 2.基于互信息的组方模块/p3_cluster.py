# -*- coding: utf-8 -*-

import utils
import pandas as pd
import itertools
import copy
from timeHelper import clock
import numpy as np


# [[a,b], [a,c], [a,d]]变为[b, c, d]
def duplicate_removal(relatives_list, list_name):
    result = []
    for item in relatives_list:
        new2 = []
        for n in item:
            for q in n:
                new2.append(q)
        guo_du = list(set(new2))
        guo_du.sort(key=new2.index)
        if guo_du:
            guo_du.remove(list_name[relatives_list.index(item)])
        result.append(guo_du)
    return result


def cluster_main2(relatives_list, list_name):
    list_qyt = duplicate_removal(relatives_list, list_name)
    # 使用数字代替列表中的项
    list_num = utils.word_2_num(list_name, list_qyt)
    for group_num in range(11, 21):
        new_list = utils.cut_by_num(list_num, group_num)
        list_num2 = del_by_correlation(new_list)
        reWord = utils.num_2_word(list_name, list_num2)
        # 创建二元组
        doubleSet = create_double_set(list_num2)
        max_num, bestSet = merge_loop(doubleSet, list_name, 'data/group' + str(group_num) + '.csv')
        # 信息利用率
        print(max_num, '/', group_num, '=', max_num / group_num)

    # tribleSet = mergeGroup(doubleSet)
    # triWord = num_2_word(list_name,tribleSet)
    # forthSet = mergeGroup(tribleSet,doubleSet)
    # fifth_set = mergeGroup(forthSet,doubleSet)
    # max,bestSet = merge_loop(doubleSet)


def merge_loop(double_set, list_name, file=None):
    """
    进行团合并操作，循环直到不能合并
    :param double_set:
    :return:团成员最大数，最终的团
    """
    bestSet = set()
    oldSet = double_set
    num_list = []
    count_list = []
    group_list = []
    while len(oldSet) > 0:
        print('成员数:', len(list(oldSet)[0]))
        print('个数:', len(oldSet))
        print(oldSet)
        num_list.append(len(list(oldSet)[0]))
        count_list.append(len(oldSet))
        group_list.append(oldSet)
        bestSet = oldSet
        oldSet = merge_group(oldSet, double_set)
    if file is not None:
        group_list = utils.num_2_word(list_name, group_list)
        utils.write_csv(['成员数', '个数', '团'], file, num_list, count_list, group_list)
        utils.save_pickle(file + '.pkl', group_list)
    return len(list(bestSet)[0]), bestSet


def merge_group(old_set, double_set=None):
    """
    合并亲友团，亲友团成员+1
    方法：输入亲友团成员数量为n，所有亲友团相互比较，如果有n-1个成员相同，比较剩余两个不同成员的强相关性，如果强相关，组成新团
    :param old_set:输入的亲友团
    :param double_set:成员数为2的亲友团，用来验证是否强相关
    :return:得到亲友团成员数比输入亲友团成员数多1
    """
    if double_set is None:
        double_set = old_set
    new_set = set()
    old_list = list(old_set)
    item_len = len(old_list[0])
    for comb in itertools.combinations(old_list, 2):
        set1 = set(comb[0])
        set2 = set(comb[1])
        if len(set1 & set2) == item_len - 1:
            otherSet = (set1 | set2) - (set1 & set2)
            other_tup = tuple(sorted(list(otherSet)))
            if other_tup in double_set:
                new_set.add(tuple(sorted(list(set1 | set2))))
    return new_set


def del_by_correlation(new_list):
    """
    根据强相关，删除不相关项。 强相关：1的list里有2，且2的list有1
    :param new_list:
    :return:
    """
    # 创建副本，便于进行删除操作
    list_num2 = copy.deepcopy(new_list)
    # 保留强相关的项，其余删除
    for i, row in enumerate(new_list):
        for numb in row:
            if i not in new_list[numb]:
                list_num2[i].remove(numb)
    return list_num2


def create_double_set(list_num2):
    """
    创建二元组的set
    :param list_num2:
    :return:
    """
    double_set = set()
    for i, row in enumerate(list_num2):
        for item in row:
            two = [i, item]
            two.sort()
            double_set.add(tuple(two))
    return double_set

def information_utilization():
    for key in range(3,11):
        group = utils.load_pickle('data/group'+str(key)+'.csv.pkl')
        core = 0
        count=0
        group_add = 0
        print(key)
        for i in range(len(group)):
            num = len(group[i])
            print(i+2,num)
            group_add+=i+2
            count+=num
            core +=(i+2)*num
        print(core/(count))
if __name__ == "__main__":
    # relatives_list = utils.load_pickle('relatives_list.txt')
    # list_name = utils.load_pickle('list_name.txt')
    # # 修改后的主程序
    # cluster_main2(relatives_list, list_name)
    information_utilization()
