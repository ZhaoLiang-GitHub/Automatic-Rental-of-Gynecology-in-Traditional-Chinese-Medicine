# -*- coding: utf-8 -*-
import pandas as pd
from math import log
import utils


def calculate_correlation(combinations_list, combinations_fre, list_fre):
    correlation = []  # 关联度系数
    for i in range(len(combinations_list)):
        flag_1, flag_2, flag_3 = 1, 1, 1
        pre, suf = combinations_list[i]
        H_pre = -list_fre[pre] * log(list_fre[pre]) - (1 - list_fre[pre]) * log((1 - list_fre[pre]))
        H_suf = -list_fre[suf] * log(list_fre[suf]) - (1 - list_fre[suf]) * log((1 - list_fre[suf]))
        param_1 = list_fre[pre] - combinations_fre[i]
        param_2 = list_fre[suf] - combinations_fre[i]
        param_3 = 1 + combinations_fre[i] - list_fre[pre] - list_fre[suf]
        if param_1 == 0:
            flag_1 = 0
            param_1 = 1
        if param_2 == 0:
            flag_2 = 0
            param_2 = 1
        if param_3 == 0:
            flag_3 = 0
            param_3 = 1
        H_pre_suf = -combinations_fre[i] * log(combinations_fre[i]) - flag_1 * param_1 * log(
            param_1) - flag_2 * param_2 * log(param_2) - flag_3 * param_3 * log(param_3)
        result = H_pre + H_suf - H_pre_suf
        # result = combinations_fre[i] * log(combinations_fre[i]/(list_fre[pre]*list_fre[suf]))
        correlation.append(result)
    return correlation


def relatives(list_name, data, relatives_num):
    relatives_list = []  # 药物亲友团
    length = data.shape[0]
    for item in list_name:
        list_ = []
        for i in range(length):
            words = data['药物'][i]
            if item in words:
                list_.append(words)
        relatives_list.append(list_)
    return utils.cut_by_num(relatives_list, relatives_num)


def relatives_2(list_name, data, relatives_num):
    """
    根据互信息得到每项的亲友团
    :param list_name:所有词的list
    :param data:dataFrame，{组合，关联度系数}
    :param relatives_num:限制亲友团个数
    :return:[[]] 所有项的亲友团
    """
    relatives_list = [[] for i in range(len(list_name))]
    length = data.shape[0]
    for i in range(length):
        words = data['组合'][i]
        # words = data['药物'][i]
        pre_index = list_name.index(words[0])
        relatives_list[pre_index].append(words)
        suf_index = list_name.index(words[1])
        relatives_list[suf_index].append(words)
    return utils.cut_by_num(relatives_list, relatives_num)

def save_relatives(list_name,relatives_list):
    relative_list = [[] for x in list_name]
    for i,item in enumerate(relatives_list):
        name = list_name[i]
        for j in item:
            if(name!=j[0]):
                relative_list[i].append(j[0])
            if(name!=j[1]):
                relative_list[i].append(j[1])
    utils.write_csv(['症状', '亲友团'], 'data/relatives.csv', list_name, relative_list)

if __name__ == "__main__":
    dd = calculate_correlation([(0,1)], [0.1], [0.2,0.3])
    dd2 = calculate_correlation([(1, 0)], [0.1], [0.2, 0.3])
    list_name = utils.load_pickle('list_name.txt')
    list_fre = utils.load_pickle('list_fre.txt')
    combinations_list = utils.load_pickle('combinations_list.txt')
    combinations_fre = utils.load_pickle('combinations_fre.txt')
    correlation = calculate_correlation(combinations_list, combinations_fre, list_fre)
    # combinations_name = comb_names(list_name,combinations_list)
    combinations_name = utils.num_2_word(list_name, combinations_list)
    column_1 = pd.Series(combinations_name, name='组合')
    column_2 = pd.Series(correlation, name='关联度系数')
    data = pd.concat([column_1, column_2], axis=1)
    data = data.sort_values(by='关联度系数', ascending=False)
    data.to_csv('rel2.csv', index=False, encoding='utf-8')
    # relatives_list = relatives(list_name, data, 5)#舍弃改方法
    relatives_list = relatives_2(list_name, data, 8)
    utils.save_pickle('relatives_list.txt', relatives_list)
