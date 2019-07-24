# -*- coding: utf-8 -*-
import utils
import gol
from collections import Counter

def count_prescript(series_two, list_name, group_list_word):
    """
    计算症状团在药方数据中出现的次数
    :param series:
    :param list_name:
    :param group_list_word:
    :return:
    """
    group_list = utils.word_2_num(list_name, group_list_word)
    series_list,index_list = series_2_list(series_two, list_name)
    # utils.save_pickle('data.txt',series_list)
    count_list = []
    id_list = []
    for group in group_list:
        count = 0
        item_li = []
        for i,item in enumerate(series_list):
            if (utils.is_in(group, item)):
                # 如果该项包括聚类团
                item_li.append(index_list[i])
                count += 1
        count_list.append(count)
        id_list.append(item_li)
    '''对每个症状团对应的药方进行排序'''
    for list_index,item_li in enumerate(id_list):
        # query_rows = series_two.query('序号==' + str(li))
        if len(item_li)<3: continue
        query_rows = series_two.loc[series_two['序号'].isin(item_li)]
        med_all = []
        '''汇总该类药方所有药物'''
        for index,row in query_rows.iterrows():
            med = row['处方'].split('、')
            med_all.extend(med)
        count_result = Counter(med_all)
        med_list = [x for x in count_result if count_result[x]>1]      # 词频大于1的药物
        count_dict = {}
        for index,row in query_rows.iterrows():
            med = row['处方'].split('、')
            id = row['序号']
            count_dict[id] = len([x for x in med if x in med_list])     # 每个方剂包含的重要药物个数
        sorted_id = sorted(count_dict.items(),key=lambda x:x[1],reverse = True)    #方剂排序
        id_list[list_index] = [x[0] for x in sorted_id]
    data = utils.write_csv(['聚类', '数字', '数量','id'], 'data/count.csv', group_list_word, group_list, count_list,id_list)  # 保存的csv无法提取list
    utils.save_pickle('group_count.pkl',data)
    return data

def calculate(series, list_name, group_list_word):
    """
    统计显示每个药方对应的团
    :param series:
    :param list_name:
    :param group_list_word:
    :return:
    """
    group_list = utils.word_2_num(list_name, group_list_word)
    series_list,index_list = series_2_list(series, list_name)
    pattern_list = []
    for item in series_list:
        pattern = []
        for group in group_list:
            if (utils.is_in(group, item)):
                pattern.append(group)
        pattern_list.append(pattern)
    series_list = utils.num_2_word(list_name, series_list)
    pattern_list = utils.num_2_word(list_name, pattern_list)
    utils.write_csv(['主治', '功能团'], 'data/pattern.csv', series_list, pattern_list)


def series_2_list(series, list_name):
    """
    文件内方法，series格式的主治转为list格式的num
    :param series:
    :param list_name:
    :return:
    """

    dic_map = utils.gene_dic_2(gol.get_value('SYNONYM_FILE'))
    word_map = dict(zip(list_name, range(len(list_name))))  # '症状'：编号
    series_list = []
    index_list = []
    for indexs in series.index:
        item_str = series.loc[indexs,'symptom']
        item_list = item_str.strip().split()
        li = list(map(lambda x: word_map[dic_map[x]], item_list))
        li.sort()
        series_list.append(li)
        index_list.append(series.loc[indexs,'序号'])
    # series_list = utils.delete_duplicate(series_list)#从源头删除，修改原文件，不适用该去重
    return series_list,index_list


