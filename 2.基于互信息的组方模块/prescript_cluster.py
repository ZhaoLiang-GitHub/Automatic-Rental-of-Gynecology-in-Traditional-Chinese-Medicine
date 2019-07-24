# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import itertools
import copy
import utils
import p1_preprocess as clus1
import p2_relatives as clus2
import p3_cluster as clus3
import p4_validate as clus4
import p5_recommend as clus5
import p6_diversity as clus6
import p7_med_replace as clus7
import whoosh_op as who
import gol

def init_program():
    """
    初始化程序：初始化变量，方剂数据格式化，聚类算法
    :return:
    """
    '''初始化全局变量'''
    gol._init()
    ENTITY_FILE = gol.set_value('ENTITY_FILE', 'data/symptom_entity.csv')  # 添加实体的方剂文件
    CODE = gol.set_value('CODE', 'UTF-8')  # code
    SYNONYM_FILE = gol.set_value('SYNONYM_FILE', 'data/clean2.txt')  # 同义词
    '''whoosh初始化索引'''
    who.get_index(ENTITY_FILE, 'index/')
    # SYMPTOM_LIST = gol.set_value('SYMPTOM_LIST','')   #症状列表
    # GROUP_COUNT = gol.set_value('GROUP_COUNT','') #症状团
    # word_all, word_map = clus1.create_list()
    # clus1.write_symptom(word_all,word_map)
    df = pd.read_csv(ENTITY_FILE, encoding='utf8')
    series_two = pd.DataFrame(df, columns=['序号', '方名', '主治', 'symptom', '处方'])
    series_two.dropna(inplace=True)  # 删掉nan
    series = series_two['symptom']
    utils.save_pickle('series.pkl', series_two)
    # series = series.replace(np.nan, '')
    # series = series.reindex()#重新索引，使索引连续，该方法不能用
    new_dic = utils.gene_dic(SYNONYM_FILE)
    new_dic_2 = utils.gene_dic_2(SYNONYM_FILE)
    # 创建dataFrame存放onehot
    df = pd.DataFrame(np.zeros((len(series), len(new_dic))), columns=new_dic.keys())
    dd = []
    count = 0  # 因为series的索引和df不同，引入count变量，做为df的索引
    for indexs in series.index:
        item_str = series[indexs]
        # if item_str == '':
        #     continue
        item_list = item_str.strip().split()
        for item in item_list:
            if item in new_dic_2:
                df[new_dic_2[item]].loc[count] = 1
            else:
                dd.append(item)
                print('未匹配到：', item)  # 输出没有匹配的字符
        count += 1
    dd = list(set(dd))
    dd.sort()
    with open('dd.txt', 'w', encoding='utf8') as f:
        f.writelines([x + '\n' for x in dd])
    # 删除没有任何匹配的词列
    max_value = df.max()
    drop_list = list(max_value[max_value == 0].index)
    df = df.drop(drop_list, axis=1)
    # we = df.columns.size
    # 计算所有症状中每个词的频数，排序
    count_dic = dict(df.sum())
    list_name, list_frequency = clus1.dic_list(count_dic)
    utils.save_pickle('list_name.txt', list_name)
    df = df.ix[:, list_name]  # 按照词频对列重新排序
    utils.save_pickle('vector.pkl', df)
    # 两两组合的频数，排序
    combinations_dic_fre = clus1.combinations_dic_2(df)
    combinations_list, combinations_frequency = clus1.dic_list(combinations_dic_fre)
    # 每个词的频率，每个两两组合的频率
    row_len = df.iloc[:, 0].size
    # list_fre = [i/sum(list_frequency) for i in list_frequency]
    list_fre = [i / row_len for i in list_frequency]
    # combinations_fre = [i/sum(combinations_frequency) for i in combinations_frequency]
    combinations_fre = [i / row_len for i in combinations_frequency]

    '''2、计算互信息'''
    correlation = clus2.calculate_correlation(combinations_list, combinations_fre, list_fre)
    combinations_name = utils.num_2_word(list_name, combinations_list)  # num_2_word 数字转文字
    data = utils.write_csv(['组合', '关联度系数'], 'data/correlation.csv', combinations_name, correlation)
    # 得到每个症状的亲友团list
    relatives_list = clus2.relatives_2(list_name, data, 15)
    clus2.save_relatives(list_name, relatives_list)
    '''3、亲友团聚类'''
    clus3.cluster_main2(relatives_list, list_name)
    # 对每个group进行合并，得到最终的聚类结果
    group_5_all = utils.group_clean2('data/group5.csv.pkl')
    group_6_all = utils.group_clean2('data/group6.csv.pkl')
    group_7_all = utils.group_clean2('data/group7.csv.pkl')
    group_8_all = utils.group_clean2('data/group8.csv.pkl')
    utils.write_csv(['group8', 'group7', 'group6', 'group5'], 'data/group_all.csv', group_8_all, group_7_all,
                    group_6_all, group_5_all)
    '''4、建立亲友团和方剂的映射'''
    # group_count = clus4.count_prescript(series_two,list_name,group_10_all)
    # utils.save_pickle('group_count.pkl',group_count)
    # clus4.calculate(series_two,list_name,group_10_all)


def init_parameter():
    '''初始化全局变量'''
    gol._init()
    ENTITY_FILE = gol.set_value('ENTITY_FILE', 'data/symptom_entity.csv')  # 添加实体的方剂文件
    CODE = gol.set_value('CODE', 'UTF-8')  # code
    SYNONYM_FILE = gol.set_value('SYNONYM_FILE', 'data/clean2.txt')  # 同义词

if __name__ == '__main__':
    init_parameter()
    # 初始化程序，执行一次
    # init_program()
    # 医案数据集等相关测试程序
    # clus5.recommend_test()
    # 输入症状进行推荐
    '''
    不同输入类型：
    ['产后']
    ['产后','胸烦','饮食不下','恶露未尽']
    ['带下', '腹中疼痛', '色黄' ,'臭气' ,'胸烦','心神多躁','口苦','小便短赤','舌质红','舌苔黄腻','脉滑']
    ['汗出', '眩晕', '心慌', '不睡', '舌质红', '舌苔花白', '无力']
    '''
    insert = ['带下' ,'瘙痒', '流产', '出血', '色黄', '心神多躁', '不安', '口干', '小便短赤', '大便结燥', '舌质红', '舌苔黄腻', '脉滑']
    num,result = clus5.recomment_main(insert)
    #多样性
    result_2 = clus6.deal_diversity(result)
    # 药物替换
    # 对药物进行替换,如：对推荐列表中的第一个药物进行替换
    clus7.med_replace_main(result[0],'黄连')
