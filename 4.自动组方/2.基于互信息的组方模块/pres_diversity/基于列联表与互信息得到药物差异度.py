import pandas as pd
import numpy as np
from itertools import combinations
from math import log,sqrt
import pickle
import time
from collections import Counter


def getonehot(f_list,list):
    '''

    :param f_list:全部属性值的列表
    :param list: 需要转化为onehot的向量
    :return: onehot向量
    '''
    onehot = [0]*len(f_list)
    for i in range(len(f_list)):
        if f_list[i] in list:
            onehot[i] += 1
    # print(all)
    # print(list)
    # print(onehot)
    return onehot

def cascading_table(list1,list2):
    a = 0
    p = 0
    d = 0

    list_sum = list1 + list2
    list_count = Counter(list_sum)
    for i in list_count:
        if (i == 0):
            d = list_count[i]
        elif (i == 1):
            p = list_count[i]
        else:
            a = list_count[i]
    try:
        return p / (a + p)
    except ZeroDivisionError:
        return 0



if __name__ == "__main__":
    f_prescription_add = '../方剂多样性推荐/公司数据_标准药物名称.csv'
    f_medicine_add = '../方剂多样性推荐/药物数据集.csv'
    f_channel_add = '../方剂多样性推荐/药物归经_12.txt'
    f_property_add = '../方剂多样性推荐/药物性味_21.txt'
    f_effect_add = '../方剂多样性推荐/药物功效_1058.txt'
    f = open('药物功效差异度.txt','rb')
    effect_difference_matrix = pickle.load(f)
    f_prescription = pd.read_csv(open(f_prescription_add, 'r', encoding='utf-8'))
    f_medicine = pd.read_csv(open(f_medicine_add, 'r', encoding='utf-8'))
    channel_list = [i.strip() for i in open(f_channel_add, 'r', encoding='utf-8').readlines()]
    property_list = [i.strip() for i in open(f_property_add, 'r', encoding='utf-8').readlines()]
    # effect_list = [i.strip() for i in open(f_effect_add, 'r', encoding='utf-8').readlines()]

    differencr_medicine_matrix = []
    for i in range(f_medicine.shape[0]):
        start =time.time()
        a = [0]*f_medicine.shape[0]
        m1_property = f_medicine['性味'].loc[i].split('、')
        m1_property_onehot = getonehot(property_list, m1_property)

        m1_channel = f_medicine['归经'].loc[i].split('、')
        m1_channel_onehot = getonehot(channel_list, m1_channel)

        m1_effect = f_medicine['功效'].loc[i].split('、')

        for j in range(i+1,f_medicine.shape[0]):
            m2_property = f_medicine['性味'].loc[j].split('、')
            m2_property_onehot = getonehot(property_list, m2_property)

            m2_channel = f_medicine['归经'].loc[j].split('、')
            m2_channel_onehot = getonehot(channel_list, m2_channel)

            m2_effect = f_medicine['功效'].loc[j].split('、')

            channel_difference = cascading_table(m1_channel_onehot, m2_channel_onehot)
            property_difference = cascading_table(m1_property_onehot, m2_property_onehot)

            effect_difference_list = []
            for k in range(len(m1_effect)):
                for t in range(len(m2_effect)):
                    result = effect_difference_matrix[k][t] + effect_difference_matrix[t][k]
                    effect_difference_list.append(1-result)  # 得到药物的功效属性值差异度
                    # print(result)
            effect_difference = np.sum(effect_difference_list) / (
                        len(m1_effect) * len(m2_effect))  # 计算两个药物功效差异度，为药物之间所有两两功效差异度（互信息）求平均
            difference = 0.1 * channel_difference + 0.1 * property_difference + 0.8 * effect_difference
            # print(channel_difference)
            # print(property_difference)
            # print(effect_difference)
            a[j]= difference
        # print(len(a))
        print(i)
        differencr_medicine_matrix.append(a)
        # print("时间差",time.time()-start)

    a1= differencr_medicine_matrix[0:1000]
    a2= differencr_medicine_matrix[1000:2000]
    a3= differencr_medicine_matrix[2000:3000]
    a4= differencr_medicine_matrix[3000:4000]
    a5= differencr_medicine_matrix[4000:5000]
    a6= differencr_medicine_matrix[5000:6000]
    a7= differencr_medicine_matrix[6000:7000]
    a8= differencr_medicine_matrix[7000:]
    with open('药物差异度矩阵1.txt','wb') as f:
        pickle.dump(a1,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(111)
    with open('药物差异度矩阵2.txt','wb') as f:
        pickle.dump(a2,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(222)
    with open('药物差异度矩阵3.txt','wb') as f:
        pickle.dump(a3,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(333)
    with open('药物差异度矩阵4.txt','wb') as f:
        pickle.dump(a4,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(444)
    with open('药物差异度矩阵5.txt','wb') as f:
        pickle.dump(a5,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(555)
    with open('药物差异度矩阵6.txt','wb') as f:
        pickle.dump(a6,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(666)
    with open('药物差异度矩阵7.txt','wb') as f:
        pickle.dump(a7,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(777)
    with open('药物差异度矩阵8.txt','wb') as f:
        pickle.dump(a8,f,protocol=pickle.HIGHEST_PROTOCOL)
        print(888)
