from medicine_similarity.function_cluster_entropy import ClusterEntropy
from medicine_similarity.taste_cluster_various import cluster_various_main
from medicine_similarity.data_utils import get_data
import pickle
import numpy as np
import pandas as pd
import re

medicine_path = 'data/data_labeld_kmodes.csv'    # 药物数据的路径（该药物数据已根据性味归经的聚类结果打上标签）
function_to_medicine_path = "data/function_to_medicine.pkl"
all_relatives_path = "data/all_relatives.csv"
all_medicines_cluster_path = "data/all_medicines_cluster.pkl"   # 保存所有能被聚类的药物
medicines_to_taste_label_path = "data/medicines_to_taste_label.pkl"  # 保存药物到性味归经聚类标签的字典


def cluster():
    """
    对两种聚类算法进行整合
    :return:
    """
    cluster_various_main()  # 利用kmodes针对性味归经的聚类结果，将聚类结果作为标签输出到原药物数据中
    cluster_entropy = ClusterEntropy()  # 初始化复杂系统熵聚类
    cluster_entropy.cluster_entropy_main()   # 基于上一次的聚类结果，获取利用复杂系统熵、基于功效团的聚类结果


def word_to_index(word):
    """
    根据药物名称获取其在药物数据中的索引
    :param word:药物名称
    :return:
    """
    data, series = get_data(medicine_path)  # data为完整药物数据
    index = data.loc[data["名称"] == word].index[0]
    return index


def search_relatives(function_to_medicine, medicine_index, medicines_to_taste_label):
    """
    基于聚类结果寻找药物的相似药物
    :param function_to_medicine:基于功效团的聚类结果
    :param medicine_index:需要寻找相似药物的目标药物所在索引
    :param medicines_to_taste_label:药物到性味归经标签的映射字典
    :return:relatives_list目标药物的相似药物列表
    """
    relatives_list = []  # 用于保存相似药物的列表
    data, series = get_data(medicine_path)  # data为完整药物数据，series为功效数据
    medicine_label = data["Label"].loc[medicine_index]  # 获取目标药物在性味归经聚类结果中的标签
    function_list = re.split("、|；", data["Function"].loc[medicine_index])   # 获取目标药物的功效列表
    function_set = set(function_list)
    print("function_set:", function_set)
    for group in function_to_medicine.keys():   # 遍历字典中的功效团
        # 若功效团是目标药物功效的子集或者目标药物功效是功效团的子集，则认为该药物属于该功效团，与该功效图案中的药物有相似的可能性
        if set(group).issubset(function_set) or function_set.issubset(set(group)):
        # 若功效团是目标药物功效的子集，则认为该药物属于该功效团，与该功效图案中的药物有相似的可能性
        # if set(group).issubset(function_set):
            medicine_list = function_to_medicine[tuple(group)]  # 获取属于该功效团的药物列表
            # print("medicine_list:", medicine_list)
            # 遍历属于功效团的药物列表，若基于性味归经的聚类结果的标签相同，则认为目标药物和该药物相似
            for i in medicine_list:
                if medicines_to_taste_label[i] == medicine_label and medicines_to_taste_label[i] != medicine_index:
                    relatives_list.append(i)  # 添加相似药物的名称
    relatives_list = set(relatives_list)    # 去除重复项
    print("relatives_list:", relatives_list)
    return relatives_list


def main(is_cluster=False, search_all=False):
    """
    主函数，根据is_cluster进行聚类或者相似药物的寻找
    :param is_cluster: 决定是进行聚类，还是根据聚类结果搜索目标药物的相似药物
    :param search_all: 决定是否输出所有药物的相似药物到文件中
    :return:
    """
    if is_cluster:  # 若需要进行聚类
        cluster()
    else:   # 若需要进行相似药物寻找
        with open(function_to_medicine_path, 'rb') as f:
            function_to_medicine_dict = pickle.load(f)   # 加载聚类结果
        print("function_to_medicine:", function_to_medicine_dict)
        with open(all_medicines_cluster_path, "rb") as f:
            all_medicines_cluster = pickle.load(f)  # 加载所有被聚类的药物
        with open(medicines_to_taste_label_path, "rb") as f:
            medicines_to_taste_label = pickle.load(f)
        all_medicines_cluster = set(all_medicines_cluster)
        if search_all:
            data, _ = get_data(medicine_path)
            length = data.shape[0]
            # all_relatives = pd.DataFrame(np.zeros((length, 2)), columns=["药物名称", "相似药物"])
            all_medicine_word_list = []
            all_relatives_list = []
            for i in range(length):
                medicine_word = data["名称"].loc[i]
                all_medicine_word_list.append(medicine_word)
                if medicine_word not in all_medicines_cluster:
                    all_relatives_list.append("无相似药物")
                else:
                    medicine = word_to_index(medicine_word)  # 获取目标药物在总药物数据中的索引
                    relatives_list = search_relatives(function_to_medicine_dict, medicine, medicines_to_taste_label)  # 获得药物的相似药物的索引列表
                    all_relatives_list.append("、".join(relatives_list))
                    # all_relatives["药物名称"].loc[i] = medicine_word
                    # all_relatives["相似药物"].loc[i] = "、".join(relatives_list)
                    # all_relatives.to_csv(all_relatives_path, encoding="utf-8")
            all_medicine_word_series = pd.Series(all_medicine_word_list, name="药物名称")
            all_relatives_series = pd.Series(all_relatives_list, name="相似药物")
            series_list = [all_medicine_word_series, all_relatives_series]
            all_relatives_data = pd.concat(series_list, axis=1)
            all_relatives_data.to_csv(all_relatives_path, index=False, encoding="utf-8")
        else:
            medicine_word = "川贝母"   # 目标药物
            medicine = word_to_index(medicine_word)  # 获取目标药物在总药物数据中的索引
            relatives_list = search_relatives(function_to_medicine_dict, medicine, medicines_to_taste_label)  # 获得药物的相似药物的索引列表
            return relatives_list

if __name__ == "__main__":
    main(is_cluster=False, search_all=False)
