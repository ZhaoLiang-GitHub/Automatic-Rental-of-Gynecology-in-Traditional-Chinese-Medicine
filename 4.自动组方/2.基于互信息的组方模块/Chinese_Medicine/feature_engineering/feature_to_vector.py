import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np

file_path = "../data/data_treat.csv"    # 需要进行特征向量化的数据


def get_data(path):
    """
    读取原数据并进行处理
    :param path:
    :return:
    """
    data = pd.read_csv(path)
    print(data.info())
    length = data.shape[0]
    # print(length)
    # print(data)
    for i in range(length):
        data["Taste"].loc[i] = re.split("、", data["Taste"].loc[i])
        data["Type"].loc[i] = re.split("、", data["Type"].loc[i])
        # data["Effect"].loc[i] = re.split("、", data["Effect"].loc[i])
    # print(data)
    return data, length


class FeatureToVector2:
    """
    #特征one-hot向量化（不保留序列特征）：
    """
    def __init__(self, data, length):
        self.data = data
        self.length = length
        self.feature_name = ["Taste", "Type"]

    def feature_to_id(self):
        """
        获取所有样本的所有特征
        :return:
        """
        feature_all = []
        for i in range(self.length):
            for f in self.feature_name:
                feature_all.append(self.data[f][i])
        # print(np.asarray(feature_all).shape)
        # 将所有特征转换为字典，key为特征名，value为出现次数
        feature_dict = {}
        for feature_list in feature_all:
            for f in feature_list:
                feature_dict[f] = feature_dict[f]+1 if f in feature_dict else 0
        # print(feature_dict)
        # 根据特征字典feature_dict创建特征映射字典
        feature_dict_sorted = sorted(feature_dict.items(), key=lambda x: (-x[1], x[0]))  # 根据特征出现频次进行排序
        # print(feature_dict_sorted, len(feature_dict_sorted))
        id_to_feature = {i: v[0] for i, v in enumerate(feature_dict_sorted)}  # id（根据词频排序从0开始）到word的映射字典
        feature_to_id = {v: k for k, v in id_to_feature.items()}  # 反转上一个映射字典，得到word到id的映射字典
        # print(feature_to_id)
        return feature_to_id

    def feature_to_vector(self):
        feature_to_id = self.feature_to_id()
        feature_length = len(feature_to_id)
        # print(feature_length)
        feature_vector = np.zeros((self.length, feature_length))
        # print(feature_to_vector)
        for i in range(self.length):
            for f in self.feature_name:
                for j in self.data[f].loc[i]:
                    # print(j)
                    feature_vector[i][feature_to_id[j]] = 1
        return feature_vector

    def data_to_pandas(self):
        feature_vector = self.feature_to_vector()
        data = pd.DataFrame(feature_vector)
        data.to_csv("../data/feature_vector.csv", index=False)

if __name__ == "__main__":
    data_treat, length_treat = get_data(file_path)
    # 特征one-hot向量化（不保留序列特征）
    feature_to_vector_2 = FeatureToVector2(data_treat, length_treat)
    feature_to_vector_2.data_to_pandas()
