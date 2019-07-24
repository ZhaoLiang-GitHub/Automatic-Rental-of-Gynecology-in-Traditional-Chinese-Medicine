import pandas as pd
import os
import re
import jieba


def word_cut_label_to_txt():
    """
    将人工审核为被正确拆分的词和被错误拆分的词存储到一个csv中，以不同列的方式进行存储，作为word_cut程序中优先级高于规则的拆分指标
    :return:
    """
    path_set3_labeld_function = "data/dict_function/set3_true_dict_labeld.txt"   # 经过人工审核并打上标签的被拆分的功效3字词
    path_set4_labeld_function = "data/dict_function/set4_true_dict_labeld.txt"   # 经过人工审核并打上标签的被拆分的功效4字词
    path_set3_labeld_effect = "data/dict_effect/set3_true_dict_labeld.txt"   # 经过人工审核并打上标签的被拆分的主治3字词
    path_set4_labeld_effect = "data/dict_effect/set4_true_dict_labeld.txt"   # 经过人工审核并打上标签的被拆分的主治4字词
    path_cut_labled = "data/cut_labled.csv"  # 可以被拆分的多字词以及不可拆分的多字词
    # 对功效特征词进行处理
    cut_true_set_function = set()    # 保存可以被拆分的功效多字词
    cut_false_set_function = set()   # 保存不可以被拆分的功效多字词
    with open(path_set3_labeld_function, "r", encoding="utf-8") as f_3:
        lines = f_3.readlines()
        for line in lines:
            word_tag_list = line.strip().split()
            # print(word_tag_list)
            if word_tag_list[-1] == "1":
                cut_true_set_function.add(word_tag_list[0])
            else:
                cut_false_set_function.add(word_tag_list[0])
    with open(path_set4_labeld_function, "r", encoding="utf-8") as f_4:
        lines = f_4.readlines()
        for line in lines:
            word_tag_list = line.strip().split()
            # print(word_tag_list)
            if word_tag_list[-1] == "1":
                cut_true_set_function.add(word_tag_list[0])
            else:
                cut_false_set_function.add(word_tag_list[0])
    cut_true_list_function = list(cut_true_set_function)
    cut_false_list_function = list(cut_false_set_function)
    cut_true_series_function = pd.Series(cut_true_list_function, name="cut_true_function")
    cut_false_series_function = pd.Series(cut_false_list_function, name="cut_false_function")
    # 对症状特征词进行处理
    cut_true_set_effect = set()    # 保存可以被拆分的症状多字词
    cut_false_set_effect = set()   # 保存不可以被拆分的症状多字词
    with open(path_set3_labeld_effect, "r", encoding="utf-8") as f_3:
        lines = f_3.readlines()
        for line in lines:
            word_tag_list = line.strip().split()
            # print(word_tag_list)
            if word_tag_list[-1] == "1":
                cut_true_set_effect.add(word_tag_list[0])
            else:
                cut_false_set_effect.add(word_tag_list[0])
    with open(path_set4_labeld_effect, "r", encoding="utf-8") as f_4:
        lines = f_4.readlines()
        for line in lines:
            word_tag_list = line.strip().split()
            # print(word_tag_list)
            if word_tag_list[-1] == "1":
                cut_true_set_effect.add(word_tag_list[0])
            else:
                cut_false_set_effect.add(word_tag_list[0])
    cut_true_list_effect = list(cut_true_set_effect)
    cut_false_list_effect = list(cut_false_set_effect)
    cut_true_series_effect = pd.Series(cut_true_list_effect, name="cut_true_effect")
    cut_false_series_effect = pd.Series(cut_false_list_effect, name="cut_false_effect")
    series_list = [cut_true_series_function, cut_false_series_function,
                   cut_true_series_effect, cut_false_series_effect,
                   ]
    cut_labled = pd.concat(series_list, axis=1)
    # print(cut_labled)
    cut_labled.to_csv(path_cut_labled, encoding="utf-8")


if __name__ == "__main__":
    word_cut_label_to_txt()
