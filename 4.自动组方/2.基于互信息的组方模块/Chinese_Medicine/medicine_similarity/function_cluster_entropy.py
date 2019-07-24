import pandas as pd
import numpy as np
import os
import re
import pickle
from medicine_similarity.data_utils import get_data, root_to_word, word_to_root, dict_sort, combine_count, index_2_word, \
    write_csv, word_2_index, cut_by_num, group_clean, write_qyt
from medicine_similarity.relatives import calculate_correlation, create_relatives
from medicine_similarity.cluster import duplicate_removal, del_by_correlation, create_double_set, merge_loop

medicine_path = 'data/data_labeld_kmodes.csv'    # 药物数据的路径（该药物数据已根据性味归经的聚类结果打上标签）
thesaurus_path = "../data/function_tongyici.txt"  # 功效同义词字典的路径
correlation_path = 'data/correlation.csv'   # 保存互信息的文件路径
max_relatives_nums = 21  # 最大的亲友团数量
min_relatives_nums = 20  # 最小的亲友团数量
group_all_path = "data/group_all.csv"
group_best_path = "data/group_best.csv"
function_to_medicine_path = "data/function_to_medicine.pkl"  # 保存功效团到相似药物的映射
all_medicines_cluster_path = "data/all_medicines_cluster.pkl"   # 保存所有能被聚类的药物


class ClusterEntropy:
    def __init__(self):
        self.df = None  # 存储特征的one-hot变量
        self.root_name = None   # 存储排序后的词根名称，由于是有序的，所以可以作为词根到索引的映射字典，详见word_2_num和num_2_word
        self.root_fre = None    # 存储词根出现的频率，用于计算互信息
        self.combine_index = None  # 存储排序后的组合，用词根的索引表示
        self.combine_fre = None  # 存储词根的两两组合出现的频率，同样用于计算互信息
        self.combine_name = None  # 存储用词根名称表示的
        self.relatives_list = None  # 保存最大亲友团数量时的亲友团
        self.group_all_2 = None  # 保存各种亲友团数量下的聚类得到的所有功效亲友团
        self.group_best = None  # 保存最佳聚类下得到的功效亲友团
        self.data = None    # 保存全药物数据的DataFrame
        self.series = None  # 保存药物功效数据的Series

    def feature_to_vector(self):
        """
        创建并初始化ont-hot向量：从csv中读取药物数据，然后将功效特征转换为one-hot向量
        :return:
        """
        self.data, self.series = get_data(medicine_path)
        # print("data:", self.data)
        # print("series:", self.series)
        root_2_word = root_to_word(thesaurus_path)  # 获取同义词根到词的映射字典
        # print("root_2_word", len(root_2_word), root_2_word)
        word_2_root = word_to_root(thesaurus_path)     # 获取词到同义词根的映射字典
        # print("word_2_root", len(word_2_root), word_2_root)
        # 创建并初始化一个DataFrame存储one-hot向量，第一行的列索引为词根
        self.df = pd.DataFrame(np.zeros((len(self.series), len(root_2_word))), columns=root_2_word.keys())
        for indexs in self.series.index:  # series去掉了nan值，index是不连贯的,所以用这种方法遍历
            item_str = self.series[indexs]
            if item_str == '':
                continue
            # item_list = item_str.strip().split()  # 针对以空格作为分隔符的症状数据
            item_list = re.split("、|；", item_str)  # 针对以“、”作为分隔符的功效数据
            for item in item_list:
                if item in word_2_root:
                    # 找到每个功效特征词的词根，然后在one-hot向量的相应索引处进行激活
                    self.df[word_2_root[item]].loc[indexs] = 1
                else:
                    print(item)  # 输出没有匹配的词，进行人工处理
        # 删除没有任何匹配的词根
        max_value = self.df.max()    # 返回df中每一列的最大值
        # print("max_value:", max_value)
        drop_list = list(max_value[max_value == 0].index)   # 找到最大值为0的列的索引（即没有出现过的词根）
        # print("drop_list:", len(drop_list))
        self.df = self.df.drop(drop_list, axis=1)   # 删除未出现过的词根

    def root_frequency(self):
        """
        根据词频对df的列进行重新排序,并获得排列后的特征词和相应的频数,然后计算词根出现的频率
        :return:
        """
        root_count = dict(self.df.sum())  # sum对每一列求和，即能得到每个词根出现的频数
        # print("count_dic:" ,count_dic)
        self.root_name, root_nums = dict_sort(root_count)
        print("list_name:", self.root_name, "\n", "list_frequency:", root_nums)
        self.df = self.df.ix[:, self.root_name]
        # print(self.df)
        row_len = self.df.iloc[:, 0].size
        self.root_fre = [i / row_len for i in root_nums]   # 用于后面对互信息的计算
        # print(self.root_fre)

    def combine_frequency(self):
        """
        对属于同一个词根进行两两组合，并获取每个组合的频数，然后计算每个组合的频率
        :return:
        """
        combine_counts = combine_count(self.df)
        # print("combinations_dic_fre", combinations_dic_fre)
        self.combine_index, combine_nums = dict_sort(combine_counts)
        # print("combinations_list", combinations_list, "\n","combinations_frequency",combinations_frequency)
        row_len = self.df.iloc[:, 0].size
        """
        计算每个两两组合的频率，后面用于计算互信息，因为互信息是根据边缘熵和联合熵得到的，前面的单个词根的频率用于计算边缘熵
        两两组合的频率用于计算联合熵
        """
        self.combine_fre = [i / row_len for i in combine_nums]
        # print(self.combine_fre)

    def search_relatives(self):
        """
        先计算每一对两两组合之间的互信息，然后根据最大亲友团数量找到每个词根的亲友团
        :return:
        """
        correlation = calculate_correlation(self.combine_index, self.combine_fre, self.root_fre)
        self.combine_name = index_2_word(self.root_name, self.combine_index)  # 将单独的词根和组合中的索引转换为词
        # 将互信息按照大小降序排列大小，然后再写入到csv中
        data = write_csv(['组合', '关联度系数'], correlation_path, [self.combine_name, correlation])
        # 获取每个症状的亲友团list
        self.relatives_list = create_relatives(self.root_name, data, max_relatives_nums)
        print("relatives_list", self.relatives_list)  # 这里的亲友团是用嵌套列表存储的，用字典存储应该更好吧？键值对为变量-亲友团列表

    def cluster(self):
        """
        对亲友团进行聚类
        :return:
        """
        # print("relatives_list", relatives_list)
        list_qyt = duplicate_removal(self.relatives_list, self.root_name)   # 将每个词根的亲友提取出来、组成亲友团
        print("list_qyt:", list_qyt)
        write_qyt(list_qyt, self.root_name)
        list_index = word_2_index(self.root_name, list_qyt)  # 使用索引代替列表中的项
        best_info_value = 0
        best_group_num = 0
        for group_num in range(min_relatives_nums, max_relatives_nums+1):   # 限制亲友团的数量，根据不同的亲友团数量进行聚类
            new_list = cut_by_num(list_index, group_num)    # 对亲友团进行裁剪
            list_index_2 = del_by_correlation(new_list)    # 删除弱相关项，留下强相关的两两组合
            # re_word = index_2_word(root_name, list_index_2)
            double_set = create_double_set(list_index_2)  # 创建二元组
            # print("doubleSet:", doubleSet)
            # 进行团合并操作，一直到无法继续合并
            max_num, best_set = merge_loop(double_set, self.root_name, 'data/group' + str(group_num) + '.csv')
            # 计算信息利用率
            info_value = max_num / group_num
            print(max_num, '/', group_num, '=', info_value)
            if info_value > best_info_value:
                best_info_value = info_value
                best_group_num = group_num
        print("最佳信息利用率：", best_info_value, "最佳亲友团个数：", best_group_num)

    def group_all(self):
        """
        先将基于不同数量亲友团的聚类结果进行清理、删除被包含在更大团中的团，然后将清洗后的聚类结果进行结合并输出
        :return:
        """
        group_all = []
        group_name = []
        for i in range(max_relatives_nums, min_relatives_nums-1, -1):
            group_path = os.path.join("data", "group"+str(i)+".csv.pkl")
            group_name.append("group"+str(i))
            group_all.append(group_clean(group_path))
        # print("group_all:", group_all)
        write_csv(group_name, group_all_path, group_all)    # 将不同亲友团数量下的所有聚类结果进行输出
        group_all_dict = {}  # 记录各个团出现的次数
        self.group_all_2 = []   # 用于保存所有亲友团数量下的功效团
        for group_list in group_all:
            for group in group_list:
                group_all_dict[tuple(group)] = group_all_dict[tuple(group)]+1 if tuple(group) in group_all_dict else 1
        print("group_all_dict:", group_all_dict)
        for group in group_all_dict.keys():
            self.group_all_2.append(group)  # 确保self.group_all_2中不出现重复的团
        print("self.group_all_2:", self.group_all_2)
        # 人工审核找到信息利用率最高的亲友团数量，即最佳的亲友团数量，然后以该亲友团数量下的结果作为最佳结果
        group_best_name = []
        group_best = []
        for i in range(21, 22):
            group_path = os.path.join("data", "group"+str(i)+".csv.pkl")
            group_best_name.append("group"+str(i))
            group_best.append(group_clean(group_path))
        write_csv(group_best_name, group_best_path, group_best)
        self.group_best = group_best[0]
        print(self.group_best)


    def cluster_entropy_main(self):
        """
        主聚类函数：调用其他函数进行功效特征的聚类、找到功效的亲友团，然后找到属于每个亲友团的药物，完成最终药物的聚类
        :return:
        """
        self.feature_to_vector()
        self.root_frequency()
        self.combine_frequency()
        self.search_relatives()
        self.cluster()
        self.group_all()
        # group_all_2 = self.group_all_2   # 用所有亲友团数量下的功效团进行药物聚类
        group_all_2 = self.group_best   # 用最佳亲友团数量下的功效团进行聚类
        function_to_medicine = {}   # 用于保存功效团到包含该功效团的药物
        # 预先为每个团创建一个列表作为value，这里要注意，dict的key必须是可哈希的，所以需要使用元祖tuple
        for group in group_all_2:
            function_to_medicine[tuple(group)] = []
        count = 0   # 记录多少个药物能根据功效团进行聚类
        for index in self.series.index:  # 由于有些缺失值被删除，所以用这种遍历方式比较保险
            function_list = re.split("、|；", self.series.loc[index])
            function_set = set(function_list)
            # print("function_set:", function_set)
            for group in group_all_2:
                # 若功效团是药物功效的子集或者药物功效是功效团的子集，则认为该药物属于该功效团
                if set(group).issubset(function_set) or function_set.issubset(set(group)):
                # 若功效团是药物功效的子集，则认为该药物属于该功效团
                # if set(group).issubset(function_set):
                    # print(index, group, function_set)
                    count += 1
                    function_to_medicine[tuple(group)].append(self.data["名称"].iloc[index])
        # 根据功效团拥有的药物数量进行排序
        function_to_medicine_sort = sorted(function_to_medicine.items(), key=lambda x: len(x[1]))
        print(function_to_medicine_sort)
        function_group_list = []    # 功效团列表
        medicines_list = []  # 功效团拥有的药物列表
        for function_medicine in function_to_medicine_sort:
            function_group_list.append(function_medicine[0])
            medicines_list.append(function_to_medicine[function_medicine[0]])
        # 获取拥有最多药物的功效团以及其具体拥有的药物
        all_medicines_cluster = []
        max_functions = 0
        max_medicines = 0
        for functions, medicines in function_to_medicine.items():
            print("功效团：", functions, "药物数量：", len(medicines))
            all_medicines_cluster.extend(medicines)
            if len(medicines) > max_medicines:
                max_medicines = len(medicines)
                max_functions = functions
        print("一共被能被聚类的药物数量：", len(all_medicines_cluster))
        print("拥有最多药物的功效团：", max_functions, "药物数量：", max_medicines)
        print("能够被聚类的药物数量:", count, "function_to_medicine:", len(function_to_medicine), function_to_medicine)
        function_group_series = pd.Series(function_group_list, name="功效团")
        medicines_series = pd.Series(medicines_list, name="药物列表")
        function_to_medicine_df_list = [function_group_series, medicines_series]
        function_to_medicine_df = pd.concat(function_to_medicine_df_list, axis=1)
        with open(all_medicines_cluster_path, "wb") as f:
            pickle.dump(all_medicines_cluster, f)  # 将所有被聚类的药物保存
        with open(function_to_medicine_path, "wb") as f:
            pickle.dump(function_to_medicine, f)    # 将聚类结果保存
        function_to_medicine_df.to_csv("data/result_function_cluster.csv", index=False, encoding="utf-8")

if __name__ == "__main__":
    Cluster = ClusterEntropy()
    Cluster.cluster_entropy_main()
