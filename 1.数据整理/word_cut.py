import pandas as pd
import re
import os
import jieba.posseg as pseg
import numpy as np


def get_data():
    """
    数据预处理:示例原始数据是药名、性味+归经为一列、功效+主治为一列,将性味归经、功效主治进行分离，
    并将其以，、作为分隔符处理成列表形式
    :return:data:经处理的DataFrame["Name","Taste","Type","Function","Effect"]
    """
    # 若文件读取错误只要在记事本或者编辑器中打开以utf-8的编码格式重新打开即可
    data = pd.read_csv("data/data_all_v10.csv")    # delimiter指定分隔符，根据具体数据调整
    length = data.shape[0]
    print(data.info())
    data = data.fillna("missing")
    print(data.info())
    # 插入列，分别保存“归经”和“主治”的数据
    # data.insert(2, "Type", None)
    # data.insert(4, "Effect", None)

    # 对性味部分数据进行处理，分为性味和归经两列并以，、作为分隔符返回列表
    for i in range(length):
        l = data["Taste"].loc[i]
        data["Taste"].loc[i] = re.split("、", l)
        l = data["Type"].loc[i]
        # print("l:", l)
        data["Type"].loc[i] = re.split("、", l)

    # 对功用主治部分进行处理,分为功用和主治两列并以，、作为分隔符返回列表
    for i in range(length):
        # l = data["Function"].loc[i].split("。")
        data["Function"].loc[i] = re.split("[，、；]", data["Function"].loc[i])
        data["Effect"].loc[i] = re.split("[，、；]", data["Effect"].loc[i])

    return data, length


class WordCut:
    """
    对功效和主治部分数据进行分词和存储
    """
    def __init__(self, data, length):
        """
        引入语料库并创建词库
        :param data:原药物数据
        :param length:样本数量
        :return:
        """
        self.data = data
        self.length = length
        self.set2 = {}
        self.set3_true = {}  # 保存可以拆分的3字词
        self.set3_false = {}    # 保存不可拆分的3字词
        self.set4_true = {}  # 保存可以拆分的4字词
        self.set4_false = {}    # 保存不可拆分的4字词
        self.cut_true_set_function = None    # 可被拆分的功效多字词库
        self.cut_false_set_function = None    # 不可被拆分的功效多字词库
        self.cut_true_set_effect = None    # 可被拆分的症状多字词库
        self.cut_false_set_effect = None    # 不可被拆分的症状多字词库

    def get_cut_lable(self):
        cut_lable_data = pd.read_csv("data/cut_labled.csv")
        # 创建可被拆分与不可拆分的功效多字词库
        cut_true_list_function = list(cut_lable_data["cut_true_function"])
        cut_false_list_function = list(cut_lable_data["cut_false_function"])
        # print(cut_true_list)
        self.cut_true_set_function = set(cut_true_list_function)
        # print(self.cut_true_set)
        self.cut_false_set_function = set(cut_false_list_function)
        # 创建可被拆分与不可拆分的症状多字词库
        cut_true_list_effect = list(cut_lable_data["cut_true_effect"])
        cut_false_list_effect = list(cut_lable_data["cut_false_effect"])
        self.cut_true_set_effect = set(cut_true_list_effect)
        self.cut_false_set_effect = set(cut_false_list_effect)

    def word_clean_taste(self):
        """
        对性味部分数据进行清洗和分词
        :return:
        """
        print("对性味部分数据进行清洗")
        for i in range(self.length):
            list_taste = self.data["Taste"].loc[i]
            length2 = len(list_taste)
            for j in range(length2):
                list_taste[j] = re.sub("性|味", "", list_taste[j])
            # print(type(list_taste))
            # 列表转为字符串，以便后续输出数据
            s = list_taste[0]
            for j in range(1, length2):
                s = "%s%s%s" % (s, "、", list_taste[j])
            self.data["Taste"].loc[i] = s

    def word_clean_type(self):
        """
        对归经部分数据进行清洗和分词
        :param self:
        :return:
        """
        print("对归经部分数据进行清洗")
        for i in range(self.length):
            list_type = self.data["Type"].loc[i]
            # print(type(list_type))
            length2 = len(list_type)
            for j in range(length2):
                list_type[j] = re.sub("归|经|入", "", list_type[j])
            # 列表转为字符串，以便后续输出数据
            s = list_type[0]
            for j in range(1, length2):
                s = "%s%s%s" % (s, "、", list_type[j])
            self.data["Type"].loc[i] = s

    def word_cut_function(self):
        """
        对功效部分的数据进行分词处理，依次对2字词、3字词、4字词进行处理
        """
        self.get_cut_lable()    # 获取可拆分的多字词库和不可拆分的多字词库
        print("对功用部分进行分词处理")
        # 先建立2字词库
        for i in range(self.length):
            list_function = self.data["Function"].loc[i]
            length2 = len(list_function)
            # print(list_function)
            for j in range(length2):
                if len(list_function[j]) == 2:
                    self.word_cut_2(list_function[j])
            # print(type(listFunction))

        # 对3字词进行处理并建立3字词库
        for i in range(self.length):
            list_function = self.data["Function"].loc[i]
            length2 = len(list_function)
            for j in range(length2):
                word = list_function[j]
                if len(list_function[j]) == 3:
                    # 首先判断该词是否在可拆分的多字词库或者不可拆分的多字词库中（以前拆分过且经过人工审核）
                    if word in self.cut_true_set_function:
                        word1 = '%s%s' % (word[0], word[1])
                        word2 = '%s%s' % (word[0], word[2])
                        self.set2[word1] = (self.set2[word1] if word1 in self.set2 else 0) + 1
                        self.set2[word2] = (self.set2[word2] if word2 in self.set2 else 0) + 1
                        word = '%s%s%s' % (word1, "、", word2)
                    elif word in self.cut_false_set_function:
                        pass
                    else:
                        word = self.word_cut_3(word)
                    list_function[j] = word
            # print(type(listFunction))

        # 对4字词进行处理并建立4字词库
        for i in range(self.length):
            list_function = self.data["Function"].loc[i]
            length2 = len(list_function)
            for j in range(length2):
                word = list_function[j]
                if len(list_function[j]) == 4:
                    if word in self.cut_true_set_function:
                        word_list = re.findall('.{2}', word)
                        self.set2[word_list[0]] = (self.set2[word_list[0]] if word_list[0] in self.set2 else 0) + 1
                        self.set2[word_list[1]] = (self.set2[word_list[1]] if word_list[1] in self.set2 else 0) + 1
                        word = '%s%s%s' % (word_list[0], "、", word_list[1])
                    elif word in self.cut_false_set_function:
                        pass
                    else:
                        word = self.word_cut_4(word)
                    list_function[j] = word
            # print(type(listFunction))

    def word_cut_effect(self):
        """
        主治部分的数据进行分词处理,依次对2字词、3字词、4字词进行处理
        """
        print("对主治部分进行分词处理")
        # 先建立2字词库
        for i in range(self.length):
            list_effect = self.data["Effect"].loc[i]
            length2 = len(list_effect)
            # print(list_effect)
            for j in range(length2):
                # 清理“主治、或”等停用词
                list_effect[j] = re.sub("主治|主|治|用于|常用|可用|可|用|等症|等|以及|各种", "", list_effect[j])
                if len(list_effect[j]) == 2:
                    self.word_cut_2(list_effect[j])

        # 对3字词进行处理并建立3字词库
        for i in range(self.length):
            list_effect = self.data["Effect"].loc[i]
            length2 = len(list_effect)
            for j in range(length2):
                word = list_effect[j]
                if len(word) == 3:
                    if word in self.cut_true_set_effect:
                        word1 = '%s%s' % (word[0], word[1])
                        word2 = '%s%s' % (word[0], word[2])
                        self.set2[word1] = (self.set2[word1] if word1 in self.set2 else 0) + 1
                        self.set2[word2] = (self.set2[word2] if word2 in self.set2 else 0) + 1
                        word = '%s%s%s' % (word1, "、", word2)
                    elif word in self.cut_false_set_effect:
                        pass
                    else:
                        word = self.word_cut_3(word)
                    list_effect[j] = word

        # 对4字词进行处理并建立4字词库
        for i in range(self.length):
            list_effect = self.data["Effect"].loc[i]
            length2 = len(list_effect)
            for j in range(length2):
                word = list_effect[j]
                if len(word) == 4:
                    if word in self.cut_true_set_effect:
                        word_list = re.findall('.{2}', word)
                        self.set2[word_list[0]] = (self.set2[word_list[0]] if word_list[0] in self.set2 else 0) + 1
                        self.set2[word_list[1]] = (self.set2[word_list[1]] if word_list[1] in self.set2 else 0) + 1
                        word = '%s%s%s' % (word_list[0], "、", word_list[1])
                    elif word in self.cut_false_set_effect:
                        pass
                    else:
                        word = self.word_cut_4(word)
                    list_effect[j] = word

    def word_cut_2(self, word):
        """
        对2字词的处理：直接保存到2字词库
        :param word:2字词
        :return:
        """
        # 保存到词库并计数
        self.set2[word] = (self.set2[word] if word in self.set2 else 0) + 1

    def word_cut_3(self, word):
        """
        根据词性进行拆分
        :param word:3字词
        :return:
        """
        # 用两个列表记录单字及其词性
        word_list = []
        char_list = []
        # 判断3字词中每个字的词性
        for s in word:
            # 每个word_jieba是一个生成器，包含单字及其词性
            word_jieba = pseg.cut(s)
            for w in word_jieba:
                word_list.append(w.word)
                char_list.append(w.flag)
        word1 = '%s%s' % (word_list[0], word_list[1])
        word2 = '%s%s' % (word_list[0], word_list[2])
        word_new = '%s%s%s' % (word1, "、", word2)
        # 根据三字词中每个字的词性进行进一步处理
        # 若满足“动名名”的规律且拆分后的某个词在2字词库中出现了，则认为该词是可分的，分配到set_true词库中(这样做应该更合理，但是在测试数据集上会出错)
        if char_list == ['v', 'n', 'n'] and (word1 in self.set2 or word2 in self.set2):
            self.set3_true[word] = (self.set3_true[word] if word in self.set3_true else 0) + 1
            # 字符串拼接
            # 将拆分成的2字词添加到2字词库中
            self.set2[word1] = (self.set2[word1] if word1 in self.set2 else 0) + 1
            self.set2[word2] = (self.set2[word2] if word2 in self.set2 else 0) + 1
            return word_new
        else:
            # 无法拆分，先分配到set3_false词库中
            self.set3_false[word] = (self.set3_false[word] if word in self.set3_false else 0) + 1
            return word

    def word_cut_4(self, word):
        """
        对4字词进行拆分：两两拆分后与2字词库中的2字词进行对比，若拆分后的某个词存在于2字词库中，则认为可拆分，若不存在于2字词库中，
        则与可拆分的3字词库中的3字词进行对比，若存在与该4字词编辑距离为1的可拆分的3字词，则认为可拆分、且作为3字词进行处理。
        :param word:4字词
        :return:
        """
        # 按照长度2进行分割，与set2中的2字词进行对比并处理
        word_list = re.findall('.{2}', word)
        temp = False
        for i in word_list:
            if i in self.set2:
                temp = True
            else:
                pass
        if temp:
            # 若22拆分后的单词在2字词库中出现过
            self.set4_true[word] = (self.set4_true[word] if word in self.set4_true else 0) + 1
            self.set2[word_list[0]] = (self.set2[word_list[0]] if word_list[0] in self.set2 else 0) + 1
            self.set2[word_list[1]] = (self.set2[word_list[1]] if word_list[1] in self.set2 else 0) + 1
            word = '%s%s%s' % (word_list[0], "、", word_list[1])
            # print(word)
            return word
        else:
            # 若22拆分后的单词在2字词库中未出现过，则先与set3_true中的3字词进行对比
            for word_3 in self.set3_true:
                dis = self.difflib_leven(word_3, word)
                if dis == 1:
                    # print("true")
                    # 若与set3_true中某个3字词只有一个编辑距离
                    word = self.word_cut_3(word_3)
                    # print(word)
                    return word
                else:
                    self.set4_false[word] = (self.set4_false[word] if word in self.set4_false else 0) + 1
                    # print(word)
                return word

    @staticmethod
    def difflib_leven(str1, str2):
        """
        用动态规划对字符串间的编辑距离进行计算
        :param str1:需要对比的字符串1
        :param str2:需要对比的字符串2
        :return:
        """
        len_str1 = len(str1) + 1
        len_str2 = len(str2) + 1
        # 创建dp矩阵
        matrix = [0 for n in range(len_str1 * len_str2)]
        # 初始化X轴
        for i in range(len_str1):
            matrix[i] = i
        # 初始化Y轴
        for j in range(0, len(matrix), len_str1):
            if j % len_str1 == 0:
                matrix[j] = j // len_str1

        for i in range(1, len_str1):
            for j in range(1, len_str2):
                if str1[i-1] == str2[j-1]:
                    cost = 0
                else:
                    cost = 1
                matrix[j*len_str1+i] = min(matrix[(j-1)*len_str1+i]+1,
                                           matrix[j*len_str1+(i-1)]+1,
                                           matrix[(j-1)*len_str1+(i-1)] + cost)
        return matrix[-1]

    def list_to_str(self):
        """
        分词后将功效和主治的列表全部转换为字符串，方便输出到CSV后的读取与处理，否则将数据写入csv时，写入的字符串会包含[]和""等。
        :return:
        """
        for i in range(self.length):
            list_function = self.data["Function"].loc[i]
            # print(type(listFunction))
            length2 = len(list_function)
            # print(length2)
            s = list_function[0]
            for j in range(1, length2):
                s = "%s%s%s" % (s, "、", list_function[j])
            self.data["Function"].loc[i] = s
        for i in range(self.length):
            list_effect = self.data["Effect"].loc[i]
            length2 = len(list_effect)
            s = list_effect[0]
            for j in range(1, length2):
                s = "%s%s%s" % (s, "、", list_effect[j])
            self.data["Effect"].loc[i] = s

    def data_analyse(self):
        """
        处理后的数据分析
        :return:
        """
        print("结果数据处理")
        print("========2字词数量=========：", len(self.set2))
        for i in self.set2:
            print(i, self.set2[i])
        print("========被拆分的3字词数量==========：", len(self.set3_true))
        for i in self.set3_true:
            print(i, self.set3_true[i])
        print("============未被拆分的3字词数量=========：", len(self.set3_false))
        for i in self.set3_false:
            print(i, self.set3_false[i])
        print("=========被拆分的4字词数量============：", len(self.set4_true))
        for i in self.set4_true:
            print(i, self.set4_true[i])
        print("=========未被拆分的4字词数量===========：", len(self.set4_false))
        for i in self.set4_false:
            print(i, self.set4_false[i])

        # 各字典按照值的大小进行排序得到相应元祖，然后转换为DataFrame并导出到CSV，最后绘图
        # self.set2 = sorted(self.set2.items(), key=lambda item:item[1], reverse = True)
        # df_2 = pd.DataFrame(self.set2)
        # df_2.to_csv("data_words_2.csv")
        # words_2 = df_2.iloc[0:15]
        # words_2.plot(kind = 'bar')
        # plt.title("words_2")
        # plt.show()

    def word_clean(self):
        """
        清理词频过低的词（特征）:找出所有词频等于或者低于5的词，将这些词存入列表，然后通过读取列表中的词创建相应的正则表达式，
        再遍历整个文件去除低词频的词（此方法过于麻烦，需要优化）
        :return:
        """
        word_low_freq = []
        # print(type(self.set2))
        list_set = [self.set2, self.set3_false, self.set4_false]
        for d in list_set:
            for key, value in d.items():
                # print(key, value)
                if value <= 5:
                    word_low_freq.append(key)
        # print(word_low_freq)
        # print(len(word_low_freq))
        pattern_string = word_low_freq[0]
        for i in word_low_freq[1:]:
            pattern_string = "%s%s%s" % (pattern_string, "|", i)
        # print(pattern_string)

        for i in range(self.length):
            # 清理词频低的词
            self.data["Function"].loc[i] = re.sub(pattern_string, "", self.data["Function"].loc[i])
            self.data["Effect"].loc[i] = re.sub(pattern_string, "", self.data["Effect"].loc[i])
            # 清理遗留的符号
            self.data["Function"].loc[i] = re.sub("、{2,}", "、", self.data["Function"].loc[i])
            self.data["Function"].loc[i] = re.sub("^\、|\、$", "", self.data["Function"].loc[i])
            self.data["Effect"].loc[i] = re.sub("、{2,}", "、", self.data["Effect"].loc[i])
            self.data["Effect"].loc[i] = re.sub("^\、|\、$", "", self.data["Effect"].loc[i])

    def write_csv(self):
        """
        对数据输出写入到CSV文件中
        :return:
        """
        self.data.to_csv("../data/data_treat.csv", index=False, encoding="utf-8")    # 输出所有经过处理的数据
        function_data = self.data["Function"]
        # print(function_data)
        for i in range(function_data.shape[0]):
            function_line = function_data.loc[i]
            function_list = function_line.strip().split("、")
            function_list_234 = []
            for word in function_list:
                if len(word) == 2 or len(word) == 3 or len(word) == 4:
                    function_list_234.append(word)
            function_line = "、".join(function_list_234)
        function_data.to_csv("../data/function_treat.csv", index=False, encoding="utf-8")    # 输出功效数据，用于进行针对功效的复杂系统熵聚类
        # 输出所有功效特征词，用于进行功效特征词的同义词分类和复杂系统熵聚类
        func_dict = {}
        for i in range(self.length):
            word_list = re.split("[；、]", function_data.loc[i])
            for word in word_list:
                func_dict[word] = func_dict[word]+1 if word in func_dict else 1
        print(func_dict)
        func_list = []
        for word in func_dict.keys():
            func_list.append(word)
        with open("../data/function_tongyici.txt", "w", encoding="utf-8") as f:
            for word in func_list:
                if len(word) == 2 or len(word) == 3 or len(word) == 4:
                    f.write(word+"\n")

    def write_txt(self, str_type):
        if str_type == "function":
            path_dir = "data/dict_function"
        else:
            path_dir = "data/dict_effect"
        set2_dict = os.path.join(path_dir, "set2_dict"+".txt")
        set3_true_dict = os.path.join(path_dir, "set3_true_dict"+".txt")
        set3_false_dict = os.path.join(path_dir, "set3_false_dict"+".txt")
        set4_true_dict = os.path.join(path_dir, "set4_true_dict"+".txt")
        set4_false_dict = os.path.join(path_dir, "set4_false_dict"+".txt")
        with open(set2_dict, "w", encoding="utf-8") as f:
            for word in self.set2:
                f.write(word+"\n")
        with open(set3_true_dict, "w", encoding="utf-8") as f:
            for word in self.set3_true:
                f.write(word+"\n")
        with open(set3_false_dict, "w", encoding="utf-8") as f:
            for word in self.set3_false:
                f.write(word+"\n")
        with open(set4_true_dict, "w", encoding="utf-8") as f:
            for word in self.set4_true:
                f.write(word+"\n")
        with open(set4_false_dict, "w", encoding="utf-8") as f:
            for word in self.set4_false:
                f.write(word+"\n")

    def cut_main(self):
        # 对性味归经部分数据进行清洗
        self.word_clean_taste()
        self.word_clean_type()
        # 重复对功效部分数据进行分词处理，使词库完整
        self.word_cut_function()
        self.word_cut_function()
        self.data_analyse()  # 分词结果分析
        # 将词库输出写入到文件中
        self.write_txt("function")  # 初始化词库字典、用于存储主治特征词
        # # 初始化存储特征词的dict
        self.set2 = {}
        self.set3_true = {}  # 保存可以拆分的3字词
        self.set3_false = {}    # 保存不可拆分的3字词
        self.set4_true = {}  # 保存可以拆分的4字词
        self.set4_false = {}    # 保存不可拆分的4字词
        # # 重复对主治部分数据进行分词处理，使词库完整
        self.word_cut_effect()
        self.word_cut_effect()
        self.data_analyse()
        self.write_txt("effect")    # 将词库输出写入到文件中
        self.list_to_str()
        self.write_csv()

if __name__ == "__main__":
    # 读取数据并进行预处理
    data_treat, length_treat = get_data()
    word_cut = WordCut(data_treat, length_treat)
    word_cut.cut_main()

    # # 实例化的类能够直接调用上面的data_treat和length_treat，所以要注意类内部参数的调用，之前参数调用出错了，上一次commit就是解决该问题的
    # # 创建对性味和归经的分词类
    # word_cut_Taste = WordCut(data_treat, length_treat)
    # # 对性味和归经部分的数据进行清洗
    # word_cut_Taste.word_clean_taste()
    # word_cut_Taste.word_clean_type()

    # 创建主治和功效的分词类
    # word_cut = WordCut(data_treat, length_treat)
    # # 对功用部分进行分词处理
    # word_cut.word_cut_function()
    # # 对主治部分进行分词处理
    # word_cut.word_cut_effect()
    # # 重复对功效和主治部分进行分词处理，使词库完整
    # word_cut.word_cut_function()
    # word_cut.word_cut_effect()
    # # 列表转为字符串
    # word_cut.list_to_str()
    #
    # # 结果数据分析
    # word_cut.data_analyse()
    # # 数据清理，清理主治和功效中词频过低的词
    # word_cut.word_clean()
    # # 数据清理后再次进行数据分析，需要重新对data_treat进行读取和记录
    # # 结果导出
    # word_cut.write_csv()
