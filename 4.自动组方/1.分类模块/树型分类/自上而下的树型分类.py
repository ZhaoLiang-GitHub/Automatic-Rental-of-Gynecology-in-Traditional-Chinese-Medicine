# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 14:00
# @Author  : xieyunshen
# @FileName: sim_sen_v1.py
# @Software: PyCharm
import math
import jieba
import pandas as pd
import operator


class GetDicts:

    def __init__(self, combine_file, stopwords_file, synonym_file, body_parts_file, add_words_file, sugg_freq_file, negative_words_file, antonym_file):
        self.combine_file = combine_file
        self.stopwords_file = stopwords_file
        self.add_words_file = add_words_file
        self.sugg_freq_file = sugg_freq_file
        self.body_parts_file = body_parts_file
        self.synonym_file = synonym_file
        self.negative_words_file = negative_words_file
        self.antonym_file = antonym_file

    # 创建同义词词典
    def get_combine_dict(self):
        combine_dict = {}
        for line in open(self.combine_file, 'r', encoding='utf-8'):
            seperate_word = line.strip().split(' ')
            num = len(seperate_word)
            for i in range(1, num):
                combine_dict[seperate_word[i]] = seperate_word[0]
        return combine_dict

    # 创建停用词词典
    # 从stopwords中读取停用词，生成停用词列表。
    def get_stop_words(self):
        stopwords_list = []
        for line in open(self.stopwords_file, 'r',encoding='utf-8'):
            stopwords_list.append(line.rstrip('\n'))
        # 生成词典
        stopwords_dict = {}.fromkeys(stopwords_list)

        return stopwords_dict

    # 创建部位词典
    # 词典起始位置为1。
    def get_body_parts_dict(self):
        body_parts_list = []
        for line in open(self.body_parts_file, 'r', encoding='utf-8'):
            body_parts_list.append(line.rstrip('\n'))
        # print(body_parts_list)
        # print(len(body_parts_list))
        body_parts_dict = {}
        i = 1
        for i in range(1, 1+len(body_parts_list)):
            # print(i)
            body_parts_dict[body_parts_list[i-1]] = i
            jieba.add_word(body_parts_list[i-1])
        # print(len(body_parts_dict))
        # print(body_parts_dict)
        return body_parts_dict

    # 获取近义词词典
    def get_synonym_dict(self):
        '''
        :return:
        '''
        synonym_dict = {}
        for line in open(self.synonym_file, 'r', encoding='utf-8'):
            separate_word = line.strip().split(' ')
            num = len(separate_word)
            for i in range(1, num):
                synonym_dict[separate_word[i]] = separate_word[0]
        return synonym_dict

    # 读取add_word_in_dict.txt文件中的词语，加入到结巴的词典中，以达到较好的分词结果。
    def add_words(self):
        for line in open(self.add_words_file, 'r', encoding='utf-8'):
            add_words = line.rstrip('\n')
            jieba.add_word(add_words)

    # 读取suggest_freq.txt 将分词错误的词语进行处理，以达到更好的分词结果
    def sugg_freq(self):
        for line in open(self.sugg_freq_file, 'r', encoding='utf-8'):
            suggest_freq = line.rstrip('\n').split(' ')
            # print(suggest_freq
            jieba.suggest_freq(suggest_freq, True)

    # 创建否定词词典
    def get_negative_words_dict(self):
        '''
        :return: 否定词词典
        '''
        negative_word_list = []
        for line in open(self.negative_words_file, 'r', encoding='utf-8'):
            negative_word_list.append(line.strip('\n'))
        negative_word_dict = {}.fromkeys(negative_word_list)
        return negative_word_dict

    # 创建反义词词典
    def get_antonym_dict(self):
        '''
        :return: 反义词词典
        '''
        i = 1
        antonym_dict = {}
        for line in open(self.antonym_file, 'r', encoding='utf-8'):
            antonym_word = line.strip('\n')
            antonym_dict[antonym_word] = i
            i += 1
        return antonym_dict


# 调用get_dicts类，获取所需的各个词典和参数
dicts = GetDicts(
                                                  combine_file='./dict/同义词.txt',
                                                  body_parts_file='./dict/部位词.txt',
                                                  synonym_file='./dict/近义词.txt',
                                                  stopwords_file='./dict/停用词',
                                                  add_words_file='./dict/add_word_in_dict.txt',
                                                  sugg_freq_file='./dict/suggest_freq_dict.txt',
                                                  antonym_file='./dict/反义词',
                                                  negative_words_file='./dict/否定词')
# 获取同义词词典
combine_dict = dicts.get_combine_dict()
# 获取停用词词典
stopwords_dict = dicts.get_stop_words()
# 获取身体部位词典
body_parts_dict = dicts.get_body_parts_dict()
# 获取近义词词典，作用是对症状词语进行预先标准化
synonym_dict = dicts.get_synonym_dict()
# 获取否定词词典
negative_words_dict = dicts.get_negative_words_dict()
# 获取反义词词典
antonym_dict = dicts.get_antonym_dict()
# 调用函数，在结巴词典中添加词语
dicts.add_words()
# 调用函数，调整切分容易出错的词语
dicts.sugg_freq()


class DealSentences:
    def __init__(self, sentence):
        self.sentence = sentence

    def pre_process_sen(self):
        '''
        :return: 预处理过的句子
        '''
        words = self.sentence.rstrip(' ').split(' ')
        # print(words)
        pre_process_sentence = ''
        # 在近义词词典中判断是否存在近义词，进行替换，将部分症状标准化
        for word in words:
            # 判断不进行分词的词语是否在近义词词典中出现，如果出现，则进行替换
            if word in synonym_dict:
                word = synonym_dict[word]
                pre_process_sentence += word+' '
                continue
            # 将词语进行分词处理，进行部位词和描述词的同义替换
            segs = jieba.lcut(word)
            pre_process_word = ''   # 存储处理后的词语
            for seg in segs:
                if seg in combine_dict:
                    seg = combine_dict[seg]
                    pre_process_word += seg
                    if seg not in pre_process_word:   # 防止重复替换出现，如白带气味臭秽，‘气味’和‘臭秽’都替换为‘臭味’
                        pre_process_word += seg
                    else:
                        pass
                # 除去停用词
                elif seg in stopwords_dict:
                    pass
                else:
                    pre_process_word += seg
            pre_process_sentence += pre_process_word+' '
        self.pre_process_sentence = pre_process_sentence
        return pre_process_sentence

    def process_sen(self):
        '''
        :return: 症状词语按部位词分类的列表
        '''
        sen = self.pre_process_sentence.rstrip(' ')
        # 创建部位列表
        body_parts_list = []
        # 创建各个部位对应的子列表
        for i in range(len(body_parts_dict)+1):
            body_parts_list.append([])
        # print(len(body_parts_list))
        words = sen.rstrip('\n').split(' ')
        for word in words:
            segs = jieba.lcut(word)

            if len(segs) == 1:  # 判断是否分词成功，len(segs)==1 说明并未进行分词
                # 对二字词进行处理
                segs_word = list(word)  # 将词语按字分隔，判断是否存在单字部位词
                cut_bool = False    # 判断是否存在单字部位词的符号
                for seg in segs_word:
                    # 如果分割开的词语中存在单字部位词，则进行进一步处理，并且将cut_bool设置为True
                    if seg in body_parts_dict:  # body_parts_dict是部位词典
                        word_rep = word.replace(seg, '')    # 替换掉部位词，只留下描述词
                        # 根据seg查找到body_parts_dict词典中的该部位的value，该value是存储部位对应症状列表的index
                        body_parts_list[body_parts_dict[seg]].append(word_rep)
                        cut_bool = True
                        continue
                # cut_bool为False说明词语中不存在部位词，所以将其放入无部位症状的子列表中。
                # body_parts_list[0]中存储无部位的症状
                if cut_bool is False:
                    body_parts_list[0].append(word)
                continue
            # 对多字部位词进行相似的处理
            has_body_parts = False
            for seg in segs:
                if seg in body_parts_dict:
                    word_rep = word.replace(seg, '')
                    body_parts_list[body_parts_dict[seg]].append(word_rep)
                    # print(body_parts_dict[seg])
                    has_body_parts = True
            if has_body_parts is False:
                body_parts_list[0].append(word)
        # print(body_parts_list)
        return body_parts_list


# 计算句子的相似度
class SimSentence:
    def __init__(self, list1, list2, cilin_file, body_parts_len):
        self.list_standard = list1
        self.list_symptom_input = list2
        # 同意词林文件路径
        self.cilin_file = cilin_file
        # 部位词典的长度，用来判断列表中的子列表的数量
        self.body_parts_len = int(body_parts_len)
        # 存在部位的症状列表
        self.word_pairs_with_bodyparts = []
        # 不存在部位词的症状列表
        self.word_pairs_with_nobodyparts = []
        # 同义词林处理后列表
        self.word_list = []
        self.word = []

    def get_word_pairs(self):
        '''
        symptom_list1:标准症型的症状描述词的集合
        symptom_list2: 输入症状描述词的集合
        :return: （需要比对的词语对） word_pairs_with_bodyparts结构如下：症状集合->同一部位症状集合->某一标准症状作为基准的症状集合->症状词语对
        '''
        symptom_list1, symptom_list2 = self.list_standard, self.list_symptom_input
        body_parts_len = self.body_parts_len

        # 不存在部位的描述词词对
        words_1 = symptom_list1[0]
        words_2 = symptom_list2[0]
        word_pairs_with_nobodyparts = []
        word_pairs_with_bodyparts = []
        for i in words_1:
            words = []
            for j in words_2:
                words.append([i, j])
            if len(words) != 0:
                word_pairs_with_nobodyparts.append(words)
        # i = int(1)
        for i in range(1, body_parts_len + 1):
            words = []
            for w1 in symptom_list1[i]:
                word_i = []
                for w2 in symptom_list2[i]:
                    word_i.append([w1, w2])
                if len(word_i) != 0:
                    words.append(word_i)
            if len(words) != 0:
                word_pairs_with_bodyparts.append(words)
        # 将词语对分成含有部位的症状对和不含部位词的症状对
        self.word_pairs_with_bodyparts, self.word_pairs_with_nobodyparts = word_pairs_with_bodyparts, word_pairs_with_nobodyparts
        # return word_pairs_with_bodyparts, word_pairs_with_nobodyparts

    def sim_sentence_v1(self):
        '''
        :return:
        '''
        word_pairs_with_bodyparts, word_pairs_with_nobodyparts = self.word_pairs_with_bodyparts, self.word_pairs_with_nobodyparts
        # 处理带有部位的症状
        max_sim_value = []
        for words_bodyspart in word_pairs_with_bodyparts:
            # print('\n这是同一部位症状集中的症状！！\n')
            for symptoms_words in words_bodyspart:
                # print('\n这是同一标准症状为基准的症状！！\n')
                # print(symptoms_words)
                sim_value_list = []
                for words in symptoms_words:
                    sim_value = self.judge_words(words)
                    sim_value_list.append(sim_value)
                # print(sim_value_list)
                sim_value_max = max(sim_value_list)
                max_sim_value.append(sim_value_max)
                # print(sim_value_max)
                # print('最相似词', symptoms_words[sim_value_list.index(sim_value_max)])
                # print('\n')
        # 处理无部位的症状
        for words in word_pairs_with_nobodyparts:
            # print('\n这是同一标准症状为基准的症状！！\n')
            # print(words)
            sim_value_list = []
            for w in words:
                sim_value = self.judge_words(w)
                sim_value_list.append(sim_value)
            # print('sim_value_list:', sim_value_list)
            sim_value_max = max(sim_value_list)
            max_sim_value.append(sim_value_max)
            # print('sim_value_max:', sim_value_max)
            # print('最相似词', words[sim_value_list.index(sim_value_max)])
            # print('\n')
        # 返回以标准症状为基准的症状对中的最大相似值，
        # 其中，如果存在含有部位的标准症状，没有对应的输入症状的，不计算该标准症状。
        sum_value = float()
        for i in max_sim_value:
            sum_value += i
        sum_value = sum_value / len(max_sim_value)
        return max_sim_value, sum_value

    # 查找在反义词词典中的字符，并返回所在反义词词典键值的value
    @staticmethod
    def result_of_antonym(word_set):
        '''
        :param word_set: 字符的set集合
        :return:[字符所在反义词词典键值的value,字符]
        '''
        word_value = []
        for character in word_set:
            for key in antonym_dict.keys():
                if character in key:
                    word_value.append([antonym_dict[key], character])
        return word_value

    # 判断词语对中是否存在一对反义词
    @staticmethod
    def judge_antonym_in_word(word_value1, word_value2):
        '''
        :param word_value1: 待判断的词语字符和其再反义词词典中的键值
        :param word_value2: 待判断的词语字符和其再反义词词典中的键值
        :return:
        '''
        # 反义词词典键值的集合
        value1_list = set()
        for i in word_value1:
            sss = i[0]
            value1_list.add(sss)
        value2_list = set()
        for i in word_value2:
            sss = i[0]
            value2_list.add(sss)
        # 求反义词词典键值的交集
        value_intersection = value2_list & value1_list
        # 如果没有交集，则说明不存在反义词，直接返回False
        if len(value_intersection) == 0:
            return False
        # 存在交集，则判断两个字符是否相同
        for i in value_intersection:
            # print(i)
            character_1 = 'c1'
            for j in word_value1:
                if i == j[0]:
                    character_1 = j[1]
                    # print('character', character_1)
            character_2 = 'c2'
            for j in word_value2:
                if i == j[0]:
                    character_2 = j[1]
                    # print('character', character_2)
            if character_1 == character_2:
                return False
        return True

    # 判断是否存在否定词，若两个词语都出现否定词和都不出现否定词同样返回False，若只出现一个否定词则返回True
    @staticmethod
    def judge_negative_word(word1_set, word2_set):
        '''
        :param word1_set: 词语1字的集合
        :param word2_set: 词语2字的集合
        :return: True 只出现一个否定词
                 False 其他情况
        '''
        has_negative_word = 0
        for f in negative_words_dict.keys():
            if f in word1_set:
                has_negative_word += 1
            if f in word2_set:
                has_negative_word += 1
        if has_negative_word == 1:
            return True
        return False

    # 根据两个词中相同汉字个数来计算其相似度
    def sim_word_zimian(self, word1, word2):
        '''
        :param word1: 待计算词语1
        :param word2: 待计算词语2
        :return: 词语相似值
        '''
        word1_list = set(word1)
        word2_list = set(word2)
        # 第一个词语的词典value和该字符的列表
        word1_value = self.result_of_antonym(word1_list)
        word2_value = self.result_of_antonym(word2_list)
        # has_antonym 是bool型变量，判断两个词语中是否存在一个反义词
        has_antonym = self.judge_antonym_in_word(word1_value, word2_value)
        # 判断词语中是否存在反义词
        has_negative_word = self.judge_negative_word(word1_list, word2_list)
        # 如果同时存在反义词和否定词，则判定其相似度为1
        if has_negative_word is True and has_antonym is True:
            sim = 1
            return sim
        # 如果只存在反义词，不存在否定词，则判定相似度为0。
        if has_antonym is True and has_negative_word is False:
            sim = 0
            return sim

        # 不存在反义词，则按照普通计算相似字符个数的方法计算相似值
        same_word = word1_list & word2_list
        word1_len = len(word1_list)
        word2_len = len(word2_list)
        same_word_len = len(same_word)
        sim = (same_word_len / word1_len + same_word_len / word2_len) / 2
        return sim

    # 将词林中的词读作列表，以便后续处理，返回值如[['Aa01A01=', '人', '士', '人物', '人士', '人氏', '人选'], ['Ae01B26#', '宣传工作者', '传播者']]
    def get_word_list(self):
        '''
        :return: 读取的词林中的词语列表
        '''
        file = self.cilin_file
        word_list = []
        f = open(file, 'r', encoding='utf-8')
        lines = f.readlines()
        for line in lines:
            line = line.replace('\n', '')
            l = line.encode('utf-8').decode('utf-8-sig')
            word = l.split(' ')
            word_list.append(word)
        self.word_list = word_list

    # 查找列表中与item相同的词语的编号，返回值以列表的形式
    def find_all_index(self, item):
        '''
        :param item: 待查找词语
        :return: 词林中的词语义项
        '''
        lists = self.word_list
        index = []
        for arr in lists:
            # print(arr)
            for i in arr:
                # print(i)
                if i == item:
                    # print(arr[0])
                    index.append(arr[0])
        return index

    # 计算给定两个词之间的相似度
    def sim_word(self, word1_indexs, word2_indexs):
        '''
        :param word1_indexs: 词语的义项列表
        :param word2_indexs: 词语的义项列表
        :return: 词语的最大相似值
        '''
        # 在同义词词林中查找两个词的节点位置
        # word_list = get_word_list('data/dict/同义词词林.txt')
        word1_index = word1_indexs
        word2_index = word2_indexs

        # 设定层数初值
        a = 0.65
        b = 0.8
        c = 0.9
        d = 0.96
        e = 0.5
        f = 0.1

        # 获取词语义项编号对
        index_pairs = []
        for i in word1_index:
            for j in word2_index:
                index_pairs.append([i, j])
        # print(index_pairs)    # 没问题

        # 根据公式计算相似度，有多个义项编号的需要交叉计算，即计算词语义项编号对的相似度
        sim_list = []
        for index_pair in index_pairs:
            # print('index_pair:', index_pair)
            # 判断词语义项编码的不同处
            symbols = self.word_index(index_pair)
            sim = None
            if symbols == '=':
                sim = 1
            elif symbols == '#':
                sim = e
            elif symbols == '@':
                sim = 0
                # sim_bool = True
            elif symbols == 1:
                sim = f
            else:
                if symbols == 2:
                    sym = a
                elif symbols == 3:
                    sym = b
                elif symbols == 4:
                    sym = c
                else:
                    sym = d
                # 获取分支层节点总数n
                n = self.get_node_number(index_pair)
                # 获取分支间的距离k
                k = self.get_distance_branch(index_pair)
                sim = sym * (math.cos(n * (math.pi) / 180)) * (n - k + 1) / n
            sim_list.append(str(sim))

        # 选择最大的相似度
        sim_list = sorted(sim_list, reverse=True)
        return float(sim_list[0])

    # 处理词语义项的编号，若在同一行内，
    # 则返回标记‘=’、‘#’、‘@’，否则返回哪一级不同
    @ staticmethod
    def word_index(index):
        '''
        :param index: 词语义项编号对
        :return: 范围为（1，5）的级别
        '''
        # 输入值为词语义项编号对，如果相同则返回标记，否则返回范围为（1，5）的级别
        index1 = index[0]
        index2 = index[1]
        if index1 == index2:
            return list(index2)[7]
        else:
            index1 = list(index1)
            index2 = list(index2)
            # print(index1,index2)
            i = int(1)
            for i in range(1, 8):
                if index1[i - 1] == index2[i - 1]:
                    i += 1
                else:
                    # 判断词语在哪一级别不同，范围为（1，5）
                    if i == 4 or i == 3:
                        return 3
                    if i == 5:
                        return 4
                    if i == 7 or i == 6:
                        return 5
                    return i

    # 获取分支层节点总数n
    def get_node_number(self, index_pair):
        '''
        :param index_pair: 词语义项
        :return: 分支层节点总数
        '''
        # 获取相同的层数num
        num = self.word_index(index_pair)
        word_lists = self.word_list
        node_num = 0
        # 根据num的返回值，找到最大相同子串
        if num == 1:
            return
        if num == 2:
            string = index_pair[0][0:1]
            # print(string)
            sub_str = 1
            set_len = 2
        elif num == 3:
            string = index_pair[0][0:2]
            sub_str = 2
            set_len = 4
        elif num == 4:
            string = index_pair[0][0:4]
            sub_str = 4
            set_len = 5
        else:
            string = index_pair[0][0:5]
            sub_str = 5
            set_len = 7
        # 判断义项编码中，所有存在公共子串的节点，即分支层节点总数
        # print('string=', string)
        node_set = set()
        for word in word_lists:
            # print(str(word[0]))
            if string == str(word[0])[0:sub_str]:
                # print(str(word[0])[0:sub_str])
                node_set.add(str(word[0])[0:set_len])
        node_num = len(node_set)
        # print(node_set)
        # print('node_num:',len(node_set))
        return node_num

    # 获取分支间的距离k
    def get_distance_branch(self, index_pair):
        '''
        :param index_pair: 词语义项组
        :return:  分支间的距离k        '''
        # 获取相同的层数num
        num = self.word_index(index_pair)
        # 根据num的返回值，找到不同的级别
        if num == 2:
            string1 = str(index_pair[0])[1]
            string2 = str(index_pair[1])[1]
            # print(string1 + string2)
            distance = abs(ord(string1) - ord(string2))
        elif num == 3:
            string1 = index_pair[0][2:4]
            string2 = index_pair[1][2:4]
            if string1[0] == '0' and string1[1] != '0':
                str1 = string1[1]
            else:
                str1 = string1
            if string2[0] == '0' and string2[1] != '0':
                str2 = string2[1]
            else:
                str2 = string2
            distance = abs(int(str1) - int(str2))

        elif num == 4:
            string1 = index_pair[0][4]
            string2 = index_pair[1][4]
            distance = abs(ord(string1) - ord(string2))
        else:
            string1 = index_pair[0][5:7]
            string2 = index_pair[1][5:7]
            if string1[0] == '0' and string1[1] != '0':
                str1 = string1[1]
            else:
                str1 = string1
            if string2[0] == '0' and string2[1] != '0':
                str2 = string2[1]
            else:
                str2 = string2
            distance = abs(int(str1) - int(str2))
        # 返回距离distance
        return distance

    # 判断是调用同义词词林相似度计算函数还是调用字面相似度计算函数
    def judge_words(self, words):
        '''
        :param words: 词语组
        :return: 词语相似值
        '''
        word1_index = self.find_all_index(words[0])
        word2_index = self.find_all_index(words[1])
        if len(word1_index) != 0 and len(word2_index) != 0:
            sim = self.sim_word(word1_index, word2_index)
        else:
            sim = self.sim_word_zimian(words[0], words[1])
        return sim


# 获取指定文档中的原始症状，处理后返回。
def get_standard_symptom(file):
    '''
    :param file: 标准症状文档
    :return: 标准症状列表
    '''
    f = open(file, 'r', encoding='utf-8')
    data = pd.read_csv(f)
    # 症型数量
    num_type = data.shape[0]
    # 原始症型-症状数据
    symptom_list = []
    for i in range(num_type):
        # 将csv文件中的主要症状和次要症状拼接起来,以空格分隔
        sen = data['主要症状'].loc[i]+' '+data['次要症状'].loc[i]
        symptom_list.append([data['症型'].loc[i], sen])
    # 列表存储所有处理过的标准症状
    symptom_list_after_process = []
    for sen in symptom_list:
        deal_sen = DealSentences(sen[1])
        deal_sen.pre_process_sen()
        s = deal_sen.process_sen()
        symptom_list_after_process.append([sen[0], s])
    return symptom_list_after_process


# 对比输入症状和标准症状文件中的症状对,返回倒序的相似值列表
def compare_symptom(input_symptom, standard_symptom_file):
    '''
    :param input_symptom: 输入症状
    :param standard_symptom_file: 标准症状文档
    :return: 症状证型相似值列表
    '''
    # 对输入症状进行预处理
    deal_input_symptom = DealSentences(input_symptom)
    deal_input_symptom.pre_process_sen()
    symptom_processed = deal_input_symptom.process_sen()
    # 获取预处理过的标准症状
    standard_symptom_list = get_standard_symptom(standard_symptom_file)
    sim_value_list = []
    for standard_symptom in standard_symptom_list:
        # 获取句子的相似值
        s = SimSentence(standard_symptom[1], symptom_processed, cilin_file='./dict/同义词词林.txt', body_parts_len=len(body_parts_dict))
        s.get_word_list()
        s.get_word_pairs()
        value_list, value_average = s.sim_sentence_v1()
        # 将症型名和相似值添加到列表中
        sim_value_list.append([standard_symptom[0], value_average])
    # 以sim值倒叙排序
    sim_value_list.sort(key=operator.itemgetter(1), reverse=True)
    # print(sim_value_list)
    return sim_value_list


if __name__ == '__main__':
    input_symptom_1 = '白带量多 带下色白 带下质稀薄 带下无特殊气味 身疲倦怠 周身乏力 四肢不温 纳少便溏 时感两足跗肿 面色萎黄 舌质淡胖 苔白腻 脉缓弱'
    result = compare_symptom(input_symptom_1, './data/带下.csv')
    print(result)