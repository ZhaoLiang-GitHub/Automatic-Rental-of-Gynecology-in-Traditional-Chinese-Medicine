import re
import math
import random
import numpy as np
import jieba
jieba.initialize()


def create_dico(item_list):
    """
    创建字典，记录每个字符出现的频数
    :param item_list: 字符列表
    :return:
    """
    assert type(item_list) is list
    dico = {}
    for items in item_list:
        for item in items:
            if item not in dico:
                dico[item] = 1
            else:
                dico[item] += 1
    return dico


def create_mapping(dico):
    """
    根据词频字典dico创建两种映射字典
    :param dico:
    :return:
    """
    # 根据字典dico创建两种映射字典
    sorted_items = sorted(dico.items(), key=lambda x: (-x[1], x[0]))    # 按照词频对字典进行排序
    # for i, v in enumerate(sorted_items):
    #     print(i, v)
    id_to_item = {i: v[0] for i, v in enumerate(sorted_items)}  # id（根据词频排序从0开始）到word
    item_to_id = {v: k for k, v in id_to_item.items()}  # 反转映射
    return item_to_id, id_to_item


def zero_digits(s):
    """
    将所有数字替换成0
    :param s:
    :return:
    """
    return re.sub('\d', '0', s)


def iob2(tags):
    """
    检测标注模式，并将IOB1转换为IOB2
    :param tags:
    :return:
    """
    for i, tag in enumerate(tags):
        if tag == 'O':
            continue
        split = tag.split('-')
        if len(split) != 2 or split[0] not in ['I', 'B']:
            return False
        if split[0] == 'B':
            continue
        elif i == 0 or tags[i - 1] == 'O':  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
        elif tags[i - 1][1:] == tag[1:]:
            continue
        else:  # conversion IOB1 to IOB2
            tags[i] = 'B' + tag[1:]
    return True


def iob_iobes(tags):
    """
    将IOB标注模式改为IOBES
    :param tags:
    :return:
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag == 'O':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'B':
            if i + 1 != len(tags) and \
               tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('B-', 'S-'))
        elif tag.split('-')[0] == 'I':
            if i + 1 < len(tags) and \
                    tags[i + 1].split('-')[0] == 'I':
                new_tags.append(tag)
            else:
                new_tags.append(tag.replace('I-', 'E-'))
        else:
            raise Exception('Invalid IOB format!')
    return new_tags


def iobes_iob(tags):
    """
    将IOBES标注模式转换为IOB
    :param tags:
    :return:
    """
    new_tags = []
    for i, tag in enumerate(tags):
        if tag.split('-')[0] == 'B':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'I':
            new_tags.append(tag)
        elif tag.split('-')[0] == 'S':
            new_tags.append(tag.replace('S-', 'B-'))
        elif tag.split('-')[0] == 'E':
            new_tags.append(tag.replace('E-', 'I-'))
        elif tag.split('-')[0] == 'O':
            new_tags.append(tag)
        else:
            raise Exception('Invalid format!')
    return new_tags


def insert_singletons(words, singletons, p=0.5):
    new_words = []
    for word in words:
        if word in singletons and np.random.uniform() < p:
            new_words.append(0)
        else:
            new_words.append(word)
    return new_words


def get_seg_features(string):
    """
    序列的分割特征（或词特征），标注方法模仿boies
    :param string:
    :return:
    """
    seg_feature = []
    # print(string)
    jieba.load_userdict("data/diseases_train.txt")
    jieba.load_userdict("data/pattern_train.txt")
    jieba.load_userdict("data/symptom_train.txt")
    jieba.load_userdict("data/treat_train.txt")
    jieba.load_userdict("data/diseases_test.txt")
    jieba.load_userdict("data/pattern_test.txt")
    jieba.load_userdict("data/symptom_test.txt")
    jieba.load_userdict("data/treat_test.txt")
    for word in jieba.cut(string):
        # print(word)
        if len(word) == 1:
            seg_feature.append(0)   # 单个字符标注为0
        else:
            tmp = [2] * len(word)
            tmp[0] = 1  # 词的中间字符标注为1
            tmp[-1] = 3  # 词的末尾字符标注为3
            seg_feature.extend(tmp)
    return seg_feature


def create_input(data):
    inputs = list()
    inputs.append(data['chars'])
    inputs.append(data["segs"])
    inputs.append(data['tags'])
    return inputs


def full_to_half(s):
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        char = chr(num)
        n.append(char)
    return ''.join(n)


def cut_to_sentence(text):
    sentence = []
    sentences = []
    len_p = len(text)
    pre_cut = False
    for idx, word in enumerate(text):
        sentence.append(word)
        cut = False
        if pre_cut:
            cut = True
            pre_cut = False
        if word in u"。;!?\n":
            cut = True
            if len_p > idx+1:
                if text[idx+1] in ".。”\"\'“”‘’?!":
                    cut = False
                    pre_cut = True

        if cut:
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append("".join(list(sentence)))
    return sentences


def replace_html(s):
    s = s.replace('&quot;', '"')
    s = s.replace('&amp;', '&')
    s = s.replace('&lt;', '<')
    s = s.replace('&gt;', '>')
    s = s.replace('&nbsp;', ' ')
    s = s.replace("&ldquo;", "“")
    s = s.replace("&rdquo;", "”")
    s = s.replace("&mdash;", "")
    s = s.replace("\xa0", " ")
    return s


def input_from_line(line, char_to_id):
    """
    将一个语句处理成可预测的数据格式
    :param line: 一个语句sentence
    :param char_to_id:
    :return:
    """
    line = full_to_half(line)
    line = replace_html(line)
    inputs = list()
    inputs.append([line])
    line.replace(" ", "$")
    inputs.append([[char_to_id[char] if char in char_to_id else char_to_id["<UNK>"]
                   for char in line]])
    inputs.append([get_seg_features(line)])
    inputs.append([[]])
    return inputs


class BatchManager(object):
    """
    用于生成batch数据的batch管理类
    """
    def __init__(self, data,  batch_size):
        self.batch_data = self.get_batch(data, batch_size)  # 根据batch_size生成所有batch数据并存入batch_data列表
        self.len_data = len(self.batch_data)    # batch数量

    def get_batch(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        # sorted_data = sorted(data, key=lambda x: len(x[0]))   # 按照序列长度进行排序，此方法废弃
        batch_data = list()
        for i in range(num_batch):
            # batch_data.append(self.pad_data(sorted_data[i*batch_size:(i+1)*batch_size]))
            batch_data.append(self.pad_data(data[i*batch_size:(i+1)*batch_size]))
        return batch_data

    @staticmethod
    def pad_data(data):
        """
        对序列进行padding补长处理，padding字符全设置为0
        :param data:
        :return:
        """
        strings = []
        chars = []
        segs = []
        targets = []
        max_length = max([len(sentence[0]) for sentence in data])
        for line in data:
            string, char, seg, target = line
            padding = [0] * (max_length - len(string))
            strings.append(string + padding)
            chars.append(char + padding)
            segs.append(seg + padding)
            targets.append(target + padding)
        return [strings, chars, segs, targets]

    def iter_batch(self, shuffle=False):
        if shuffle:
            random.shuffle(self.batch_data)
        for idx in range(self.len_data):
            yield self.batch_data[idx]  # 利用生成器逐个返回batch
