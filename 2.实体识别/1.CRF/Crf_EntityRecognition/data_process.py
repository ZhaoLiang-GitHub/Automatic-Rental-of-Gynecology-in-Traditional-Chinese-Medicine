from imp import reload
import jieba
import jieba.posseg
import jieba.analyse
import sys
from utils import *

reload(sys)


def segmentation(str):
    # jieba.load_userdict('./data1/dict_train/all_dict.txt')
    if len(str) != 0:
        cut_sentence = []
        # key_words = jieba.analyse.extract_tags(str,topK=10)
        segment_list = jieba.posseg.cut(str)  # 精确模式
        for item in segment_list:
            cut_sentence.append((item.word, item.flag))
        return cut_sentence
    else:
        pass


def cut_sentence(file1, file2):
    file_in = open(file1, 'r', encoding='utf-8')
    file_out = open(file2, 'w', encoding='utf-8')
    lines = file_in.readlines()
    cut_sentences = []
    for line in lines:
        line = line.strip()
        if line != '' and line != '\n':
            line = line.strip('\n')
            cut_sentence = segmentation(line)  # cut_sentence 是分好词的句子
            for item in cut_sentence:
                word = item[0]
                flag = item[1]
                file_out.write(str(word) + '    ' + str(flag) + '\n')
            cut_sentences.append(cut_sentence)
            file_out.write('\n')


def combineConceptAnnotation(file_path1, file_path2, dic_path):
    disease_list = changetoList(dic_path + '/disease_dict.txt')
    treat_list = changetoList(dic_path + '/treat_dict.txt')
    symptom_list = changetoList(dic_path + '/symptom_dict.txt')
    pattern_list = changetoList(dic_path + '/pattern_dict.txt')
    dict = changeListToDict(symptom_list, treat_list, disease_list, pattern_list)
    annotation(file_path1, file_path2, dict)



