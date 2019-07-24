import os
import codecs
import subprocess
import pandas as pd
from MaxForwardMatch_byWords import *


# 将字典list转换成键值对
def changeListToDict(symptom, treat, disease, pattern):
    ner_dict = []
    for item in symptom:
        ner_dict.append((item, 'symptom'))
    for item in treat:
        ner_dict.append((item, 'treat'))
    for item in disease:
        ner_dict.append((item, 'disease'))
    for item in pattern:
        ner_dict.append((item, 'pattern'))
    ner_dict = dict(ner_dict)
    return ner_dict


# 读取字典，将字典转换成list
def changetoList(infile):
    input_data = codecs.open(infile, 'r', 'utf-8')
    list = []
    for line in input_data.readlines():
        list.append(line.strip())
    return list


# 生成学习训练命令
def GenerateLearnCommand(command1):
    commend_file = open('./crf/learn_commend.bat', 'w', encoding='utf-8')
    commend_file.write(command1)
    commend_file.close()


# 生成测试训练命令
def GenerateMutualTestCommend(command2):
    commend_file = open('./crf/crossLearn_commend.bat', 'w', encoding='utf-8')
    commend_file.write(command2)
    commend_file.close()


# 执行.bat批处理文件
def run_bat(batfile_path):
    # p = subprocess.Popen("E:\\5_02code\\NER\\crf\\learn_commend.bat", stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    p = subprocess.Popen(batfile_path, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    curline = p.stdout.readline()
    while (curline != b''):
        curline = curline.decode('gb2312')
        print(curline)
        curline = p.stdout.readline()
    p.wait()
    print(p.returncode)


def annotation(file_path1, file_path2, dict):
    fp = open(file_path1, 'r', encoding='utf-8')
    fo = open(file_path2, 'w', encoding='utf-8')
    lines = fp.readlines()
    total_list = []
    an_list = []
    j = 0
    for line in lines:
        j += 1
        if (line == '\n' or line == ''):
            total_list.append('')
            an_list.append('')
        else:
            # list = line.strip().split()
            list = line.split('    ')
            total_list.append(list[0])
            an_list.append(list[1].strip())
    word_list = total_list
    mfmatch = MaxForwardMatch(dict, word_list)
    tag_list = mfmatch.match()
    for i in range(len(word_list)):
        if (word_list[i] != '' and tag_list[i] != ''):
            fo.write(word_list[i] + '    ' + an_list[i] + '    ' + tag_list[i] + '\n')
        else:
            fo.write('\n')
    fp.close()
    fo.close()


if __name__ == '__main__':
    os.chdir(r'E:\5_02code\EntityRecognition_byCRF\crf')
    run_bat("E:\\5_02code\\EntityRecognition_byCRF\\crf\\learn_commend.bat")
    run_bat("E:\\5_02code\\EntityRecognition_byCRF\\crf\\crossLearn_commend.bat")