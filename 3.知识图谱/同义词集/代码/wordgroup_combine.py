import copy


# 对词语相似度计算出的同义词词组进行合并
def word_set_combine(read_file, write_file):
    f = open(read_file, 'r', encoding='utf-8')
    data = []
    # 获取同义词词集文件中的同义词词组
    for line in f.readlines():
        data.append(line.split())
    # 删除同义词词集中完全相同的词组
    del_samewords_difforder(data)
    # 删除同义词词组中的子集
    del_sub_set(data)
    # 合并相同词语个数超过一定数量的词组
    combine_subset(data)

    # 将处理后词语写入新文件中
    f_write = open(write_file, 'a', encoding='utf-8')
    for cell in data:
        cell_1 = set(cell)
        cell = list(cell_1)
        f_write.write(' '.join(cell)+'\n')
        f_write.flush()


# 判断词组是否完全相同
def judge_identical(word_list1, word_list2):
    '''
    :param word_list1: 待比较的列表1
    :param word_list2: 待比较的列表2
    :return: 两个列表相同返回True，不同返回False
    '''
    if len(word_list1) != len(word_list2):
        return False
    for cell in word_list1:
        if cell not in word_list2:
            return False
    return True


# 判断相同词语的个数
def judge_identical_num(word_list_1, word_list_2):
    '''
    :param word_list_1: 待比较的列表1
    :param word_list_2: 带比较的列表2
    :return: 两个列表中相同元素的个数
    '''
    # 将列表转化为集合，通过集合求交集判断词语列表相同词语的个数
    word_set_1 = set(word_list_1)
    word_set_2 = set(word_list_2)
    same_word_set = word_set_1 & word_set_2
    # print(same_word_set)
    return len(same_word_set)


# 删除同义词词组中的词组子集
def del_sub_set(word_list):
    '''
    :param word_list: 同义词词集
    :return: 删除词组子集的word_list
    '''
    word_list_1 = copy.deepcopy(word_list)
    for i in word_list:
        for j in word_list_1:
            set1 = set(i)
            set2 = set(j)
            if set1.issubset(set2) and set1 != set2:
                word_list.remove(i)
                break
    return word_list


# 删除同义词词集中完全相同的词组
def del_samewords_difforder(word_list):
    '''
    :param word_list:
    :return: 处理后的word_list
    '''
    word_list_1 = copy.deepcopy(word_list)
    num_log = []
    for i in range(len(word_list_1)):
        for j in range(i+1, len(word_list_1)):
            word_set_1 = set(word_list_1[i])
            word_set_2 = set(word_list_1[j])
            if word_set_1 == word_set_2 and j not in num_log:
                word_list.remove(word_list_1[j])
                num_log.append(j)
    return word_list


# 将相同词语个数超过50%的词组合并
def combine_subset(word_list):
    '''
    :param word_list:
    :return:
    '''
    word_list_1 = copy.deepcopy(word_list)
    # 记录被合并过的词组的序号
    num_log = []
    for i in range(len(word_list_1)):
        if i in num_log:
            continue
        for j in range(i, len(word_list_1)):
            # 如果当前词组被合并过一次，则跳过
            if j in num_log:
                continue
            # 获取当前词组的长度
            num_compare = len(word_list_1[j])
            # 当前词组与待比较词组中相同词语的个数
            num = judge_identical_num(word_list_1[i],word_list_1[j])
            if num == 0:
                continue
            # 如果相同词语个数是待比较词组中词语总数的一半，则将待比较词组并入当前词组
            if num/num_compare >=0.6:
                word_list[i].extend(word_list[j])
                num_log.append(j)
    # print(len(num_log))
    # print(len(set(num_log)))
    # exit()
    #　删除被合并过的词组
    for i in num_log:
        if word_list_1[i] in word_list:
            word_list.remove(word_list_1[i])
        else:
            print(word_list_1[i])
    return word_list


if __name__ == '__main__':
    word_set_combine('../output_9_26/symptom_synonyms_v0.txt', '../output_9_26/symptom_synonyms_v1.txt')