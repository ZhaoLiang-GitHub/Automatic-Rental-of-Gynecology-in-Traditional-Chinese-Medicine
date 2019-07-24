
import os


# 通过计算词语相似度得到词语的同义词，针对字数不同的词语采用不同的阈值
def get_sim_words(words):
    # judge the length of the word
    words_len = words_length(words)

    # 判断两字词的情况
    if words_len[0] == 2:
        if words_len[1] == 2:
            gate = 0.9
        else:
            gate = 0.8
    elif words_len[0] == 3:
        if words_len[1] == 3:
            gate = 0.8
        else:
            gate = 0.75
    else:
        gate = 0.7

    if sim_word_zimian(words[1], words[0]) >= gate:
        return True
    return False

def sim_symptom(read_file, write_file,other_file):
    '''
    :param read_file:带读取的文件
    :param write_file: 同义词写入的文件
    :param other_file: 没有同义词的文件
    :return:
    '''

    f = open(read_file, 'r', encoding='utf-8')
    if os.path.exists(write_file):
        print('文件已存在，请重新输入文件名！')
        return
    f_write = open(write_file, 'w', encoding='utf-8')
    f_other = open(other_file, 'a', encoding='utf-8')
    data = f.read()
    data_list_1 = data.strip().split()
    data_list = list(set(data_list_1))
    tyc = []
    for word_i in data_list:
        word_set = []
        word_set.append(word_i)
        for word_j in data_list:
            if word_j ==word_i:
                continue
            if get_sim_words([word_i,word_j]):
                word_set.append(word_j)
        if len(word_set) == 1:
            f_other.write(word_set[0]+'\n')
            f_other.flush()
            continue
        tyc.append(word_set)
    word_count = get_count_num(tyc)
    f_write.write('无重复词语共%d个，%d个同义词词组，覆盖率为%.4f\n' % (word_count, len(tyc), word_count/5175))
    f_write.flush()
    f_write.write('=' * 88 + '\n')
    f_write.flush()
    for cell in tyc:
        line = ' '.join(cell)
        f_write.write(line+'\n')
        f_write.flush()


# 计算词语的长度
def words_length(words):
    word_len = []
    for cell in words:
        word_len.append(len(cell))
    word_len = sorted(word_len)
    return word_len


# 计算同义词集中无重复词语的数量
def get_count_num(tyc):
    '''
    :param tyc:
    :return:
    '''
    word = set()
    for cell in tyc:
        for c in cell:
            word.add(c)
    return len(word)


# 字面相似度计算
def sim_word_zimian(word1,word2):
    word1_list = set(word1)
    word2_list = set(word2)
    same_word = word1_list&word2_list
    # print(len(same_word))
    word1_len = len(word1_list)
    word2_len = len(word2_list)
    same_word_len = len(same_word)
    sim = (same_word_len/word1_len + same_word_len/word2_len)/2
    return sim


if __name__ == '__main__':

    sim_symptom('../data_new/pattern.txt', '../output_9_26/pattern_output.txt', '../output_9_26/pattern_other.txt')
    # sim_symptom('../data_new/treat.txt', '../output_9_26/treat_output.txt', '../output_9_26/treat_other.txt')
