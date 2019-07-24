def entity_extract():
    """
    从实体识别的结果中提取出实体词
    输入：result.txt  crf识别结果文件
    :return: result2.txt  识别出的所有实体词
    """
    f = open('./data/result.txt', 'r', encoding='utf-8')
    f2 = open('./data/result2.txt', 'w', encoding='utf-8')
    lines = f.readlines()
    word = ""
    testAnnotationFlag = ""
    tagConcepts = []
    for line in lines:
        # print(line)
        if len(line) >= 12:
            line = line.strip('\n').split('\t')
            line1 = line[3].split('/')
            if line[0] == str(1):
                word = line[0]
                testAnnotationFlag = line1[0]
                tagConcepts.append([word, testAnnotationFlag])
                word = ''
                testAnnotationFlag = ''
                # print(line1[0])
            elif (line1[0] == 'S-pattern' or line1[0] == 'S-disease' or line1[0] == 'S-symptom' or line1[0]
                  == 'S-treat'):
                word = line[0]
                testAnnotationFlag = line1[0]
                tagConcepts.append([word, testAnnotationFlag])
                word = ''
                testAnnotationFlag = ''
            elif (line1[0] == 'B-pattern' or line1[0] == 'B-disease' or line1[0] == 'B-symptom' or line1[0]
                  == 'B-treat' or line1[0] == 'I-pattern' or line1[0] == 'I-disease' or line1[0] == 'I-symptom' or
                  line1[0]
                  == 'I-treat'):
                word += line[0]
            elif (line1[0] == 'E-pattern' or line1[0] == 'E-disease' or line1[0] == 'E-symptom' or line1[0]
                  == 'E-treat'):
                word += line[0]
                testAnnotationFlag = line1[0]
                tagConcepts.append([word, testAnnotationFlag])
                word = ''
                testAnnotationFlag = ''
    for tagConcept in tagConcepts:
        concept = tagConcept[0]
        f2.write(concept + '\n')
    f.close()
    f2.close()


def check_new_word():
    """
    检测实体识别中的新词
    :return:
    """
    with open('./data/result2.txt', encoding='utf8') as f:
        result = f.readlines()
    with open('./data/dict_train/all_dict.txt', encoding='utf8') as f:
        all = f.readlines()
    with open('./data/dict_test/all_dict.txt', encoding='utf8') as f:
        test = f.readlines()
    new_list = []
    new_list_all = []
    for word in result:
        if (word not in all):
            new_list_all.append(word)
            if (word in test):
                new_list.append(word)
    new_list = list(set(new_list))
    new_list_all = list(set(new_list_all))
    print('共检测到新词：', len(new_list_all), '个')
    print('其中测试集中已标注的新词：', len(new_list), '个')
    print('基准准确率（下限）：', len(new_list), '/', len(new_list_all), '=', len(new_list) / len(new_list_all))
    with open('./data/new_word_all.txt', 'w', encoding='utf8') as f:
        f.writelines(new_list)


if __name__ == '__main__':
    entity_extract()
    check_new_word()
