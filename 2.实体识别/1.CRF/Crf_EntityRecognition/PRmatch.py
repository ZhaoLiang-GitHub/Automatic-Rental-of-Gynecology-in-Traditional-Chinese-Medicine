from decimal import Decimal


def PRmatch(filepath1, filepath2):
    tagConcepts = []
    file_in = open(filepath1, 'r', encoding='utf-8')
    file_out = open(filepath2, 'w', encoding='utf-8')
    lines = file_in.readlines()
    # 计算准确率与召回率
    '''标注正确的数目'''
    x = 0
    '''标注数目'''
    y = 0
    '''原始概念数目'''
    z = 0

    word = ""
    initAnnotationFlag = ""
    testAnnotationFlag = ""
    '''概念行标识'''
    flag = False

    for line in lines:
        if line != '':
            # 去掉首尾空格
            line = line.strip('')
            # print(type(line))
            # 去掉文档开头# 0.738938
            if len(line) < 17:
                # print(line)
                continue
            # '''将一行使用tab健分开'''
            line1 = line.split("\t")
            # print(line1)
            # '''原文概念出现一次'''
            if line1[2] == 'B-symptom' or line1[2] == 'S-symptom' or line1[2] == 'S-pattern' or line1[2] == 'B-pattern' \
                    or line1[2] == 'B-treat' or line1[2] == 'S-treat' or line1[2] == 'B-disease' or line1[
                2] == 'S-disease':
                z += 1
                # '''标注概念出现一次'''
            # elif line1[2] == 'B-symptom' or line1[2] == 'S-symptom':
            #     z += 1
            line2 = line1[3].split('/')
            # print(line2[0])
            if line2[0] == 'B-symptom' or line2[0] == 'S-symptom' or line2[0] == 'S-pattern' or line2[0] == 'B-pattern' \
                    or line2[0] == 'B-treat' or line2[0] == 'S-treat' or line2[0] == 'B-disease' or line2[
                0] == 'S-disease':
                y += 1
                '''概念开始'''
            if line1[2] == 'B-symptom' or line1[2] == 'S-symptom' or line2[0] == 'B-symptom' or line2[0] == 'S-symptom' \
                    or line1[2] == 'B-pattern' or line1[2] == 'S-pattern' or line2[0] == 'B-pattern' or line2[
                0] == 'S-pattern' \
                    or line1[2] == 'B-treat' or line1[2] == 'S-treat' or line2[0] == 'B-treat' or line2[0] == 'S-treat' \
                    or line1[2] == 'B-disease' or line1[2] == 'S-disease' or line2[0] == 'B-disease' or line2[
                0] == 'S-disease':
                flag = True
            '''概念结束'''
            if line1[2] == 'S-symptom' or line1[2] == 'E-symptom' or line2[0] == 'S-symptom' or line2[0] == 'E-symptom' \
                    or line1[2] == 'S-pattern' or line1[2] == 'E-pattern' or line2[0] == 'S-pattern' or line2[
                0] == 'E-pattern' \
                    or line1[2] == 'S-treat' or line1[2] == 'E-treat' or line2[0] == 'S-treat' or line2[0] == 'E-treat' \
                    or line1[2] == 'S-disease' or line1[2] == 'E-disease' or line2[0] == 'S-disease' or line2[
                0] == 'E-disease':
                word += line1[0] + " "
                initAnnotationFlag += line1[2]
                testAnnotationFlag += line2[0]
                tagConcepts.append([word, initAnnotationFlag, testAnnotationFlag])
                # print(tagConcepts)
                '''初始化'''
                word = ""
                initAnnotationFlag = ""
                testAnnotationFlag = ""
                flag = False
            if flag:
                word += line1[0] + " "
                initAnnotationFlag += line1[2]
                testAnnotationFlag += line2[0]
    file_in.close()
    for tagConcept in tagConcepts:
        # print(tagConcept)
        if tagConcept[1] == tagConcept[2]:
            x += 1
    # print(str(x))
    # print(str(y))
    # print(str(z))
    p = x / y if (y) else 0
    # 正确率
    r = x / z if (z) else 0
    # 召回率
    f = 2 * p * r / (p + r) if (p + r) else 0
    # 保留三位小数
    p = Decimal(p).quantize(Decimal('0.000'))
    r = Decimal(r).quantize(Decimal('0.000'))
    f = Decimal(f).quantize(Decimal('0.000'))
    print("精确率：" + str(x) + "/" + str(y) + "=" + str(p) + "\n" + "召回率：" + str(x) + "/" + str(z) + "=" + str(
        r) + "\n" + "F1值：" + " f=" + str(f))
    print('-----------------------------------')
    file_out.write("模型实验结果如下：" + "\n")
    file_out.write("精确率：" + str(x) + "/" + str(y) + "=" + str(p) + "\n")
    file_out.write("召回率：" + str(x) + "/" + str(z) + "=" + str(r) + "\n")
    file_out.write("F1值：" + " f=" + str(f))
    file_out.close()


if __name__ == "__main__":
    PRmatch('./data/result.txt', './data/probability.txt')
