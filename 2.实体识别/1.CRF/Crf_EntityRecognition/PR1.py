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
                print(line)
                continue
            # '''将一行使用tab健分开'''
            line1 = line.split("\t")
            # print(line1)
            # '''原文概念出现一次'''
            if line1[2] == 'B-3' or line1[2] == 'S-3' or line1[2] == 'S-1' or line1[2] == 'B-1' \
                    or line1[2] == 'B-2' or line1[2] == 'S-2' or line1[2] == 'B-0' or line1[
                2] == 'S-0':
                z += 1
                # '''标注概念出现一次'''
            # elif line1[2] == 'B-3' or line1[2] == 'S-3':
            #     z += 1
            line2 = line1[3].split('/')
            # print(line2[0])
            if line2[0] == 'B-3' or line2[0] == 'S-3' or line2[0] == 'S-1' or line2[0] == 'B-1' \
                    or line2[0] == 'B-2' or line2[0] == 'S-2' or line2[0] == 'B-0' or line2[
                0] == 'S-0':
                y += 1
                '''概念开始'''
            if line1[2] == 'B-3' or line1[2] == 'S-3' or line2[0] == 'B-3' or line2[0] == 'S-3' \
                    or line1[2] == 'B-1' or line1[2] == 'S-1' or line2[0] == 'B-1' or line2[
                0] == 'S-1' \
                    or line1[2] == 'B-2' or line1[2] == 'S-2' or line2[0] == 'B-2' or line2[0] == 'S-2' \
                    or line1[2] == 'B-0' or line1[2] == 'S-0' or line2[0] == 'B-0' or line2[
                0] == 'S-0':
                flag = True
            '''概念结束'''
            if line1[2] == 'S-3' or line1[2] == 'E-3' or line2[0] == 'S-3' or line2[0] == 'E-3' \
                    or line1[2] == 'S-1' or line1[2] == 'E-1' or line2[0] == 'S-1' or line2[
                0] == 'E-1' \
                    or line1[2] == 'S-2' or line1[2] == 'E-2' or line2[0] == 'S-2' or line2[0] == 'E-2' \
                    or line1[2] == 'S-0' or line1[2] == 'E-1' or line2[0] == 'S-0' or line2[
                0] == 'E-1':
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
        print(tagConcept)
        if tagConcept[1] == tagConcept[2]:
            x += 1
    print(str(x))
    print(str(y))
    print(str(z))
    p = x / y if (y) else 0
    # 正确率
    r = x / z if (z) else 0
    # 召回率
    f = 2 * p * r / (p + r) if (p + r) else 0
    # 保留三位小数
    p = Decimal(p).quantize(Decimal('0.000'))
    r = Decimal(r).quantize(Decimal('0.000'))
    f = Decimal(f).quantize(Decimal('0.000'))
    print("正确率：" + str(x) + "/" + str(y) + "=" + str(p) + "\n" + "召回率：" + str(x) + "/" + str(z) + "=" + str(
        r) + "\n" + "F-Measure：" + " f=" + str(f))
    file_out.write("mutual_result_5的统计结果如下：" + "\n")
    file_out.write("正确率：" + str(x) + "/" + str(y) + "=" + str(p) + "\n")
    file_out.write("召回率：" + str(x) + "/" + str(z) + "=" + str(r) + "\n")
    file_out.write("F-Measure：" + " f=" + str(f))
    file_out.close()


if __name__ == "__main__":
    PRmatch('./data/result5.txt', './data/probability5.txt')
