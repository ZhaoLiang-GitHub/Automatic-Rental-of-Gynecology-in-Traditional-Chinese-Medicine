class MaxForwardMatch:

    def __init__(self, dict_map, word_list):
        self.dict_map = dict_map
        self.word_list = word_list

    def match(self):
        """
        {月经不调:病名，气血两虚：证型}
        :param dict: user dict (key是实体,value是实体的类型)
        :param word_list: split word list
        :return: annotation listp
        """
        tag_list = []
        # global i
        i = 0
        # print(self.word_list)
        while (i < len(self.word_list)):

            # #匹配字符串10
            #     str10 = ''
            #     for m in range(i,i+10):
            #         str10 += str(word_list[i])
            #     str10 = word_list[i]+word_list[i+1]+word_list[i+2]+word_list[i+3]+word_list[i+4]+word_list[i+5]+word_list[i+6]+word_list[i+7]+word_list[i+8]+word_list[i+9]
            #     flag10 = False
            #     if str10 in dict_map:
            #         if(dict_map(str10)=='pattern'):
            #             tag_list.append('B-pattern')
            #             for j in range(i+1,i+9):
            #                 tag_list.append('I-pattern')
            #             tag_list.append('E-pattern')
            #         elif(dict_map(str10)=='disease'):
            #             tag_list.append('B-disease')
            #             for j in range(i+1,i+9):
            #                 tag_list.append('I-disease')
            #             tag_list.append('E-disease')
            #         elif(dict_map(str10)=='symptom'):
            #             tag_list.append('B-symptom')
            #             for j in range(i + 1, i + 9):
            #                 tag_list.append('I-symptom')
            #             tag_list.append('E-symptom')
            #         else:
            #             pass
            #         i+=10
            #         #下一行为空行的情况
            #         if len(word_list[i]) == 0:
            #             i+=1
            #         flag10 = True
            #         if(flag10):
            #             continue
            #
            # # 匹配字符串9
            #     str9 = ''
            #     for m in range(i,i+9):
            #         str9 += str(word_list[i])
            #     #str9 = word_list[i] + word_list[i + 1] + word_list[i + 2] + word_list[i + 3] + word_list[i + 4] + word_list[i + 5] + \
            #            #word_list[i + 6] + word_list[i + 7] + word_list[i + 8]
            #     flag9 = False
            #     if str9 in dict_map:
            #         if (dict_map(str9) == 'pattern'):
            #             tag_list.append('B-pattern')
            #             for j in range(i + 1, i + 8):
            #                 tag_list.append('I-pattern')
            #             tag_list.append('E-pattern')
            #         elif (dict_map(str10) == 'disease'):
            #             tag_list.append('B-disease')
            #             for j in range(i + 1, i + 8):
            #                 tag_list.append('I-disease')
            #             tag_list.append('E-disease')
            #         elif (dict_map(str10) == 'symptom'):
            #             tag_list.append('B-symptom')
            #             for j in range(i + 1, i + 8):
            #                 tag_list.append('I-symptom')
            #             tag_list.append('E-symptom')
            #         else:
            #             pass
            #         i += 9
            #         # 下一行为空行的情况
            #         if len(word_list[i]) == 0:
            #             i += 1
            #         flag10 = True
            #         if (flag9):
            #             continue
            #
            #
            #
            #
            # # 匹配字符串8
            #     str8 = ''
            #     for m in range(i,i+8):
            #         str8 += str(word_list[m])
            # #     str8 = word_list[i] + word_list[i + 1] + word_list[i + 2] + word_list[i + 3] + word_list[i + 4] + word_list[i + 5] + \
            # #            word_list[i + 6] + word_list[i + 7]
            #     flag8 = False
            #     if str8 in dict_map:
            #         if (dict_map(str8) == 'pattern'):
            #             tag_list.append('B-pattern')
            #             for j in range(i + 1, i + 7):
            #                 tag_list.append('I-pattern')
            #             tag_list.append('E-pattern')
            #         elif (dict_map(str8) == 'disease'):
            #             tag_list.append('B-disease')
            #             for j in range(i + 1, i + 7):
            #                 tag_list.append('I-disease')
            #             tag_list.append('E-disease')
            #         elif (dict_map(str8) == 'symptom'):
            #             tag_list.append('B-symptom')
            #             for j in range(i + 1, i + 7):
            #                 tag_list.append('I-symptom')
            #             tag_list.append('E-symptom')
            #         else:
            #             pass
            #         i += 8
            #         # 下一行为空行的情况
            #         if len(word_list[i]) == 0:
            #             i += 1
            #         flag8 = True
            #         if (flag8):
            #             continue
            #
            # # 匹配字符串7
            #     str7= ''
            #     for m in range(i,i+7):
            #         str7 += str(word_list[m])
            # #     str7 = word_list[i] + word_list[i + 1] + word_list[i + 2] + word_list[i + 3] + word_list[i + 4] + \
            # #            word_list[i + 5] + word_list[i + 6]
            #     flag7 = False
            #
            #     if str7 in dict_map:
            #         if (dict_map(str7) == 'pattern'):
            #             tag_list.append('B-pattern')
            #             for j in range(i + 1, i + 6):
            #                 tag_list.append('I-pattern')
            #             tag_list.append('E-pattern')
            #         elif (dict_map(str7) == 'disease'):
            #             tag_list.append('B-disease')
            #             for j in range(i + 1, i + 6):
            #                 tag_list.append('I-disease')
            #             tag_list.append('E-disease')
            #         elif (dict_map(str7) == 'symptom'):
            #             tag_list.append('B-symptom')
            #             for j in range(i + 1, i + 6):
            #                 tag_list.append('I-symptom')
            #             tag_list.append('E-symptom')
            #         else:
            #             pass
            #         i += 7
            #         # 下一行为空行的情况
            #         if len(word_list[i]) == 0:
            #             i += 1
            #         flag7 = True
            #         if (flag7):
            #             continue

            # # 匹配字符串6
            #     str6 = ''
            #     for m in range(i,i+6):
            #         str6 += str(word_list[m])
            #     # str6 = word_list[i] + word_list[i + 1] + word_list[i + 2] + word_list[i + 3] + word_list[i + 4] + \
            #     #        word_list[i + 5]
            #     flag = False
            #
            #     if str6 in dict_map:
            #         if (dict_map(str6) == 'pattern'):
            #             tag_list.append('B-pattern')
            #             for j in range(i + 1, i + 5):
            #                 tag_list.append('I-pattern')
            #             tag_list.append('E-pattern')
            #         elif (dict_map(str6) == 'disease'):
            #             tag_list.append('B-disease')
            #             for j in range(i + 1, i + 5):
            #                 tag_list.append('I-disease')
            #             tag_list.append('E-disease')
            #         elif (dict_map(str6) == 'symptom'):
            #             tag_list.append('B-symptom')
            #             for j in range(i + 1, i + 5):
            #                 tag_list.append('I-symptom')
            #             tag_list.append('E-symptom')
            #         else:
            #             pass
            #         i += 6
            #         # 下一行为空行的情况
            #         if len(word_list[i]) == 0:
            #             i += 1
            #         flag6 = True
            #         if (flag6):
            #             continue
            #
            # # 匹配字符串5
            #     str5 = ''
            #     for m in range(i,i+5):
            #         str5 += str(word_list[m])
            #     #str5 = word_list[i] + word_list[i + 1] + word_list[i + 2] + word_list[i + 3] + word_list[i + 4]
            #     flag = False
            #     if str5 in dict_map:
            #         if (dict_map(str5) == 'pattern'):
            #             tag_list.append('B-pattern')
            #             for j in range(i + 1, i + 4):
            #                 tag_list.append('I-pattern')
            #             tag_list.append('E-pattern')
            #         elif (dict_map(str5) == 'disease'):
            #             tag_list.append('B-disease')
            #             for j in range(i + 1, i + 4):
            #                 tag_list.append('I-disease')
            #             tag_list.append('E-disease')
            #         elif (dict_map(str5) == 'symptom'):
            #             tag_list.append('B-symptom')
            #             for j in range(i + 1, i + 4):
            #                 tag_list.append('I-symptom')
            #             tag_list.append('E-symptom')
            #         else:
            #             pass
            #         i += 5
            #         # 下一行为空行的情况
            #         if len(word_list[i]) == 0:
            #             i += 1
            #         flag5 = True
            #         if (flag5):
            #             continue

            # # 匹配字符串4
            #     str4= ''
            #     try:
            #         for m in range(i,i+4):
            #             str4 += str(word_list[m])
            #         #str4 = word_list[i] + word_list[i + 1] + word_list[i + 2] + word_list[i + 3]
            #
            #         if str4 in dict_map:
            #             if (dict_map(str4) == 'pattern'):
            #                 tag_list.append('B-pattern')
            #                 for j in range(i + 1, i + 3):
            #                     tag_list.append('I-pattern')
            #                 tag_list.append('E-pattern')
            #             elif (dict_map(str4) == 'disease'):
            #                 tag_list.append('B-disease')
            #                 for j in range(i + 1, i + 3):
            #                     tag_list.append('I-disease')
            #                 tag_list.append('E-disease')
            #             elif (dict_map(str4) == 'symptom'):
            #                 tag_list.append('B-symptom')
            #                 for j in range(i + 1, i + 3):
            #                     tag_list.append('I-symptom')
            #                 tag_list.append('E-symptom')
            #             else:
            #                 pass
            #             i += 4
            #             # 下一行为空行的情况
            #             if len(word_list[i]) == 0:
            #                 i += 1
            #             flag4 = True
            #             if (flag4):
            #                 continue
            #     except:
            #         pass
            str6 = ''
            flag6 = False
            try:
                # for m in range(i,i+3):
                #     str3 += str(self.word_list[m])
                str6 = self.word_list[i] + self.word_list[i + 1] + self.word_list[i + 2] + self.word_list[i + 3] + \
                       self.word_list[i + 4]+self.word_list[i + 5]
            except:
                pass
            if str6 != '' and str6 in self.dict_map and self.word_list[i + 5] != '':
                if (self.dict_map[str6] == 'pattern'):
                    tag_list.append('B-pattern')
                    tag_list.append('I-pattern')
                    tag_list.append('I-pattern')
                    tag_list.append('I-pattern')
                    tag_list.append('I-pattern')
                    tag_list.append('E-pattern')
                elif (self.dict_map[str6] == 'disease'):
                    tag_list.append('B-disease')
                    tag_list.append('I-disease')
                    tag_list.append('I-disease')
                    tag_list.append('I-disease')
                    tag_list.append('I-disease')
                    tag_list.append('E-disease')
                elif (self.dict_map[str6] == 'symptom'):
                    tag_list.append('B-symptom')
                    tag_list.append('I-symptom')
                    tag_list.append('I-symptom')
                    tag_list.append('I-symptom')
                    tag_list.append('I-symptom')
                    tag_list.append('E-symptom')
                elif (self.dict_map[str6] == 'treat'):
                    tag_list.append('B-treat')
                    tag_list.append('I-treat')
                    tag_list.append('I-treat')
                    tag_list.append('I-treat')
                    tag_list.append('I-treat')
                    tag_list.append('E-treat')
                else:
                    pass
                i += 6
                flag6 = True
                # 下一行为空行的情况
                if i < len(self.word_list) and self.word_list[i] == '':
                    i += 1
                    tag_list.append('')
                if (flag6):
                    continue
            else:
                pass

            # 匹配5个分词结果
            str5 = ''
            flag5 = False
            try:
                # for m in range(i,i+3):
                #     str3 += str(self.word_list[m])
                str5 = self.word_list[i] + self.word_list[i + 1] + self.word_list[i + 2] + self.word_list[i + 3] + self.word_list[i + 4]
            except:
                pass
            if str5 != '' and str5 in self.dict_map and self.word_list[i + 4] != '':
                if (self.dict_map[str5] == 'pattern'):
                    tag_list.append('B-pattern')
                    tag_list.append('I-pattern')
                    tag_list.append('I-pattern')
                    tag_list.append('I-pattern')
                    tag_list.append('E-pattern')
                elif (self.dict_map[str5] == 'disease'):
                    tag_list.append('B-disease')
                    tag_list.append('I-disease')
                    tag_list.append('I-disease')
                    tag_list.append('I-disease')
                    tag_list.append('E-disease')
                elif (self.dict_map[str5] == 'symptom'):
                    tag_list.append('B-symptom')
                    tag_list.append('I-symptom')
                    tag_list.append('I-symptom')
                    tag_list.append('I-symptom')
                    tag_list.append('E-symptom')
                elif (self.dict_map[str5] == 'treat'):
                    tag_list.append('B-treat')
                    tag_list.append('I-treat')
                    tag_list.append('I-treat')
                    tag_list.append('I-treat')
                    tag_list.append('E-treat')
                else:
                    pass
                i += 5
                flag5 = True
                # 下一行为空行的情况
                if i < len(self.word_list) and self.word_list[i] == '':
                    i += 1
                    tag_list.append('')
                if (flag5):
                    continue
            else:
                pass

            # 匹配4个分词结果
            str4 = ''
            flag4 = False
            try:
                # for m in range(i,i+3):
                #     str3 += str(self.word_list[m])
                str4 = self.word_list[i] + self.word_list[i + 1] + self.word_list[i + 2]+self.word_list[i + 3]
            except:
                pass
            if str4 != '' and str4 in self.dict_map and self.word_list[i + 3] != '':
                if (self.dict_map[str4] == 'pattern'):
                    tag_list.append('B-pattern')
                    tag_list.append('I-pattern')
                    tag_list.append('I-pattern')
                    tag_list.append('E-pattern')
                elif (self.dict_map[str4] == 'disease'):
                    tag_list.append('B-disease')
                    tag_list.append('I-disease')
                    tag_list.append('I-disease')
                    tag_list.append('E-disease')
                elif (self.dict_map[str4] == 'symptom'):
                    tag_list.append('B-symptom')
                    tag_list.append('I-symptom')
                    tag_list.append('I-symptom')
                    tag_list.append('E-symptom')
                elif (self.dict_map[str4] == 'treat'):
                    tag_list.append('B-treat')
                    tag_list.append('I-treat')
                    tag_list.append('I-treat')
                    tag_list.append('E-treat')
                else:
                    pass
                i += 4
                flag4 = True
                # 下一行为空行的情况
                if i < len(self.word_list) and self.word_list[i] == '':
                    i += 1
                    tag_list.append('')
                if (flag4):
                    continue
            else:
                pass

            # 匹配字符串3
            str3 = ''
            flag3 = False
            try:
                # for m in range(i,i+3):
                #     str3 += str(self.word_list[m])
                str3 = self.word_list[i] + self.word_list[i + 1] + self.word_list[i + 2]
            except:
                pass
            if str3 != '' and str3 in self.dict_map and self.word_list[i + 2] != '':
                if (self.dict_map[str3] == 'pattern'):
                    tag_list.append('B-pattern')
                    tag_list.append('I-pattern')
                    tag_list.append('E-pattern')
                elif (self.dict_map[str3] == 'disease'):
                    tag_list.append('B-disease')
                    tag_list.append('I-disease')
                    tag_list.append('E-disease')
                elif (self.dict_map[str3] == 'symptom'):
                    tag_list.append('B-symptom')
                    tag_list.append('I-symptom')
                    tag_list.append('E-symptom')
                elif (self.dict_map[str3] == 'treat'):
                    tag_list.append('B-treat')
                    tag_list.append('I-treat')
                    tag_list.append('E-treat')
                else:
                    pass
                i += 3
                flag3 = True
                # 下一行为空行的情况
                if i < len(self.word_list) and self.word_list[i] == '':
                    i += 1
                    tag_list.append('')
                if (flag3):
                    continue
            else:
                pass

            # 匹配字符串2
            str2 = ''
            flag2 = False
            try:
                # for m in range(i,i+2):
                #     str2 += str(self.word_list[m])
                str2 = self.word_list[i] + self.word_list[i + 1]
            except:
                pass
            if str2 != '' and str2 in self.dict_map and self.word_list[i + 1] != '':
                if (self.dict_map[str2] == 'pattern'):
                    tag_list.append('B-pattern')
                    tag_list.append('E-pattern')
                elif (self.dict_map[str2] == 'disease'):
                    tag_list.append('B-disease')
                    tag_list.append('E-disease')
                elif (self.dict_map[str2] == 'symptom'):
                    tag_list.append('B-symptom')
                    tag_list.append('E-symptom')
                elif (self.dict_map[str2] == 'treat'):
                    tag_list.append('B-treat')
                    tag_list.append('E-treat')
                else:
                    pass
                i += 2
                flag2 = True
                # 下一行为空行的情况
                if i < len(self.word_list) and self.word_list[i] == '':
                    i += 1
                    tag_list.append('')
                if (flag2):
                    continue
            else:
                pass

            # 匹配字符串1
            str1 = ''
            str1 += str(self.word_list[i])
            flag1 = False
            if str1 != '' and str1 in self.dict_map:  # 匹配成功
                flag1 = True
                if (self.dict_map[str1] == 'pattern'):
                    tag_list.append('S-pattern')
                elif (self.dict_map[str1] == 'disease'):
                    tag_list.append('S-disease')
                elif (self.dict_map[str1] == 'symptom'):
                    tag_list.append('S-symptom')
                elif (self.dict_map[str1] == 'treat'):
                    tag_list.append('S-treat')
                else:
                    pass
            else:  # 没有匹配到任何类型的实体 或者 字符串为空/换行符 的情况
                if str1 == '\n' or str1 == '':
                    i += 1
                    tag_list.append('')
                elif str != '':
                    i += 1
                    tag_list.append('O')
                else:
                    pass
                # 下一行存在且为空行
                if (i < len(self.word_list)):
                    if self.word_list[i] == '':
                        i += 1
                        tag_list.append('')
                    continue
            if (flag1):  # 匹配成功
                i += 1
                # 下一行为空行的情况
                if i < len(self.word_list) and self.word_list[i] == '':
                    i += 1
                    tag_list.append('')
                continue

        return tag_list
