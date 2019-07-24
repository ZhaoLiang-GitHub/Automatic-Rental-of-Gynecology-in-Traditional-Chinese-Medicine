from math import log
from medicine_similarity.data_utils import cut_by_num


def calculate_correlation(combine_name, combine_fre, root_fre):
    """
    计算每个两两组合之间的互信息
    :param combine_name: 组合名称列表
    :param combine_fre: 组合出现的频率列表
    :param root_fre: 词根名称列表
    :return:
    """
    correlation = []  # 关联度系数
    for i in range(len(combine_name)):
        flag_1, flag_2, flag_3 = 1, 1, 1
        pre, suf = combine_name[i]
        # 熵的计算公式有点不对劲？
        h_pre = -root_fre[pre] * log(root_fre[pre]) - (1 - root_fre[pre]) * log((1 - root_fre[pre]))
        h_suf = -root_fre[suf] * log(root_fre[suf]) - (1 - root_fre[suf]) * log((1 - root_fre[suf]))
        param_1 = root_fre[pre] - combine_fre[i]
        param_2 = root_fre[suf] - combine_fre[i]
        param_3 = 1 + combine_fre[i] - root_fre[pre] - root_fre[suf]
        if param_1 == 0:
            flag_1 = 0
            param_1 = 1
        if param_2 == 0:
            flag_2 = 0
            param_2 = 1
        if param_3 == 0:
            flag_3 = 0
            param_3 = 1
        h_pre_suf = -combine_fre[i] * log(combine_fre[i]) - flag_1 * param_1 * log(
            param_1) - flag_2 * param_2 * log(param_2) - flag_3 * param_3 * log(param_3)
        result = h_pre + h_suf - h_pre_suf
        # result = combinations_fre[i] * log(combinations_fre[i]/(list_fre[pre]*list_fre[suf]))
        correlation.append(result)
    return correlation


def create_relatives(root_name, data, max_relatives_nums):
    """
    根据互信息得到每项的亲友团
    :param root_name:所有词根的list
    :param data:dataFrame，{组合，关联度系数}
    :param max_relatives_nums:限制亲友团的最大个数
    :return:嵌套list，表示所有项的亲友团:[[组合1,组合2...],...]
    """
    relatives_list = [[] for i in range(len(root_name))]
    length = data.shape[0]
    for i in range(length):
        # 先遍历所有两两组合的互信息，并分别添加到每个变量的列表中
        words = data['组合'][i]
        # words = data['药物'][i]
        pre_index = root_name.index(words[0])
        relatives_list[pre_index].append(words)
        suf_index = root_name.index(words[1])
        relatives_list[suf_index].append(words)
    return cut_by_num(relatives_list, max_relatives_nums)  # 对互信息列表进行裁剪，得到的就是亲友团
