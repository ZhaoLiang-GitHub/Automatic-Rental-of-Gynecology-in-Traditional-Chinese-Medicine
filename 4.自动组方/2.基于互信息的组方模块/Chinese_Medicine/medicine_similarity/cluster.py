import copy
import itertools
from medicine_similarity.data_utils import word_2_index, cut_by_num, index_2_word, write_csv, save_pickle


def duplicate_removal(relatives_list, root_name):
    """
    提取亲友团：之前的亲友是以词根和亲友的两两组合列表形式存在的，
    将每个词根的亲友提取出来组成该词根的亲友团列表
    :param relatives_list:亲友团列表[[[词根1,亲友1], [词根1, 亲友2], ...], [[词根2,亲友1], [词根2, 亲友2], ...], ...]
    :param root_name:词根列表
    :return:list_qyt:亲友团列表[[亲友1, 亲友2,...], [亲友1, 亲友2,...],...]
    """
    # print("relatives_list:", relatives_list)
    list_qyt = []
    for item in relatives_list:
        new2 = []
        for n in item:
            for q in n:
                new2.append(q)
        guo_du = list(set(new2))
        guo_du.sort(key=new2.index)
        if guo_du:
            guo_du.remove(root_name[relatives_list.index(item)])
        list_qyt.append(guo_du)
    # print("list_qyt:", list_qyt)
    return list_qyt


def del_by_correlation(new_list):
    """
    根据强相关的要求（词根1的list里有词根2，且词根2的list有词根1），删除不相关项。
    :param new_list:
    :return:
    """
    # 创建副本，便于进行删除操作
    list_index_2 = copy.deepcopy(new_list)
    # 保留强相关的项，其余删除
    for i, row in enumerate(new_list):
        for numb in row:
            if i not in new_list[numb]:
                list_index_2[i].remove(numb)
    return list_index_2


def create_double_set(list_num2):
    """
    创建二元组的set
    :param list_num2:
    :return:
    """
    double_set = set()  # 使用元组能够排除重复项
    for i, row in enumerate(list_num2):
        for item in row:
            two = [i, item]
            two.sort()
            double_set.add(tuple(two))
    return double_set


def merge_loop(double_set, root_name, file=None):
    """
    进行团合并操作，循环直到不能合并
    :param double_set:强相关的两两组合
    :param root_name:词根列表
    :param file:对聚类结果进行dump的目标路径
    :return:团成员最大数，最终的团
    """
    best_set = set()
    old_set = double_set
    num_list = []
    count_list = []
    group_list = []
    while len(old_set) > 0:
        # oldSet为需要继续进行合并操作的团
        print('成员数:', len(list(old_set)[0]))  # oldSet中团的成员数量
        print('个数:', len(old_set))   # oldSet中团的数量
        # print("old_set", old_set)
        num_list.append(len(list(old_set)[0]))
        count_list.append(len(old_set))
        group_list.append(old_set)
        best_set = old_set
        old_set = merge_group(old_set, double_set)    # 返回新组合成的团，对这些团继续进行合并操作
    # 若oldSet不存在，则说明聚类收敛、合并到最大的团了，无法继续合并了
    if file is not None:
        group_list = index_2_word(root_name, group_list)
        write_csv(['成员数', '个数', '团'], file, [num_list, count_list, group_list])
        save_pickle(file + '.pkl', group_list)
    # print("best_set", best_set)
    return len(list(best_set)[0]), best_set


def merge_group(old_set, double_set=None):
    """
    合并亲友团，亲友团成员+1
    方法：输入亲友团成员数量为n，所有亲友团相互比较，如果有n-1个成员相同，比较剩余两个不同成员的强相关性，如果强相关，组成新团
    :param old_set:输入的亲友团
    :param double_set:成员数为2的亲友团，用来验证是否强相关
    :return:得到亲友团成员数比输入亲友团成员数多1
    """
    if double_set is None:
        double_set = old_set
    new_set = set()
    old_list = list(old_set)
    item_len = len(old_list[0])
    for comb in itertools.combinations(old_list, 2):    # itertools.combinations创建一个迭代器，返回old_list中所有长度为2的子序列
        set1 = set(comb[0])
        set2 = set(comb[1])
        if len(set1 & set2) == item_len - 1:
            other_set = (set1 | set2) - (set1 & set2)
            other_tup = tuple(sorted(list(other_set)))
            if other_tup in double_set:
                new_set.add(tuple(sorted(list(set1 | set2))))
    return new_set
