import pandas as pd
from itertools import combinations
from math import log,sqrt
import pickle
def get_effect_Probability(f_medicine):
    '''
    得到药物功效值，单个值和功效对 出现词频
    :param f_medicine: 药物数据集
    :return: 单个功效出现词频，功效值对出现词频
    '''
    group={}
    all_group = []
    all_sigle = []
    for i in range(f_medicine.shape[0]):
        a = list(combinations(f_medicine['功效'].loc[i].split('、'), 2))
        all_group.extend(a)
        all_sigle.extend(f_medicine['功效'].loc[i].split('、'))
        for j in a :
            if j not in group.keys():
                group[j] = 1
            else:group[j] += 1
    sigle_effect_probability = []
    group_effect_probability = []
    for i in set(all_sigle):
        sigle_effect_probability.append((i,all_sigle.count(i)/len(all_sigle)))
    for i in group.keys():
        group_effect_probability.append((i, group[i]/len(all_group)))
    group_effect_probability = dict(group_effect_probability)
    sigle_effect_probability = dict(sigle_effect_probability)
    return group_effect_probability,sigle_effect_probability




f_prescription_add = '../方剂多样性推荐/公司数据_标准药物名称.csv'
f_medicine_add = '../方剂多样性推荐/药物数据集.csv'
f_channel_add = '../方剂多样性推荐/药物归经_12.txt'
f_property_add = '../方剂多样性推荐/药物性味_21.txt'
f_effect_add = '../方剂多样性推荐/药物功效_1058.txt'####功效数据集错了！！！！！！！！！

f_prescription = pd.read_csv(open(f_prescription_add, 'r', encoding='utf-8'))
f_medicine = pd.read_csv(open(f_medicine_add, 'r', encoding='utf-8'))
group_effect_probability, sigle_effect_probability = get_effect_Probability(f_medicine)
channel_list = [i.strip() for i in open(f_channel_add, 'r', encoding='utf-8').readlines()]
property_list = [i.strip() for i in open(f_property_add, 'r', encoding='utf-8').readlines()]
effect_list = list(sigle_effect_probability.keys())
effect_difference_matrix = []
print(group_effect_probability.keys())
for i in range(len(effect_list)):
	print(i)
	item = [0] * len(effect_list)
	a = sigle_effect_probability[effect_list[i]]
	for j in range(i + 1, len(effect_list)):
		print(j)
		b = sigle_effect_probability[effect_list[j]]
		# 非常小的正数，防止两个性味没有同事出现，联合概率为0的情况
		combine = 1.4E-45
		if (effect_list[i], effect_list[j]) in group_effect_probability.keys():
			combine = group_effect_probability[(effect_list[i], effect_list[j])]
		elif (effect_list[j], effect_list[i]) in group_effect_probability.keys():
			combine = group_effect_probability[(effect_list[j], effect_list[i])]

		flag_1, flag_2, flag_3 = 1, 1, 1
		h_pre = -a * log(a) - (1 - a) * log((1 - a))
		h_suf = -b * log(b) - (1 - b) * log((1 - b))
		param_1 = a - combine
		param_2 = b - combine
		param_3 = 1 + combine - a - b
		if param_1 == 0:
			flag_1 = 0
			param_1 = 1
		if param_2 == 0:
			flag_2 = 0
			param_2 = 1
		if param_3 == 0:
			flag_3 = 0
			param_3 = 1
		h_pre_suf = -combine * log(combine) - flag_1 * param_1 * log(param_1) - flag_2 * param_2 * log(
			param_2) - flag_3 * param_3 * log(param_3)
		result = (h_pre + h_suf - h_pre_suf) / (sqrt(h_pre) * sqrt(h_suf))
		result = 1 - result  # 计算差异度
		item[j] = result
	effect_difference_matrix.append(item)

f = open('药物功效差异度.txt','wb')
pickle.dump(effect_difference_matrix,f)