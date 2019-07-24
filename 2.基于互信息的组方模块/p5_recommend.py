# -*- coding: utf-8 -*-
import utils
import p4_validate as clus4
import whoosh_op as who
import pandas as pd
from collections import Counter
import gol


def recomment_main(insert):
    """
    推荐主程序，综合聚类结果和whoosh进行搜索推荐
    :param insert:
    :return:
    """
    # insert = ['带下赤白', '四肢乏力'] #全包含
    # insert = ['头痛', '赤白带下', '腹痛', '饮食减少', '经闭', '风寒']  # 团
    # insert = ['经闭', '头晕', '食症','恶心','风寒']#不包含

    list_name = utils.load_pickle('list_name.txt')
    series = utils.load_pickle('series.pkl')  # 症状series
    group_list = utils.group_clean2('data/group10.csv.pkl')
    group_count = clus4.count_prescript(series, list_name, group_list)
    # 存放索引的文件夹
    indexfile = 'index/'
    # 同义词词典路径
    dictfile = 'data/clean2.txt'
    # 方剂数据集路径
    # datasetfile = 'data/symptom_entity.csv'
    # 创建索引
    # who.get_index(datasetfile, indexfile)
    # 用户查询

    # 和查询
    my_query = who.add_synonym(dictfile, insert)
    result = who.get_recommend(indexfile, my_query)
    num_4_group = 2
    num_4_other = 2
    if (result):  # 如果可以直接匹配
        print('可以直接匹配')
        flag=0      #匹配团标签
        #使用症状团进行推理
        for index, group in enumerate(group_list):
            if (utils.is_in(insert,group) and len(insert)!=len(group)):
                flag=1
                ids = group_count.loc[index, 'id']
                print('症状团',group)
                print_info(ids,4)
        if not flag:        #不匹配症状团
            print_info(result,1)
        return 1, result
    else:
        # 从文件中初始化对象
        word_map = utils.gene_dic_2('data/clean2.txt')
        # insert标准化为同义词的标准词
        insert = [word_map[x] for x in insert]
        insert = utils.delete_duplicate(insert)  # 去重 不考虑重复词权重
        group_new = []
        # 查找insert包含的团
        match_symptom = []
        for index, group in enumerate(group_list):
            if (utils.is_in(group, insert)):
                match_symptom.extend(group)
                group_new.append(index)  # 团的列表
        group_new.sort(reverse=True)
        other_symptom = list(set(insert)-set(match_symptom))
        series_ids = []  # 症状团对应的方剂id
        group_clean = []  # 对团进行清理，删除重复包含的团
        for index in group_new:
            group = group_count.loc[index, '聚类']
            if (utils.is_in2(group, group_clean)):
                continue
            group_clean.append(group)
            ids = group_count.loc[index, 'id']
            print('药物团', group)
            # series_ids.append((group,ids))
            for i,id in enumerate(ids):
                if i>num_4_group-1:
                    break
                series_ids.append(str(id))
        # 存在对应的方剂
        if (len(series_ids) > 0):
            # 使用whoosh搜索剩余症状的方剂
            my_query = who.add_synonym2(dictfile, insert)  # or查询语句
            recommend_result = who.get_recommend(indexfile, my_query)
            recommend_result = recommend_result[:num_4_other]
            series_ids.extend(recommend_result)
            # series_ids = list(set(series_ids)) #顺序可能改变
            series_ids = utils.delete_duplicate(series_ids)  # 不会改变顺序
            print('存在症状团匹配')
            print_info(series_ids,2)
            return 2, series_ids
        # 没有对应的方剂
        else:
            print('没有症状团匹配，or搜索')
            my_query = who.add_synonym2(dictfile, insert)  # or查询语句
            result = who.get_recommend(indexfile, my_query)
            print_info(result,3)
            return 3, result

def recommend_test():
    """
    使用医案数据集做测试，计算准确率
    :return:
    """
    '''全局变量'''
    for group_num in [20]:
        group_num = str(group_num)
        # group_num = str(20)
        '''导入文件'''
        list_name = utils.load_pickle('list_name.txt')
        series = utils.load_pickle('series.pkl')  # 症状series
        group_list = utils.group_clean2('data/group' + group_num + '.csv.pkl')
        group_count = clus4.count_prescript(series, list_name, group_list)
        # group_count = pd.read_csv('data/count.csv')
        ''''''
        script_df = pd.read_csv('data/symptom_entity.csv', encoding='utf8')  # 方剂实体
        df = pd.read_csv('data/ZYFKX_2.csv', encoding='utf8')  # 中医妇科学病案
        '''添加列'''
        col_name = df.columns.tolist()
        col_name.insert(5, 'rec_med')  # 推荐药
        df = df.reindex(columns=col_name)
        ''''''
        count = 0
        precision_all = 0  # 准确率
        index_dict = {}  # 标记病案序号和方剂数
        with open('test/recommend_group/rec' + group_num + '_all.csv', 'w', encoding='utf8') as f:
            for index, row in df.iterrows():
                if pd.isnull(row["entity"]):
                    continue
                if pd.isnull(row["med"]):
                    continue
                med = row["med"].split('、')
                entity = row["entity"].split()
                num, result = recomment_main(entity)
                print(result)
                index_dict[index] = len(result)
                f.write(row['病案'] + ',' + row["entity"] + ',' + str(num) + ',')
                if not result:
                    print(index, '无推荐结果')
                    f.write('无推荐结果,')
                else:
                    f.write(' '.join(result) + ',')
                if num != 2:
                    f.write(' '.join(med) + ',,\n')
                else:
                    script_med_dic = {}
                    script_med_list = []
                    for i, id in enumerate(result):
                        # if i > 2:     # 取前8个药方的所有药物
                        #     break
                        script_row = script_df.query('序号==' + str(id))
                        script_med = script_row['处方'].values
                        script_med = script_med[0]
                        for word in script_med.split('、'):
                            if (word not in script_med_list):
                                script_med_list.append(word)
                    intersection = set(script_med_list) & set(med)
                    precision = len(intersection) / len(med)
                    count += 1
                    precision_all += precision
                    f.write(' '.join(med) + ',' + ' '.join(script_med_list) + ',')
                    f.write(str(precision) + ',' + str(len(result)) + '\n')
        utils.save_pickle('test/index_dict' + group_num + '.pkl', index_dict)
    # print(precision_all/count)
def id_2_prescript(ids):
    """
    方剂id转换为方剂信息
    :param ids:
    :return:
    """
    script_df = pd.read_csv('data/prescription_10000.csv', encoding='utf8')  # 方剂实体
    new_list = []
    for id in ids:
        pres_obj = {}
        script_row = script_df.query('id==' + str(id))
        pres_obj['方剂名'] = script_row['title'].values[0]
        pres_obj['药物组成'] = script_row['药物组成'].values[0]
        pres_obj['药物'] = script_row['标准药物名称'].values[0]
        pres_obj['用法用量'] = script_row['用法用量'].values[0]
        pres_obj['主治'] = script_row['主治'].values[0]
        new_list.append(pres_obj)
    return new_list
def cal_main_med(prescript_list):
    """
    核心药物
    :return:
    """
    med_list = []
    for i, prescript in enumerate(prescript_list):
        med_str =prescript['药物']
        med_list.extend(med_str.split('、'))
    med_count = Counter(med_list)
    most_common = med_count.most_common(4)
    return most_common
def print_info(ids,flag):
    prescript_list = id_2_prescript(ids)
    main_med = cal_main_med(prescript_list)
    if(flag==4 or flag==3):
        for i, prescript in enumerate(prescript_list):
            if (i > 2):
                break
            print('方剂名：  ', prescript['方剂名'])
            print('推荐药物：',prescript['药物组成'])
            print('用法用量：',prescript['用法用量'])
            print('主治:     ',prescript['主治'])
            print()
    elif(flag==1):
        for i, prescript in enumerate(prescript_list):
            if (i > 5):
                break
            print('方剂名：  ', prescript['方剂名'])
            print('推荐药物：',prescript['药物组成'])
            print('用法用量：',prescript['用法用量'])
            print('主治:     ',prescript['主治'])
            print()
    else:
        for i, prescript in enumerate(prescript_list):
            print('方剂名：  ', prescript['方剂名'])
            print('推荐药物：',prescript['药物组成'])
            print('用法用量：',prescript['用法用量'])
            print('主治:     ',prescript['主治'])
            print()
    print('核心药物集合：','、'.join([x[0] for x in main_med]))
if __name__ == '__main__':
    gol._init()
    ENTITY_FILE = gol.set_value('ENTITY_FILE', 'data/symptom_entity.csv')  # 添加实体的方剂文件
    CODE = gol.set_value('CODE', 'UTF-8')  # code
    SYNONYM_FILE = gol.set_value('SYNONYM_FILE', 'data/clean2.txt')  # 同义词
    num_4_group = 2
    num_4_other = 2
    '''构建索引'''
    # 存放索引的文件夹
    indexfile = 'index/'
    # 同义词词典路径
    # dictfile = 'dict/synonym.txt'
    dictfile = 'data/clean2.txt'
    # 方剂数据集路径
    datasetfile = 'data/symptom_entity.csv'
    # 创建索引
    # who.get_index(datasetfile, indexfile)

    insert = ['寒热','恶露未尽']
    insert = ['饮食不下','月经不通']
    recomment_main(insert)