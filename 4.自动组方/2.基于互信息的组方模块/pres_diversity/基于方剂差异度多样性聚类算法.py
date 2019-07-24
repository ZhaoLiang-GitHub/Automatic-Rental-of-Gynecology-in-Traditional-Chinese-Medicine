import pandas as pd
import re
import numpy as np
import random
import matplotlib.pyplot as plt
import time
import pickle


def hierarchical_clustering(difference_matrix, prescription_list, max):
    cluster = [[i] for i in prescription_list]
    while len(cluster) > max:
        difference_cluster = []
        for i in range(len(cluster)):
            for j in range(i + 1, len(cluster)):
                difference = 0
                for k in cluster[i]:
                    for t in cluster[j]:
                        difference += (difference_matrix[prescription_list.index(k)][prescription_list.index(t)]
                                       + difference_matrix[prescription_list.index(t)][prescription_list.index(k)])
                difference = difference / (len(cluster[i]) * len(cluster[j]))
                difference_cluster.append([difference, [i, j]])
        difference_cluster = sorted(difference_cluster, key=lambda x: x[0])
        item = []
        for i in difference_cluster[0][1]:
            item.extend(cluster[i])
        new_list = []
        for i in range(len(cluster)):
            if i not in difference_cluster[0][1]:
                new_list.append(cluster[i])
        cluster = new_list
        cluster.append(item)

    cluster_score = 0
    # DB指数  https://blog.csdn.net/sinat_33363493/article/details/52496011
    C_list = []  # 簇内的平均距离
    for i in range(len(cluster)):
        c = 0
        for j in cluster[i]:
            for k in cluster[i]:
                c += (difference_matrix[prescription_list.index(j)][prescription_list.index(k)]
                      + difference_matrix[prescription_list.index(k)][prescription_list.index(j)])
        C_list.append(c / len(cluster[i]) ** 2)
    for i in range(len(cluster)):
        score_list = []
        for j in range(i + 1, len(cluster)):
            if cluster[j] != cluster[i]:
                w = 0
                for k in cluster[i]:
                    for t in cluster[j]:
                        w += (difference_matrix[prescription_list.index(k)][prescription_list.index(t)]
                              + difference_matrix[prescription_list.index(t)][prescription_list.index(k)])
                w = w / len(cluster[i]) * len(cluster[j])
                score_list.append((C_list[j] + C_list[i]) / w)
            score_list.sort(reverse=True)
            cluster_score += score_list[0]
    cluster_score = cluster_score / max
    # print("聚类团个数为%d时的簇"%max,cluster)

    return cluster, cluster_score


def kmeans_prescription(difference_matrix, prescription_list, K, Maximum_iteration):
    seed_k = prescription_list[:K]  # 取前k个方剂作为聚类中心，不使用随机中心
    # seed_k = random.sample(prescription_list,K)#随机选择初试簇中心
    Num_iterations = 0
    Residual_prescription = set(prescription_list) - set(seed_k)  # 剩余方剂列表
    cluster = [[i] for i in seed_k]  # 聚类簇
    # print("聚类团个数为%d第%d次迭代的聚类簇" % (K,Num_iterations), cluster)
    # print("聚类团个数为%d第%d次迭代的簇中心" % (K,Num_iterations), seed_k)

    # 当聚类簇中心不再发生变化时或者以达到最大迭代次数时退出循环
    while Num_iterations < Maximum_iteration:
        # 对于除中心外其他方剂，找到距离最小的簇并加入该簇
        for i in Residual_prescription:
            a = []
            for j in range(len(seed_k)):
                # 在方剂差异度矩阵找查找，根据方剂列表的下表
                a.append([j, difference_matrix[prescription_list.index(i)][prescription_list.index(seed_k[j])]
                          + difference_matrix[prescription_list.index(seed_k[j])][prescription_list.index(i)]])
            a = sorted(a, key=lambda x: x[1])
            cluster[a[0][0]].append(i)
        cluster = [list(set(i)) for i in cluster]
        # print("聚类团个数为%d第%d次迭代的聚类簇" % (K, Num_iterations+1), cluster)

        # 重新定位簇中心，中心为簇内一点与其他点的距离和最小的一个
        new_seed = [''] * len(seed_k)
        for i in range(len(cluster)):
            # 当簇内只有一个或两个方剂时，簇中心保持不变
            if len(cluster[i]) > 2:
                difference_list = []
                # 簇内两两比较距离，找到距离和最小的一个作为新中心
                for j in cluster[i]:
                    a = 0
                    for t in cluster[i]:
                        a += (difference_matrix[prescription_list.index(j)][prescription_list.index(t)]
                              + difference_matrix[prescription_list.index(t)][prescription_list.index(j)])
                    difference_list.append([j, a])
                mindifference = sorted(difference_list, key=lambda x: x[1])
                new_seed[i] = mindifference[0][0]
            else:
                new_seed[i] = seed_k[i]
        # print("聚类团个数为%d第%d次迭代的簇中心" % (K, Num_iterations+1), new_seed)
        Residual_prescription = set(prescription_list) - set(new_seed)  # 剩余方剂列表
        if new_seed == seed_k:
            # #print(new_seed,seed_k)
            # print('聚类团个数为%d的方剂聚类簇中心不再发生变化，退出循环'%K)
            break
        else:
            Num_iterations += 1
            seed_k = new_seed
            cluster = [[i] for i in new_seed]

    # 基于轮廓系数得到聚类的评分,同类别样本距离越近不同类别样本距离越远分数越高，分数越高聚类效果越好
    # 轮廓系数 a是单个样本的轮廓系数为与同类别内的其他样本的平均距离，b是与距离最近的类别中样本的平均距离，s = （b-a）/(max (a.b))
    # 对于集合轮廓系数为所有轮廓系数平均值
    # for i in prescription_list:
    #     aa = 0
    #     num_same_categories = 0
    #     min_distance_categorie = []
    #     for j in range(len(cluster)):
    #         if i in cluster[j] :#对于单个个体而言 计算同类别内每个个体距离的平均值
    #             for t in cluster[j]:
    #                 #因为方剂距离矩阵是上三角矩阵，因为没有判断i t大小，加上相对位置的值，以免出现某值为0但相对应的值有实数
    #                 aa += (difference_matrix[prescription_list.index(i)][prescription_list.index(t)]
    #                       +difference_matrix[prescription_list.index(t)][prescription_list.index(i)])
    #                 num_same_categories += 1
    #         else:#单个个体 计算与距离最近的类别内其他个体距离的平均值
    #             bb = 0
    #             num_difference_categories = 0
    #             for k in cluster[j]:
    #                 bb += (difference_matrix[prescription_list.index(i)][prescription_list.index(k)] +
    #                       difference_matrix[prescription_list.index(k)][prescription_list.index(i)])
    #                 num_difference_categories += 1
    #             min_distance_categorie.append(bb/num_difference_categories)
    #     a = aa/num_same_categories#一个个体的 平均簇内距离
    #     min_distance_categorie.sort()
    #     b = min_distance_categorie[0]#一个个体的 最短的 平均簇间距离
    #     s = (b-a)/(max(a,b))
    #     cluster_score += s
    # cluster_score =cluster_score/k

    cluster_score = 0
    # DB指数  https://blog.csdn.net/sinat_33363493/article/details/52496011
    C_list = []  # 簇内的平均距离
    for i in range(len(cluster)):
        c = 0
        for j in cluster[i]:
            # for k in cluster[i]:
            c += (difference_matrix[prescription_list.index(j)][prescription_list.index(seed_k[i])]
                  + difference_matrix[prescription_list.index(seed_k[i])][prescription_list.index(j)])
        C_list.append(c / len(cluster[i]))
    for i in range(len(cluster)):
        score_list = []
        for j in range(len(cluster)):
            if j != i:
                w = (difference_matrix[prescription_list.index(seed_k[i])][prescription_list.index(seed_k[j])]
                     + difference_matrix[prescription_list.index(seed_k[j])][prescription_list.index(seed_k[i])])
                score_list.append((C_list[j] + C_list[i]) / w)
        score_list.sort(reverse=True)
        cluster_score += score_list[0]
    cluster_score = cluster_score / K

    return cluster, cluster_score


def cluster_score(prescription_list, min_group, max_group, f_prescription, f_prescription_list,
                  difference_medicine_matrix, f_medicine_list):
    if len(prescription_list) < max_group: max_group = len(prescription_list)
    cluster_score_list = []

    difference_matrix = []
    for i in range(len(prescription_list)):
        item_list = [0] * len(prescription_list)
        p1_medicine = f_prescription['标准药物名称'].loc[f_prescription_list.index(prescription_list[i])].split('、')

        for j in range(i + 1, len(prescription_list)):
            p2_medicine = f_prescription['标准药物名称'].loc[f_prescription_list.index(prescription_list[j])].split('、')
            a = []
            for k in p2_medicine:
                for t in p1_medicine:
                    a.append(difference_medicine_matrix[f_medicine_list.index(k)][f_medicine_list.index(t)] +
                             difference_medicine_matrix[f_medicine_list.index(t)][f_medicine_list.index(k)])
            difference_p = round(sum(a) / (len(p1_medicine) * len(p2_medicine)), 10)
            item_list[j] = difference_p
        difference_matrix.append(item_list)
    # #print(difference_matrix)

    for i in range(min_group, max_group):
        # prescription_cluster , cluster_score = kmeans_prescription(difference_matrix=difference_matrix,prescription_list=prescription_list, K=i,Maximum_iteration=10)
        prescription_cluster, cluster_score = hierarchical_clustering(difference_matrix=difference_matrix,
                                                                      prescription_list=prescription_list, max=i)
        cluster_score_list.append([prescription_cluster, cluster_score])
    num_cluster = np.array([i for i in range(min_group, max_group)])
    score = np.array([i[1] for i in cluster_score_list])
    plt.plot(num_cluster, score)
    # plt.show()
    cluster_score_list = sorted(cluster_score_list, key=lambda x: x[1], reverse=True)
    # cluster_score_list = sorted(cluster_score_list, key=lambda x: x[1])
    # print("最好的聚类结果团个数为", len(cluster_score_list[0][0]), "\n最好的聚类结果是", cluster_score_list[0][0])
    best_cluster = cluster_score_list[0][0]

    # 从每个聚类簇中挑选评分最高的一个方剂
    # 评分方法为统计在簇内出现过的所有药物，每个药物的词频作为该药物的重要新程度，每个方剂的分数为方剂内所有药物评分总和
    all_medicine = []
    for i in best_cluster:
        for j in i:
            a = f_prescription['标准药物名称'].loc[f_prescription_list.index(j)].split('、')
            all_medicine.extend(a)
    best_prescription = []
    for prescription_list_item in best_cluster:
        if len(prescription_list_item) == 1:
            best_prescription.append(prescription_list_item[0])
        else:
            score_list = [0] * len(prescription_list_item)
            for j in range(len(prescription_list_item)):
                b = f_prescription['标准药物名称'].loc[f_prescription_list.index(prescription_list_item[j])].split('、')
                for k in b:
                    score_list[j] += all_medicine.count(k)
            score_list = [[i, score_list.index(i)] for i in score_list]
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)
            best_prescription.append(prescription_list_item[score_list[0][1]])
    # print("多样性好且评分高的方剂推荐列表是",best_prescription)

    return best_cluster, best_prescription


def diversity_main(prescription_list):
    f_prescription_add = '方剂多样性推荐/公司数据_标准药物名称.csv'
    f_prescription = pd.read_csv(open(f_prescription_add, 'r', encoding='utf8'))
    f_prescription_list = f_prescription['id'].tolist()
    f_medicine_add = '方剂多样性推荐/药物数据集.csv'
    f_medicine = pd.read_csv(open(f_medicine_add, 'r', encoding='utf-8'))
    f_medicine_list = f_medicine['名称'].tolist()
    with open('方剂多样性推荐/药物差异度矩阵1.txt', 'rb') as f:
        a1 = pickle.load(f)
    with open('方剂多样性推荐/药物差异度矩阵2.txt', 'rb') as f:
        a2 = pickle.load(f)
    with open('方剂多样性推荐/药物差异度矩阵3.txt', 'rb') as f:
        a3 = pickle.load(f)
    with open('方剂多样性推荐/药物差异度矩阵4.txt', 'rb') as f:
        a4 = pickle.load(f)
    with open('方剂多样性推荐/药物差异度矩阵5.txt', 'rb') as f:
        a5 = pickle.load(f)
    with open('方剂多样性推荐/药物差异度矩阵6.txt', 'rb') as f:
        a6 = pickle.load(f)
    with open('方剂多样性推荐/药物差异度矩阵7.txt', 'rb') as f:
        a7 = pickle.load(f)
    with open('方剂多样性推荐/药物差异度矩阵8.txt', 'rb') as f:
        a8 = pickle.load(f)
    difference_medicine_matrix = a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8
    prescription_list = [int(i) for i in prescription_list]
    best_cluster, best_prescription = cluster_score(prescription_list=prescription_list, min_group=2, max_group=50,
                                                    f_prescription=f_prescription,
                                                    f_prescription_list=f_prescription_list,
                                                    difference_medicine_matrix=difference_medicine_matrix,
                                                    f_medicine_list=f_medicine_list)
    return best_cluster