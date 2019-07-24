import numpy as np
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans, Birch, SpectralClustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics

file_path = "../data/feature_vector.csv"    # 需要进行聚类的数据特征向量
file_data_treat = "../data/data_treat.csv"  # 需要进行聚类的原数据文件
medicines_to_taste_label_path = "data/medicines_to_taste_label.pkl"  # 保存药物到性味归经聚类标签的字典


def get_data(path):
    data = pd.read_csv(path)
    # print(data.info())
    # print(data)
    data = np.asarray(data)  # DataFrame转array
    data = data[:, 1:]
    # print(data, data.shape)
    return data


def cluster_kmodes(n_clusters, data):
    """
    kmodes聚类方法，处理离散的原始one-hot特征向量
    :param n_clusters:质心数量
    :param data:需要进行聚类的数据
    :return:
    """
    # visual_data(data)  #可视化原数据
    kmodes = KModes(n_clusters=n_clusters, init="Huang", n_init=2, verbose=1)
    clusters = kmodes.fit_predict(data)
    print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(data, clusters))
    # print("每个样本点所属类别索引", clusters)  # 输出每个样本的类别
    # print("簇中心", kmodes.cluster_centroids_)    # 输出聚类结束后的簇中心
    # 聚类结果输出到CSV中，包括加上聚类结果标签的全体数据和单独的性味归经数据
    data_labeled_to_csv(clusters, file_data_treat, "data/data_labeld_kmodes.csv", "data/taste_group.csv")
    # visual_cluster(n_clusters, data, clusters)


def cluster_kmeans(n_clusters):
    """
    kmeans聚类方法，处理经过PCA处理的特征向量
    :param n_clusters:质心数量
    :return:
    """
    data = get_data("../data/feature_vector.csv")
    # visual_data(data)
    kmeans = KMeans(n_clusters=n_clusters)
    clusters = kmeans.fit_predict(data)
    # print("聚类性能", kmeans.inertia_)
    print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(data, clusters))
    # print("每个样本点所属类别索引", clusters)
    # print("簇中心", kmeans.cluster_centers_)
    data_labeled_to_csv(clusters, file_data_treat, "data/data_labeld_kmodes.csv", "data/taste_group.csv")
    # visual_cluster(n_clusters, data, clusters)


def cluster_birch(n_clusters):
    """
    birch聚类方法，处理经过PCA处理的特征向量
    :param n_clusters:质心数量
    :return:
    """
    data = get_data("../data/feature_vector_pca.csv")
    birch = Birch(n_clusters=n_clusters, threshold=0.4, branching_factor=50)
    clusters = birch.fit_predict(data)
    print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(data, clusters))
    print("每个样本点所属类别索引", clusters)
    # print("簇中心", birch.cluster_centers_)
    data_labeled_to_csv(clusters, "data/data_labeld_birch.csv")
    # visual_cluster(n_clusters, data, clusters)


def cluster_spectralclustering(n_clusters):
    """
    SpectralClustering聚类算法
    :param n_clusters:质心数量
    :return:
    """
    data = get_data("../data/feature_vector_pca.csv")
    spectral = SpectralClustering(n_clusters=n_clusters, gamma=0.01)
    clusters = spectral.fit_predict(data)
    # 遍历超参以寻找最优参数
    # for index, gamma in enumerate((0.01, 0.1, 1, 10)):
    #     for index2, k in enumerate((15, 20, 25, 30)):
    #         clusters = SpectralClustering(n_clusters=k, gamma=gamma).fit_predict(data)
    #         print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", k, "score:",
    #               metrics.calinski_harabaz_score(data, clusters))
    print("Calinski-Harabasz Score", metrics.calinski_harabaz_score(data, clusters))
    print("每个样本点所属类别索引", clusters)
    data_labeled_to_csv(clusters, "data/data_labeld_birch.csv")
    # visual_cluster(n_clusters, data, clusters)


def visual_data(data):
    """
    对原始结果进行可视化
    :param data:原始样本数据
    :return:
    """
    length = len(data[0])
    # print(type(data))
    x_length, y_length, z_length = length//3, 2*(length//3), 3*(length//3)
    # 各个轴可以是同样长度的数组，不必须是单一的数值，尚不清楚数组如何转换为指定坐标的值
    x, y, z = data[:, :x_length], data[:, x_length:y_length], data[:, y_length:z_length]
    # print(x, y, z)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, marker="o", c="y")  # 如果要进行3D绘图，要用ax调用scatter才行
    plt.show()


def clusters_label_class(n_clusters, data, clusters):
    """
    将聚类后的样本分别存入各标签数组中
    :param n_clusters:质心数量
    :param data:原始数据样本
    :param clusters:聚类结果（即每个样本所属的标签）
    :return:
    """
    data_labeled_lists = []
    # print(data, clusters)
    length = len(data)
    for i in range(n_clusters):
        data_labeled_lists.append([])
    # print(data_labeled_lists)
    for i in range(length):
        data_labeled_lists[clusters[i]].append(data[i].tolist())
    # 列表转数组
    for i in range(n_clusters):
        data_labeled_lists[i] = np.asarray(data_labeled_lists[i])
    data_labeled_lists = np.asarray(data_labeled_lists)
    # print(data_labeled_lists)
    # 验证是否所有样本均归类
    num_count = 0
    for data_labeled_list in data_labeled_lists:
        for data_labeled in data_labeled_list:
            num_count += 1
    # print("Right!" if num_count==length else "Error!")
    return data_labeled_lists


def visual_cluster(n_clusters, data, clusters):
    """
    聚类结果可视化
    :param n_clusters:质心数量
    :param data:原始数据样本
    :param clusters:聚类结果（即每个样本所属的标签）
    :return:
    """
    data_labeled_lists = clusters_label_class(n_clusters, data, clusters)   # 将聚类后的样本分别存入各标签数组中
    length = len(data[0])
    x_length, y_length, z_length = length//3, 2*(length//3), 3*(length//3)
    colors = ['b', 'g', 'r', 'k', 'c', 'm', 'y', 'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
              'beige', 'bisque', 'black', 'blanchedalmond', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
              'chartreuse', 'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue',
              'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkkhaki', 'darkmagenta']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(n_clusters):
        x, y, z = data_labeled_lists[i][:, :x_length], data_labeled_lists[i][:, x_length:y_length], \
                  data_labeled_lists[i][:, y_length:z_length]
        # print(x, y, z)
        ax.scatter(x, y, z, marker="o", c=colors[i])
    plt.show()


def data_labeled_to_csv(clusters, data_treat, filename, filename_taste):
    """
    输出聚类结果：将每个样本的标签合并到原始data中、根据label重新排序，再输出到csv中
    :param clusters:聚类结果
    :param data_treat:需要进行聚类的原数据
    :param filename:输出打上聚类结果标签的全体数据的目标文件路径
    :param filename_taste:输出打上聚类结果标签的性味归经数据的目标路径文件
    :return:
    """
    data = pd.read_csv(data_treat, index_col=0)
    print(data.info())
    data.insert(1, "Label", None)
    length = data.shape[0]
    # 验证长度是否对齐
    length_clusters = len(clusters)
    # print(data)
    print("length_clusters:", length_clusters)
    print("Right!" if length == length_clusters else "Error!" + str(length) + ":" + str(length_clusters))
    medicines_to_taste_label = {}
    for i in range(length):
        data["Label"].iloc[i] = clusters[i]
        medicines_to_taste_label[data["名称"].iloc[i]] = clusters[i]
    with open(medicines_to_taste_label_path, "wb") as f:
        pickle.dump(medicines_to_taste_label, f)
    print("medicines_to_taste_label has been dump!!!!!!!!")
    print(data.info())
    data = data.sort_values(by='Label', ascending=True)  # 这里要注意sort_value是返回一个已排序的对象，而不是原地进行修改
    data.to_csv(filename, index=False, encoding="utf-8")
    # data_taste = pd.concat([data["Label"], data["名称"], data["Taste", data["Type"]]], axis=1)
    # data_taste.to_csv(filename_taste, index=False, encoding="utf-8")
    data_taste = pd.DataFrame(np.zeros((length, 4)), columns=["Label", "名称", "Taste", "Type"])
    print(data_taste.info())
    for i in range(length):
        data_taste["Label"].iloc[i] = data["Label"].iloc[i]
        data_taste["名称"].iloc[i] = data["名称"].iloc[i]
        data_taste["Taste"].iloc[i] = data["Taste"].iloc[i]
        data_taste["Type"].iloc[i] = data["Type"].iloc[i]
    # print("data_taste", data_taste)
    data_taste_new = data_taste.sort_values(by='Label', ascending=True)  # 根据label进行排序
    data_taste_new.to_csv(filename_taste, index=False, encoding="utf-8")


def opti_para_select(cluster_name, data):
    """
    专门用于寻找最优参数的函数
    :param cluster_name:聚类方法名称
    :param data:需要进行聚类的数据
    :return:
    """
    if cluster_name == SpectralClustering:
        max_score = 0
        opti_gamma, opti_n_clusters = 0, 0
        for gamma in (0.01, 0.1, 1):
            for n_clusters in (15, 20, 25, 30):
                clusters = SpectralClustering(n_clusters=n_clusters, gamma=gamma).fit_predict(data)
                score = metrics.calinski_harabaz_score(data, clusters)
                # print("Calinski-Harabasz Score with gamma=", gamma, "n_clusters=", n_clusters,"score:", score)
                if max_score < score:
                    max_score = score
                    opti_gamma, opti_n_clusters = gamma, n_clusters
        print("max_score:", max_score, "opti_gamma:", opti_gamma, "opti_n_clusters:", opti_n_clusters)

    if cluster_name == "k_modes":
        max_score = 0
        opti_n_clusters = 0
        cluster_num_list = [30, 40, 50, 60, 70, 80, 90, 100]
        # for n in range(30, 100):
        for n in cluster_num_list:
            kmodes = KModes(n_clusters=n, init="Huang", n_init=5, verbose=1)
            clusters = kmodes.fit_predict(data)
            score = metrics.calinski_harabaz_score(data, clusters)
            print("Calinski-Harabasz Score——", "n_clusters=", n, "score:", score)
            if max_score < score:
                max_score = score
                opti_n_clusters = n
        print("max_score:", max_score, "opti_n_clusters:", opti_n_clusters)

    if cluster_name == "k_means":
        max_score = 0
        opti_n_clusters = 0
        for n in range(2, 30):
            kmodes = KModes(n_clusters=n, init="Huang", n_init=10, verbose=1)
            clusters = kmodes.fit_predict(data)
            score = metrics.calinski_harabaz_score(data, clusters)
            print("Calinski-Harabasz Score——", "n_clusters=", n, "score:", score)
            if max_score < score:
                max_score = score
                opti_n_clusters = n
        print("max_score:", max_score, "opti_n_clusters:", opti_n_clusters)


def cluster_various_main():
    """
    主聚类函数
    :return:
    """
    data_treat = get_data(file_path)
    # opti_para_select("k_modes", data_treat)
    num_clusters = 30
    cluster_kmodes(num_clusters, data_treat)

if __name__ == "__main__":
    cluster_various_main()
