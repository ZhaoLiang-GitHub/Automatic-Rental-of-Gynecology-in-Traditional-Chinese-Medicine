# -*- coding: utf-8 -*-
# @Time    : 2018/7/11 14:00
# @Author  : zhaoliang
# @FileName: tongjifenxi
# @Software: Sublime

import pandas as pd
import re
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.decomposition import PCA
import numpy as np


def getdict(f_dict):
    a = f_dict.readlines()
    dict = []
    for i in range(len(a)):
        b = a[i].strip().split('\t')
        dict.append(b)
    # print(dict)
    return dict

def get_onehotandlabels(dict,f_excel):
    onehot = []
    labels = []
    for i in range(f_excel.shape[0]):
        a = [0] * len(dict)
        for j in range(len(dict)):
            for t in range(len(dict[j])):
                try:
                    if dict[j][t] in f_excel['主治'].loc[i]:
                        a[j] = 1
                except TypeError:
                    pass
        onehot.append(a)
        labels.append(str(f_excel['证型'].loc[i]))
    # print(onehot)
    # print(labels)
    return onehot,labels

"""高斯朴素贝叶斯"""
def NB(X_train, X_test, y_train, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    print("Gaussian Naive Bayes model accuracy(in %):", metrics.accuracy_score(y_test, y_pred) * 100)

"""KNN"""
def KNN(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier()  # 定义一个knn分类器对象
    knn.fit(X_train, y_train)  # 调用该对象的训练方法，主要接收两个参数：训练数据集及其样本标签
    y_predict = knn.predict(X_test)  # 调用该对象的测试方法，主要接收一个参数：测试数据集
    probility = knn.predict_proba(X_test)  # 计算各测试样本基于概率的预测
    score = knn.score(X_test, y_test)
    # 调用该对象的打分方法，计算出准确率
    # print('y_predict = ')
    # print(y_predict)
    # 输出测试的结果
    # print('y_test = ')
    # print(y_test)
    # 输出原始测试数据集的正确标签，以方便对比
    print('KNN_Accuracy:', score)
    # 输出准确率计算结果
    # print('neighborpoint of last test sample:', neighborpoint)
    # print('probility:', probility)

"""逻辑回归"""
def LR(X_train, X_test, y_train, y_test):
    cls = LogisticRegression()  # 把数据交给模型训练
    cls.fit(X_train, y_train)  # 选择模型
    cls = LogisticRegression(multi_class='multinomial', solver='lbfgs')
    # 把数据交给模型训练
    cls.fit(X_train, y_train)
    # print("Coefficients:%s, intercept %s" % (cls.coef_, cls.intercept_))
    y_predict = cls.predict(X_test)
    # print("Residual sum of squares: %.2f" % np.mean((y_predict - y_test) ** 2))
    print('logistic回归_Score: %.2f' % cls.score(X_test, y_test))

"""SVM"""
def SVM(X_train, X_test, y_train, y_test):
    model = OneVsRestClassifier(svm.SVC(kernel='linear'))
    clf = model.fit(X_train, y_train)
    y_result = clf.predict(X_test)
    print("SVM_score:", clf.score(X_test, y_test))

"""CART"""
def CART(X_train, X_test, y_train, y_test):
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    matchCount = 0
    for i in range(len(y_predict)):
        if y_predict[i] == y_test[i]:
            matchCount += 1
    accuracy = float(matchCount / len(y_predict))
    print('Testing_CART :Testing completed.Accuracy: %.3f%%' % (accuracy * 100))

"""PCA"""
def pca(onehot_matrix):
    x = np.array(onehot_matrix)
    print(x.shape)
    pca = PCA(n_components=0.9)
    pca.fit(x)
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    print(pca.n_components_)
    x_pca = pca.transform(x)
    print(x_pca.shape)
    return x

if __name__ =="__main__":
    f_dict = open(r'data\dict.txt', 'r',encoding='utf-8')
    f_excel = pd.read_excel(r'data\pattern_class.xlsx')
    dict = getdict(f_dict)
    onehot, labels = get_onehotandlabels(dict, f_excel)
    X_train, X_test, y_train, y_test = train_test_split(onehot, labels, test_size=0.9, random_state=0)
    CART(X_train, X_test, y_train, y_test)
    # x = pca(onehot)
