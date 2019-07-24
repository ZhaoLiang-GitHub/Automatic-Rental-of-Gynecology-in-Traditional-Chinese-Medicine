## 主要文件：

#### feature_engineering：特征工程文件夹，主要是基于规则对症状和功效特征词进行分词、将药物的性味归经特征进行向量化、训练词向量：

1. data：保存程序所生成的文件，其中：
   - dict_effect：保存对功效词分词的结果，set3_false_dict.txt存储未被拆分的三字词，set3_true_dict.txt存储被拆分的三字词，对set3_true_dict.txt中的词进行人工标注得到set3_true_dict_labeld.txt，1表示该词是被正确拆分，0表示该词被错误拆分；同理，set4_false_dict.txt存储未被拆分的四字词，set4_true_dict.txt存储被拆分的四字词，对set4_true_dict.txt中的词进行人工标注得到set4_true_dict_labeld.txt。
   - dict_function：保存对症状词分词的结果，同上。
   - cut_labled.csv：保存被人工标注为被错误拆分和被正确拆分的词，作为优先级高于规则的拆分指标。
   - data_all_v10.csv：全体药物数据集
2. word_cut.py：对特征词进行拆分的主程序
3. word_cut_label_to_txt.py：将被人工标注为被错误拆分和被正确拆分的词整合并处理为cut_labled.csv
4. feature_to_vector.py：将药物的性味归经特征进行向量化
5. gensim_wor2vec.py：基于word2vec、以药物数据中的功效信息和主治信息作为语料库训练词向量。

#### data：保存feature_engineering中程序所生成的文件

#### medicine_similarity：药物聚类文件夹，主要是基于k-modes和互信息对药物进行聚类：

1. data：保存程序所生成的文件
2. taste_cluster_various.py：基于k-modes方法对药物的性味归经进行聚类
3. function_cluster_entropy.py：基于互信息对药物的功效特征进行聚类
4. cluster_main.py：聚类的主程序，将上面两种聚类方法得到的结果进行整合，得到最终的聚类结果。
5. cluster.py、data_utils.py、relatives.py：均用于保存对数据进行处理的函数

## 运行环境：

- python (3.5.4)
- jieba (0.37)
- numpy (1.14.3)
- pandas (0.23.3)
- scikit-learn (0.19.1)

## 运行方法：

1. 分词：在word_cut.py中设置需要对特征词进行拆分的药物数据的路径，然后运行该程序
2. 聚类：
   1. 运行feature_to_vector.py，得到性味归经的向量化表示。
   2. 在cluster_main.py中将参数**is_cluster**设置为True，进行聚类。
3. 寻找相似药物：
   1. 在cluster_main.py中将参数**is_cluster**设置为False，若要找到所有药物的相似药物，则将参数**search_all**设置为True，然后运行程序，否则设置为False、设置函数**main()**中的变量**medicine_word**为需要寻找相似药物的目标药物，再运行程序。