## 主要文件：

1. ckpt：用于保存模型的文件夹
2. data：保存训练集、验证集和实体词库的文件夹
3. log：用于保存训练日志的文件夹
4. result：用于保存实体识别结果的文件夹
5. config_file：保存模型超参数
6. conlleval.py：对实体识别结果进行性能上的评价
7. data_utils.py：对数据进行处理的函数
8. loader.py：加载数据的函数
9. main.py：主函数，控制模型的训练和预测
10. maps.pkl：保存训练过程中产生的字典
11. model.py：模型的构建
12. result_output.py：将识别并在测试集上标注出的实体输出成标准的实体
13. rnncell.py：模型部分
14. train_dev.pkl：保存训练过程中产生的训练集和验证集
15. utils.py：对文件进行操作的函数
16. word2vec_model.txt：以中医数据作为语料库训练得到的word2vec词向量

## 运行环境：

- python (3.5.4)
- gensim (3.6.0)
- jieba (0.37)
- numpy (1.14.3)
- pandas (0.23.3)
- scikit-learn (0.19.1)
- tensorflow (1.8.0)

## 运行方法

1. 训练：在**main.py**中将参数**train**设置为True，然后运行**main.py**。
2. 预测：在**main.py**中将参数**train**设置为False，若要对一整个文件中的数据进行实体识别，则将**predict_line**设置成False，若要通过控制台对一个句子进行实体识别，则将**predict_line**设置成True，然后在控制台输入需要识别的句子得到识别结果。
3. 实体标准化：实体的识别是在数据集上进行标注的，若需要将其转换成标准的词的形式，运行**result_output.py**即可。