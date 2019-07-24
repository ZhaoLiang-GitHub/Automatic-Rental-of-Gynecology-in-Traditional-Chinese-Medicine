sim_sentence.py作用是计算输入病案与标准病案的相似度，返回与标准病案中某症型的相似度，按相似度倒叙排列
使用示例：
compare_symptom(input_string, '带下.csv')
其中input_string是带比较的病案，带下.csv是标准病案文件，标准病案文件的格式为[症型,主要症状,次要症状],详情参考data文件夹下带下.csv文件
代码运行需要的外部词典在dict文件下，词典的具体作用参见sim_sentence.py中的注释内容