synonyms.py 代码的作用是对输入文件中的词语作相似度计算,输入文件中的词语应用空格或换行隔开；输出若干个同义词词组，词组格式为按行排列，每一行是一个词组.
使用示例：
 sim_symptom('input.txt', 'output.txt', 'other.txt')
 input.txt为词语存储文件，词语以空格或换行符隔开；
 output.txt为输出同义词词组文件，文件中同义词词组按行排列
 other.txt为输入文件中，没有同义词的词语。

wordgroup_combine.py 作用是处理计算同义词时容易出现的词组重复现象及词组A和词组B有重合情况。
使用示例：
word_set_combine('input.txt', 'output.txt')
input.txt为synonyms.py输出的同义词词组文件
output.txt是合并同义词词组后的文件
