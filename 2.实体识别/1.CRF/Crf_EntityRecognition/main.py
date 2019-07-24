from data_process import *
from PRmatch import *
from extract_dict import *
from entity_extract_byCrf import *

if __name__ == "__main__":
    # # '''提取词典'''
    # # extract_alltypeDictionary('data/token_train','data/dict_train')
    # # extract_alltypeDictionary('data/token_test', 'data/dict_test')
    # # '''对训练集做分词处理'''
    # cut_sentence('./data/original_data/train_data.txt', './data/cut_sentence_train.txt')
    # '''对测试集做分词处理'''
    # cut_sentence('./data/original_data/test_data.txt', './data/cut_sentence_test.txt')
    # '''对训练集和测试集标注，标注后的数据做为模型的训练语料'''
    # combineConceptAnnotation('./data/cut_sentence_train.txt', './data/annotation_train.txt', 'data/dictionary')
    # combineConceptAnnotation('./data/cut_sentence_test.txt', './data/annotation_test.txt', 'data/dictionary')
    # '''用训练语料训练crf生成训练模板'''
    # command1 = 'crf_learn  ./template   ../data/annotation_train.txt  ../crf/model'
    # GenerateLearnCommand(command1)
    # '''用训练好的模板对测试数据做测试'''
    # command2 = 'crf_test -v1 -m  ..\\crf\\model  ..\\data\\annotation_test.txt  >>  ..\\data\\result.txt'
    # GenerateMutualTestCommend(command2)
    # file_dir = os.getcwd()
    # os.chdir(file_dir + '/crf')
    # run_bat(file_dir + "/crf/learn_commend.bat")
    # run_bat(file_dir + "/crf/crossLearn_commend.bat")
    # os.chdir(file_dir)
    # ''''模型测试结果'''
    PRmatch('./data/result.txt', './data/probability.txt')
    # '''提取新词和准确率计算'''
    # entity_extract()
    # check_new_word()
