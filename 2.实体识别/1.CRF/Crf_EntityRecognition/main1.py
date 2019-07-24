from data_process import *
from PRmatch import *
from  extract_dict import *
from entity_extract_byCrf import *

if __name__ == "__main__":
    # '''对测试集做分词处理和标注'''
    # cut_sentence('./split_data/test_data.txt', './data/cut_sentence_test.txt')
    # combineConceptAnnotation('./data/cut_sentence_test.txt', './data/annotation_test.txt', './data/dictionary')
    # '''对训练集做分词处理'''
    # cut_sentence('./split_data/train_data3000.txt','./data/cut_sentence_train1.txt')
    # '''对训练集标注，标注后的数据做为模型的训练语料'''
    # combineConceptAnnotation('./data/cut_sentence_train1.txt', './data/annotation_train1.txt','./data/dictionary')
    # '''用训练语料训练crf生成训练模板'''
    # command1 = 'crf_learn  ./template   ../data/annotation_train1.txt  ../crf/model1'
    # GenerateLearnCommand(command1)
    # '''用训练好的模板对测试数据做测试'''
    # command2 = 'crf_test -v1 -m  ..\\crf\\model1  ..\\data\\annotation_test.txt  >>  ..\\data\\result1.txt'
    # GenerateMutualTestCommend(command2)
    # file_dir = os.getcwd()
    # os.chdir(file_dir+'/crf')
    # run_bat(file_dir+"/crf/learn_commend.bat")
    # run_bat(file_dir+"/crf/crossLearn_commend.bat")
    # os.chdir(file_dir)
    PRmatch('./data/result1.txt', './data/probability1.txt')
    # '''对训练集做分词处理'''
    # cut_sentence('./split_data/train_data5000.txt', './data/cut_sentence_train2.txt')
    # '''对训练集标注，标注后的数据做为模型的训练语料'''
    # combineConceptAnnotation('./data/cut_sentence_train2.txt', './data/annotation_train2.txt', './data/dictionary')
    # '''用训练语料训练crf生成训练模板'''
    # command1 = 'crf_learn  ./template   ../data/annotation_train2.txt  ../crf/model2'
    # GenerateLearnCommand(command1)
    # '''用训练好的模板对测试数据做测试'''
    # command2 = 'crf_test -v1 -m  ..\\crf\\model2  ..\\data\\annotation_test.txt  >>  ..\\data\\result2.txt'
    # GenerateMutualTestCommend(command2)
    # file_dir = os.getcwd()
    # os.chdir(file_dir + '/crf')
    # run_bat(file_dir + "/crf/learn_commend.bat")
    # run_bat(file_dir + "/crf/crossLearn_commend.bat")
    # os.chdir(file_dir)
    PRmatch('./data/result2.txt', './data/probability2.txt')
    # '''对训练集做分词处理'''
    # cut_sentence('./split_data/train_data7000.txt', './data/cut_sentence_train3.txt')
    # '''对训练集标注，标注后的数据做为模型的训练语料'''
    # combineConceptAnnotation('./data/cut_sentence_train3.txt', './data/annotation_train3.txt', './data/dictionary')
    # '''用训练语料训练crf生成训练模板'''
    # command1 = 'crf_learn  ./template   ../data/annotation_train3.txt  ../crf/model3'
    # GenerateLearnCommand(command1)
    # '''用训练好的模板对测试数据做测试'''
    # command2 = 'crf_test -v1 -m  ..\\crf\\model3  ..\\data\\annotation_test.txt  >>  ..\\data\\result3.txt'
    # GenerateMutualTestCommend(command2)
    # file_dir = os.getcwd()
    # os.chdir(file_dir + '/crf')
    # run_bat(file_dir + "/crf/learn_commend.bat")
    # run_bat(file_dir + "/crf/crossLearn_commend.bat")
    # os.chdir(file_dir)
    PRmatch('./data/result3.txt', './data/probability3.txt')
    # '''对训练集做分词处理'''
    # cut_sentence('./split_data/train_data10000.txt', './data/cut_sentence_train4.txt')
    # '''对训练集标注，标注后的数据做为模型的训练语料'''
    # combineConceptAnnotation('./data/cut_sentence_train4.txt', './data/annotation_train4.txt', './data/dictionary')
    # '''用训练语料训练crf生成训练模板'''
    # command1 = 'crf_learn  ./template   ../data/annotation_train4.txt  ../crf/model4'
    # GenerateLearnCommand(command1)
    # '''用训练好的模板对测试数据做测试'''
    # command2 = 'crf_test -v1 -m  ..\\crf\\model4  ..\\data\\annotation_test.txt  >>  ..\\data\\result4.txt'
    # GenerateMutualTestCommend(command2)
    # file_dir = os.getcwd()
    # os.chdir(file_dir + '/crf')
    # run_bat(file_dir + "/crf/learn_commend.bat")
    # run_bat(file_dir + "/crf/crossLearn_commend.bat")
    # os.chdir(file_dir)
    PRmatch('./data/result4.txt', './data/probability4.txt')