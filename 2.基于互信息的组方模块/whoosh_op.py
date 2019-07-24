from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh.query import *
from whoosh.fields import *
import pandas as pd
import re
import utils

def get_index(datasetfile, indexdir):
    '''

    :param datasetfile: 方剂数据集文件
    :param indexdir: 存放index的文件夹
    :return:保存index文件
    '''
    #读取方剂数据
    data = pd.read_csv(datasetfile, delimiter=',', encoding='utf-8')
    #创建schema，field
    schema = Schema(id=TEXT(stored=True), name=TEXT(stored=True), source=TEXT(stored=True), content=TEXT(stored=True))
    #构建索引
    ix = create_in(indexdir, schema)
    writer = ix.writer()
    for i in range(data.shape[0]):
        id = str(data.ix[i,'序号'])
        name = str(data.ix[i,'方名'])
        source = str(data.ix[i,'出处'])
        content = str(data.ix[i,'symptom'])
        writer.add_document(id=id, name=name, source=source, content=content)
    writer.commit()

def add_synonym(dictfile,query_list):
    '''

    :param dictfile:同义词词典路径
    :param query: 用户查询信息列表
    :return: 加入同义词的用户查询语句
    '''

    # 读取同义词词典,syndict是同义词列表
    # f = open(dictfile,'r',encoding='utf-8')
    # syn_lines = f.readlines()
    # f.close()
    # syndict = list()
    # for line in syn_lines:
    #     # line = re.split(' ',re.sub('\n','',line))
    #     line = line.split()
    #     word_list = list()
    #     for word in line:
    #         word = re.sub(' ','',word)
    #         word_list.append(word)
    #     syndict.append(word_list)
    dic_map = utils.gene_dic_2(dictfile)
    syn_list = [dic_map[x] for x in query_list]
    #为用户查询添加同义词信息
    # syn_list = list()
    # for kw in query_list:
    #     kw_syn = list()
    #     kw_syn.append(kw)
    #     for syn in syndict:
    #         if kw in syn:
    #             kw_syn.extend(syn)
    #             break
    #     kw_syn = list(set(kw_syn))
    #     syn_list.append(kw_syn)
    #创建whoosh query语句
    # term_list = list()
    # for syn in syn_list:
    #     terms = [Term('content',word) for word in syn]
    #     query = Or(terms)
    #     term_list.append(query)
    # my_query = And(term_list)
    my_query = And([Term('content',word) for word in syn_list])
    return my_query

def add_synonym2(dictfile,query_list):
    '''
    or 查询语句
    :param dictfile:
    :param query_list:
    :return:
    '''
    my_query = Or([Term('content', word) for word in query_list])
    return my_query


def get_recommend(indexfile,myquery):
    '''

    :param indexfile: 保存index的文件夹
    :param query: 分好词的病症字符串
    :return:
    '''
    ix = open_dir(indexfile)
    with ix.searcher() as searcher:
        results = searcher.search(myquery,limit=None)
        if(len(results)==0):
            return False
        else:
            # return results
            count=0
            id_list = []
            for r in results:
                count+=1
                if(count>20):
                    break
                id_list.append(r['id'])
                 #score:评分
                 #
                # print('评分：{} id：{} 方名：{} 主治：{}'.format(round(r.score,2),r['id'],r['name'],r['content']))
            return id_list

if __name__ == '__main__':
    #存放索引的文件夹
    indexfile = 'index/'
    #同义词词典路径
    # dictfile = 'dict/synonym.txt'
    dictfile = 'data/tongyici_4.txt'
    #方剂数据集路径
    datasetfile = 'data/symptom_entity.csv'
    # datasetfile = 'data/entity.csv'

    #创建索引
    get_index(datasetfile,indexfile)

    # 用户查询
    query = ['带下赤白','四肢乏力']
    my_query = add_synonym(dictfile,query)
    get_recommend(indexfile,my_query)

