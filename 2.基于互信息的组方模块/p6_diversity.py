# -*- coding: utf-8 -*-
from pres_diversity import 基于方剂差异度多样性聚类算法 as div
import p5_recommend as clus5
def deal_diversity(insert_list):
    print('多样化分组结果：')
    # best = div.diversity_main(['18019','80839','80920','81476']) #['11359','72842','80883','21177','6663','80844','80868','80872','80925','78091','80870','14779','11368','69818','11347','11352','66382','11360','80849','11998']
    best = div.diversity_main(insert_list)
    # print(best)
    print('共分成',len(best),'组')
    for index,item in enumerate(best):
        pre_list = clus5.id_2_prescript(item)
        print('第'+str(index+1)+'组:',pre_list)
if __name__ == '__main__':
    deal_diversity()