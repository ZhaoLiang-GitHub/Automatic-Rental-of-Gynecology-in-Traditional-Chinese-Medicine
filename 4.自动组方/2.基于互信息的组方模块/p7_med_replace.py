# -*- coding: utf-8 -*-
import pandas as pd
import utils
import p5_recommend as clus5
def count_med():
    df = pd.read_csv('data/symptom_entity.csv',encoding='utf8')
    med_count = {}
    all= 0
    for index,row in df.iterrows():
        med_list = row['处方'].split('、')
        all+=len(med_list)
        for med in med_list:
            if(med in med_count):
                med_count[med]+=1
            else:
                med_count[med]=1
    med_count = sorted(med_count.items(),key=lambda x:x[1])
    med_less = [x[0] for x in med_count if x[1]<4]
    replace_med = []
    with open('medicine_similarity/data/all_relatives.csv','r',encoding='utf8') as f:
        lines = f.readlines()
    lines = lines[1:]
    for line in lines:
        line = line.strip()
        seg = line.split(',')
        if(seg[0]==seg[1] or seg[1]=='无相似药物'):
            pass
        else:
            replace_med.append(seg[0])
            replace_med.extend(seg[1].split('、'))
    replace_med = list(set(replace_med))
    merge_list = list(set(med_less) & set(replace_med))
    with open('medicine_similarity/data/med_count.txt','w',encoding='utf8') as f:
        f.writelines([x[0]+'\n' for x in med_count])
    merge_dict ={}
    for line in lines:
        line = line.strip()
        seg = line.split(',')
        if(seg[0] in merge_list):
            merge_dict[seg[0]]=seg[1].split('、')
    utils.save_pickle('medicine_similarity/data/replace_med.pkl',replace_med)
    utils.save_pickle('medicine_similarity/data/merge_dict.pkl',merge_dict)
    with open('medicine_similarity/data/med_replace2.txt','r',encoding='utf8') as f:
        lines = f.readlines()
    lines = [x.split(',')[0] for x in lines]
    dd = set(lines)&set(replace_med)
    print('')
def med_replace_main(id,med):
    prescript = clus5.id_2_prescript([id])[0]
    med_same = utils.load_pickle('data/相似药物.txt')
    med_obj = {}
    for item in med_same:
        med_obj[item[0]]=item[1:]
    if(med not in med_obj):
        print('该药物没有相似药物。')
    else:
        print(med,'可以替换成药物:','、'.join(med_obj[med]))

if __name__ == '__main__':
    count_med()