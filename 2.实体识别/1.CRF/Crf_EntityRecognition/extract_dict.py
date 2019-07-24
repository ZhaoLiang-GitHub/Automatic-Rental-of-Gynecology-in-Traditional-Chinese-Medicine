import os
import re
from utils import changeListToDict


def get_concept_from_entfile(file_path1, file_path2):
    files = os.listdir(file_path1)
    symptom_fp = open(file_path2 + '/symptom_dict.txt', 'w', encoding='utf-8')
    treat_fp = open(file_path2 + '/treat_dict.txt', 'w', encoding='utf-8')
    pattern_fp = open(file_path2 + '/pattern_dict.txt', 'w', encoding='utf-8')
    disease_fp = open(file_path2 + '/disease_dict.txt', 'w', encoding='utf-8')
    all_fp = open(file_path2 + '/all_dict.txt', 'w', encoding='utf-8')
    symptom_dict = []
    treat_dict = []
    disease_dict = []
    pattern_dict = []

    pattern_symptom = 'C=(.*) P=\d*:\d* T=symptom L=null'
    pattern_treat = 'C=(.*) P=\d*:\d* T=treat L=null'
    pattern_pattern = 'C=(.*) P=\d*:\d* T=pattern L=null'
    pattern_disease = 'C=(.*) P=\d*:\d* T=diseases L=null'

    for file in files:
        full_name = (os.sep).join([file_path1, file])
        with open(full_name, encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                result1 = re.match(pattern_symptom, line)
                result2 = re.match(pattern_treat, line)
                result3 = re.match(pattern_pattern, line)
                result4 = re.match(pattern_disease, line)

                if (result1):
                    symptom_dict.append(result1.group(1))
                elif (result2):
                    treat_dict.append(result2.group(1))
                elif (result3):
                    pattern_dict.append(result3.group(1))
                elif (result4):
                    disease_dict.append(result4.group(1))
                else:
                    pass

    symptom_dict = list(set(symptom_dict))
    treat_dict = list(set(treat_dict))
    pattern_dict = list(set(pattern_dict))
    disease_dict = list(set(disease_dict))

    for item1 in symptom_dict:
        symptom_fp.write(item1 + '\n')
        all_fp.write(item1 + '\n')
    for item2 in treat_dict:
        treat_fp.write(item2 + '\n')
        all_fp.write(item2 + '\n')
    for item3 in pattern_dict:
        pattern_fp.write(item3 + '\n')
        all_fp.write(item3 + '\n')
    for item4 in disease_dict:
        disease_fp.write(item4 + '\n')
        all_fp.write(item4 + '\n')
    symptom_fp.close()
    treat_fp.close()
    pattern_fp.close()
    disease_fp.close()
    all_fp.close()
    return symptom_dict, treat_dict, disease_dict, pattern_dict


def extract_alltypeDictionary(file_path1, file_path2):
    symptom_dict, treat_dict, disease_dict, pattern_dict = get_concept_from_entfile(file_path1, file_path2)
    dict = changeListToDict(symptom_dict, treat_dict, disease_dict, pattern_dict)
    return dict


if __name__ == "__main__":
    extract_alltypeDictionary('./test_token')
