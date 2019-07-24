import pandas as pd
import os
import logging
from gensim.models import word2vec

path_data_all = "data/data_all_v5.csv"
path_word2vec_string = "data/word2vec_string.txt"


def get_data():
    data_all = pd.read_csv(path_data_all)
    print(data_all.info())
    data_all = data_all.fillna("missing")
    print(data_all.info())
    len_data = data_all.shape[0]
    data_function = data_all['功效']
    data_effect = data_all["主治"]
    string = ""
    for i in range(len_data):
        list_function = []
        for char in data_function.loc[i]:
            if char != "m" \
                    and char != "i" and char != "s" and char != "n" and char != "g":
                list_function.append(char)
        string_function = " ".join(list_function)
        list_effect = []
        for char in data_effect.loc[i]:
            if char != "m" \
                    and char != "i" and char != "s" and char != "n" and char != "g":
                list_effect.append(char)
        string_effect = " ".join(list_effect)
        string = string + string_function + " " + string_effect + " "
    # print(string)
    with open(path_word2vec_string, "w", encoding="utf-8") as f:
        f.write(string)


def get_word2vec():
    sentences = word2vec.LineSentence(path_word2vec_string)
    # model = word2vec.Word2Vec(sentences, sg=0, hs=1, min_count=1, window=3, size=100)  # CBOW
    # model.save("data/word2vec_model")
    # model.wv.save_word2vec_format('data/word2vec_model.txt', binary=False)
    model = word2vec.Word2Vec(sentences, sg=1, hs=1, min_count=1, window=3, size=100)   # skipgram
    model.save("data/word2vec_model_sg")
    model.wv.save_word2vec_format('data/word2vec_model_sg.txt', binary=False)
    print("痛", model["痛"])
    print("疼", model["疼"])
    print(model.most_similar(["痛"]))
    print(model.wv.similarity('痛', '虚'))

if __name__ == "__main__":
    # get_data()
    get_word2vec()
