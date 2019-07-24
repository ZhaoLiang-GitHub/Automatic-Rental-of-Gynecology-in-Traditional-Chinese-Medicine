import os
import json
import shutil
import logging
import numpy as np
import codecs
import re
import tensorflow as tf
from conlleval import return_report


def get_logger(log_file):
    """
    初始化logger
    :param log_file:
    """
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


def result_write_evaluate(results, path, name):
    """
    将对验证集的预测识别结果写入到原数据中并进行输出，然后计算识别的性能；将对测试集的预测识别结果写入到原数据中并进行输出
    :param results:
    :param path:
    :param name:
    :return:
    """
    if name == "dev":
        output_file = os.path.join(path, "ner_predict_dev.utf8")
        with open(output_file, "w", encoding="utf8") as f:
            to_write = []
            for block in results:
                for line in block:
                    to_write.append(line + "\n")
                to_write.append("\n")
            f.writelines(to_write)
        eval_lines = return_report(output_file)
        return eval_lines
    elif name == "test":
        output_file = os.path.join(path, "ner_predict_test.utf8")
        with open(output_file, "w", encoding="utf8") as f:
            to_write = []
            for block in results:
                for line in block:
                    to_write.append(line + "\n")
                to_write.append("\n")
            f.writelines(to_write)


def print_config(config, logger):
    for k, v in config.items():
        logger.info("{}:\t{}".format(k.ljust(15), v))


def make_path(params):
    if not os.path.isdir(params.result_path):
        os.makedirs(params.result_path)
    if not os.path.isdir(params.ckpt_path):
        os.makedirs(params.ckpt_path)
    if not os.path.isdir("log"):
        os.makedirs("log")


def clean(params):
    if os.path.isfile(params.train_dev_file):
        os.remove(params.train_dev_file)
    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)
    if os.path.isfile(params.map_file):
        os.remove(params.map_file)
    if os.path.isdir(params.ckpt_path):
        shutil.rmtree(params.ckpt_path)
    if os.path.isdir(params.summary_path):
        shutil.rmtree(params.summary_path)
    if os.path.isdir(params.result_path):
        shutil.rmtree(params.result_path)
    if os.path.isdir("log"):
        shutil.rmtree("log")
    if os.path.isdir("__pycache__"):
        shutil.rmtree("__pycache__")
    if os.path.isfile(params.config_file):
        os.remove(params.config_file)
    if os.path.isfile(params.vocab_file):
        os.remove(params.vocab_file)


def save_config(config, config_file):
    with open(config_file, "w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def load_config(config_file):
    with open(config_file, encoding="utf8") as f:
        return json.load(f)


def convert_to_text(line):
    to_print = []
    for item in line:
        try:
            if item[0] == " ":
                to_print.append(" ")
                continue
            word, gold, tag = item.split(" ")
            if tag[0] in "SB":
                to_print.append("[")
            to_print.append(word)
            if tag[0] in "SE":
                to_print.append("@" + tag.split("-")[-1])
                to_print.append("]")
        except:
            print(list(item))
    return "".join(to_print)


def save_model(sess, model, path, logger):
    checkpoint_path = os.path.join(path, "ner.ckpt")
    model.saver.save(sess, checkpoint_path)
    logger.info("model saved")


def create_model(session, model_class, path, config, id_to_char, logger):
    """
    创建模型
    :param session:
    :param model_class:模型类名
    :param path:已训练完成的模型参数文件
    :param config:模型参数列表
    :param id_to_char:索引id到字符的映射字典
    :param logger:
    :return:
    """
    model = model_class(config)
    ckpt = tf.train.get_checkpoint_state(path)
    # 若存在被保存的模型参数，则直接从文件中恢复模型
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        logger.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logger.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        if config["pre_emb"]:   # 若使用预训练的词向量
            emb_weights = session.run(model.char_lookup.read_value())
            # 将预训练文件的词向量加载到字典中
            emb_weights = load_word2vec(config["emb_file"], id_to_char, config["char_dim"], emb_weights)
            session.run(model.char_lookup.assign(emb_weights))
            logger.info("Load pre-trained embedding.")
    return model


def load_word2vec(emb_path, id_to_word, word_dim, old_weights):
    """
    从预训练词向量文件中加载词向量，根据emb_path得到键值对word-词向量的字典，然后又根据键值对为id-word的id_to_word，
    最终得到id-词向量的字典，作为最终的词向量矩阵。
    :param emb_path:预训练的词向量文件，其数据为word-词向量
    :param id_to_word:索引id到word的映射字典，键值对为id-word
    :param word_dim:词向量维度
    :param old_weights:原词向量参数
    :return:new_weights[id:vector]
    """
    new_weights = old_weights
    print('Loading pretrained embeddings from {}...'.format(emb_path))
    pre_trained = {}
    emb_invalid = 0
    for i, line in enumerate(codecs.open(emb_path, 'r', 'utf-8')):
        line = line.rstrip().split()
        if len(line) == word_dim + 1:
            pre_trained[line[0]] = np.asarray([float(x) for x in line[1:]]).astype(np.float32)
        else:
            emb_invalid += 1
    if emb_invalid > 0:
        print('WARNING: %i invalid lines' % emb_invalid)
    c_found = 0
    c_lower = 0
    c_zeros = 0
    n_words = len(id_to_word)
    # Lookup table initialization
    for i in range(n_words):
        word = id_to_word[i]
        if word in pre_trained:
            new_weights[i] = pre_trained[word]
            c_found += 1
        elif word.lower() in pre_trained:
            new_weights[i] = pre_trained[word.lower()]
            c_lower += 1
        elif re.sub('\d', '0', word.lower()) in pre_trained:
            new_weights[i] = pre_trained[
                re.sub('\d', '0', word.lower())
            ]
            c_zeros += 1
    print('Loaded %i pretrained embeddings.' % len(pre_trained))
    print('%i / %i (%.4f%%) words have been initialized with ''pretrained embeddings.' %
          (c_found + c_lower + c_zeros, n_words, 100. * (c_found + c_lower + c_zeros) / n_words))
    print('%i found directly, %i after lowercasing, ''%i after lowercasing + zero.' % (c_found, c_lower, c_zeros))
    return new_weights


def result_to_json(string, tags):
    """
    将对一个语句的实体识别结果整理成规范化数据并返回
    :param string:
    :param tags:
    :return:
    """
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx+1, "type": tag[2:]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I":
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2:]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item
