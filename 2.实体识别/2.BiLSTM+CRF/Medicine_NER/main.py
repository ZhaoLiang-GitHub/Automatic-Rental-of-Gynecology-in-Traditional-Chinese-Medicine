import os
import pickle
import itertools
from collections import OrderedDict
import tensorflow as tf
import numpy as np
from model import Model
from loader import load_sentences, update_tag_scheme
from loader import char_mapping, tag_mapping
from loader import augment_with_pretrained, prepare_dataset
from utils import get_logger, make_path, clean, create_model, save_model
from utils import print_config, save_config, load_config, result_write_evaluate
from data_utils import input_from_line, BatchManager

flags = tf.flags
# 若要训练则将clean和train设置为True
flags.DEFINE_boolean("clean",       True,      "clean train folder")
flags.DEFINE_boolean("train",       True,      "Wither train the model")
# 若要进行预测则将clean和train均设置为False
# flags.DEFINE_boolean("clean",       False,      "clean train folder")
# flags.DEFINE_boolean("train",       False,      "Wither train the model")
# flags.DEFINE_boolean("predict_line", False, "predict one line data or all dataset")
# 模型参数
# seg_dim为分割特征的维度，分割特征即为词向量，对应的char_dim为词向量的维度，分别对应于英语文本中的词向量和字符向量
flags.DEFINE_integer("seg_dim",     20,         "Embedding size for segmentation, 0 if not used")
flags.DEFINE_integer("char_dim",    100,        "Embedding size for characters")
flags.DEFINE_integer("lstm_dim",    100,        "Num of hidden units in LSTM")
flags.DEFINE_string("tag_schema",   "iobes",    "tagging schema iobes or iob")  # 标注格式
# 模型训练参数
flags.DEFINE_float("clip",          5,          "Gradient clip")
flags.DEFINE_float("dropout",       0.2,        "Dropout rate")
flags.DEFINE_float("batch_size",    256,         "batch size")
flags.DEFINE_float("lr",            0.001,      "Initial learning rate")
flags.DEFINE_string("optimizer",    "adam",     "Optimizer for training")
flags.DEFINE_boolean("pre_emb",     True,       "Wither use pre-trained embedding")
flags.DEFINE_boolean("zeros",       False,      "Wither replace digits with zero")
flags.DEFINE_boolean("lower",       True,       "Wither lower case")
flags.DEFINE_integer("max_epoch",   100,        "maximum training epochs")
flags.DEFINE_integer("steps_check", 100,        "steps per checkpoint")
# 文件路径参数设置
flags.DEFINE_string("ckpt_path",    "ckpt",      "Path to save model")
flags.DEFINE_string("summary_path", "summary",      "Path to store summaries")
flags.DEFINE_string("log_file",     "train.log",    "File for log")
flags.DEFINE_string("train_dev_file",     "train_dev.pkl",     "file for train data and dev data")
flags.DEFINE_string("map_file",     "maps.pkl",     "file for maps")
flags.DEFINE_string("vocab_file",   "vocab.json",   "File for vocab")
flags.DEFINE_string("config_file",  "config_file",  "File for config")
flags.DEFINE_string("script",       "conlleval",    "evaluation script")
flags.DEFINE_string("result_path",  "result",       "Path for results")
flags.DEFINE_string("emb_file",     "word2vec_model.txt", "Path for pre_trained embedding")
# 中医数据集
flags.DEFINE_string("train_file",   os.path.join("data", "3000.train"),  "Path for train data")
flags.DEFINE_string("dev_file",     os.path.join("data", "1150.dev"),    "Path for dev data")
flags.DEFINE_string("test_file",    os.path.join("data", "test.test"),   "Path for test data")

FLAGS = tf.flags.FLAGS
assert FLAGS.clip < 5.1, "gradient clip should't be too much"
assert 0 <= FLAGS.dropout < 1, "dropout rate between 0 and 1"
assert FLAGS.lr > 0, "learning rate must larger than zero"
assert FLAGS.optimizer in ["adam", "sgd", "adagrad"]


class Main:
    def __init__(self):
        self.train_sentences = None  # 用于存储训练集语句中的字符及标签
        self.dev_sentences = None  # 用于存储验证集语句中字符及标签
        self.char_to_id = None  # 字符char到索引id的映射字典
        self.id_to_char = None  # 索引id到字符char的映射字典
        self.tag_to_id = None   # 标签tag到索引id的映射字典
        self.id_to_tag = None   # 索引id到标签tag的映射字典
        self.train_batch_manager = None   # 训练集的batch管理类
        self.dev_batch_manager = None  # 验证集的batch管理类

    @staticmethod
    def config_model(char_to_id, tag_to_id):
        """
        设置模型参数
        :param char_to_id:词到索引的映射字典
        :param tag_to_id:标签到索引的映射字典
        :return:config：dict
        """
        config = OrderedDict()
        config["num_chars"] = len(char_to_id)
        config["char_dim"] = FLAGS.char_dim
        config["num_tags"] = len(tag_to_id)
        config["seg_dim"] = FLAGS.seg_dim
        config["lstm_dim"] = FLAGS.lstm_dim
        config["batch_size"] = FLAGS.batch_size
        config["emb_file"] = FLAGS.emb_file
        config["clip"] = FLAGS.clip
        config["dropout_keep"] = 1.0 - FLAGS.dropout
        config["optimizer"] = FLAGS.optimizer
        config["lr"] = FLAGS.lr
        config["tag_schema"] = FLAGS.tag_schema
        config["pre_emb"] = FLAGS.pre_emb
        config["zeros"] = FLAGS.zeros
        config["lower"] = FLAGS.lower
        return config

    @staticmethod
    def evaluate(sess, model, name, data, id_to_tag, logger):
        if name == "dev":
            logger.info("evaluate dev data......")
            ner_results = model.predict(sess, data, id_to_tag)  # 对验证集进行预测，得到对各个实体的预测
            # 将预测结果写入到原数据并输出，然后计算并评估识别性能
            eval_lines = result_write_evaluate(ner_results, FLAGS.result_path, name)
            for line in eval_lines:
                logger.info(line)
            f1 = float(eval_lines[1].strip().split()[-1])
            best_test_f1 = model.best_dev_f1.eval()
            if f1 > best_test_f1:
                tf.assign(model.best_dev_f1, f1).eval()
                logger.info("new best dev f1 score:{:>.3f}".format(f1))
            return f1 > best_test_f1

    def get_sentences_dict(self):
        """
        加载数据集中的语句，将每个语句的字符和标签存储为列表，然后生成字符和标签与索引id的双向映射字典
        :return:
        """
        # 加载数据集中的语句，将每个语句的字符和标签存储为列表
        self.train_sentences = load_sentences(FLAGS.train_file, FLAGS.lower, FLAGS.zeros)
        self.dev_sentences = load_sentences(FLAGS.dev_file, FLAGS.lower, FLAGS.zeros)
        # print("dev_sentences:", self.dev_sentences)

        # 原数据的标注模式与需要的标注模式不同时用update_tag_scheme函数对标注模式进行转换，转换成指定的IOB或者IOBES
        # update_tag_scheme(train_sentences, FLAGS.tag_schema)
        # update_tag_scheme(test_sentences, FLAGS.tag_schema)

        if not os.path.isfile(FLAGS.map_file):
            # 若map_file不存在，则根据数据集和预训练词向量文件初始化各个映射字典
            # 若使用预训练的词向量
            if FLAGS.pre_emb:
                # 得到train_sentences中字符的字典，键值对为word-词频
                dico_chars_train = char_mapping(self.train_sentences, FLAGS.lower)[0]
                # 用预训练词向量文件扩充字典（目的为尽可能地扩充字典、使更多字符能基于预训练的词向量进行初始化）并得到word与id的双向映射字典。
                dico_chars, self.char_to_id, self.id_to_char = augment_with_pretrained(
                    dico_chars_train.copy(),
                    FLAGS.emb_file,
                    list(itertools.chain.from_iterable(
                        [[w[0] for w in s] for s in self.dev_sentences])
                    )
                )
            else:   # 若不使用预训练的词向量
                _c, self.char_to_id, self.id_to_char = char_mapping(self.train_sentences, FLAGS.lower)
            _t, self.tag_to_id, self.id_to_tag = tag_mapping(self.train_sentences)  # 标签和索引之间的双向映射字典
            print("tag_to_id", self.tag_to_id, len(self.tag_to_id))
            # 将得到的映射字典存入文件，以免重复初始化
            with open(FLAGS.map_file, "wb") as f:
                pickle.dump([self.char_to_id, self.id_to_char, self.tag_to_id, self.id_to_tag], f)
        else:
            # 若map_file存在，则直接从文件中恢复各个映射字典
            with open(FLAGS.map_file, "rb") as f:
                self.char_to_id, self.id_to_char, self.tag_to_id, self.id_to_tag = pickle.load(f)

    def get_batch_data(self):
        """
        得到训练集和验证集的batch管理类：首先基于各映射字典对训练集和验证集的语句序列进行处理，得到每个语句的各特征列表以及
        真实标签列表，然后获取batch管理类，用于生成batch数据
        :return:
        """
        if not os.path.isfile(FLAGS.train_dev_file):
            train_data = prepare_dataset(self.train_sentences, self.char_to_id, self.tag_to_id, FLAGS.lower)
            dev_data = prepare_dataset(self.dev_sentences, self.char_to_id, self.tag_to_id, FLAGS.lower)
            with open(FLAGS.train_dev_file, "wb") as f:
                pickle.dump([train_data, dev_data], f)
        else:
            with open(FLAGS.train_dev_file, "rb") as f:
                train_data, dev_data = pickle.load(f)
        print("%i / %i  sentences in train / dev ." % (len(train_data), len(dev_data)))
        self.train_batch_manager = BatchManager(train_data, int(FLAGS.batch_size))
        self.dev_batch_manager = BatchManager(dev_data, int(FLAGS.batch_size))

    def get_config(self):
        """
        从模型参数配置文件中获取参数或者用config_model函数生成参数并存储
        :return:日志logger及参数列表config
        """
        make_path(FLAGS)
        if os.path.isfile(FLAGS.config_file):
            config = load_config(FLAGS.config_file)
        else:
            config = self.config_model(self.char_to_id, self.tag_to_id)
            save_config(config, FLAGS.config_file)
        log_path = os.path.join("log", FLAGS.log_file)
        logger = get_logger(log_path)
        print_config(config, logger)
        return logger, config

    def train(self):
        self.get_sentences_dict()
        self.get_batch_data()
        logger, config = self.get_config()
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True  # limit GPU memory
        steps_per_epoch = self.train_batch_manager.len_data  # 每一轮epoch的batch数量
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, FLAGS.ckpt_path, config, self.id_to_char, logger)
            logger.info("start training")
            loss = []
            for i in range(FLAGS.max_epoch):
                for batch in self.train_batch_manager.iter_batch(shuffle=True):
                    step, batch_loss = model.run_step(sess, True, batch)
                    loss.append(batch_loss)
                    if step % FLAGS.steps_check == 0:
                        iteration = step // steps_per_epoch + 1
                        logger.info("iteration:{} step:{}/{}, ""NER loss:{:>9.6f}".format(
                            iteration, step % steps_per_epoch, steps_per_epoch, np.mean(loss)))
                        loss = []
                # 对验证集进行预测和评估
                best = self.evaluate(sess, model, "dev", self.dev_batch_manager, self.id_to_tag, logger)
                if best:
                    save_model(sess, model, FLAGS.ckpt_path, logger)

    @ staticmethod
    def predict():
        """
        对一个数据集进行实体识别
        :return:
        """
        config = load_config(FLAGS.config_file)
        logger = get_logger(FLAGS.log_file)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True   # limit GPU memory
        # 从训练阶段生成的map_file中恢复各映射字典
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
        test_sentences = load_sentences(FLAGS.test_file, FLAGS.lower, FLAGS.zeros)
        test_data = prepare_dataset(test_sentences, char_to_id, tag_to_id, FLAGS.lower, train=False)
        test_manager = BatchManager(test_data, 1)
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, FLAGS.ckpt_path, config, id_to_char, logger)
            logger.info("predict data......")
            ner_results = model.predict(sess, test_manager, id_to_tag)
            result_write_evaluate(ner_results, FLAGS.result_path, "test")

    @staticmethod
    def predict_line():
        """
        对一个语句实例进行实体识别测试
        :return:
        """
        config = load_config(FLAGS.config_file)
        logger = get_logger(FLAGS.log_file)
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        with open(FLAGS.map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)
        with tf.Session(config=tf_config) as sess:
            model = create_model(sess, Model, FLAGS.ckpt_path, config, id_to_char, logger)
            # 对单个句子进行预测
            while True:
                line = input("请输入测试句子:")
                result = model.predict_line(sess, input_from_line(line, char_to_id), id_to_tag)
                print(result)

if __name__ == "__main__":
    main = Main()
    if FLAGS.train:
        if FLAGS.clean:
            clean(FLAGS)
        main.train()
    elif FLAGS.predict_line:
        main.predict_line()
    else:
        main.predict()
