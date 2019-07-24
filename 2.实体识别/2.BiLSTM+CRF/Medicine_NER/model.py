# encoding = utf8
import numpy as np
import tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
# from crf_test import crf_log_likelihood
from tensorflow.contrib.layers.python.layers import initializers
import rnncell as rnn
from utils import result_to_json
from data_utils import create_input, iobes_iob


class Model(object):
    def __init__(self, config):

        # 从参数列表中获取模型参数
        self.config = config
        self.lr = config["lr"]
        self.char_dim = config["char_dim"]  # 字符的词向量维度
        self.lstm_dim = config["lstm_dim"]  # lstm隐层神经元数量
        self.seg_dim = config["seg_dim"]    # 字符的分割特征维度
        self.num_tags = config["num_tags"]  # 标签数量
        self.num_chars = config["num_chars"]    # 字符数量
        self.num_segs = 4   # 分割特征的数量
        # 设置全局变量
        self.global_step = tf.Variable(0, trainable=False)
        self.best_dev_f1 = tf.Variable(0.0, trainable=False)
        self.best_test_f1 = tf.Variable(0.0, trainable=False)
        self.initializer = initializers.xavier_initializer()
        # 设置输入占位符
        self.char_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="ChatInputs")    # 字符特征，由字符的索引id组成
        self.seg_inputs = tf.placeholder(dtype=tf.int32, shape=[None, None], name="SegInputs")  # 分割特征，由每个字符的分割特征索引组成
        self.targets = tf.placeholder(dtype=tf.int32, shape=[None, None], name="Targets")   # 真实标签
        self.dropout = tf.placeholder(dtype=tf.float32, name="Dropout")
        # 设置变量
        self.char_lookup = None    # 词向量矩阵，初始化模型的时候，通过预训练词向量进行初始化
        self.seg_lookup = None  # 分割特征向量矩阵
        self.trans = None  # 状态转移矩阵，在loss层中进行计算

        used = tf.sign(tf.abs(self.char_inputs))    # 计算序列中索引非0字符的数量
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)    # 记录序列除去padding（索引为0）的真实长度
        self.batch_size = tf.shape(self.char_inputs)[0]
        self.num_steps = tf.shape(self.char_inputs)[-1]  # 序列总长度

        # 构造tensor的传递
        embedding = self.embedding_layer()  # 通过embedding_layer得到字词向量拼接后的特征向量
        lstm_inputs = tf.nn.dropout(embedding, self.dropout)    # dropout层
        lstm_outputs = self.bilstm_layer(lstm_inputs)  # 双向BiLSTM层
        self.logits = self.project_layer(lstm_outputs)  # 进行预测，得到对每个字符是每个标签的概率
        self.loss = self.loss_layer(self.logits)  # 计算loss
        # 设置训练阶段的优化算法
        with tf.variable_scope("optimizer"):
            optimizer = self.config["optimizer"]
            if optimizer == "sgd":
                self.opt = tf.train.GradientDescentOptimizer(self.lr)
            elif optimizer == "adam":
                self.opt = tf.train.AdamOptimizer(self.lr)
            elif optimizer == "adgrad":
                self.opt = tf.train.AdagradOptimizer(self.lr)
            else:
                raise KeyError
            # 设置梯度裁剪（grad clip）以避免梯度爆炸
            grads_vars = self.opt.compute_gradients(self.loss)
            capped_grads_vars = [[tf.clip_by_value(g, -self.config["clip"], self.config["clip"]), v]
                                 for g, v in grads_vars]
            self.train_op = self.opt.apply_gradients(capped_grads_vars, self.global_step)
        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)   # 模型保存设置

    def embedding_layer(self):
        """
        词嵌入层，将语句的词序列转换为词向量与分割特征序列转换为词向量
        :return:[batch_size, num_steps, embedding size]
        """
        embedding = []
        with tf.variable_scope("char_embedding"):
            # print("num_chars", self.num_chars)
            self.char_lookup = tf.get_variable(name="char_embedding", shape=[self.num_chars, self.char_dim],
                                               initializer=self.initializer)
            embedding.append(tf.nn.embedding_lookup(self.char_lookup, self.char_inputs))
            if self.config["seg_dim"]:
                with tf.variable_scope("seg_embedding"):
                    # print("num_segs", self.seg_dim)
                    self.seg_lookup = tf.get_variable(name="seg_embedding", shape=[self.num_segs, self.seg_dim],
                                                      initializer=self.initializer)
                    embedding.append(tf.nn.embedding_lookup(self.seg_lookup, self.seg_inputs))
            embed = tf.concat(embedding, axis=-1)
            # print(embed.shape)
        return embed

    def bilstm_layer(self, lstm_inputs):
        """
        BiLSTM层
        :param lstm_inputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, 2*lstm_dim] 
        """
        with tf.variable_scope("char_BiLSTM"):
            lstm_cell = {}
            for direction in ["forward", "backward"]:
                with tf.variable_scope(direction):
                    lstm_cell[direction] = rnn.CoupledInputForgetGateLSTMCell(
                        self.lstm_dim,   # 每个LSTM cell内部的神经元数量（即隐层参数维度）
                        use_peepholes=True, initializer=self.initializer, state_is_tuple=True)
            outputs, final_states = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell["forward"],   # 前向传播cell
                lstm_cell["backward"],  # 后向传播cell
                lstm_inputs,
                dtype=tf.float32,
                sequence_length=self.lengths)
        return tf.concat(outputs, axis=2)

    def project_layer(self, lstm_outputs):
        """
        根据lstm的输出对序列中每个字符进行预测，得到每个字符是每个标签的概率
        :param lstm_outputs: [batch_size, num_steps, emb_size] 
        :return: [batch_size, num_steps, num_tags]
        """
        with tf.variable_scope("project"):
            # 隐层的计算
            with tf.variable_scope("hidden"):
                w = tf.get_variable("W", shape=[self.lstm_dim*2, self.lstm_dim],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.lstm_dim], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                output = tf.reshape(lstm_outputs, shape=[-1, self.lstm_dim*2])
                hidden = tf.tanh(tf.nn.xw_plus_b(output, w, b))
            # 得到标签概率
            with tf.variable_scope("logits"):
                w = tf.get_variable("W", shape=[self.lstm_dim, self.num_tags],
                                    dtype=tf.float32, initializer=self.initializer)

                b = tf.get_variable("b", shape=[self.num_tags], dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

                pred = tf.nn.xw_plus_b(hidden, w, b)

            return tf.reshape(pred, [-1, self.num_steps, self.num_tags])

    def loss_layer(self, project_logits):
        """
        通过CRF层计算loss
        :param project_logits: [1, num_steps, num_tags]
        :return: scalar loss
        """
        with tf.variable_scope("crf_loss"):
            small = -1000.0
            # 设置计算首个字符的概率，用于在CRF中计算真实首个字符的转移概率
            start_logits = tf.concat([small * tf.ones(shape=[self.batch_size, 1, self.num_tags]),
                                      tf.zeros(shape=[self.batch_size, 1, 1])], axis=-1)
            pad_logits = tf.cast(small * tf.ones([self.batch_size, self.num_steps, 1]), tf.float32)
            logits = tf.concat([project_logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)
            targets = tf.concat(
                [tf.cast(self.num_tags*tf.ones([self.batch_size, 1]), tf.int32), self.targets], axis=-1)
            print("targets:", targets)
            self.trans = tf.get_variable("transitions", shape=[self.num_tags + 1, self.num_tags + 1],
                                         initializer=self.initializer)
            log_likelihood, self.trans = crf_log_likelihood(inputs=logits, tag_indices=targets,
                                                            transition_params=self.trans,
                                                            sequence_lengths=self.lengths+1)
            return tf.reduce_mean(-log_likelihood)

    def create_feed_dict(self, is_train, batch):
        """
        创建feed_dict
        :param is_train: Flag, True for train batch, False for dev batch and test batch
        :param batch: batch列表，主要包括string, chars, segs, tags
        :return: structured data to feed
        """
        _, chars, segs, tags = batch
        feed_dict = {
            self.char_inputs: np.asarray(chars),
            self.seg_inputs: np.asarray(segs),
            self.dropout: 1.0,
        }
        if is_train:
            feed_dict[self.targets] = np.asarray(tags)
            feed_dict[self.dropout] = self.config["dropout_keep"]
        return feed_dict

    def run_step(self, sess, is_train, batch):
        """
        # 运行sess
        :param sess: session
        :param is_train: 指定是训练还是验证的flag
        :param batch:
        :return: batch result, loss of the batch or logits
        """
        feed_dict = self.create_feed_dict(is_train, batch)
        if is_train:
            global_step, loss, _ = sess.run([self.global_step, self.loss, self.train_op], feed_dict)
            return global_step, loss
        else:
            lengths, logits = sess.run([self.lengths, self.logits], feed_dict)
            return lengths, logits

    def decode(self, logits, lengths, trans):
        """
        通过project_layer层得到的每个字符的标签概率和通过loss层得到的标签转移概率矩阵后，
        利用维特比算法对序列标签进行预测
        :param logits: 对序列中字符标签的预测[batch_size, num_steps, num_tags]
        :param lengths: 每个序列除去padding字符的真实长度[batch_size]
        :param trans: 状态转移概率矩阵
        :return:
        """
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.num_tags + [0]])   # start是pad的概率为最大
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])  # 极小化预测出pad的概率
            logits = np.concatenate([score, pad], axis=1)   # 添加每个字符是pad的概率
            logits = np.concatenate([start, logits], axis=0)    # 将start的概率添加到序列前面
            path, _ = viterbi_decode(logits, trans)
            paths.append(path[1:])
        return paths

    def predict(self, sess, data_manager, id_to_tag):
        """
        对一个数据集进行预测
        :param sess:
        :param data_manager:batch管理类
        :param id_to_tag:
        :return:results:预测结果列表
        """
        results = []
        trans = self.trans.eval()  # tensor.eval()作用类似于sess.run()，目的在于运行图获取tensor,返回一个array
        for batch in data_manager.iter_batch():
            # batch = [sentences, chars(word的id), segs(分割特征), tags]
            strings = batch[0]  # 原语句的字符列表
            tags = batch[-1]    # 原语句的tags列表
            lengths, logits = self.run_step(sess, False, batch)  # 运行sess进行预测，获取对每个字符的预测
            batch_paths = self.decode(logits, lengths, trans)   # 利用维特比算法基于状态概率和状态转移概率进行解码
            # print("batch_paths", batch_paths)
            for i in range(len(strings)):
                result = []
                string = strings[i][:lengths[i]]
                # print("string:", string)
                gold = [id_to_tag[int(x)] for x in tags[i][:lengths[i]]]
                pred = [id_to_tag[int(x)] for x in batch_paths[i][:lengths[i]]]
                for char, gold, pred in zip(string, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                results.append(result)
        # print("results", results)
        return results

    def predict_line(self, sess, sentence, id_to_tag):
        """
        一个语句实例进行实体识别测试
        :param sess:
        :param sentence:
        :param id_to_tag:
        :return:
        """
        trans = self.trans.eval()
        lengths, logits = self.run_step(sess, False, sentence)
        batch_paths = self.decode(logits, lengths, trans)
        tags = [id_to_tag[idx] for idx in batch_paths[0]]
        return result_to_json(sentence[0][0], tags)
