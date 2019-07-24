import os
import re
import codecs

from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features, iobes_iob


def load_sentences(path, lower, zeros):
    """
    加载数据集中的语句，将语句中的字符及对应的标签存储为列表，然后每个语句又单独形成一个列表
    :param path:数据集路径
    :param lower:是否将英文字符小写
    :param zeros:是否将数字全赋值为0
    :return:
    """
    sentences = []  # 存储所有语句
    sentence = []   # 存储一个语句的所有字符及相应的标签
    num = 0
    for line in codecs.open(path, 'r', 'utf8'):
        # print(line)
        num += 1
        # 根据zero参数的值决定是否将所有的数字设为0
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        # print(list(line))
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    # 每句话结束的时候将sentence添加到sentences中
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":
                line = "$" + line[1:]
                # 将每个词与相应的标注存储为一个数组word
                word = line.split()
                # word[0] = " "
            else:
                word = line.split()
            # assert len(word) >= 2, print([word[0]])  # 若训练数据每一行只有一个字符串，则报错（因为每一行应该是word+标签）
            if len(word) == 1:
                word.append("O")
            # 每个word数组添加到sentence中
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    对标注模式进行检查和转换
    :param sentences:
    :param tag_scheme:
    :return:
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # 对IOB标注模式的数据进行处理，确定原数据的标注模式是IOB2格式
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        # 若要转换成IOBES格式
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
    根据数据集的词频创建字典，然后得到字符与索引id的双向映射字典
    :param sentences:
    :param lower:
    :return:
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]
    # 用creat创建字典
    dico = create_dico(chars)   # 创建字典，键值对为word-词频frequency
    # padding字符<PAD>的频数，极大化这一数值，保证最终得到的映射字典中<PAD>的索引为0，因为对序列进行补长的时候，补充的是0
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000    # unknown字符的索引
    # 根据字典得到两种映射
    char_to_id, id_to_char = create_mapping(dico)
    print("Found %i unique words (%i in total)" % (
        len(dico), sum(len(x) for x in chars)
    ))
    # print("char_to_id:", char_to_id)
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    根据数据集的标签频数创建字典，然后得到标签与索引id的双向映射字典
    :param sentences:
    :return:
    """
    tags = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)    # 根据标签出现的频数创建字典
    tag_to_id, id_to_tag = create_mapping(dico)
    print("Found %i unique named entity tags" % len(dico))
    return dico, tag_to_id, id_to_tag


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    基于各映射字典对训练集和验证集的语句序列进行处理，得到将要输入模型的特征列表以及真实标签列表
    :param sentences:
    :param char_to_id:
    :param tag_to_id:
    :param lower:
    :param train:决定对训练集还是测试集进行处理，默认测试集没有标签，所以全部标注为0
    :return:
    """
    none_index = tag_to_id["O"]
    # print("none_index:", none_index)

    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        string = [w[0] for w in s]
        # print(string)
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>']
                 for w in string]
        segs = get_seg_features("".join(string))
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
            # print("tags", tags)
        else:
            tags = [none_index for _ in chars]
        # print("chars:", chars)
        data.append([string, chars, segs, tags])
    # print(segs)
    return data


def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    用预训练词向量文件扩充字典并得到双向映射字典，扩充原则为：存在于预训练词向量文件ext_emb_path和验证集字符列表chars中
    但是不存在于dictionary中的词添加到dictionary中，若chars为空，
    则将所有存在于ext_emb_path但是不存在于dictionary中的词添加到dictionary中。
    :param dictionary:训练集中字符的字典
    :param ext_emb_path:预训练词向量文件
    :param chars:验证集字符列表chars
    :return:
    """
    print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)
    # 将预训练词向量文件中的所有词存入集合
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [
                char,
                char.lower(),
                re.sub('\d', '0', char.lower())
            ]) and char not in dictionary:
                dictionary[char] = 0
    word_to_id, id_to_word = create_mapping(dictionary)  # 根据词频字典得到word-id的双向映射字典
    return dictionary, word_to_id, id_to_word
