# coding: UTF-8
import os
import torch
import numpy as np
import pickle as pkl
from tqdm import tqdm
import time
from datetime import timedelta


MAX_VOCAB_SIZE = 10000  # 词表长度限制
UNK, PAD = '<UNK>', '<PAD>'

#构造词汇表具体过程
def build_vocab(file_path, tokenizer, max_size, min_freq):
    vocab_dic = {}
    with open(file_path, 'r', encoding='UTF-8') as f:
        # tqdm是一个快速，可扩展的Python进度条，可以在Python长循环中添加一个进度提示信息，只需要添加任意一个迭代器即可
        for line in tqdm(f):
            # str.strip()就是把字符串(str)的头和尾的空格，以及位于头尾的\n \t之类给删掉。
            lin = line.strip()
            if not lin:
                continue
            content = lin.split('\t')[0]
            #针对训练集文件中的每句话的每个字进行词汇表字典构建，最终字典里存放的key-value分别对应训练集的每个字和该字出现的次数
            for word in tokenizer(content):
                # mydict.get(label,num)返回字典mydict中label元素对应的值, 若无，则初始化为num
                vocab_dic[word] = vocab_dic.get(word, 0) + 1
        #取出词汇表字典出现频率最高的前max_size个字作为词汇表list
        # （进行字频排序时，先选出那些出现频率不小于要求的最低频率的字组成列表，列表的每个元素为一个元组形式，即key-value）
        vocab_list = sorted([_ for _ in vocab_dic.items() if _[1] >= min_freq], key=lambda x: x[1], reverse=True)[:max_size]
        #根据每个字出现次数从高到低的排名来决定该字对应的id，最终的词汇表格式即为{字1：id1，字2：id2...}
        #enumerate可以将一个可遍历的数据对象组合为一个索引序列，同时列出数据下标和数据，即下方的idx取值为：0,1...，word_count取值为('字',num)
        vocab_dic = {word_count[0]: idx for idx, word_count in enumerate(vocab_list)}
        #加入特殊字和对应的长度，'PAD'是句子长度不够是进行填充的字符，'UNK'是当训练集/测试集/验证集中出现词汇表中没有的字符时，将这些字对应的id设置为'UNK'的id
        vocab_dic.update({UNK: len(vocab_dic), PAD: len(vocab_dic) + 1})
    return vocab_dic

#获取词汇表、训练集、测试集、验证集
def build_dataset(config, ues_word):
    #词级
    if ues_word:
        tokenizer = lambda x: x.split(' ')  # 以空格隔开，word-level
    else:
        #字符级
        tokenizer = lambda x: [y for y in x]  # char-level

    #如果词汇表存在，就打开，否则用train文件构造词汇表
    if os.path.exists(config.vocab_path):
        vocab = pkl.load(open(config.vocab_path, 'rb'))
    else:
        #得到的词汇表是一个字典，key是每个字，value是对应的id，其中key包括了两个特殊字符'UNK'和'PAD'，对应的id分别为不包括这两个特殊字符时词汇表的长度和词汇表长度+1
        vocab = build_vocab(config.train_path, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        pkl.dump(vocab, open(config.vocab_path, 'wb'))
    print(f"Vocab size: {len(vocab)}")

    # pad_size是每句话处理成的长度，短填长切
    def load_dataset(path, pad_size=32):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label = lin.split('\t')
                words_line = []

                #如果是词级则获取当前句子每个字的列表，如果是字符级则获取当前句子每个字符构成的列表
                token = tokenizer(content)

                #获取当前句子长度，并和要求处理的长度进行对比，短则用[PAD]填，长则切掉多余的
                seq_len = len(token)
                if pad_size:
                    if len(token) < pad_size:
                        token.extend([PAD] * (pad_size - len(token)))
                    else:
                        token = token[:pad_size]
                        seq_len = pad_size
                #将当前句子中每个字对应的id加入到words_line列表中，如果是词汇表中没出现的字，则id记为构造词汇表时添加的'UNK'的value
                for word in token:
                    words_line.append(vocab.get(word, vocab.get(UNK)))

                #contents是一个列表，其每个元素是一个元组，每个元组有三个分量：
                #第一个分量表示每个句子对应的id组成的list，第二个分量是当前句子的类别，
                #第三个分量是句子长度（如果句子大于等于pad_size，则长度为pad_size，否则为实际句子长度）
                contents.append((words_line, int(label), seq_len))
        return contents  # [([一个句子对应的每个字的id列表], 0,32), ([...], 1,32), ...]
    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return vocab, train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        #这里的batches对应训练、验证、测试集经过build_dataset后的数据集
        self.batches = batches

        self.n_batches = len(batches) // batch_size

        # 记录batch数量是否为整数，取值为False则batch为整数
        self.residue = False

        #len(batches)即为该集中的句子数
        if len(batches) % self.n_batches != 0:
            self.residue = True

        #表示当前迭代到第几个batch的数据
        self.index = 0

        self.device = device


    #将当前batch的每个句子转为Longtensor，x是一个列表，这个列表的每个元素是这个batch中每个句子的id列表，y是每个句子对应的类别列表，seq_len是每个句子的长度列表
    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)

        #将句子id列表和句子长度组合为一个元组，再整体和y组成一个元组
        return (x, seq_len), y

    def __next__(self):
        #如果不是整数个batch，那么到最后一个batch时，把剩下的句子都取出来
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        #如果能取整并且训练的batch数已经不少于一共有的batch数，就停止迭代
        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            #如果是正常过程期间，就取一个batch的句子
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            #执行的batch数加1
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


if __name__ == "__main__":
    '''提取预训练词向量'''
    # 下面的目录、文件名按需更改。
    train_dir = "./THUCNews/data/train.txt"
    vocab_dir = "./THUCNews/data/vocab.pkl"
    pretrain_dir = "./THUCNews/data/sgns.sogou.char"
    emb_dim = 300
    filename_trimmed_dir = "./THUCNews/data/embedding_SougouNews"

    #如果有词汇表存在则打开已有的词汇表，否则根据现有的train文件来构造词汇表
    if os.path.exists(vocab_dir):
        word_to_id = pkl.load(open(vocab_dir, 'rb'))
    else:
        # tokenizer = lambda x: x.split(' ')  # 以词为单位构建词表(数据集中词之间以空格隔开)
        tokenizer = lambda x: [y for y in x]  # 以字为单位构建词表
        word_to_id = build_vocab(train_dir, tokenizer=tokenizer, max_size=MAX_VOCAB_SIZE, min_freq=1)
        #pickle.dump(obj,file,[,protocol])序列化对象，将对象obj保存到file中
        # file可以是'w'或'wb'打开的文件或任何实现write接口的对象，最后一个参数控制序列化模式：文本形式/二进制形式
        pkl.dump(word_to_id, open(vocab_dir, 'wb'))

    embeddings = np.random.rand(len(word_to_id), emb_dim)
    f = open(pretrain_dir, "r", encoding='UTF-8')
    #用预训练词汇来更新embedding
    for i, line in enumerate(f.readlines()):
        #strip()用于去掉字符串首尾的空格、\n 和\t
        lin = line.strip().split(" ")
        if lin[0] in word_to_id:
            idx = word_to_id[lin[0]]
            emb = [float(x) for x in lin[1:301]]
            embeddings[idx] = np.asarray(emb, dtype='float32')
    f.close()
    #np.savez_compressed将得到的文件进行打包并压缩，numpy.savez是将得到的文件打包,不压缩
    np.savez_compressed(filename_trimmed_dir, embeddings=embeddings)
