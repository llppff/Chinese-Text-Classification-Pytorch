# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'TextRNN_Att'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 10                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.hidden_size2 = 64


# '''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            # nn.Embedding这是一个矩阵类，里面初始化了一个随机矩阵，矩阵的长是字典的大小，宽是用来表示字典中每个元素的属性向量，
            # 类实例化之后可以根据字典中元素的下标来查找元素对应的向量。输入是一个下标的列表，输出是对应的词嵌入
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        # 参数batch_first如果为True则输入输出的数据格式为(batch, seq, feature)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        # 将一个不可训练的类型Tensor转换成可以训练的类型parameter
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        # self.tanh2 = nn.Tanh()
        #torch.nn.Linear(in_features, out_features,bias=True)用于设置全连接层，输入输出都是二维张量
        # in_features是输入的二维张量的大小，即输入的[batch_size,size]中的size
        # out_features是输出的二维张量的大小，即输出的二维张量的形状为[batch_size,output_size]
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.num_classes)

    def forward(self, x):
        # 参数中的x是一个元组，元组第一个元素是一个长度为batch_size的列表，这个列表的每个元素是这个batch中每个句子的id列表，
        # 第二个元素是一个batch中每个句子长度构成的列表
        # 经过下方语句，x只存放句子id列表，_存放句子长度
        x, _ = x
        #得到一个batch中每个句子的每个字的embedding，所以是三维
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        # H是上一刻时间产生的中间值与当前时刻的输入共同产生的状态
        H, _ = self.lstm(emb)  # H:[batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)# [batch_size, seq_len, hidden_size * num_direction]  # [128, 32, 256]
        # 矩阵相乘有torch.mm和torch.matmul两个函数。其中前一个是针对二维矩阵，后一个是高维。当torch.mm用于大于二维时将报错
        # 例如A:(D,x,y),B:(D,y,z),C=torch.matmul(A,B),C的维度是(D,x,z)
        # unsqueeze(-1)表示在最后一维上增加一维
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1) # 只有M和self.w矩阵相乘结果的维度是[128,32],增加维度后是[batch_size, seq_len, 1] # [128, 32, 1]
        out = H * alpha# [batch_size, seq_len, hidden_size * num_direction]  # [128, 32, 256]
        out = torch.sum(out, 1)#[batch_size,hidden_size * num_direction]  # [128, 256]
        out = F.relu(out)#[batch_size,hidden_size * num_direction]
        out = self.fc1(out)#[batch_size,hidden_size2][128, 64]
        out = self.fc(out) #[batch_size,num_classes] [64,10]
        return out
