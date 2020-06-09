# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
import models.TextRNN_Att as TextRNN_Att
from utils import build_dataset, build_iterator, get_time_dif

if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集

    #经过utils数据预处理之后得到的embedding.npz文件

    # 搜狗新闻:embedding_SougouNews.npz
    embedding = 'embedding_SougouNews.npz'
    model_name = 'TextRNN_Att'


    #导入参数配置
    config = TextRNN_Att.Config(dataset, embedding)

    #设置种子
    np.random.seed(1)
    torch.manual_seed(1)#为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.cuda.manual_seed_all(1)#为所有的GPU设置种子
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")

    # wd_or_ch取值为True代表词级, 取值为False代表字符级
    wd_or_ch = False
    #获取词汇表、训练集、测试集、验证集
    #得到的词汇表形式是{字1：id1，字2：id2...}，
    #训练、验证、测试集形式是[([句子id列表1], label1, seq_len/pad_size), ([句子id列表2], label2, seq_len/pad_size), ...]
    vocab, train_data, dev_data, test_data = build_dataset(config, wd_or_ch)

    #分别由训练集、测试集、验证集构造迭代器，每次迭代是一个batch的数据
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)

    #获取已运行时间
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train

    # 根据生成的词汇表获取词表大小
    config.n_vocab = len(vocab)
    #用上述定义的参数来构造一个模型
    model = TextRNN_Att.Model(config).to(config.device)

    #初始化模型参数
    init_network(model)
    #输出model所有的参数名
    print(model.parameters)

    #开始训练
    train(config, model, train_iter, dev_iter, test_iter)
