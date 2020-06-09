# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    # 迭代打印model.named_parameters()将会打印每一次迭代元素的名字和param
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                # Xavier初始化基本思想是通过网络层时，输入和输出的方差相同，包括前向传播和后向传播
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                    #kaiming是针对ReLU的初始化方法
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    #正态分布
                    nn.init.normal_(w)
            elif 'bias' in name:
                #初始化为常数
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)


    # 记录进行到多少batch
    total_batch = 0

    #float('inf')表示正无穷
    dev_best_loss = float('inf')

    # 记录上次验证集loss下降的batch数
    last_improve = 0

    # 记录是否很久没有效果提升
    flag = False

    #这里使用tensorboardX可视化工具，先定义一个SummaryWriter()实例，用scalar方式进行可视化，log_dir为生成的文件所放的目录
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H_%M', time.localtime()))

    print("num epochs:{}".format(config.num_epochs))
    for epoch in range(config.num_epochs):
        print('epoch:{}'.format(epoch + 1))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            #这里的trains是一个元组，元组第一个元素是一个长度为batch的列表，这个列表的每个元素是这个batch中每个句子的id列表，
            # 第二个元素是一个batch中每个句子长度构成的列表
            #得到的输出维度是[batch_size,num_classes]
            outputs = model(trains)
            # zero_grad()直接把模型的参数梯度设成0
            # 当optimizer使用optim.Optimzer进行定义后，那么在模型训练过程中model.zero_grad()和optimzier.zero_grad()两者是等效的
            #关于这里为什么要将参数的梯度设为0：pytorch中梯度是自动累加的。如果梯度不清零，pytorch中会将上次计算的梯度和本次计算的梯度累加
            #如果不想让之前的梯度结果影响本次梯度结果，就需要手动设为0
            model.zero_grad()
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                #求一个batch的每个句子预测的类别
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                #如果本次验证集的损失小于训练到目前为止最小的损失，就存储当前的验证集损失，并且记录当前是第几个batch，同时输出验证集损失时后面加*
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))

                # writer.add_scalar将所需要的数据保存在文件里面供可视化使用，
                # 第一个参数可以理解为保存图的名称，第二个参数可以理解为Y轴数据，第三个参数可以理解为X轴数据
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)

#输出在测试集的精度、损失，矩阵F1得分及困惑度
def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

#test为True时表示是在测试集上，为False时表示是在验证集上
def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    # with torch.no_grad()或者 @ torch.no_grad()中的数据不需要计算梯度，也不会进行反向传播
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)