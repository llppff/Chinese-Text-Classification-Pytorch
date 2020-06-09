# Chinese-Text-Classification-Pytorch

中文文本分类，BiLSTM_Attention

## 介绍
双向LSTM+Attention

数据以字为单位输入模型，预训练词向量使用 [搜狗新闻 Word+Character 300d]

## 环境
python 3.7  
pytorch 1.1  
tqdm  
sklearn  
tensorboardX

## 中文数据集
THUCNews抽取了20万条，一共十个类别，每个类别二万条

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

预训练sgns.sogou.char文件数据集下载：https://pan.baidu.com/s/1Km9Pa7LGQJE8RE24ha2Aew  提取码：s0ij
## 使用说明
```
# 训练并测试：
python run.py 

# 模型定义：
models/TextRNN_Att.py

# 具体训练和测试过程：
train_eval.py

# 词汇表构造，加载训练集、测试集、验证集数据、并分别构造迭代器：
utils.py

# 数据存放
THUCNews/data文件夹下
```

### 参数
定义模型的文件在models目录下，超参定义和模型定义在同一文件中。  


## 结果可视化步骤
将目录切换到THUCNews/log/TextRNN_Att/根据时间命名的文件夹下，在命令行输入tensorboard --logdir + 参数，将返回的网址复制到浏览器中，断开本地网络后即可查看
