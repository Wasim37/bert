'''
Author: your name
Date: 2021-08-28 10:27:02
LastEditTime: 2021-09-02 14:53:02
LastEditors: Please set LastEditors
Description: In User Settings Edit
'''
# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from pytorch_pretrained import BertModel, BertTokenizer
import os

root_dir = os.path.abspath('.')


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        # 基于BERT的fine-tuning，简单的分类任务来，通常仅需要2-4个epoch 基本就可以收敛，而对于复杂的对话生成、阅读理解等任务，则往往需要10个以上
        self.num_epochs = 3
        # mini-batch大小. 当使用的 GPU 是一台 16G 显存的 TITAN Xp，在内存不爆的情况下，对 batch size 的选取可以遵循：
        # 1w左右小数据32、10w左右中规模数据64、100w以上大规模数据128
        self.batch_size = 64
        # 每句话处理成的长度(短填长切). 可以对训练数据的文本进行可视化处理, 例如采用超过95%文本长度的sent_len作为模型的最大输入长度
        self.pad_size = 32                                              
        # 和从零训练模型不同，由于 BERT 模型本身已经得到充分训练，对指定任务进行 fine-tune 时，学习率设置不应该过大。
        # 《How to Fine-Tune BERT for Text Classification？》一文中提到，BERT 的 fine-tune 学习率设置为 [5e-5, 3e-5, 2e-5] 通常可以取得较好的学习效果。
        self.learning_rate = 5e-5                                       # 学习率
        self.bert_path = './bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.debug = True                                               # 是否Debug调试


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out
