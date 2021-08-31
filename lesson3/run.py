'''
Author: your name
Date: 2021-08-28 10:27:02
LastEditTime: 2021-08-31 16:22:56
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: wasim_bert\lesson3\run.py
'''
# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

# parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
# args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'THUCNews'  # 数据集
    model_name = 'bert'  # bert
    
    # 相对导入
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    
    # 设置随机种子 确保每次运行的条件(模型参数初始化、数据集的切分或打乱等)是一样的
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    start_time = time.time()
    print("Loading data...")
    # 数据预处理: 构建词典、训练集、验证集、测试集
    train_data, dev_data, test_data = build_dataset(config)
    
    # 构建训练集、验证集、测试集的迭代器/生成器（节约内存、避免溢出）
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    # 构建模型对象 并to_device
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter, test_iter)
