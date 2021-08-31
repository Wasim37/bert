'''
Author: your name
Date: 2021-08-28 10:27:02
LastEditTime: 2021-08-31 14:37:48
LastEditors: Please set LastEditors
Description: 微博NER数据集处理
FilePath: wasim_bert\bert_torch_lession245\ner_prep.py
'''
# !/usr/bin/env python
# -*- coding: utf-8 -*-
import codecs
import csv
import pandas as pd
import sys

# assert len(sys.argv) == 3, "usage: python ner_prep.py <原始文件路径> <保存路径> "
# file_path = sys.argv[1]
# save_name = sys.argv[2]

# file_path = 'F:/git_ml/wasim_bert/bert_torch_lession245/data/chnsenticorp/train.tsv'
file_path = 'F:/git_ml/wasim_bert/bert_torch_lession245/data/nlp-public-dataset-master/ner-data/weibo/weiboNER_2nd_conll.dev'
save_name = 'dev.tsv'

ner_labels_dict = []
data = []
sentence = []
label_list = []
with codecs.open(file_path, "r", "utf-8") as f:
    for line in f:
        if line.strip() != "":
            tmp = line.strip().split("\t")
            if tmp[-1] not in ner_labels_dict:
                ner_labels_dict.append(tmp[-1])
            sentence.append(tmp[0][0])
            label_list.append(tmp[-1])
        elif line == "\n":
            data.append((sentence, label_list))
            sentence = []
            label_list = []
        else:
            continue

with codecs.open(save_name, "w", "utf-8") as f:
    for ex in data:
        sentence_tmp = "".join(ex[0])
        label_list_tmp = " ".join(ex[1])
        f.write(sentence_tmp + "\t" + label_list_tmp + "\n")

if save_name.find("train.tsv") > -1:
    with open("label.txt", "w") as f:
        for k in ner_labels_dict:
            f.write(k + "\n")
