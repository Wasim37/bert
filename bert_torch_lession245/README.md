### 代码结构
代码分为以下几个部分
- run.py： 模型入口，供大家一键运行。
- utils.py： 功能函数，主要包含数据准备与预处理。
- train_eval.py： 训练、验证、测试主函数。
- models/bert.py： BERT模型结构，包含对模型超参数的封装。
- bert_pretrain： 预训练好的 BERT 权重，下载地址见文件中的 README。


### 数据集
当前目录新建chinese_wwm_pytorch和data文件夹，存放预训练模型文件和数据集。

- 分类数据集：中文情感分析数据集 ChnSentiCorp
下载链接：https://bj.bcebos.com/paddlehub-dataset/chnsenticorp.tar.gz

- 匹配数据集：哈工大语义相似度数据集 LCQMC
下载链接：https://bj.bcebos.com/paddlehub-dataset/lcqmc.tar.gz

- 序列标注数据集：中文微博命名实体识别数据集 weiboNer
下载链接：https://github.com/quincyliang/nlp-public-dataset/tree/master/ner-data/weibo


### 训练与测试
# 训练和测试脚本(以分类为例，匹配和NER任务参考scripts中的脚本。)
python train.sh

# 不训练，只测试
python eval.sh

