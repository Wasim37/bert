<!--
 * @Author: your name
 * @Date: 2021-08-28 17:29:54
 * @LastEditTime: 2021-08-31 10:28:00
 * @LastEditors: your name
 * @Description: In User Settings Edit
 * @FilePath: \wasim_bert\lesson3\README.md
-->
### 代码结构
代码分为以下几个部分
- run.py： 模型入口，供大家一键运行。
- utils.py： 功能函数，主要包含数据准备与预处理。
- train_eval.py： 训练、验证、测试主函数。
- models/bert.py： BERT模型结构，包含对模型超参数的封装。
- bert_pretrain： 预训练好的 BERT 权重，下载地址见文件中的 README。


### 训练与测试
- 训练和测试脚本(以分类为例，匹配和NER任务参考scripts中的脚本。)
```
python train.sh
```

- 不训练，只测试
```
python eval.sh
```