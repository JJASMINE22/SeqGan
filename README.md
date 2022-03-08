## SeqGAN: Sequence Generative Adversarial Nets的pytorch实现
---
## 目录
1. [所需环境 Environment](#所需环境)
2. [模型结构 Structure](#模型结构)
3. [注意事项 Attention](#注意事项)
4. [文件下载 Download](#文件下载)
5. [训练步骤 How2train](#训练步骤)
6. [参考资料 Reference](#Reference)

## 所需环境
Python3.7
PyTorch>=1.7.0+cu110  
Numpy==1.19.5
CUDA 11.0+
nltk==3.7
tqdm==4.62.3

## 模型结构
![image](https://github.com/JJASMINE22/SeqGan/blob/main/structure/model_seqgan.png)

## 注意事项
1. 新增基于生成器、对抗器权重的l2正则化误差，降低过拟合概率
2. 新增基于HingeLoss的对抗器误差
3. 更改基于参考资料[1]的蒙特卡洛补偿计算方法，使其与数据特征能正确匹配
4. 数据路径、训练目标等参数均位于config.py，默认使用image_coco数据集

## 文件下载    
链接：https://pan.baidu.com/s/1SNc7uJ3PMxX6gxLrELfjEQ 
提取码：LEAK 
下载解压后放置于config.py中设置的路径即可。  

## 训练步骤
1. 默认使用image_coco进行训练，并使用nltk划分单词。  
2. 运行train.py即可开始训练。

## Reference
1. https://github.com/williamSYSU/TextGAN-PyTorch
