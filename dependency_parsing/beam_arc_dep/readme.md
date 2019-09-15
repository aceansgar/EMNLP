
# HanLP KBeamArcEagerDependencyParser

transition-based 

不使用预训练模型，自行根据训练集训练

只需两次迭代训练的模型即可达到如下准确率

UAS=72.24 LAS=65.99

## notes

HanLP的词典等数据是分开存储的，需要修改propertie文件

在jar包目录下创建文件夹data/，存储train.conllu,test.conllu以及生成的数据、模型等

可能程序运行结束却不停止，可能是HanLP在while循环中试图读别的文件，终止程序即可

## train ("train" means train+test)

    java -jar train data/train.conllu data/test.conllu 5

分别为训练文件，测试文件，训练迭代次数

## test

    java -jar test data/test.conllu
