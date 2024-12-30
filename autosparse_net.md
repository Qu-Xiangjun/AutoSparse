# AutoSparse Net

### 稀疏张量特征提取网络模块
waco 的代码中，使用的多层Conv，实际代码里面只有stride 步长，即只是用了跨步卷积
那么是否可以尝试空洞卷积呢？
> 参考
> https://www.zhihu.com/search?type=content&q=%E7%A9%BA%E6%B4%9E%E5%8D%B7%E7%A7%AF
> https://zhuanlan.zhihu.com/p/50369448

空洞卷积可以有更大的感受野。
但其中连续dilated参数相同时候最后面层的感受野会漏掉附近的一些元素，因此一般不同的空洞参数互相组合为一个 Hybrid Dilated Convolution (HDC) block。

此外，waco文中说明由于其GPU显存问题，无法使用更大的Channel数来容纳特征，只能使用32，所以将多层的卷积Pooling结果拼接作为结果。那我们是否是可以在GPU足够大情况下可以不这样做多层拼接，直接使用最终层的池化结果。

那么就是多个 HDC block 连接形成我们的网络。



### Tokenizer
注意各个映射出来的数字太大要做 Normalize，缩减为小的浮点数

```
2 i 2 i0 256 i1 16 k 2 k0 4096 k1 16 1 A 4 i0 i1 k0 k1 A 4 i0 1 i1 1 k0 1 k1 1 1 j 2 j0 2 j1 64 6 j0 i0 i1 k0 k1 j1 j0 j1 None 0 128 512 445.855559
这个的组成来自 schedule.Schedule.GenConfigCommand的返回值[1] + latency lable
```
除了schedule编码，还需要编码矩阵的信息，参考论文 SpTFS, 包括 规则化后的 shape、nnz、sparsity

1. 符号表映射

2. 原语 one-hot 编码
fsplit freorder fmode lsplit lreorder parallel vectorize unroll sparse_matirx_info sparse_matirx_embedding
3. 特征向量编码组装

### 网络结构
- 使用Transformer decoder模块，NLP语言方便输入各种算子的调度编码
- 网络层数：使用一层还是两层attention 模块
- 稀疏张量特征编码
    - 可以放在第一层 attention模块和输入token一起输入，
    - 也可以放在第一层的结果进行拼凑
- 最终的结果使用SUM求和，然后用rank loss，
    - 可以是WACO 的方式，
    - 也可以是TLP的方式


### Loss function 与 训练方案
WACO 使用的是nn.MarginRankingLoss——排序损失 https://blog.csdn.net/qq_50001789/article/details/128974619
WACO是一个算子一个模型，训练时候一个输入稀疏矩阵的不同数据（大约200个）按照batch=32开始累计做互相的rank loss

TLP的是 randomRankingLoss，然后训练的数据是先对label做处理，将 latency/min_latency 获得同一shape和算子下的相对值，则label=1时候最好，则所有算子的所有shape、所有输入稀疏张量数据都可以放在一起混合训练