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

### 训练时问题
- 总是出现OOM，
    - 尝试batch_size 小一点，小到32还是有OOM，检查发现可能是sparse matrix太大，导致conv net 会很大，
    - 因此筛选哪些matrix不能用，一般来说 nnz越大，占显存就会越大
    - 那么对total_dataset.txt 中数据都跑一遍，注意schedule_batch 提前跳出即可，看什么数据有OOM


### 模型评估指标
1. loss值
loss值越趋近0越好

2. topk acc
见Tenset和TLP中的定义，即预测出的topk，拿到他们在原来 true runtimes列表中的runtime值，找其中最小的 pre_true_min ，将全true runtimes最小的 true_min 除以 pre_true_min，即  true_min / pre_true_min, 可知这个结果在[0,1]区间，找的越好，则值越接近1，就是acc准确率
![alt text](image-3.png)


### 结果呈现

1. loss 的训练下降图
    ![alt text](image-5.png)
    baseline选择：WACONet_spmm, AutoSparseNet_WACONet_spmm, AutosparseNet_spmm, AutoSparseNet_Full
    结果：AutosparseNet_spmm, AutoSparseNet_Full的下降效果最好，loss能达到最低，或者是更快收敛
2. TopK Acc
    baseline选择：WACONet_spmm, AutoSparseNet_WACONet_spmm, AutosparseNet_spmm, AutoSparseNet_Full
    结果1：AutosparseNet_spmm 比 前两个效果都好
    结果2：AutoSparseNet_Full能在数据和但算子持平情况达到和单算子一样的效果
    ![alt text](image-2.png)

3. 带costmodel的搜索找到的算子性能和普通搜索齐平

4. 加速搜索效率效果
    造3个图：
    - 图1 3中每一个的搜索时间对比，是否有costmodel，柱状图
    - 图2 是搜索的收敛图，在3中每一个，更快达到最佳性能结果
    - 图3 从3中调一组，搜索程序中各个部分时间分布图，测试性能部分降低多少倍，或者是说减少的硬件测试次数
    ![alt text](image-4.png)
    ![alt text](image-7.png)
    ![alt text](image.png)
    ![alt text](image-6.png)
    

    
### 模型训练日志
##### 2025-01-04
| 模型名称 | 模型结构 | 数据集选择 | TrainLoss | ValLoss | Top1Acc | Top5Acc | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 行1数据1 | 行1数据2 | 行1数据3 | 行1数据1 | 行1数据2 | 行1数据3 | 行1数据1 | 行1数据2 |
| 行1数据1 | 行1数据2 | 行1数据3 | 行1数据1 | 行1数据2 | 行1数据3 | 行1数据1 | 行1数据2 |
| 行1数据1 | 行1数据2 | 行1数据3 | 行1数据1 | 行1数据2 | 行1数据3 | 行1数据1 | 行1数据2 |
| 行1数据1 | 行1数据2 | 行1数据3 | 行1数据1 | 行1数据2 | 行1数据3 | 行1数据1 | 行1数据2 |
| 行1数据1 | 行1数据2 | 行1数据3 | 行1数据1 | 行1数据2 | 行1数据3 | 行1数据1 | 行1数据2 |
| 行1数据1 | 行1数据2 | 行1数据3 | 行1数据1 | 行1数据2 | 行1数据3 | 行1数据1 | 行1数据2 |
| 行1数据1 | 行1数据2 | 行1数据3 | 行1数据1 | 行1数据2 | 行1数据3 | 行1数据1 | 行1数据2 |
| 行2数据1 | 行2数据2 | 行2数据3 |


### TODO
- ~~配置AutoSparse 测试各个部分的时间在 EPYC_7543 服务器~~
- WACONet的baseline训练
    - 按照WACO的方案设计一个新的AutoSparse spmm 算子模版
    - 训练spmm单算子基于已有的AutoSparse数据
    - 按照WACO的方案设计一个新的AutoSparse spmv 算子模版
    - 训练spmv单算子基于已有的AutoSparse数据
    - 按照WACO的方案设计一个新的AutoSparse sddmm 算子模版
    - 训练sddmm单算子基于已有的AutoSparse数据 
- TrainNaive 中增加训练并行度，第一步解决跨算子级合并一次计算与backward，第二步解决跨矩阵合并
- 集成进入AutoSparse的搜索中
- 画出loss图，修改模型结构
- 尝试TrainNaive 上使用 randomloss func，看看结果效果
- 实现TrainMix版本，同时考虑如何增加并行性，比如分矩阵类型减少矩阵conv的批大小，然后合并所有的小批进行batch atention计算， 最后分矩阵的批次进行backward
- 