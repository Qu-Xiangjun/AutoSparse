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


### Loss function
nn.MarginRankingLoss——排序损失 https://blog.csdn.net/qq_50001789/article/details/128974619

### Tokenizer
注意各个映射出来的数字太大要做 Normalize，缩减为小的浮点数


