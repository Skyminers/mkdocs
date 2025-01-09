# ResNet 面试资料整理

---

## 简介

> [《Deep Residual Learning for Image Recognition》](https://arxiv.org/pdf/1512.03385)

Deep Residual Network, 深度残差神经网络，简称 ResNet。通过残差学习使训练更深的网络成为可能。ResNet 有很多种不同深度的网络，ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 等。

### 网络架构

结构组成：

1. 初始卷积层：初步进行特征提取
   - 初始卷积层 7x7，步长 2，padding 3，将维度减半。后面跟 BN 层和 ReLU 层。
   - 进行图像特征进行基础的提取
2. 残差块组：包含多个残差单元
   - 每个残差块可以从前一组提取的特征中提取更高级的特征
   - 通过残差连接，每个残差块能学习非线形映射
   - Skip Connection 可以更好地传递梯度
3. 全局平均池化：减少维度
   - 全局平均池化将每个特征图缩减为一个单一的值，显著减少了计算量
   - 防止过拟合，提升泛化能力。减少参数有助于防止模型过拟合
4. 全链接层：可以用于分类或其他任务
   - 可以根据任务需求根据前层特征进行分类或回归
   - 完全整合之前各层的信息，输出一个固定大小的特征向量。

### 残差单元

![](./assets/residual.png)

传统的 CNN 中，每个卷积层学习的都是输入与输出之间的映射。残差块则采用了不同的策略，尝试学习输入和输出之间的残差映射。

---

## 问答

### 为什么有效

通过 BN 和合理的初始化结局了梯度消失和梯度爆炸的问题。残差连接可以更容易地学到类恒等映射，这使 ResNet 的学习任务更简单，可以有效地解决梯度弥散的问题。

### ResNet152 中有什么特殊的设计

在更深的 ResNet 中，位了减少计算量通常先使用 1x1 的卷积核进行降维，然后再进行 3x3 卷积，最后再通过 1x1 卷积恢复维度。

### 请简要介绍 ResNet

ResNet 通过将多个神经网络的层聚合成一个块，然后再块的一侧加入恒等映射，使这个块从原本的 $F(x)$ 变成了 $F(x)+x$，从而解决了神经网络的退化问题。

> ResNet 引入跳跃链接，使得梯度能够更好地回传，从而缓解了梯度消失与梯度爆炸问题。

上面这个描述实际上不准确，论文中明确说明优化困难不是由于梯度消失而导致的。

### 什么是梯度消失？传统的激活函数在两端会进入梯度饱和区，从而导致梯度消失。但现代激活函数比如ReLU，他在输入为正时恒为1，那么应当就没有梯度消失问题才对啊？

ReLU 在输入为负数时恒为 0，因此对梯度消失的效果有限。ResNet 的梯度消失和爆炸的问题主要是通过 BN 和初始化来解决的。

### Batchnorm 与 LayerNorm

作用都是将输入的数据归一化，均值变为 $0$，方差变为 $1$。

Batchnorm 是对每个 batch 的某一个维度组合成的特征进行归一化，LayerNorm 是对每个样本的特征进行归一化。Transformers 中使用 LayerNorm，因为理解上来说不同 Batch 样本中的某个特征不存在相关性，而同一个样本中的特征之间存在相关性。大部分样本 embedding 是没有意义的，少部分是有意义的，如果使用 Batchnorm 可能反而会破坏这些有意义的 embedding。

![](./assets/batch-layer-norm.png)

### Batchnorm 有哪些参数

可学习参数：

- $\gamma$：缩放因子，控制标准化后的数据 scale，训练过程中每一个特征通道都有一个可学习的缩放因子。
- $\beta$：偏移因子，控制标准化后的数据 shift，训练过程中每一个特征通道都有一个可学习的偏移因子。

不可学习参数：全局均值和全局方差。

### 什么是BatchNorm？BatchNorm如何计算并起到了什么样的效果？

BatchNorm 可以让某一维度的数据在 Batch 内进行归一化，这种操作为数据引入了较为随机的扰动，可以起到正则化的效果。也有的解释说 BatchNorm 通过使损失的landscape更加平滑降低了优化的难度。

### BatchNorm 的公式及代码实现

$$
\mu_{\mathcal{B}} = \frac{1}{|\mathcal{B}|}\sum_{x \in \mathcal{B}} x , \sigma^2_{\mathcal{B}} = \frac{1}{|\mathcal{B}|}\sum_{x \ in \mathcal{B}}(x - \mu)^2 \\
\hat{x} = \frac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma^2_{\mathcal{B}} + \epsilon}} \\
y_i = \gamma \hat{x}_i + \beta
$$

```python
def Batchnorm_simple_for_train(x, gamma, beta, bn_param):
"""
param:x    : 输入数据，设shape(B,L)
param:gama : 缩放因子  γ
param:beta : 平移因子  β
param:bn_param   : batchnorm所需要的一些参数
    eps      : 接近0的数，防止分母出现0
    momentum : 动量参数，一般为0.9， 0.99， 0.999
    running_mean ：滑动平均的方式计算新的均值，训练时计算，为测试数据做准备
    running_var  : 滑动平均的方式计算新的方差，训练时计算，为测试数据做准备
"""
    running_mean = bn_param['running_mean']  #shape = [B]
    running_var = bn_param['running_var']    #shape = [B]
    results = 0. # 建立一个新的变量

    x_mean=x.mean(axis=0)  # 计算x的均值
    x_var=x.var(axis=0)    # 计算方差
    x_normalized=(x-x_mean)/np.sqrt(x_var+eps)       # 归一化
    results = gamma * x_normalized + beta            # 缩放平移

    running_mean = momentum * running_mean + (1 - momentum) * x_mean
    running_var = momentum * running_var + (1 - momentum) * x_var

    #记录新的值
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return results , bn_param
```

### 初始化是如何缓解梯度消失/梯度爆炸问题的呢？为什么有\(Xavier\)、\(Kaiming\)这样的初始化呢？

> Xavier: bias初始化为 0，为 Normalize 后的参数乘以一个 rescale 系数：$1/\sqrt n$, n 是输入参数的个数
>
> Kaiming: 因为 relu 会抛弃掉小于 0 的值，对于一个均值为 0 的 data 来说，这就相当于砍掉了一半的值。这样一来，均值就会变大，前面 Xavier 初始化公式中 E(x) = mean = 0 的情况就不成立了。根据新公式的推导，最终得到新的 rescale 系数：$1 / \sqrt{2/n}$。

我们通常希望网络的输出是一个均值为 $0$ 方差为 $1$ 的标准正态分布 —— 与输入保持一致。对于一个只有卷积或者全连接层构成的线性的神经网络来说，常规做法是使用 Xavier 来归一化权重使得其输出满足正态分布。然而，这个做法在引入了非线性变换的神经网络中效果有限，在 ReLU 激活层出现时更是如此，所以有了 Kaiming。

---

## 参考链接

- [ResNet面试简介](https://zyc.ai/sketch/career/interview_resnet/#_4)
- [基础 | batchnorm原理及代码详解](https://www.cnblogs.com/adong7639/p/9145911.html)
- [一文搞懂深度网络初始化（Xavier and Kaiming initialization）](https://cloud.tencent.com/developer/article/1587082)
