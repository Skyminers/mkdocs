
# Transformer 面试资料整理

---

## 简介

> [《Attention is All You Need》](https://arxiv.org/pdf/1706.03762)

![](./assets/transformer.png)

<!-- ### Vit 为什么加入 cls 分类 token？相比直接在最后用第一个有什么好处？

Token 可以捕获全局的信息，如果直接用第一个的话可能会跟第一个 Patch 有很强的关系，无法有效捕获其他 Patch 的信息。 -->

### 简单介绍一下模型

Transformer 由编码器和解码器组成，编码器内部由一个多头自注意力机制和一个前馈神经网络组成，解码器由一个 Masked 多头自注意力机制、一个多头注意力机制和一个前馈神经网络组成。在 Transformer 中利用了 Resnet 的残差连接，使模型更容易训练并且能够训练更深的网络，同时也缓解了梯度震荡、网络退化等问题。

Encoder 中，Self-Attention 的作用是获取上下文信息这一层的参数并不多，主要是融合上下文信息。接下来通过参数量较大的前馈神经网络储存知识。

Decoder 的自注意力机制部分可以用于信息融合，在多模块、机器翻译等场景有所应用。融合时 Encoder 向 decoder 的这一层提供 $K$ 和 $V$，decoder 的这一层提供 $Q$。

因为 Attention 的计算复杂度跟输入的 Token 相关是输入的平方级别（所以许多大模型都限制输入的最长 Token），所以在推理时可以通过 KV-cache 进行优化：每次 decoder 将之前的 $Q$ 和 $K$ 的点乘结果缓存起来，避免重复计算。用空间换时间。

### 介绍一下 FFN 计算公式

FFN由两个全连接层（即前馈神经网络）和一个激活函数组成。下面是FFN块的计算公式：

$$
\operatorname{FFN}(\boldsymbol{x})=\operatorname{Relu}\left(\boldsymbol{x} \boldsymbol{W}_{1}+\boldsymbol{b}_{1}\right) \boldsymbol{W}_{2}+\boldsymbol{b}_{2}
$$

假设输入是一个向量 $x$，FFN块的计算过程如下：

1. 第一层全连接层（线性变换）：$z = xW1 + b1$ 其中，W1 是第一层全连接层的权重矩阵，b1 是偏置向量。
2. 激活函数：$a = g(z)$ 其中，g() 是激活函数，常用的激活函数有ReLU（Rectified Linear Unit）等。
3. 第二层全连接层（线性变换）：$y = aW2 + b2$ 其中，W2 是第二层全连接层的权重矩阵，b2 是偏置向量。

增大前馈子层隐状态的维度有利于提升最终翻译结果的质量，因此，前馈子层隐状态的维度一般比自注意力子层要大。

需要注意的是，上述公式中的 W1、b1、W2、b2 是FFN块的可学习参数，它们会通过训练过程进行学习和更新。

---

## 问答

### 为什么 Nrom 选择 LayerNorm 而不是 BatchNorm？

1. BN 是在不同 Batch 之间计算的。在 NLP 任务中，不同序列的长度不一样，所以 BatchNorm 会遇到对齐的问题。LN 是在一个 Batch 内计算的，所以不会遇到这个问题。

2. BN 在 NLP 的任务中比较弱，因为 NLP 中同一个位置的词的作用可能完全不一样，BN 将这些作用不一样的词的信息混合在一起，这样会导致信息的丢失。

### 为什么要除以根号 dk

论文中的解释是：在计算 Attention 时，计算的是 $Q \cdot K^T$，如果不除以 $\sqrt {d_k}$，那么 $Q \cdot K^T$ 的值会很大，导致 Softmax 后的值很小，这样会导致梯度消失。通过这样的 Scale 可以缓解这个问题。

为什么是 $\sqrt {d_k}$ 呢？从公式来解释，假设向量 $q$ 和 $k$ 的各个分量是互相独立的随机变量，均值是 $0$，方差是 $1$，那么点积 $q\cdot k$ 的均值是 $0$，方差是 $d_k$。

已知 $E(q_i) = E(k_i) = 0, Q(q_i) = Q(k_i) = 1$ 推导过程如下:

$$
\begin{align}
E(q_ik_i) &= E(q_i)\cdot E(k_i) = 0 \\
Var(q_ik_i) &= Var(q_i) \cdot Var(k_i) + Var(q_i)\cdot E(k_i)^2 + Var(k_i)\cdot E(q_i)^2\\
            &= 1 \times 1 + 1 \times 0 + 1 \times 0 = 1 \\
Var(q\cdot k) &= \sum_{i=1}^{d_k} Var(q_ik_i) = d_k
\end{align}
$$

所以乘积后的数字方差为 $d_k$，需要除以 $\sqrt {d_k}$ 使得方差为 $1$。

### 为什么要使用多头注意力

多头注意力可以注意到不同子空间的信息，捕捉到更加丰富的特征信息。不同的头的关注点可能不相同，例如一个头关注到了语义信息，另一个头关注语法信息。有的实验表明，进行可视化后总有一两个头是独一无二的，与其他的头关注不一样。

多头注意力并不是必须的，去掉一些头也还是会有不错的效果，因为剩下的头已经有足够的对位置信息、语法信息或罕见词的关注能力了。

### 为什么的 Q 和 K 使用不同的权重矩阵生成，为何不能使用同一个值进行自身的点乘？

如果相同的话，我们会得到一个泛化能力较差的对称矩阵。不相同的权重矩阵可以保证 Q/K/V 在不同空间进行投影，这增强了表达能力，提高了泛化能力。

### 为什么多头自注意力机制中的 QKV 要用三个不同的矩阵？

1. 不同的矩阵可以引入更多的参数量，增大语义空间，提升表达能力。。
2. 在进行 self-attntion 时，我们更加希望 $Q\neq K \neq V$，引入不同的矩阵可以更好地做到这一点。如果 $Q=K=V$，那么在进行点乘时，每一个 Query 对自己的 Key 都会有很大的值，这会影响每一个 Query 对其他的 Key 的注意力权重，导致 Transformer 的上下文信息的提取能力大幅度下降。

### 计算 attention 为何选择点乘而不是加法？两者计算复杂度和效果上有什么区别？

通过点乘的方式计算会有更高的效率。论文中有实验，在 dk 比较大的时候加法有更好的效果。

### Positional Encoding 为什么选用 sin 和 cos 函数？

$$
\left\{
             \begin{array}{lr}
             PE(pos, 2i) &= sin\left( pos / 10000 ^ {2i / d_{model}}\right) &  \\
             PE(pos, 2i+1) &= cos\left( pos / 10000 ^ {2i / d_{model}}\right) &
             \end{array}
\right.
$$

```python
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)
```

pos 即 position，意味 token 在句中的位置，设句子长度为 $L$，则 $pos = 0,1,\ldots, L-1$; $i$ 为向量的某一个维度，例如 $d_{model} = 512, i = 0,1,\ldots, 255$。这种编码方式对不同维度使用了不同频率的正/余弦公式生成了不同位置的高纬位置编码，利用正余弦函数实现相对位置信息的表示。

通过三角函数的性质，我们可以得到：

$$
\left\{
             \begin{array}{lr}
             PE(pos+k, 2i) &= PE(pos, 2i)\times PE(k, 2i+1) + PE(pos, 2i+1)\times PE(k, 2i) &  \\
             PE(pos+k, 2i+1) &= PE(pos, 2i+1)\times PE(k, 2i+1) - PE(pos, 2i)\times PE(k,2i) &
             \end{array}
\right.
$$

也就是说，对于 $pos+k$ 位置的位置向量的某一维度而言，其值可以被表示为 $pos$ 和 $k$ 位置的位置向量的线性组合。这意味着位置向量之中包含了相对位置信息。但是我有看到有人提出：在经过线性变换层之后，相对位置的信息被破坏。Transformer 被提出时作者可能就发现了这个问题，后续他们又提出了一个新的方法来解决这个问题，加入了一个可训练的相对位置参数。

### PreNorm 和 PostNorm 的区别

![](./assets/norm.jpeg)

PreNorm 是 LayerNorm 放在了 Attention/FFN 运算之前，PostNorm 是 LayerNorm 放在了 Attention/FFN 之后。

- Post-norm 在残差之后做归一化，对参数的正则化效果更好。
- Pre-norm 相对 Post-norm ，因为有一部分参数未经过 Norm 加在了后面，这防止了模型出现梯度消失的问题。
- 如果层数比较少的话 PostNorm 的效果会更好，如果层数比较高的话就需要通过 PreNorm 来防止梯度爆炸或梯度消失。

PostNorm 梯度：$\frac{\partial \varepsilon}{\partial x_l} = \frac{\partial \varepsilon}{\partial x_{l+1}}\times \prod_{k=l}^{L-1}\frac{\partial LN(y_k)}{\partial y_k}\times \prod_{k=1}^{L-1}\left(1 + \frac{\partial F(x_k; \theta_k)}{\partial x_k}\right)$

PreNorm 梯度：$\frac{\partial \varepsilon }{\partial x_l} = \frac{\partial \varepsilon}{\partial x_{l+1}}\times \prod_{k=1}^{L-1}\left(1 + \frac{\partial x_{l+1}}{\partial LN(x_l)}\right)$

所以 PreNorm 的梯度更加稳定，因为 $\prod$ 中的数字均大于 $1$，所以不会出现梯度消失的问题。

### 为何在获取输入词向量之后需要对矩阵乘以embedding size的开方？意义是什么？

Embedding matrix 的初始化方式是 xavier 初始化，这种方式的方差是 $\frac{1}{\text{Embedding size}}$，因此乘以 Embedding size 的开方使得 Embedding matrix 的方差是 $1$，在这个 scale 下可能更有利于 Embedding matrix 的收敛。

---

## 代码

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
seaborn.set_context(context='talk')
%matplotlib inline

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        """
        编码器、解码器、输入嵌入层、目标嵌入层、输出层
        """
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        src --> memory
        memory + tgt --> output
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        """
        src --> memory
        """
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        """
        memory + tgt --> output
        """
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        """
        需要自主生成 mask
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    """
    inputs: batch, seq_len, features
    沿输入数据的特征维度归一化
    """
    def __init__(self, features, eps=1e-6):
        # 需要指定特征数量 features
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.ones(features))
        self.eps = eps

    def forward(self, x):
        """
        x --> (x - x.mean) / x.std
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        指定内部的结构 sublayer，是 attention 层，还是 feed_forward 层
        """
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """size: d_model"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size  # 作为参数用于 layernorm 层
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clone(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

# 解码器一次输入序列中向量，当前步后面的序列需要被遮盖
# 需要被遮盖的单词被标记为 False

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


plt.figure(figsize=(5, 5))
plt.imshow(subsequent_mask(20)[0])

def attention(query, key, value, mask=None, dropout=None):
    """
    query : batch, target_len, feats
    key   : batch, seq_len,    feats
    value : batch, seq_len,    val_feats

    return: batch, target_len, val_feats
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """
        h, num_heads
        d_model, features
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clone(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query,key,value: batch,seq_len,d_model

        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(
            query,  # batch,num_head,seq_len,feats
            key,
            value,
            mask=mask,
            dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1,
                                                self.h * self.d_k)
        # batch,seq_len,num_head*feats
        return self.linears[-1](x)

def test_multi_head():
    x = torch.randn(2, 4, 12)
    d_model = x.shape[-1]
    model = MultiHeadedAttention(2, d_model)
    attn = model(x, x, x)
    assert attn.shape == (2, 4, 12)
    print("Test passed!")

test_multi_head()


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

def make_model(src_vocab,
               tgt_vocab,
               N=6,
               d_model=512,
               d_ff=2048,
               h=8,
               dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class Batch:
    def __init__(self, src, trg=None, pad=0):
        """
        src: 输入序列
        trg: 目标序列
        """
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        """
        将 pad 产生的 mask，和序列一次预测下一个单词产生的 mask 结合起来
        """
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

src = torch.tensor([[3, 5, 7, 0, 0], [2, 4, 6, 8, 0]])  # batch=2,seq_len=5
trg = torch.tensor([[2, 3, 4, 5, 0, 0], [3, 5, 6, 0, 0,
                                         0]])  # batch=2,seq_len=6

sample = Batch(src, trg)
sample.src_mask

def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask,
                            batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens  # 总 tokens 数
        tokens += batch.ntokens  # 50 批训练时的总 tokens 数
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens
```

## 参考链接

- [Transformer、Like-Bert、对比学习、ChatGPT相关面试集锦](https://zhuanlan.zhihu.com/p/149634836)
- [Transformer相关——（8）Transformer模型](https://ifwind.github.io/2021/08/18/Transformer%E7%9B%B8%E5%85%B3%E2%80%94%E2%80%94%EF%BC%888%EF%BC%89Transformer%E6%A8%A1%E5%9E%8B/#decoder%E5%B1%82)
- [llm_interview_note](https://github.com/wdndev/llm_interview_note/tree/main)
- [24年大模型面试必看，基础知识Transformer面试题-北大博士后卢菁博士授课](https://www.bilibili.com/video/BV13x421S7bi/?share_source=copy_web&vd_source=014f498a4d1a2f08f4e4de7b447bdb63)
- [30-Transformer模型及源代码(PyTorch)](<https://github.com/YangBin1729/nlp_notes/blob/master/06-%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86/30-Transformer%E6%A8%A1%E5%9E%8B%E5%8F%8A%E6%BA%90%E4%BB%A3%E7%A0%81(PyTorch).ipynb>)
