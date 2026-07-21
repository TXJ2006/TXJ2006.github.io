---
title: '从零读懂 Transformer：自注意力、位置编码与现代序列建模'
subtitle: '从 RNN 的结构瓶颈，到 QKV、缩放点积、多头注意力、编码器与解码器的逐步推导'
summary: '本文从序列建模的基本困难出发，以通俗语言解释 Transformer 为什么要抛弃递归，并从加权平均开始逐步推导 Query、Key、Value、缩放点积注意力、多头注意力、位置编码、因果掩码、残差连接、归一化与前馈网络。随后给出自注意力的置换等变性、核回归视角、完整反向传播、复杂度与 KV Cache 分析，并系统比较 BERT、GPT 与编码器—解码器范式及其跨模态推广。'
description: '本文从序列建模的基本困难出发，以通俗语言解释 Transformer 为什么要抛弃递归，并从加权平均开始逐步推导 Query、Key、Value、缩放点积注意力、多头注意力、位置编码、因果掩码、残差连接、归一化与前馈网络。随后给出自注意力的置换等变性、核回归视角、完整反向传播、复杂度与 KV Cache 分析，并系统比较 BERT、GPT 与编码器—解码器范式及其跨模态推广。'
date: 2026-07-21
lastmod: 2026-07-21
weight: 55
tags: ["Transformer", "Self-Attention", "Deep Learning", "Sequence Modeling", "Large Language Models", "Representation Learning"]
draft: false
ShowToc: false
hideMeta: true
libraryFolder: "ai-foundations"
libraryFolderName: "人工智能基础"
libraryFolderColor: 0
---

## 引言

递归神经网络处理序列的方式非常符合人的直觉：从左到右阅读，每读到一个新元素，就把它与过去的记忆合并。问题是，这种直觉同时带来了两种结构性代价。

第一，信息必须沿着时间一步一步传递。序列中第一个 token 若要影响最后一个 token，至少要经过 $n-1$ 次状态更新。只要每一步都损失一点信息，长距离信号就会逐渐衰减。

第二，计算也必须沿着时间一步一步进行。第 $t$ 个隐藏状态依赖第 $t-1$ 个隐藏状态，所以即使拥有大量 GPU，也不能一次性算出所有时间步。

Transformer 改变了问题的问法。它不再要求信息沿时间链传递，而是让序列中的每个位置直接检查所有位置，并根据当前输入动态决定“应该从谁那里取多少信息”。这一操作就是自注意力。

最简洁的表达是

<div class="display-equation">
$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$
</div>

这条公式只有一行，但它压缩了至少六个问题：

第一，为什么要把同一个输入映射成 $Q,K,V$ 三种表示？

第二，为什么相似度使用内积？

第三，为什么要做 softmax？

第四，为什么必须除以 $\sqrt{d_k}$？

第五，为什么一层注意力还不够，需要多头、残差、归一化和前馈网络？

第六，没有递归以后，模型如何知道 token 的先后顺序？

本文不把 Transformer 当成一张需要背诵的结构图，而是从一个最基本的“动态加权平均”问题开始，一步一步推导出完整架构。每个对象先解释直觉，再给出定义、维度、公式、证明和数值例子。

## 零、先不看公式：Transformer 到底改变了什么

考虑一句话：

> 小明没有把书交给小王，因为**他**还没有看完。

理解“他”指谁，不能只看“他”附近的字。模型必须从句子中寻找与当前词最相关的对象。

传统全连接层会使用一组训练后固定的权重。无论输入句子是什么，第 $j$ 个位置对第 $i$ 个位置的影响规则都不随内容改变。

卷积层允许局部位置共享参数，但它的窗口通常固定。若两个相关词相距很远，信息必须经过多层卷积才能相遇。

RNN 会把前文压入隐藏状态，但所有历史都必须经过同一个有限维“瓶颈”，并沿时间逐步传递。

自注意力采取另一种方式。对当前位置“他”而言，模型先生成一个查询，表示“我现在需要寻找什么信息”；每个候选 token 生成一个键，表示“我具有什么可被匹配的属性”；同时生成一个值，表示“如果我被选中，我实际要传递什么内容”。

于是，一次注意力可以被通俗地拆成三句话：

1. 用 Query 与所有 Key 比较，得到相关性分数；
2. 把分数归一化为总和等于一的权重；
3. 用这些权重对 Value 做加权平均。

因此，Transformer 最核心的变化不是“去掉了循环”这么简单，而是把固定的信息传递图改成了由当前数据动态生成的信息路由图。

<div class="display-equation">
$$
\boxed{
\text{固定聚合规则}
\quad\longrightarrow\quad
\text{由输入内容决定的动态聚合规则}
}
$$
</div>

### 四种层的根本区别

| 结构 | 信息从哪里来 | 聚合权重是否依赖当前输入 | 长距离路径 | 时间并行性 |
|---|---|---:|---:|---:|
| 全连接层 | 所有输入坐标 | 否 | 一步 | 高 |
| 卷积层 | 固定局部窗口 | 否 | 需要多层 | 高 |
| RNN | 当前输入与上一步状态 | 间接依赖 | $O(n)$ 步 | 低 |
| 自注意力 | 所有 token | 是 | 一步 | 训练时高 |

这里需要强调一个容易被忽略的事实：Transformer 并不是在所有方面都优于 RNN。它用全局直接交互换来了 $O(n^2)$ 的注意力矩阵；而自回归生成在推理时仍然必须逐 token 进行。它消除的是网络内部沿序列的递归依赖，而不是生成任务本身的因果顺序。

## 一、把序列写成矩阵

设输入序列包含 $n$ 个 token，每个 token 被表示为一个 $d_{\mathrm{model}}$ 维向量。把这些向量按行排列，得到

<div class="display-equation">
$$
X=
\begin{bmatrix}
x_1^\top\\
x_2^\top\\
\vdots\\
x_n^\top
\end{bmatrix}
\in\mathbb R^{n\times d_{\mathrm{model}}}.
$$
</div>

这里第 $i$ 行 $x_i^\top$ 是第 $i$ 个 token 的表示。

这种记法同时包含两个维度：

- 行维度表示序列位置；
- 列维度表示每个位置的特征。

后续所有公式都应先检查形状。若形状不一致，即使直觉正确，矩阵公式也不可能成立。

### 定义 1（逐位置线性映射）

给定权重矩阵

<div class="display-equation">
$$
W\in\mathbb R^{d_{\mathrm{model}}\times d'},
$$
</div>

矩阵乘法

<div class="display-equation">
$$
XW\in\mathbb R^{n\times d'}
$$
</div>

表示对每个 token 使用同一个线性变换：

<div class="display-equation">
$$
(XW)_{i,:}=x_i^\top W.
$$
</div>

参数在位置之间共享，所以序列长度改变时，权重矩阵本身不需要改变。这一点是 Transformer 能处理可变长度序列的基础。

## 二、从加权平均推导注意力

先不讨论 Query、Key 和 Value。假设第 $i$ 个位置希望从所有位置收集信息，最自然的形式是

<div class="display-equation">
$$
y_i=\sum_{j=1}^n\alpha_{ij}v_j,
$$
</div>

其中 $v_j\in\mathbb R^{d_v}$ 是第 $j$ 个位置可提供的内容，$\alpha_{ij}$ 是位置 $i$ 对位置 $j$ 的权重。

为了让 $y_i$ 是稳定的加权平均，通常要求

<div class="display-equation">
$$
\alpha_{ij}\geq0,
\qquad
\sum_{j=1}^n\alpha_{ij}=1.
$$
</div>

于是问题只剩下：怎样根据输入自动生成 $\alpha_{ij}$？

### 第一步：定义匹配分数

令位置 $i$ 产生查询向量 $q_i\in\mathbb R^{d_k}$，位置 $j$ 产生键向量 $k_j\in\mathbb R^{d_k}$。用内积定义原始匹配分数：

<div class="display-equation">
$$
s_{ij}=q_i^\top k_j.
$$
</div>

若两个向量方向接近，内积通常较大；若方向相反，内积较小。它是可微、计算高效并适合大规模矩阵乘法的相似度函数。

### 第二步：把任意分数变成概率权重

对固定查询 $i$，使用按行 softmax：

<div class="display-equation">
$$
\alpha_{ij}
=
\frac{\exp(s_{ij})}
{\sum_{\ell=1}^n\exp(s_{i\ell})}.
$$
</div>

显然有 $\alpha_{ij}>0$ 且 $\sum_j\alpha_{ij}=1$。

### 第三步：聚合内容

最终输出为

<div class="display-equation">
$$
y_i
=
\sum_{j=1}^n
\frac{\exp(s_{ij})}
{\sum_{\ell=1}^n\exp(s_{i\ell})}
\,v_j.
$$
</div>

这已经是完整的单位置注意力。将所有位置一起写成矩阵，令

<div class="display-equation">
$$
Q=
\begin{bmatrix}q_1^\top\\ \vdots\\q_n^\top\end{bmatrix},
\quad
K=
\begin{bmatrix}k_1^\top\\ \vdots\\k_n^\top\end{bmatrix},
\quad
V=
\begin{bmatrix}v_1^\top\\ \vdots\\v_n^\top\end{bmatrix},
$$
</div>

则

<div class="display-equation">
$$
S=QK^\top\in\mathbb R^{n\times n},
$$
</div>

第 $(i,j)$ 个元素正是 $q_i^\top k_j$。令 softmax 对每一行独立执行，得到

<div class="display-equation">
$$
A=\operatorname{softmax}_{\mathrm{row}}(S),
\qquad
Y=AV.
$$
</div>

其中

<div class="display-equation">
$$
A\in\mathbb R^{n\times n},
\qquad
Y\in\mathbb R^{n\times d_v}.
$$
</div>

矩阵 $A$ 就是注意力矩阵。第 $i$ 行描述位置 $i$ 从所有位置读取信息的比例。

## 三、为什么必须分成 Query、Key 和 Value

在自注意力中，$Q,K,V$ 通常都来自同一个输入矩阵 $X$：

<div class="display-equation">
$$
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V,
$$
</div>

其中

<div class="display-equation">
$$
W_Q,W_K\in\mathbb R^{d_{\mathrm{model}}\times d_k},
\qquad
W_V\in\mathbb R^{d_{\mathrm{model}}\times d_v}.
$$
</div>

初学者常问：既然三者都来自 $X$，为什么不直接使用 $XX^\top$，再对 $X$ 加权？

答案有三层。

### 第一层：匹配空间与内容空间不应被强制相同

Key 与 Query 负责回答“谁应该与谁连接”，Value 负责回答“连接建立后传递什么”。

一个词可能因为语法角色而与另一个词匹配，但传递的内容可能是语义、实体属性或位置信息。若直接令 $Q=K=V=X$，模型就被迫使用同一组特征同时完成路由和内容表达。

独立投影允许

<div class="display-equation">
$$
\text{路由空间}
eq\text{内容空间}.
$$
</div>

### 第二层：Query 与 Key 的角色本身不对称

$q_i$ 表示位置 $i$ 正在寻找什么，$k_j$ 表示位置 $j$ 可以被怎样检索。这类似数据库中的查询与索引。即使两个 token 内容相同，它们在不同上下文中也可能承担不同的“提问者”与“被检索者”角色。

若 $W_Q\neq W_K$，则匹配函数为

<div class="display-equation">
$$
s_{ij}
=x_i^\top W_QW_K^\top x_j,
$$
</div>

这是一种可学习的双线性相似度，而不再局限于原空间中的普通内积。

### 第三层：独立投影提高表达能力

直接使用 $XX^\top$ 只允许模型按原始表示的几何结构匹配。使用 $W_Q,W_K$ 后，模型可以把输入投影到更适合当前任务的子空间。

例如，同一个输入表示中可能同时含有词性、语义、位置和实体信息。一个注意力头可以把 Query 与 Key 投影到“句法关系”子空间，另一个头投影到“指代关系”子空间。

### 结论

<div class="display-equation">
$$
\boxed{
Q,K\text{ 决定信息流向，}
\qquad
V\text{ 决定实际传递的信息。}
}
$$
</div>

这个区分是理解注意力的最重要直觉之一。注意力权重本身并不是内容，它只是路由系数。

## 四、缩放点积中的 $\sqrt{d_k}$ 从哪里来

标准注意力不是

<div class="display-equation">
$$
\operatorname{softmax}(QK^\top)V,
$$
</div>

而是

<div class="display-equation">
$$
\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V.
$$
</div>

这个缩放不是经验装饰，而是方差归一化。

### 引理 1（随机内积的方差）

设

<div class="display-equation">
$$
q=(q_1,\ldots,q_{d_k}),
\qquad
k=(k_1,\ldots,k_{d_k}),
$$
</div>

各坐标相互独立，且

<div class="display-equation">
$$
\mathbb E[q_r]=\mathbb E[k_r]=0,
\qquad
\operatorname{Var}(q_r)=\operatorname{Var}(k_r)=1.
$$
</div>

则

<div class="display-equation">
$$
\mathbb E[q^\top k]=0,
\qquad
\operatorname{Var}(q^\top k)=d_k.
$$
</div>

**证明.** 由

<div class="display-equation">
$$
q^\top k=\sum_{r=1}^{d_k}q_rk_r,
$$
</div>

以及零均值与独立性，

<div class="display-equation">
$$
\mathbb E[q_rk_r]
=
\mathbb E[q_r]\mathbb E[k_r]=0.
$$
</div>

故总和的期望为零。又因为不同坐标乘积相互独立，

<div class="display-equation">
$$
\operatorname{Var}(q^\top k)
=
\sum_{r=1}^{d_k}\operatorname{Var}(q_rk_r).
$$
</div>

对每个 $r$，

<div class="display-equation">
$$
\operatorname{Var}(q_rk_r)
=
\mathbb E[q_r^2k_r^2]
=
\mathbb E[q_r^2]\mathbb E[k_r^2]
=1.
$$
</div>

因此方差等于 $d_k$。$\square$

### 推论 1（缩放后的方差）

<div class="display-equation">
$$
\operatorname{Var}\!\left(\frac{q^\top k}{\sqrt{d_k}}\right)=1.
$$
</div>

**证明.** 方差满足 $\operatorname{Var}(cX)=c^2\operatorname{Var}(X)$，所以

<div class="display-equation">
$$
\operatorname{Var}\!\left(\frac{q^\top k}{\sqrt{d_k}}\right)
=
\frac{1}{d_k}\operatorname{Var}(q^\top k)
=1.
$$
</div>

$\square$

### 为什么大方差会伤害 softmax

softmax 的第 $i$ 个分量为

<div class="display-equation">
$$
p_i=\frac{e^{z_i}}{\sum_j e^{z_j}}.
$$
</div>

其 Jacobian 为

<div class="display-equation">
$$
\frac{\partial p_i}{\partial z_j}
=p_i(\mathbf 1\{i=j\}-p_j).
$$
</div>

若某个 logit 远大于其余值，则对应 $p_i\approx1$，其余 $p_j\approx0$。此时所有 $p_i(1-p_i)$ 与 $p_ip_j$ 都接近零，softmax 进入饱和区，梯度难以流向非最大位置。

除以 $\sqrt{d_k}$ 的作用不是保证注意力永远不尖锐，而是在随机初始化和训练早期让 logit 的典型尺度不随维度增长，从而避免仅仅因为 $d_k$ 较大就让 softmax 过早饱和。

### 例 1（缩放前后的注意力）

假设某个 Query 与四个 Key 的未缩放内积为

<div class="display-equation">
$$
(20,5,-3,8).
$$
</div>

未缩放 softmax 几乎把全部质量放到第一个位置。若 $d_k=64$，除以 $8$ 后得到

<div class="display-equation">
$$
(2.5,0.625,-0.375,1.0),
$$
</div>

其权重约为

<div class="display-equation">
$$
(0.698,0.107,0.039,0.156).
$$
</div>

模型仍然偏向第一个位置，但其他位置不再完全失去梯度。

## 五、单头自注意力的完整公式与维度检查

定义

<div class="display-equation">
$$
Q=XW_Q\in\mathbb R^{n\times d_k},
$$
</div>

<div class="display-equation">
$$
K=XW_K\in\mathbb R^{n\times d_k},
$$
</div>

<div class="display-equation">
$$
V=XW_V\in\mathbb R^{n\times d_v}.
$$
</div>

然后

<div class="display-equation">
$$
S=\frac{QK^\top}{\sqrt{d_k}}
\in\mathbb R^{n\times n},
$$
</div>

<div class="display-equation">
$$
A=\operatorname{softmax}_{\mathrm{row}}(S)
\in\mathbb R^{n\times n},
$$
</div>

<div class="display-equation">
$$
Y=AV\in\mathbb R^{n\times d_v}.
$$
</div>

逐元素写为

<div class="display-equation">
$$
Y_{i,:}
=
\sum_{j=1}^n
\frac{
\exp(q_i^\top k_j/\sqrt{d_k})
}{
\sum_{\ell=1}^n
\exp(q_i^\top k_\ell/\sqrt{d_k})
}
v_j^\top.
$$
</div>

这条式子说明：每个输出位置仍然对应一个 token，但其表示已经融合了整个序列的信息。

### 例 2（三个二维 token 的完整计算）

令

<div class="display-equation">
$$
X=
\begin{bmatrix}
1&0\\
0&1\\
1&1
\end{bmatrix},
\qquad
W_Q=W_K=W_V=I_2.
$$
</div>

于是 $Q=K=V=X$，且 $d_k=2$。首先

<div class="display-equation">
$$
QK^\top
=
\begin{bmatrix}
1&0&1\\
0&1&1\\
1&1&2
\end{bmatrix}.
$$
</div>

缩放后

<div class="display-equation">
$$
S
=
\frac{1}{\sqrt2}
\begin{bmatrix}
1&0&1\\
0&1&1\\
1&1&2
\end{bmatrix}
\approx
\begin{bmatrix}
0.707&0&0.707\\
0&0.707&0.707\\
0.707&0.707&1.414
\end{bmatrix}.
$$
</div>

逐行 softmax 得

<div class="display-equation">
$$
A\approx
\begin{bmatrix}
0.401&0.198&0.401\\
0.198&0.401&0.401\\
0.248&0.248&0.503
\end{bmatrix}.
$$
</div>

最后

<div class="display-equation">
$$
Y=AX
\approx
\begin{bmatrix}
0.802&0.599\\
0.599&0.802\\
0.752&0.752
\end{bmatrix}.
$$
</div>

以第一行为例，

<div class="display-equation">
$$
y_1
=0.401(1,0)+0.198(0,1)+0.401(1,1)
=(0.802,0.599).
$$
</div>

第一个 token 最关注自身与第三个 token，因为二者与查询 $(1,0)$ 的内积相同；对第二个 token 的关注较低。

## 六、注意力是一种可学习的核回归

对固定查询 $q_i$，定义核函数

<div class="display-equation">
$$
k(q_i,k_j)
:=
\exp\!\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right).
$$
</div>

则注意力输出为

<div class="display-equation">
$$
y_i
=
\frac{\sum_{j=1}^n k(q_i,k_j)v_j}
{\sum_{\ell=1}^n k(q_i,k_\ell)}.
$$
</div>

这与 Nadaraya–Watson 核回归完全同形：根据查询与样本的相似度，对响应值做归一化加权平均。

不同之处在于，传统核回归通常预先选择核函数；自注意力中的“核”由 $W_Q,W_K$ 学习。模型不是只学习 Value，而是在学习“什么叫相似”。

### 这一视角解释了什么

第一，注意力天然是一种平滑器。若多个 Key 与 Query 相近，它会混合对应的 Value。

第二，softmax 温度控制平滑程度。若把分母写成温度 $\tau$：

<div class="display-equation">
$$
A(\tau)=\operatorname{softmax}\!\left(\frac{QK^\top}{\tau}\right),
$$
</div>

则 $\tau$ 小时分布尖锐，接近最近邻检索；$\tau$ 大时分布平坦，接近全局平均。

第三，线性注意力与随机特征方法可以理解为对指数核进行低秩近似，从而避免显式构造 $n\times n$ 矩阵。

不过，注意力不等于普通的对称核。若 $W_Q\neq W_K$，则一般有

<div class="display-equation">
$$
q_i^\top k_j\neq q_j^\top k_i,
$$
</div>

并且按行 softmax 进一步引入方向性。因此位置 $i$ 关注 $j$ 的程度，通常不同于位置 $j$ 关注 $i$ 的程度。

## 七、没有位置编码时，自注意力不知道顺序

自注意力只依赖 token 之间的成对匹配。若把输入行重新排列，输出也只会以相同方式重新排列。

### 定义 2（置换矩阵）

置换矩阵 $P\in\mathbb R^{n\times n}$ 的每行每列恰有一个元素为 $1$，其余为 $0$。左乘 $PX$ 表示重排 $X$ 的行，也就是重排 token 顺序。

### 定理 1（无位置自注意力的置换等变性）

令

<div class="display-equation">
$$
\operatorname{SA}(X)
=
\operatorname{softmax}_{\mathrm{row}}\!\left(
\frac{XW_QW_K^\top X^\top}{\sqrt{d_k}}
\right)XW_V.
$$
</div>

对任意置换矩阵 $P$，有

<div class="display-equation">
$$
\operatorname{SA}(PX)=P\operatorname{SA}(X).
$$
</div>

**证明.** 令

<div class="display-equation">
$$
Q=XW_Q,
\qquad
K=XW_K,
\qquad
V=XW_V.
$$
</div>

输入变成 $PX$ 后，

<div class="display-equation">
$$
Q'=PQ,
\qquad
K'=PK,
\qquad
V'=PV.
$$
</div>

故分数矩阵为

<div class="display-equation">
$$
S'
=
\frac{Q'K'^\top}{\sqrt{d_k}}
=
\frac{PQK^\top P^\top}{\sqrt{d_k}}
=PSP^\top.
$$
</div>

按行 softmax 与同时重排行、列相容：

<div class="display-equation">
$$
\operatorname{softmax}_{\mathrm{row}}(PSP^\top)
=P\operatorname{softmax}_{\mathrm{row}}(S)P^\top.
$$
</div>

因此

<div class="display-equation">
$$
\begin{aligned}
\operatorname{SA}(PX)
&=
PAP^\top PV\\
&=PAV\\
&=P\operatorname{SA}(X).
\end{aligned}
$$
</div>

$\square$

“置换等变”不等于“输出完全不变”。输入 token 被重排后，输出 token 也按相同方式重排。但模型无法仅凭自注意力区分“我打你”与“你打我”中的顺序结构，因为它没有任何额外坐标系。

因此必须注入位置信息。

## 八、位置编码：为集合函数加入顺序

输入通常写为

<div class="display-equation">
$$
X^{(0)}=E+P,
$$
</div>

其中 $E$ 是 token embedding，$P$ 是位置表示。

### 8.1 正弦—余弦绝对位置编码

原始 Transformer 对位置 $p$ 与频率索引 $r$ 定义

<div class="display-equation">
$$
\operatorname{PE}(p,2r)
=
\sin\!\left(
\frac{p}{10000^{2r/d_{\mathrm{model}}}}
\right),
$$
</div>

<div class="display-equation">
$$
\operatorname{PE}(p,2r+1)
=
\cos\!\left(
\frac{p}{10000^{2r/d_{\mathrm{model}}}}
\right).
$$
</div>

不同维度使用不同频率：高频维度敏感于局部位移，低频维度可表达更长尺度的位置变化。

### 引理 2（正弦位置编码的相对位移可线性表示）

固定角频率 $\omega$，定义二维位置向量

<div class="display-equation">
$$
p_t=
\begin{bmatrix}
\sin(t\omega)\\
\cos(t\omega)
\end{bmatrix}.
$$
</div>

对任意偏移 $\Delta$，存在只依赖 $\Delta$ 的矩阵 $R_\Delta$，使

<div class="display-equation">
$$
p_{t+\Delta}=R_\Delta p_t.
$$
</div>

**证明.** 由和角公式，

<div class="display-equation">
$$
\sin((t+\Delta)\omega)
=
\cos(\Delta\omega)\sin(t\omega)
+
\sin(\Delta\omega)\cos(t\omega),
$$
</div>

<div class="display-equation">
$$
\cos((t+\Delta)\omega)
=
-\sin(\Delta\omega)\sin(t\omega)
+
\cos(\Delta\omega)\cos(t\omega).
$$
</div>

所以

<div class="display-equation">
$$
R_\Delta
=
\begin{bmatrix}
\cos(\Delta\omega)&\sin(\Delta\omega)\\
-\sin(\Delta\omega)&\cos(\Delta\omega)
\end{bmatrix}.
$$
</div>

$\square$

这说明固定相对位移对应固定旋转。模型可以通过线性运算学习相对位置关系。

### 8.2 可学习绝对位置编码

定义参数矩阵

<div class="display-equation">
$$
P\in\mathbb R^{n_{\max}\times d_{\mathrm{model}}},
$$
</div>

第 $p$ 行直接作为位置 $p$ 的向量。优点是灵活；缺点是训练长度以外的位置没有被可靠学习，外推能力通常较弱。

### 8.3 相对位置偏置

不把位置向量加到输入，而是直接修改注意力分数：

<div class="display-equation">
$$
s_{ij}
=
\frac{q_i^\top k_j}{\sqrt{d_k}}
+b_{i-j}.
$$
</div>

这里 $b_{i-j}$ 只依赖相对距离。模型由此直接学习“相隔一个位置”“相隔十个位置”对注意力的影响。

### 8.4 RoPE 的核心公式

旋转位置编码对第 $i$ 个位置的 Query 和 Key 施加位置相关旋转：

<div class="display-equation">
$$
\widetilde q_i=R_iq_i,
\qquad
\widetilde k_j=R_jk_j.
$$
</div>

则内积为

<div class="display-equation">
$$
\widetilde q_i^\top\widetilde k_j
=q_i^\top R_i^\top R_jk_j
=q_i^\top R_{j-i}k_j.
$$
</div>

因此位置只通过相对差 $j-i$ 进入匹配分数。这也是 RoPE 适合自回归长上下文建模的关键结构。

### 8.5 ALiBi 的简化思想

ALiBi 直接对远距离位置施加线性惩罚：

<div class="display-equation">
$$
s_{ij}
=
\frac{q_i^\top k_j}{\sqrt{d_k}}
-m_h|i-j|,
$$
</div>

其中不同注意力头可以使用不同斜率 $m_h$。它不需要显式位置 embedding，而是让距离成为 logit 的先验偏置。

## 九、因果掩码：让模型不能偷看未来

编码任务可以让每个位置观察整个序列，但自回归生成要求第 $t$ 个位置只能依赖 $1,\ldots,t$。

定义掩码矩阵

<div class="display-equation">
$$
M_{ij}
=
\begin{cases}
0,&j\leq i,\\
-\infty,&j>i.
\end{cases}
$$
</div>

因果注意力为

<div class="display-equation">
$$
A
=
\operatorname{softmax}_{\mathrm{row}}\!\left(
\frac{QK^\top}{\sqrt{d_k}}+M
\right).
$$
</div>

因为 $e^{-\infty}=0$，所以当 $j>i$ 时，

<div class="display-equation">
$$
A_{ij}=0.
$$
</div>

### 引理 3（因果性）

带上述掩码的第 $i$ 个输出只依赖 $v_1,\ldots,v_i$。

**证明.** 输出为

<div class="display-equation">
$$
y_i=\sum_{j=1}^nA_{ij}v_j.
$$
</div>

对所有 $j>i$，$A_{ij}=0$，故

<div class="display-equation">
$$
y_i=\sum_{j=1}^iA_{ij}v_j.
$$
</div>

$\square$

### 训练为什么仍然可以并行

虽然第 $i$ 个位置不能看未来，但训练时整段真实序列已知。我们可以一次性构造所有 Query、Key、Value，并用同一个下三角掩码并行计算所有位置的损失。

因此：

<div class="display-equation">
$$
\boxed{
\text{因果掩码限制信息依赖，}
\quad
\text{但不强迫训练计算按时间串行。}
}
$$
</div>

推理不同。生成第 $t+1$ 个 token 之前，必须先知道第 $t$ 个 token，所以自回归推理仍然是串行的。

## 十、多头注意力：在多个子空间中同时检索

单头注意力只有一套 $W_Q,W_K,W_V$。它必须用同一个匹配几何同时处理局部、句法、语义、实体与位置关系。

多头注意力为每个头分配独立投影：

<div class="display-equation">
$$
\operatorname{head}_r
=
\operatorname{Attention}
\left(
XW_Q^{(r)},
XW_K^{(r)},
XW_V^{(r)}
\right),
\qquad r=1,\ldots,h.
$$
</div>

将各头拼接并再次线性变换：

<div class="display-equation">
$$
\operatorname{MHA}(X)
=
\operatorname{Concat}
(\operatorname{head}_1,\ldots,\operatorname{head}_h)W_O.
$$
</div>

通常取

<div class="display-equation">
$$
d_k=d_v=d_h:=\frac{d_{\mathrm{model}}}{h}.
$$
</div>

每个头在较低维子空间中工作，拼接后恢复到 $d_{\mathrm{model}}$ 维。

### 引理 4（标准多头注意力的参数量不随头数线性增加）

若 $hd_h=d_{\mathrm{model}}$，忽略偏置，则 Q、K、V 投影与输出投影的总参数量为

<div class="display-equation">
$$
4d_{\mathrm{model}}^2,
$$
</div>

与头数 $h$ 无关。

**证明.** 所有头的 Query 投影参数总数为

<div class="display-equation">
$$
h\,d_{\mathrm{model}}d_h
=d_{\mathrm{model}}^2.
$$
</div>

Key 与 Value 同理，共 $3d_{\mathrm{model}}^2$。拼接后的矩阵维度仍为 $d_{\mathrm{model}}$，输出投影 $W_O$ 含 $d_{\mathrm{model}}^2$ 个参数。总计 $4d_{\mathrm{model}}^2$。$\square$

多头并不是免费增加容量。头数增加时，每头维度下降。头太少，关系类型可能挤在同一空间；头太多，每头维度过低，单头难以表达复杂匹配。头数是子空间多样性与单头容量之间的折中。

### 注意力头的冗余应怎样理解

实验中常能删掉部分头而性能下降不大。这不说明多头在训练时没有价值。冗余可能提供：

- 多种初始化与优化路径；
- 对部分关系的重复编码；
- 训练阶段的集成效应；
- 单个头失败时的补偿能力。

但也不能仅凭某个头的注意力图，就断言它“理解了语法”。注意力权重只是信息路由的一部分，Value 投影、残差路径和后续层都会改变最终表示。

## 十一、完整 Transformer 层不只有注意力

单独的自注意力只是动态混合 token。一个标准 Transformer block 还需要残差连接、归一化与逐位置前馈网络。

### 11.1 逐位置前馈网络

对每个 token 独立应用同一个两层 MLP：

<div class="display-equation">
$$
\operatorname{FFN}(x)
=W_2\phi(W_1x+b_1)+b_2,
$$
</div>

其中

<div class="display-equation">
$$
W_1\in\mathbb R^{d_{\mathrm{model}}\times d_{\mathrm{ff}}},
\qquad
W_2\in\mathbb R^{d_{\mathrm{ff}}\times d_{\mathrm{model}}}.
$$
</div>

注意力负责 token 之间的信息交换，FFN 负责每个 token 内部的非线性特征变换。

可以把二者概括为：

<div class="display-equation">
$$
\boxed{
\text{Attention：跨位置通信；}
\qquad
\text{FFN：逐位置计算。}
}
$$
</div>

### 11.2 残差连接

若子层为 $F$，残差形式为

<div class="display-equation">
$$
y=x+F(x).
$$
</div>

其 Jacobian 为

<div class="display-equation">
$$
\frac{\partial y}{\partial x}
=I+J_F(x).
$$
</div>

即使 $J_F$ 的某些方向很小，恒等项 $I$ 仍提供直接梯度路径。残差也允许子层只学习相对于输入的修正，而不必重新构造完整表示。

### 11.3 Layer Normalization

对单个 token 的特征维度进行标准化。若 $x\in\mathbb R^d$，

<div class="display-equation">
$$
\mu(x)=\frac1d\sum_{r=1}^dx_r,
$$
</div>

<div class="display-equation">
$$
\sigma^2(x)=\frac1d\sum_{r=1}^d(x_r-\mu(x))^2,
$$
</div>

<div class="display-equation">
$$
\operatorname{LN}(x)
=
\gamma\odot
\frac{x-\mu(x)\mathbf 1}
{\sqrt{\sigma^2(x)+\varepsilon}}
+\beta.
$$
</div>

它不依赖 batch 中其他样本，因此适合可变长度序列与自回归推理。

### 11.4 Post-LN 与 Pre-LN

原始 Transformer 使用 Post-LN：

<div class="display-equation">
$$
X_1=\operatorname{LN}(X+\operatorname{MHA}(X)),
$$
</div>

<div class="display-equation">
$$
X_2=\operatorname{LN}(X_1+\operatorname{FFN}(X_1)).
$$
</div>

现代深层模型常使用 Pre-LN：

<div class="display-equation">
$$
X_1=X+\operatorname{MHA}(\operatorname{LN}(X)),
$$
</div>

<div class="display-equation">
$$
X_2=X_1+\operatorname{FFN}(\operatorname{LN}(X_1)).
$$
</div>

Pre-LN 中残差主干更接近纯恒等映射，深层梯度通常更容易传播；Post-LN 的输出尺度控制更直接，但深层训练更依赖初始化、warmup 与其他稳定化技巧。

## 十二、编码器、解码器与交叉注意力

### 12.1 编码器层

编码器接收完整输入序列，使用无因果掩码的双向自注意力：

<div class="display-equation">
$$
Q=K=V=X
\quad\text{经过各自投影。}
$$
</div>

每个位置都可以读取左右两侧的所有位置。

### 12.2 解码器的 masked self-attention

解码器生成目标序列时，第 $t$ 个位置只能读取目标前缀。它使用因果掩码的自注意力。

### 12.3 交叉注意力

编码器—解码器模型还需要让解码器读取源序列。设编码器输出为

<div class="display-equation">
$$
H_{\mathrm{enc}}\in\mathbb R^{n_x\times d_{\mathrm{model}}},
$$
</div>

解码器当前状态为

<div class="display-equation">
$$
H_{\mathrm{dec}}\in\mathbb R^{n_y\times d_{\mathrm{model}}}.
$$
</div>

交叉注意力定义

<div class="display-equation">
$$
Q=H_{\mathrm{dec}}W_Q,
\qquad
K=H_{\mathrm{enc}}W_K,
\qquad
V=H_{\mathrm{enc}}W_V.
$$
</div>

于是注意力矩阵形状为

<div class="display-equation">
$$
QK^\top\in\mathbb R^{n_y\times n_x}.
$$
</div>

第 $i$ 行表示目标位置 $i$ 从所有源位置读取信息的权重。

因此三种注意力可以严格区分：

| 类型 | Query 来源 | Key/Value 来源 | 是否因果掩码 |
|---|---|---|---:|
| 编码器自注意力 | 编码器序列 | 同一编码器序列 | 否 |
| 解码器自注意力 | 目标前缀 | 同一目标前缀 | 是 |
| 交叉注意力 | 解码器状态 | 编码器输出 | 通常否 |

## 十三、三种预训练范式

### 13.1 编码器模型：BERT

编码器模型使用双向注意力，适合形成上下文相关表示。典型预训练目标是掩码语言模型：随机遮盖部分 token，并用两侧上下文预测它们。

目标可写为

<div class="display-equation">
$$
\mathcal L_{\mathrm{MLM}}
=-\sum_{i\in\mathcal M}
\log p_\theta(x_i\mid x_{\setminus\mathcal M}),
$$
</div>

其中 $\mathcal M$ 是被遮盖位置集合。

BERT 的优势是每个 token 可以同时利用左右上下文，适合分类、序列标注与抽取式问答。它本身不天然定义从左到右的生成过程。

### 13.2 解码器模型：GPT

解码器模型使用因果自注意力，按概率链式法则建模序列：

<div class="display-equation">
$$
p_\theta(x_1,\ldots,x_n)
=
\prod_{t=1}^n
p_\theta(x_t\mid x_1,\ldots,x_{t-1}).
$$
</div>

最大似然训练等价于最小化负对数似然：

<div class="display-equation">
$$
\mathcal L_{\mathrm{AR}}
=-\sum_{t=1}^n
\log p_\theta(x_t\mid x_{&lt;t}).
$$
</div>

训练时用真实前缀并行计算所有位置；推理时递归生成下一个 token。

### 13.3 编码器—解码器模型

对输入序列 $x$ 与输出序列 $y$，条件生成分解为

<div class="display-equation">
$$
p_\theta(y\mid x)
=
\prod_{t=1}^{n_y}
 p_\theta(y_t\mid y_{&lt;t},x).
$$
</div>

编码器先得到源序列表示，解码器在每一步通过交叉注意力读取源信息。翻译、摘要和文本到文本转换天然适合这一范式。

### 三种范式的本质比较

| 范式 | 可见上下文 | 核心目标 | 典型优势 | 主要限制 |
|---|---|---|---|---|
| Encoder-only | 双向全文 | 重建或判别 | 理解、表征 | 不天然自回归生成 |
| Decoder-only | 左侧前缀 | 下一 token 预测 | 统一生成接口 | 每步不能看未来 |
| Encoder–Decoder | 源端双向，目标端因果 | 条件序列生成 | 输入输出解耦 | 结构与计算更复杂 |

### 为什么 decoder-only 容易形成统一接口

任何任务只要能表示成“给定前缀，继续生成”，就可使用同一个自回归目标。分类可以生成标签，问答可以生成答案，推理可以生成中间步骤，工具调用可以生成结构化字符串。

这并不证明 decoder-only 在所有统计意义上都最优，而是说明它在训练目标、推理接口、缓存机制和任务统一之间形成了非常强的工程闭环。

## 十四、Tokenizer 与嵌入：进入 Transformer 之前发生了什么

Transformer 接收连续向量，而原始文本是字符串。文本进入模型前至少经历两步：tokenization 与 embedding。

### 14.1 Tokenizer 的核心矛盾

若 token 太粗，词汇表巨大，罕见词稀疏；若 token 太细，序列过长，注意力计算昂贵。

因此 tokenizer 实际在优化一种折中：

<div class="display-equation">
$$
\text{词汇表大小}
\quad\leftrightarrow\quad
\text{序列长度与语义完整性}.
$$
</div>

### 14.2 BPE 的逐步思想

BPE 从字符或字节开始，反复合并最频繁的相邻符号对。

假设语料为

<div class="display-equation">
$$
\texttt{low},\quad
\texttt{lower},\quad
\texttt{newest},\quad
\texttt{widest}.
$$
</div>

初始 token 是字符。若 `e` 与 `s` 最常相邻，则合并为 `es`；若 `es` 与 `t` 最常相邻，再合并为 `est`。经过多轮后，常见词根与词缀会成为独立 token，罕见词仍可由更小单元拼成。

这使模型不会因为遇到新词就完全失去表示能力。

### 14.3 Embedding 是可学习查表

设词汇表大小为 $V$，embedding 维度为 $d$，参数矩阵为

<div class="display-equation">
$$
E\in\mathbb R^{V\times d}.
$$
</div>

token 索引 $t$ 的表示是第 $t$ 行：

<div class="display-equation">
$$
x=E_{t,:}.
$$
</div>

若用 one-hot 向量 $e_t\in\mathbb R^V$，则

<div class="display-equation">
$$
e_t^\top E=E_{t,:}.
$$
</div>

所以 embedding lookup 与 one-hot 乘矩阵数学上等价，但查表避免了显式构造巨大稀疏向量。

### 静态表示与上下文表示

embedding 层中同一个 token 初始向量相同；经过多层自注意力后，其表示会依赖上下文。

例如 `bank` 在 “river bank” 与 “bank account” 中起始 embedding 相同，但高层 hidden state 不同。Transformer 的核心能力不只是学词向量，而是把静态 token 表示逐层改造成上下文表示。

## 十五、注意力的完整反向传播

理解前向公式以后，可以逐步推导梯度。设

<div class="display-equation">
$$
S=\frac{QK^\top}{\sqrt{d_k}},
\qquad
A=\operatorname{softmax}_{\mathrm{row}}(S),
\qquad
Y=AV.
$$
</div>

令上游梯度为

<div class="display-equation">
$$
G_Y:=\frac{\partial L}{\partial Y}.
$$
</div>

### 第一步：对 $V$ 与 $A$ 求梯度

由 $Y=AV$，矩阵微分为

<div class="display-equation">
$$
dY=dA\,V+A\,dV.
$$
</div>

利用 Frobenius 内积

<div class="display-equation">
$$
dL=\langle G_Y,dY\rangle_F,
$$
</div>

得到

<div class="display-equation">
$$
\frac{\partial L}{\partial V}
=A^\top G_Y,
$$
</div>

<div class="display-equation">
$$
\frac{\partial L}{\partial A}
=G_YV^\top.
$$
</div>

第一条表示：一个 Value 的梯度由所有读取它的 Query 加权累积。第二条表示：改变注意力权重的价值取决于对应 Value 与上游梯度的内积。

### 第二步：穿过 row-wise softmax

固定第 $i$ 行，记

<div class="display-equation">
$$
a_i=\operatorname{softmax}(s_i),
\qquad
g_i=\frac{\partial L}{\partial a_i}.
$$
</div>

softmax Jacobian 为

<div class="display-equation">
$$
J_i=\operatorname{diag}(a_i)-a_ia_i^\top.
$$
</div>

故

<div class="display-equation">
$$
\frac{\partial L}{\partial s_i}
=J_i g_i.
$$
</div>

逐元素可写成更高效的形式：

<div class="display-equation">
$$
\frac{\partial L}{\partial s_i}
=
a_i\odot
\left(
 g_i-\langle g_i,a_i\rangle\mathbf 1
\right).
$$
</div>

这个公式说明，每一行的 logit 梯度会减去加权平均，因此其元素和为零：

<div class="display-equation">
$$
\mathbf 1^\top
\frac{\partial L}{\partial s_i}=0.
$$
</div>

原因是给一整行 logits 同时加同一个常数不会改变 softmax。

### 第三步：对 $Q$ 与 $K$ 求梯度

令

<div class="display-equation">
$$
G_S:=\frac{\partial L}{\partial S}.
$$
</div>

由

<div class="display-equation">
$$
S=\frac{QK^\top}{\sqrt{d_k}},
$$
</div>

得到

<div class="display-equation">
$$
\frac{\partial L}{\partial Q}
=
\frac{G_SK}{\sqrt{d_k}},
$$
</div>

<div class="display-equation">
$$
\frac{\partial L}{\partial K}
=
\frac{G_S^\top Q}{\sqrt{d_k}}.
$$
</div>

Query 的梯度由所有 Key 加权形成；Key 的梯度由所有 Query 的反向影响累积。

### 第四步：传到投影矩阵与输入

若

<div class="display-equation">
$$
Q=XW_Q,
\qquad K=XW_K,
\qquad V=XW_V,
$$
</div>

则

<div class="display-equation">
$$
\frac{\partial L}{\partial W_Q}
=X^\top\frac{\partial L}{\partial Q},
$$
</div>

<div class="display-equation">
$$
\frac{\partial L}{\partial W_K}
=X^\top\frac{\partial L}{\partial K},
$$
</div>

<div class="display-equation">
$$
\frac{\partial L}{\partial W_V}
=X^\top\frac{\partial L}{\partial V}.
$$
</div>

输入同时沿三条路径接收梯度：

<div class="display-equation">
$$
\frac{\partial L}{\partial X}
=
\frac{\partial L}{\partial Q}W_Q^\top
+
\frac{\partial L}{\partial K}W_K^\top
+
\frac{\partial L}{\partial V}W_V^\top.
$$
</div>

这也说明，Q、K、V 不只是前向中的三个角色，它们在反向传播中提供三种不同的学习信号。

## 十六、复杂度：真正昂贵的部分在哪里

设序列长度为 $n$，模型维度为 $d$，前馈隐藏维度为 $d_{\mathrm{ff}}$。

### 16.1 线性投影

计算 $Q,K,V$ 的复杂度约为

<div class="display-equation">
$$
O(nd^2).
$$
</div>

### 16.2 注意力分数与 Value 聚合

<div class="display-equation">
$$
QK^\top:
O(n^2d),
$$
</div>

<div class="display-equation">
$$
AV:
O(n^2d).
$$
</div>

标准注意力核心复杂度为 $O(n^2d)$，显式注意力矩阵需要 $O(n^2)$ 级空间（多头时乘头数，但每头维度相应缩小）。

### 16.3 前馈网络

<div class="display-equation">
$$
O(ndd_{\mathrm{ff}}).
$$
</div>

当序列较短而 $d_{\mathrm{ff}}$ 很大时，FFN 可能占主要计算；当序列很长时，$n^2$ 注意力成为瓶颈。

### 16.4 与 RNN 的比较

| 模块 | 总算术复杂度 | 序列方向最短依赖路径 | 可并行性 |
|---|---:|---:|---:|
| RNN | $O(nd^2)$ | $O(n)$ | 低 |
| 全局自注意力 | $O(n^2d+nd^2)$ | $O(1)$ | 高 |
| 局部注意力 | 约 $O(nwd)$ | $O(n/w)$ 或分层传播 | 高 |

Transformer 用更大的全局交互代价，换取了极短的信息路径与硬件并行性。

### 16.5 高效注意力的三条路线

第一，稀疏化：只关注局部窗口、全局标记或结构化邻域。

第二，低秩化：假设注意力矩阵有效秩较低，用投影压缩 Key 和 Value。

第三，核近似：把指数内积核写成特征映射内积，从

<div class="display-equation">
$$
\operatorname{softmax}(QK^\top)V
$$
</div>

重排成不显式构造 $n\times n$ 矩阵的形式。

需要区分“算术复杂度”与“显存复杂度”。一些精确注意力实现通过分块和重计算避免存储完整注意力矩阵，显著降低显存，但标准全局注意力的乘加次数仍是二次量级。

## 十七、KV Cache：为什么自回归推理不必反复重算过去

在 decoder-only 模型中，生成第 $t$ 个 token 时，前 $t-1$ 个 token 的 Key 与 Value 已经算过。

对每一层缓存

<div class="display-equation">
$$
K_{1:t-1},
\qquad
V_{1:t-1}.
$$
</div>

新 token 只需计算

<div class="display-equation">
$$
q_t,
\qquad
k_t,
\qquad
v_t,
$$
</div>

再把 $k_t,v_t$ 追加到缓存，并计算

<div class="display-equation">
$$
y_t
=
\operatorname{softmax}\!\left(
\frac{q_tK_{1:t}^\top}{\sqrt{d_k}}
\right)V_{1:t}.
$$
</div>

每个新 token 的注意力代价随当前长度 $t$ 线性增长，总生成 $n$ 个 token 的注意力代价仍约为 $O(n^2d)$。但若每一步都从头重算整个前缀，代价会进一步恶化。

KV Cache 的代价是显存。每层都要保存所有过去位置的 Key 和 Value，缓存大小与序列长度、层数、batch size 和模型宽度近似线性增长。

因此长上下文推理的核心矛盾是：

<div class="display-equation">
$$
\text{避免重复计算}
\quad\Longleftrightarrow\quad
\text{保存更大的历史状态。}
$$
</div>

## 十八、Transformer 为什么能跨模态

Transformer 并不要求 token 必须是单词。它只要求输入可以表示成一组向量，并且这些向量之间存在值得学习的关系。

### 18.1 图像：把 patch 当成 token

设图像大小为 $H\times W$，patch 大小为 $P\times P$，则 token 数量为

<div class="display-equation">
$$
n=\frac{HW}{P^2}.
$$
</div>

每个 patch 展平后线性投影为 $d$ 维向量，再加入位置编码。

若 patch 边长减半，token 数量变为原来的四倍，注意力矩阵元素数变为原来的十六倍：

<div class="display-equation">
$$
P\mapsto \frac P2
\quad\Longrightarrow\quad
n\mapsto4n
\quad\Longrightarrow\quad
n^2\mapsto16n^2.
$$
</div>

这解释了高分辨率视觉 Transformer 为什么需要窗口注意力、层次化下采样或稀疏机制。

### 18.2 语音：把时间—频率块当成 token

音频可先转为梅尔频谱图，再沿时间或二维 patch 划分为序列。编码器处理声学 token，解码器生成文本 token，交叉注意力负责声学与文字对齐。

### 18.3 多模态：统一路由，不一定统一编码器

图像与文本可以各自编码后投影到共享空间，也可以把视觉 token 与文本 token 拼接后共同送入 Transformer，还可以让文本 Query 通过交叉注意力读取视觉 Key/Value。

因此多模态的核心不是简单地“把所有数据拼起来”，而是设计：

- 每种模态怎样变成 token；
- 各模态使用独立还是共享表示空间；
- 哪些层内部交互，哪些层通过交叉注意力交互；
- 如何定义对齐与生成目标。

### 18.4 科学数据

蛋白质残基、图节点、表格字段、时间序列片段都可以被 token 化。注意力提供的是对成对关系的通用建模器，而不是语言专属规则。

## 十九、常见误解与严格边界

### 误解 1：注意力权重就是模型解释

注意力矩阵只能说明某一层某一头在前向传播中怎样混合 Value。最终预测还经过 Value 投影、输出投影、残差、FFN 和后续层。高注意力不必然意味着该 token 对最终输出具有最大因果影响。

### 误解 2：Transformer 完全没有序列性

训练时可并行计算，但因果语言模型的生成仍按 token 串行。位置编码与掩码也明确注入了序列结构。

### 误解 3：多头只是把一个大头拆小

参数量可能相同，但不同头拥有独立投影和独立 softmax。多个低维注意力分布不是一个高维单头的简单坐标切分，它们定义不同的信息路由。

### 误解 4：位置编码只是给 token 编号

位置编码的目标不是记住绝对序号本身，而是让注意力分数能够表达顺序、距离、方向和相对位移。不同方案对应不同的位置归纳偏置与外推性质。

### 误解 5：注意力解决了所有长程依赖

路径长度虽为一步，但模型能否真正利用远距离信息仍取决于训练数据、优化、位置表示、注意力稀疏性与有限精度。理论上可直接连接，不等于实践中一定有效使用。

### 误解 6：$O(n^2)$ 意味着所有 Transformer 都无法处理长序列

二次复杂度针对标准全局注意力。局部、稀疏、低秩、核近似、状态空间混合与分块精确算法可以改变时间或显存代价，但通常会引入结构假设、近似误差或新的工程权衡。

## 二十、从结构到思想：Transformer 最值得保留的五个原则

### 原则一：把信息路由变成数据依赖函数

固定卷积核或递归矩阵规定了预先设计的信息通道；注意力让通道随输入改变。

### 原则二：将“在哪里找”与“找到后传什么”分离

Query/Key 负责匹配，Value 负责内容。路由与计算解耦以后，结构具有更强的组合性。

### 原则三：用短路径换并行与长程交互

任意两个位置一层即可直接通信，代价是显式或隐式处理大量成对关系。

### 原则四：通过残差主干叠加小修正

深层模型不必每层重写表示，而可以在稳定主干上逐步修改。残差与归一化是可训练性的核心，不是外围附件。

### 原则五：把模态差异推到 tokenization 之前

一旦图像、音频、文本或科学结构被映射到 token 序列，后续可共享同一套注意力与前馈计算。这种统一性是 Transformer 跨领域扩张的根本原因。

## 结论

Transformer 的核心可以被压缩成一个动态加权平均，但真正理解它，必须把公式中的每个部分拆开。

Query 表示当前位置需要寻找什么；Key 表示每个位置可以怎样被匹配；Value 表示匹配成功后实际传递什么。内积构造可学习相似度，softmax 将相似度转成稳定权重，$\sqrt{d_k}$ 保持随机内积的尺度，多头在不同子空间中并行建立多种路由。

没有位置编码的自注意力只是置换等变的集合运算。绝对位置、相对偏置、RoPE 或 ALiBi 为它加入顺序。因果掩码把全局访问限制为只看过去，从而得到自回归语言模型。

完整 Transformer block 还依赖残差连接、LayerNorm 和 FFN：注意力完成跨 token 通信，FFN 完成逐 token 非线性计算，残差主干保证深层信息与梯度能够传播。

从理论上看，自注意力是一种可学习核回归；从计算上看，它用 $O(n^2)$ 成对交互换取 $O(1)$ 的长程路径；从系统上看，KV Cache 以显存换取自回归推理中的计算复用；从表示学习上看，它把文本、图像、音频和结构化科学数据统一为 token 之间的动态关系建模。

因此，Transformer 最深刻的贡献并不是某一条具体公式，而是一种通用计算思想：

<div class="display-equation">
$$
\boxed{
\text{先根据当前内容构造信息路由，}
\quad
\text{再沿路由聚合与变换表示。}
}
$$
</div>

这套思想既解释了它为什么能够突破 RNN 的串行瓶颈，也解释了为什么注意力会成为现代人工智能中最普遍的基础算子之一。

## 参考资料

1. Ashish Vaswani et al. *Attention Is All You Need*. NeurIPS, 2017.
2. Jacob Devlin et al. *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL, 2019.
3. Alec Radford et al. *Improving Language Understanding by Generative Pre-Training*. 2018.
4. Colin Raffel et al. *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer*. JMLR, 2020.
5. Alexey Dosovitskiy et al. *An Image Is Worth 16×16 Words: Transformers for Image Recognition at Scale*. ICLR, 2021.
6. Ze Liu et al. *Swin Transformer: Hierarchical Vision Transformer Using Shifted Windows*. ICCV, 2021.
7. Peter Shaw, Jakob Uszkoreit, and Ashish Vaswani. *Self-Attention with Relative Position Representations*. NAACL, 2018.
8. Jianlin Su et al. *RoFormer: Enhanced Transformer with Rotary Position Embedding*. Neurocomputing, 2024.
9. Ofir Press, Noah A. Smith, and Mike Lewis. *Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation*. ICLR, 2022.
10. Tay et al. *Efficient Transformers: A Survey*. ACM Computing Surveys, 2022.
