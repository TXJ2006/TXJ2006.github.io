---
title: "Gan Note"
subtitle: ""
summary: "第七章 对抗学习"
description: "第七章 对抗学习"
date: 2026-07-23
lastmod: 2026-07-23
weight: 90
tags: []
draft: false
ShowToc: false
hideMeta: true
libraryFolder: "ai-foundations"
libraryFolderName: "人工智能基础"
libraryFolderColor: 0
---

第七章 对抗学习

前面的章节中，我们一直在做一件相对明确的事情：先给模型一个输入，再告诉模型正确输出是什么。分类问题有类别标签，回归问题有目标数值，语言模型虽然没有人工标注，却可以把序列中的下一个 token 当作监督信号。模型做错以后，损失函数能够直接指出它错了多少。

生成任务没有这么方便。假设我们希望模型生成一张新的猫的图片。模型输出一张图片以后，我们并不存在一张与它逐像素对应的“标准答案”。训练集中当然有很多猫的图片，但把其中任意一张拿出来与生成图片做均方误差并不合理，因为两张都很真实的猫图也可能在每一个像素上都完全不同。

所以生成模型要解决的不是一个普通的点对点映射问题，而是一个分布问题。模型不必把某个噪声 $z$ 映射成某张指定图片，它只需要保证：当 $z$ 按照参考分布反复采样时，所有输出共同形成的分布与真实数据分布尽可能接近。

设参考随机变量满足

<div class="display-equation">
$$
z\sim P_z,
$$
</div>

生成器是由参数 $\theta$ 控制的神经网络

<div class="display-equation">
$$
G_\theta:\mathcal Z\to\mathcal X.
$$
</div>

生成样本为

<div class="display-equation">
$$
x_g=G_\theta(z).
$$
</div>

由此诱导出的生成分布记为

<div class="display-equation">
$$
P_g=(G_\theta)_\#P_z.
$$
</div>

这里的前推符号并不神秘。对任意可测集合 $A\subseteq\mathcal X$，有

<div class="display-equation">
$$
P_g(A)
=
P_z\bigl(G_\theta^{-1}(A)\bigr).
$$
</div>

它表示生成样本落入区域 $A$ 的概率，等于潜变量落入 $A$ 的原像中的概率。生成建模的目标可以写成

<div class="display-equation">
$$
P_g\approx P_{\mathrm{data}}.
$$
</div>

问题随即变成：怎样计算两个复杂分布之间的差距？

GAN 的回答是，不直接写出这个差距，而是训练一个判别器，让判别器主动寻找两个分布之间最容易被利用的区别。生成器再根据判别器暴露出的区别修正自己。于是，损失函数不再是一个完全固定的公式，而是由另一个不断学习的模型动态产生。

<div class="display-equation">
$$
\boxed{
\text{生成器寻找更像真的样本，}
\qquad
\text{判别器寻找仍然不像真的证据。}
}
$$
</div>

这便是对抗学习的起点。

7.1 生成对抗网络的基本思想

7.1.1 为什么不能把噪声与真实样本直接配对

在进入 GAN 以前，先考虑一个看起来很自然、实际上必然失败的方案。每次随机抽取噪声 $z\sim P_z$，再独立抽取真实样本 $x\sim P_{\mathrm{data}}$，让生成器最小化均方误差

<div class="display-equation">
$$
L(\theta)
=
\mathbb E_{z,x}
\bigl[\|G_\theta(z)-x\|_2^2\bigr].
$$
</div>

固定某个潜变量 $z$，记生成器输出为 $g=G_\theta(z)$。由于 $x$ 与 $z$ 独立，固定 $z$ 后需要最小化

<div class="display-equation">
$$
\mathbb E_x\|g-x\|_2^2.
$$
</div>

令数据均值为

<div class="display-equation">
$$
\mu=\mathbb E[x].
$$
</div>

将 $g-x$ 写成 $(g-\mu)+(\mu-x)$，得到

<div class="display-equation">
$$
\begin{aligned}
\mathbb E_x\|g-x\|_2^2
&=
\mathbb E_x
\bigl\|(g-\mu)+(\mu-x)\bigr\|_2^2\\
&=
\|g-\mu\|_2^2
+
2(g-\mu)^\top\mathbb E_x(\mu-x)
+
\mathbb E_x\|x-\mu\|_2^2.
\end{aligned}
$$
</div>

因为

<div class="display-equation">
$$
\mathbb E_x(\mu-x)=0,
$$
</div>

所以

<div class="display-equation">
$$
\mathbb E_x\|g-x\|_2^2
=
\|g-\mu\|_2^2
+
\mathbb E_x\|x-\mu\|_2^2.
$$
</div>

第二项与 $g$ 无关，因此唯一最优输出是

<div class="display-equation">
$$
g^\star=\mu.
$$
</div>

这个结论对每个 $z$ 都成立，所以最优生成器会满足

<div class="display-equation">
$$
G_\theta(z)
\equiv
\mathbb E[x].
$$
</div>

所有噪声都被映射成同一个数据均值。对于图像而言，这通常是一张模糊的平均图。这个推导说明，如果真实样本与噪声之间没有天然配对关系，那么逐样本回归会把分布学习错误地变成条件均值估计，最终主动消灭生成多样性。

因此，生成模型必须比较整批样本所形成的分布，而不能要求每个生成样本接近某个随意抽取的真实样本。GAN 的判别器正是为分布层面的比较而设计的。

7.1.2 从分布比较变成真假分类

记判别器为

<div class="display-equation">
$$
D_\phi:\mathcal X\to(0,1).
$$
</div>

$D_\phi(x)$ 被解释为样本 $x$ 来自真实数据而非生成器的概率。为了训练它，我们构造一个平衡二分类问题。令类别变量 $Y$ 满足

<div class="display-equation">
$$
\mathbb P(Y=1)
=
\mathbb P(Y=0)
=
\frac12,
$$
</div>

其中 $Y=1$ 表示真实样本，$Y=0$ 表示生成样本，并规定

<div class="display-equation">
$$
x\mid Y=1
\sim P_{\mathrm{data}},
\qquad
x\mid Y=0
\sim P_g.
$$
</div>

若两个分布相对于同一参考测度存在密度，由 Bayes 公式，真实类别的后验概率为

<div class="display-equation">
$$
\begin{aligned}
\mathbb P(Y=1\mid x)
&=
\frac{p_{\mathrm{data}}(x)\mathbb P(Y=1)}
{p_{\mathrm{data}}(x)\mathbb P(Y=1)+p_g(x)\mathbb P(Y=0)}\\
&=
\frac{p_{\mathrm{data}}(x)}
{p_{\mathrm{data}}(x)+p_g(x)}.
\end{aligned}
$$
</div>

这表明一个理想判别器实际上在估计密度比。把上式改写为 odds，得到

<div class="display-equation">
$$
\frac{D^\star(x)}{1-D^\star(x)}
=
\frac{p_{\mathrm{data}}(x)}{p_g(x)}.
$$
</div>

再取对数，得到

<div class="display-equation">
$$
\operatorname{logit}D^\star(x)
=
\log p_{\mathrm{data}}(x)
-
\log p_g(x).
$$
</div>

所以判别器并不只是学习一些固定的“真图特征”。它试图估计某个位置在真实分布下与在生成分布下的相对可能性。生成器改变以后，$p_g$ 改变，最优判别规则也会跟着改变。这就是 GAN 的损失曲面不断移动的原因。

7.1.3 原始极小极大目标

判别器在真实样本上希望输出一，在生成样本上希望输出零。使用 Bernoulli 对数似然，它希望最大化

<div class="display-equation">
$$
V(D_\phi,G_\theta)
=
\mathbb E_{x\sim P_{\mathrm{data}}}
[\log D_\phi(x)]
+
\mathbb E_{z\sim P_z}
[\log(1-D_\phi(G_\theta(z)))].
$$
</div>

生成器希望判别器无法区分真假，于是试图最小化同一个目标。原始 GAN 写成

<div class="display-equation">
$$
\min_\theta
\max_\phi
V(D_\phi,G_\theta).
$$
</div>

固定生成器时，判别器寻找最有力的分布差异；固定判别器时，生成器沿着当前差异所产生的梯度移动。若分别使用学习率 $\eta_D$ 和 $\eta_G$，最朴素的交替更新为

<div class="display-equation">
$$
\phi_{t+1}
=
\phi_t
+
\eta_D
\nabla_\phi
V(D_{\phi_t},G_{\theta_t}),
$$
</div>

<div class="display-equation">
$$
\theta_{t+1}
=
\theta_t
-
\eta_G
\nabla_\theta
V(D_{\phi_{t+1}},G_{\theta_t}).
$$
</div>

普通监督学习是在一张固定损失曲面上下降。GAN 中，生成器下降一步后，判别器会重新塑造曲面；判别器上升一步后，生成器所面对的方向也会变化。因此，即使每个子问题单独看都可以优化，二者的联立动力学也未必收敛。

7.1.4 最优判别器

固定生成器 $G$，即固定 $P_g$。目标函数可以写成

<div class="display-equation">
$$
V(D,G)
=
\int_{\mathcal X}
\left[
p_{\mathrm{data}}(x)\log D(x)
+
p_g(x)\log(1-D(x))
\right]dx.
$$
</div>

由于积分中不同位置的 $D(x)$ 可以分别优化，只需要研究一元函数

<div class="display-equation">
$$
f(d)
=
a\log d+b\log(1-d),
\qquad
0&lt;d&lt;1,
$$
</div>

其中

<div class="display-equation">
$$
a=p_{\mathrm{data}}(x),
\qquad
b=p_g(x).
$$
</div>

定理 7.1.1（最优判别器）　固定生成器后，在 $a+b>0$ 的位置，最优判别器满足

<div class="display-equation">
$$
D^\star(x)
=
\frac{p_{\mathrm{data}}(x)}
{p_{\mathrm{data}}(x)+p_g(x)}.
$$
</div>

证明很直接。对 $d$ 求导，得到

<div class="display-equation">
$$
f'(d)
=
\frac{a}{d}
-
\frac{b}{1-d}.
$$
</div>

令导数为零，得到

<div class="display-equation">
$$
a(1-d)=bd,
$$
</div>

所以

<div class="display-equation">
$$
d
=
\frac{a}{a+b}.
$$
</div>

二阶导数为

<div class="display-equation">
$$
f''(d)
=
-\frac{a}{d^2}
-
\frac{b}{(1-d)^2}
&lt;0,
$$
</div>

因此该驻点是唯一最大值。证毕。

当 $p_{\mathrm{data}}(x)=p_g(x)$ 时，最优判别器输出

<div class="display-equation">
$$
D^\star(x)=\frac12.
$$
</div>

这不是判别器能力不足，而是在两个分布相同的情况下，任何分类器都不可能得到优于随机猜测的 Bayes 错误率。

7.1.5 从最优判别器推导 Jensen--Shannon 散度

将 $D^\star$ 代回目标函数，得到

<div class="display-equation">
$$
\begin{aligned}
V(D^\star,G)
=&
\int p_{\mathrm{data}}(x)
\log
\frac{p_{\mathrm{data}}(x)}
{p_{\mathrm{data}}(x)+p_g(x)}dx\\
&+
\int p_g(x)
\log
\frac{p_g(x)}
{p_{\mathrm{data}}(x)+p_g(x)}dx.
\end{aligned}
$$
</div>

定义混合分布

<div class="display-equation">
$$
M
=
\frac12
(P_{\mathrm{data}}+P_g),
$$
</div>

其密度为

<div class="display-equation">
$$
m(x)
=
\frac12
\bigl(
p_{\mathrm{data}}(x)+p_g(x)
\bigr).
$$
</div>

于是

<div class="display-equation">
$$
p_{\mathrm{data}}(x)+p_g(x)=2m(x).
$$
</div>

第一项可以写成

<div class="display-equation">
$$
\begin{aligned}
\int p_{\mathrm{data}}
\log
\frac{p_{\mathrm{data}}}
{p_{\mathrm{data}}+p_g}
&=
\int p_{\mathrm{data}}
\log
\frac{p_{\mathrm{data}}}{2m}\\
&=
\int p_{\mathrm{data}}
\log
\frac{p_{\mathrm{data}}}{m}
-
\log2
\int p_{\mathrm{data}}\\
&=
D_{\mathrm{KL}}
(P_{\mathrm{data}}\|M)
-
\log2.
\end{aligned}
$$
</div>

同理，第二项为

<div class="display-equation">
$$
D_{\mathrm{KL}}(P_g\|M)-\log2.
$$
</div>

Jensen--Shannon 散度定义为

<div class="display-equation">
$$
D_{\mathrm{JS}}(P\|Q)
=
\frac12D_{\mathrm{KL}}(P\|M)
+
\frac12D_{\mathrm{KL}}(Q\|M),
\qquad
M=\frac12(P+Q).
$$
</div>

因此有

定理 7.1.2（GAN 的理想分布目标）　若判别器对固定生成器达到最优，则

<div class="display-equation">
$$
V(D^\star,G)
=
-\log4
+
2D_{\mathrm{JS}}
(P_{\mathrm{data}}\|P_g).
$$
</div>

由于

<div class="display-equation">
$$
D_{\mathrm{JS}}(P_{\mathrm{data}}\|P_g)\geq0,
$$
</div>

且等号当且仅当两个分布相等时成立，理想全局最优点满足

<div class="display-equation">
$$
P_g=P_{\mathrm{data}},
$$
</div>

此时

<div class="display-equation">
$$
V(D^\star,G)
=
-\log4.
$$
</div>

需要注意，分布层面的最优解可以唯一，参数层面的最优解却通常不唯一。不同的生成器参数甚至不同的潜空间重参数化，都可能诱导相同的 $P_g$。GAN 要求的是输出分布相等，并不要求生成映射本身唯一。

这个定理也依赖非常理想的条件。它假设判别器在每次生成器更新前都达到函数空间中的最优，生成器与判别器容量足够，积分与密度操作合法，而且数值优化能够找到全局解。实际 GAN 只是在有限神经网络类中交替走少量梯度步，因此“理论上等价于最小化 JS 散度”不能被理解为每一次实际更新都严格沿 JS 散度下降。

7.1.6 原始生成器损失为什么会饱和

原始极小极大形式要求生成器最小化

<div class="display-equation">
$$
L_G^{\mathrm{MM}}
=
\mathbb E_z
\bigl[
\log(1-D(G(z)))
\bigr].
$$
</div>

训练初期，生成样本很差，判别器常有

<div class="display-equation">
$$
D(G(z))\approx0.
$$
</div>

令判别器最后的 logit 为 $a(x)$，并写成

<div class="display-equation">
$$
D(x)=\sigma(a(x)).
$$
</div>

利用

<div class="display-equation">
$$
\sigma'(a)
=
\sigma(a)
(1-\sigma(a)),
$$
</div>

可以计算

<div class="display-equation">
$$
\frac{d}{da}
\log(1-\sigma(a))
=
-\sigma(a).
$$
</div>

当 $D=\sigma(a)$ 接近零时，这个导数也接近零。生成器的梯度还要继续通过判别器关于输入的 Jacobian 和生成器关于参数的 Jacobian 传播，最后得到

<div class="display-equation">
$$
\nabla_\theta
L_G^{\mathrm{MM}}
=
-
\mathbb E_z
\left[
D(G_\theta(z))
J_{G_\theta}(z)^\top
\nabla_xa(x)
\big|_{x=G_\theta(z)}
\right].
$$
</div>

前面的因子 $D(G_\theta(z))$ 使梯度在判别器非常自信时被压到接近零。于是，判别器越容易识别生成样本，生成器反而越难得到修正方向。

实践中通常使用非饱和生成器损失

<div class="display-equation">
$$
L_G^{\mathrm{NS}}
=
-
\mathbb E_z
\bigl[
\log D(G(z))
\bigr].
$$
</div>

因为

<div class="display-equation">
$$
\frac{d}{da}
[-\log\sigma(a)]
=
-(1-\sigma(a)),
$$
</div>

所以当 $D(G(z))\approx0$ 时，导数约为 $-1$，不会消失。其参数梯度为

<div class="display-equation">
$$
\nabla_\theta
L_G^{\mathrm{NS}}
=
-
\mathbb E_z
\left[
(1-D(G_\theta(z)))
J_{G_\theta}(z)^\top
\nabla_xa(x)
\big|_{x=G_\theta(z)}
\right].
$$
</div>

非饱和损失与原始损失具有相同的理想平衡点，却在远离平衡点时提供更强梯度。这说明神经网络训练不能只比较最优点，还必须比较到达最优点的动力学路径。

7.1.7 一个可以完全算清楚的高斯例子

设真实数据与生成数据都是一维高斯分布，并且方差相同：

<div class="display-equation">
$$
P_{\mathrm{data}}
=
\mathcal N(\mu_r,\sigma^2),
\qquad
P_g
=
\mathcal N(\mu_g,\sigma^2).
$$
</div>

高斯密度为

<div class="display-equation">
$$
p_\mu(x)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp
\left(
-\frac{(x-\mu)^2}{2\sigma^2}
\right).
$$
</div>

最优判别器的 logit 是对数密度比：

<div class="display-equation">
$$
\operatorname{logit}D^\star(x)
=
\log
\frac{p_{\mu_r}(x)}
{p_{\mu_g}(x)}.
$$
</div>

代入高斯密度，归一化常数抵消，得到

<div class="display-equation">
$$
\begin{aligned}
\log
\frac{p_{\mu_r}(x)}
{p_{\mu_g}(x)}
&=
-\frac{(x-\mu_r)^2}{2\sigma^2}
+
\frac{(x-\mu_g)^2}{2\sigma^2}\\
&=
\frac{
2(\mu_r-\mu_g)x
+
\mu_g^2-\mu_r^2
}{2\sigma^2}.
\end{aligned}
$$
</div>

因此

<div class="display-equation">
$$
D^\star(x)
=
\sigma
\left(
\frac{\mu_r-\mu_g}{\sigma^2}x
-
\frac{\mu_r^2-\mu_g^2}{2\sigma^2}
\right).
$$
</div>

也就是说，当真假分布具有相同方差、只在均值上不同的时候，最优判别器恰好是一个 logistic 线性分类器。它的决策边界满足 logit 为零，即

<div class="display-equation">
$$
x^\star
=
\frac{\mu_r+\mu_g}{2}.
$$
</div>

判别器把两个均值的中点作为分界。均值距离越大，斜率

<div class="display-equation">
$$
\frac{|\mu_r-\mu_g|}{\sigma^2}
$$
</div>

越大，判别器越接近一个硬阈值；方差越大，两个分布重叠越明显，判别器越平缓。

这个例子把“判别器估计密度比”变成了一个可以直接看见的公式。它也解释了为什么在训练初期，若真假分布相距很远，最优判别器会非常陡峭，生成器只在狭窄的决策边界附近得到明显梯度。

7.1.8 损失函数改变的是梯度形状

原始 GAN 使用 logistic 损失，但对抗学习并不要求判别器一定采用二分类交叉熵。不同损失往往具有相同或相近的平衡点，却给出不同的局部梯度。

Least Squares GAN 令判别器回归真假标签。若真实标签取 $b$、生成标签取 $a$，判别器损失为

<div class="display-equation">
$$
L_D^{\mathrm{LS}}
=
\frac12
\mathbb E_{x\sim P_{\mathrm{data}}}
(D(x)-b)^2
+
\frac12
\mathbb E_{z\sim P_z}
(D(G(z))-a)^2.
$$
</div>

生成器希望判别输出接近另一个目标值 $c$：

<div class="display-equation">
$$
L_G^{\mathrm{LS}}
=
\frac12
\mathbb E_z
(D(G(z))-c)^2.
$$
</div>

平方损失不会像 sigmoid 交叉熵那样在分类已经正确时立即进入极端饱和区。生成样本即使位于决策边界的错误一侧，只要判别分数与目标仍有距离，就会继续获得线性增长的梯度。

现代高质量图像 GAN 中还经常使用 hinge loss。判别器损失写成

<div class="display-equation">
$$
L_D^{\mathrm{hinge}}
=
\mathbb E_{x\sim P_{\mathrm{data}}}
[\max(0,1-D(x))]
+
\mathbb E_z
[\max(0,1+D(G(z)))].
$$
</div>

生成器损失为

<div class="display-equation">
$$
L_G^{\mathrm{hinge}}
=
-
\mathbb E_z
[D(G(z))].
$$
</div>

真实样本只要分数超过一，判别器就不再继续把它推高；生成样本只要分数低于负一，也不再继续被推低。判别器因而把容量集中在决策边界附近，而不是无止境增大已经正确分类样本的 logit。

这些损失的共同目的不是改变“真假分布最终应当一致”这一基本方向，而是塑造一个更适合有限步梯度优化的局部几何。GAN 的训练效果对损失形状高度敏感，正是因为实际算法远未处于“每轮判别器都达到理论最优”的极限状态。

7.2 GAN 的训练动力学与困难

7.2.1 支集分离与 JS 散度的不连续性

高维自然数据通常集中在一个低维流形附近。生成器若从低维潜变量连续映射到图像空间，其输出也集中在另一个低维集合附近。训练初期，这两个集合几乎必然不重合。

考虑最简单的一维分布

<div class="display-equation">
$$
P_r=\delta_0,
\qquad
P_\theta=\delta_\theta.
$$
</div>

只要 $\theta\neq0$，两个分布的支集完全分离。混合分布为

<div class="display-equation">
$$
M_\theta
=
\frac12\delta_0
+
\frac12\delta_\theta.
$$
</div>

在点 $0$ 上，$P_r$ 相对于 $M_\theta$ 的密度比为 $2$，所以

<div class="display-equation">
$$
D_{\mathrm{KL}}(P_r\|M_\theta)
=
\log2.
$$
</div>

同理

<div class="display-equation">
$$
D_{\mathrm{KL}}(P_\theta\|M_\theta)
=
\log2.
$$
</div>

因此

<div class="display-equation">
$$
D_{\mathrm{JS}}(\delta_0\|\delta_\theta)
=
\begin{cases}
0,
&\theta=0,\\
\log2,
&\theta\neq0.
\end{cases}
$$
</div>

当 $\theta$ 从 $10$ 逐渐变到 $10^{-6}$ 时，两个点已经越来越接近，但 JS 散度始终等于 $\log2$。只有在 $\theta$ 精确等于零时，它才突然跳到零。

更值得注意的是，随机变量对应的分布满足

<div class="display-equation">
$$
\delta_\theta
\Rightarrow
\delta_0
\qquad
\text{当 }\theta\to0,
$$
</div>

但

<div class="display-equation">
$$
D_{\mathrm{JS}}(\delta_0\|\delta_\theta)
\nrightarrow0.
$$
</div>

所以 JS 散度诱导的拓扑比弱收敛强得多。它能够判断支集是否重合，却在支集尚未重合时缺少平滑的几何方向。

实际神经网络判别器不会真正成为不连续的 Bayes 分类器，因此有限训练阶段仍可能提供梯度；但判别器越接近完美分类，生成器看到的梯度越容易退化。这解释了为什么“把判别器训练得更好”并不总能改善 GAN。

7.2.2 极小极大问题中的旋转方向

普通最小化问题的梯度通常指向下降方向。极小极大博弈的梯度场中还可能存在旋转分量。考虑最简单的双线性问题

<div class="display-equation">
$$
\min_\theta
\max_\phi
L(\theta,\phi)
=
\theta\phi.
$$
</div>

唯一鞍点是 $(0,0)$。连续时间梯度下降--上升满足

<div class="display-equation">
$$
\dot\theta
=
-\frac{\partial L}{\partial\theta}
=
-\phi,
$$
</div>

<div class="display-equation">
$$
\dot\phi
=
\frac{\partial L}{\partial\phi}
=
\theta.
$$
</div>

计算半径平方，得到

<div class="display-equation">
$$
\begin{aligned}
\frac{d}{dt}
(\theta^2+\phi^2)
&=
2\theta\dot\theta
+
2\phi\dot\phi\\
&=
-2\theta\phi
+
2\phi\theta\\
&=0.
\end{aligned}
$$
</div>

所以连续轨迹不会靠近鞍点，而是沿圆周不断旋转。

同步离散更新为

<div class="display-equation">
$$
\begin{pmatrix}
\theta_{t+1}\\
\phi_{t+1}
\end{pmatrix}
=
\begin{pmatrix}
1&-\eta\\
\eta&1
\end{pmatrix}
\begin{pmatrix}
\theta_t\\
\phi_t
\end{pmatrix}.
$$
</div>

该更新矩阵的特征值为

<div class="display-equation">
$$
\lambda_\pm
=
1\pm i\eta,
$$
</div>

其模长为

<div class="display-equation">
$$
|\lambda_\pm|
=
\sqrt{1+\eta^2}
>1.
$$
</div>

等价地，直接计算可得

<div class="display-equation">
$$
\theta_{t+1}^2+\phi_{t+1}^2
=
(1+\eta^2)
(\theta_t^2+\phi_t^2).
$$
</div>

离散轨迹不仅不收敛，反而螺旋式向外发散。

这个玩具问题说明，GAN 的振荡并不只是损失非凸或学习率选择失误。即使目标是最简单的双线性函数，普通梯度下降--上升也会因为反对称 Jacobian 产生旋转。

一种经典修正是 extragradient。它先用当前梯度预测一个中间点

<div class="display-equation">
$$
\widetilde\theta
=
\theta-\eta\phi,
\qquad
\widetilde\phi
=
\phi+\eta\theta,
$$
</div>

再使用中间点的梯度更新原参数

<div class="display-equation">
$$
\theta^+
=
\theta-\eta\widetilde\phi,
\qquad
\phi^+
=
\phi+\eta\widetilde\theta.
$$
</div>

代入后得到

<div class="display-equation">
$$
\begin{pmatrix}
\theta^+\\
\phi^+
\end{pmatrix}
=
\begin{pmatrix}
1-\eta^2&-\eta\\
\eta&1-\eta^2
\end{pmatrix}
\begin{pmatrix}
\theta\\
\phi
\end{pmatrix}.
$$
</div>

其特征值模长为

<div class="display-equation">
$$
\sqrt{(1-\eta^2)^2+\eta^2}
=
\sqrt{1-\eta^2+\eta^4}.
$$
</div>

当 $0<\eta<1$ 时，该值小于一，因此双线性博弈中的轨迹会收缩。现代对抗优化中的 optimistic gradient、extragradient 和两时间尺度方法，都在不同程度上试图消除这种旋转分量。

7.2.3 模式坍塌与生成映射的几何结构

模式坍塌是指不同潜变量被映射到相同或少数几个输出区域：

<div class="display-equation">
$$
G_\theta(z_1)
\approx
G_\theta(z_2)
\approx\cdots\approx
x_0.
$$
</div>

即使 $z_1,z_2$ 相距很远，生成样本仍几乎相同。生成器可能输出非常清晰的人脸，却总是在生成相似的少数人脸。

必须先说明，模式坍塌不是理想 JS 目标的全局最优解。若模型容量无限、判别器始终达到最优、优化最终到达全局平衡，则定理 7.1.2 仍然要求

<div class="display-equation">
$$
P_g=P_{\mathrm{data}}.
$$
</div>

坍塌发生在有限容量与局部交替优化的动力学中。生成器每一步只面对当前判别器。若某个模式暂时获得较高判别分数，把更多潜变量推向该模式能够立即改善生成器损失；缺失的其他模式要等判别器重新学会利用这种缺失后，才会产生反向压力。生成器和判别器便可能在若干模式之间来回追逐，而不是同时覆盖所有模式。

生成器的局部几何由 Jacobian

<div class="display-equation">
$$
J_G(z)
=
\frac{\partial G_\theta(z)}
{\partial z}
$$
</div>

描述。若其奇异值为

<div class="display-equation">
$$
\sigma_1(z)
\geq
\sigma_2(z)
\geq\cdots\geq0,
$$
</div>

那么这些奇异值表示潜空间不同方向在数据空间中被放大或压缩的程度。当许多奇异值接近零时，潜空间中的大体积区域会被压入低维集合，输出多样性随之减少。

<div class="display-equation">
$$
\operatorname{rank}J_G(z)
\ll d_z
$$
</div>

可以被视为局部坍塌的一个信号。

还有一个更基本的拓扑障碍。若潜变量支持集 $\mathcal Z$ 是连通的，且生成器 $G$ 连续，则其像集

<div class="display-equation">
$$
G(\mathcal Z)
$$
</div>

也必然连通。证明只需使用连续映射保持连通性。若真实数据的理想支持集由多个彼此分离的部分组成，那么连续确定性生成器无法在集合意义上精确得到这些互不连通的部分，而不在它们之间产生某种连接路径。

实际神经网络可以把连接路径压得极细，使其概率质量很小，从而近似多模态分布；但这仍然说明，潜空间拓扑、生成器连续性与数据支持集结构之间可能存在天然错配。模式坍塌并不只是一项优化技巧没调好，它还与生成模型的参数化方式有关。

7.2.4 判别器过拟合与实例噪声

训练中使用的是有限样本经验目标

<div class="display-equation">
$$
\widehat V(D,G)
=
\frac1n
\sum_{i=1}^n
\log D(x_i)
+
\frac1m
\sum_{j=1}^m
\log(1-D(G(z_j))).
$$
</div>

若判别器容量过大，它可以记住有限的真实训练样本，并把当前生成样本全部判为假，而不必学习两个总体分布之间可泛化的差异。此时经验判别准确率很高，生成器收到的却可能只是围绕有限样本形成的尖锐梯度。

一个早期而有效的思路是对真实样本和生成样本同时加入噪声：

<div class="display-equation">
$$
\widetilde x_r
=
x_r+\xi,
\qquad
\widetilde x_g
=
x_g+\xi',
$$
</div>

其中

<div class="display-equation">
$$
\xi,\xi'
\sim
\mathcal N(0,\sigma^2I).
$$
</div>

加噪后的分布是卷积

<div class="display-equation">
$$
P_r^{(\sigma)}
=
P_r*\mathcal N(0,\sigma^2I),
\qquad
P_g^{(\sigma)}
=
P_g*\mathcal N(0,\sigma^2I).
$$
</div>

只要 $\sigma>0$，高斯卷积通常使两个分布获得处处为正的平滑密度，从而缓解支集完全分离。随着训练进行，再逐渐减小 $\sigma$，相当于先比较两个平滑分布，再逐步恢复细节。这种思路与后来的扩散模型存在某种思想上的呼应：噪声可以把难以比较的低维数据流形变成更平滑、更容易学习的密度。

DCGAN 则从网络结构改善训练条件。生成器用转置卷积逐层放大空间分辨率，判别器用带步长卷积逐层压缩；生成器内部使用 ReLU，输出使用 Tanh，判别器使用 LeakyReLU，使负区间仍保留梯度；批量归一化控制中间激活尺度；Adam 的动量和自适应缩放缓解高噪声梯度。它没有改变 GAN 的理论目标，却证明了合理的架构约束能够显著改善极小极大博弈的数值性质。

7.2.5 一个两模式的追逐例子

设真实数据只有两个模式：

<div class="display-equation">
$$
P_{\mathrm{data}}
=
\frac12\delta_{-1}
+
\frac12\delta_{1}.
$$
</div>

为了突出动力学问题，假设生成器在每一轮只能把全部潜变量映射到一个点 $g\in{-1,1}$。这个生成器当然容量不足，无法同时表达两个模式，但它能清楚展示 best-response 追逐。

若当前生成器只输出 $-1$，则

<div class="display-equation">
$$
P_g=\delta_{-1}.
$$
</div>

判别器很快会发现：真实数据在 $1$ 处仍有质量，而生成器从不产生 $1$。因此 $1$ 会获得较高真实分数，$-1$ 的真假证据则相互混合。生成器下一步最有利的反应可能是全部转向 $1$。

转向以后，

<div class="display-equation">
$$
P_g=\delta_1.
$$
</div>

此时判别器又发现 $-1$ 成为缺失模式，生成器再被吸引回 $-1$。于是训练轨迹可能表现为

<div class="display-equation">
$$
-1
\longrightarrow
1
\longrightarrow
-1
\longrightarrow
1
\longrightarrow\cdots.
$$
</div>

这个例子中的根本问题一部分来自生成器容量不足，但它同时揭示了交替 best response 的短视性：当前判别器只对当前缺失模式产生高奖励，生成器若大幅更新，就可能完全抛弃上一轮已覆盖的模式。

实际神经网络的输出不是两个离散点，生成器容量也更大，但大步长更新、有限 batch 和滞后的判别器仍会产生相似现象。所谓“模式坍塌后又突然切换到另一个模式”，常常正是这种追逐动力学的连续高维版本。

7.2.6 怎样理解判别器准确率

训练日志中，判别器准确率很高不一定是好事，也不一定是坏事。若生成器确实很差，高准确率只是忠实反映两个分布相差很大；若判别器已经记住有限训练样本，高准确率可能没有任何可泛化价值；若判别器被正则化得过强，接近 $50%$ 的准确率也可能只是判别器能力不足。

在理想二分类问题中，最优错误率与总变差距离有关。对平衡类别先验，Bayes 最小分类错误满足

<div class="display-equation">
$$
\varepsilon^\star
=
\frac12
\left(
1-
\operatorname{TV}
(P_{\mathrm{data}},P_g)
\right),
$$
</div>

其中

<div class="display-equation">
$$
\operatorname{TV}(P,Q)
=
\sup_A
|P(A)-Q(A)|.
$$
</div>

若两个分布相同，则 $\operatorname{TV}=0$，最优错误率为 $1/2$；若两个分布支集完全分离，则 $\operatorname{TV}=1$，最优错误率为零。

但实际判别器只在有限函数类中训练，所以观测到的错误率是函数类、优化程度、训练样本数量和正则化共同作用的结果。判别准确率只能作为诊断信号之一，不能单独充当生成质量的可靠度量。

7.2.7 生成质量为什么不能只看训练损失

监督学习中，验证集损失通常与模型性能有直接联系。GAN 的训练损失却很难横向解释，因为判别器本身也在变化。同一个生成器，在弱判别器面前可能得到很小损失，在强判别器面前可能得到很大损失；两个训练时刻的损失值并不是在同一个评价函数下计算的。

原始 GAN 到达理想平衡时，

<div class="display-equation">
$$
D(x)=\frac12,
$$
</div>

所以

<div class="display-equation">
$$
\mathbb E_{P_{\mathrm{data}}}\log D(x)
+
\mathbb E_{P_g}\log(1-D(x))
=
-\log4.
$$
</div>

但实际训练中，目标接近 $-\log4$ 可能有完全不同的原因。它可能表示两个分布接近，也可能表示判别器尚未学会分类，甚至可能表示判别器被正则化得过弱。因此需要在独立于当前判别器的特征空间中评价生成样本。

Fréchet Inception Distance 先用固定特征提取器把真实样本和生成样本映射到特征空间，再分别用高斯分布近似：

<div class="display-equation">
$$
h(x_r)
\approx
\mathcal N(\mu_r,\Sigma_r),
$$
</div>

<div class="display-equation">
$$
h(x_g)
\approx
\mathcal N(\mu_g,\Sigma_g).
$$
</div>

两个高斯分布的平方 Wasserstein-2 距离给出

<div class="display-equation">
$$
\begin{aligned}
\operatorname{FID}
=&
\|\mu_r-\mu_g\|_2^2\\
&+
\operatorname{Tr}
\left(
\Sigma_r+\Sigma_g
-
2
\left(
\Sigma_r^{1/2}
\Sigma_g
\Sigma_r^{1/2}
\right)^{1/2}
\right).
\end{aligned}
$$
</div>

第一项比较特征均值，第二项比较协方差结构。若生成器只覆盖少数模式，协方差往往过小；若图像质量差，均值和协方差都会偏离真实特征分布。

FID 仍然只是近似。它把复杂特征分布压成均值和协方差，无法区分所有高阶差异；有限样本估计还存在偏差；特征提取器的训练域也会影响评价。但它至少使用固定的外部表示，而不是使用正在与生成器共同变化的判别器。

生成模型还存在质量与覆盖之间的矛盾。只生成少数极其逼真的样本，单样本质量很高，但分布覆盖很差；生成多样性很强却包含许多不真实样本，又会降低精度。可以把生成分布的评价类比为分类中的 precision 与 recall：precision 关注生成样本有多少落在真实数据流形附近，recall 关注真实数据流形有多少被生成分布覆盖。

模式坍塌主要伤害 recall，模糊或带伪影的样本主要伤害 precision。单一指标往往把二者混在一起，所以实际分析需要同时观察样本质量、覆盖程度和训练动力学。

7.3 Wasserstein GAN

7.3.1 从分布是否重叠转向概率质量搬运

原始 GAN 通过分类器比较分布。Wasserstein GAN 改用最优传输距离，使分布差异具有明确的空间几何意义。

设真实分布为 $P_r$，生成分布为 $P_g$。一个 coupling $\gamma$ 是定义在样本对 $(x,y)$ 上的联合分布，其两个边缘分别为 $P_r$ 和 $P_g$。所有合法 coupling 构成

<div class="display-equation">
$$
\Pi(P_r,P_g).
$$
</div>

若把 $P_r$ 看成一堆土，把 $P_g$ 看成目标土堆，$\gamma$ 决定从位置 $x$ 向位置 $y$ 搬运多少概率质量。以欧氏距离为单位搬运成本，一阶 Wasserstein 距离定义为

<div class="display-equation">
$$
W_1(P_r,P_g)
=
\inf_{\gamma\in\Pi(P_r,P_g)}
\mathbb E_{(x,y)\sim\gamma}
\bigl[
\|x-y\|_2
\bigr].
$$
</div>

对于

<div class="display-equation">
$$
P_r=\delta_0,
\qquad
P_\theta=\delta_\theta,
$$
</div>

唯一 coupling 把点 $0$ 的全部质量搬到点 $\theta$，所以

<div class="display-equation">
$$
W_1(\delta_0,\delta_\theta)
=
|\theta|.
$$
</div>

这与 JS 散度形成鲜明对比：

<div class="display-equation">
$$
D_{\mathrm{JS}}(\delta_0\|\delta_\theta)
=
\log2,
\qquad
\theta\neq0,
$$
</div>

而

<div class="display-equation">
$$
W_1(\delta_0,\delta_\theta)
\to0
\qquad
\text{当 }\theta\to0.
$$
</div>

Wasserstein 距离不仅知道两个分布尚未重合，还知道它们应该移动多远。

在一维中，若 $F_P^{-1}$ 和 $F_Q^{-1}$ 是分位数函数，则

<div class="display-equation">
$$
W_1(P,Q)
=
\int_0^1
\left|
F_P^{-1}(u)
-
F_Q^{-1}(u)
\right|du.
$$
</div>

它表示用同一分位数 $u$ 对两个分布进行配对。小分位数与小分位数对应，大分位数与大分位数对应；这种共单调配对是一维绝对距离下的最优运输方案。

7.3.2 生成器参数变化为什么会带来连续的 Wasserstein 变化

Wasserstein 距离与生成器参数之间的连续性可以通过一个简单 coupling 看出来。取同一个潜变量

<div class="display-equation">
$$
z\sim P_z,
$$
</div>

并同时构造

<div class="display-equation">
$$
x=G_\theta(z),
\qquad
y=G_{\theta'}(z).
$$
</div>

$(x,y)$ 的联合分布是 $P_\theta$ 与 $P_{\theta'}$ 的一个合法 coupling，所以根据下确界定义，

<div class="display-equation">
$$
W_1(P_\theta,P_{\theta'})
\leq
\mathbb E_z
\bigl[
\|G_\theta(z)-G_{\theta'}(z)\|_2
\bigr].
$$
</div>

若存在可积函数 $L(z)$，使

<div class="display-equation">
$$
\|G_\theta(z)-G_{\theta'}(z)\|_2
\leq
L(z)
\|\theta-\theta'\|_2,
$$
</div>

则

<div class="display-equation">
$$
W_1(P_\theta,P_{\theta'})
\leq
\mathbb E[L(z)]
\|\theta-\theta'\|_2.
$$
</div>

因此，生成器参数的小变化会导致生成分布在 Wasserstein 距离下的小变化。这个性质并不保证优化一定容易，却说明目标至少不会像 Dirac 分布的 JS 散度那样在几乎所有非零参数处保持常数。

7.3.3 Kantorovich--Rubinstein 对偶

直接在所有 coupling 上求下确界通常不可行。对偶理论把运输问题改写成函数优化：

<div class="display-equation">
$$
W_1(P_r,P_g)
=
\sup_{\|f\|_{\mathrm{Lip}}\leq1}
\left\{
\mathbb E_{x\sim P_r}[f(x)]
-
\mathbb E_{y\sim P_g}[f(y)]
\right\}.
$$
</div>

先证明容易的一半。对任意 $1$-Lipschitz 函数 $f$ 和任意 coupling $\gamma\in\Pi(P_r,P_g)$，有

<div class="display-equation">
$$
\begin{aligned}
\mathbb E_{P_r}[f]
-
\mathbb E_{P_g}[f]
&=
\mathbb E_{(x,y)\sim\gamma}
[f(x)-f(y)]\\
&\leq
\mathbb E_\gamma
|f(x)-f(y)|\\
&\leq
\mathbb E_\gamma
\|x-y\|_2.
\end{aligned}
$$
</div>

对所有 $1$-Lipschitz 函数取上确界，再对所有 coupling 取下确界，得到

<div class="display-equation">
$$
\sup_{\|f\|_{\mathrm{Lip}}\leq1}
\left(
\mathbb E_{P_r}f
-
\mathbb E_{P_g}f
\right)
\leq
W_1(P_r,P_g).
$$
</div>

Kantorovich--Rubinstein 定理的深刻之处在于，在适当条件下等号成立。也就是说，最优搬运成本可以完全由一个最优 Lipschitz 势函数的期望差恢复。

WGAN 用神经网络 $f_w$ 近似这个势函数。它不再输出概率，因此通常称为 critic 而不是 discriminator。critic 希望最大化

<div class="display-equation">
$$
L_C(w;\theta)
=
\mathbb E_{x\sim P_r}
[f_w(x)]
-
\mathbb E_{z\sim P_z}
[f_w(G_\theta(z))],
$$
</div>

同时必须近似满足

<div class="display-equation">
$$
\|f_w\|_{\mathrm{Lip}}
\leq1.
$$
</div>

生成器最小化估计的 Wasserstein 距离，等价于最小化

<div class="display-equation">
$$
L_G^{\mathrm W}
=
-
\mathbb E_z
[f_w(G_\theta(z))].
$$
</div>

固定 critic 后，利用链式法则得到

<div class="display-equation">
$$
\nabla_\theta
L_G^{\mathrm W}
=
-
\mathbb E_z
\left[
J_{G_\theta}(z)^\top
\nabla_xf_w(x)
\big|_{x=G_\theta(z)}
\right].
$$
</div>

$\nabla_xf_w$ 直接在数据空间中提供提高 critic 分数的方向，不再经过 sigmoid 概率的饱和因子。这是 WGAN 梯度通常更加平滑的重要原因。

7.3.4 为什么 Lipschitz 约束不可缺少

若不限制 $f$，只要存在某个函数满足

<div class="display-equation">
$$
\mathbb E_{P_r}f
-
\mathbb E_{P_g}f
>0,
$$
</div>

把它乘以任意常数 $c$，便有

<div class="display-equation">
$$
\mathbb E_{P_r}[cf]
-
\mathbb E_{P_g}[cf]
=
c
\left(
\mathbb E_{P_r}f
-
\mathbb E_{P_g}f
\right),
$$
</div>

上确界会变成无穷大。Lipschitz 约束给 critic 的斜率设置统一单位，使分数差具有距离尺度。

设 $f$ 在凸区域内可微，并且

<div class="display-equation">
$$
\|\nabla f(x)\|_2
\leq1
$$
</div>

处处成立。对任意 $x,y$，沿线段积分：

<div class="display-equation">
$$
\begin{aligned}
f(y)-f(x)
&=
\int_0^1
\nabla f(x+t(y-x))^\top
(y-x)dt\\
&\leq
\int_0^1
\|\nabla f(x+t(y-x))\|_2
\|y-x\|_2dt\\
&\leq
\|y-x\|_2.
\end{aligned}
$$
</div>

交换 $x,y$ 后可得

<div class="display-equation">
$$
|f(y)-f(x)|
\leq
\|y-x\|_2,
$$
</div>

因此 $f$ 是 $1$-Lipschitz。

原始 WGAN 通过权重裁剪近似限制函数类：

<div class="display-equation">
$$
w_i
\leftarrow
\operatorname{clip}(w_i,-c,c).
$$
</div>

但参数小不等于函数恰好具有合适的 Lipschitz 常数。裁剪阈值过小会把 critic 压成近似线性函数，限制表达能力；阈值过大又不足以控制斜率。网络还可能把大量权重推到裁剪边界，使优化变得僵硬。

7.3.5 WGAN-GP 的梯度惩罚

WGAN-GP 不直接裁剪权重，而是检查 critic 对输入的梯度。抽取真实样本 $x_r$、生成样本 $x_g$ 与

<div class="display-equation">
$$
\varepsilon
\sim
\operatorname{Unif}[0,1],
$$
</div>

构造插值点

<div class="display-equation">
$$
\widehat x
=
\varepsilon x_r
+
(1-\varepsilon)x_g.
$$
</div>

梯度惩罚定义为

<div class="display-equation">
$$
L_{\mathrm{GP}}
=
\lambda
\mathbb E_{\widehat x}
\left[
\|\nabla_{\widehat x}f_w(\widehat x)\|_2
-
1
\right]^2.
$$
</div>

若用最小化形式训练 critic，可写成

<div class="display-equation">
$$
L_D
=
\mathbb E_{x_g}[f_w(x_g)]
-
\mathbb E_{x_r}[f_w(x_r)]
+
L_{\mathrm{GP}}.
$$
</div>

为什么惩罚梯度范数偏离一，而不只是惩罚超过一？考虑一对在最优运输计划中相互对应的点 $(x_r,x_g)$。若最优 Kantorovich 势函数在这对点上达到

<div class="display-equation">
$$
f(x_r)-f(x_g)
=
\|x_r-x_g\|_2,
$$
</div>

同时又满足 $1$-Lipschitz，那么沿连接二者的线段，Cauchy--Schwarz 不等式需要近似取等。于是梯度应与运输方向对齐，并且范数接近一。WGAN-GP 正是针对真实样本与生成样本之间可能的运输路径施加这种约束。

它仍然不是全局 Lipschitz 性的严格证明。惩罚只发生在有限采样到的插值点上，数据空间其他区域未必受控。它是一种利用最优运输几何设计出的有效近似。

7.3.6 谱归一化

设一个前馈 critic 为

<div class="display-equation">
$$
f(x)
=
W_L\sigma_{L-1}
\left(
W_{L-1}\cdots
\sigma_1(W_1x)
\right).
$$
</div>

线性映射 $x\mapsto Wx$ 的 Lipschitz 常数是谱范数

<div class="display-equation">
$$
\|W\|_2
=
\sup_{x\neq0}
\frac{\|Wx\|_2}{\|x\|_2}
=
\sigma_{\max}(W).
$$
</div>

若激活函数都是 $1$-Lipschitz，则复合函数满足

<div class="display-equation">
$$
\|f\|_{\mathrm{Lip}}
\leq
\prod_{\ell=1}^L
\|W_\ell\|_2.
$$
</div>

谱归一化把每层权重替换为

<div class="display-equation">
$$
\overline W
=
\frac{W}{\sigma_{\max}(W)}.
$$
</div>

最大奇异值可以通过幂迭代近似。给定单位向量 $u,v$，反复更新

<div class="display-equation">
$$
v
\leftarrow
\frac{W^\top u}{\|W^\top u\|_2},
$$
</div>

<div class="display-equation">
$$
u
\leftarrow
\frac{Wv}{\|Wv\|_2},
$$
</div>

最后用

<div class="display-equation">
$$
\sigma_{\max}(W)
\approx
u^\top Wv
$$
</div>

估计谱范数。

梯度惩罚是在数据附近直接约束函数梯度，谱归一化则从参数矩阵控制全局上界。前者更贴近当前数据几何，但需要额外自动微分；后者开销稳定，却可能给出偏松的乘积上界。二者都是对 Lipschitz 对偶条件的近似实现。

7.3.7 WGAN 解决了什么，又没有解决什么

WGAN 的主要改进是让分布支集不重合时仍然存在连续而有意义的距离信号。critic loss 也往往比原始 GAN 的判别准确率更能反映训练进展。

但实际 critic 只在有限神经网络函数类中优化，Lipschitz 约束又是近似的，所以

<div class="display-equation">
$$
\mathbb E_{P_r}f_w
-
\mathbb E_{P_g}f_w
$$
</div>

通常只是 Wasserstein 距离的代理量，而不是精确无偏估计。

WGAN 也不从理论上彻底消除模式坍塌。若 critic 优化不足、生成器容量受限或交替动力学失衡，生成器仍可能忽略部分模式。更合适的距离改善了梯度几何，却没有自动解决所有非凸博弈问题。



7.3.8 一个离散最优传输例子

考虑两个离散分布

<div class="display-equation">
$$
P
=
\frac12\delta_0
+
\frac12\delta_2,
$$
</div>

<div class="display-equation">
$$
Q
=
\frac12\delta_1
+
\frac12\delta_3.
$$
</div>

令 $\gamma_{ij}$ 表示从 $P$ 的第 $i$ 个位置向 $Q$ 的第 $j$ 个位置搬运的质量。运输矩阵满足边缘约束

<div class="display-equation">
$$
\gamma_{11}+\gamma_{12}
=
\frac12,
\qquad
\gamma_{21}+\gamma_{22}
=
\frac12,
$$
</div>

<div class="display-equation">
$$
\gamma_{11}+\gamma_{21}
=
\frac12,
\qquad
\gamma_{12}+\gamma_{22}
=
\frac12,
$$
</div>

并且 $\gamma_{ij}\geq0$。距离成本矩阵为

<div class="display-equation">
$$
C
=
\begin{pmatrix}
|0-1|&|0-3|\\
|2-1|&|2-3|
\end{pmatrix}
=
\begin{pmatrix}
1&3\\
1&1
\end{pmatrix}.
$$
</div>

运输成本是

<div class="display-equation">
$$
\langle C,\gamma\rangle
=
\gamma_{11}
+
3\gamma_{12}
+
\gamma_{21}
+
\gamma_{22}.
$$
</div>

最优方案是把 $0$ 处的质量搬到 $1$，把 $2$ 处的质量搬到 $3$：

<div class="display-equation">
$$
\gamma^\star
=
\begin{pmatrix}
1/2&0\\
0&1/2
\end{pmatrix}.
$$
</div>

于是

<div class="display-equation">
$$
W_1(P,Q)
=
\frac12\cdot1
+
\frac12\cdot1
=
1.
$$
</div>

若错误地把 $0$ 搬到 $3$，再把 $2$ 搬到 $1$，成本会变成

<div class="display-equation">
$$
\frac12\cdot3
+
\frac12\cdot1
=
2.
$$
</div>

这个例子说明 coupling 不是普通的独立配对。它需要在满足两个边缘分布的前提下，寻找全局最便宜的匹配方式。

从对偶角度，取函数

<div class="display-equation">
$$
f(x)=-x.
$$
</div>

它是 $1$-Lipschitz，并且

<div class="display-equation">
$$
\mathbb E_Pf
=
-\frac12(0+2)
=
-1,
$$
</div>

<div class="display-equation">
$$
\mathbb E_Qf
=
-\frac12(1+3)
=
-2.
$$
</div>

所以

<div class="display-equation">
$$
\mathbb E_Pf-\mathbb E_Qf
=
1.
$$
</div>

它恰好达到原始运输成本，直接验证了这个例子中的强对偶。

7.4 条件生成与图像翻译

7.4.1 条件 GAN 的分布含义

原始 GAN 学习的是一个总体分布

<div class="display-equation">
$$
P_g(x)
\approx
P_{\mathrm{data}}(x).
$$
</div>

这只能保证生成样本整体看起来真实，却无法控制生成什么。若希望指定类别、文本描述、属性或另一张输入图像，需要引入条件变量 $c$。

生成器变为

<div class="display-equation">
$$
x_g
=
G_\theta(z,c),
$$
</div>

判别器同时观察样本和条件：

<div class="display-equation">
$$
D_\phi(x,c).
$$
</div>

条件 GAN 的目标函数为

<div class="display-equation">
$$
\begin{aligned}
\min_\theta\max_\phi
V_c(D,G)
=&
\mathbb E_{(x,c)\sim P_{\mathrm{data}}}
[\log D_\phi(x,c)]\\
&+
\mathbb E_{c\sim P_c,\;z\sim P_z}
[\log(1-D_\phi(G_\theta(z,c),c))].
\end{aligned}
$$
</div>

条件不能只送给生成器。若判别器不知道 $c$，它只能判断图片是否真实，无法判断图片是否符合条件。假设条件要求生成数字 $7$，生成器输出一张非常逼真的数字 $3$，无条件判别器仍可能给出很高分数。

在条件分布 $P_c$ 对真实样本和生成样本相同的情况下，固定生成器后，最优条件判别器满足

<div class="display-equation">
$$
D^\star(x,c)
=
\frac{p_{\mathrm{data}}(x\mid c)}
{p_{\mathrm{data}}(x\mid c)+p_g(x\mid c)}.
$$
</div>

证明与无条件情形完全相同，只需固定 $c$ 后逐点优化。将其代回目标，得到

<div class="display-equation">
$$
V_c(D^\star,G)
=
-\log4
+
2\mathbb E_{c\sim P_c}
\left[
D_{\mathrm{JS}}
\bigl(
P_{\mathrm{data}}(\cdot\mid c)
\|
P_g(\cdot\mid c)
\bigr)
\right].
$$
</div>

因此，条件 GAN 不是简单地在生成器输入后面多拼一个向量，而是在同时拟合一整族条件分布：

<div class="display-equation">
$$
P_g(\cdot\mid c)
\approx
P_{\mathrm{data}}(\cdot\mid c),
\qquad
\forall c.
$$
</div>

条件越丰富，模型需要学习的分布族越复杂。类别标签只把总体分布拆成有限个类别分布；文本条件则可能对应几乎无限多种语义约束。

7.4.2 pix2pix 中重构损失的作用

在成对图像翻译中，训练数据是配对样本

<div class="display-equation">
$$
(x,y)\sim P_{\mathrm{pair}},
$$
</div>

其中 $x$ 可能是边缘图，$y$ 是与之对应的真实照片。生成器输出 $G(x)$，判别器判断图像对 $(x,y)$ 是否来自真实配对。

条件对抗损失为

<div class="display-equation">
$$
\begin{aligned}
L_{\mathrm{cGAN}}(G,D)
=&
\mathbb E_{x,y}
[\log D(x,y)]\\
&+
\mathbb E_x
[\log(1-D(x,G(x)))].
\end{aligned}
$$
</div>

只使用这个损失，生成器只需要让输出属于目标图像域，却不一定严格保持输入图像的具体结构。对某张边缘图而言，输出另一张结构合理、纹理真实但内容不同的照片，也可能骗过判别器。

pix2pix 因此加入逐样本重构损失

<div class="display-equation">
$$
L_{L_1}(G)
=
\mathbb E_{x,y}
\|y-G(x)\|_1.
$$
</div>

完整目标为

<div class="display-equation">
$$
G^\star
=
\arg\min_G\max_D
L_{\mathrm{cGAN}}(G,D)
+
\lambda L_{L_1}(G).
$$
</div>

$L_1$ 项负责保持输入与输出之间的结构对应，对抗项负责让输出的局部统计具有真实感。两者的作用不能互相替代。

为什么不只用 $L_2$ 重构损失？对固定输入 $x$，若可能的真实输出 $Y$ 具有条件分布 $P(Y\mid x)$，最小化

<div class="display-equation">
$$
\mathbb E
\bigl[
\|Y-a\|_2^2
\mid x
\bigr]
$$
</div>

的最优点是条件均值

<div class="display-equation">
$$
a^\star
=
\mathbb E[Y\mid x].
$$
</div>

若一个边缘图可能对应多种合理纹理，条件均值会把这些纹理平均起来，产生模糊结果。标量情形下，最小化

<div class="display-equation">
$$
\mathbb E
\bigl[
|Y-a|
\mid x
\bigr]
$$
</div>

的最优点是条件中位数，因此 $L_1$ 比 $L_2$ 更不容易受极端像素影响，但当条件分布本身高度多模态时，它同样无法单独选择一个清晰模式。

对抗损失的作用正是在这里出现。判别器不要求输出接近所有可能目标的平均值，而要求输出落在真实图像流形附近。因此，重构损失提供内容约束，对抗损失提供分布约束。

pix2pix 常使用 PatchGAN。判别器不是为整张图输出单个分数，而是在许多局部窗口上输出真假判断。若第 $k$ 个局部窗口判别分数为 $D_k(x,y)$，整体判别损失可以理解为

<div class="display-equation">
$$
L_D
=
\frac1K
\sum_{k=1}^K
\ell_{\mathrm{BCE}}
\bigl(D_k(x,y),1\bigr)
+
\frac1K
\sum_{k=1}^K
\ell_{\mathrm{BCE}}
\bigl(D_k(x,G(x)),0\bigr).
$$
</div>

这种设计把判别器的注意力放在局部纹理、边缘和高频统计上；全局结构则主要由输入条件和重构项维持。

7.4.3 CycleGAN 与映射不可辨识性

许多图像翻译任务没有成对数据。我们只有

<div class="display-equation">
$$
x\sim P_X,
\qquad
y\sim P_Y,
$$
</div>

却不知道哪个 $x$ 应该对应哪个 $y$。只训练映射 $G\to Y$ 的对抗损失，只能要求

<div class="display-equation">
$$
G_\#P_X
\approx
P_Y.
$$
</div>

满足这一条件的映射通常不唯一。生成器甚至可以忽略输入内容，把不同的 $x$ 映射到少数几个真实感很强的 $Y$ 域样本。

CycleGAN 同时学习反向映射

<div class="display-equation">
$$
F:Y\to X,
$$
</div>

并要求往返映射尽量恢复原样本：

<div class="display-equation">
$$
F(G(x))
\approx x,
$$
</div>

<div class="display-equation">
$$
G(F(y))
\approx y.
$$
</div>

循环一致性损失为

<div class="display-equation">
$$
\begin{aligned}
L_{\mathrm{cyc}}(G,F)
=&
\mathbb E_{x\sim P_X}
\|F(G(x))-x\|_1\\
&+
\mathbb E_{y\sim P_Y}
\|G(F(y))-y\|_1.
\end{aligned}
$$
</div>

完整目标包含两个方向的对抗损失：

<div class="display-equation">
$$
\begin{aligned}
L(G,F,D_X,D_Y)
=&
L_{\mathrm{GAN}}(G,D_Y;X,Y)\\
&+
L_{\mathrm{GAN}}(F,D_X;Y,X)\\
&+
\lambda_{\mathrm{cyc}}
L_{\mathrm{cyc}}(G,F).
\end{aligned}
$$
</div>

循环一致性显著缩小了可行映射集合，却不能保证恢复人类期望的语义对应。这个问题可以用一个简单命题看清。

命题 7.4.1（循环一致性不保证语义唯一）　若存在任意可逆映射 $T\to Y$，使

<div class="display-equation">
$$
T_\#P_X=P_Y,
$$
</div>

则取

<div class="display-equation">
$$
G=T,
\qquad
F=T^{-1},
$$
</div>

可以同时使两个方向的分布匹配成立，并使循环损失为零。

证明只需注意

<div class="display-equation">
$$
F(G(x))
=
T^{-1}(T(x))
=
x,
$$
</div>

<div class="display-equation">
$$
G(F(y))
=
T(T^{-1}(y))
=
y.
$$
</div>

因此，只要 $T$ 是一个把 $P_X$ 推到 $P_Y$ 的双射，无论它是否符合人类语义，都能成为理想解。比如，一个映射完全可以系统地交换两类对象，只要反向映射再交换回来，循环一致性就不会发现问题。

这说明无配对图像翻译存在不可辨识性。对抗损失确定输出分布，循环损失要求映射可逆，却仍不足以决定“哪一个 $x$ 应该对应哪一个 $y$”。实际模型依赖卷积网络的局部归纳偏置、身份损失、颜色约束和数据结构，偏向较自然的映射。

常见身份损失写成

<div class="display-equation">
$$
L_{\mathrm{id}}(G,F)
=
\mathbb E_{y\sim P_Y}
\|G(y)-y\|_1
+
\mathbb E_{x\sim P_X}
\|F(x)-x\|_1.
$$
</div>

它要求已经属于目标域的样本不应被大幅修改，从而抑制无意义的颜色或几何变化。但它同样只是额外偏置，并不能彻底解决无配对映射的语义唯一性。

7.4.4 条件信息怎样进入判别器

最简单的条件判别器把 $x$ 的特征与条件嵌入直接拼接，再送入后续网络。但在类别条件生成中，一种更有结构的方式是 projection discriminator。

设判别器先提取样本特征

<div class="display-equation">
$$
h(x)\in\mathbb R^d,
$$
</div>

条件 $c$ 对应可学习嵌入

<div class="display-equation">
$$
v_c\in\mathbb R^d.
$$
</div>

判别分数写成

<div class="display-equation">
$$
s(x,c)
=
u^\top h(x)
+
v_c^\top h(x).
$$
</div>

第一项判断样本整体是否真实，第二项判断样本特征是否与条件相容。若 $x$ 是一张真实数字图片，但 $c$ 给出错误类别，第一项可能很高，第二项却会因为条件嵌入与图像特征不匹配而降低总分。

projection 形式还可以理解为条件特征空间中的双线性兼容函数。它要求同一条件下的真实样本特征与 $v_c$ 方向对齐，使条件信息不只是作为一个额外输入被网络“自行处理”，而是直接进入真假评分的核心内积。

7.5 对抗样本与鲁棒优化

7.5.1 从局部线性化推导 FGSM

对抗思想并不限于生成模型。对于分类器，攻击者可以不训练另一个判别网络，而是在输入附近主动寻找最坏扰动。

设模型为 $f_\theta$，样本为 $(x,y)$，损失为

<div class="display-equation">
$$
L(\theta;x,y)
=
\ell(f_\theta(x),y).
$$
</div>

攻击者在约束集合中寻找

<div class="display-equation">
$$
\max_{\|\delta\|_p\leq\varepsilon}
L(\theta;x+\delta,y).
$$
</div>

若扰动很小，对输入做一阶 Taylor 展开：

<div class="display-equation">
$$
L(\theta;x+\delta,y)
\approx
L(\theta;x,y)
+
\nabla_xL(\theta;x,y)^\top\delta.
$$
</div>

记

<div class="display-equation">
$$
g
=
\nabla_xL(\theta;x,y).
$$
</div>

在 $\ell_\infty$ 约束下，

<div class="display-equation">
$$
|\delta_i|\leq\varepsilon.
$$
</div>

线性项为

<div class="display-equation">
$$
g^\top\delta
=
\sum_i g_i\delta_i.
$$
</div>

每个坐标可以独立最大化。当 $g_i>0$ 时取 $\delta_i=\varepsilon$，当 $g_i<0$ 时取 $\delta_i=-\varepsilon$。于是

<div class="display-equation">
$$
\delta^\star
=
\varepsilon
\operatorname{sign}(g).
$$
</div>

得到快速梯度符号法

<div class="display-equation">
$$
x_{\mathrm{adv}}
=
x
+
\varepsilon
\operatorname{sign}
\bigl(
\nabla_xL(\theta;x,y)
\bigr).
$$
</div>

一阶近似下的最大损失增量为

<div class="display-equation">
$$
\max_{\|\delta\|_\infty\leq\varepsilon}
g^\top\delta
=
\varepsilon
\|g\|_1.
$$
</div>

这解释了高维空间中的累积效应。每个像素只改变很小的 $\varepsilon$，但攻击者让所有坐标都与梯度符号对齐，影响不会相互抵消，而是累积成 $\varepsilon|g|_1$。

更一般地，设 $p$ 与 $q$ 满足

<div class="display-equation">
$$
\frac1p+\frac1q=1.
$$
</div>

由 Hölder 不等式，

<div class="display-equation">
$$
g^\top\delta
\leq
\|g\|_q
\|\delta\|_p
\leq
\varepsilon\|g\|_q.
$$
</div>

并且可以选择合适方向达到等号，所以

<div class="display-equation">
$$
\max_{\|\delta\|_p\leq\varepsilon}
g^\top\delta
=
\varepsilon\|g\|_q.
$$
</div>

对抗脆弱性因此与输入梯度的对偶范数直接相关。

7.5.2 PGD 与约束内层优化

FGSM 只沿当前梯度走一步。若损失曲面高度非线性，一步近似可能不够。Projected Gradient Descent 反复做梯度上升，并把结果投影回允许的扰动集合。

从随机初值

<div class="display-equation">
$$
\delta^{(0)}
\in
\mathcal B_p(\varepsilon)
$$
</div>

开始，迭代

<div class="display-equation">
$$
\delta^{(k+1)}
=
\Pi_{\mathcal B_p(\varepsilon)}
\left[
\delta^{(k)}
+
\alpha
\nabla_\delta
L(\theta;x+\delta^{(k)},y)
\right].
$$
</div>

对于 $\ell_\infty$ 攻击，常写成

<div class="display-equation">
$$
\delta^{(k+1)}
=
\operatorname{clip}_{[-\varepsilon,\varepsilon]}
\left[
\delta^{(k)}
+
\alpha
\operatorname{sign}
\nabla_x
L(\theta;x+\delta^{(k)},y)
\right].
$$
</div>

投影操作保证每一步都满足威胁模型。随机初始化则使攻击从扰动球中的不同位置出发，降低固定起点带来的局部性。

PGD 面对的是非凸内层问题，所以一般不能保证找到全局最大值。但在给定范数球的白盒攻击中，多步 PGD 通常比单步 FGSM 更接近当前模型的局部最坏扰动。

7.5.3 对抗训练的鲁棒目标

对抗训练求解

<div class="display-equation">
$$
\min_\theta
\mathbb E_{(x,y)\sim P_{\mathrm{data}}}
\left[
\max_{\|\delta\|_p\leq\varepsilon}
L(\theta;x+\delta,y)
\right].
$$
</div>

内层攻击者寻找当前参数下最坏的输入，外层模型再降低该最坏损失。它与 GAN 同样是极小极大问题，但对手的含义不同。GAN 的对手学习一个分布判别函数；鲁棒训练的对手直接选择输入扰动。

利用一阶近似，内层目标约为

<div class="display-equation">
$$
L(\theta;x,y)
+
\varepsilon
\|\nabla_xL(\theta;x,y)\|_q.
$$
</div>

所以对抗训练可以近似理解为

<div class="display-equation">
$$
\min_\theta
\mathbb E
\left[
L(\theta;x,y)
+
\varepsilon
\|\nabla_xL(\theta;x,y)\|_q
\right].
$$
</div>

模型不仅要在数据点上损失小，还要让损失在数据点附近变化缓慢。对抗训练因而与梯度正则化、局部 Lipschitz 控制和大间隔分类有直接联系。

在线性二分类器中，这种联系可以精确写出。设分类分数为

<div class="display-equation">
$$
s(x)=w^\top x,
$$
</div>

标签 $y\in{-1,1}$，分类间隔为

<div class="display-equation">
$$
m(x,y)=y\,w^\top x.
$$
</div>

对 $\ell_p$ 扰动，最坏间隔为

<div class="display-equation">
$$
\begin{aligned}
\min_{\|\delta\|_p\leq\varepsilon}
y\,w^\top(x+\delta)
&=
y\,w^\top x
+
\min_{\|\delta\|_p\leq\varepsilon}
(yw)^\top\delta\\
&=
y\,w^\top x
-
\varepsilon
\|w\|_q.
\end{aligned}
$$
</div>

因此，样本在对抗扰动下保持正确分类，需要

<div class="display-equation">
$$
y\,w^\top x
>
\varepsilon
\|w\|_q.
$$
</div>

普通分类只要求间隔为正；鲁棒分类要求间隔大于一个由权重对偶范数决定的安全余量。对 $\ell_\infty$ 扰动，$q=1$，鲁棒性与 $|w|_1$ 直接相关。

7.5.4 威胁模型决定“鲁棒”是什么意思

对抗鲁棒性必须相对于攻击集合定义。一个模型在 $\ell_\infty$ 半径 $\varepsilon$ 下鲁棒，并不意味着它对空间旋转、遮挡、光照变化或语义修改同样鲁棒。

数学上，攻击集合可以写成

<div class="display-equation">
$$
\mathcal U(x)
=
\{x+\delta:\|\delta\|_p\leq\varepsilon\}.
$$
</div>

鲁棒风险为

<div class="display-equation">
$$
R_{\mathrm{rob}}(\theta)
=
\mathbb E_{(x,y)}
\left[
\sup_{x'\in\mathcal U(x)}
\ell(f_\theta(x'),y)
\right].
$$
</div>

若不同类别的扰动集合发生重叠，就可能不存在同时对所有样本鲁棒的分类器。设两个样本 $x_1,x_2$ 标签不同，但存在 $x'$ 满足

<div class="display-equation">
$$
x'
\in
\mathcal U(x_1)
\cap
\mathcal U(x_2).
$$
</div>

那么模型在 $x'$ 上不可能同时输出两个不同标签。鲁棒精度与标准精度之间的张力，有时并不只是优化算法不够好，而是威胁模型本身提出了互相冲突的要求。

7.6 域对抗神经网络

7.6.1 为什么域偏移会让模型失效

设源域分布为 $P_S(x,y)$，目标域分布为 $P_T(x,y)$。模型只在源域上有标签，却要在目标域上使用。即使任务本身相同，只要

<div class="display-equation">
$$
P_S(x)\neq P_T(x),
$$
</div>

源域上学到的特征也可能依赖域特有信息。例如，训练图片全部来自摄影棚，测试图片来自街景；模型可能错误地把背景光线当作类别线索。

域对抗神经网络希望学习一个特征表示，使它既能完成源域任务，又尽量不包含可用于区分源域和目标域的信息。

设特征提取器为

<div class="display-equation">
$$
h=F_{\theta_f}(x),
$$
</div>

任务分类器为

<div class="display-equation">
$$
\widehat y
=
C_{\theta_y}(h),
$$
</div>

域分类器为

<div class="display-equation">
$$
\widehat d
=
D_{\theta_d}(h),
$$
</div>

其中 $d=0$ 表示源域，$d=1$ 表示目标域。

任务损失只在有标签的源域上计算：

<div class="display-equation">
$$
L_y(\theta_f,\theta_y)
=
\mathbb E_{(x_s,y_s)\sim P_S}
\ell_y
\bigl(
C_{\theta_y}(F_{\theta_f}(x_s)),
y_s
\bigr).
$$
</div>

域分类损失同时使用源域和目标域样本：

<div class="display-equation">
$$
L_d(\theta_f,\theta_d)
=
\mathbb E_{x,d}
\ell_d
\bigl(
D_{\theta_d}(F_{\theta_f}(x)),
d
\bigr).
$$
</div>

域分类器希望最小化 $L_d$，准确区分两个域；特征提取器希望最大化 $L_d$，让两个域在特征空间中不可区分。同时，特征提取器还要最小化任务损失。因此目标可以写成

<div class="display-equation">
$$
\min_{\theta_f,\theta_y}
\max_{\theta_d}
\left[
L_y(\theta_f,\theta_y)
-
\lambda
L_d(\theta_f,\theta_d)
\right].
$$
</div>

7.6.2 梯度反转层

梯度反转层在前向传播中什么都不做：

<div class="display-equation">
$$
R_\lambda(h)=h.
$$
</div>

但在反向传播中规定

<div class="display-equation">
$$
\frac{\partial R_\lambda}{\partial h}
=
-\lambda I.
$$
</div>

域分类器按照普通方向更新，最小化 $L_d$；域损失传回特征提取器时，梯度符号被反转，使特征提取器最大化 $L_d$。

特征提取器的总梯度为

<div class="display-equation">
$$
\nabla_{\theta_f}
L_y
-
\lambda
\nabla_{\theta_f}
L_d.
$$
</div>

这个机制把一个极小极大目标嵌入普通反向传播，不需要显式编写两套复杂优化器。

7.6.3 域分类误差与分布差异

若一个足够强的域分类器能够几乎完美地区分源域与目标域，说明两个域的特征分布相差很大。若最优域分类器只能达到随机猜测，说明在当前函数类看来两个域接近不可区分。

常用的 proxy $\mathcal A$-distance 可以写成

<div class="display-equation">
$$
d_{\mathcal A}
=
2(1-2\varepsilon_d),
$$
</div>

其中 $\varepsilon_d$ 是最优域分类器的泛化错误率。当

<div class="display-equation">
$$
\varepsilon_d=\frac12
$$
</div>

时，

<div class="display-equation">
$$
d_{\mathcal A}=0,
$$
</div>

表示两个域难以区分；当 $\varepsilon_d=0$ 时，

<div class="display-equation">
$$
d_{\mathcal A}=2,
$$
</div>

表示域分类完全容易。

域适应理论中的典型风险界具有形式

<div class="display-equation">
$$
R_T(h)
\leq
R_S(h)
+
\frac12
d_{\mathcal H\Delta\mathcal H}
(P_S,P_T)
+
\lambda^\star.
$$
</div>

这里 $R_S$ 是源域风险，$R_T$ 是目标域风险，$d_{\mathcal H\Delta\mathcal H}$ 衡量两个域在假设类中的差异，而

<div class="display-equation">
$$
\lambda^\star
=
\min_{h\in\mathcal H}
\bigl(
R_S(h)+R_T(h)
\bigr)
$$
</div>

表示两个域是否存在共同表现良好的分类器。

DANN 主要试图降低中间的域差异项，却不能自动减小 $\lambda^\star$。若两个域的标签规则本身不同，或者域信息与类别信息不可分离，强行让特征域不变反而可能损害任务性能。这说明域不可辨识不是越强越好，它必须建立在两个域共享任务语义的假设上。

7.7 对抗学习的统一视角

GAN、WGAN、对抗训练和 DANN 表面上解决不同问题，但它们共享一个结构：

<div class="display-equation">
$$
\min_\theta
\max_{a\in\mathcal A}
\mathcal L(\theta,a).
$$
</div>

在 GAN 中，对手 $a$ 是判别器参数，它寻找真实分布与生成分布最明显的差异；在 WGAN 中，对手是受 Lipschitz 约束的 critic，它寻找最大的期望差；在对抗训练中，对手是输入扰动 $\delta$，它寻找模型局部最脆弱的方向；在 DANN 中，对手是域分类器，它寻找源域和目标域仍然可区分的信息。

普通经验风险最小化研究平均情况：

<div class="display-equation">
$$
\min_\theta
\mathbb E_{x\sim P}
\ell(\theta;x).
$$
</div>

对抗学习则主动构造一个会暴露缺陷的环境。更抽象地，可以写成分布鲁棒优化

<div class="display-equation">
$$
\min_\theta
\sup_{Q\in\mathcal U(P)}
\mathbb E_{x\sim Q}
\ell(\theta;x),
$$
</div>

其中 $\mathcal U(P)$ 是围绕名义分布 $P$ 的不确定集合。模型不再只要求在已观察数据的平均情况下表现良好，而要经受一组可能变化中的最坏检验。

从分布比较的角度，许多生成目标都可以写成积分概率度量

<div class="display-equation">
$$
d_{\mathcal F}(P,Q)
=
\sup_{f\in\mathcal F}
\left|
\mathbb E_Pf
-
\mathbb E_Qf
\right|.
$$
</div>

函数类 $\mathcal F$ 决定了模型能够观察到哪些分布差异。Wasserstein-1 距离对应所有 $1$-Lipschitz 函数。若 $\mathcal F$ 是某个再生核 Hilbert 空间中的单位球，就得到最大均值差异 MMD。原始 GAN 的 logistic 判别目标则与 $f$-散度和密度比估计联系在一起。

因此，设计对抗生成模型，本质上是在选择一种“由什么样的测试函数来比较两个分布”的方法。判别器越强，能够发现的差异越丰富；但函数类越复杂，统计估计和优化也越困难。判别器容量、正则化与生成器梯度之间始终存在权衡。

7.8 为什么生成领域后来转向扩散模型

GAN 的采样极快。训练完成后，只需要

<div class="display-equation">
$$
z\sim\mathcal N(0,I),
\qquad
x=G_\theta(z),
$$
</div>

一次前向传播便得到样本。

但这种速度来自把全部难度压进一个单步映射。生成器必须一次把简单高斯分布变成复杂数据分布，判别器又在训练中不断改变监督信号。目标映射高度非线性，优化还是非凸--非凹博弈，这两个困难叠加在一起。

扩散模型选择另一条路径。它先构造一条逐渐加噪的分布路径

<div class="display-equation">
$$
P_0=P_{\mathrm{data}}
\longrightarrow
P_1
\longrightarrow\cdots\longrightarrow
P_T
\approx
\mathcal N(0,I),
$$
</div>

再学习反向的小步去噪。每一步只需完成一个接近恒等映射的局部任务，训练可以变成相对稳定的回归或得分匹配。

GAN 把代价集中在训练上，换取一步采样；扩散模型把困难映射拆成许多简单步骤，换取稳定训练，却增加采样成本。

<div class="display-equation">
$$
\text{GAN：一步生成，动态对手；}
$$
</div>

<div class="display-equation">
$$
\text{扩散：多步生成，固定回归目标。}
$$
</div>

这并不意味着对抗思想已经过时。高分辨率图像恢复、超分辨率、视频生成和生成蒸馏中，判别损失仍常用于增强感知真实感；对抗训练已经成为鲁棒学习的核心方法；域对抗仍是迁移学习的重要工具。

GAN 作为主流通用图像生成架构的地位有所下降，但它最重要的思想已经留在机器学习中：当人类无法直接写出一个足够好的评价函数时，可以训练另一个模型，让它主动寻找当前系统尚未解决的问题。

结论

生成对抗网络的表面故事是“造假者与鉴定者互相竞争”，但真正的数学核心是分布比较。

不能把随机噪声与随机真实样本直接配对做均方误差，因为这种目标会把所有噪声映射到数据均值：

<div class="display-equation">
$$
\arg\min_g
\mathbb E_x
\|g-x\|_2^2
=
\mathbb E[x].
$$
</div>

GAN 因此引入判别器，把分布比较转化为二分类。固定生成器时，最优判别器是密度比的函数：

<div class="display-equation">
$$
D^\star(x)
=
\frac{p_{\mathrm{data}}(x)}
{p_{\mathrm{data}}(x)+p_g(x)}.
$$
</div>

将它代回目标，得到

<div class="display-equation">
$$
V(D^\star,G)
=
-\log4
+
2D_{\mathrm{JS}}
(P_{\mathrm{data}}\|P_g).
$$
</div>

这个结论解释了理想 GAN 在做什么，却没有保证实际交替梯度能够稳定到达全局平衡。支集分离会让 JS 散度饱和；原始生成器损失会在判别器过强时产生梯度消失；极小极大梯度场具有旋转分量；生成映射的 Jacobian 退化、拓扑错配和反馈滞后又会诱发模式坍塌。

WGAN 改用最优传输距离：

<div class="display-equation">
$$
W_1(P_r,P_g)
=
\inf_{\gamma\in\Pi(P_r,P_g)}
\mathbb E_\gamma
\|x-y\|_2.
$$
</div>

它在支集不重合时仍能连续反映分布间的几何距离。Kantorovich--Rubinstein 对偶把原始运输问题变成受 Lipschitz 约束的 critic 优化：

<div class="display-equation">
$$
W_1(P_r,P_g)
=
\sup_{\|f\|_{\mathrm{Lip}}\leq1}
\left(
\mathbb E_{P_r}f
-
\mathbb E_{P_g}f
\right).
$$
</div>

梯度惩罚和谱归一化由此出现。它们不是孤立的经验技巧，而是在有限神经网络中近似实现对偶理论所要求的函数约束。

对抗思想随后扩展到条件生成、图像翻译、模型鲁棒性和域适应。FGSM 来自局部一阶最大化，PGD 是约束内层优化，对抗训练是局部最坏风险最小化；DANN 则让域分类器主动寻找域差异，再迫使特征提取器消除这些差异。

这些方法共同揭示了一种与普通监督学习不同的训练哲学。普通损失告诉模型当前输出与目标相差多少；对抗机制则训练一个会不断进化的检验者，主动寻找模型最薄弱的地方。

<div class="display-equation">
$$
\boxed{
\text{学习者不断修正自己，}
\qquad
\text{对手不断提高检验强度，}
\qquad
\text{能力在二者的共同演化中形成。}
}
$$
</div>

参考资料

Goodfellow 等人的 Generative Adversarial Nets 提出了原始 GAN，并给出最优判别器与 Jensen--Shannon 散度分析。Radford 等人的 DCGAN 系统总结了卷积 GAN 的稳定训练结构。Arjovsky、Chintala 与 Bottou 提出的 WGAN 将最优传输引入生成建模，Gulrajani 等人的 WGAN-GP 以梯度惩罚替代权重裁剪，Miyato 等人进一步用谱归一化控制判别器的 Lipschitz 常数。

Mirza 与 Osindero 提出了条件 GAN。pix2pix 把条件对抗损失与成对重构损失结合，CycleGAN 则使用循环一致性处理无配对图像翻译。Szegedy 等人发现了深度网络的对抗样本，Goodfellow 等人由局部线性化提出 FGSM，Madry 等人以 PGD 内层攻击建立了对抗训练的鲁棒优化框架。Ganin 等人的 DANN 将极小极大思想推广到了无监督域适应。
