---
title: "只观察被选择的动作，如何学习全部动作：重要性加权、指数权重与 EXP3"
date: 2026-09-04 10:00:00
categories:
  - 机器学习理论
tags:
  - 多臂老虎机
  - 对抗性 Bandit
  - EXP3
  - 重要性加权
  - 指数权重
  - 在线学习
  - 后悔分析
  - 数学证明
mathjax: true
toc: true
toc_number: false
comments: true
---

随机 Bandit 把每个臂的奖励看成某个固定概率分布的样本。UCB 通过置信上界安排探索，Thompson Sampling 通过后验抽样安排探索；两者都在估计臂的未知均值。

现在去掉固定均值。广告的效果可以随日期、竞争对手与用户组成而变；一条路线的耗时可以被天气与临时管制改变。如果不再假设这些损失来自一个稳定分布，算法还能学习吗？

困难不只是环境会变。每一轮选择一个动作后，算法只能观察这个动作的损失。其他动作在同一时刻本会产生什么结果，始终是反事实。

EXP3 的推导正是从这个缺口开始。指数权重告诉我们如何累积已知损失；重要性加权则把一次随机观察变成对完整损失向量的无偏估计。两者结合后，一条全信息算法就成了 Bandit 算法。

<!--more-->

## 1. 同一个决策问题，两种反馈

设有 $K\geq2$ 个动作，并记

$$
[K]:=\lbrace1,2,\ldots,K\rbrace.
$$

符号 $[K]$ 表示前 $K$ 个正整数组成的集合。交互共进行 $T$ 轮。在第 $t$ 轮，环境为每个动作 $i\in[K]$ 指定一个损失

$$
\ell&#95;{t,i}\in[0,1].
$$

$\ell$ 是希腊字母 ell，损失越小越好。本轮的完整损失向量记作

$$
\ell&#95;t
:=
(\ell&#95;{t,1},\ldots,\ell&#95;{t,K}).
$$

算法选择动作 $A&#95;t\in[K]$，并承担损失 $\ell&#95;{t,A&#95;t}$。

在**全信息反馈**中，本轮结束后可以看到整个 $\ell&#95;t$。证券组合在收盘后可以查看所有股票的当日收益，就接近这种反馈。

在**Bandit 反馈**中，本轮结束后只能看到 $\ell&#95;{t,A&#95;t}$。平台向用户展示广告 $A&#95;t$，可以观察这次展示是否产生点击；同一位用户对其余广告的即时反应无法同时观察。

这两种反馈的差别会在后悔界中留下一个 $\sqrt K$ 因子。它来自单点观测的估计方差，第 9 节会把这一点算出来。

## 2. 对抗性环境与外部后悔

把所有损失排成一个 $T\times K$ 的表：

$$
\begin{pmatrix}
\ell&#95;{1,1}&\cdots&\ell&#95;{1,K}\\\\
\vdots&\ddots&\vdots\\\\
\ell&#95;{T,1}&\cdots&\ell&#95;{T,K}
\end{pmatrix}.
$$

对抗性模型允许这张表任意变化，只要每个元素都位于 $[0,1]$。本篇使用的时间顺序是：环境先确定损失表，算法后进行随机抽样。“对抗性”因此表示不为损失表假设概率分布，不表示环境可以在看到当前动作后再修改这一轮损失。

动作 $i$ 的累计损失是

$$
L&#95;{T,i}
:=
\sum&#95;{t=1}^{T}\ell&#95;{t,i}.
$$

事后最好的固定动作可以选为

$$
i^\star
\in
\operatorname*{argmin}&#95;{i\in[K]}L&#95;{T,i}.
$$

$\operatorname*{argmin}$ 表示使后面的量取得最小值的动作集合；若有多个最小者，$i^\star$ 可以取其中任意一个。

算法到第 $T$ 轮的期望外部后悔定义为

$$
\overline R&#95;T
:=
\mathbb E\left[
\sum&#95;{t=1}^{T}\ell&#95;{t,A&#95;t}
\right]
-
\min&#95;{i\in[K]}L&#95;{T,i}.
$$

$\mathbb E$ 表示对算法的随机性取期望。比较对象是从第一轮到第 $T$ 轮始终使用同一个动作的最优策略。它与每轮都挑选当轮最优动作的动态比较对象不同。

当 $\overline R&#95;T/T\to0$ 时，平均每轮后悔趋于零。即使损失序列没有固定均值，算法的长期平均表现仍然不会落后于事后最好的固定动作。

## 3. 先从全信息问题开始

假设算法在每轮结束后都能观察完整损失向量。为每个动作维护一个正权重 $w&#95;{t,i}$，并初始化为

$$
w&#95;{1,i}:=1.
$$

第 $t$ 轮的概率是

$$
p&#95;{t,i}
:=
\frac{w&#95;{t,i}}{W&#95;t},
\qquad
W&#95;t
:=
\sum&#95;{j=1}^{K}w&#95;{t,j}.
$$

$W&#95;t$ 是所有权重之和，因此 $p&#95;{t,i}>0$ 且 $\sum&#95;{i=1}^{K}p&#95;{t,i}=1$。数组 $p&#95;t=(p&#95;{t,1},\ldots,p&#95;{t,K})$ 是一个概率分布。

观察 $\ell&#95;t$ 后，按下式更新：

$$
w&#95;{t+1,i}
:=
w&#95;{t,i}\exp(-\eta\ell&#95;{t,i}).
$$

$\exp(x)=e^x$ 是指数函数，$\eta>0$ 称为学习率。展开递推式得

$$
\begin{aligned}
w&#95;{t+1,i}
&=\prod&#95;{s=1}^{t}\exp(-\eta\ell&#95;{s,i})\\\\
&=\exp\left(
-\eta\sum&#95;{s=1}^{t}\ell&#95;{s,i}
\right).
\end{aligned}
$$

所以两个动作 $i$ 与 $j$ 的权重比满足

$$
\frac{w&#95;{t+1,i}}{w&#95;{t+1,j}}
=
\exp\left[
-\eta
\left(
\sum&#95;{s=1}^{t}\ell&#95;{s,i}
-
\sum&#95;{s=1}^{t}\ell&#95;{s,j}
\right)
\right].
$$

累计损失比动作 $j$ 少 $d$ 的动作 $i$，其权重会是后者的 $\exp(\eta d)$ 倍。指数权重不是每轮只选择当前最好的动作，而是把累计损失差平滑地翻译成概率比。

这个算法通常称为 **Hedge**。附录 C 将证明，对任意非负数列 $z&#95;{t,i}$，指数权重都满足确定性不等式

$$
\begin{aligned}
&\sum&#95;{t=1}^{T}\sum&#95;{i=1}^{K}p&#95;{t,i}z&#95;{t,i}
-\sum&#95;{t=1}^{T}z&#95;{t,i^\star}\\\\
&\qquad\leq
\frac{\log K}{\eta}
+\frac{\eta}{2}
\sum&#95;{t=1}^{T}\sum&#95;{i=1}^{K}
p&#95;{t,i}z&#95;{t,i}^2.
\end{aligned}
$$

$\log$ 表示自然对数。右边有两部分：$\log K/\eta$ 是初始时在 $K$ 个动作之间无法区分产生的代价；二次项记录每次更新的波动。

在全信息问题中令 $z&#95;{t,i}=\ell&#95;{t,i}$。由于 $0\leq\ell&#95;{t,i}\leq1$，有

$$
\sum&#95;{i=1}^{K}p&#95;{t,i}\ell&#95;{t,i}^2
\leq
\sum&#95;{i=1}^{K}p&#95;{t,i}
=1.
$$

因此 Hedge 的后悔至多为

$$
\frac{\log K}{\eta}+\frac{\eta T}{2}.
$$

取 $\eta=\sqrt{2\log K/T}$，上式等于

$$
\sqrt{2T\log K}.
$$

全信息问题的算法骨架已经出现。Bandit 反馈的关键是：如何在没有 $\ell&#95;t$ 的情况下，仍然向这个权重更新式提供可用的输入。

## 4. 把未观察损失填成零会发生什么

第 $t$ 轮抽到 $A&#95;t$后，对动作 $i$ 定义指示变量

$$
\mathbf 1\lbrace A&#95;t=i\rbrace
:=
\begin{cases}
1,&A&#95;t=i,\\\\
0,&A&#95;t\neq i.
\end{cases}
$$

指示变量只记录事件是否发生。一个看似自然的损失估计是

$$
\widetilde\ell&#95;{t,i}
:=
\ell&#95;{t,i}\mathbf 1\lbrace A&#95;t=i\rbrace.
$$

它把被选动作的观测值放在对应坐标，把未被选择的坐标填成零。如果算法以概率 $p&#95;{t,i}$ 选择动作 $i$，那么

$$
\mathbb E&#95;t[
\widetilde\ell&#95;{t,i}
]
=
p&#95;{t,i}\ell&#95;{t,i}.
$$

$\mathbb E&#95;t$ 表示在第 $t$ 轮开始时的概率分布 $p&#95;t$ 已经确定后，只对本轮的随机抽样取期望。

期望中多出的 $p&#95;{t,i}$ 造成了系统性缩小。一个动作被选得越少，它的损失就越容易被记为零；估计值的小反映的也许只是观测次数少，并不是真实损失小。

因此，指数权重不能直接把未观察坐标当成零。这些零表示“没有数据”，而指数权重会把它们读成“没有损失”。

## 5. 重要性加权：用观测概率校正缺失

为了消去期望中的 $p&#95;{t,i}$，将被观察到的损失除以它的观测概率：

$$
\widehat\ell&#95;{t,i}
:=
\frac{\ell&#95;{t,i}
\mathbf 1\lbrace A&#95;t=i\rbrace}
{p&#95;{t,i}}.
$$

符号 $\widehat\ell&#95;{t,i}$ 读作“ell hat”，表示 $\ell&#95;{t,i}$ 的估计。由于所有指数权重始终为正，$p&#95;{t,i}>0$，分母始终有定义。

固定动作 $i$，并对本轮抽样取条件期望：

$$
\begin{aligned}
\mathbb E&#95;t[
\widehat\ell&#95;{t,i}
]
&=p&#95;{t,i}
\frac{\ell&#95;{t,i}}{p&#95;{t,i}}
+(1-p&#95;{t,i})\cdot0\\\\
&=\ell&#95;{t,i}.
\end{aligned}
$$

所以 $\widehat\ell&#95;{t,i}$ 是 $\ell&#95;{t,i}$ 的**无偏估计**：它在某一次实现中可以不等于真实损失，但对算法的抽样随机性取平均后，正好回到真实损失。

一个具体时刻可以看清这个校正。设

$$
p&#95;t=(0.6,0.3,0.1),
\qquad
\ell&#95;t=(0.2,0.8,0.4).
$$

如果本轮抽到第 $2$ 个动作，算法只看到 $0.8$，估计向量为

$$
\widehat\ell&#95;t
=
\left(0,\frac{0.8}{0.3},0\right)
=
\left(0,\frac83,0\right).
$$

估计值 $8/3$ 超过了原始损失的范围 $[0,1]$。这是刻意的放大：一个概率为 $0.3$ 才能观察到的事件，一旦发生，就用 $1/0.3$ 倍的权重代表那些没有发生的抽样。

![Bandit 反馈下的重要性加权估计](/images/notes/assets/bandits/exp3-importance-weighting.svg)

无偏性的代价是方差。附录 D 将逐项计算出

$$
\operatorname{Var}&#95;t(
\widehat\ell&#95;{t,i}
)
=
\ell&#95;{t,i}^2
\left(
\frac1{p&#95;{t,i}}-1
\right).
$$

$\operatorname{Var}&#95;t$ 表示在本轮条件分布下的方差。当 $p&#95;{t,i}$ 很小时，对动作 $i$ 的单次估计可以非常剧烈。

## 6. EXP3 的每一步

EXP3 是 Exponential-weight algorithm for Exploration and Exploitation 的缩写。使用损失记号时，它的基本形式如下。

给定学习率 $\eta>0$，初始化

$$
w&#95;{1,i}:=1
\qquad(i\in[K]).
$$

对每个 $t=1,\ldots,T$，依次执行：

1. 把权重归一化为概率

   $$
   p&#95;{t,i}
   :=
   \frac{w&#95;{t,i}}
   {\sum&#95;{j=1}^{K}w&#95;{t,j}};
   $$

2. 按分布 $p&#95;t$ 随机抽取 $A&#95;t$，即

   $$
   \mathbb P&#95;t(A&#95;t=i)=p&#95;{t,i};
   $$

3. 观察被选动作的损失 $\ell&#95;{t,A&#95;t}$，并为所有 $i\in[K]$ 定义

   $$
   \widehat\ell&#95;{t,i}
   :=
   \frac{\ell&#95;{t,A&#95;t}
   \mathbf 1\lbrace A&#95;t=i\rbrace}
   {p&#95;{t,i}};
   $$

4. 用估计损失更新权重

   $$
   w&#95;{t+1,i}
   :=
   w&#95;{t,i}
   \exp(-\eta\widehat\ell&#95;{t,i}).
   $$

$\mathbb P&#95;t$ 表示给定过去信息后本轮抽样的条件概率。在第 3 步中，当 $i=A&#95;t$ 时，分子中的 $\ell&#95;{t,A&#95;t}$ 就是观察值；当 $i\neq A&#95;t$ 时，指示变量为零，不需要知道 $\ell&#95;{t,i}$。所以整个更新只使用 Bandit 反馈。

未被选中的动作满足 $\widehat\ell&#95;{t,i}=0$，其权重在本轮不变。被选中的动作则根据 $\ell&#95;{t,A&#95;t}/p&#95;{t,A&#95;t}$ 减小权重。一个少被选择的动作如果恰好暴露出较大损失，它的权重会受到更大幅度的修正。

## 7. 势函数为什么选权重之和

定义

$$
W&#95;t:=\sum&#95;{i=1}^{K}w&#95;{t,i}.
$$

$W&#95;t$ 称为势函数。它同时提供两个观察方向。

从整体看，

$$
\begin{aligned}
\frac{W&#95;{t+1}}{W&#95;t}
&=
\sum&#95;{i=1}^{K}
p&#95;{t,i}
\exp(-\eta\widehat\ell&#95;{t,i})\\\\
&\leq
1-\eta
\sum&#95;{i=1}^{K}p&#95;{t,i}\widehat\ell&#95;{t,i}
+\frac{\eta^2}{2}
\sum&#95;{i=1}^{K}p&#95;{t,i}\widehat\ell&#95;{t,i}^2.
\end{aligned}
$$

不等式来自

$$
e^{-x}\leq1-x+\frac{x^2}{2}
\qquad(x\geq0).
$$

右边的一次项是算法在估计损失上的平均表现，二次项是为随机估计的波动付出的代价。

从单个比较动作 $i^\star$ 看，

$$
\begin{aligned}
W&#95;{T+1}
&\geq w&#95;{T+1,i^\star}\\\\
&=\exp\left(
-\eta\sum&#95;{t=1}^{T}
\widehat\ell&#95;{t,i^\star}
\right).
\end{aligned}
$$

第一个方向描述算法的加权平均，第二个方向抓住任意一个固定动作。对 $\log(W&#95;{T+1}/W&#95;1)$ 分别建立上界和下界，再把它们合并，就得到第 3 节的指数权重不等式。

这一步完全是确定性的：对每一条抽样路径上产生的 $\widehat\ell&#95;t$，不等式都成立。概率只在下一步进入，用来把估计损失换回真实损失。

## 8. EXP3 的期望后悔界

**定理。** 设 $K\geq2$，并且对每个 $t=1,\ldots,T$ 与 $i\in[K]$ 都有 $\ell&#95;{t,i}\in[0,1]$。对任意 $\eta>0$，上述 EXP3 算法满足

$$
\overline R&#95;T
\leq
\frac{\log K}{\eta}
+\frac{\eta TK}{2}.
$$

特别地，取

$$
\eta
:=
\sqrt{\frac{2\log K}{TK}},
$$

则

$$
\boxed{
\overline R&#95;T
\leq
\sqrt{2TK\log K}
}.
$$

证明的骨架只有三步。

第一步，把指数权重不等式应用到估计损失 $z&#95;{t,i}=\widehat\ell&#95;{t,i}$：

$$
\begin{aligned}
&\sum&#95;{t=1}^{T}\sum&#95;{i=1}^{K}
p&#95;{t,i}\widehat\ell&#95;{t,i}
-\sum&#95;{t=1}^{T}\widehat\ell&#95;{t,i^\star}\\\\
&\qquad\leq
\frac{\log K}{\eta}
+\frac{\eta}{2}
\sum&#95;{t=1}^{T}\sum&#95;{i=1}^{K}
p&#95;{t,i}\widehat\ell&#95;{t,i}^2.
\end{aligned}
$$

第二步，左边的算法项在每条路径上都可简化为

$$
\begin{aligned}
\sum&#95;{i=1}^{K}
p&#95;{t,i}\widehat\ell&#95;{t,i}
&=
\sum&#95;{i=1}^{K}
p&#95;{t,i}
\frac{\ell&#95;{t,i}
\mathbf 1\lbrace A&#95;t=i\rbrace}
{p&#95;{t,i}}\\\\
&=\ell&#95;{t,A&#95;t}.
\end{aligned}
$$

而比较动作的估计是无偏的：

$$
\mathbb E&#95;t[
\widehat\ell&#95;{t,i^\star}
]
=
\ell&#95;{t,i^\star}.
$$

第三步，二次项的条件期望满足

$$
\begin{aligned}
\mathbb E&#95;t\left[
\sum&#95;{i=1}^{K}
p&#95;{t,i}\widehat\ell&#95;{t,i}^2
\right]
&=
\sum&#95;{i=1}^{K}\ell&#95;{t,i}^2\\\\
&\leq K.
\end{aligned}
$$

对势函数不等式取期望，再对 $t$ 求和，就得到定理。附录 E 会从初始权重开始，把这三步与最后的学习率选择完整展开。

## 9. 根号 K 从哪里来

全信息 Hedge 中的二次项每轮至多为

$$
\sum&#95;{i=1}^{K}
p&#95;{t,i}\ell&#95;{t,i}^2
\leq1.
$$

EXP3 使用重要性加权估计后，对二次项取条件期望：

$$
\begin{aligned}
&\mathbb E&#95;t\left[
\sum&#95;{i=1}^{K}
p&#95;{t,i}\widehat\ell&#95;{t,i}^2
\right]\\\\
&\qquad=
\sum&#95;{i=1}^{K}
p&#95;{t,i}
\mathbb E&#95;t\left[
\frac{\ell&#95;{t,i}^2
\mathbf 1\lbrace A&#95;t=i\rbrace}
{p&#95;{t,i}^2}
\right]\\\\
&\qquad=
\sum&#95;{i=1}^{K}\ell&#95;{t,i}^2
\leq K.
\end{aligned}
$$

每轮的二阶代价从 $1$ 变成了 $K$。把它代入

$$
\frac{\log K}{\eta}
+\frac{\eta TK}{2}
$$

并平衡两项，就出现 $\sqrt K$。这是部分反馈的价格：一次只观察一个坐标，需要用更大的随机波动来换取对完整向量的无偏代表。

经典 EXP3 界为 $O(\sqrt{TK\log K})$，这里 $O$ 表示忽略与 $T$、$K$ 无关的常数后得到的渐近上界。对抗性 $K$ 臂 Bandit 的最小最大最优阶为 $\Theta(\sqrt{TK})$，其中 $\Theta$ 表示上界与下界只相差常数因子。使用不同的正则化几何，例如 INF 或 Tsallis 熵，可以去掉这个 $\sqrt{\log K}$ 差距。EXP3 的价值在于它把部分反馈的两个基本技巧暴露得最清楚：无偏估计与指数势函数。

## 10. 显式探索解决的是另一个问题

基本版 EXP3 的所有权重始终严格为正，所以每个 $p&#95;{t,i}$ 也始终为正。期望后悔证明只需要这一点，不要求为每个动作额外混入固定的均匀概率。

但 $p&#95;{t,i}$ 可以非常小。如果一个概率为百万分之一的动作恰好被抽中，重要性加权因子 $1/p&#95;{t,i}$ 会制造一次极大的更新。期望可以平均掉这种稀有事件，单条实现路径上却可能出现明显波动。

一种处理方法是先从权重得到分布 $q&#95;t$，再与均匀分布混合：

$$
p&#95;{t,i}
:=
(1-\gamma)q&#95;{t,i}
+\frac{\gamma}{K},
\qquad
0<\gamma<1.
$$

$\gamma$ 是显式探索比例。此时

$$
p&#95;{t,i}\geq\frac{\gamma}{K},
$$

从而

$$
0\leq\widehat\ell&#95;{t,i}
\leq\frac K\gamma.
$$

估计量获得了确定的上界。代价是，每轮有 $\gamma$ 的概率按均匀分布行动，累计后悔最多因此增加 $\gamma T$。

附录 F 会证明这个版本满足

$$
\overline R&#95;T
\leq
\frac{\log K}{\eta}
+\frac{\eta TK}{2}
+\gamma T.
$$

选择 $\eta$ 与 $\gamma$ 时，需要在初始复杂度、估计方差和均匀探索代价之间平衡。经典 EXP3.P 与隐式探索方法会进一步修改估计量或更新式，从期望保证走向高概率保证。

## 11. 在线系统中，这个估计量表示什么

假设广告 $i$ 在当前轮的展示概率是 $p&#95;{t,i}=0.02$。它被展示后产生损失 $1$，则重要性加权估计为

$$
\widehat\ell&#95;{t,i}=\frac1{0.02}=50.
$$

$50$ 不是广告的真实单次损失，而是这条低概率观测在随机设计中代表的份量。它与调查抽样中给稀有群体更大权重是同一种数学动作。

这也说明了为什么工业实现中必须记录当时的选择概率。只保存“选了什么”与“结果如何”，不保存 $p&#95;{t,A&#95;t}$，便无法在事后正确重建这个无偏估计。策略的随机化概率是数据生成过程的一部分。

如果系统还对最小流量、单次更新幅度或风险暴露有要求，显式探索下界 $p&#95;{t,i}\geq\gamma/K$ 就不只是证明工具，也是系统约束。$\gamma$ 决定最少要为每个动作保留多少曝光，同时也决定重要性权重最多可以放大多少。

## 12. 从 Hedge 到 EXP3 的真正转折

EXP3 的更新式并不复杂。整个推导中最关键的转折是，要把“没有观察”与“损失为零”分开。

指数权重处理已经写成数字的损失；重要性加权处理哪些损失有机会被写成数字。前者是在线优化，后者是统计估计。二次势函数项把两者连在一起：估计的方差最终变成后悔界中的 $K$。

因此，EXP3 的 $\sqrt{TK\log K}$ 不只是一个算法界。每个因子都能追溯到证明中的一个具体来源：$T$ 来自累积时间，$K$ 来自单点反馈的二阶代价，$\log K$ 来自初始时对 $K$ 个动作的均匀权重。

## 参考文献

1. Nick Littlestone and Manfred K. Warmuth, *The Weighted Majority Algorithm*, Information and Computation, 108(2):212--261, 1994.
2. Yoav Freund and Robert E. Schapire, *A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting*, Journal of Computer and System Sciences, 55(1):119--139, 1997.
3. Peter Auer, Nicolò Cesa-Bianchi, Yoav Freund, and Robert E. Schapire, *The Nonstochastic Multiarmed Bandit Problem*, SIAM Journal on Computing, 32(1):48--77, 2002.
4. Nicolò Cesa-Bianchi and Gábor Lugosi, *Prediction, Learning, and Games*, Cambridge University Press, 2006.
5. Jean-Yves Audibert and Sébastien Bubeck, *Minimax Policies for Adversarial and Stochastic Bandits*, Proceedings of COLT, 2009.
6. Sébastien Bubeck and Nicolò Cesa-Bianchi, *Regret Analysis of Stochastic and Nonstochastic Multi-Armed Bandit Problems*, Foundations and Trends in Machine Learning, 5(1):1--122, 2012.
7. Gergely Neu, *Explore No More: Improved High-Probability Regret Bounds for Non-Stochastic Bandits*, Advances in Neural Information Processing Systems, 2015.
8. Tor Lattimore and Csaba Szepesvári, *Bandit Algorithms*, Cambridge University Press, 2020.

---

## 附录

如下为正文附录补充。

## A. 概率符号与时间顺序

### A.1 条件概率与条件期望

记 $\mathcal F&#95;{t-1}$ 为第 $t$ 轮抽取动作前已经知道的全部信息。符号 $\mathcal F$ 读作花体 F，可以把它理解为历史记录。概率向量 $p&#95;t$ 是这些历史的函数，因此在给定 $\mathcal F&#95;{t-1}$ 后，$p&#95;t$ 已经是确定的。

算法的抽样规则写作

$$
\mathbb P(A&#95;t=i\mid\mathcal F&#95;{t-1})
=p&#95;{t,i}.
$$

符号 $\mathbb P$ 表示概率，竖线后的条件表示过去信息被固定。正文中的简写是

$$
\mathbb P&#95;t(A&#95;t=i):=
\mathbb P(A&#95;t=i\mid\mathcal F&#95;{t-1}).
$$

对随机变量 $X$ 的条件期望简写为

$$
\mathbb E&#95;t[X]
:=
\mathbb E[X\mid\mathcal F&#95;{t-1}].
$$

对任意 $i\in[K]$，指示变量的条件期望是

$$
\begin{aligned}
\mathbb E&#95;t[
\mathbf 1\lbrace A&#95;t=i\rbrace
]
&=1\cdot\mathbb P&#95;t(A&#95;t=i)\\\\
&\quad+0\cdot\mathbb P&#95;t(A&#95;t\neq i)\\\\
&=p&#95;{t,i}.
\end{aligned}
$$

完全期望可以通过条件期望逐轮计算。全期望公式给出

$$
\mathbb E[
\mathbb E&#95;t[X]
]
=
\mathbb E[X].
$$

这条恒等式使我们可以先固定历史，计算第 $t$ 轮的抽样期望，然后再对历史本身取期望。

### A.2 为什么损失必须先于当前动作确定

如果环境可以先看见 $A&#95;t$ 再决定本轮损失，它可以设置

$$
\ell&#95;{t,i}
:=
\mathbf 1\lbrace A&#95;t=i\rbrace.
$$

算法每轮选中的动作损失都为 $1$，所以累计损失等于 $T$。动作 $i$ 的累计损失则等于它被选中的次数

$$
N&#95;i(T)
:=
\sum&#95;{t=1}^{T}
\mathbf 1\lbrace A&#95;t=i\rbrace.
$$

因为每轮恰好选中一个动作，

$$
\sum&#95;{i=1}^{K}N&#95;i(T)=T.
$$

所以至少有一个动作 $j$ 满足 $N&#95;j(T)\leq T/K$。否则所有 $K$ 个计数都严格大于 $T/K$，它们的和就会严格大于 $T$，与上式矛盾。

因此这种环境下的路径后悔至少为

$$
\begin{aligned}
T-\min&#95;{i\in[K]}N&#95;i(T)
&\geq T-\frac TK\\\\
&=T\left(1-\frac1K\right).
\end{aligned}
$$

它与 $T$ 成正比，平均每轮后悔不会趋于零。因此对抗性 Bandit 会要求当前损失在 $A&#95;t$ 抽样之前已经确定。

## B. 势函数证明中的两个标量不等式

### B.1 指数函数的二次上界

对 $x\geq0$，要证明

$$
e^{-x}\leq1-x+\frac{x^2}{2}.
$$

定义两边之差

$$
f(x)
:=
1-x+\frac{x^2}{2}-e^{-x}.
$$

直接计算导数：

$$
f'(x)=-1+x+e^{-x},
$$

$$
f''(x)=1-e^{-x}.
$$

当 $x\geq0$ 时，$e^{-x}\leq1$，因此 $f''(x)\geq0$。这表示 $f'$ 在 $[0,\infty)$ 上单调不减。又因为

$$
f'(0)=0,
$$

所以 $f'(x)\geq0$，从而 $f$ 在 $[0,\infty)$ 上单调不减。最后，

$$
f(0)=0,
$$

所以 $f(x)\geq0$。这就是要证的不等式。

### B.2 对数函数的线性上界

对每个 $u>0$，要证明

$$
\log u\leq u-1.
$$

定义

$$
g(u):=u-1-\log u.
$$

则

$$
g'(u)=1-\frac1u=\frac{u-1}{u}.
$$

当 $0<u<1$ 时，$g'(u)<0$；当 $u>1$ 时，$g'(u)>0$。因此 $g$ 在 $u=1$ 处取得最小值。由

$$
g(1)=0,
$$

可得 $g(u)\geq0$，也就是 $\log u\leq u-1$。

## C. 指数权重不等式的完整证明

设 $z&#95;{t,i}\geq0$，初始权重 $w&#95;{1,i}=1$，更新式为

$$
w&#95;{t+1,i}
=
w&#95;{t,i}\exp(-\eta z&#95;{t,i}),
$$

并定义

$$
W&#95;t:=\sum&#95;{i=1}^{K}w&#95;{t,i},
\qquad
p&#95;{t,i}:=\frac{w&#95;{t,i}}{W&#95;t}.
$$

先从整体权重变化得到上界。由附录 B.1，

$$
\begin{aligned}
\frac{W&#95;{t+1}}{W&#95;t}
&=
\sum&#95;{i=1}^{K}
\frac{w&#95;{t,i}}{W&#95;t}
\exp(-\eta z&#95;{t,i})\\\\
&=
\sum&#95;{i=1}^{K}
p&#95;{t,i}\exp(-\eta z&#95;{t,i})\\\\
&\leq
\sum&#95;{i=1}^{K}p&#95;{t,i}
\left(
1-\eta z&#95;{t,i}
+\frac{\eta^2z&#95;{t,i}^2}{2}
\right)\\\\
&=1
-\eta\sum&#95;{i=1}^{K}p&#95;{t,i}z&#95;{t,i}
+\frac{\eta^2}{2}
\sum&#95;{i=1}^{K}p&#95;{t,i}z&#95;{t,i}^2.
\end{aligned}
$$

对两边取对数。由 $W&#95;{t+1}/W&#95;t>0$ 且对数函数单调递增，再使用附录 B.2，得

$$
\begin{aligned}
\log\frac{W&#95;{t+1}}{W&#95;t}
&\leq
-\eta\sum&#95;{i=1}^{K}p&#95;{t,i}z&#95;{t,i}\\\\
&\quad+
\frac{\eta^2}{2}
\sum&#95;{i=1}^{K}p&#95;{t,i}z&#95;{t,i}^2.
\end{aligned}
$$

对 $t=1,\ldots,T$ 求和，左边望远镜式相消：

$$
\begin{aligned}
\sum&#95;{t=1}^{T}
\log\frac{W&#95;{t+1}}{W&#95;t}
&=
\log\prod&#95;{t=1}^{T}
\frac{W&#95;{t+1}}{W&#95;t}\\\\
&=\log\frac{W&#95;{T+1}}{W&#95;1}.
\end{aligned}
$$

所以

$$
\begin{aligned}
\log\frac{W&#95;{T+1}}{W&#95;1}
&\leq
-\eta
\sum&#95;{t=1}^{T}\sum&#95;{i=1}^{K}
p&#95;{t,i}z&#95;{t,i}\\\\
&\quad+
\frac{\eta^2}{2}
\sum&#95;{t=1}^{T}\sum&#95;{i=1}^{K}
p&#95;{t,i}z&#95;{t,i}^2.
\end{aligned}
$$

再固定任意比较动作 $i^\star$。由于 $W&#95;{T+1}$ 是所有正权重之和，

$$
W&#95;{T+1}\geq w&#95;{T+1,i^\star}.
$$

展开动作 $i^\star$ 的权重递推：

$$
\begin{aligned}
w&#95;{T+1,i^\star}
&=w&#95;{1,i^\star}
\prod&#95;{t=1}^{T}
\exp(-\eta z&#95;{t,i^\star})\\\\
&=\exp\left(
-\eta\sum&#95;{t=1}^{T}z&#95;{t,i^\star}
\right),
\end{aligned}
$$

其中使用了 $w&#95;{1,i^\star}=1$。另一方面，

$$
W&#95;1=\sum&#95;{i=1}^{K}w&#95;{1,i}=K.
$$

因此

$$
\begin{aligned}
\log\frac{W&#95;{T+1}}{W&#95;1}
&\geq
\log\frac{w&#95;{T+1,i^\star}}K\\\\
&= -\eta\sum&#95;{t=1}^{T}z&#95;{t,i^\star}
-\log K.
\end{aligned}
$$

将这个下界与前面的上界合并：

$$
\begin{aligned}
&-\eta\sum&#95;{t=1}^{T}z&#95;{t,i^\star}
-\log K\\\\
&\quad\leq
-\eta
\sum&#95;{t=1}^{T}\sum&#95;{i=1}^{K}
p&#95;{t,i}z&#95;{t,i}\\\\
&\qquad+
\frac{\eta^2}{2}
\sum&#95;{t=1}^{T}\sum&#95;{i=1}^{K}
p&#95;{t,i}z&#95;{t,i}^2.
\end{aligned}
$$

把一次项移到左边，再除以正数 $\eta$，得

$$
\begin{aligned}
&\sum&#95;{t=1}^{T}\sum&#95;{i=1}^{K}
p&#95;{t,i}z&#95;{t,i}
-\sum&#95;{t=1}^{T}z&#95;{t,i^\star}\\\\
&\qquad\leq
\frac{\log K}{\eta}
+\frac{\eta}{2}
\sum&#95;{t=1}^{T}\sum&#95;{i=1}^{K}
p&#95;{t,i}z&#95;{t,i}^2.
\end{aligned}
$$

证明完成。

## D. 重要性加权估计的完整矩计算

给定第 $t$ 轮开始时的历史，$p&#95;{t,i}$ 与 $\ell&#95;{t,i}$ 都可以当作固定数。对任意 $i\in[K]$，

$$
\widehat\ell&#95;{t,i}
=
\frac{\ell&#95;{t,i}}
{p&#95;{t,i}}
\mathbf 1\lbrace A&#95;t=i\rbrace.
$$

### D.1 一阶矩与无偏性

由条件期望的线性性质以及附录 A.1，

$$
\begin{aligned}
\mathbb E&#95;t[
\widehat\ell&#95;{t,i}
]
&=
\frac{\ell&#95;{t,i}}
{p&#95;{t,i}}
\mathbb E&#95;t[
\mathbf 1\lbrace A&#95;t=i\rbrace
]\\\\
&=
\frac{\ell&#95;{t,i}}
{p&#95;{t,i}}p&#95;{t,i}\\\\
&=\ell&#95;{t,i}.
\end{aligned}
$$

### D.2 二阶矩

指示变量只能取 $0$ 或 $1$，因此它的平方等于自身：

$$
\mathbf 1\lbrace A&#95;t=i\rbrace^2
=
\mathbf 1\lbrace A&#95;t=i\rbrace.
$$

所以

$$
\begin{aligned}
\mathbb E&#95;t[
\widehat\ell&#95;{t,i}^2
]
&=
\frac{\ell&#95;{t,i}^2}
{p&#95;{t,i}^2}
\mathbb E&#95;t[
\mathbf 1\lbrace A&#95;t=i\rbrace
]\\\\
&=
\frac{\ell&#95;{t,i}^2}
{p&#95;{t,i}^2}p&#95;{t,i}\\\\
&=
\frac{\ell&#95;{t,i}^2}
{p&#95;{t,i}}.
\end{aligned}
$$

### D.3 方差

条件方差的定义是

$$
\operatorname{Var}&#95;t(X)
:=
\mathbb E&#95;t[X^2]
-\mathbb E&#95;t[X]^2.
$$

将 D.1 和 D.2 代入，

$$
\begin{aligned}
\operatorname{Var}&#95;t(
\widehat\ell&#95;{t,i}
)
&=
\frac{\ell&#95;{t,i}^2}{p&#95;{t,i}}
-\ell&#95;{t,i}^2\\\\
&=
\ell&#95;{t,i}^2
\left(
\frac1{p&#95;{t,i}}-1
\right).
\end{aligned}
$$

### D.4 势函数中的加权二阶矩

由于 $p&#95;{t,i}$ 在条件期望下是固定数，

$$
\begin{aligned}
&\mathbb E&#95;t\left[
\sum&#95;{i=1}^{K}
p&#95;{t,i}\widehat\ell&#95;{t,i}^2
\right]\\\\
&\qquad=
\sum&#95;{i=1}^{K}p&#95;{t,i}
\mathbb E&#95;t[
\widehat\ell&#95;{t,i}^2
]\\\\
&\qquad=
\sum&#95;{i=1}^{K}p&#95;{t,i}
\frac{\ell&#95;{t,i}^2}{p&#95;{t,i}}\\\\
&\qquad=
\sum&#95;{i=1}^{K}\ell&#95;{t,i}^2\\\\
&\qquad\leq K.
\end{aligned}
$$

最后一步使用了 $0\leq\ell&#95;{t,i}\leq1$，因此 $\ell&#95;{t,i}^2\leq1$，$K$ 个不超过 $1$ 的数之和不超过 $K$。

## E. EXP3 期望后悔上界的完整证明

选择一个事后最优固定动作

$$
i^\star
\in
\operatorname*{argmin}&#95;{i\in[K]}
\sum&#95;{t=1}^{T}\ell&#95;{t,i}.
$$

附录 C 的不等式对每一条实现的估计损失路径成立。取 $z&#95;{t,i}=\widehat\ell&#95;{t,i}$，得

$$
\begin{aligned}
&\sum&#95;{t=1}^{T}\sum&#95;{i=1}^{K}
p&#95;{t,i}\widehat\ell&#95;{t,i}
-\sum&#95;{t=1}^{T}\widehat\ell&#95;{t,i^\star}\\\\
&\qquad\leq
\frac{\log K}{\eta}
+\frac{\eta}{2}
\sum&#95;{t=1}^{T}\sum&#95;{i=1}^{K}
p&#95;{t,i}\widehat\ell&#95;{t,i}^2.
\end{aligned}
$$

先简化左边第一项。对固定的 $t$，只有 $i=A&#95;t$ 时指示变量为 $1$，因此

$$
\begin{aligned}
\sum&#95;{i=1}^{K}
p&#95;{t,i}\widehat\ell&#95;{t,i}
&=
\sum&#95;{i=1}^{K}
\ell&#95;{t,i}
\mathbf 1\lbrace A&#95;t=i\rbrace\\\\
&=\ell&#95;{t,A&#95;t}.
\end{aligned}
$$

这个等式是逐路径成立的，因此

$$
\mathbb E\left[
\sum&#95;{t=1}^{T}\sum&#95;{i=1}^{K}
p&#95;{t,i}\widehat\ell&#95;{t,i}
\right]
=
\mathbb E\left[
\sum&#95;{t=1}^{T}\ell&#95;{t,A&#95;t}
\right].
$$

对左边第二项，附录 D.1 和全期望公式给出

$$
\begin{aligned}
\mathbb E[
\widehat\ell&#95;{t,i^\star}
]
&=
\mathbb E\left[
\mathbb E&#95;t[
\widehat\ell&#95;{t,i^\star}
]
\right]\\\\
&=\mathbb E[
\ell&#95;{t,i^\star}
]\\\\
&=\ell&#95;{t,i^\star}.
\end{aligned}
$$

最后一个等号使用了损失表在互动开始前已经确定这一条件。对二次项，附录 D.4 给出

$$
\begin{aligned}
&\mathbb E\left[
\sum&#95;{t=1}^{T}\sum&#95;{i=1}^{K}
p&#95;{t,i}\widehat\ell&#95;{t,i}^2
\right]\\\\
&\qquad=
\sum&#95;{t=1}^{T}
\mathbb E\left[
\mathbb E&#95;t\left[
\sum&#95;{i=1}^{K}
p&#95;{t,i}\widehat\ell&#95;{t,i}^2
\right]
\right]\\\\
&\qquad\leq
\sum&#95;{t=1}^{T}K\\\\
&\qquad=TK.
\end{aligned}
$$

对附录 C 的路径不等式取期望，代入上面三个结果：

$$
\begin{aligned}
&\mathbb E\left[
\sum&#95;{t=1}^{T}\ell&#95;{t,A&#95;t}
\right]
-\sum&#95;{t=1}^{T}\ell&#95;{t,i^\star}\\\\
&\qquad\leq
\frac{\log K}{\eta}
+\frac{\eta TK}{2}.
\end{aligned}
$$

按 $i^\star$ 的定义，

$$
\sum&#95;{t=1}^{T}\ell&#95;{t,i^\star}
=
\min&#95;{i\in[K]}
\sum&#95;{t=1}^{T}\ell&#95;{t,i}.
$$

因此

$$
\overline R&#95;T
\leq
\frac{\log K}{\eta}
+\frac{\eta TK}{2}.
$$

现在选择学习率。令

$$
h(\eta)
:=
\frac{\log K}{\eta}
+\frac{\eta TK}{2}.
$$

对 $\eta>0$ 求导：

$$
h'(\eta) = -\frac{\log K}{\eta^2}
+\frac{TK}{2}.
$$

令 $h'(\eta)=0$，得

$$
\eta^2=\frac{2\log K}{TK}.
$$

因为 $\eta>0$，取

$$
\eta=\sqrt{\frac{2\log K}{TK}}.
$$

又因为

$$
h''(\eta)=\frac{2\log K}{\eta^3}>0,
$$

这个驻点是最小值点。代回 $h$：

$$
\begin{aligned}
h(\eta)
&=
\frac{\log K}
{\sqrt{2\log K/(TK)}}
+\frac{TK}{2}
\sqrt{\frac{2\log K}{TK}}\\\\
&=
\sqrt{\frac{TK\log K}{2}}
+\sqrt{\frac{TK\log K}{2}}\\\\
&=
\sqrt{2TK\log K}.
\end{aligned}
$$

主定理证明完成。

## F. 混合均匀探索的代价

设指数权重归一化后的分布为

$$
q&#95;{t,i}
:=
\frac{w&#95;{t,i}}{\sum&#95;{j=1}^{K}w&#95;{t,j}},
$$

实际抽样分布为

$$
p&#95;{t,i}
:=
(1-\gamma)q&#95;{t,i}
+\frac\gamma K,
\qquad
0<\gamma<1.
$$

估计量仍然使用实际抽样概率：

$$
\widehat\ell&#95;{t,i}
:=
\frac{\ell&#95;{t,i}
\mathbf 1\lbrace A&#95;t=i\rbrace}
{p&#95;{t,i}}.
$$

因此无偏性仍然成立。附录 C 的势函数分布现在是 $q&#95;t$，所以

$$
\begin{aligned}
&\mathbb E\left[
\sum&#95;{t=1}^{T}
\sum&#95;{i=1}^{K}q&#95;{t,i}
\widehat\ell&#95;{t,i}
-\sum&#95;{t=1}^{T}
\widehat\ell&#95;{t,i^\star}
\right]\\\\
&\qquad\leq
\frac{\log K}{\eta}
+\frac\eta2
\sum&#95;{t=1}^{T}
\mathbb E\left[
\sum&#95;{i=1}^{K}q&#95;{t,i}
\widehat\ell&#95;{t,i}^2
\right].
\end{aligned}
$$

加权二阶矩的条件期望是

$$
\begin{aligned}
\mathbb E&#95;t\left[
\sum&#95;{i=1}^{K}q&#95;{t,i}
\widehat\ell&#95;{t,i}^2
\right]
&=
\sum&#95;{i=1}^{K}
q&#95;{t,i}
\frac{\ell&#95;{t,i}^2}{p&#95;{t,i}}.
\end{aligned}
$$

由于

$$
p&#95;{t,i}
\geq
(1-\gamma)q&#95;{t,i},
$$

对每个 $i$ 都有

$$
\frac{q&#95;{t,i}}{p&#95;{t,i}}
\leq
\frac1{1-\gamma}.
$$

因此

$$
\begin{aligned}
\mathbb E&#95;t\left[
\sum&#95;{i=1}^{K}q&#95;{t,i}
\widehat\ell&#95;{t,i}^2
\right]
&\leq
\frac1{1-\gamma}
\sum&#95;{i=1}^{K}\ell&#95;{t,i}^2\\\\
&\leq\frac K{1-\gamma}.
\end{aligned}
$$

利用无偏性并对时间求和，得到权重分布 $q&#95;t$ 的期望后悔界

$$
\begin{aligned}
&\mathbb E\left[
\sum&#95;{t=1}^{T}
\sum&#95;{i=1}^{K}q&#95;{t,i}\ell&#95;{t,i}
\right]
-L&#95;{T,i^\star}\\\\
&\qquad\leq
\frac{\log K}{\eta}
+\frac{\eta TK}{2(1-\gamma)}.
\end{aligned}
$$

实际抽样分布是 $p&#95;t$。令 $u=(1/K,\ldots,1/K)$ 表示 $K$ 个动作上的均匀分布，则 $p&#95;t=(1-\gamma)q&#95;t+\gamma u$。于是

$$
\begin{aligned}
&\sum&#95;{t=1}^{T}
\sum&#95;{i=1}^{K}p&#95;{t,i}\ell&#95;{t,i}
-L&#95;{T,i^\star}\\\\
&\quad=
(1-\gamma)
\left(
\sum&#95;{t=1}^{T}
\sum&#95;{i=1}^{K}q&#95;{t,i}\ell&#95;{t,i}
-L&#95;{T,i^\star}
\right)\\\\
&\qquad+
\gamma
\left(
\sum&#95;{t=1}^{T}
\sum&#95;{i=1}^{K}u&#95;i\ell&#95;{t,i}
-L&#95;{T,i^\star}
\right).
\end{aligned}
$$

因为每轮均匀分布的期望损失不超过 $1$，并且 $L&#95;{T,i^\star}\geq0$，第二个括号至多为 $T$。对上式取期望，并代入 $q&#95;t$ 的后悔界：

$$
\begin{aligned}
\overline R&#95;T
&\leq
(1-\gamma)
\left(
\frac{\log K}{\eta}
+\frac{\eta TK}{2(1-\gamma)}
\right)
+\gamma T\\\\
&=
\frac{(1-\gamma)\log K}{\eta}
+\frac{\eta TK}{2}
+\gamma T\\\\
&\leq
\frac{\log K}{\eta}
+\frac{\eta TK}{2}
+\gamma T.
\end{aligned}
$$

这就是正文中的结论。

## G. 奖励记号与损失记号的等价性

如果问题以奖励 $r&#95;{t,i}\in[0,1]$ 表示，可定义

$$
\ell&#95;{t,i}:=1-r&#95;{t,i}.
$$

最大化累计奖励与最小化累计损失等价，因为

$$
\begin{aligned}
\sum&#95;{t=1}^{T}\ell&#95;{t,i}
&=\sum&#95;{t=1}^{T}(1-r&#95;{t,i})\\\\
&=T-\sum&#95;{t=1}^{T}r&#95;{t,i}.
\end{aligned}
$$

对任意比较动作 $i$，奖励后悔与损失后悔逐路径相等：

$$
\begin{aligned}
&\sum&#95;{t=1}^{T}r&#95;{t,i}
-\sum&#95;{t=1}^{T}r&#95;{t,A&#95;t}\\\\
&\quad=
\sum&#95;{t=1}^{T}(1-\ell&#95;{t,i})
-\sum&#95;{t=1}^{T}(1-\ell&#95;{t,A&#95;t})\\\\
&\quad=
\sum&#95;{t=1}^{T}\ell&#95;{t,A&#95;t}
-\sum&#95;{t=1}^{T}\ell&#95;{t,i}.
\end{aligned}
$$

所以在奖励形式与损失形式之间切换，不会改变后悔的数值。本篇使用损失记号，是因为“损失越大，指数权重越小”可以直接写成 $\exp(-\eta\widehat\ell&#95;{t,i})$。
