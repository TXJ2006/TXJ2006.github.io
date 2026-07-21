---
title: '从零读懂 CTS3：单种子耦合、$k^\star$ 边界与后续解法'
subtitle: '论文背景、核心思想、逐步证明拆解、Blocker 反例、MNL 与 Delay-Max'
summary: '本文从组合半 Bandit 的基本模型出发，系统解读 CTS3 如何通过单种子共单调耦合消除标准 CTS 的指数型联合乐观障碍，并追踪原论文最坏情形界中 $k^\star$ 因子的来源。随后给出一般 CMAB 下不可无条件摘除 $k^\star$ 的 blocker 反例，分析 MNL-Bandit 的结构性正面结果，以及修改算法 Delay-Max 如何通过保留未被消费的乐观样本得到 $k^\star$-free regret。'
description: '本文从组合半 Bandit 的基本模型出发，系统解读 CTS3 如何通过单种子共单调耦合消除标准 CTS 的指数型联合乐观障碍，并追踪原论文最坏情形界中 $k^\star$ 因子的来源。随后给出一般 CMAB 下不可无条件摘除 $k^\star$ 的 blocker 反例，分析 MNL-Bandit 的结构性正面结果，以及修改算法 Delay-Max 如何通过保留未被消费的乐观样本得到 $k^\star$-free regret。'
date: 2026-07-21
lastmod: 2026-07-21
weight: 60
tags: ["Combinatorial Bandits", "Thompson Sampling", "CTS3", "Regret Analysis", "MNL Bandit", "Coupling"]
draft: false
ShowToc: false
hideMeta: true
---

## 引言

组合多臂老虎机的困难，不只是可行动作很多。更本质的困难是：一次动作由多个未知基础臂共同组成，而算法必须根据这些坐标的随机估计，判断整个组合是否值得选择。

标准 Combinatorial Thompson Sampling（CTS）对每个基础臂独立进行 posterior sampling。若最优组合包含 $k^\star$ 个基础臂，而 oracle 只有在这些坐标同时“足够好”时才会返回最优组合，那么联合乐观概率会发生乘法：

<div class="display-equation">
$$
\mathbb P(\text{all coordinates are good})
=
\prod_{i\in S^\star}p_i.
$$
</div>

只要每个 $p_i<1$，这个概率就会随 $k^\star$ 指数衰减。原论文提出的 CTS3 只改动了一行：不再给每个坐标分别抽取随机种子，而是令所有基础臂共享同一个随机种子。这个改动保持了每个坐标的边缘采样分布，却彻底重写了联合分布。

本文重点回答四个问题。

第一，原论文究竟解决了什么？

第二，为什么消除了 $L^{k^\star}$ 以后，最坏情形界中仍然留下 $\sqrt{k^\star}$？

第三，这个剩余因子只是证明技术造成的，还是一般模型中确实存在结构性障碍？

第四，在哪些附加结构或算法改动下，$k^\star$ 可以真正被摘除？

后面的结论需要严格区分：

<div class="display-equation">
$$
\boxed{
\begin{array}{c}
\text{原始 CTS3 在一般 CMAB 假设下：不能无条件摘除 }k^\star;\\[2mm]
\text{特殊模型结构或修改算法以后：可以得到 }k^\star\text{-free 界。}
\end{array}
}
$$
</div>

## 零、先不看公式：这篇论文到底在解决什么困难

先考虑一个最直观的网络选择问题。一个通信网络里有很多条边，每条边都有未知的平均质量。每一轮，我们必须选择一棵生成树来传输信息。生成树不是一条边，而是一组必须共同工作的边。选择以后，我们能观察这棵树中每条边本轮的质量，却看不到没有选择的边。

在这个例子里，每条边是一个 base arm，一棵生成树是一个 super arm，所有可行生成树组成集合族 $\mathcal I$。学习算法面对的不是“哪一条边最好”，而是“哪一个组合的期望总收益最高”。

困难从这里开始。假设真正最优的生成树包含 $k^\star$ 条边。算法若低估其中任何一条关键边，oracle 都可能转而选择另一棵树。于是，一个组合是否被选中，不只取决于某个坐标是否乐观，而可能取决于很多坐标是否同时处在有利位置。

经典 Thompson Sampling 的通俗理解是：根据当前数据，算法对每个未知均值维护一个后验分布；每轮从后验中抽取一个“可能的世界”，然后假装这个世界就是真实世界，在其中选择最优动作。若动作只是单个 arm，这个思想非常自然。组合动作中，标准 CTS 为不同坐标分别抽样，于是得到的“世界”由许多彼此独立的局部随机波动拼接而成。

这会产生一种组合上的不协调。最优集合中的第一个坐标可能被抽高，第二个坐标被抽低，第三个坐标又被抽高。单个坐标的后验抽样都没有问题，但把它们组合起来以后，最优结构可能长期无法整体显得足够好。

CTS3 的思想可以用“共同分位数”来理解。假设本轮抽到一个均匀随机数 $U_t=0.95$。标准 CTS 会给每个臂各抽一个不同的分位数；CTS3 则让所有臂都取各自后验分布的第 $95\%$ 分位数。若本轮抽到 $U_t=0.10$，所有臂都取各自的第 $10\%$ 分位数。每个臂自己的后验分布没有改变，改变的是不同臂之间如何共同波动。

因此，原论文真正研究的不是“怎样把采样方差调大”，而是下面这个更根本的问题：

<div class="display-equation">
$$
\text{在保持每个坐标边缘分布不变的前提下，}
\quad
\text{怎样设计联合依赖，使组合探索不再指数困难？}
$$
</div>

这也是整篇论文最值得保留的思想。Thompson Sampling 的探索能力不只由每个坐标抽什么分布决定，也由这些坐标怎样耦合决定。

### 四个数学对象

上面的网络例子对应四个必须分清的对象。

第一，环境参数是未知均值 $\mu_i$。它们固定但不可见。

第二，环境观测是 $Z_i(t)$。它们是每轮真正产生的随机数据。

第三，经验均值 $\widehat\mu_i(t)$ 是算法根据历史数据形成的估计。

第四，后验样本 $\theta_i(t)$ 是算法为了做本轮决策而额外生成的随机分数。它既不是真实均值，也不是环境本轮的观测。

许多阅读错误都来自把这四个对象混在一起。后文每次出现公式时，都应先问：这个量属于环境、历史统计，还是算法内部随机化？

## 一、组合半 Bandit 的数学模型

共有 $m$ 个基础臂：

<div class="display-equation">
$$
[m]:=\{1,2,\ldots,m\}.
$$
</div>

算法每轮不是选择一个基础臂，而是从可行集合族

<div class="display-equation">
$$
\mathcal I\subseteq 2^{[m]}\setminus\{\varnothing\}
$$
</div>

中选择一个集合 $S(t)\in\mathcal I$。每个 $S\in\mathcal I$ 称为 super arm。定义

<div class="display-equation">
$$
k:=\max_{S\in\mathcal I}|S|.
$$
</div>

第 $t$ 轮，基础臂 $i$ 产生观测

<div class="display-equation">
$$
Z_i(t)\sim\mathcal N(\mu_i,1),
$$
</div>

其中真实均值向量

<div class="display-equation">
$$
\mu=(\mu_1,\ldots,\mu_m)
$$
</div>

未知。选择 $S(t)$ 后，semi-bandit feedback 给出

<div class="display-equation">
$$
Q(t)=\{(i,Z_i(t)):i\in S(t)\}.
$$
</div>

因此，算法能观察所选组合内每个基础臂的逐坐标反馈，而不是只看到组合总收益。这里并不要求同一轮不同基础臂的环境观测彼此独立。

### 定义 1（期望收益）

对任意 $S\in\mathcal I$，其期望收益写为

<div class="display-equation">
$$
r(S,\mu)=\mathbb E[R(S,Z)].
$$
</div>

原论文考虑以下四类结构条件。

### 假设 1（均值依赖）

收益只依赖所选集合中的均值坐标：

<div class="display-equation">
$$
r(S,\mu)=r(S,\mu_S).
$$
</div>

### 假设 2（$\ell_1$ 有界平滑性）

存在常数 $B>0$，使对任意 $u,v\in\mathbb R^m$，

<div class="display-equation">
$$
|r(S,u)-r(S,v)|
\leq
B\|u_S-v_S\|_1.
$$
</div>

它表示参数估计误差不会被 reward 任意放大。

### 假设 3（逐坐标单调性）

若 $u_i\leq v_i$ 对所有 $i$ 成立，则

<div class="display-equation">
$$
r(S,u)\leq r(S,v),
\qquad \forall S\in\mathcal I.
$$
</div>

### 假设 4（精确优化 oracle）

给定参数向量 $\theta$，可以调用

<div class="display-equation">
$$
\operatorname{ORACLE}(\theta)
\in
\arg\max_{S\in\mathcal I}r(S,\theta).
$$
</div>

固定一个真实最优组合

<div class="display-equation">
$$
S^\star
\in
\arg\max_{S\in\mathcal I}r(S,\mu),
\qquad
k^\star:=|S^\star|.
$$
</div>

需要始终区分

<div class="display-equation">
$$
1\leq k^\star\leq k\leq m.
$$
</div>

这里 $k$ 是任意可行 super arm 的最大大小，而 $k^\star$ 只是一个最优 super arm 的实际大小。

对任意 $S\in\mathcal I$，定义 gap

<div class="display-equation">
$$
\Delta_S
:=
r(S^\star,\mu)-r(S,\mu)
\geq0.
$$
</div>

累计 pseudo-regret 为

<div class="display-equation">
$$
R_{\mathcal I}(T)
:=
\sum_{t=1}^T
\bigl(r(S^\star,\mu)-r(S(t),\mu)\bigr)
=
\sum_{t=1}^T\Delta_{S(t)}.
$$
</div>

## 二、标准 CTS 为什么产生指数障碍

令 $N_i(t-1)$ 表示第 $t$ 轮开始前基础臂 $i$ 已被观察的次数，$\widehat\mu_i(t-1)$ 表示经验均值。标准 Gaussian CTS 对每个坐标独立抽样：

<div class="display-equation">
$$
X_{i,t}\stackrel{\mathrm{i.i.d.}}{\sim}\mathcal N(0,1),
$$
</div>

并令

<div class="display-equation">
$$
\theta_i(t)
=
\widehat\mu_i(t-1)
+
\frac{X_{i,t}}{\sqrt{N_i(t-1)+1}}.
$$
</div>

随后选择

<div class="display-equation">
$$
S(t)=\operatorname{ORACLE}(\theta(t)).
$$
</div>

单个坐标得到乐观样本并不困难。困难在于，oracle 可能要求最优组合中的许多坐标同时达到一定精度。

### 引理 1（独立联合乐观的乘法代价）

设对每个 $i\in S^\star$，好事件记为 $G_i(t)$。固定历史后，若这些事件独立，且

<div class="display-equation">
$$
\mathbb P(G_i(t)\mid\mathcal F_{t-1})\geq p,
\qquad i\in S^\star,
$$
</div>

则

<div class="display-equation">
$$
\mathbb P\left(
\bigcap_{i\in S^\star}G_i(t)
\,\middle|\,
\mathcal F_{t-1}
\right)
\geq p^{k^\star}.
$$
</div>

**证明.** 条件于历史，事件相互独立，因此

<div class="display-equation">
$$
\mathbb P\left(
\bigcap_{i\in S^\star}G_i(t)
\mid\mathcal F_{t-1}
\right)
=
\prod_{i\in S^\star}
\mathbb P(G_i(t)\mid\mathcal F_{t-1})
\geq p^{k^\star}.
$$
</div>

$\square$

若 $p=1/L$，等待一次联合好事件的平均尺度就是 $L^{k^\star}$。

### 例 1（指数等待的量级）

若每个最优坐标独立地以概率 $1/2$ 足够好，则

<div class="display-equation">
$$
\mathbb P(\text{all good})=2^{-k^\star}.
$$
</div>

当 $k^\star=10$ 时，平均约等待 $1024$ 轮；当 $k^\star=20$ 时，平均等待约为一百万轮。

这说明旧 regret 分析中的指数项并不是纯粹的符号放大，而是与独立联合事件的真实等待时间相对应。

### 方差膨胀为什么不能自动解决一般 CMAB

假设方差膨胀把每个坐标的乐观概率提高到常数 $c\in(0,1)$。若一般非线性 reward 仍要求 $k^\star$ 个坐标共同达到阈值，那么联合概率仍然是

<div class="display-equation">
$$
c^{k^\star}.
$$
</div>

对对称 Gaussian 分布，任意方差下都有

<div class="display-equation">
$$
\mathbb P(Z\geq \mathbb E Z)=\frac12.
$$
</div>

所以单纯增大方差至多改变指数底数，不能消灭指数结构。

在线性 reward 中，一个坐标的大幅上偏可以补偿其他坐标的小幅下偏，因此不一定要求所有坐标同时乐观。这解释了为什么 variance inflation 在特殊线性模型中可能得到多项式 regret，却不能直接推广到一般 CMAB。

## 三、CTS3：保持边缘，重写联合分布

CTS3 每轮只抽一个共同种子

<div class="display-equation">
$$
X_t\sim\mathcal N(0,1),
$$
</div>

然后对所有基础臂定义

<div class="display-equation">
$$
\theta_i(t)
=
\widehat\mu_i(t-1)
+
\frac{X_t}{\sqrt{N_i(t-1)+1}}.
$$
</div>

### 引理 2（每个坐标的边缘分布保持不变）

固定历史 $\mathcal F_{t-1}$ 后，

<div class="display-equation">
$$
\theta_i(t)\mid\mathcal F_{t-1}
\sim
\mathcal N\left(
\widehat\mu_i(t-1),
\frac{1}{N_i(t-1)+1}
\right).
$$
</div>

**证明.** 固定历史以后，$\widehat\mu_i(t-1)$ 与 $N_i(t-1)$ 均为常数。结论直接来自标准正态随机变量的仿射变换。$\square$

因此单独看任一坐标，CTS3 与标准 Gaussian CTS 使用同样的 marginal sampling law。变化发生在坐标之间的依赖关系。

令

<div class="display-equation">
$$
a_i(t):=\frac{1}{\sqrt{N_i(t-1)+1}}.
$$
</div>

则

<div class="display-equation">
$$
\theta_i(t)-\widehat\mu_i(t-1)=a_i(t)X_t.
$$
</div>

故对任意 $i,j$，

<div class="display-equation">
$$
\operatorname{Cov}\bigl(\theta_i(t),\theta_j(t)\mid\mathcal F_{t-1}\bigr)
=a_i(t)a_j(t),
$$
</div>

并且

<div class="display-equation">
$$
\operatorname{Corr}\bigl(\theta_i(t),\theta_j(t)\mid\mathcal F_{t-1}\bigr)=1.
$$
</div>

相关系数为一不意味着不同坐标数值相等。它表示所有中心化波动由同一个随机方向驱动，但每个坐标仍有不同中心和不同尺度。

### 引理 3（联合好事件化为单一阈值）

设对每个 $i\in S^\star$，希望满足

<div class="display-equation">
$$
\theta_i(t)\geq\mu_i-\varepsilon_i.
$$
</div>

定义

<div class="display-equation">
$$
b_i(t)
:=
\sqrt{N_i(t-1)+1}
\bigl(\mu_i-\varepsilon_i-\widehat\mu_i(t-1)\bigr).
$$
</div>

则

<div class="display-equation">
$$
\bigcap_{i\in S^\star}
\{\theta_i(t)\geq\mu_i-\varepsilon_i\}
=
\left\{
X_t\geq\max_{i\in S^\star}b_i(t)
\right\}.
$$
</div>

**证明.** 对每个 $i$，

<div class="display-equation">
$$
\theta_i(t)\geq\mu_i-\varepsilon_i
\iff
X_t\geq b_i(t).
$$
</div>

所有条件同时成立，当且仅当 $X_t$ 超过所有坐标阈值，即超过其最大值。$\square$

因此，标准 CTS 中的概率乘积

<div class="display-equation">
$$
\prod_{i\in S^\star}p_i
$$
</div>

被替换为一个一维 Gaussian tail：

<div class="display-equation">
$$
\overline\Phi\left(
\max_{i\in S^\star}b_i(t)
\right).
$$
</div>

这就是 single-seed coupling 消除指数联合概率的核心。

### 补充引理（共同分位数达到最大的联合乐观概率）

这一点可以比“相关系数等于一”说得更精确。设第 $i$ 个坐标的边缘分布为连续分布 $F_i$，给定阈值 $a_i$，记

<div class="display-equation">
$$
p_i:=\mathbb P(X_i\geq a_i).
$$
</div>

对任意具有这些边缘分布的联合构造，都必有

<div class="display-equation">
$$
\mathbb P(X_i\geq a_i,\ \forall i)
\leq
\min_i p_i,
$$
</div>

因为交事件包含在每一个单独事件中。若使用共同分位数耦合

<div class="display-equation">
$$
X_i=F_i^{-1}(U),
\qquad U\sim\operatorname{Unif}[0,1],
$$
</div>

则这个上界被恰好达到：

<div class="display-equation">
$$
\mathbb P(X_i\geq a_i,\ \forall i)
=
\min_i p_i.
$$
</div>

证明. 对连续分布，事件 $X_i\geq a_i$ 等价于

<div class="display-equation">
$$
U\geq F_i(a_i).
$$
</div>

因此所有事件同时发生等价于

<div class="display-equation">
$$
U\geq \max_i F_i(a_i).
$$
</div>

于是

<div class="display-equation">
$$
\begin{aligned}
\mathbb P(X_i\geq a_i,\ \forall i)
&=1-\max_iF_i(a_i)\\
&=\min_i\{1-F_i(a_i)\}\\
&=\min_i p_i.
\end{aligned}
$$
</div>

证毕。

这个引理给出了 CTS3 的概率论本质。对“每个坐标都超过各自阈值”这一类递增事件，共同分位数耦合把联合成功概率从独立情形下的乘积

<div class="display-equation">
$$
\prod_i p_i
$$
</div>

提高到了所有合法耦合能够达到的最大尺度

<div class="display-equation">
$$
\min_i p_i.
$$
</div>

因此，single-seed 不是一个随意的相关采样技巧，而是在这类联合阈值事件上达到 Fréchet 上界的极端正相关耦合。

### 一般 posterior 的逆 CDF 版本

若基础臂 $i$ 的 posterior CDF 为 $F_{i,t}$，抽取

<div class="display-equation">
$$
U_t\sim\operatorname{Unif}[0,1]
$$
</div>

并令

<div class="display-equation">
$$
\theta_i(t)=F_{i,t}^{-1}(U_t),
\qquad i\in[m],
$$
</div>

则每个坐标仍具有正确的 posterior 边缘分布，而所有坐标关于 $U_t$ 共单调变化。Gaussian single-seed 只是这一构造的显式特例。

## 四、论文的深层解读：问题、算法与证明思想

### 4.1 原论文面对的开放问题

标准 CTS 已知可以获得关于时间 $T$ 的对数 regret，但一般分析中带有形如 $L^{k^\star}$ 的项。这里真正危险的不是 $T$，而是最优组合大小 $k^\star$：当 $k^\star$ 增大时，即使时间长度固定，理论常数也可能指数爆炸。

已有 variance-inflated CTS 能在线性组合 reward 中把指数依赖改成多项式依赖。原因是线性求和允许一个坐标的大幅上偏补偿其他坐标的轻微下偏。但一般 CMAB 的 reward 可能是覆盖概率、可靠性、最小值、非线性聚合等形式；这时一个坐标的超额乐观未必能补偿另一个关键坐标的低估。

因此，原论文的问题不是“怎样重新证明线性 CTS”，而是：

<div class="display-equation">
$$
\text{能否对一般 monotone、$\ell_1$-smooth reward，}
\quad
\text{构造具有多项式 regret 的 Thompson-style 算法？}
$$
</div>

### 4.2 算法贡献为何看似简单却并不浅

CTS3 的伪代码只改动标准 CTS 的随机化方式。标准 CTS 使用

<div class="display-equation">
$$
X_{1,t},\ldots,X_{m,t}
\stackrel{\mathrm{i.i.d.}}{\sim}\mathcal N(0,1),
$$
</div>

而 CTS3 使用一个共同变量

<div class="display-equation">
$$
X_t\sim\mathcal N(0,1).
$$
</div>

算法其余部分——经验均值、观测计数、oracle、semi-bandit 更新——都没有改变。

这种设计有两个同时成立、通常很难兼得的性质。

第一，每个坐标的采样边缘保持不变。因此从单臂角度看，它仍使用原来的 Gaussian posterior sampling law。

第二，最优组合的联合乐观事件不再需要多个独立随机变量同时成功，而被压缩成一个 seed 超过最大阈值的事件。

论文的关键思想可以概括为：

<div class="display-equation">
$$
\text{不修改 uncertainty 的边缘大小，}
\quad
\text{只修改 uncertainty 的协同方式。}
$$
</div>

这比单纯增加方差更精确。方差膨胀扩大每个坐标的波动，却仍可能保留联合事件的乘法结构；single-seed 直接改变乘法结构本身。

### 4.3 证明为什么比算法困难得多

算法引入强相关以后，标准 CTS 证明中的独立性工具立即失效。原论文必须同时完成两件看似矛盾的工作：利用强正相关提高联合乐观概率，又在 regret 分析中控制这种相关性对 oracle 选择造成的复杂影响。

证明的第一步是 gap bucketing。把所有次优组合按照 gap 大小分层：

<div class="display-equation">
$$
\mathcal M_r
=
\{S:2^{-r}&lt;\Delta_S\leq2^{-r+1}\}.
$$
</div>

这样，在固定层 $r$ 内，所有错误选择的单轮损失都处于同一尺度。证明可以先控制该层被选择多少次，再乘以这一层的 gap。

第二步是使用 posterior 的上下分位数区分两类错误。若一个次优组合的上分位 reward 已明显超过真实 reward，则它被选择可以归因于 selected side 的过估；若它没有明显过估却仍被选择，则问题只能来自更优组合，尤其是 $S^\star$，被低估。

第三步是对 selected-side 过估做 charging。只要一个次优组合仍能靠上分位数获得过高分数，其中至少有一个基础臂尚未达到相应采样阈值。每选择一次该组合，就会使某个低计数臂的 $N_i$ 增加一次。因此这类坏轮次可以支付给有限的计数增量，而不是逐轮无界累积。

第四步是处理最难的 optimal-side underestimation。论文构造 rescue event：共同 seed 足够高时，最优组合中的所有关键坐标同时不再严重低估。但在 single-seed 世界中，高 seed 同时抬高所有竞争组合，因此不能简单地说“$S^\star$ 乐观就必然被选”。论文为此引入 seed slicing，把实数轴上的 seed 按 oracle 最终选择的动作类型划分，并进一步区分能被计数收费的 rescued rounds 与没有直接更新最优坐标的轮次。

第五步是 inverse-probability reduction。若某个 critical round 的 rescue 概率是 $p_t$，则 critical rounds 的期望数量可以转化为 rescued rounds 上 $1/p_t$ 的加权和。问题因此变成：$p_t$ 可能有多小，以及极小的 $p_t$ 能持续多久。

第六步是 time-uniform anti-concentration 与 peeling。论文不是给 $p_t$ 一个极粗的统一下界，而是把 $p_t$ 的可能取值分成几何区间。在每个区间里，极小 rescue 概率会迫使某个经验均值在某个采样时刻出现异常下偏；时间一致浓缩不等式说明这种异常不能频繁发生。把所有区间求和以后，得到关于 $k^\star$ 的多项式账本。

因此，原论文的技术创新不只是“用一个 seed”。更完整的证明思想是：

<div class="display-equation">
$$
\text{共同种子的一维阈值结构}
\;+
\text{gap 分层}
\;+
\text{seed slicing}
\;+
\text{inverse-probability charging}
\;+
\text{time-uniform peeling}.
$$
</div>

### 4.4 主定理逐项解释

原论文给出的实例依赖界为

<div class="display-equation">
$$
\mathbb E[R_{\mathcal I}(T)]
=
O\left(
\frac{mkk^\star B^2}{\Delta_{\min}}
\operatorname{polylog}(T,m,\Delta_{\min}^{-1})
\right).
$$
</div>

这里每个量的含义都不同。

$m$ 衡量基础臂总数；$k$ 衡量每轮最多观察多少个基础臂；$k^\star$ 是最优组合的实际大小；$B$ 衡量坐标误差对 reward 的放大；$\Delta_{\min}$ 衡量最难区分的最优与次优动作之间的距离。

最重要的变化是：旧界中的 $L^{k^\star}$ 被 $k^\star$ 的多项式依赖替代。只要 $k^\star$ 增长，指数与多项式的差别不是常数改进，而是可扩展性层面的改变。

原论文的最坏情形界为

<div class="display-equation">
$$
\mathbb E[R_{\mathcal I}(T)]
=
O\left(
B\sqrt{mkk^\star T}
\operatorname{polylog}(T,m)
\right).
$$
</div>

已知一般组合 semi-bandit 的最坏情形下界为

<div class="display-equation">
$$
\Omega(\sqrt{mkT}).
$$
</div>

所以论文完成了从指数障碍到近最优多项式界的主要突破，但仍留下 $\sqrt{k^\star}$ 的差距。这个剩余差距就是后续笔记研究的起点。

### 4.5 实验应该怎样解读

原论文在 stochastic maximum spanning tree 上比较 CUCB、标准 CTS 与 CTS3。图中的节点数依次为 $10,20,30,50,100$，随机图采用较高连接概率，因此候选边和可行生成树数量随着节点数迅速增长。

每条边是一个 base arm，每棵生成树是一个 super arm。因为一棵 $n$ 个节点的生成树包含 $n-1$ 条边，所以节点数增加时，最优组合大小也同步增加。这正好放大标准 CTS 的弱点：独立边样本很难形成一棵在整体上协调乐观的树。

曲线所表达的现象是：在小图上三种方法差距有限；随着图规模增加，标准 CTS 在早期和中期积累 regret 更快，CTS3 的优势变得明显。CUCB 是稳定的 frequentist 基线，而 CTS3 在这些实验中取得更低的累计 regret。

但实验不能被过度解释。首先，maximum spanning tree 使用线性 reward，而主定理覆盖更一般的 monotone smooth reward；实验验证的是 coupling 的探索机制，而不是替代一般定理。其次，实验展示有限 horizon 与特定图分布，不能单独证明最坏情形最优性。最后，CTS3 的优势随组合规模增长而扩大，与理论动机一致，却不能由曲线反推出精确的 $k^\star$ 依赖指数。

## 五、一步步追踪原证明中的 $k^\star$

### 5.1 为什么先做 gap 分桶

定义

<div class="display-equation">
$$
\mathcal M_r
:=
\left\{
S\in\mathcal I:
2^{-r}&lt;\Delta_S\leq2^{-r+1}
\right\},
$$
</div>

并令

<div class="display-equation">
$$
\delta_r\asymp\frac{2^{-r}}{B}.
$$
</div>

固定 $r$ 后，所有 $S\in\mathcal M_r$ 的单轮 regret 都与 $B\delta_r$ 同阶。因此只要证明这一层至多被错误选择 $N_r$ 次，就有

<div class="display-equation">
$$
R_{\mathcal M_r}(T)
\lesssim
B\delta_r N_r.
$$
</div>

这一步把“不同错误损失不同”的问题变成“固定损失尺度下数坏轮次”。

### 5.2 Easy term：次优组合自身被过估

设 $\mathcal W_r(t)$ 表示在第 $t$ 轮没有被明显过估的 bucket-$r$ 组合集合。若

<div class="display-equation">
$$
S(t)\in\mathcal M_r,
\qquad
S(t)\notin\mathcal W_r(t),
$$
</div>

则所选组合中至少有一些坐标的 posterior 上分位数仍远高于真实均值。证明按 dyadic sample scale 定义阈值

<div class="display-equation">
$$
\tau_{r,s}
\asymp
\frac{s^2\log(Tmk)}{\delta_r^2},
$$
</div>

其中 $s$ 表示一个组合中可能共同贡献误差的坐标数尺度。若一个臂已有至少 $\tau_{r,s}$ 次观测，其分位数误差应小于 $\delta_r/s$。因此若总过估仍超过 $\delta_r$，必有足够多的臂尚未达到相应阈值。

每个此类坏轮次都会选择这些臂并使其计数增加。对每个臂、每个 dyadic scale 最多收费 $\tau_{r,s}$ 次，最后得到

<div class="display-equation">
$$
N_r^{\mathrm{over}}
\lesssim
\frac{m\tau_r\log k}{k},
$$
</div>

其中

<div class="display-equation">
$$
\tau_r
\asymp
\frac{k^2\log(Tmk)}{\delta_r^2}.
$$
</div>

这一部分没有 $k^\star$ 的核心困难，因为收费对象是本轮真正被选中的基础臂。

### 5.3 Hard term：次优组合没有过估，最优组合却被低估

真正困难的轮次满足

<div class="display-equation">
$$
S(t)\in\mathcal M_r,
\qquad
S(t)\in\mathcal W_r(t).
$$
</div>

既然被选的次优组合没有明显过估，它还能胜过更优组合，只能说明 $S^\star$ 或某个接近最优的组合在 sampled world 中被压低。

原证明定义逐坐标 rescue

<div class="display-equation">
$$
E_{\mathrm{under}}(t)
:=
\bigcap_{i\in S^\star}
\left\{
\theta_i(t)
\geq
\mu_i-\frac{c\delta_r}{k^\star}
\right\}.
$$
</div>

分母 $k^\star$ 来自 $\ell_1$ smoothness：若每个最优坐标最多低估 $O(\delta_r/k^\star)$，则总坐标误差最多为 $O(\delta_r)$，从而最优组合 reward 只损失 $O(B\delta_r)$。

### 5.4 Single-seed 已经消除了概率乘积

由共同种子公式，事件 $E_{\mathrm{under}}(t)$ 等价于

<div class="display-equation">
$$
X_t
\geq
\max_{i\in S^\star}
\sqrt{N_i(t-1)+1}
\left(
\mu_i-\frac{c\delta_r}{k^\star}
-
\widehat\mu_i(t-1)
\right).
$$
</div>

所以这里没有 $p^{k^\star}$。若只看本轮条件概率，single-seed 已经完成了最重要的一步。

### 5.5 为什么证明中仍需要对所有最优坐标做时间一致控制

问题是阈值由最差的那个最优坐标决定。为了控制所有 critical rounds 上的最坏阈值，原证明需要保证：对每个 $i\in S^\star$，在其观测次数 $1,\ldots,\tau_r$ 的任何时刻，经验均值都不能异常低。

于是构造事件

<div class="display-equation">
$$
A(x)
:=
\bigcap_{i\in S^\star}
\bigcap_{1\leq n\leq\tau_r}
\left\{
\widehat\mu_{i,n}
\geq
\mu_i-
\sqrt{\frac{2\log(k^\star x)}{n+1}}
\right\}.
$$
</div>

对固定 $i$，时间一致 Gaussian 下尾界给出近似

<div class="display-equation">
$$
\mathbb P\left(
\exists n\leq\tau_r:
\widehat\mu_{i,n}
&lt;
\mu_i-
\sqrt{\frac{2\log(k^\star x)}{n+1}}
\right)
\lesssim
\frac{\operatorname{polylog}(k^\star x,\tau_r)}{k^\star x}.
$$
</div>

再对 $k^\star$ 个最优坐标求和，得到

<div class="display-equation">
$$
\mathbb P(A(x)^c)
\lesssim
\frac{\operatorname{polylog}(k^\star x,\tau_r)}{x}.
$$
</div>

注意，$k^\star$ 虽在 union bound 外面被消掉，却已经进入阈值中的 $\log(k^\star x)$。这个阈值会进一步影响 Gaussian tail，从而使 inverse-rescue 代价具有 $k^\star$ 量级。

### 5.6 Inverse-rescue 账本怎样产生 $k^\star$

令 $p_t$ 表示第 $t$ 个 critical round 的条件 rescue 概率。通过条件期望恒等式，critical rounds 数量可以写成

<div class="display-equation">
$$
\mathbb E|\mathcal L_r|
=
\mathbb E
\sum_{t\in\mathcal L_r:\,E_{\mathrm{under}}(t)}
\frac1{p_t}.
$$
</div>

rescued rounds 的数量可以由 sample-count charging 控制，但每个 rescued round 要乘 $1/p_t$。对 $p_t$ 做几何 peeling，并用上面的时间一致尾界后，原证明得到

<div class="display-equation">
$$
\mathbb E\left[
\frac1{\min_{t\in\mathcal L_r}p_t}
\right]
\lesssim
k^\star\operatorname{polylog}(T,m,k).
$$
</div>

因此 hard term 的 bucket 账本中保留一个 $k^\star$。

### 5.7 从 instance-dependent 界到 worst-case 界

把各 bucket 求和后，得到近似形式

<div class="display-equation">
$$
\mathbb E[R_{\mathcal I}(T)]
\lesssim
\frac{B^2mkk^\star}{\Delta_{\min}}
\operatorname{polylog}(T,m,\Delta_{\min}^{-1}).
$$
</div>

要去掉对 $\Delta_{\min}$ 的依赖，取任意截断尺度 $\varepsilon>0$。gap 不超过 $\varepsilon$ 的轮次总 regret 最多为

<div class="display-equation">
$$
\varepsilon T.
$$
</div>

gap 大于 $\varepsilon$ 的部分由实例依赖界控制为

<div class="display-equation">
$$
\frac{CB^2mkk^\star}{\varepsilon}
\operatorname{polylog}(T,m).
$$
</div>

所以

<div class="display-equation">
$$
\mathbb E[R_{\mathcal I}(T)]
\lesssim
\varepsilon T
+
\frac{CB^2mkk^\star}{\varepsilon}
\operatorname{polylog}(T,m).
$$
</div>

令两项平衡，即取

<div class="display-equation">
$$
\varepsilon
\asymp
B\sqrt{\frac{mkk^\star}{T}}
\operatorname{polylog}^{1/2}(T,m),
$$
</div>

得到

<div class="display-equation">
$$
\mathbb E[R_{\mathcal I}(T)]
\lesssim
B\sqrt{mkk^\star T}
\operatorname{polylog}(T,m).
$$
</div>

因此，最坏情形界中的 $\sqrt{k^\star}$ 是实例依赖账本中 $k^\star$ 经过 gap 截断平方根化后的结果。

### 5.8 为什么不能把问题简单归结为 union bound

到这里容易产生一个误判：既然 $k^\star$ 在逐坐标时间一致控制中出现，只要改用 reward-level rescue，不再逐坐标 union bound，是否就一定能完成 $k^\star$\-free 证明？

答案是否定的。

union bound 解释了原证明为什么留下 $k^\star$，但它还没有回答这个因子是否只是证明松弛。要回答后一个问题，必须分析一次 reward-level rescue 发生以后，oracle 是否真的更新了造成低估的最优坐标。

如果 rescue 发生却被一个不包含低估 witness 的竞争动作截走，那么本轮没有修复最优侧统计量。下一轮重新抽 seed 后，刚才的 optimism 完全消失。一个实例若能让这种“rescue 被消费但 witness 不更新”的过程连续发生 $\Theta(k^\star)$ 次，那么即使完全避开逐坐标 union bound，也仍然无法获得统一的 $k^\star$\-free 界。

下一节的 reward-level rescue 与 blocker 反例正是为了回答这个更深的问题。

## 六、Reward-level rescue 与真正的阻碍

固定历史，把 seed 写成普通变量 $x$：

<div class="display-equation">
$$
\theta_i(t;x)
=
\widehat\mu_i(t-1)
+
\frac{x}{\sqrt{N_i(t-1)+1}}.
$$
</div>

由单调性，若 $x\leq y$，则

<div class="display-equation">
$$
\theta(t;x)\leq\theta(t;y)
\quad\Longrightarrow\quad
r(S^\star,\theta(t;x))
\leq
r(S^\star,\theta(t;y)).
$$
</div>

定义 reward-level rescue event

<div class="display-equation">
$$
\mathcal R_{r,t}
:=
\left\{
 r(S^\star,\theta(t;X_t))
\geq
r(S^\star,\mu)-c_0B\delta_r
\right\}.
$$
</div>

由于左侧关于 $X_t$ 单调，存在一个由历史决定的阈值 $q_{r,t}$，使

<div class="display-equation">
$$
\mathcal R_{r,t}=\{X_t\geq q_{r,t}\}.
$$
</div>

记

<div class="display-equation">
$$
p_{r,t}
:=
\mathbb P(\mathcal R_{r,t}\mid\mathcal F_{t-1})
=
\overline\Phi(q_{r,t}).
$$
</div>

### 引理 4（Inverse-rescue 恒等式）

若 $\mathcal L_r$ 是由第 $t$ 轮之前历史决定的 critical rounds 集合，则

<div class="display-equation">
$$
\mathbb E|\mathcal L_r|
=
\mathbb E
\sum_{t\in\mathcal L_r:\,\mathcal R_{r,t}}
\frac1{p_{r,t}}.
$$
</div>

**证明.** 对每轮先固定历史。由于 $\mathbf 1\{t\in\mathcal L_r\}$ 是 $\mathcal F_{t-1}$\-可测的，

<div class="display-equation">
$$
\mathbb E\left[
\frac{
\mathbf 1\{t\in\mathcal L_r\}
\mathbf 1\{\mathcal R_{r,t}\}
}{p_{r,t}}
\,\middle|\,
\mathcal F_{t-1}
\right]
=
\mathbf 1\{t\in\mathcal L_r\}.
$$
</div>

对 $t$ 求和并再次取期望即得。$\square$

这条公式完全没有对 $S^\star$ 内部坐标做 union bound。问题在于：rescue 发生以后，oracle 未必选择能够更新低估 witness 的动作。

一次高 seed 会同时抬高最优组合和大量竞争组合。若某个竞争组合截走了本轮选择，最优组合在 reward 层面虽已得到 rescue，但导致低估的坐标并没有被观察，下一轮又会重新抽取 seed，刚才的乐观性也随之消失。

这类轮次称为 non-full blocker rounds。

定义 release number

<div class="display-equation">
$$
\rho
:=
\sup
\left\{
\text{某个低估 witness 被真正更新前，连续发生的 blocker 次数}
\right\}.
$$
</div>

若能够证明

<div class="display-equation">
$$
\rho=\operatorname{polylog}(T,m,k),
$$
</div>

则 reward-level rescue 可以闭合为 $k^\star$\-free 证明。一般 CMAB 假设却允许

<div class="display-equation">
$$
\rho=\Theta(k^\star).
$$
</div>

下面给出严格反例。

## 七、一般 CMAB 下的 Blocker 反例

### 定理 1（原始 CTS3 的 $k^\star$\-free no-go）

不存在与问题规模无关的固定常数 $A,a<\infty$，使得对所有满足原始 mean-dependent、monotone、$\ell_1$\-smooth 与 exact-oracle 假设的实例和所有 $T$，原始 CTS3 都满足

<div class="display-equation">
$$
\mathbb E[R_{\mathcal I}(T)]
\leq
AB\sqrt{mkT}
\bigl\{\log(eTmk)\bigr\}^a.
$$
</div>

### 反例构造

取整数 $K\to\infty$，定义

<div class="display-equation">
$$
u_K:=\sqrt{2\log K},
\qquad
\Delta_K:=\frac{c_\Delta}{u_K},
$$
</div>

其中 $c_\Delta>0$ 足够小。再令

<div class="display-equation">
$$
h_K:=\sqrt2\,u_K-3\Delta_K,
\qquad
C_K:=h_K+\Delta_K
=
\sqrt2\,u_K-2\Delta_K.
$$
</div>

基础臂分为三类：

<div class="display-equation">
$$
\{1,\ldots,K\}
\cup\{f\}
\cup\{d_1,\ldots,d_K\}.
$$
</div>

只允许选择

<div class="display-equation">
$$
S^\star:=\{1,\ldots,K\},
\qquad
F:=\{f\},
\qquad
D_\ell:=\{d_\ell\},\quad \ell\in[K].
$$
</div>

因此

<div class="display-equation">
$$
m=2K+1,
\qquad
k=K,
\qquad
k^\star=K.
$$
</div>

所有真实均值均取零。定义 reward

<div class="display-equation">
$$
r(S^\star,x):=\min_{1\leq i\leq K}x_i,
$$
</div>

<div class="display-equation">
$$
r(F,x):=-\Delta_K,
$$
</div>

<div class="display-equation">
$$
r(D_\ell,x):=x_{d_\ell}-C_K.
$$
</div>

若希望所有 reward 非负，可以对三类 reward 同时加上常数 $C_K$；若还需要统一归一化，也可以再除以 $C_K+1$。正仿射变换不改变 oracle 的排序，只会按同一比例缩放 gap 与 smoothness 常数。

$S^\star$ 的收益由最差坐标决定；$F$ 是低 seed 时的安全 blocker；尚未观测的 $D_\ell$ 是高 seed 时的 decoy blocker。

### 引理 5（合法性）

上述实例满足均值依赖、逐坐标单调性、$B=1$ 的 $\ell_1$ bounded smoothness，并存在 exact oracle。

**证明.** 均值依赖显然成立。若 $x\leq y$，则

<div class="display-equation">
$$
\min_{i\leq K}x_i
\leq
\min_{i\leq K}y_i,
\qquad
x_{d_\ell}-C_K
\leq
y_{d_\ell}-C_K.
$$
</div>

故单调性成立。对 $S^\star$，

<div class="display-equation">
$$
\left|
\min_{i\leq K}x_i-
\min_{i\leq K}y_i
\right|
\leq
\max_{i\leq K}|x_i-y_i|
\leq
\sum_{i=1}^K|x_i-y_i|.
$$
</div>

对 decoy，

<div class="display-equation">
$$
|r(D_\ell,x)-r(D_\ell,y)|
=|x_{d_\ell}-y_{d_\ell}|.
$$
</div>

安全臂 reward 为常数。因此可统一取 $B=1$。精确比较有限个可行动作即可得到 exact oracle。$\square$

在真实均值 $\mu=0$ 处，

<div class="display-equation">
$$
r(S^\star,\mu)=0,
\qquad
r(F,\mu)=-\Delta_K,
\qquad
r(D_\ell,\mu)=-C_K.
$$
</div>

故 $S^\star$ 唯一最优，最小正 gap 为 $\Delta_K$。

### 第一轮产生低估 witness

初始时所有坐标的 sample 都等于共同 seed：

<div class="display-equation">
$$
\theta_i(1)=X_1.
$$
</div>

若 $X_1>-\Delta_K$，则

<div class="display-equation">
$$
r(S^\star,\theta(1))=X_1>-\Delta_K=r(F,\theta(1)),
$$
</div>

且

<div class="display-equation">
$$
r(D_\ell,\theta(1))=X_1-C_K&lt;X_1.
$$
</div>

所以第一轮选择 $S^\star$，并观察

<div class="display-equation">
$$
Y_i:=Z_i(1)
\stackrel{\mathrm{i.i.d.}}{\sim}
\mathcal N(0,1),
\qquad i\in[K].
$$
</div>

定义

<div class="display-equation">
$$
M_K:=-\min_{1\leq i\leq K}Y_i.
$$
</div>

坏初始化事件为

<div class="display-equation">
$$
H_K
:=
\left\{
X_1>-\Delta_K,
\quad
M_K\in[u_K,u_K+u_K^{-1}]
\right\}.
$$
</div>

Gaussian 极值估计给出

<div class="display-equation">
$$
\mathbb P(H_K)\geq\frac{c}{u_K}.
$$
</div>

这不是指数小事件。$K$ 个标准正态样本中的最小值自然会达到约 $-\sqrt{2\log K}$。

### 引理 6（全 seed 阻断）

在事件 $H_K$ 上，只要仍存在一个未观测 decoy，则对任意 seed $x\in\mathbb R$，oracle 都不会选择 $S^\star$。

**证明.** 在 $H_K$ 上存在最差坐标 $j^\star$，其第一轮经验均值为 $-M_K$。只要 $S^\star$ 此后未被选择，该坐标就不会更新。固定任意后续 seed $x$，

<div class="display-equation">
$$
r(S^\star,\theta(x))
=-M_K+\frac{x}{\sqrt2}.
$$
</div>

若 $x<h_K$，则由 $M_K\geq u_K$，

<div class="display-equation">
$$
r(S^\star,\theta(x))
\leq
-u_K+\frac{h_K}{\sqrt2}
=
-\frac{3}{\sqrt2}\Delta_K
&lt;
-\Delta_K
=
r(F,\theta(x)).
$$
</div>

因此低 seed 时，安全臂 $F$ 击败 $S^\star$。

若 $x\geq h_K$，取任一未观测 decoy。因为 $N_{d_\ell}=0$ 且 $\widehat\mu_{d_\ell}=0$，

<div class="display-equation">
$$
r(D_\ell,\theta(x))=x-C_K.
$$
</div>

于是

<div class="display-equation">
$$
\begin{aligned}
r(D_\ell,\theta(x))-r(S^\star,\theta(x))
&=
\left(1-\frac1{\sqrt2}\right)x+M_K-C_K\\
&\geq
\left(1-\frac1{\sqrt2}\right)h_K+u_K-C_K\\
&=
\left(-1+\frac3{\sqrt2}\right)\Delta_K
>0.
\end{aligned}
$$
</div>

因此高 seed 时，未观测 decoy 击败 $S^\star$。两个区间覆盖整条实轴，结论成立。$\square$

最关键的斜率比较是

<div class="display-equation">
$$
\text{冻结后的最优瓶颈斜率}=\frac1{\sqrt2},
\qquad
\text{未观测 decoy 斜率}=1.
$$
</div>

seed 越大，decoy 相对最优组合反而越有优势。一次高 seed 没有修复低估 witness，而是被 decoy 消耗。

### 清空 decoy 的时间尺度

定义高 seed 概率

<div class="display-equation">
$$
p_K:=\mathbb P(X_t\geq h_K)=\overline\Phi(h_K).
$$
</div>

Mills ratio 给出

<div class="display-equation">
$$
\frac{c}{K^2u_K}
\leq
p_K
\leq
\frac{C}{K^2u_K}.
$$
</div>

低 seed 轮不会首次选择 decoy；每个高 seed 轮至多首次观察一个 decoy。要清空 $K$ 个 decoys，至少需要 $K$ 次高 seed。因此典型等待尺度为

<div class="display-equation">
$$
\frac{K}{p_K}
\asymp
K^3u_K.
$$
</div>

取

<div class="display-equation">
$$
T_K
:=
1+
\left\lfloor
\eta\frac{K}{p_K}
\right\rfloor,
\qquad
0&lt;\eta&lt;\frac14.
$$
</div>

令第 $2$ 至第 $T_K$ 轮的高 seed 次数为

<div class="display-equation">
$$
N_K^+
:=
\sum_{t=2}^{T_K}
\mathbf 1\{X_t\geq h_K\}.
$$
</div>

则

<div class="display-equation">
$$
\mathbb E[N_K^+]
\leq
\eta K.
$$
</div>

由 Markov 不等式，

<div class="display-equation">
$$
\mathbb P(N_K^+\geq K)
\leq
\eta,
\qquad
\mathbb P(N_K^+&lt;K)
\geq
1-\eta.
$$
</div>

在事件

<div class="display-equation">
$$
E_K:=H_K\cap\{N_K^+&lt;K\}
$$
</div>

上，始终至少有一个 decoy 未被观测，因此第 $2$ 至第 $T_K$ 轮均不选择 $S^\star$，且

<div class="display-equation">
$$
\mathbb P(E_K)
\geq
\frac{c}{u_K}.
$$
</div>

### regret 下界

每个非最优动作的 gap 至少为 $\Delta_K=c_\Delta/u_K$。因此

<div class="display-equation">
$$
\begin{aligned}
\mathbb E[R_{\mathcal I}(T_K)]
&\geq
\mathbb P(E_K)(T_K-1)\Delta_K\\
&\geq
\frac{c}{u_K}
\cdot
\frac{cK}{p_K}
\cdot
\frac{c}{u_K}\\
&\geq
c\frac{K^3}{u_K}.
\end{aligned}
$$
</div>

三个因子分别来自

<div class="display-equation">
$$
\underbrace{u_K^{-1}}_{\text{坏初始化概率}}
\times
\underbrace{K/p_K}_{\text{阻断持续时间}}
\times
\underbrace{u_K^{-1}}_{\text{每轮最小 gap}}.
$$
</div>

另一方面，

<div class="display-equation">
$$
m\asymp K,
\qquad
k=K,
\qquad
T_K\asymp K^3u_K.
$$
</div>

故不含 $k^\star$ 的目标尺度为

<div class="display-equation">
$$
\sqrt{mkT_K}
\asymp
K^{5/2}u_K^{1/2}.
$$
</div>

若存在统一的 $\widetilde O(\sqrt{mkT})$ 上界，则必须有

<div class="display-equation">
$$
\frac{K^3}{u_K}
\leq
C K^{5/2}u_K^{1/2}(\log K)^a,
$$
</div>

即

<div class="display-equation">
$$
K^{1/2}
\leq
C u_K^{3/2}(\log K)^a
=
C(\log K)^{a+3/4},
$$
</div>

与 $K\to\infty$ 矛盾。

### 反例的边界

这个反例否定的是以下三个条件同时成立：

<div class="display-equation">
$$
\text{算法保持原始 CTS3}
+
\text{假设保持一般 CMAB}
+
\text{结论对所有实例统一成立}.
$$
</div>

它不否定原论文已有的 $\widetilde O(\sqrt{mkk^\star T})$ 上界，不否定 MNL、线性或其他结构化子类中摘除 $k^\star$，也不否定修改算法后获得更优界。

事实上，在该反例上，原论文尺度为

<div class="display-equation">
$$
\sqrt{mkk^\star T_K}
\asymp
K^3u_K^{1/2},
$$
</div>

而下界只有 $K^3/u_K$，二者完全兼容。

## 八、结构性正面结果：MNL-Bandit

一般 CMAB 的 no-go 不意味着结构化模型无法摘除 $k^\star$。MNL-Bandit 是一个典型正例。

### MNL 模型

有 $N$ 个商品，商品 $i$ 的 attraction parameter 为 $v_i>0$，outside option 标准化为 $v_0=1$。展示 assortment $S$ 后，用户选择 $i\in S$ 的概率为

<div class="display-equation">
$$
p_i(S;v)
=
\frac{v_i}{1+\sum_{j\in S}v_j},
$$
</div>

不购买概率为

<div class="display-equation">
$$
p_0(S;v)
=
\frac{1}{1+\sum_{j\in S}v_j}.
$$
</div>

若商品收入为 $r_i\in[0,1]$，则期望 revenue 为

<div class="display-equation">
$$
R(S,v)
=
\frac{\sum_{i\in S}r_iv_i}
{1+\sum_{i\in S}v_i}.
$$
</div>

### 一般 revenue MNL 并非逐坐标单调

固定 $S$，对 $v_i$ 求导：

<div class="display-equation">
$$
\frac{\partial R(S,v)}{\partial v_i}
=
\frac{r_i-R(S,v)}{1+\sum_{j\in S}v_j}.
$$
</div>

若 $r_i<R(S,v)$，增加该低收入商品的 attraction 反而会降低总 revenue。因此不能把一般 revenue MNL 不加说明地当作一般 CMAB 中的逐坐标单调 reward。

真正可用的是最优 assortment 上的 restricted monotonicity。

### 引理 7（最优 assortment 的 restricted monotonicity）

设

<div class="display-equation">
$$
S^\star
\in
\arg\max_{|S|\leq K}R(S,v).
$$
</div>

若 $w_i\geq v_i$ 对所有 $i$ 成立，则

<div class="display-equation">
$$
R(S^\star,w)
\geq
R(S^\star,v).
$$
</div>

**证明.** 记 $R^\star:=R(S^\star,v)$。先证明每个 $i\in S^\star$ 都满足 $r_i\geq R^\star$。否则若 $r_i<R^\star$，删除商品 $i$ 后

<div class="display-equation">
$$
R(S^\star\setminus\{i\},v)-R^\star
=
\frac{v_i(R^\star-r_i)}
{1+\sum_{j\in S^\star}v_j-v_i}
>0,
$$
</div>

与 $S^\star$ 最优矛盾。

又因为

<div class="display-equation">
$$
\sum_{i\in S^\star}(r_i-R^\star)v_i
=R^\star,
$$
</div>

且每个系数 $r_i-R^\star\geq0$，故

<div class="display-equation">
$$
\sum_{i\in S^\star}(r_i-R^\star)w_i
\geq
R^\star.
$$
</div>

展开并除以正分母，即得 $R(S^\star,w)\geq R^\star$。$\square$

### Epoch 与相关采样

经典 MNL-Bandit 算法在一个 epoch 内反复展示同一 assortment，直到第一次出现 outside option。若该 epoch 选择 $S_\ell$，则

<div class="display-equation">
$$
\mathbb E[|E_\ell|\mid S_\ell]
=
1+
\sum_{i\in S_\ell}v_i
\leq K+1.
$$
</div>

同时，epoch 内商品 $i$ 被购买的次数给出 $v_i$ 的无偏观测。这使复杂的 multinomial 反馈可以逐商品更新。

经典相关采样算法并不是原始 CTS3。它在每个 epoch 生成 $K$ 组共同 Gaussian seeds：

<div class="display-equation">
$$
Z_\ell^{(j)}\sim\mathcal N(0,1),
\qquad j=1,\ldots,K,
$$
</div>

同一组内所有商品共享一个 seed：

<div class="display-equation">
$$
\mu_i^{(j)}(\ell)
=
\widehat v_i(\ell)
+Z_\ell^{(j)}\widehat\sigma_i(\ell).
$$
</div>

再逐坐标取最大：

<div class="display-equation">
$$
\mu_i(\ell)
=
\max_{1\leq j\leq K}
\mu_i^{(j)}(\ell).
$$
</div>

若某一组满足 $Z_\ell^{(j)}\geq z$ 就能同时抬高最优 assortment 的全部坐标，则一组成功概率为

<div class="display-equation">
$$
p_z=\overline\Phi(z),
$$
</div>

而 $K$ 组全部失败的概率为

<div class="display-equation">
$$
(1-p_z)^K
\leq
e^{-Kp_z}.
$$
</div>

独立坐标的 $2^{-K}$ 被“一个共同种子成功，再做 $K$ 组 boosting”所取代。

### MNL regret 公式链

令总 epoch 数为 $L$。加上再减去 sampled instance 上的 revenue：

<div class="display-equation">
$$
\begin{aligned}
\operatorname{Reg}(T,v)
=
\mathbb E\sum_{\ell=1}^L|E_\ell|
\Bigl[&R(S^\star,v)-R(S_\ell,\mu(\ell))\\
&+R(S_\ell,\mu(\ell))-R(S_\ell,v)\Bigr].
\end{aligned}
$$
</div>

第一项控制 sampled optimum 是否足够乐观，第二项控制 sample 与真实参数之间的估计误差。

估计误差按每个商品自己的 epoch 计数求和。使用

<div class="display-equation">
$$
\sum_{s=1}^n\frac1{\sqrt s}
\leq
2\sqrt n
$$
</div>

与 Cauchy-Schwarz，可得

<div class="display-equation">
$$
\operatorname{Reg}_2
\leq
C_1\sqrt{NT}\log(TK)
+C_2N\log^2(TK).
$$
</div>

对第一项，乐观 epoch 不产生正 regret；非乐观 epoch 收费给最近的前一个乐观 epoch。相关采样的 spacing 性质使两个乐观 epoch 之间的非乐观间隔代价为 $O(1/K)$，而每个 epoch 的平均长度为 $O(K)$，两者抵消：

<div class="display-equation">
$$
\underbrace{O(K)}_{\text{epoch 平均长度}}
\times
\underbrace{O(1/K)}_{\text{乐观间隔代价}}
=O(1).
$$
</div>

### 定理 2（MNL-Bandit 的 $k^\star$\-free regret）

<div class="display-equation">
$$
\operatorname{Reg}_{\mathrm{MNL}}(T)
\leq
C_1\sqrt{NT}\log(TK)
+C_2N\log^2(TK).
$$
</div>

右端没有最优 assortment 大小 $k^\star$ 的乘法因子。

### 例 2（三商品 MNL）

令

<div class="display-equation">
$$
r=(1,0.7,0.4),
\qquad
v=(0.8,0.5,0.2).
$$
</div>

展示 $S=\{1,2\}$ 时，

<div class="display-equation">
$$
p_0=\frac1{2.3},
\qquad
p_1=\frac{0.8}{2.3},
\qquad
p_2=\frac{0.5}{2.3},
$$
</div>

且

<div class="display-equation">
$$
R(S,v)
=
\frac{1\cdot0.8+0.7\cdot0.5}{2.3}
=0.5.
$$
</div>

这个例子说明 outside option 为什么必然进入分母，也说明一般 revenue MNL 的单调性必须按最优集合结构单独证明。

### 三条结论不能混为一谈

到这里必须把三种对象严格分开。

原论文 CTS3 使用一个共同 seed，保持每个基础臂的 Gaussian sampling marginal，并在一般 monotone smooth CMAB 中消除指数依赖。

经典 MNL 相关采样算法使用 epoch、Gaussian approximation、$K$ 组共同 seeds 以及逐坐标取最大。它不是原始 CTS3 的直接特例，其 $k^\star$\-free 结果依赖 MNL 的 restricted monotonicity 与 epoch 结构。

后文的 Delay-Max 则进一步改变采样状态：未被选择的高 sample 被保留。因此它不再具有原始 posterior marginal，但能直接破坏 blocker 反例依赖的“乐观性下一轮消失”机制。

因此，严谨的表述不是“后续证明把原论文定理里的 $k^\star$ 直接删掉”，而是：

<div class="display-equation">
$$
\begin{array}{ll}
\text{原始 CTS3：} & \text{一般 CMAB 下不能无条件删掉；}\\
\text{MNL：} & \text{利用特殊结构得到 $k^\star$-free 结果；}\\
\text{Delay-Max：} & \text{修改算法后在更明确的 Gaussian 条件下得到 $k^\star$-free 结果。}
\end{array}
$$
</div>

## 九、算法性正面结果：Delay-Max

这一后续结果分析的是修改后的算法，而不是对原始 CTS3 定理的直接改写。为建立下面的 renewal 账本，使用 Gaussian semi-bandit、各臂观测序列独立、每个基础臂预先至少初始化观察一次，以及逐坐标单调和 $\ell_1$ bounded smoothness 等条件。

blocker 反例的失败机制是：未被选择的臂即使本轮抽到极高 sample，下一轮也会完全丢失。若这个高 sample 被某个 blocker 截走，最优 witness 没有更新，而刚才的乐观性也没有保留。

Delay-Max 直接修改这一点。

### 算法定义

每个基础臂维护经验均值 $\widehat\mu_i(t)$、样本数 $N_i(t)$ 与 retained sample $H_i(t)$。初始化

<div class="display-equation">
$$
H_i(0)=-\infty.
$$
</div>

第 $t$ 轮抽共同种子

<div class="display-equation">
$$
X_t\sim\mathcal N(0,1),
$$
</div>

生成 fresh sample

<div class="display-equation">
$$
\widetilde\theta_i(t)
=
\widehat\mu_i(t-1)
+
\frac{X_t}{\sqrt{N_i(t-1)+1}}.
$$
</div>

送入 oracle 的 index 为

<div class="display-equation">
$$
Q_i(t)
=
\max\{\widetilde\theta_i(t),H_i(t-1)\}.
$$
</div>

若 oracle 允许 additive error $\varepsilon_t$，要求

<div class="display-equation">
$$
r(S(t),Q(t))
\geq
\max_{S\in\mathcal I}r(S,Q(t))-\varepsilon_t.
$$
</div>

选择并观察 $S(t)$ 后，更新 retained sample：

<div class="display-equation">
$$
H_i(t)
=
\begin{cases}
-\infty, & i\in S(t),\\
Q_i(t), & i\notin S(t).
\end{cases}
$$
</div>

被选择的臂清空旧 sample，因为其统计状态已经更新；未被选择的臂保留当前最大 sample。

Delay-Max 不再保持精确 posterior marginal。它应被理解为 Thompson-style randomized optimistic index，而不是原封不动的 posterior sampling。

### 引理 8（乐观性保留）

若基础臂 $i$ 在 $\tau,\tau+1,\ldots,t-1$ 均未被选择，则

<div class="display-equation">
$$
Q_i(t)
\geq
Q_i(\tau).
$$
</div>

特别地，一旦 $Q_i(\tau)\geq\mu_i$，在它下一次被选择前始终有 $Q_i(t)\geq\mu_i$。

**证明.** 未被选择时 $H_i(s)=Q_i(s)$，所以

<div class="display-equation">
$$
Q_i(s+1)
=
\max\{\widetilde\theta_i(s+1),H_i(s)\}
\geq
H_i(s)
=
Q_i(s).
$$
</div>

逐轮迭代即得。$\square$

这条 persistence 性质直接破坏了 blocker 循环：一次 rare high seed 产生的乐观性不会在下一轮消失。

## 十、Delay-Max 的 regret 证明骨架

定义第 $t$ 轮 gap

<div class="display-equation">
$$
\Delta_t
:=
r(S^\star,\mu)-r(S(t),\mu).
$$
</div>

定义最优集合在 index 下的低估量

<div class="display-equation">
$$
d_t
:=
\bigl[r(S^\star,\mu)-r(S^\star,Q(t))\bigr]_+.
$$
</div>

### 引理 9（三项分解）

每一轮确定性地满足

<div class="display-equation">
$$
\Delta_t
\leq
 d_t
+
\varepsilon_t
+
B\sum_{i\in S(t)}(Q_i(t)-\mu_i)_+.
$$
</div>

**证明.** 加上再减去两个 index-reward：

<div class="display-equation">
$$
\begin{aligned}
\Delta_t
=&\ r(S^\star,\mu)-r(S^\star,Q(t))\\
&+r(S^\star,Q(t))-r(S(t),Q(t))\\
&+r(S(t),Q(t))-r(S(t),\mu).
\end{aligned}
$$
</div>

第一项不大于 $d_t$。第二项由 additive oracle 不大于 $\varepsilon_t$。对第三项，令 $Q\vee\mu$ 表示逐坐标最大值。由单调性与 bounded smoothness，

<div class="display-equation">
$$
\begin{aligned}
r(S(t),Q(t))-r(S(t),\mu)
&\leq
r(S(t),Q(t)\vee\mu)-r(S(t),\mu)\\
&\leq
B\sum_{i\in S(t)}(Q_i(t)-\mu_i)_+.
\end{aligned}
$$
</div>

三项相加即得。$\square$

这个分解形成三个独立账本：最优侧低估、oracle 误差、selected side 过度乐观。

### Selected-side 路径预算

在统一高概率事件上，当 $N_i(t-1)=n$ 时，

<div class="display-equation">
$$
(Q_i(t)-\mu_i)_+
\leq
C\sqrt{\frac{\Lambda}{n+1}},
\qquad
\Lambda\asymp\log(mT).
$$
</div>

每当 $i\in S(t)$，其计数增加一次。因此

<div class="display-equation">
$$
\begin{aligned}
\sum_{t=1}^T\sum_{i\in S(t)}(Q_i(t)-\mu_i)_+
&\leq
C\sqrt\Lambda
\sum_{i=1}^m
\sum_{n=1}^{N_i(T)}\frac1{\sqrt n}\\
&\leq
2C\sqrt\Lambda
\sum_{i=1}^m\sqrt{N_i(T)}.
\end{aligned}
$$
</div>

semi-bandit 总观察数满足

<div class="display-equation">
$$
\sum_{i=1}^mN_i(T)
=
\sum_{t=1}^T|S(t)|
\leq kT.
$$
</div>

由 Cauchy-Schwarz，

<div class="display-equation">
$$
\sum_{i=1}^m\sqrt{N_i(T)}
\leq
\sqrt{m\sum_{i=1}^mN_i(T)}
\leq
\sqrt{mkT}.
$$
</div>

故

<div class="display-equation">
$$
\sum_{t=1}^T\sum_{i\in S(t)}(Q_i(t)-\mu_i)_+
\leq
C\sqrt{mkT\Lambda}.
$$
</div>

### 最优侧 deficit 降到逐臂 deficit

### 引理 10（Reward deficit 的坐标上界）

<div class="display-equation">
$$
d_t
\leq
B\sum_{i\in S^\star}(\mu_i-Q_i(t))_+.
$$
</div>

**证明.** 令 $Q\wedge\mu$ 为逐坐标最小值。因为 $Q\geq Q\wedge\mu$，由单调性，

<div class="display-equation">
$$
r(S^\star,Q)
\geq
r(S^\star,Q\wedge\mu).
$$
</div>

再由 bounded smoothness，

<div class="display-equation">
$$
\begin{aligned}
d_t
&\leq
r(S^\star,\mu)-r(S^\star,Q\wedge\mu)\\
&\leq
B\sum_{i\in S^\star}|\mu_i-(Q_i\wedge\mu_i)|\\
&=
B\sum_{i\in S^\star}(\mu_i-Q_i)_+.
\end{aligned}
$$
</div>

$\square$

### 每个基础臂的 renewal block

固定一个最优臂 $i$。它第 $n$ 次被观察后，retained sample 清空，经验均值固定为 $\widehat\mu_{i,n}$。直到下一次被选择前，fresh sample 为

<div class="display-equation">
$$
\widetilde\theta_i
=
\widehat\mu_{i,n}
+
\frac{X}{\sqrt{n+1}}.
$$
</div>

定义标准化低估阈值

<div class="display-equation">
$$
A_{i,n}
:=
\sqrt{n+1}(\mu_i-\widehat\mu_{i,n}).
$$
</div>

fresh sample 达到真实均值当且仅当

<div class="display-equation">
$$
X\geq A_{i,n}.
$$
</div>

条件于 $A_{i,n}=a$，每轮成功概率为 $\overline\Phi(a)$。第一次成功前的等待是几何等待。一旦成功，若该臂未被选择，乐观 sample 被保留，后续 deficit 为零；若本轮被选择，则该 block 当场结束。

由于

<div class="display-equation">
$$
\widehat\mu_{i,n}-\mu_i
\sim
\mathcal N(0,1/n),
$$
</div>

有

<div class="display-equation">
$$
A_{i,n}
\sim
\mathcal N\left(0,\frac{n+1}{n}\right).
$$
</div>

通过 layer-cake 恒等式与 Gaussian Mills ratio，可以控制截断 inverse-tail 等待：

<div class="display-equation">
$$
\mathbb E\left[
\left(
\frac1{\overline\Phi(A_{i,n})}
\right)\wedge T
\right]
\leq
C\operatorname{polylog}(T)
\left[
1+(n+1)(T^{1/(n+1)}-1)
\right].
$$
</div>

括号中的 $-1$ 不能丢掉。因为当 $n\gg\log T$ 时，

<div class="display-equation">
$$
(n+1)(T^{1/(n+1)}-1)
\asymp
\log T,
$$
</div>

而不是 $O(n)$。

按 $n\leq\lceil\log T\rceil$ 与 $n>\lceil\log T\rceil$ 分段求和，可得单臂累计 deficit

<div class="display-equation">
$$
\mathbb E
\sum_{t=1}^T
(\mu_i-Q_i(t))_+
\leq
C\bigl(\sqrt T+\sqrt{N_i(T)}\bigr)
\operatorname{polylog}(T,m).
$$
</div>

对 $i\in S^\star$ 求和：

<div class="display-equation">
$$
\mathbb E
\sum_{t=1}^T
\sum_{i\in S^\star}
(\mu_i-Q_i(t))_+
\leq
C\left(
 k^\star\sqrt T
+
\sum_{i\in S^\star}\sqrt{N_i(T)}
\right)
\operatorname{polylog}(T,m).
$$
</div>

此处暂时出现了 $k^\star$，但它可以被目标尺度吸收。因为 $k^\star\leq k\leq m$，

<div class="display-equation">
$$
(k^\star)^2\leq mk
\quad\Longrightarrow\quad
k^\star\sqrt T
\leq
\sqrt{mkT}.
$$
</div>

同时

<div class="display-equation">
$$
\sum_{i\in S^\star}\sqrt{N_i(T)}
\leq
\sqrt{k^\star\sum_{i\in S^\star}N_i(T)}
\leq
\sqrt{k^\star kT}
\leq
\sqrt{mkT}.
$$
</div>

因此

<div class="display-equation">
$$
\mathbb E\sum_{t=1}^Td_t
\leq
CB\sqrt{mkT}
\operatorname{polylog}(T,m).
$$
</div>

### 定理 3（Delay-Max 的 $k^\star$\-free regret）

在 Gaussian semi-bandit、逐坐标单调、$\ell_1$ bounded smoothness 条件下，Delay-Max 配合 additive oracle 满足

<div class="display-equation">
$$
\mathbb E[\operatorname{Reg}(T)]
\leq
CB\sqrt{mkT}
\operatorname{polylog}(T,m)
+
\sum_{t=1}^T\mathbb E[\varepsilon_t]
+O(m).
$$
</div>

exact oracle 对应 $\varepsilon_t=0$。

若 $\varepsilon_t\equiv\varepsilon>0$，则会留下 $\varepsilon T$ 线性项。若使用固定比例 $\alpha$\-approximation oracle，正确 benchmark 应改为 $\alpha$\-regret；一般不能保证相对 exact optimum 的次线性 regret。

## 十一、四臂线性 CMAB 的完整案例

考虑四个基础臂，每轮恰好选择两个：

<div class="display-equation">
$$
[m]=[4],
\qquad
\mathcal I=\{S\subseteq[4]:|S|=2\}.
$$
</div>

真实均值为

<div class="display-equation">
$$
\mu=(0.2,0.8,0.6,0.1),
$$
</div>

使用线性 reward

<div class="display-equation">
$$
r(S,\mu)=\sum_{i\in S}\mu_i.
$$
</div>

六个组合的真实收益为

<div class="display-equation">
$$
\begin{array}{c|c}
S & r(S,\mu)\\
\hline
\{1,2\} & 1.0\\
\{1,3\} & 0.8\\
\{1,4\} & 0.3\\
\{2,3\} & 1.4\\
\{2,4\} & 0.9\\
\{3,4\} & 0.7
\end{array}
$$
</div>

因此

<div class="display-equation">
$$
S^\star=\{2,3\},
\qquad
k=k^\star=2,
\qquad
\Delta_{\min}=0.4.
$$
</div>

假设某轮开始前

<div class="display-equation">
$$
\widehat\mu=(0.3,0.7,0.4,0.2),
\qquad
N=(3,8,2,3),
$$
</div>

且共同 seed 为 $X_t=1$。CTS3 样本为

<div class="display-equation">
$$
\theta_i
=
\widehat\mu_i+
\frac1{\sqrt{N_i+1}}.
$$
</div>

逐坐标计算得到

<div class="display-equation">
$$
\theta
\approx
(0.8,1.033,0.977,0.7).
$$
</div>

oracle 在 sampled world 中比较六个组合，得到

<div class="display-equation">
$$
r(\{2,3\},\theta)
\approx
2.010,
$$
</div>

为最大值，故本轮选择 $\{2,3\}$。只观察并更新基础臂 $2,3$。

共同 seed 的作用不是让所有坐标相等，而是让它们同时向同一方向移动。基础臂 $3$ 只有两次观测，随机尺度为 $1/\sqrt3$；基础臂 $2$ 有八次观测，随机尺度为 $1/3$。因此同一个正 seed 会把更不确定的臂 $3$ 推得更远。

## 十二、原始 CTS3 与 Delay-Max 的机制对照

| 问题 | 原始 CTS3 | Delay-Max |
| --- | --- | --- |
| 未选臂本轮出现高 sample | 下一轮丢失 | 存入 $H_i$ |
| 下一轮 fresh seed 变低 | 乐观性消失 | $Q_i=\max\{\text{fresh},H_i\}$，乐观性保留 |
| 一个 blocker 被选择 | 只更新 blocker，其他高 sample 全丢失 | 被选 blocker 清空；其他未选臂的高 sample 继续存在 |
| 清空 $K$ 个 blockers | 可能反复等待 $K$ 次 rare tail，尺度 $K/p$ | 一次 tail 后乐观性持续，典型尺度接近 $1/p+K$ |
| 证明账本 | inverse-tail 代价可能被重复支付 | 每个更新 block 只支付一次真实 reset 代价 |
| 是否保持精确 posterior marginal | 是 | 否，成为 randomized optimistic index |

Delay-Max 的改动不是常数级优化。它改变了算法的信息状态：未被使用的乐观证据不再立即消失。

## 十三、整条理论路线的准确结论

### 命题 1（标准 CTS 的问题）

独立 posterior sampling 使最优组合的联合好事件产生概率乘积：

<div class="display-equation">
$$
\mathbb P(\text{all good})
\approx
p^{k^\star}.
$$
</div>

### 命题 2（CTS3 的突破）

单种子 coupling 把联合事件改写为一个一维阈值：

<div class="display-equation">
$$
\bigcap_{i\in S^\star}\{\theta_i\geq b_i\}
=
\{X_t\geq\max_i c_i\}.
$$
</div>

因此指数依赖被降为多项式依赖。

### 命题 3（原始 CTS3 的一般边界）

Reward-level rescue 不保证低估 witness 被更新。一般 CMAB 允许长度为 $\Theta(k^\star)$ 的 blocker release chain，因此原始 CTS3 不能在原一般假设下无条件得到 $\widetilde O(\sqrt{mkT})$。

### 命题 4（MNL 的结构性解法）

MNL 的 epoch、restricted monotonicity、相关采样与 $K$ 组 boosting 共同保证 optimistic epochs 足够密，最终得到不含 $k^\star$ 乘法因子的 regret。

### 命题 5（一般 CMAB 的算法性解法）

Delay-Max 保留未被选择臂的乐观 sample，使一次高 seed 的作用持续到真正更新发生。按基础臂的真实 reset clock 建立 renewal 账本后，$k^\star$ 可以被 $\sqrt{mkT}$ 吸收。

## 十四、进一步的问题

### 1\. 最小结构条件

bounded-release、persistence、no-blocker 与 reward-level threshold 之间的关系仍需要进一步形式化。理想结果应给出近似必要且充分的结构条件，而不是只分别处理一般反例与若干特殊模型。

### 2\. Posterior marginal 与 optimism persistence 能否兼得

原始 CTS3 的优势是每个坐标保持正确 posterior marginal；Delay-Max 的优势是未使用乐观性能够保留。二者目前存在张力。一个更强的算法可能通过带状态 coupling、lazy resampling 或 posterior-consistent memory，同时保留统计解释与释放效率。

### 3\. 近似 oracle

additive oracle error 直接进入累计账本：

<div class="display-equation">
$$
\operatorname{Reg}(T)
\lesssim
\widetilde O(B\sqrt{mkT})
+
\sum_{t=1}^T\varepsilon_t.
$$
</div>

因此近似精度需要随时间提高。固定比例 $\alpha$\-oracle 一般只能保证 $\alpha$\-regret，不能无条件保证相对 exact optimum 的次线性 regret。

### 4\. 一般 outcome 分布

CTS3 的 inverse-CDF coupling 可以扩展到 Beta、Gamma 与其他 posterior。Delay-Max 的 Gaussian inverse-tail 分析则依赖正态尾部。对次高斯、指数族乃至重尾分布，需要建立相应的持久化采样与截断 inverse-tail 工具。

### 5\. 更一般的序贯决策

single-seed coupling 的本质是重新设计联合随机性，而不改变边缘不确定性；Delay-Max 的本质是让一次探索证据在被消费前持续存在。二者可能推广到 combinatorial RL、多智能体探索、结构化 posterior sampling，以及大动作空间中的概率保持耦合。

## 结论

CTS3 的贡献不是简单地“把多个随机数换成一个随机数”。它揭示了 Thompson Sampling 理论中的一个核心事实：边缘采样分布不能单独决定探索效率，联合依赖结构同样重要。

标准 CTS 的指数困难来自多个独立坐标必须同时乐观。CTS3 通过共单调 coupling 将其压缩为一个共同 seed 的 tail event，从而在一般非线性 CMAB 中把 $L^{k^\star}$ 型障碍降为关于 $k^\star$ 的多项式依赖。

但从多项式到完全 $k^\star$\-free 仍有一道结构性障碍。一般 CMAB 中，乐观 seed 可能被 non-full blockers 连续消耗。反例证明，原始 CTS3 在原一般假设下不能无条件达到 $\widetilde O(B\sqrt{mkT})$。这不是某个 union bound 写得不够紧，而是算法会丢失尚未被使用的乐观证据。

正面结果有两条路线。MNL-Bandit 利用特殊 reward 与 epoch 结构，使相关采样的乐观间隔代价抵消组合容量；Delay-Max 则修改算法，使未被选择臂的乐观 sample 持续保留，并按真实更新次数建立逐臂账本，最终得到一般单调光滑 CMAB 中的 $k^\star$\-free regret。

最终结论应准确写成：

<div class="display-equation">
$$
\boxed{
\begin{array}{c}
\text{原始 CTS3 + 一般 CMAB：不能无条件摘除 }k^\star;\\[2mm]
\text{特殊结构或修改算法：可以摘除。}
\end{array}
}
$$
</div>

## 参考资料

1.  _Avoiding $\exp(k^\star)$ Scaling for Thompson Sampling in Combinatorial Semi-Bandits: From Multiple Seeds to a Single Seed_, COLT 2026 under review.
2.  Siwei Wang and Wei Chen. _Thompson Sampling for Combinatorial Semi-Bandits_. ICML, 2018.
3.  Raymond Zhang and Richard Combes. _Thompson Sampling for Combinatorial Bandits: Polynomial Regret and Mismatched Sampling Paradox_. NeurIPS, 2024.
4.  Shipra Agrawal, Vashist Avadhanula, Vineet Goyal, and Assaf Zeevi. _Thompson Sampling for the MNL-Bandit_. COLT, 2017.
5.  Branislav Kveton, Zheng Wen, Azin Ashkan, and Csaba Szepesvári. _Tight Regret Bounds for Stochastic Combinatorial Semi-Bandits_. AISTATS, 2015.
