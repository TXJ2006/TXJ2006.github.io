---
title: "当手臂不再有限：线性 Bandit、置信椭球与乐观优化"
date: 2026-08-30 20:00:00
categories:
  - 机器学习理论
tags:
  - 多臂老虎机
  - 线性 Bandit
  - 在线学习
  - 最优化
  - 数学证明
mathjax: true
toc: true
toc_number: false
comments: true
---

经典多臂老虎机把每个选项看成一只彼此独立的臂。广告 A 的点击数据只更新 A，广告 B 的点击数据只更新 B。这个模型适合解释探索与利用，却忽略了工业系统中最重要的一类信息：不同选项通常共享特征。

一部科幻电影和另一部科幻电影不是毫无关系；两条经过同一城区的配送路线也不是两个完全独立的对象。只要动作可以表示成特征向量，一次观测就可能同时改变我们对许多动作的判断。

线性 Bandit 正是研究这种“信息迁移”的最小模型。它仍然是 Bandit：算法每轮选择动作，只观察被选动作的奖励，并以累计遗憾衡量决策质量。变化只在于，未知对象不再是 $K$ 个互不相干的均值，而是一个所有动作共同依赖的参数向量。

本文从这个模型出发，逐步推导岭回归估计、置信椭球和 OFUL 的乐观选择规则，并证明它的高概率遗憾上界。附录会把正文使用的矩阵范数、行列式引理、自归一化集中不等式与椭圆势能逐项展开。

<!--more-->

## 1. 为什么有限臂模型不够了

设推荐系统有一百万个候选内容。如果把每个内容当作一只独立的臂，系统就需要分别估计一百万个均值。新内容刚上线时没有历史数据，它与已有内容之间的相似性也完全无法利用。

一种更自然的做法是为每个“用户--内容”组合构造特征向量

$$
x=(x&#95;1,\ldots,x&#95;d)^\top\in\mathbb R^d.
$$

每个坐标有具体含义，例如题材匹配度、作者偏好、新鲜度或价格敏感度。这里 $d$ 是特征维数，$\mathbb R^d$ 是所有 $d$ 维实向量组成的空间。假设存在未知偏好参数

$$
\theta^\star\in\mathbb R^d,
$$

使动作 $x$ 的期望奖励为

$$
\mathbb E[Y\mid x]=\langle x,\theta^\star\rangle=x^\top\theta^\star.
$$

符号 $\langle x,\theta^\star\rangle$ 表示内积，$x^\top$ 表示把列向量 $x$ 转置成行向量。展开后就是

$$
x^\top\theta^\star
=\sum&#95;{j=1}^{d}x&#95;j\theta^\star&#95;j.
$$

一次奖励不再只属于某个离散编号。它提供的是关于同一个 $\theta^\star$ 的线性方程，因此会影响所有与当前动作方向相近的候选项。

此后每轮奖励都在估计同一个 $\theta^\star$。因此，观测动作 $x$ 不仅更新这个动作本身，还会改变所有与 $x$ 具有相同特征方向的候选动作的预测。

## 2. 随机线性 Bandit 模型

第 $t$ 轮开始时，算法看到一个动作集合

$$
\mathsf{D}&#95;t\subseteq\mathbb R^d.
$$

集合 $\mathsf{D}&#95;t$ 可以随轮次变化。推荐系统中，它可以是当前用户面对的候选内容特征；动态定价中，它可以是本轮允许选择的价格与商品特征组合。

算法从中选择

$$
x&#95;t\in\mathsf{D}&#95;t,
$$

随后只观察所选动作的奖励

$$
Y&#95;t=x&#95;t^\top\theta^\star+\varepsilon&#95;t.
$$

$\varepsilon&#95;t$ 是噪声。我们不要求它一定服从高斯分布，只假设在给定过去信息后，它的条件均值为零，并且是 $R$-次高斯的：对任意实数 $\eta$，

$$
\mathbb E\left[
\exp(\eta\varepsilon&#95;t)
\mid\mathsf{F}&#95;{t-1}
\right]
\leq
\exp\left(\frac{\eta^2R^2}{2}\right).
$$

$\mathsf{F}&#95;{t-1}$ 表示第 $t$ 轮选择动作之前已经知道的全部信息，$R>0$ 控制噪声尾部的尺度。条件次高斯假设允许 $x&#95;t$ 依赖过去数据，这正是自适应 Bandit 与普通固定设计回归的区别。

本文使用三个有界性条件：

$$
\lVert x\rVert&#95;2\leq L
\quad (x\in\mathsf{D}&#95;t),
\qquad
\lVert \theta^\star\rVert&#95;2\leq S,
\qquad
\lambda>0.
$$

$\lVert x\rVert&#95;2=(\sum&#95;{j=1}^d x&#95;j^2)^{1/2}$ 是欧氏范数，$L$ 限制动作长度，$S$ 限制未知参数长度；$\lambda$ 是稍后岭回归中的正则化系数。

令

$$
x&#95;t^\star
\in\arg\max&#95;{x\in\mathsf{D}&#95;t}x^\top\theta^\star
$$

为第 $t$ 轮在真实参数下的最优动作。到 $T$ 轮的伪遗憾定义为

$$
R&#95;T
:=
\sum&#95;{t=1}^{T}
\left(
(x&#95;t^\star)^\top\theta^\star
-x&#95;t^\top\theta^\star
\right).
$$

噪声本身的偶然波动不计入伪遗憾；它衡量的是算法选择造成的期望奖励损失。

## 3. 数据如何汇成一个几何对象

观察前 $t$ 轮数据后，考虑带正则化的最小二乘问题

$$
\widehat\theta&#95;t
\in
\arg\min&#95;{\theta\in\mathbb R^d}
\left\lbrace
\sum&#95;{s=1}^{t}(Y&#95;s-x&#95;s^\top\theta)^2
+\lambda\lVert \theta\rVert&#95;2^2
\right\rbrace.
$$

定义设计矩阵

$$
V&#95;t
:=
\lambda I&#95;d+
\sum&#95;{s=1}^{t}x&#95;sx&#95;s^\top,
$$

以及加权奖励向量

$$
b&#95;t:=\sum&#95;{s=1}^{t}x&#95;sY&#95;s.
$$

$I&#95;d$ 是 $d\times d$ 单位矩阵。由于 $\lambda>0$，$V&#95;t$ 是正定矩阵，因而可逆。对目标函数求梯度并令其为零，得到

$$
\boxed{
\widehat\theta&#95;t=V&#95;t^{-1}b&#95;t.
}
$$

矩阵 $V&#95;t$ 不只是计算公式中的中间量。它记录算法在哪些方向上收集过多少信息。

如果所有 $x&#95;s$ 几乎平行，那么 $V&#95;t$ 沿这条方向增长很快，垂直方向却仍然缺少信息；如果动作覆盖了许多不同方向，估计就会更均衡。于是“不确定性”不再是一个统一宽度，而是一个随方向变化的几何形状。

把奖励模型代入 $b&#95;t$，可得估计误差的关键分解：

$$
\begin{aligned}
\widehat\theta&#95;t-\theta^\star
&=V&#95;t^{-1}
\left(
\sum&#95;{s=1}^{t}x&#95;s\varepsilon&#95;s
-\lambda\theta^\star
\right).
\end{aligned}
$$

右侧有两部分。随机项 $\sum x&#95;s\varepsilon&#95;s$ 来自噪声，确定项 $-\lambda\theta^\star$ 来自正则化偏差。附录 D 会从正规方程开始逐行证明这条分解。

## 4. 从置信区间到置信椭球

有限臂 UCB 为每只臂维护一个置信区间。线性 Bandit 的未知量是 $d$ 维向量，所以对应对象是置信椭球。

对正定矩阵 $V$，定义矩阵诱导范数

$$
\lVert z\rVert&#95;V:=\sqrt{z^\top Vz}.
$$

在置信水平 $1-\delta$ 下，令

$$
\beta&#95;t(\delta)
:=
R\sqrt{
\log\frac{\det(V&#95;t)}{\det(\lambda I&#95;d)}
+2\log\frac1\delta
}
+\sqrt\lambda S,
$$

其中 $\det(V)$ 表示矩阵 $V$ 的行列式，$\delta\in(0,1)$ 是允许失败的概率。定义

$$
\mathsf{C}&#95;t(\delta)
:=
\left\lbrace
\theta\in\mathbb R^d:
\lVert \theta-\widehat\theta&#95;t\rVert&#95;{V&#95;t}
\leq\beta&#95;t(\delta)
\right\rbrace.
$$

自归一化集中不等式说明：至少以 $1-\delta$ 的概率，

$$
\boxed{
\theta^\star\in\mathsf{C}&#95;t(\delta)
\quad \forall t\geq0.
}
$$

“同时成立”很重要。算法每轮都根据过去数据改变动作方向，不能只对一个预先固定的 $t$ 证明普通集中不等式。这里需要的是一个对整条自适应路径有效的事件。

椭球在信息多的方向上窄，在信息少的方向上宽。对任意候选动作 $x$，其方向不确定性为

$$
\lVert x\rVert&#95;{V&#95;t^{-1}}
=\sqrt{x^\top V&#95;t^{-1}x}.
$$

这就是线性 Bandit 中“这只臂被探索了多少次”的推广。有限臂中我们看 $1/\sqrt{N&#95;i(t)}$；线性模型中，动作之间共享信息，计数被矩阵逆所取代。

![二维线性 Bandit 中的置信椭球与乐观动作](/images/notes/assets/bandits/linear-bandit-confidence-ellipse.svg)

图中的椭球是参数可能所在的区域。某个动作的乐观值等于它在整个椭球上能取得的最大内积；算法选择的不是离椭球中心最近的动作，而是仍可能带来最高奖励的方向。

## 5. 乐观原则如何变成一个优化问题

在第 $t$ 轮，OFUL 使用上一轮结束后的置信集合，选择

$$
x&#95;t
\in
\arg\max&#95;{x\in\mathsf{D}&#95;t}
\max&#95;{\theta\in\mathsf{C}&#95;{t-1}(\delta)}
x^\top\theta.
$$

OFUL 是 Optimism in the Face of Uncertainty for Linear bandits 的缩写。它把“对不确定性保持乐观”写成一个双层优化：对每个动作，先问置信集合中是否存在一个参数使它非常好，再选择这个最好情形最高的动作。

内层优化有闭式解。矩阵形式的 Cauchy--Schwarz 不等式给出

$$
\max&#95;{\theta\in\mathsf{C}&#95;{t-1}(\delta)}x^\top\theta
=
x^\top\widehat\theta&#95;{t-1}
+\beta&#95;{t-1}(\delta)
\lVert x\rVert&#95;{V&#95;{t-1}^{-1}}.
$$

因此选择规则可以写成

$$
\boxed{
x&#95;t\in
\arg\max&#95;{x\in\mathsf{D}&#95;t}
\left\lbrace
x^\top\widehat\theta&#95;{t-1}
+\beta&#95;{t-1}(\delta)
\lVert x\rVert&#95;{V&#95;{t-1}^{-1}}
\right\rbrace.
}
$$

第一项是当前预测，负责利用；第二项是方向相关的置信宽度，负责探索。这个公式看起来像 UCB，但它不再为每只臂单独加一个奖励，而是通过同一个 $V&#95;{t-1}^{-1}$ 比较所有动作的未知方向。

若 $\mathsf{D}&#95;t$ 是有限集合，直接枚举候选动作即可，这一形式通常称为 LinUCB。若动作集合连续，就需要求解带椭球奖励的优化问题。统计问题和计算问题在这里第一次明确分开：置信集合可以完全正确，但如果外层优化无法有效完成，算法仍然不能实际运行。

## 6. 一步遗憾为什么由置信宽度控制

假设当前处在好事件上，即 $\theta^\star\in\mathsf{C}&#95;{t-1}(\delta)$。最优动作的真实期望不超过它的乐观值：

$$
(x&#95;t^\star)^\top\theta^\star
\leq
\max&#95;{\theta\in\mathsf{C}&#95;{t-1}(\delta)}
(x&#95;t^\star)^\top\theta.
$$

OFUL 又选择了乐观值最大的动作，所以

$$
\max&#95;{\theta\in\mathsf{C}&#95;{t-1}(\delta)}
(x&#95;t^\star)^\top\theta
\leq
x&#95;t^\top\widehat\theta&#95;{t-1}
+\beta&#95;{t-1}(\delta)
\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}.
$$

另一方面，置信集合保证

$$
x&#95;t^\top
(\widehat\theta&#95;{t-1}-\theta^\star)
\leq
\beta&#95;{t-1}(\delta)
\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}.
$$

把三式相连，第 $t$ 轮瞬时遗憾

$$
r&#95;t
:=
(x&#95;t^\star)^\top\theta^\star
-x&#95;t^\top\theta^\star
$$

满足

$$
\boxed{
r&#95;t
\leq
2\beta&#95;{t-1}(\delta)
\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}.
}
$$

这条式子把决策损失变成了几何不确定性。算法可能暂时选择一个真实次优动作，但只有当这个动作所在方向仍然足够未知时，它才能在乐观规则下胜出。

## 7. 椭圆势能：探索为什么不会无限重复

还需要控制所有方向不确定性的总和。定义

$$
q&#95;t
:=
\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}^2.
$$

矩阵行列式引理给出

$$
\det(V&#95;t)
=
\det(V&#95;{t-1})(1+q&#95;t).
$$

因此

$$
\log\frac{\det(V&#95;T)}{\det(V&#95;0)}
=
\sum&#95;{t=1}^{T}\log(1+q&#95;t).
$$

如果取 $\lambda\geq L^2$，那么 $V&#95;{t-1}\succeq\lambda I&#95;d$，从而 $0\leq q&#95;t\leq1$。在区间 $[0,1]$ 上有 $q\leq2\log(1+q)$，于是

$$
\sum&#95;{t=1}^{T}q&#95;t
\leq
2\log\frac{\det(V&#95;T)}{\det(\lambda I&#95;d)}.
$$

这就是椭圆势能引理的核心。每选择一次 $x&#95;t$，设计矩阵沿这个方向增加 $x&#95;tx&#95;t^\top$；同一方向被反复观察后，$V^{-1}$ 会把它的置信宽度压低。因此一个方向不可能永久以“高度未知”为理由赢得选择。

进一步使用普通 Cauchy--Schwarz 不等式，

$$
\begin{aligned}
\sum&#95;{t=1}^{T}
\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}
&\leq
\sqrt{
T\sum&#95;{t=1}^{T}
\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}^2
}\\\\
&\leq
\sqrt{
2T\log\frac{\det(V&#95;T)}{\det(\lambda I&#95;d)}
}.
\end{aligned}
$$

## 8. OFUL 的遗憾上界

由于 $V&#95;t\succeq V&#95;{t-1}$，$\det(V&#95;t)$ 随 $t$ 不减，所以 $\beta&#95;t(\delta)$ 也不减。把瞬时遗憾界与椭圆势能相加，得到

$$
\boxed{
R&#95;T
\leq
2\beta&#95;T(\delta)
\sqrt{
2T\log\frac{\det(V&#95;T)}{\det(\lambda I&#95;d)}
}
}
$$

至少以 $1-\delta$ 的概率成立。

为了看出维数和时间的关系，记 $V&#95;T$ 的特征值并用算术--几何平均不等式，可以证明

$$
\log\frac{\det(V&#95;T)}{\det(\lambda I&#95;d)}
\leq
d\log\left(1+\frac{TL^2}{\lambda d}\right).
$$

于是

$$
\beta&#95;T(\delta)
\leq
R\sqrt{
d\log\left(1+\frac{TL^2}{\lambda d}\right)
+2\log\frac1\delta
}
+\sqrt\lambda S,
$$

并得到显式上界

$$
\begin{aligned}
R&#95;T
\leq 2
&\left[
R\sqrt{
d\log\left(1+\frac{TL^2}{\lambda d}\right)
+2\log\frac1\delta
}
+\sqrt\lambda S
\right]\\\\
&\times
\sqrt{
2Td\log\left(1+\frac{TL^2}{\lambda d}\right)
}.
\end{aligned}
$$

忽略对数项与固定参数后，主导量级通常写成

$$
\widetilde O(d\sqrt T).
$$

$\widetilde O$ 表示省略对数因子。证明中的 $\sqrt T$ 来自 Cauchy--Schwarz 求和；自归一化置信半径含有一个 $\sqrt d$，椭圆势能界又含有一个 $\sqrt d$，两者相乘形成主导项 $d\sqrt T$。这三个因子都能在前面的不等式中找到对应来源。

## 9. 行列式项从哪里来，它何时等于信息增益

前面的遗憾证明并不需要贝叶斯先验。现在额外考虑一个固定设计的高斯模型，只为解释行列式项的统计含义。把 $X&#95;t$ 看作预先给定、与参数和噪声独立的矩阵，并设

$$
\theta^\star\sim\mathsf N(0,\lambda^{-1}I&#95;d),
\qquad
\varepsilon&#95;s\overset{\mathrm{i.i.d.}}{\sim}\mathsf N(0,1),
$$

并在给定动作 $x&#95;1,\ldots,x&#95;t$ 的条件下考察奖励 $Y&#95;1,\ldots,Y&#95;t$。记 $X&#95;t\in\mathbb R^{t\times d}$ 为第 $s$ 行等于 $x&#95;s^\top$ 的设计矩阵。此时

$$
V&#95;t=\lambda I&#95;d+X&#95;t^\top X&#95;t.
$$

高斯熵公式和 Sylvester 行列式恒等式给出

$$
\begin{aligned}
I(\theta^\star;Y&#95;{1:t}\mid X&#95;t)
&=\frac12\log\det\left(
I&#95;t+\lambda^{-1}X&#95;tX&#95;t^\top
\right)\\\\
&=\frac12
\log\frac{\det(V&#95;t)}{\det(\lambda I&#95;d)}.
\end{aligned}
$$

$I(\theta^\star;Y&#95;{1:t}\mid X&#95;t)$ 表示给定设计矩阵后，奖励序列包含的关于 $\theta^\star$ 的条件互信息。只有在上述高斯先验、单位方差高斯噪声和固定设计条件下，正文中的行列式项才与它严格相等；若噪声方差为 $\sigma^2$，设计矩阵需要相应乘以 $\sigma^{-2}$。在真正的自适应 Bandit 中，$x&#95;t$ 依赖过去奖励，仍可用行列式作势函数，但不能跳过策略产生数据的条件结构，直接套用这个固定设计互信息等式。完整计算见附录 N。

这也说明椭圆势能为什么使用行列式，而不是简单地把每个动作的长度相加。$\det(V&#95;t)$ 同时汇总了所有特征方向上的收缩：在已经反复观测的方向再取一个样本，行列式增长很少；沿尚未覆盖的方向取样，增长更多。

主动学习与线性 Bandit 都会利用这种方向信息，但目标函数不同。主动学习通常关心收集数据以后估计误差减少多少，查询本身没有即时奖励；Bandit 在选择动作的当轮就承担机会成本。如果只最大化行列式增长，算法可能长期选择信息很多但收益很低的动作。OFUL 只在某个动作的置信上界足以与当前最优预测竞争时探索它，这正是瞬时遗憾证明能够成立的原因。

## 10. Contextual bandit 如何化为线性模型

在 contextual bandit 中，第 $t$ 轮先到达上下文 $c&#95;t$，例如用户画像、设备状态或当前库存。对每个可选动作 $a$，构造联合特征

$$
x&#95;{t,a}=\phi(c&#95;t,a)\in\mathbb R^d,
$$

其中 $\phi$ 是特征映射。若期望奖励满足

$$
\mathbb E[Y&#95;t\mid c&#95;t,a]
=x&#95;{t,a}^\top\theta^\star,
$$

若第 $t$ 轮可选动作的集合为 $\mathsf A&#95;t$，令

$$
\mathsf{D}&#95;t
=\lbrace \phi(c&#95;t,a):a\in\mathsf A&#95;t\rbrace,
$$

这时问题在记号上就是前面的线性 Bandit，OFUL 的证明也可以原样使用，但前提是特征映射 $\phi$ 已经固定，并且确实存在同一个 $\theta^\star$ 使线性期望关系对所有轮次成立。

如果 $\phi$ 由神经网络或随机森林根据历史数据持续更新，事情就不同了。新的表示会改变旧样本的坐标，之前定义的 $V&#95;t$ 不再是同一个固定参数模型的设计矩阵；如果仍直接使用原来的置信半径，$\theta^\star\in\mathsf C&#95;t(\delta)$ 未必成立。同样，若动作空间带有组合约束，统计上的乐观值已经算出以后，还要证明外层最大化能够精确或近似完成。这些问题分别属于模型设定与计算误差，不能由 OFUL 的现有遗憾界自动解决。

## 11. 遗憾界依赖的四个条件

第一，**线性可实现性**。理论假设存在固定的 $\theta^\star$ 使所有条件期望都精确线性。现实中特征遗漏、交互效应和模型漂移会产生 misspecification，即模型设定偏差。此时置信椭球可能越来越窄，却围住了错误的参数。

第二，**反馈时机**。本文默认选择后立即看到 $Y&#95;t$。广告转化、临床结局和长期留存可能延迟数小时甚至数月，设计矩阵增长与真实可用信息之间会出现时间差。

第三，**动作外生性**。动作集合 $\mathsf{D}&#95;t$ 可以依赖过去，但必须在本轮噪声产生前确定。如果候选集本身泄露了本轮不可见结果，条件次高斯结构就不再成立。

第四，**优化可计算性**。OFUL 的定义使用精确的 $\arg\max$。连续、组合或带约束的动作空间可能使外层优化成为主要瓶颈。近似 oracle 带来的优化误差需要单独进入遗憾分解，不能被统计置信宽度自动吸收。

所以一个可靠的系统不只要画出置信椭球，还要检查椭球为什么覆盖真实参数、反馈何时进入估计、动作集合怎样生成，以及乐观目标是否真的被求解。

## 12. 结语：为什么同一个矩阵同时控制估计和探索

整篇证明只围绕设计矩阵 $V&#95;t$ 展开。它先出现在岭回归的正规方程中，随后决定每个动作的置信宽度；选择动作以后，$V&#95;t$ 的更新又记录了本轮观测带来的变化。

先看估计。岭回归正规方程给出

$$
V&#95;t(\widehat\theta&#95;t-\theta^\star)
=S&#95;t-\lambda\theta^\star.
$$

因此，沿动作 $x$ 的预测误差由 $\lVert x\rVert&#95;{V&#95;t^{-1}}$ 控制。OFUL 把这同一个量放进动作的置信上界，所以一个真实次优动作若被选中，它必须在当前置信集合中仍有机会成为最优。由此得到

$$
r&#95;t
\leq
2\beta&#95;{t-1}(\delta)
\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}.
$$

再看更新。选择 $x&#95;t$ 后，

$$
V&#95;t=V&#95;{t-1}+x&#95;tx&#95;t^\top,
$$

而行列式的增量满足

$$
\log\det(V&#95;t)-\log\det(V&#95;{t-1})
=
\log\left(
1+\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}^2
\right).
$$

这说明造成当轮遗憾的置信宽度，也会在观测奖励后增加设计矩阵的信息量。一个方向若不断以“不确定”为理由被选择，它的宽度就会随数据积累而下降。椭圆势能引理只是把这个逐轮事实求和，再用 Cauchy--Schwarz 转成 $R&#95;T$ 的上界。

有限臂 Bandit 中，这个机制退化为计数 $N&#95;i(t)$：选择第 $i$ 只臂只会缩小它自己的区间。在线性模型中，可以用 Sherman--Morrison 公式精确计算任意候选方向 $z$ 的宽度变化：

$$
\begin{aligned}
\lVert z\rVert&#95;{V&#95;t^{-1}}^2
&=
\lVert z\rVert&#95;{V&#95;{t-1}^{-1}}^2\\\\
&\quad-
\frac{
\left(z^\top V&#95;{t-1}^{-1}x&#95;t\right)^2
}{
1+\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}^2
}.
\end{aligned}
$$

若 $z^\top V&#95;{t-1}^{-1}x&#95;t$ 的绝对值较大，观测 $x&#95;t$ 会明显缩小 $z$ 的置信宽度；若它等于零，这次观测对 $z$ 没有帮助。所谓动作之间共享信息，具体就是这个减项。

## 参考文献

1. H. Robbins, “Some Aspects of the Sequential Design of Experiments,” *Bulletin of the American Mathematical Society*, 1952. [Project Euclid](https://projecteuclid.org/journals/bulletin-of-the-american-mathematical-society/volume-58/issue-5/Some-aspects-of-the-sequential-design-of-experiments/10.1090/S0002-9904-1952-09620-8.full).
2. V. Dani, T. P. Hayes, and S. M. Kakade, “Stochastic Linear Optimization under Bandit Feedback,” *Proceedings of COLT*, 2008. [PDF](https://homes.cs.washington.edu/~sham/papers/ml/bandit_linear.pdf).
3. P. Rusmevichientong and J. N. Tsitsiklis, “Linearly Parameterized Bandits,” *Mathematics of Operations Research*, 2010. [INFORMS](https://doi.org/10.1287/moor.1100.0446).
4. Y. Abbasi-Yadkori, D. Pál, and C. Szepesvári, “Improved Algorithms for Linear Stochastic Bandits,” *Advances in Neural Information Processing Systems*, 2011. [NeurIPS](https://proceedings.neurips.cc/paper/2011/hash/e1d5be1c7f2f456670de3d53c7b54f4a-Abstract.html).
5. W. Chu, L. Li, L. Reyzin, and R. Schapire, “Contextual Bandits with Linear Payoff Functions,” *Proceedings of AISTATS*, 2011. [PMLR](https://proceedings.mlr.press/v15/chu11a.html).
6. T. Lattimore and C. Szepesvári, *Bandit Algorithms*, Cambridge University Press, 2020. [Online edition](https://tor-lattimore.com/downloads/book/book.pdf).

---

# 附录

如下为正文附录补充。

## A. 符号与时间顺序

$d$ 是特征维数，$T$ 是总轮数。$\mathbb R^d$ 表示 $d$ 维实向量空间。向量默认写成列向量，$x^\top$ 是其转置。

第 $t$ 轮的顺序是：

1. 过去信息形成 $\sigma$-代数 $\mathsf{F}&#95;{t-1}$；
2. 动作集合 $\mathsf{D}&#95;t$ 与所选动作 $x&#95;t$ 对 $\mathsf{F}&#95;{t-1}$ 可测；
3. 环境产生噪声 $\varepsilon&#95;t$；
4. 算法观察 $Y&#95;t=x&#95;t^\top\theta^\star+\varepsilon&#95;t$；
5. 新信息形成 $\mathsf{F}&#95;t$。

“对 $\mathsf{F}&#95;{t-1}$ 可测”表示在噪声 $\varepsilon&#95;t$ 出现以前，动作已经由过去信息确定；算法可以随机化，只需把算法在本轮选择前使用的随机种子也纳入 $\mathsf{F}&#95;{t-1}$。

主要符号如下：

- $\theta^\star$：固定但未知的真实参数；
- $x&#95;t$：第 $t$ 轮选择的动作特征；
- $Y&#95;t$：第 $t$ 轮观察到的标量奖励；
- $\varepsilon&#95;t$：条件均值为零的噪声；
- $R$：条件次高斯尺度；
- $L$：动作欧氏范数上界；
- $S$：真实参数欧氏范数上界；
- $\lambda$：岭正则化系数；
- $V&#95;t$：正则化设计矩阵；
- $b&#95;t$：特征加权奖励向量；
- $\widehat\theta&#95;t$：岭回归估计；
- $\beta&#95;t(\delta)$：置信椭球半径；
- $\mathsf{C}&#95;t(\delta)$：参数置信集合；
- $R&#95;T$：累计伪遗憾。

$A\succeq B$ 表示 $A-B$ 是半正定矩阵；$A\succ0$ 表示 $A$ 是正定矩阵。$\det(A)$ 和 $\operatorname{tr}(A)$ 分别表示行列式与迹。$\arg\max$ 表示使目标达到最大值的点组成的集合。

## B. 正定矩阵、矩阵范数与对偶不等式

### B.1 半正定与正定

对称矩阵 $A\in\mathbb R^{d\times d}$ 称为半正定，如果

$$
z^\top Az\geq0
\qquad \forall z\in\mathbb R^d
$$

成立；若对所有非零 $z$ 都严格大于零，则称为正定。

每个外积 $xx^\top$ 都是半正定的，因为

$$
z^\top xx^\top z
=(x^\top z)^2\geq0.
$$

因此

$$
V&#95;t
=\lambda I&#95;d+
\sum&#95;{s=1}^{t}x&#95;sx&#95;s^\top
\succeq\lambda I&#95;d\succ0.
$$

正定性保证 $V&#95;t$ 可逆，并且 $V&#95;t^{-1}$ 也正定。

### B.2 矩阵诱导范数

对正定矩阵 $V$，定义

$$
\lVert z\rVert&#95;V:=\sqrt{z^\top Vz},
\qquad
\lVert z\rVert&#95;{V^{-1}}:=\sqrt{z^\top V^{-1}z}.
$$

令 $V^{1/2}$ 为 $V$ 的正定平方根，则

$$
\lVert z\rVert&#95;V=\lVert V^{1/2}z\rVert&#95;2.
$$

对任意 $x,z\in\mathbb R^d$，插入 $V^{-1/2}V^{1/2}=I&#95;d$：

$$
\begin{aligned}
|x^\top z|
&=|(V^{-1/2}x)^\top(V^{1/2}z)|\\\\
&\leq
\lVert V^{-1/2}x\rVert&#95;2
\lVert V^{1/2}z\rVert&#95;2\\\\
&=
\lVert x\rVert&#95;{V^{-1}}\lVert z\rVert&#95;V.
\end{aligned}
$$

第二行使用欧氏空间的 Cauchy--Schwarz 不等式。这就是正文使用的矩阵对偶不等式。

### B.3 椭球上线性函数的最大值

设

$$
\mathsf{C}
=\lbrace\theta:\lVert \theta-\widehat\theta\rVert&#95;V\leq\beta\rbrace.
$$

写 $z=\theta-\widehat\theta$，则

$$
x^\top\theta
=x^\top\widehat\theta+x^\top z
\leq
x^\top\widehat\theta
+\beta\lVert x\rVert&#95;{V^{-1}}.
$$

当 $x\neq0$ 时，取

$$
z^\star
:=
\frac{\beta V^{-1}x}{\lVert x\rVert&#95;{V^{-1}}}.
$$

先验证它位于椭球边界：

$$
\begin{aligned}
\lVert z^\star\rVert&#95;V^2
&=
\frac{\beta^2}{\lVert x\rVert&#95;{V^{-1}}^2}
x^\top V^{-1}VV^{-1}x\\\\
&=
\frac{\beta^2}{\lVert x\rVert&#95;{V^{-1}}^2}
x^\top V^{-1}x\\\\
&=\beta^2.
\end{aligned}
$$

再代入目标：

$$
x^\top z^\star
=\frac{\beta x^\top V^{-1}x}
{\lVert x\rVert&#95;{V^{-1}}}
=\beta\lVert x\rVert&#95;{V^{-1}}.
$$

因此上界可以取到，故

$$
\max&#95;{\theta\in\mathsf{C}}x^\top\theta
=x^\top\widehat\theta
+\beta\lVert x\rVert&#95;{V^{-1}}.
$$

$x=0$ 时两边都等于 $0$，结论仍成立。

## C. 条件次高斯噪声

### C.1 定义与条件均值

若对所有 $\eta\in\mathbb R$，

$$
\mathbb E[exp(\eta\varepsilon&#95;t)\mid\mathsf{F}&#95;{t-1}]
\leq\exp(\eta^2R^2/2),
$$

则称 $\varepsilon&#95;t$ 在给定 $\mathsf{F}&#95;{t-1}$ 后是 $R$-次高斯的。

在适当可积条件下，对上式在 $\eta=0$ 处比较一阶导数，可得

$$
\mathbb E[\varepsilon&#95;t\mid\mathsf{F}&#95;{t-1}]=0.
$$

本文直接把条件均值为零包含在假设中，避免在边界正则性上增加无关讨论。

### C.2 高斯噪声为什么满足定义

若给定过去后

$$
\varepsilon&#95;t\sim\mathsf{N}(0,\sigma^2),
$$

则其矩母函数为

$$
\mathbb E[e^{\eta\varepsilon&#95;t}\mid\mathsf{F}&#95;{t-1}]
=e^{\eta^2\sigma^2/2}.
$$

下面通过配方验证。高斯密度是

$$
\frac1{\sqrt{2\pi}\sigma}
\exp\left(-\frac{u^2}{2\sigma^2}\right).
$$

因此

$$
\begin{aligned}
\mathbb E[e^{\eta\varepsilon&#95;t}]
&=
\frac1{\sqrt{2\pi}\sigma}
\int&#95;{-\infty}^{\infty}
\exp\left(
\eta u-\frac{u^2}{2\sigma^2}
\right)du\\\\
&=
\frac1{\sqrt{2\pi}\sigma}
\int&#95;{-\infty}^{\infty}
\exp\left(
-\frac{(u-\eta\sigma^2)^2}{2\sigma^2}
+\frac{\eta^2\sigma^2}{2}
\right)du\\\\
&=
e^{\eta^2\sigma^2/2}.
\end{aligned}
$$

最后一步中，平移后的高斯密度积分为 $1$。所以高斯噪声是 $R=\sigma$ 的次高斯噪声。

## D. 岭回归闭式解与误差分解

定义目标函数

$$
J&#95;t(\theta)
:=
\sum&#95;{s=1}^{t}(Y&#95;s-x&#95;s^\top\theta)^2
+\lambda\theta^\top\theta.
$$

对 $\theta$ 求梯度：

$$
\nabla J&#95;t(\theta)
=-2\sum&#95;{s=1}^{t}x&#95;s(Y&#95;s-x&#95;s^\top\theta)
+2\lambda\theta.
$$

令梯度为零：

$$
-\sum&#95;{s=1}^{t}x&#95;sY&#95;s
+\sum&#95;{s=1}^{t}x&#95;sx&#95;s^\top\theta
+\lambda\theta=0.
$$

整理为

$$
\left(
\lambda I&#95;d+
\sum&#95;{s=1}^{t}x&#95;sx&#95;s^\top
\right)\theta
=
\sum&#95;{s=1}^{t}x&#95;sY&#95;s,
$$

即

$$
V&#95;t\theta=b&#95;t.
$$

由于 $V&#95;t\succ0$，目标函数严格凸，唯一极小点为

$$
\widehat\theta&#95;t=V&#95;t^{-1}b&#95;t.
$$

再代入 $Y&#95;s=x&#95;s^\top\theta^\star+\varepsilon&#95;s$：

$$
\begin{aligned}
b&#95;t
&=\sum&#95;{s=1}^{t}x&#95;s
(x&#95;s^\top\theta^\star+\varepsilon&#95;s)\\\\
&=\left(\sum&#95;{s=1}^{t}x&#95;sx&#95;s^\top\right)\theta^\star
+\sum&#95;{s=1}^{t}x&#95;s\varepsilon&#95;s\\\\
&=(V&#95;t-\lambda I&#95;d)\theta^\star
+\sum&#95;{s=1}^{t}x&#95;s\varepsilon&#95;s.
\end{aligned}
$$

于是

$$
\begin{aligned}
\widehat\theta&#95;t
&=V&#95;t^{-1}b&#95;t\\\\
&=V&#95;t^{-1}
\left[
(V&#95;t-\lambda I&#95;d)\theta^\star
+\sum&#95;{s=1}^{t}x&#95;s\varepsilon&#95;s
\right]\\\\
&=\theta^\star
-\lambda V&#95;t^{-1}\theta^\star
+V&#95;t^{-1}\sum&#95;{s=1}^{t}x&#95;s\varepsilon&#95;s.
\end{aligned}
$$

因此

$$
\boxed{
\widehat\theta&#95;t-\theta^\star
=V&#95;t^{-1}
\left(
\sum&#95;{s=1}^{t}x&#95;s\varepsilon&#95;s
-\lambda\theta^\star
\right).
}
$$

## E. 两个矩阵恒等式

### E.1 Sherman--Morrison 公式

若 $A$ 可逆且 $1+u^\top A^{-1}u\neq0$，则

$$
(A+uu^\top)^{-1}
=A^{-1}
-\frac{A^{-1}uu^\top A^{-1}}
{1+u^\top A^{-1}u}.
$$

记右侧为 $B$，直接相乘：

$$
\begin{aligned}
(A+uu^\top)B
&=I&#95;d+uu^\top A^{-1}\\\\
&\quad-
\frac{uu^\top A^{-1}
+uu^\top A^{-1}uu^\top A^{-1}}
{1+u^\top A^{-1}u}.
\end{aligned}
$$

令 $q=u^\top A^{-1}u$，分子中后两项为

$$
uu^\top A^{-1}+q,uu^\top A^{-1}
=(1+q)uu^\top A^{-1}.
$$

它与前面的 $uu^\top A^{-1}$ 抵消，因此 $(A+uu^\top)B=I&#95;d$，公式得证。

### E.2 矩阵行列式引理

若 $A$ 可逆，则

$$
\det(A+uv^\top)
=\det(A)(1+v^\top A^{-1}u).
$$

先提出 $A$：

$$
\det(A+uv^\top)
=\det(A)\det(I&#95;d+A^{-1}uv^\top).
$$

还需证明

$$
\det(I&#95;d+ab^\top)=1+b^\top a.
$$

若 $b^\top a\neq0$，取一组以 $a$ 为第一个基向量的基。在这组基下，线性映射 $I&#95;d+ab^\top$ 对所有满足 $b^\top z=0$ 的 $z$ 保持不变，而在 $a$ 方向上的特征值为 $1+b^\top a$。其余 $d-1$ 个特征值为 $1$，故行列式为 $1+b^\top a$。若 $b^\top a=0$，同一结论由连续性得到。

取 $a=A^{-1}u$、$b=v$，便得到矩阵行列式引理。令 $u=v=x&#95;t$、$A=V&#95;{t-1}$，得到

$$
\det(V&#95;t)
=\det(V&#95;{t-1})
\left(1+x&#95;t^\top V&#95;{t-1}^{-1}x&#95;t\right).
$$

## F. 固定方向的指数超鞅

定义噪声加权和

$$
S&#95;t:=\sum&#95;{s=1}^{t}x&#95;s\varepsilon&#95;s,
\qquad S&#95;0:=0.
$$

对任意固定向量 $u\in\mathbb R^d$，定义

$$
M&#95;t(u)
:=
\exp\left(
u^\top S&#95;t
-\frac{R^2}{2}
u^\top(V&#95;t-\lambda I&#95;d)u
\right).
$$

由于

$$
S&#95;t=S&#95;{t-1}+x&#95;t\varepsilon&#95;t
$$

以及

$$
V&#95;t-\lambda I&#95;d
=(V&#95;{t-1}-\lambda I&#95;d)+x&#95;tx&#95;t^\top,
$$

比值为

$$
\frac{M&#95;t(u)}{M&#95;{t-1}(u)}
=
\exp\left(
(u^\top x&#95;t)\varepsilon&#95;t
-\frac{R^2}{2}(u^\top x&#95;t)^2
\right).
$$

$x&#95;t$ 对 $\mathsf{F}&#95;{t-1}$ 可测。令 $\eta=u^\top x&#95;t$，条件次高斯假设给出

$$
\begin{aligned}
\mathbb E\left[
\frac{M&#95;t(u)}{M&#95;{t-1}(u)}
\middle|\mathsf{F}&#95;{t-1}
\right]
&=e^{-R^2\eta^2/2}
\mathbb E[e^{\eta\varepsilon&#95;t}
\mid\mathsf{F}&#95;{t-1}]\\\\
&\leq1.
\end{aligned}
$$

所以

$$
\mathbb E[M&#95;t(u)\mid\mathsf{F}&#95;{t-1}]
\leq M&#95;{t-1}(u).
$$

这说明 $\lbrace M&#95;t(u)\rbrace$ 是非负超鞅，并且 $M&#95;0(u)=1$。

## G. 混合超鞅与自归一化集中不等式

固定方向 $u$ 的结论还不够，因为估计误差最终会选择一个依赖数据的方向。解决方法是把所有方向用高斯密度混合起来。

### G.1 多维高斯积分

若 $A\succ0$，则

$$
\begin{aligned}
\int&#95;{\mathbb R^d}
\exp\left(-\frac12u^\top Au+h^\top u\right)du
&=(2\pi)^{d/2}\det(A)^{-1/2}\\\\
&\quad\times
\exp\left(\frac12h^\top A^{-1}h\right).
\end{aligned}
$$

证明来自矩阵配方：

$$
\begin{aligned}
-\frac12u^\top Au+h^\top u
&=-\frac12(u-A^{-1}h)^\top A(u-A^{-1}h)\\\\
&\quad
+\frac12h^\top A^{-1}h.
\end{aligned}
$$

平移 $v=u-A^{-1}h$ 后，再令 $w=A^{1/2}v$。变量替换的 Jacobian 为 $\det(A)^{-1/2}$，而

$$
\int&#95;{\mathbb R^d}e^{-\lVert w\rVert&#95;2^2/2}dw=(2\pi)^{d/2}.
$$

代回即得结论。

### G.2 混合计算

令 $U$ 服从均值为 $0$、协方差为 $(R^2\lambda)^{-1}I&#95;d$ 的高斯分布，其密度为

$$
f(u)
=\left(\frac{R^2\lambda}{2\pi}\right)^{d/2}
\exp\left(-\frac{R^2\lambda}{2}\lVert u\rVert&#95;2^2\right).
$$

定义混合过程

$$
\overline M&#95;t
:=\int&#95;{\mathbb R^d}M&#95;t(u)f(u),du.
$$

非负超鞅的非负加权积分仍是非负超鞅，且 $\overline M&#95;0=1$。把 $M&#95;t(u)$ 与 $f(u)$ 的指数合并：

$$
\begin{aligned}
&u^\top S&#95;t
-\frac{R^2}{2}u^\top(V&#95;t-\lambda I&#95;d)u
-\frac{R^2\lambda}{2}u^\top u\\\\
&\qquad=
u^\top S&#95;t
-\frac{R^2}{2}u^\top V&#95;tu.
\end{aligned}
$$

应用 G.1，其中 $A=R^2V&#95;t$、$h=S&#95;t$：

$$
\begin{aligned}
\overline M&#95;t
&=
\left(\frac{R^2\lambda}{2\pi}\right)^{d/2}\\\\
&\quad\times
\int
\exp\left(
u^\top S&#95;t
-\frac{R^2}{2}u^\top V&#95;tu
\right)du\\\\
&=
\left(\frac{R^2\lambda}{2\pi}\right)^{d/2}
(2\pi)^{d/2}\\\\
&\quad\times
\det(R^2V&#95;t)^{-1/2}
\exp\left(
\frac1{2R^2}S&#95;t^\top V&#95;t^{-1}S&#95;t
\right)\\\\
&=
\left(
\frac{\det(\lambda I&#95;d)}{\det(V&#95;t)}
\right)^{1/2}\\\\
&\quad\times
\exp\left(
\frac{\lVert S&#95;t\rVert&#95;{V&#95;t^{-1}}^2}{2R^2}
\right).
\end{aligned}
$$

### G.3 Ville 不等式

若 $Z&#95;t$ 是 $Z&#95;0=1$ 的非负超鞅，则对任意 $a>0$，

$$
\mathbb P\left(\sup&#95;{t\geq0}Z&#95;t\geq a\right)
\leq\frac1a.
$$

证明如下。先固定有限时域 $T$，定义首次越过 $a$ 的停止时刻

$$
\tau:=\min\lbrace t\leq T:Z&#95;t\geq a\rbrace,
$$

若从未越过则令 $\tau=T$。有界停止时刻的超鞅可选停止性质给出

$$
\mathbb E[Z&#95;\tau]\leq\mathbb E[Z&#95;0]=1.
$$

在事件 $\lbrace\max&#95;{t\leq T}Z&#95;t\geq a\rbrace$ 上，$Z&#95;\tau\geq a$，所以

$$
1\geq\mathbb E[Z&#95;\tau]
\geq
a\,
\mathbb P\left(\max&#95;{t\leq T}Z&#95;t\geq a\right).
$$

令 $T\to\infty$，这些事件单调增加，概率连续性给出 Ville 不等式。

对 $Z&#95;t=\overline M&#95;t$ 和 $a=1/\delta$ 使用该不等式。至少以 $1-\delta$ 的概率，对所有 $t\geq0$，

$$
\overline M&#95;t<\frac1\delta.
$$

代入 G.2 的闭式表达并取对数：

$$
\frac{\lVert S&#95;t\rVert&#95;{V&#95;t^{-1}}^2}{2R^2}
-\frac12\log
\frac{\det(V&#95;t)}{\det(\lambda I&#95;d)}
<\log\frac1\delta.
$$

移项并乘以 $2R^2$：

$$
\boxed{
\lVert S&#95;t\rVert&#95;{V&#95;t^{-1}}
\leq
R\sqrt{
\log\frac{\det(V&#95;t)}{\det(\lambda I&#95;d)}
+2\log\frac1\delta
}
}
$$

对所有 $t\geq0$ 同时成立。这就是本文所需的自归一化集中不等式。

## H. 置信椭球覆盖真实参数

由附录 D，

$$
\widehat\theta&#95;t-\theta^\star
=V&#95;t^{-1}(S&#95;t-\lambda\theta^\star).
$$

取 $V&#95;t$-范数并使用三角不等式：

$$
\begin{aligned}
\lVert \widehat\theta&#95;t-\theta^\star\rVert&#95;{V&#95;t}
&\leq
\lVert V&#95;t^{-1}S&#95;t\rVert&#95;{V&#95;t}
+\lambda\lVert V&#95;t^{-1}\theta^\star\rVert&#95;{V&#95;t}\\\\
&=
\lVert S&#95;t\rVert&#95;{V&#95;t^{-1}}
+\lambda\lVert \theta^\star\rVert&#95;{V&#95;t^{-1}}.
\end{aligned}
$$

因为 $V&#95;t\succeq\lambda I&#95;d$，所以

$$
V&#95;t^{-1}\preceq\lambda^{-1}I&#95;d.
$$

从而

$$
\lambda\lVert \theta^\star\rVert&#95;{V&#95;t^{-1}}
\leq
\lambda\sqrt{\lambda^{-1}}
\lVert \theta^\star\rVert&#95;2
\leq\sqrt\lambda S.
$$

再使用附录 G 的自归一化界：至少以 $1-\delta$ 的概率，对所有 $t$，

$$
\begin{aligned}
\lVert \widehat\theta&#95;t-\theta^\star\rVert&#95;{V&#95;t}
&\leq
R\sqrt{
\log\frac{\det(V&#95;t)}{\det(\lambda I&#95;d)}
+2\log\frac1\delta
}
+\sqrt\lambda S\\\\
&=\beta&#95;t(\delta).
\end{aligned}
$$

所以 $\theta^\star\in\mathsf{C}&#95;t(\delta)$ 对所有 $t$ 同时成立。

## I. 乐观性与瞬时遗憾的逐步证明

定义乐观值

$$
U&#95;t(x)
:=
\max&#95;{\theta\in\mathsf{C}&#95;{t-1}(\delta)}x^\top\theta.
$$

由附录 B，

$$
U&#95;t(x)
=x^\top\widehat\theta&#95;{t-1}
+\beta&#95;{t-1}(\delta)
\lVert x\rVert&#95;{V&#95;{t-1}^{-1}}.
$$

OFUL 选择 $x&#95;t\in\arg\max&#95;{x\in\mathsf{D}&#95;t}U&#95;t(x)$。在好事件上，$\theta^\star\in\mathsf{C}&#95;{t-1}(\delta)$，所以

$$
(x&#95;t^\star)^\top\theta^\star
\leq U&#95;t(x&#95;t^\star)
\leq U&#95;t(x&#95;t).
$$

于是

$$
\begin{aligned}
r&#95;t
&=(x&#95;t^\star)^\top\theta^\star
-x&#95;t^\top\theta^\star\\\\
&\leq U&#95;t(x&#95;t)-x&#95;t^\top\theta^\star\\\\
&=x&#95;t^\top(\widehat\theta&#95;{t-1}-\theta^\star)
+\beta&#95;{t-1}(\delta)
\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}.
\end{aligned}
$$

矩阵对偶不等式与置信集合再给出

$$
\begin{aligned}
x&#95;t^\top(\widehat\theta&#95;{t-1}-\theta^\star)
&\leq
\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}
\lVert \widehat\theta&#95;{t-1}-\theta^\star\rVert&#95;{V&#95;{t-1}}\\\\
&\leq
\beta&#95;{t-1}(\delta)
\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}.
\end{aligned}
$$

合并可得

$$
r&#95;t
\leq
2\beta&#95;{t-1}(\delta)
\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}.
$$

## J. 椭圆势能引理

令

$$
q&#95;t
:=x&#95;t^\top V&#95;{t-1}^{-1}x&#95;t
=\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}^2.
$$

由矩阵行列式引理，

$$
\frac{\det(V&#95;t)}{\det(V&#95;{t-1})}=1+q&#95;t.
$$

对 $t=1,\ldots,T$ 相乘：

$$
\frac{\det(V&#95;T)}{\det(V&#95;0)}
=\prod&#95;{t=1}^{T}(1+q&#95;t).
$$

取自然对数：

$$
\log\frac{\det(V&#95;T)}{\det(V&#95;0)}
=\sum&#95;{t=1}^{T}\log(1+q&#95;t).
$$

由于 $V&#95;{t-1}\succeq\lambda I&#95;d$，

$$
q&#95;t
\leq\frac{\lVert x&#95;t\rVert&#95;2^2}{\lambda}
\leq\frac{L^2}{\lambda}.
$$

当 $\lambda\geq L^2$ 时，$0\leq q&#95;t\leq1$。定义

$$
g(q):=2\log(1+q)-q.
$$

有

$$
g(0)=0,
\qquad
g'(q)=\frac{2}{1+q}-1
=\frac{1-q}{1+q}\geq0
$$

对 $q\in[0,1]$ 成立，因此 $q\leq2\log(1+q)$。逐项求和得到

$$
\boxed{
\sum&#95;{t=1}^{T}
\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}^2
\leq
2\log\frac{\det(V&#95;T)}{\det(\lambda I&#95;d)}.
}
$$

不要求 $\lambda\geq L^2$ 时，更一般的版本是

$$
\sum&#95;{t=1}^{T}\min\lbrace1,q&#95;t\rbrace
\leq
2\log\frac{\det(V&#95;T)}{\det(\lambda I&#95;d)}.
$$

本文采用 $\lambda\geq L^2$，是为了让主证明不再额外截断瞬时遗憾。

## K. 行列式的维数上界

写

$$
A&#95;T:=\sum&#95;{t=1}^{T}x&#95;tx&#95;t^\top,
\qquad
V&#95;T=\lambda I&#95;d+A&#95;T.
$$

$A&#95;T$ 是半正定矩阵。设其特征值为 $a&#95;1,\ldots,a&#95;d\geq0$，则 $V&#95;T$ 的特征值为 $\lambda+a&#95;1,\ldots,\lambda+a&#95;d$。所以

$$
\frac{\det(V&#95;T)}{\det(\lambda I&#95;d)}
=\prod&#95;{j=1}^{d}
\left(1+\frac{a&#95;j}{\lambda}\right).
$$

算术--几何平均不等式给出

$$
\prod&#95;{j=1}^{d}
\left(1+\frac{a&#95;j}{\lambda}\right)
\leq
\left(
1+\frac{\sum&#95;{j=1}^{d}a&#95;j}{\lambda d}
\right)^d.
$$

特征值之和等于迹，而

$$
\begin{aligned}
\sum&#95;{j=1}^{d}a&#95;j
&=\operatorname{tr}(A&#95;T)\\\\
&=\sum&#95;{t=1}^{T}\operatorname{tr}(x&#95;tx&#95;t^\top)\\\\
&=\sum&#95;{t=1}^{T}\lVert x&#95;t\rVert&#95;2^2\\\\
&\leq TL^2.
\end{aligned}
$$

因此

$$
\boxed{
\log\frac{\det(V&#95;T)}{\det(\lambda I&#95;d)}
\leq
d\log\left(1+\frac{TL^2}{\lambda d}\right).
}
$$

## L. OFUL 高概率遗憾界的完整证明

假设：

1. $x&#95;t$ 在观察 $\varepsilon&#95;t$ 前确定，且 $\lVert x&#95;t\rVert&#95;2\leq L$；
2. $\varepsilon&#95;t$ 在给定过去后是 $R$-次高斯的；
3. $\lVert \theta^\star\rVert&#95;2\leq S$；
4. $\lambda\geq L^2$；
5. 每轮的乐观最大值能够取到。

由附录 G 和 H，至少以 $1-\delta$ 的概率，$\theta^\star$ 同时属于所有 $\mathsf{C}&#95;t(\delta)$。以下固定在这个好事件上。

由附录 I，

$$
r&#95;t
\leq
2\beta&#95;{t-1}(\delta)
\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}.
$$

由于 $V&#95;t\succeq V&#95;{t-1}$，行列式不减，所以

$$
\beta&#95;{t-1}(\delta)\leq\beta&#95;T(\delta).
$$

于是

$$
R&#95;T
\leq
2\beta&#95;T(\delta)
\sum&#95;{t=1}^{T}
\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}.
$$

对最后的和使用 Cauchy--Schwarz：

$$
\begin{aligned}
\left(
\sum&#95;{t=1}^{T}
\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}
\right)^2
&\leq
\left(\sum&#95;{t=1}^{T}1^2\right)
\left(
\sum&#95;{t=1}^{T}
\lVert x&#95;t\rVert&#95;{V&#95;{t-1}^{-1}}^2
\right)\\\\
&\leq
2T\log
\frac{\det(V&#95;T)}{\det(\lambda I&#95;d)},
\end{aligned}
$$

第二步使用附录 J。因此

$$
R&#95;T
\leq
2\beta&#95;T(\delta)
\sqrt{
2T\log
\frac{\det(V&#95;T)}{\det(\lambda I&#95;d)}
}.
$$

最后使用附录 K：

$$
\begin{aligned}
R&#95;T
\leq 2
&\left[
R\sqrt{
d\log\left(1+\frac{TL^2}{\lambda d}\right)
+2\log\frac1\delta
}
+\sqrt\lambda S
\right]\\\\
&\times
\sqrt{
2Td\log\left(1+\frac{TL^2}{\lambda d}\right)
}.
\end{aligned}
$$

这就完成了正文所述高概率遗憾界的证明。

## M. 有限臂与 contextual bandit 如何嵌入线性模型

### M.1 有限臂是一个特殊线性模型

考虑 $K$ 臂 Bandit，令 $d=K$，并令第 $i$ 只臂的特征为第 $i$ 个标准基向量

$$
e&#95;i=(0,\ldots,0,1,0,\ldots,0)^\top.
$$

取

$$
\theta^\star=(\mu&#95;1,\ldots,\mu&#95;K)^\top.
$$

则

$$
e&#95;i^\top\theta^\star=\mu&#95;i.
$$

设计矩阵变成对角矩阵：

$$
V&#95;t
=\lambda I&#95;K+
\operatorname{diag}
(N&#95;1(t),\ldots,N&#95;K(t)),
$$

其中 $\operatorname{diag}$ 表示以给定数字为对角元的对角矩阵。于是

$$
\lVert e&#95;i\rVert&#95;{V&#95;t^{-1}}
=\frac1{\sqrt{\lambda+N&#95;i(t)}}.
$$

这说明有限臂 UCB 的“次数倒数平方根”正是线性置信宽度在标准基特征下的特殊情形。

### M.2 Contextual bandit 的嵌入

给定上下文 $c&#95;t$ 和动作 $a$，特征映射

$$
\phi:\mathsf{C}\times\mathsf{A}\to\mathbb R^d
$$

把上下文--动作对送到一个 $d$ 维向量，其中 $\mathsf{C}$ 是上下文空间，$\mathsf{A}$ 是动作空间。令

$$
x&#95;{t,a}:=\phi(c&#95;t,a),
\qquad
\mathsf{D}&#95;t:=\lbrace x&#95;{t,a}:a\in\mathsf{A}&#95;t\rbrace.
$$

只要条件期望是 $x&#95;{t,a}^\top\theta^\star$，所有线性 Bandit 的估计与遗憾分析即可直接应用。这里允许可用动作集合 $\mathsf{A}&#95;t$ 随上下文变化。

在有限臂嵌入中，$V&#95;t$ 是对角矩阵，各臂之间没有信息传递；在一般特征映射下，$V&#95;t$ 通常不是对角矩阵，一次观测会改变多个动作方向的置信宽度。两种模型的差异具体体现在设计矩阵的结构中。

## N. 高斯模型下的信息增益公式

本节证明正文第 9 节使用的等式。给定一个非随机设计矩阵 $X&#95;t\in\mathbb R^{t\times d}$，设

$$
\theta^\star\sim\mathsf N(0,\lambda^{-1}I&#95;d),
\qquad
\varepsilon&#95;{1:t}\sim\mathsf N(0,I&#95;t),
$$

且 $\theta^\star$ 与 $\varepsilon&#95;{1:t}$ 独立。奖励向量为

$$
Y&#95;{1:t}=X&#95;t\theta^\star+\varepsilon&#95;{1:t}.
$$

### N.1 奖励向量的条件分布

给定 $X&#95;t$ 和 $\theta^\star$ 后，随机性只来自噪声，因此

$$
Y&#95;{1:t}\mid X&#95;t,\theta^\star
\sim
\mathsf N(X&#95;t\theta^\star,I&#95;t).
$$

只给定 $X&#95;t$ 时，$X&#95;t\theta^\star$ 的协方差为

$$
\begin{aligned}
\operatorname{Cov}(X&#95;t\theta^\star\mid X&#95;t)
&=
X&#95;t
\operatorname{Cov}(\theta^\star)
X&#95;t^\top\\\\
&=
\lambda^{-1}X&#95;tX&#95;t^\top.
\end{aligned}
$$

参数项和噪声独立，所以协方差相加：

$$
Y&#95;{1:t}\mid X&#95;t
\sim
\mathsf N\left(
0,\,
I&#95;t+\lambda^{-1}X&#95;tX&#95;t^\top
\right).
$$

### N.2 用高斯微分熵计算互信息

若 $Z\sim\mathsf N(0,\Sigma)$ 是 $n$ 维高斯向量，且 $\Sigma\succ0$，其微分熵为

$$
h(Z)
=
\frac12\log\left((2\pi e)^n\det(\Sigma)\right).
$$

$h(Z)$ 表示连续随机向量的微分熵。下面先验证这个公式。$Z$ 的密度为

$$
p(z)
=
\frac{
\exp\left(-z^\top\Sigma^{-1}z/2\right)
}{
(2\pi)^{n/2}\det(\Sigma)^{1/2}
}.
$$

按照微分熵的定义，

$$
\begin{aligned}
h(Z)
&=-\mathbb E[\log p(Z)]\\\\
&=\frac n2\log(2\pi)
+\frac12\log\det(\Sigma)
+\frac12\mathbb E[Z^\top\Sigma^{-1}Z].
\end{aligned}
$$

最后一项可以用迹计算：

$$
\begin{aligned}
\mathbb E[Z^\top\Sigma^{-1}Z]
&=
\operatorname{tr}\left(
\Sigma^{-1}\mathbb E[ZZ^\top]
\right)\\\\
&=
\operatorname{tr}(\Sigma^{-1}\Sigma)
=n.
\end{aligned}
$$

代回后，

$$
\begin{aligned}
h(Z)
&=
\frac n2\log(2\pi)
+\frac12\log\det(\Sigma)
+\frac n2\\\\
&=
\frac12\log\left((2\pi e)^n\det(\Sigma)\right).
\end{aligned}
$$

现在使用条件互信息的定义：

$$
\begin{aligned}
I(\theta^\star;Y&#95;{1:t}\mid X&#95;t)
&=
h(Y&#95;{1:t}\mid X&#95;t)
-h(Y&#95;{1:t}\mid X&#95;t,\theta^\star)\\\\
&=
\frac12\log\det\left(
I&#95;t+\lambda^{-1}X&#95;tX&#95;t^\top
\right).
\end{aligned}
$$

第二步中，两个高斯熵都包含 $t\log(2\pi e)/2$，相减后抵消；条件于 $\theta^\star$ 后的噪声协方差为 $I&#95;t$，其行列式为 $1$。

### N.3 Sylvester 行列式恒等式

对 $A\in\mathbb R^{m\times n}$ 和 $B\in\mathbb R^{n\times m}$，Sylvester 恒等式为

$$
\det(I&#95;m+AB)=\det(I&#95;n+BA).
$$

考虑分块矩阵

$$
K=
\begin{pmatrix}
I&#95;m & A\\\\
-B & I&#95;n
\end{pmatrix}.
$$

这里使用一次 Schur 补公式。若方阵 $P$ 可逆，则

$$
\det
\begin{pmatrix}
P & Q\\\\
R & S
\end{pmatrix}
=
\det(P)\det(S-RP^{-1}Q).
$$

证明只需在左侧乘行列式为 $1$ 的分块下三角矩阵：

$$
\begin{pmatrix}
I & 0\\\\
-RP^{-1} & I
\end{pmatrix}
\begin{pmatrix}
P & Q\\\\
R & S
\end{pmatrix}
=
\begin{pmatrix}
P & Q\\\\
0 & S-RP^{-1}Q
\end{pmatrix}.
$$

右侧是分块上三角矩阵，其行列式为两个对角块行列式的乘积，因此 Schur 补公式成立。

先对左上角 $I&#95;m$ 取 Schur 补：

$$
\det(K)
=
\det(I&#95;m)\det(I&#95;n+BA)
=
\det(I&#95;n+BA).
$$

再对右下角 $I&#95;n$ 取 Schur 补：

$$
\det(K)
=
\det(I&#95;n)\det(I&#95;m+AB)
=
\det(I&#95;m+AB).
$$

两种计算得到同一个 $\det(K)$，故恒等式成立。

令

$$
A=\lambda^{-1/2}X&#95;t,
\qquad
B=\lambda^{-1/2}X&#95;t^\top,
$$

得到

$$
\begin{aligned}
\det\left(
I&#95;t+\lambda^{-1}X&#95;tX&#95;t^\top
\right)
&=
\det\left(
I&#95;d+\lambda^{-1}X&#95;t^\top X&#95;t
\right)\\\\
&=
\frac{
\det(\lambda I&#95;d+X&#95;t^\top X&#95;t)
}{
\det(\lambda I&#95;d)
}\\\\
&=
\frac{\det(V&#95;t)}{\det(\lambda I&#95;d)}.
\end{aligned}
$$

代入 N.2 即得

$$
\boxed{
I(\theta^\star;Y&#95;{1:t}\mid X&#95;t)
=
\frac12
\log\frac{\det(V&#95;t)}{\det(\lambda I&#95;d)}.
}
$$
