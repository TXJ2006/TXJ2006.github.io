---
title: "聪明有极限：多臂老虎机中的后悔、信息与不可避免的探索"
date: 2026-08-30 17:00:00
categories:
  - 机器学习理论
tags:
  - 多臂老虎机
  - 在线学习
  - 后悔分析
  - 信息论
  - 数学证明
mathjax: true
toc: true
comments: true
---

在一个广告平台里，每次请求到来时，系统都要决定展示哪一个广告。一个广告的点击率可能是 $10\%$，另一个是 $9\%$，但系统并不知道这些数字，只能通过展示和观察点击逐渐学习。

如果平台把所有流量都给当前估计最好的广告，它可能永远错过真正更好的方案；如果平台不断尝试新广告，又会把一部分本来可以获得的点击交给次优选项。探索不是免费的，但它又不能被简单地取消。

前两篇文章分别讨论了探索与利用的基本矛盾，以及 Thompson Sampling 如何用后验不确定性安排探索。本文继续追问一个更基础的问题：**有没有一种算法可以把探索成本压到任意接近于零？**

答案是否定的。原因不是某个具体算法不够聪明，而是两个可能的世界有时在统计上几乎无法区分。为了排除“另一个广告其实更好”的可能性，系统必须收集足够多的证据；在证据还不够时，继续尝试本身就是理性的。

这就是下界的作用。上界告诉我们某个算法最多损失多少，下界告诉我们所有算法至少要损失多少。把两者放在一起，才知道一个方法是在浪费流量，还是已经接近问题本身允许的极限。

<!--more-->

## 1. 先把问题写清楚

设有 $K$ 个臂，编号为 $1,\ldots,K$。在第 $t$ 轮，算法根据过去的信息选择臂

$$
A&#95;t\in\{1,\ldots,K\},
$$

然后观察奖励 $X&#95;t$。本文先研究 Bernoulli 多臂老虎机：若第 $t$ 轮选择了臂 $i$，则

$$
X&#95;t\mid(A&#95;t=i)\sim\operatorname{Bernoulli}(\mu&#95;i),
\qquad \mu&#95;i\in[0,1].
$$

这里 $\operatorname{Bernoulli}(\mu&#95;i)$ 是只取 $0$ 和 $1$ 的分布，取 $1$ 的概率为 $\mu&#95;i$；在广告场景中，$1$ 可以表示点击，$0$ 表示未点击。奖励在给定动作后独立产生，但动作可以依赖全部历史。

记

$$
\mu^\star:=\max&#95;{1\leq i\leq K}\mu&#95;i
$$

为最优均值，并选择一个最优臂 $i^\star\in\arg\max&#95;{1\leq i\leq K}\mu&#95;i$。臂 $i$ 的差距（gap）定义为

$$
\Delta&#95;i:=\mu^\star-\mu&#95;i\geq0.
$$

到第 $T$ 轮为止，臂 $i$ 被选择的次数是

$$
N&#95;i(T):=\sum&#95;{t=1}^{T}\mathbf{1}\{A&#95;t=i\}.
$$

每一轮只选择一个臂，所以路径上恒有

$$
\sum&#95;{i=1}^{K}N&#95;i(T)=T.
$$

本文讨论伪遗憾（pseudo-regret）：它比较每轮的期望奖励，不把 Bernoulli 奖励自身的偶然波动算成算法错误。定义为

$$
R&#95;T:=\sum&#95;{t=1}^{T}(\mu^\star-\mu&#95;{A&#95;t}).
$$

如果算法经常选择差距大的臂，$R&#95;T$ 就大；如果它只在几个相近的臂之间试错，单次损失就小。后文会把这两个因素精确拆开。

## 2. 遗憾到底在工业系统里表示什么

回到广告平台。假设广告 A 的真实点击率为 $0.10$，广告 B 的真实点击率为 $0.09$。某一次请求如果本来应该展示 A，却展示了 B，那么这次决策的期望点击损失是

$$
0.10-0.09=0.01.
$$

如果一天有一百万次请求，平均有十万次请求因为选择 B 而不是 A，那么这部分伪遗憾就是

$$
10^5\times0.01=1000
$$

次期望点击。它不是说每天一定少一千个点击，而是说在重复同样的用户随机性时，平均差异的尺度是一千。

在这个语境下，下界回答的是一个很实际的预算问题：为了确认 B 的点击率确实低于 A，而不是因为短期运气不好，平台至少要给 B 多少曝光？如果 A 和 B 的真实点击率差距很小，答案会变大；如果两个点击率相差明显，答案会变小。

所以“下界”不是一句抽象的否定。它告诉我们：

- 新广告至少需要多少探索流量，才有机会被可靠地评价；
- 一个实验预算是否足以发现预期的微小提升；
- 系统当前的损失中，有多少来自不可避免的信息收集，有多少来自策略实现低效；
- 在上线安全约束下，性能目标是否本来就不可能实现。

把这些问题写成数学，关键是分别描述“错选一次损失多少”和“观察一次提供多少证据”。前者由 gap 给出，后者由 KL 散度给出。

## 3. 第一条分解：遗憾等于差距乘以次数

对任意一条动作路径，先插入指标恒等式

$$
\sum&#95;{i=1}^{K}\mathbf{1}\{A&#95;t=i\}=1.
$$

于是

$$
\begin{aligned}
R&#95;T
&=\sum&#95;{t=1}^{T}(\mu^\star-\mu&#95;{A&#95;t})\\
&=\sum&#95;{t=1}^{T}\sum&#95;{i=1}^{K}
\mathbf{1}\{A&#95;t=i\}(\mu^\star-\mu&#95;i)\\
&=\sum&#95;{i=1}^{K}\Delta&#95;iN&#95;i(T).
\end{aligned}
$$

取期望便得到

$$
\boxed{\mathbb{E}[R&#95;T]
=\sum&#95;{i=1}^{K}\Delta&#95;i\,\mathbb{E}[N&#95;i(T)]}.
$$

这条恒等式很朴素，却决定了后面所有分析的形状。算法的任务是控制次优臂的选择次数；统计学的限制则决定这些次数不可能同时都很小。

当臂 $i$ 的差距为 $0$ 时，它不会增加伪遗憾；当 $\Delta&#95;i>0$ 时，每多选择一次，就增加 $\Delta&#95;i$ 的期望损失。因此一个下界可以有两种等价的表达：直接给出遗憾至少多大，或者先给出每个次优臂至少要被选择多少次。

## 4. 上界是承诺，下界是边界

为了看清两者的分工，先回顾一个已知时域的 UCB 规则。令

$$
\widehat\mu&#95;i(t-1)
:=\frac{1}{N&#95;i(t-1)}
\sum&#95;{s<t:A&#95;s=i}X&#95;s
$$

为臂 $i$ 的经验均值。初始化阶段让每个臂至少被选择一次，之后在第 $t$ 轮选择

$$
A&#95;t\in\arg\max&#95;{1\leq i\leq K}
\lbrace
\widehat\mu&#95;i(t-1)+\sqrt{\frac{2\log T}{N&#95;i(t-1)}}
\rbrace.
$$

经验均值是当前的利用部分，平方根项是对未知程度的补偿。样本越少，补偿越大，算法就越愿意再看一眼这个臂。

对固定的次优臂 $i$，Hoeffding 不等式可以给出

$$
\mathbb{E}[N&#95;i(T)]
\leq 1+\frac{8\log T}{\Delta&#95;i^2}+\frac{4}{T^3},
$$

从而

$$
\mathbb{E}[R&#95;T]
\leq\sum&#95;{i:\Delta&#95;i>0}
\left(\frac{8\log T}{\Delta&#95;i}
+\Delta&#95;i+\frac{4\Delta&#95;i}{T^3}\right).
$$

这是一条**上界**：它保证采用这条规则时，损失不会超过右侧的量级。它没有说明别的算法一定做不到更好。

下界则要换一个量词。它要证明的是：对任何满足基本一致性要求的算法，都存在一个不可绕开的成本。只有当上界与下界在数量级、甚至常数上接近时，我们才有理由说算法已经接近最优。

## 5. KL 散度：把“两个世界很像”量化

考虑两个 Bernoulli 分布：一个点击率为 $p$，另一个点击率为 $q$。它们的 KL 散度定义为

$$
\operatorname{kl}(p,q)
:=p\log\frac{p}{q}
+(1-p)\log\frac{1-p}{1-q},
$$

其中 $p,q\in(0,1)$，$\log$ 是自然对数。约定边界上的 $0\log0=0$，并用连续延拓处理 $p=0$ 或 $p=1$ 的情形。

一次观测 $X\sim\operatorname{Bernoulli}(p)$ 相对于 $\operatorname{Bernoulli}(q)$ 的对数似然比为

$$
\ell(X):=
X\log\frac{p}{q}
+(1-X)\log\frac{1-p}{1-q}.
$$

在真实分布 $p$ 下取期望：

$$
\begin{aligned}
\mathbb{E}&#95;{p}[\ell(X)]
&=\mathbb{E}&#95;{p}[X]\log\frac{p}{q}
+\mathbb{E}&#95;{p}[1-X]\log\frac{1-p}{1-q}\\
&=p\log\frac{p}{q}
+(1-p)\log\frac{1-p}{1-q}\\
&=\operatorname{kl}(p,q).
\end{aligned}
$$

因此 KL 散度可以读成“一次观测平均提供多少区分证据”。它越小，两个世界越像，要达到同样的可靠程度就需要更多观测。

例如，若 A 的点击率为 $0.10$，我们要排除的近邻世界是 B 的点击率从 $0.09$ 变成 $0.1001$，那么

$$
\operatorname{kl}(0.09,0.1001)\approx5.84\times10^{-4}.
$$

一次 B 的曝光只提供很少的区分信息。后文会看到，所需曝光数的主导量级正是

$$
\frac{\log T}{\operatorname{kl}(0.09,0.1001)}.
$$

## 6. 自适应实验仍然可以分解信息

算法不是预先决定每个臂观察多少次，而是根据数据自适应地选择动作。自适应性并不会破坏 KL 分解。

设真实环境为 $\nu=(\nu&#95;1,\ldots,\nu&#95;K)$，另一个环境为 $\lambda=(\lambda&#95;1,\ldots,\lambda&#95;K)$。在历史

$$
H&#95;T=(A&#95;1,X&#95;1,\ldots,A&#95;T,X&#95;T)
$$

上，假设算法的动作概率为 $\pi&#95;t(a\mid h&#95;{t-1})$，臂 $a$ 在两个环境中的奖励密度分别为 $p&#95;a$ 与 $q&#95;a$。历史密度逐项写成

$$
P&#95;\nu^T(h&#95;T)
=\prod&#95;{t=1}^{T}
\pi&#95;t(a&#95;t\mid h&#95;{t-1})p&#95;{a&#95;t}(x&#95;t),
$$

$$
P&#95;\lambda^T(h&#95;T)
=\prod&#95;{t=1}^{T}
\pi&#95;t(a&#95;t\mid h&#95;{t-1})q&#95;{a&#95;t}(x&#95;t).
$$

两式相除时，同一个算法的策略项逐项抵消，于是

$$
\frac{P&#95;\nu^T(h&#95;T)}{P&#95;\lambda^T(h&#95;T)}
=\prod&#95;{t=1}^{T}
\frac{p&#95;{a&#95;t}(x&#95;t)}{q&#95;{a&#95;t}(x&#95;t)}.
$$

取对数并在真实环境下取期望，得到

$$
\boxed{
\operatorname{KL}(P&#95;\nu^T\Vert P&#95;\lambda^T)
=\sum&#95;{i=1}^{K}
\mathbb{E}&#95;\nu[N&#95;i(T)]
\operatorname{KL}(\nu&#95;i\Vert\lambda&#95;i)}.
$$

如果两个环境只在臂 $i$ 上不同，那么总历史信息就简化为

$$
\operatorname{KL}(P&#95;\nu^T\Vert P&#95;\lambda^T)
=\mathbb{E}&#95;\nu[N&#95;i(T)]
\operatorname{kl}(\mu&#95;i,q).
$$

这句话是下界证明的桥梁：算法无论怎样安排动作，都只能通过真正拉动臂 $i$ 来区分“均值是 $\mu&#95;i$”和“均值是 $q$”这两个世界。

## 7. Lai--Robbins 下界：为什么是 $\log T$

现在假设臂 $i$ 在真实环境中是次优的：

$$
\mu&#95;i<\mu^\star.
$$

构造替代环境 $\nu^{(q)}$：只把臂 $i$ 的均值改成 $q>\mu^\star$，其他臂完全不变。于是臂 $i$ 在替代环境中变成唯一最优臂。

这两个环境的区别很小，但算法必须在长期内表现出不同的行为：

- 在真实环境中，臂 $i$ 次优，所以不应无休止地拉它；
- 在替代环境中，臂 $i$ 最优，所以必须最终大部分时间拉它。

如果观察臂 $i$ 的次数太少，两个世界的历史分布就不能被可靠地区分；如果观察次数足够多，历史似然比才会积累到能够支持这种行为差异的程度。

我们把这个直觉写成定理。

### 定理：实例相关的渐近下界

假设算法在每个固定环境上都是 uniformly efficient：对任意常数 $c>0$，都有

$$
\mathbb{E}&#95;\nu[R&#95;T]=o(T^c),
\qquad T\to\infty.
$$

则对每个满足 $\mu&#95;i<\mu^\star$ 的次优臂 $i$，

$$
\boxed{
\liminf&#95;{T\to\infty}
\frac{\mathbb{E}&#95;\nu[N&#95;i(T)]}{\log T}
\geq
\frac{1}{\operatorname{kl}(\mu&#95;i,\mu^\star)}}.
$$

因此

$$
\boxed{
\liminf&#95;{T\to\infty}
\frac{\mathbb{E}&#95;\nu[R&#95;T]}{\log T}
\geq
\sum&#95;{i:\Delta&#95;i>0}
\frac{\Delta&#95;i}
{\operatorname{kl}(\mu&#95;i,\mu^\star)}.}
$$

证明的每一步都放在附录 G。主线只保留最重要的逻辑：选一个事件来表示“臂 $i$ 被探索得很少”，分别在两个环境中估计这个事件的概率，再用 KL 散度把两个概率的差异压回观察次数。

## 8. 这个下界究竟限制了什么

在广告例子中，如果臂 B 的真实点击率为 $0.09$，而替代世界把它提高到略高于 A 的 $0.10$，那么算法必须收集足够多的 B 的点击反馈，才能排除替代世界。每次曝光只贡献

$$
\operatorname{kl}(0.09,0.10)
$$

量级的证据，因此所需曝光数至少是

$$
\frac{\log T}{\operatorname{kl}(0.09,0.10)}
$$

量级。

注意这里的“至少”不是说后台必须设置一个固定配额，也不是说每个有限的 $T$ 都精确等于这个数。它是长期渐近意义下的不可避免尺度，告诉我们问题本身需要多少信息。

如果 B 与 A 的差距很大，$\operatorname{kl}(\mu&#95;B,\mu^\star)$ 往往较大，少量样本就能看出差别；如果两者非常接近，KL 很小，任何算法都必须付出更多探索成本。这解释了一个常见现象：最难的实验不是差方案，而是两个表现几乎一样的方案。

## 9. 两种“难度”：实例相关与最坏情形

上面的 Lai--Robbins 下界针对一组固定的均值向量，常数依赖每个 gap 和 KL 散度。这叫**实例相关下界**（instance-dependent lower bound）。当每个臂的差距固定且非零时，长期遗憾的主导项是 $\log T$。

还有另一种问题：如果只知道有 $K$ 个臂，却不知道它们的均值，能否找到一个对所有环境都好的算法？这时对手可以选择一个极其困难的环境，让两个臂的均值差距随着 $T$ 变小。算法需要在几乎无法区分的世界之间做决定，最坏情形遗憾会达到

$$
\Omega(\sqrt{KT})
$$

的量级（在 $K=2$ 的情形就是 $\Omega(\sqrt T)$）。

这两个结论并不矛盾：固定差距的实例最终会被识别，遗憾是对数级；最坏情形允许差距随时间缩小，算法始终处于“证据刚刚够、但又不够”的状态，遗憾因此变成平方根级。

附录 H 用两个 Bernoulli 世界和总变差距离给出一个两臂 $\Omega(\sqrt T)$ 的完整证明。这样可以清楚看到：对数下界描述长期识别一个固定实例的成本，平方根下界描述面对最困难实例时的统一保证。

## 10. UCB、KL-UCB 与 Thompson Sampling 的位置

UCB 用置信区间上界表达“目前仍然可能很好的情况”；KL-UCB 则直接用 Bernoulli KL 约束经验均值与候选均值之间的距离；Thompson Sampling 从后验分布中抽取一个可能世界，再在该世界里利用。

它们的实现方式不同，但都必须面对同一件事：要把一个次优臂的后验或置信集合压到足够小，必须观察它；而观察次数不能低于区分近邻环境所需要的证据量。

在适当的正则条件下，KL-UCB 与 Thompson Sampling 都可以达到 Lai--Robbins 下界给出的渐近常数。对工程系统而言，这意味着算法比较不应只看某一段流量上的点击率，还要问：

- 它是否在真正困难的近邻世界上仍然有效？
- 它付出的探索成本是否接近信息论下界？
- 它的额外损失来自算法设计，还是来自安全约束、延迟反馈与非平稳性？

下界把“聪明”从一个模糊的形容词变成了可检验的标准：在必须收集这些信息的前提下，算法有没有把每一次探索用好。

## 11. 结语：先问问题允许什么，再问算法能做什么

多臂老虎机最有价值的地方，不在于它把广告点击简化成 $0$ 和 $1$，而在于它揭示了在线决策的一个基本事实：**信息本身需要成本**。

一个系统不可能一边从未见过某个方案，一边又确定它不是最好的；也不可能在两个统计上几乎相同的世界中立刻作出可靠区分。探索带来的短期遗憾，是换取长期判断能力的费用。

因此，分析一个新算法时，问题的顺序应该是：它面对的环境是什么？要区分哪些近邻世界？每次观测提供多少信息？理论下界规定了多少不可避免的成本？只有在这些问题之后，才轮到比较算法的工程实现。

## 参考文献

1. H. Robbins, “Some Aspects of the Sequential Design of Experiments,” *Bulletin of the American Mathematical Society*, 1952. [Project Euclid](https://projecteuclid.org/journals/bulletin-of-the-american-mathematical-society/volume-58/issue-5/Some-aspects-of-the-sequential-design-of-experiments/10.1090/S0002-9904-1952-09620-8.full).
2. T. L. Lai and H. Robbins, “Asymptotically Efficient Adaptive Allocation Rules,” *Advances in Applied Mathematics*, 1985. [ScienceDirect](https://doi.org/10.1016/0196-8858(85)90002-8).
3. P. Auer, N. Cesa-Bianchi, and P. Fischer, “Finite-time Analysis of the Multiarmed Bandit Problem,” *Machine Learning*, 2002. [Springer](https://doi.org/10.1023/A:1013689704352).
4. W. Hoeffding, “Probability Inequalities for Sums of Bounded Random Variables,” *Journal of the American Statistical Association*, 1963. [JSTOR](https://doi.org/10.2307/2282952).
5. A. Garivier and O. Cappé, “The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond,” *Proceedings of the 24th Annual Conference on Learning Theory*, 2011. [PMLR](https://proceedings.mlr.press/v19/garivier11a.html).
6. E. Kaufmann, N. Korda, and R. Munos, “Thompson Sampling: An Asymptotically Optimal Finite-Time Analysis,” *Algorithmic Learning Theory*, 2012. [arXiv](https://arxiv.org/abs/1205.4217).
7. S. Bubeck and N. Cesa-Bianchi, “Regret Analysis of Stochastic and Nonstochastic Multi-armed Bandit Problems,” *Foundations and Trends in Machine Learning*, 2012. [Now Publishers](https://doi.org/10.1561/2200000024).

---

# 附录

如下为正文附录补充。

## A. 符号、历史与策略

$K$ 是臂的数量，$T$ 是总轮数。第 $t$ 轮的动作和奖励分别为 $A&#95;t$ 与 $X&#95;t$。第 $t$ 轮之前的历史是

$$
H&#95;{t-1}:=(A&#95;1,X&#95;1,\ldots,A&#95;{t-1},X&#95;{t-1}),
$$

并约定 $H&#95;0$ 为空历史。

策略可以随机化。给定历史 $h&#95;{t-1}$ 后，记

$$
\pi&#95;t(a\mid h&#95;{t-1})
:=\mathbb{P}(A&#95;t=a\mid H&#95;{t-1}=h&#95;{t-1}).
$$

对每个固定历史，$\pi&#95;t(\cdot\mid h&#95;{t-1})$ 是一个概率分布，因此

$$
\pi&#95;t(a\mid h&#95;{t-1})\geq0,
\qquad
\sum&#95;{a=1}^{K}\pi&#95;t(a\mid h&#95;{t-1})=1.
$$

给定真实均值向量 $\mu=(\mu&#95;1,\ldots,\mu&#95;K)$，臂 $i$ 的奖励满足

$$
\mathbb{P}&#95;\mu(X&#95;t=1\mid A&#95;t=i,H&#95;{t-1})=\mu&#95;i,
$$

$$
\mathbb{P}&#95;\mu(X&#95;t=0\mid A&#95;t=i,H&#95;{t-1})=1-\mu&#95;i.
$$

这里的条件独立是指：给定动作和真实均值后，当前奖励的分布不再依赖更早的历史。动作仍然可以依赖历史，所以这是一个自适应实验，而不是预先固定样本数的实验。

## B. Bernoulli 分布与遗憾分解的逐行证明

若 $X\sim\operatorname{Bernoulli}(\mu)$，则对 $x\in\{0,1\}$

$$
\mathbb{P}(X=x)=\mu^x(1-\mu)^{1-x}.
$$

当 $x=1$ 时，右端是 $\mu$；当 $x=0$ 时，右端是 $1-\mu$。因此

$$
\begin{aligned}
\mathbb{E}[X]
&=0\cdot\mathbb{P}(X=0)+1\cdot\mathbb{P}(X=1)\\
&=0\cdot(1-\mu)+1\cdot\mu\\
&=\mu.
\end{aligned}
$$

因为 $X$ 只取 $0$ 和 $1$，所以 $X^2=X$，从而

$$
\mathbb{E}[X^2]=\mu.
$$

按方差定义，

$$
\begin{aligned}
\operatorname{Var}(X)
&=\mathbb{E}[X^2]-(\mathbb{E}[X])^2\\
&=\mu-\mu^2\\
&=\mu(1-\mu).
\end{aligned}
$$

下面证明遗憾分解。因为每轮恰好选择一个臂，

$$
\sum&#95;{i=1}^{K}\mathbf{1}\{A&#95;t=i\}=1.
$$

在第 $t$ 轮有

$$
\begin{aligned}
\mu^\star-\mu&#95;{A&#95;t}
&=(\mu^\star-\mu&#95;{A&#95;t})
\sum&#95;{i=1}^{K}\mathbf{1}\{A&#95;t=i\}\\
&=\sum&#95;{i=1}^{K}
\mathbf{1}\{A&#95;t=i\}(\mu^\star-\mu&#95;{A&#95;t}).
\end{aligned}
$$

当指标为 $1$ 时，$A&#95;t=i$，所以

$$
\mathbf{1}\{A&#95;t=i\}(\mu^\star-\mu&#95;{A&#95;t})
=\mathbf{1}\{A&#95;t=i\}(\mu^\star-\mu&#95;i)
=\mathbf{1}\{A&#95;t=i\}\Delta&#95;i.
$$

当指标为 $0$ 时，两边也都为 $0$。因此逐轮求和：

$$
\begin{aligned}
R&#95;T
&=\sum&#95;{t=1}^{T}\sum&#95;{i=1}^{K}
\mathbf{1}\{A&#95;t=i\}\Delta&#95;i\\
&=\sum&#95;{i=1}^{K}\Delta&#95;i
\sum&#95;{t=1}^{T}\mathbf{1}\{A&#95;t=i\}\\
&=\sum&#95;{i=1}^{K}\Delta&#95;iN&#95;i(T).
\end{aligned}
$$

最后，$K$ 是有限数，期望可以与有限和交换：

$$
\begin{aligned}
\mathbb{E}[R&#95;T]
&=\mathbb{E}\left[\sum&#95;{i=1}^{K}\Delta&#95;iN&#95;i(T)\right]\\
&=\sum&#95;{i=1}^{K}\Delta&#95;i\mathbb{E}[N&#95;i(T)].
\end{aligned}
$$

## C. KL 散度的基本性质

### C.1 从概率质量函数得到 Bernoulli KL

Bernoulli$(p)$ 与 Bernoulli$(q)$ 的概率质量函数分别为

$$
p&#95;p(x)=p^x(1-p)^{1-x},
\qquad
p&#95;q(x)=q^x(1-q)^{1-x}.
$$

离散分布的 KL 散度定义为

$$
\operatorname{KL}(P\Vert Q)
:=\sum&#95;{x\in\{0,1\}}P(x)\log\frac{P(x)}{Q(x)}.
$$

把 $x=1$ 与 $x=0$ 两项分别写出：

$$
\begin{aligned}
\operatorname{KL}(\operatorname{Bern}(p)\Vert\operatorname{Bern}(q))
&=p\log\frac{p}{q}
+(1-p)\log\frac{1-p}{1-q}\\
&=\operatorname{kl}(p,q).
\end{aligned}
$$

### C.2 非负性与等号条件

对 $u>0$，基本不等式

$$
\log u\leq u-1
$$

可由函数 $g(u)=u-1-\log u$ 的导数证明：

$$
g'(u)=1-\frac1u=\frac{u-1}{u},
$$

所以 $u=1$ 是全局最小点，且 $g(1)=0$。

令 $u=Q(x)/P(x)$，则

$$
\begin{aligned}
\operatorname{KL}(P\Vert Q)
&=-\mathbb{E}&#95;P\left[\log\frac{Q(X)}{P(X)}\right]\\
&\geq-\mathbb{E}&#95;P\left[\frac{Q(X)}{P(X)}-1\right]\\
&=-\sum&#95;xQ(x)+\sum&#95;xP(x)\\
&=0.
\end{aligned}
$$

等号成立当且仅当 $Q(x)/P(x)=1$ 在 $P$ 几乎处处成立，即两个分布相同。对于 Bernoulli 分布，这意味着 $p=q$。

### C.3 连续性与近邻展开

对 $p,q\in(0,1)$，$\operatorname{kl}(p,q)$ 是由对数和乘法组成的连续函数。固定 $p$，令 $q\to p$，则

$$
\operatorname{kl}(p,q)\to0.
$$

对 $q$ 求导：

$$
\frac{\partial}{\partial q}\operatorname{kl}(p,q)
=-\frac{p}{q}+\frac{1-p}{1-q}
=\frac{q-p}{q(1-q)}.
$$

在 $q=p$ 处一阶导数为 $0$。再次求导：

$$
\frac{\partial^2}{\partial q^2}\operatorname{kl}(p,q)
=\frac{p}{q^2}+\frac{1-p}{(1-q)^2}.
$$

代入 $q=p$ 得

$$
\left.\frac{\partial^2}{\partial q^2}\operatorname{kl}(p,q)\right|&#95;{q=p}
=\frac1{p(1-p)}.
$$

所以 Taylor 展开给出

$$
\operatorname{kl}(p,p+\varepsilon)
=\frac{\varepsilon^2}{2p(1-p)}+o(\varepsilon^2).
$$

这说明两个点击率越接近，KL 以差距的平方缩小；要得到固定证据量，样本数就会以差距倒数平方的尺度增加。

## D. Hoeffding 不等式与 UCB 上界

### D.0 Hoeffding 引理

先证明 D.1 中使用的矩母函数界。若 $Y\in[a,b]$ 且 $\mathbb{E}Y=\mu$，则对任意 $\lambda\in\mathbb{R}$，

$$
\log\mathbb{E}\left[e^{\lambda(Y-\mu)}\right]
\leq\frac{\lambda^2(b-a)^2}{8}.
$$

令

$$
\theta:=\frac{Y-a}{b-a}\in[0,1],
\qquad
p:=\mathbb{E}\theta=\frac{\mu-a}{b-a}.
$$

由于指数函数凸，任意 $z\in[0,1]$ 都满足

$$
e^{uz}\leq(1-z)+ze^u.
$$

取 $u=\lambda(b-a)$、$z=\theta$ 并取期望：

$$
\mathbb{E}e^{u\theta}
\leq1-p+pe^u.
$$

另一方面，$Y-\mu=(b-a)(\theta-p)$，所以只需证明

$$
\log(1-p+pe^u)-pu\leq\frac{u^2}{8}.
$$

令左侧为 $f(u)$。记

$$
r(u):=\frac{pe^u}{1-p+pe^u}.
$$

直接求导：

$$
f'(u)=r(u)-p,
$$

$$
f''(u)=r(u)(1-r(u))\leq\frac14.
$$

因为 $f(0)=0$、$f'(0)=0$，对 $f''(u)\leq1/4$ 积分两次可得

$$
f(u)=\int&#95;0^u\int&#95;0^v f''(w)\,\mathrm dw\,\mathrm dv
\leq\frac{u^2}{8}
$$

（$u<0$ 时同样对区间 $[u,0]$ 积分）。代回 $u=\lambda(b-a)$，即得

$$
\log\mathbb{E}e^{\lambda(Y-\mu)}
\leq\frac{\lambda^2(b-a)^2}{8}.
$$

### D.1 Hoeffding 不等式

若 $Y&#95;1,\ldots,Y&#95;n$ 独立，且 $Y&#95;j\in[0,1]$、$\mathbb{E}Y&#95;j=\mu$，令

$$
\overline Y&#95;n:=\frac1n\sum&#95;{j=1}^{n}Y&#95;j,
$$

则对任意 $\varepsilon>0$，

$$
\mathbb{P}(\overline Y&#95;n-\mu\geq\varepsilon)
\leq e^{-2n\varepsilon^2},
$$

$$
\mathbb{P}(\mu-\overline Y&#95;n\geq\varepsilon)
\leq e^{-2n\varepsilon^2}.
$$

两式相加得到

$$
\mathbb{P}(|\overline Y&#95;n-\mu|\geq\varepsilon)
\leq2e^{-2n\varepsilon^2}.
$$

证明如下。对任意 $\lambda>0$，Markov 不等式给出

$$
\begin{aligned}
\mathbb{P}\left(\sum&#95;{j=1}^{n}(Y&#95;j-\mu)\geq n\varepsilon\right)
&=\mathbb{P}\left(e^{\lambda\sum&#95;j(Y&#95;j-\mu)}
\geq e^{\lambda n\varepsilon}\right)\\
&\leq e^{-\lambda n\varepsilon}
\mathbb{E}\left[e^{\lambda\sum&#95;j(Y&#95;j-\mu)}\right].
\end{aligned}
$$

独立性使矩母函数因子化：

$$
\mathbb{E}\left[e^{\lambda\sum&#95;j(Y&#95;j-\mu)}\right]
=\prod&#95;{j=1}^{n}
\mathbb{E}[e^{\lambda(Y&#95;j-\mu)}].
$$

Hoeffding 引理给出每一项不超过 $e^{\lambda^2/8}$，因此

$$
\mathbb{P}(\overline Y&#95;n-\mu\geq\varepsilon)
\leq\exp\left(-\lambda n\varepsilon+\frac{n\lambda^2}{8}\right).
$$

右侧指数是关于 $\lambda$ 的二次函数。令导数为零：

$$
-n\varepsilon+\frac{n\lambda}{4}=0
\quad\Longrightarrow\quad
\lambda=4\varepsilon.
$$

代回得到 $e^{-2n\varepsilon^2}$。对 $\mu-\overline Y&#95;n$ 应用同样推导即可。

### D.2 UCB 的计数

固定次优臂 $i$，令

$$
m&#95;i:=\left\lceil\frac{8\log T}{\Delta&#95;i^2}\right\rceil.
$$

当 $N&#95;i(t-1)\geq m&#95;i$ 时，

$$
\sqrt{\frac{2\log T}{N&#95;i(t-1)}}
\leq\sqrt{\frac{2\log T}{8\log T/\Delta&#95;i^2}}
=\frac{\Delta&#95;i}{2}.
$$

定义好事件

$$
G&#95;t:=\bigcap&#95;{j=1}^{K}
\lbrace
\left|\widehat\mu&#95;j(t-1)-\mu&#95;j\right|
<\sqrt{\frac{2\log T}{N&#95;j(t-1)}}
\rbrace.
$$

若 $G&#95;t$ 发生且 $N&#95;i(t-1)\geq m&#95;i$，则最优臂的 UCB 满足

$$
\operatorname{UCB}&#95;{i^\star}(t)
\geq\mu^\star,
$$

而臂 $i$ 的 UCB 满足

$$
\begin{aligned}
\operatorname{UCB}&#95;i(t)
&=\widehat\mu&#95;i(t-1)
+\sqrt{\frac{2\log T}{N&#95;i(t-1)}}\\
&\leq\mu&#95;i
+2\sqrt{\frac{2\log T}{N&#95;i(t-1)}}\\
&\leq\mu&#95;i+\Delta&#95;i\\
&=\mu^\star.
\end{aligned}
$$

因此好事件上，臂 $i$ 不可能严格超过最优臂；它再次被选只能发生在 $G&#95;t$ 失败时，或还没有达到 $m&#95;i$ 次样本时。于是

$$
N&#95;i(T)
\leq m&#95;i+\sum&#95;{t=1}^{T}\mathbf{1}\{G&#95;t^c\}.
$$

对两边取期望。对固定的臂 $j$ 和固定样本数 $n$，Hoeffding 不等式取

$$
\varepsilon=\sqrt{\frac{2\log T}{n}}
$$

给出偏离概率至多 $2T^{-4}$。在时间 $t$，$N&#95;j(t-1)$ 只能取 $1,\ldots,t-1$ 中的一个值，因此并合界给出

$$
\mathbb{P}(G&#95;t^c)
\leq\sum&#95;{j=1}^{K}\sum&#95;{n=1}^{T}
2T^{-4}
=2KT^{-3}.
$$

这个粗略界带有 $K$。若把每个臂的样本序列预先耦合，并只对最优臂和当前臂做计数，则可得到正文使用的常数 $4T^{-3}$；无论采用哪一种写法，错误事件的总贡献都是可忽略的 $O(T^{-2})$。使用粗略界也得到

$$
\mathbb{E}[N&#95;i(T)]
\leq m&#95;i+2KT^{-2},
$$

从而得到对数级 UCB 上界。这里的作用只是提供一个可比较的上界；下界并不依赖 UCB 的具体形式。

## E. 自适应历史 KL 分解的逐行证明

设历史为 $h&#95;T=(a&#95;1,x&#95;1,\ldots,a&#95;T,x&#95;T)$。在环境 $\nu$ 下，其联合密度按链式法则写成

$$
\begin{aligned}
P&#95;\nu^T(h&#95;T)
&=\prod&#95;{t=1}^{T}
\mathbb{P}&#95;\nu(A&#95;t=a&#95;t\mid h&#95;{t-1})
\mathbb{P}&#95;\nu(X&#95;t=x&#95;t\mid a&#95;t,h&#95;{t-1})\\
&=\prod&#95;{t=1}^{T}
\pi&#95;t(a&#95;t\mid h&#95;{t-1})p&#95;{a&#95;t}(x&#95;t).
\end{aligned}
$$

第二个等号使用了条件独立：给定当前动作和环境，奖励分布只由被选臂决定。环境 $\lambda$ 同理：

$$
P&#95;\lambda^T(h&#95;T)
=\prod&#95;{t=1}^{T}
\pi&#95;t(a&#95;t\mid h&#95;{t-1})q&#95;{a&#95;t}(x&#95;t).
$$

假设出现的点上分母非零。两式相除：

$$
\begin{aligned}
\frac{P&#95;\nu^T(h&#95;T)}{P&#95;\lambda^T(h&#95;T)}
&=\frac{\prod&#95;t\pi&#95;t(a&#95;t\mid h&#95;{t-1})p&#95;{a&#95;t}(x&#95;t)}
{\prod&#95;t\pi&#95;t(a&#95;t\mid h&#95;{t-1})q&#95;{a&#95;t}(x&#95;t)}\\
&=\prod&#95;{t=1}^{T}
\frac{p&#95;{a&#95;t}(x&#95;t)}{q&#95;{a&#95;t}(x&#95;t)}.
\end{aligned}
$$

策略项之所以抵消，是因为比较的是同一个算法在两个不同环境中的行为。算法可以根据历史改变动作概率，但这个改变在两个世界中完全相同。

取对数：

$$
\log\frac{P&#95;\nu^T(H&#95;T)}{P&#95;\lambda^T(H&#95;T)}
=\sum&#95;{t=1}^{T}
\log\frac{p&#95;{A&#95;t}(X&#95;t)}{q&#95;{A&#95;t}(X&#95;t)}.
$$

KL 定义要求在 $P&#95;\nu^T$ 下取期望：

$$
\begin{aligned}
\operatorname{KL}(P&#95;\nu^T\Vert P&#95;\lambda^T)
&=\mathbb{E}&#95;\nu\left[
\sum&#95;{t=1}^{T}
\log\frac{p&#95;{A&#95;t}(X&#95;t)}{q&#95;{A&#95;t}(X&#95;t)}
\right]\\
&=\sum&#95;{t=1}^{T}\mathbb{E}&#95;\nu\left[
\log\frac{p&#95;{A&#95;t}(X&#95;t)}{q&#95;{A&#95;t}(X&#95;t)}
\right].
\end{aligned}
$$

对第 $t$ 项按动作分组：

$$
\begin{aligned}
&\mathbb{E}&#95;\nu\left[
\log\frac{p&#95;{A&#95;t}(X&#95;t)}{q&#95;{A&#95;t}(X&#95;t)}
\right]\\
&=\sum&#95;{a=1}^{K}
\mathbb{P}&#95;\nu(A&#95;t=a)
\mathbb{E}&#95;\nu\left[
\left.\log\frac{p&#95;a(X&#95;t)}{q&#95;a(X&#95;t)}
\right|A&#95;t=a\right]\\
&=\sum&#95;{a=1}^{K}
\mathbb{P}&#95;\nu(A&#95;t=a)
\operatorname{KL}(\nu&#95;a\Vert\lambda&#95;a).
\end{aligned}
$$

最后交换有限求和，并使用

$$
\mathbb{E}&#95;\nu[N&#95;a(T)]
=\sum&#95;{t=1}^{T}\mathbb{P}&#95;\nu(A&#95;t=a),
$$

得到

$$
\operatorname{KL}(P&#95;\nu^T\Vert P&#95;\lambda^T)
=\sum&#95;{a=1}^{K}
\mathbb{E}&#95;\nu[N&#95;a(T)]
\operatorname{KL}(\nu&#95;a\Vert\lambda&#95;a).
$$

## F. 二元数据处理不等式

令 $P,Q$ 是历史 $H&#95;T$ 的两个分布，$E$ 是一个由历史决定的事件。记

$$
p:=P(E),\qquad q:=Q(E).
$$

我们证明

$$
\operatorname{KL}(P\Vert Q)
\geq\operatorname{kl}(p,q).
$$

把样本空间分成 $E$ 和 $E^c$ 两块。对 $x\in E$，令条件密度为 $P(x\mid E)$ 和 $Q(x\mid E)$；对 $x\in E^c$ 同理。于是

$$
P(x)=
\begin{cases}
pP(x\mid E),&x\in E,\\
(1-p)P(x\mid E^c),&x\in E^c,
\end{cases}
$$

$$
Q(x)=
\begin{cases}
qQ(x\mid E),&x\in E,\\
(1-q)Q(x\mid E^c),&x\in E^c.
\end{cases}
$$

代入 KL，并把两块分别相加：

$$
\begin{aligned}
\operatorname{KL}(P\Vert Q)
&=p\log\frac pq+(1-p)\log\frac{1-p}{1-q}\\
&\quad+p\operatorname{KL}(P(\cdot\mid E)\Vert Q(\cdot\mid E))\\
&\quad+(1-p)\operatorname{KL}(P(\cdot\mid E^c)\Vert Q(\cdot\mid E^c)).
\end{aligned}
$$

条件分布的两个 KL 都非负，所以删去它们后得到

$$
\operatorname{KL}(P\Vert Q)\geq\operatorname{kl}(p,q).
$$

这说明把完整历史压缩成一个二元事件不会增加可区分信息。

## G. Lai--Robbins 下界的逐步证明

### G.1 两个环境与事件

固定次优臂 $i$，令其真实均值为 $\mu&#95;i<\mu^\star$。取任意

$$
q>\mu^\star
$$

并定义替代环境 $\nu^{(q)}$：臂 $i$ 的均值改为 $q$，其余臂均值不变。

取 $b\in(0,1)$，定义事件

$$
E&#95;T:=\{N&#95;i(T)\leq T^b\}.
$$

它表示“截至 $T$，臂 $i$ 被探索得很少”。我们将证明真实环境下 $E&#95;T$ 几乎必然发生，而替代环境下它的概率必须很小。

### G.2 真实环境中的概率

在真实环境中，臂 $i$ 次优。因此每一次选择它至少产生 $\Delta&#95;i$ 的期望遗憾，逐点有

$$
R&#95;T\geq\Delta&#95;iN&#95;i(T).
$$

所以

$$
\mathbb{E}&#95;\nu[N&#95;i(T)]
\leq\frac{\mathbb{E}&#95;\nu[R&#95;T]}{\Delta&#95;i}
=o(T^c)
$$

对任意 $c>0$ 成立。事件 $E&#95;T^c$ 意味着 $N&#95;i(T)>T^b$，因此 Markov 不等式给出

$$
\begin{aligned}
\mathbb{P}&#95;\nu(E&#95;T^c)
&=\mathbb{P}&#95;\nu(N&#95;i(T)>T^b)\\
&\leq\frac{\mathbb{E}&#95;\nu[N&#95;i(T)]}{T^b}\\
&=o(T^{c-b}).
\end{aligned}
$$

选择 $0<c<b$，因为 $c-b<0$，所以右端趋于 $0$，即

$$
p&#95;T:=\mathbb{P}&#95;\nu(E&#95;T)\longrightarrow1.
$$

### G.3 替代环境中的概率

在替代环境中，臂 $i$ 的均值为 $q$，其他臂的均值不超过 $\mu^\star$。令

$$
\delta&#95;q:=q-\mu^\star>0.
$$

每一次没有选择臂 $i$，相对于最优均值 $q$ 至少损失 $\delta&#95;q$，所以对每条路径

$$
R&#95;T\geq\delta&#95;q(T-N&#95;i(T)).
$$

若事件 $E&#95;T$ 发生，则 $N&#95;i(T)\leq T^b$，从而

$$
T-N&#95;i(T)\geq T-T^b.
$$

因此

$$
\begin{aligned}
\mathbb{P}&#95;{\nu^{(q)}}(E&#95;T)
&\leq\mathbb{P}&#95;{\nu^{(q)}}
\left(T-N&#95;i(T)\geq T-T^b\right)\\
&\leq\frac{\mathbb{E}&#95;{\nu^{(q)}}[T-N&#95;i(T)]}{T-T^b}\\
&\leq\frac{\mathbb{E}&#95;{\nu^{(q)}}[R&#95;T]}
{\delta&#95;q(T-T^b)}.
\end{aligned}
$$

对任意 $c>0$，uniform efficiency 给出 $\mathbb{E}&#95;{\nu^{(q)}}[R&#95;T]=o(T^c)$，所以

$$
q&#95;T:=\mathbb{P}&#95;{\nu^{(q)}}(E&#95;T)
=o(T^{c-1}).
$$

### G.4 把事件概率差异变成 KL

由附录 F 的二元数据处理不等式，以及附录 E 的自适应 KL 分解：

$$
\begin{aligned}
\mathbb{E}&#95;\nu[N&#95;i(T)]\operatorname{kl}(\mu&#95;i,q)
&=\operatorname{KL}(P&#95;\nu^T\Vert P&#95;{\nu^{(q)}}^T)\\
&\geq\operatorname{kl}(p&#95;T,q&#95;T).
\end{aligned}
$$

现在估计右侧。二元熵

$$
h(p):=-p\log p-(1-p)\log(1-p)
$$

满足 $0\leq h(p)\leq\log2$。展开二元 KL：

$$
\begin{aligned}
\operatorname{kl}(p,q)
&=p\log p+(1-p)\log(1-p)\\
&\quad-p\log q-(1-p)\log(1-q)\\
&=-h(p)+p\log\frac1q+(1-p)\log\frac1{1-q}\\
&\geq p\log\frac1q-\log2.
\end{aligned}
$$

取 $p=p&#95;T\to1$，并使用 $q&#95;T=o(T^{c-1})$。对任意小的 $\varepsilon>0$，当 $T$ 足够大时，$p&#95;T\geq1-\varepsilon$，且 $q&#95;T\leq T^{c-1+\varepsilon}$。于是

$$
\begin{aligned}
\operatorname{kl}(p&#95;T,q&#95;T)
&\geq(1-\varepsilon)(1-c-\varepsilon)\log T-\log2.
\end{aligned}
$$

令 $T\to\infty$，再令 $\varepsilon\downarrow0$，得到

$$
\liminf&#95;{T\to\infty}
\frac{\mathbb{E}&#95;\nu[N&#95;i(T)]}{\log T}
\geq\frac{1-c}{\operatorname{kl}(\mu&#95;i,q)}.
$$

因为 $c>0$ 可以任意小，

$$
\liminf&#95;{T\to\infty}
\frac{\mathbb{E}&#95;\nu[N&#95;i(T)]}{\log T}
\geq\frac{1}{\operatorname{kl}(\mu&#95;i,q)}.
$$

### G.5 让替代世界逼近真实边界

上式对每个固定的 $q>\mu^\star$ 都成立。由附录 C 的连续性，令 $q\downarrow\mu^\star$：

$$
\operatorname{kl}(\mu&#95;i,q)
\longrightarrow
\operatorname{kl}(\mu&#95;i,\mu^\star).
$$

于是

$$
\boxed{
\liminf&#95;{T\to\infty}
\frac{\mathbb{E}&#95;\nu[N&#95;i(T)]}{\log T}
\geq\frac{1}{\operatorname{kl}(\mu&#95;i,\mu^\star)}}.
$$

乘以 $\Delta&#95;i$ 并对所有次优臂求和，利用附录 B 的遗憾分解，得到

$$
\liminf&#95;{T\to\infty}
\frac{\mathbb{E}&#95;\nu[R&#95;T]}{\log T}
\geq\sum&#95;{i:\Delta&#95;i>0}
\frac{\Delta&#95;i}
{\operatorname{kl}(\mu&#95;i,\mu^\star)}.
$$

这就是 Lai--Robbins 型实例相关下界。证明中的关键不是某个具体算法，而是同一个策略在两个环境中的历史分布必须足够不同；而这种不同只能由实际观测积累出来。

## H. 两臂最坏情形的 $\Omega(\sqrt T)$ 下界

这一节证明一个较粗但非常有用的事实：如果要求一个算法对所有两臂 Bernoulli 环境都表现良好，那么最坏情形遗憾不可能低于 $\sqrt T$ 的量级。

### H.1 构造两个近邻环境

取 $0<\varepsilon\leq1/4$，定义环境 $+$ 与环境 $-$：

$$
\mu^+=(1/2+\varepsilon,\,1/2),
$$

$$
\mu^-=(1/2,\,1/2+\varepsilon).
$$

在环境 $+$ 中臂 $1$ 最优，在环境 $-$ 中臂 $2$ 最优。令事件

$$
E:=\{N&#95;1(T)>T/2\}.
$$

在环境 $+$ 中，如果 $E^c$ 发生，则 $N&#95;1(T)\leq T/2$，所以至少有 $T/2$ 次没有选择最优臂：

$$
R&#95;T^+\geq\varepsilon(T-N&#95;1(T))
\geq\frac{\varepsilon T}{2}.
$$

因此

$$
\mathbb{E}&#95;+R&#95;T
\geq\frac{\varepsilon T}{2}\mathbb{P}&#95;+(E^c).
$$

同理，在环境 $-$ 中，若 $E$ 发生，则至少有 $T/2$ 次没有选择臂 $2$，于是

$$
\mathbb{E}&#95;-R&#95;T
\geq\frac{\varepsilon T}{2}\mathbb{P}&#95;-(E).
$$

两式相加：

$$
\mathbb{E}&#95;+R&#95;T+\mathbb{E}&#95;-R&#95;T
\geq\frac{\varepsilon T}{2}
\left(\mathbb{P}&#95;+(E^c)+\mathbb{P}&#95;-(E)\right).
$$

### H.2 用总变差控制判别错误

对任意事件 $E$，

$$
\mathbb{P}&#95;+(E^c)+\mathbb{P}&#95;-(E)
=1-\left(\mathbb{P}&#95;+(E)-\mathbb{P}&#95;-(E)\right)
\geq1-\operatorname{TV}(P&#95;+^T,P&#95;-^T),
$$

其中总变差距离定义为

$$
\operatorname{TV}(P,Q):=\sup&#95;E|P(E)-Q(E)|.
$$

下面从二元 KL 直接证明 Pinsker 不等式，而不把它当作黑箱使用。先证明对任意 $p,q\in[0,1]$，

$$
\operatorname{kl}(p,q)\geq2(p-q)^2.
$$

固定 $q$，令

$$
g(p):=\operatorname{kl}(p,q)-2(p-q)^2.
$$

对 $p\in(0,1)$ 求导：

$$
g'(p)=\log\frac{p(1-q)}{q(1-p)}-4(p-q),
$$

$$
g''(p)=\frac1p+\frac1{1-p}-4
=\frac1{p(1-p)}-4\geq0,
$$

最后一个不等式来自 $p(1-p)\leq1/4$。又有 $g(q)=0$、$g'(q)=0$，所以 $g$ 在 $p=q$ 处取全局最小值，得到所需不等式；端点由连续性得到。

取一个达到总变差的事件 $E$（有限历史空间中必然存在；一般情形可用逼近事件），令 $p=P(E)$、$q=Q(E)$。由附录 F 的二元数据处理不等式：

$$
\operatorname{KL}(P\Vert Q)
\geq\operatorname{kl}(p,q)
\geq2(p-q)^2
=2\operatorname{TV}(P,Q)^2.
$$

于是得到 Pinsker 不等式

$$
\operatorname{TV}(P,Q)
\leq\sqrt{\frac12\operatorname{KL}(P\Vert Q)}.
$$

两环境只在被选择的臂上交换 Bernoulli 参数。利用附录 E 的自适应分解，并使用 $\operatorname{KL}(P\Vert Q)\leq\chi^2(P\Vert Q)$，单次 Bernoulli 差异满足

$$
\operatorname{KL}(\operatorname{Bern}(1/2+\varepsilon)
\Vert\operatorname{Bern}(1/2))
\leq4\varepsilon^2,
$$

反向方向也有同样上界。因此无论算法如何选臂，

$$
\operatorname{KL}(P&#95;+^T\Vert P&#95;-^T)
\leq4\varepsilon^2T.
$$

取 $\varepsilon=1/(4\sqrt T)$，右侧不超过 $1/4$，从而

$$
\operatorname{TV}(P&#95;+^T,P&#95;-^T)
\leq\frac{1}{2\sqrt2}<\frac12.
$$

于是

$$
\mathbb{P}&#95;+(E^c)+\mathbb{P}&#95;-(E)>\frac12.
$$

代回前面的遗憾和：

$$
\begin{aligned}
\mathbb{E}&#95;+R&#95;T+\mathbb{E}&#95;-R&#95;T
&>\frac{\varepsilon T}{4}\\
&=\frac{\sqrt T}{16}.
\end{aligned}
$$

两者中至少有一个不小于和的一半，因此

$$
\boxed{
\max\{\mathbb{E}&#95;+R&#95;T,\mathbb{E}&#95;-R&#95;T\}
\geq\frac{\sqrt T}{32}.}
$$

常数并不重要，重要的是 $\sqrt T$ 的尺度。最坏情形允许环境差距随 $T$ 缩小，算法就无法像固定实例那样最终轻松区分两个世界。

### H.3 单次 KL 的上界为什么成立

对离散分布，$\operatorname{KL}(P\Vert Q)\leq\chi^2(P\Vert Q)$ 可以由 $\log u\leq u-1$ 得到。令 $r(x)=P(x)/Q(x)$：

$$
\begin{aligned}
\operatorname{KL}(P\Vert Q)
&=\sum&#95;xP(x)\log r(x)\\
&\leq\sum&#95;xP(x)(r(x)-1)\\
&=\sum&#95;x\frac{P(x)^2}{Q(x)}-1\\
&=\sum&#95;x\frac{(P(x)-Q(x))^2}{Q(x)}\\
&=\chi^2(P\Vert Q).
\end{aligned}
$$

对 $P=(1/2+\varepsilon,1/2-\varepsilon)$、$Q=(1/2,1/2)$：

$$
\begin{aligned}
\chi^2(P\Vert Q)
&=\frac{\varepsilon^2}{1/2}
+\frac{\varepsilon^2}{1/2}\\
&=4\varepsilon^2.
\end{aligned}
$$

交换两臂时只是交换坐标，所以另一方向同样不超过 $4\varepsilon^2$。

## I. 参考结论之间的关系

为了避免把不同问题的结论混在一起，最后把三种常见说法并列出来。

第一，固定实例、固定正 gap 下，Lai--Robbins 型下界说明每个次优臂至少需要

$$
\frac{\log T}{\operatorname{kl}(\mu&#95;i,\mu^\star)}
$$

量级的观察次数。这是识别近邻环境的成本。

第二，UCB1 的有限时间上界通常写成 $O(\log T/\Delta&#95;i^2)$ 的次优臂选择次数，或 $O(\log T/\Delta&#95;i)$ 的遗憾。它是某个具体算法的保证，常数未必达到信息论最优。

第三，最坏情形下界是 $\Omega(\sqrt{KT})$ 量级。它允许最困难的环境随时域变化，因此并不与固定实例的对数级下界冲突。

三者的量词不同：一个谈固定环境中的长期极限，一个谈具体规则在有限时间内的上界，一个谈所有环境中最坏的选择。读到任何“下界”时，先检查它究竟对哪个环境、哪个时域和哪个评价指标成立，才不会把一个精确结论误用到另一个问题上。
