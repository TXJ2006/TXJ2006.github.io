---
title: "算法知道自己不知道什么吗？Thompson Sampling 的贝叶斯解释"
date: 2026-08-30 10:00:00
categories:
  - 机器学习理论
tags:
  - 多臂老虎机
  - Thompson Sampling
  - 贝叶斯统计
  - 在线学习
  - 概率论
  - 数学证明
mathjax: true
toc: true
comments: true
---

一个广告已经展示一千次，点击率是 $10\\%$；另一个广告只展示二十次，点击率是 $5\\%$。如果只比较经验点击率，第一个广告显然更好。但第二个广告真的更差吗？二十次展示留下的证据太少，它可能确实平庸，也可能只是暂时运气不好。

在线决策的困难从来不只是“哪个数更大”，而是：我们对这些数究竟有多确定？

UCB 的回答是给经验均值加上置信半径，主动偏向仍有乐观可能的选项。Thompson Sampling 给出了另一种回答：从当前证据允许的所有世界中随机抽取一个，然后在这个世界里选择最优动作。

它不是随意碰运气，而是让行动的随机性反映知识的不确定性。

<!--more-->

## 1. 从一个估计值变成一整个分布

设有 $K$ 个广告，编号为 $1,\ldots,K$。第 $i$ 个广告的真实点击率记为 $\theta&#95;i\in[0,1]$。在第 $t$ 轮，系统选择广告 $A&#95;t$，随后观察奖励 $X&#95;t\in\lbrace0,1\rbrace$：点击为 $1$，未点击为 $0$。若选择了广告 $i$，则

$$
X&#95;t\mid(A&#95;t=i,\theta&#95;i)
\sim\operatorname{Bernoulli}(\theta&#95;i).
$$

符号 $\operatorname{Bernoulli}(\theta&#95;i)$ 表示参数为 $\theta&#95;i$ 的 Bernoulli 分布；竖线 $\mid$ 表示“在给定右侧条件后”；符号 $\sim$ 表示“服从某个分布”。

只记录一个估计值，就无法区分“观察一千次得到的 $10\\%$”与“观察十次得到的 $10\\%$”。贝叶斯方法为每个未知点击率设置一个分布。这里取相互独立的 Beta 先验：

$$
\theta&#95;i\sim\operatorname{Beta}(\alpha&#95;i,\beta&#95;i),
\qquad \alpha&#95;i>0,\quad\beta&#95;i>0.
$$

$\alpha&#95;i$ 和 $\beta&#95;i$ 是先验参数。若没有偏向，可以取 $\alpha&#95;i=\beta&#95;i=1$；此时 $\operatorname{Beta}(1,1)$ 就是区间 $[0,1]$ 上的均匀分布。

截至当前，若广告 $i$ 获得了 $S&#95;i$ 次点击和 $F&#95;i$ 次未点击，那么它的后验分布是

$$
\theta&#95;i\mid H
\sim\operatorname{Beta}(\alpha&#95;i+S&#95;i,\beta&#95;i+F&#95;i),
$$

其中 $H$ 表示已经观察到的全部动作与奖励，称为历史。一次点击使第一个参数增加 $1$，一次未点击使第二个参数增加 $1$。

令

$$
a&#95;i:=\alpha&#95;i+S&#95;i,
\qquad
b&#95;i:=\beta&#95;i+F&#95;i.
$$

后验均值和后验方差分别为

$$
\mathbb E[\theta&#95;i\mid H]
=\frac{a&#95;i}{a&#95;i+b&#95;i},
$$

$$
\operatorname{Var}(\theta&#95;i\mid H)
=\frac{a&#95;i b&#95;i}
{(a&#95;i+b&#95;i)^2(a&#95;i+b&#95;i+1)}.
$$

$\mathbb E$ 表示期望，$\operatorname{Var}$ 表示方差。均值描述“目前认为点击率是多少”，方差描述“对这个判断有多不确定”。上述分布与公式的推导都放在附录中。

## 2. 从后验分布中选择动作

Thompson Sampling 在每一轮只做三件事。

第一步，对每个广告分别抽取一个可能的点击率：

$$
\widetilde\theta&#95;i
\sim\operatorname{Beta}(a&#95;i,b&#95;i),
\qquad i=1,\ldots,K.
$$

第二步，选择抽样值最大的广告：

$$
A&#95;t\in\arg\max&#95;{1\leq i\leq K}\widetilde\theta&#95;i.
$$

$\arg\max$ 返回使目标函数达到最大值的编号；若多个广告并列，可以用预先规定的规则处理。

第三步，观察 $X&#95;t$。若 $X&#95;t=1$，就把被选广告的 $a&#95;i$ 加 $1$；若 $X&#95;t=0$，就把它的 $b&#95;i$ 加 $1$。没有被选择的广告在这一轮不产生新数据，后验参数保持不变。

$\widetilde\theta&#95;i$ 不是新的点估计，而是从现有证据允许的范围中抽出的一个可能值。把所有抽样值放在一起，就得到一个可能的真实世界。算法只是在这个世界中选择最好的广告。

## 3. 概率匹配

令真正的最优广告为

$$
A^\star\in\arg\max&#95;{1\leq i\leq K}\theta&#95;i.
$$

给定历史 $H&#95;{t-1}$ 后，真实参数向量

$$
\theta=(\theta&#95;1,\ldots,\theta&#95;K)
$$

服从当前后验分布；Thompson Sampling 抽出的向量

$$
\widetilde\theta=(\widetilde\theta&#95;1,\ldots,\widetilde\theta&#95;K)
$$

也服从同一个后验分布。因此，对任意广告 $i$，

$$
\boxed{
\mathbb P(A&#95;t=i\mid H&#95;{t-1})
=\mathbb P(A^\star=i\mid H&#95;{t-1})}.
$$

$\mathbb P$ 表示概率。这条等式称为概率匹配：如果现有证据认为某个广告有 $30\\%$ 的概率最优，算法就大约把 $30\\%$ 的选择机会交给它。附录 G 会逐步证明这条等式。

## 4. 一个广告系统中的具体时刻

设两个广告都使用 $\operatorname{Beta}(1,1)$ 先验。广告 A 展示 $1000$ 次，获得 $100$ 次点击；广告 B 展示 $20$ 次，只获得 $1$ 次点击。它们的后验分别为

$$
\theta&#95;A\mid H
\sim\operatorname{Beta}(101,901),
$$

$$
\theta&#95;B\mid H
\sim\operatorname{Beta}(2,20).
$$

A 的后验均值和标准差约为

$$
\mathbb E[\theta&#95;A\mid H]\approx0.1008,
\qquad
\operatorname{sd}(\theta&#95;A\mid H)\approx0.0095,
$$

B 的相应数值约为

$$
\mathbb E[\theta&#95;B\mid H]\approx0.0909,
\qquad
\operatorname{sd}(\theta&#95;B\mid H)\approx0.0600.
$$

$\operatorname{sd}$ 表示标准差，即方差的平方根。B 的后验均值更低，但不确定性大得多。

在只有这两个广告时，Thompson Sampling 选择 B 的概率是

$$
\begin{aligned}
\mathbb P(A&#95;t=B\mid H)
&=\mathbb P(\widetilde\theta&#95;B>
\widetilde\theta&#95;A\mid H)\\\\
&\approx0.363.
\end{aligned}
$$

所以系统此时仍会以约 $36.3\\%$ 的概率选择 B。这不是因为算法忽视了 B 较低的点击率，而是因为二十次观察还不足以排除“B 实际上更好”的可能性。这个概率的二重积分、化简和有限和表达式见附录 H。

如果 B 继续不被点击，它的后验会依次变成 $\operatorname{Beta}(2,21)$、$\operatorname{Beta}(2,22)$ 等，右侧尾部逐渐收缩，获得流量的概率也会下降。探索率不是预先规定的常数，而是证据变化的结果。

## 5. 随机抽样为什么不会永久浪费流量

记最优点击率为

$$
\theta^\star:=\max&#95;{1\leq i\leq K}\theta&#95;i,
$$

广告 $i$ 与最优广告之间的差距为

$$
\Delta&#95;i:=\theta^\star-\theta&#95;i\geq0.
$$

到第 $T$ 轮为止，广告 $i$ 被选择的次数记为

$$
N&#95;i(T):=\sum&#95;{t=1}^{T}
\mathbf 1\lbrace A&#95;t=i\rbrace,
$$

其中 $\mathbf 1\lbrace A&#95;t=i\rbrace$ 是指标变量：事件成立时等于 $1$，否则等于 $0$。期望遗憾可以逐步分解为

$$
\boxed{
\mathbb E[R&#95;T]
=\sum&#95;{i:\Delta&#95;i>0}
\Delta&#95;i\,\mathbb E[N&#95;i(T)]}.
$$

这说明次优广告是否危险，只取决于两件事：每次误选损失多少，以及它一共会被误选多少次。完整分解见附录 I。

随着观察次数增加，Beta 后验的标准差至多按 $n^{-1/2}$ 的尺度缩小。这里的 $n$ 表示某个广告已经获得的观察数。这个结论不是图形直觉；附录 J 会从 Beta 方差公式和不等式 $(a-b)^2\geq0$ 直接推出。

对 Bernoulli 奖励、独立的 $\operatorname{Beta}(1,1)$ 先验以及唯一最优广告，Kaufmann、Korda 与 Munos 在 2012 年证明了 Thompson Sampling 的渐近最优性。定义 Bernoulli KL 散度

$$
\operatorname{kl}(p,q)
:=p\log\frac{p}{q}
+(1-p)\log\frac{1-p}{1-q},
$$

其中 $p,q\in(0,1)$，$\log$ 表示自然对数。对每个次优广告 $i$，他们的结论与 Lai--Robbins 下界结合后给出

$$
\lim&#95;{T\to\infty}
\frac{\mathbb E[N&#95;i(T)]}{\log T}
=\frac{1}{\operatorname{kl}(\theta&#95;i,\theta^\star)}.
$$

进而

$$
\lim&#95;{T\to\infty}
\frac{\mathbb E[R&#95;T]}{\log T}
=\sum&#95;{i:\Delta&#95;i>0}
\frac{\Delta&#95;i}
{\operatorname{kl}(\theta&#95;i,\theta^\star)}.
$$

附录 K 会把“拉动次数的结论怎样推出遗憾结论”逐项写出，并区分引用的渐近最优定理与由它得到的代数推论。

## 6. 另一种保证：贝叶斯遗憾

固定一组真实点击率再评价算法，得到的是频率学派遗憾。若先从先验中抽取真实参数 $\theta$，再对参数、奖励和算法的随机性一同取期望，就得到贝叶斯遗憾，记为 $\operatorname{BR}&#95;T$：

$$
\operatorname{BR}&#95;T
:=\mathbb E
\left[
\sum&#95;{t=1}^{T}
(\theta&#95;{A^\star}-\theta&#95;{A&#95;t})
\right].
$$

若先让每个广告各展示一次，再运行 Thompson Sampling，那么对 $T\geq\max\lbrace K,2\rbrace$，可以得到一条不依赖差距 $\Delta&#95;i$ 的上界：

$$
\boxed{
\operatorname{BR}&#95;T
\leq
K+4\sqrt{2KT\log T}+\frac{4K}{T^2}}.
$$

这条界不是直接引用结论。附录 L 会从 Hoeffding 不等式、概率匹配和逐臂计数开始，完整证明每一步。它与前面的对数级结论回答不同问题：前者控制任意困难实例上的总体尺度，后者描述一组固定且彼此有差距的广告在长时间下的精细常数。

## 7. UCB 与 Thompson Sampling 的区别

UCB 为每个广告计算一个确定的乐观上界，然后选择上界最大的广告。给定同一段历史，除去平局处理，它的动作通常是确定的。

Thompson Sampling 保存整个后验分布，再随机抽取一个可能世界。它不是永远选择“最乐观的可能值”，而是让一个世界出现的频率与其后验可信度一致。

两者最终都在利用不确定性，但使用的数学对象不同：UCB 使用置信集合的边界，Thompson Sampling 使用参数的后验分布。前者问“仍然合理的最好情况是什么”，后者问“如果当前某个可能世界就是真实世界，我会怎样行动”。

## 8. 工业系统真正需要警惕什么

Thompson Sampling 的随机性来自不确定性，但这不意味着任何探索都可以直接进入生产环境。医疗方案、信贷额度或高风险推荐不能仅凭后验概率自由尝试，还需要安全阈值、预算约束与人工规则。

另一个问题是环境漂移。普通 Beta 后验会永久累积历史数据；当用户偏好发生变化时，过去的海量样本可能让后验过于自信。滑动窗口、时间折扣和变化点检测，是在非平稳系统中保持不确定性有效的必要机制。

先验也不是一个可以随意填写的装饰参数。数据稀少时，先验会真实影响流量；数据充足后，它的影响才逐渐减弱。一个可靠系统需要记录先验从何而来，并检查错误先验会造成多大的冷启动代价。

Thompson Sampling 最吸引人的地方，是它没有把探索写成一项额外任务。算法每次都在利用，只不过利用的是一个从当前知识中抽出的可能世界。知识越模糊，行动越分散；证据越充分，行动越集中。

## 参考文献

1. W. R. Thompson, “On the Likelihood that One Unknown Probability Exceeds Another in View of the Evidence of Two Samples,” *Biometrika*, 1933. [JSTOR](https://doi.org/10.2307/2332286).
2. S. Agrawal and N. Goyal, “Analysis of Thompson Sampling for the Multi-armed Bandit Problem,” *Proceedings of the 25th Annual Conference on Learning Theory*, 2012. [PMLR](https://proceedings.mlr.press/v23/agrawal12.html).
3. E. Kaufmann, N. Korda, and R. Munos, “Thompson Sampling: An Asymptotically Optimal Finite-Time Analysis,” *Algorithmic Learning Theory*, 2012. [arXiv](https://arxiv.org/abs/1205.4217).
4. D. Russo and B. Van Roy, “An Information-Theoretic Analysis of Thompson Sampling,” *Journal of Machine Learning Research*, 2016. [JMLR](https://www.jmlr.org/papers/v17/14-087.html).
5. O. Chapelle and L. Li, “An Empirical Evaluation of Thompson Sampling,” *Advances in Neural Information Processing Systems*, 2011. [NeurIPS](https://proceedings.neurips.cc/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html).

---

# 附录

如下为正文附录补充。

## A. 符号与概率模型

$K$ 表示广告总数，$T$ 表示决策总轮数。第 $t$ 轮选择的广告是 $A&#95;t\in\lbrace1,\ldots,K\rbrace$，观察到的奖励是 $X&#95;t\in\lbrace0,1\rbrace$。

第 $t$ 轮之前的历史定义为

$$
H&#95;{t-1}
:=(A&#95;1,X&#95;1,\ldots,A&#95;{t-1},X&#95;{t-1}).
$$

广告 $i$ 在前 $t-1$ 轮的点击数、未点击数和展示数分别为

$$
S&#95;i(t-1)
:=\sum&#95;{u=1}^{t-1}
\mathbf 1\lbrace A&#95;u=i,X&#95;u=1\rbrace,
$$

$$
F&#95;i(t-1)
:=\sum&#95;{u=1}^{t-1}
\mathbf 1\lbrace A&#95;u=i,X&#95;u=0\rbrace,
$$

$$
N&#95;i(t-1)
:=S&#95;i(t-1)+F&#95;i(t-1).
$$

给定真实参数 $\theta=(\theta&#95;1,\ldots,\theta&#95;K)$ 和动作 $A&#95;t=i$ 后，奖励独立地产生，并满足

$$
\mathbb P(X&#95;t=1\mid A&#95;t=i,\theta)=\theta&#95;i,
$$

$$
\mathbb P(X&#95;t=0\mid A&#95;t=i,\theta)=1-\theta&#95;i.
$$

这里的条件独立指：一旦给定被选择的广告及其真实点击率，当前奖励不再依赖更早的奖励；动作本身仍然可以依赖整个历史。

## B. Bernoulli 分布的质量函数、均值与方差

若 $X\sim\operatorname{Bernoulli}(\theta)$，其中 $\theta\in[0,1]$，则概率质量函数可以统一写为

$$
\mathbb P(X=x)
=\theta^x(1-\theta)^{1-x},
\qquad x\in\lbrace0,1\rbrace.
$$

当 $x=1$ 时，右端等于 $\theta$；当 $x=0$ 时，右端等于 $1-\theta$。

期望为

$$
\begin{aligned}
\mathbb E[X]
&=0\cdot\mathbb P(X=0)+1\cdot\mathbb P(X=1)\\\\
&=0\cdot(1-\theta)+1\cdot\theta\\\\
&=\theta.
\end{aligned}
$$

因为 $X$ 只能取 $0$ 或 $1$，所以 $X^2=X$，从而

$$
\mathbb E[X^2]=\mathbb E[X]=\theta.
$$

由方差定义得到

$$
\begin{aligned}
\operatorname{Var}(X)
&:=\mathbb E[X^2]-(\mathbb E[X])^2\\\\
&=\theta-\theta^2\\\\
&=\theta(1-\theta).
\end{aligned}
$$

## C. Gamma 函数、Beta 函数与 Beta 密度的归一化

对任意 $z>0$，Gamma 函数定义为

$$
\Gamma(z):=\int&#95;0^\infty t^{z-1}e^{-t}\,\mathrm dt.
$$

对任意 $a>0$、$b>0$，Beta 函数定义为

$$
B(a,b):=\int&#95;0^1x^{a-1}(1-x)^{b-1}\,\mathrm dx.
$$

两者满足

$$
\boxed{B(a,b)=\frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}}.
$$

后文还会用到 Gamma 函数的递推式。对定义积分做分部积分：

$$
\begin{aligned}
\Gamma(z+1)
&=\int&#95;0^\infty t^z e^{-t}\,\mathrm dt\\\\
&=\left[-t^ze^{-t}\right]&#95;0^\infty
+z\int&#95;0^\infty t^{z-1}e^{-t}\,\mathrm dt\\\\
&=z\Gamma(z).
\end{aligned}
$$

边界项等于零，并且 $\Gamma(1)=\int&#95;0^\infty e^{-t}\,\mathrm dt=1$。所以对正整数 $n$，反复使用递推式得到

$$
\Gamma(n)=(n-1)!.
$$

下面证明这条恒等式。从 Gamma 函数定义出发：

$$
\Gamma(a)\Gamma(b)
=\int&#95;0^\infty\int&#95;0^\infty
u^{a-1}v^{b-1}e^{-(u+v)}\,\mathrm du\,\mathrm dv.
$$

作变量替换

$$
r:=u+v,
\qquad
x:=\frac{u}{u+v}.
$$

反解得到

$$
u=rx,
\qquad
v=r(1-x),
$$

其中 $r\in(0,\infty)$、$x\in(0,1)$。Jacobian 行列式的绝对值为

$$
\left|
\det
\begin{pmatrix}
\partial u/\partial r & \partial u/\partial x\\\\
\partial v/\partial r & \partial v/\partial x
\end{pmatrix}
\right|
=
\left|
\det
\begin{pmatrix}
x&r\\\\
1-x&-r
\end{pmatrix}
\right|
=r.
$$

因此 $\mathrm du\,\mathrm dv=r\,\mathrm dr\,\mathrm dx$，原积分变为

$$
\begin{aligned}
\Gamma(a)\Gamma(b)
&=\int&#95;0^1\int&#95;0^\infty
(rx)^{a-1}[r(1-x)]^{b-1}e^{-r}r
\,\mathrm dr\,\mathrm dx\\\\
&=\int&#95;0^1x^{a-1}(1-x)^{b-1}\,\mathrm dx
\int&#95;0^\infty r^{a+b-1}e^{-r}\,\mathrm dr\\\\
&=B(a,b)\Gamma(a+b).
\end{aligned}
$$

两边除以 $\Gamma(a+b)$，即得所需恒等式。

Beta 分布的密度定义为

$$
f(x;a,b)
:=\frac{x^{a-1}(1-x)^{b-1}}{B(a,b)},
\qquad 0<x<1.
$$

利用 Beta 函数定义，

$$
\begin{aligned}
\int&#95;0^1f(x;a,b)\,\mathrm dx
&=\frac{1}{B(a,b)}
\int&#95;0^1x^{a-1}(1-x)^{b-1}\,\mathrm dx\\\\
&=\frac{B(a,b)}{B(a,b)}\\\\
&=1.
\end{aligned}
$$

所以它确实是一个概率密度。

## D. Beta 分布均值与方差的完整推导

先由 Gamma 函数关系得到

$$
\begin{aligned}
\frac{B(a+1,b)}{B(a,b)}
&=\frac{\Gamma(a+1)\Gamma(b)}{\Gamma(a+b+1)}
\frac{\Gamma(a+b)}{\Gamma(a)\Gamma(b)}\\\\
&=\frac{a\Gamma(a)}{(a+b)\Gamma(a+b)}
\frac{\Gamma(a+b)}{\Gamma(a)}\\\\
&=\frac{a}{a+b},
\end{aligned}
$$

其中使用了 $\Gamma(z+1)=z\Gamma(z)$。因此，若 $\theta\sim\operatorname{Beta}(a,b)$，则

$$
\begin{aligned}
\mathbb E[\theta]
&=\int&#95;0^1x f(x;a,b)\,\mathrm dx\\\\
&=\frac{1}{B(a,b)}
\int&#95;0^1x^a(1-x)^{b-1}\,\mathrm dx\\\\
&=\frac{B(a+1,b)}{B(a,b)}\\\\
&=\frac{a}{a+b}.
\end{aligned}
$$

同理，

$$
\begin{aligned}
\mathbb E[\theta^2]
&=\frac{B(a+2,b)}{B(a,b)}\\\\
&=\frac{B(a+2,b)}{B(a+1,b)}
\frac{B(a+1,b)}{B(a,b)}\\\\
&=\frac{a+1}{a+b+1}\frac{a}{a+b}\\\\
&=\frac{a(a+1)}{(a+b)(a+b+1)}.
\end{aligned}
$$

令 $s:=a+b$。方差为

$$
\begin{aligned}
\operatorname{Var}(\theta)
&=\mathbb E[\theta^2]-(\mathbb E[\theta])^2\\\\
&=\frac{a(a+1)}{s(s+1)}-\frac{a^2}{s^2}\\\\
&=\frac{a(a+1)s-a^2(s+1)}{s^2(s+1)}\\\\
&=\frac{a[as+s-a s-a]}{s^2(s+1)}\\\\
&=\frac{a(s-a)}{s^2(s+1)}\\\\
&=\frac{ab}{(a+b)^2(a+b+1)}.
\end{aligned}
$$

标准差按定义为

$$
\operatorname{sd}(\theta)
:=\sqrt{\operatorname{Var}(\theta)}.
$$

## E. Beta--Bernoulli 共轭与后验预测

设单个广告的未知点击率为 $\theta$，先验为 $\operatorname{Beta}(\alpha,\beta)$。先验密度是

$$
p(\theta)
=\frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}
{B(\alpha,\beta)}.
$$

观察 $n$ 个结果 $x&#95;1,\ldots,x&#95;n\in\lbrace0,1\rbrace$。令

$$
s:=\sum&#95;{j=1}^{n}x&#95;j,
\qquad
f:=n-s.
$$

因为给定 $\theta$ 后各次奖励独立，似然函数为

$$
\begin{aligned}
p(x&#95;1,\ldots,x&#95;n\mid\theta)
&=\prod&#95;{j=1}^{n}
\theta^{x&#95;j}(1-\theta)^{1-x&#95;j}\\\\
&=\theta^{\sum&#95;jx&#95;j}
(1-\theta)^{\sum&#95;j(1-x&#95;j)}\\\\
&=\theta^s(1-\theta)^f.
\end{aligned}
$$

Bayes 公式给出

$$
p(\theta\mid x&#95;1,\ldots,x&#95;n)
=\frac{p(x&#95;1,\ldots,x&#95;n\mid\theta)p(\theta)}
{p(x&#95;1,\ldots,x&#95;n)}.
$$

先计算分子中与 $\theta$ 有关的部分：

$$
\begin{aligned}
p(x&#95;1,\ldots,x&#95;n\mid\theta)p(\theta)
&\propto
\theta^s(1-\theta)^f
\theta^{\alpha-1}(1-\theta)^{\beta-1}\\\\
&=\theta^{\alpha+s-1}
(1-\theta)^{\beta+f-1}.
\end{aligned}
$$

为了归一化，对 $\theta\in(0,1)$ 积分：

$$
\int&#95;0^1
\theta^{\alpha+s-1}(1-\theta)^{\beta+f-1}
\,\mathrm d\theta
=B(\alpha+s,\beta+f).
$$

因此

$$
p(\theta\mid x&#95;1,\ldots,x&#95;n)
=\frac{
\theta^{\alpha+s-1}(1-\theta)^{\beta+f-1}}
{B(\alpha+s,\beta+f)},
$$

也就是

$$
\boxed{
\theta\mid x&#95;1,\ldots,x&#95;n
\sim\operatorname{Beta}(\alpha+s,\beta+f)}.
$$

下一次奖励为 $1$ 的后验预测概率是

$$
\begin{aligned}
\mathbb P(X&#95;{n+1}=1\mid x&#95;1,\ldots,x&#95;n)
&=\int&#95;0^1
\mathbb P(X&#95;{n+1}=1\mid\theta)
p(\theta\mid x&#95;1,\ldots,x&#95;n)
\,\mathrm d\theta\\\\
&=\int&#95;0^1\theta
p(\theta\mid x&#95;1,\ldots,x&#95;n)
\,\mathrm d\theta\\\\
&=\mathbb E[\theta\mid x&#95;1,\ldots,x&#95;n]\\\\
&=\frac{\alpha+s}{\alpha+\beta+n}.
\end{aligned}
$$

## F. 自适应选择为什么不破坏共轭性

设策略在历史 $H&#95;{t-1}$ 后选择动作 $a$ 的概率为

$$
\pi&#95;t(a\mid H&#95;{t-1}).
$$

历史的联合概率可以逐轮分解为

$$
p(H&#95;T\mid\theta)
=\prod&#95;{t=1}^{T}
\pi&#95;t(A&#95;t\mid H&#95;{t-1})
\theta&#95;{A&#95;t}^{X&#95;t}
(1-\theta&#95;{A&#95;t})^{1-X&#95;t}.
$$

策略项由历史和算法决定，不含未知参数 $\theta$。把它们收进只依赖历史的常数 $C(H&#95;T)$，再按照广告编号收集奖励项：

$$
\begin{aligned}
p(H&#95;T\mid\theta)
&=C(H&#95;T)
\prod&#95;{i=1}^{K}
\prod&#95;{t:A&#95;t=i}
\theta&#95;i^{X&#95;t}(1-\theta&#95;i)^{1-X&#95;t}\\\\
&=C(H&#95;T)
\prod&#95;{i=1}^{K}
\theta&#95;i^{S&#95;i(T)}
(1-\theta&#95;i)^{F&#95;i(T)}.
\end{aligned}
$$

独立先验的联合密度为

$$
p(\theta)
=\prod&#95;{i=1}^{K}
\frac{\theta&#95;i^{\alpha&#95;i-1}
(1-\theta&#95;i)^{\beta&#95;i-1}}
{B(\alpha&#95;i,\beta&#95;i)}.
$$

将似然与先验相乘，并忽略不依赖 $\theta$ 的常数：

$$
\begin{aligned}
p(\theta\mid H&#95;T)
&\propto p(H&#95;T\mid\theta)p(\theta)\\\\
&\propto\prod&#95;{i=1}^{K}
\theta&#95;i^{\alpha&#95;i+S&#95;i(T)-1}
(1-\theta&#95;i)^{\beta&#95;i+F&#95;i(T)-1}.
\end{aligned}
$$

右端仍按 $i$ 分解，所以给定历史后各臂后验仍相互独立，并且

$$
\boxed{
\theta&#95;i\mid H&#95;T
\sim\operatorname{Beta}
(\alpha&#95;i+S&#95;i(T),\beta&#95;i+F&#95;i(T))}.
$$

## G. 概率匹配的完整证明

给定历史 $H&#95;{t-1}$，真实参数向量的条件分布是后验

$$
\theta\mid H&#95;{t-1}
\sim p(\theta\mid H&#95;{t-1}).
$$

Thompson Sampling 另外生成一个条件独立的后验样本

$$
\widetilde\theta\mid H&#95;{t-1}
\sim p(\theta\mid H&#95;{t-1}).
$$

令函数 $g$ 把一个参数向量映射到其最大坐标的编号：

$$
g(z&#95;1,\ldots,z&#95;K)
\in\arg\max&#95;{1\leq i\leq K}z&#95;i.
$$

使用同一个固定规则处理平局，则

$$
A^\star=g(\theta),
\qquad
A&#95;t=g(\widetilde\theta).
$$

因为 $\theta$ 与 $\widetilde\theta$ 在给定历史后同分布，所以它们经过同一个函数 $g$ 后仍然同分布。因此

$$
\begin{aligned}
\mathbb P(A&#95;t=i\mid H&#95;{t-1})
&=\mathbb P(g(\widetilde\theta)=i\mid H&#95;{t-1})\\\\
&=\mathbb P(g(\theta)=i\mid H&#95;{t-1})\\\\
&=\mathbb P(A^\star=i\mid H&#95;{t-1}).
\end{aligned}
$$

这就证明了概率匹配。

## H. 两个 Beta 后验的比较概率

设

$$
X\sim\operatorname{Beta}(a,b),
\qquad
Y\sim\operatorname{Beta}(c,d),
$$

并且 $X$ 与 $Y$ 独立。记 $F&#95;X(y):=\mathbb P(X\leq y)$ 为 $X$ 的累积分布函数。则

$$
\begin{aligned}
\mathbb P(Y>X)
&=\int&#95;0^1\int&#95;0^y
f&#95;X(x)f&#95;Y(y)\,\mathrm dx\,\mathrm dy\\\\
&=\int&#95;0^1f&#95;Y(y)
\left[\int&#95;0^yf&#95;X(x)\,\mathrm dx\right]
\mathrm dy\\\\
&=\int&#95;0^1f&#95;Y(y)F&#95;X(y)\,\mathrm dy.
\end{aligned}
$$

定义不完全 Beta 函数

$$
B&#95;y(a,b)
:=\int&#95;0^y x^{a-1}(1-x)^{b-1}\,\mathrm dx.
$$

因为 $F&#95;X(y)=B&#95;y(a,b)/B(a,b)$，所以

$$
\boxed{
\mathbb P(Y>X)
=\frac{1}{B(a,b)B(c,d)}
\int&#95;0^1
y^{c-1}(1-y)^{d-1}B&#95;y(a,b)
\,\mathrm dy}.
$$

当 $a,b$ 是正整数时，还可以把它改写成有限和。令 $m:=a+b-1$，则

$$
\frac{B&#95;y(a,b)}{B(a,b)}
=\sum&#95;{j=a}^{m}
\binom{m}{j}y^j(1-y)^{m-j},
$$

下面证明这个有限和恒等式。令

$$
G(y):=\sum&#95;{j=a}^{m}
\binom{m}{j}y^j(1-y)^{m-j}.
$$

逐项求导：

$$
\begin{aligned}
G'(y)
&=\sum&#95;{j=a}^{m}\binom{m}{j}
\left[
j y^{j-1}(1-y)^{m-j}
-(m-j)y^j(1-y)^{m-j-1}
\right]\\\\
&=m\sum&#95;{j=a}^{m}
\binom{m-1}{j-1}y^{j-1}(1-y)^{m-j}\\\\
&\quad-m\sum&#95;{j=a}^{m-1}
\binom{m-1}{j}y^j(1-y)^{m-j-1}.
\end{aligned}
$$

第一项令 $k=j-1$，得到

$$
m\sum&#95;{k=a-1}^{m-1}
\binom{m-1}{k}y^k(1-y)^{m-1-k}.
$$

它与第二项从 $k=a$ 到 $m-1$ 的部分逐项抵消，只剩 $k=a-1$：

$$
\begin{aligned}
G'(y)
&=m\binom{m-1}{a-1}
y^{a-1}(1-y)^{m-a}\\\\
&=\frac{m!}{(a-1)!(b-1)!}
y^{a-1}(1-y)^{b-1}\\\\
&=\frac{y^{a-1}(1-y)^{b-1}}{B(a,b)}.
\end{aligned}
$$

第二行使用 $m-a=b-1$；第三行使用正整数情形下

$$
B(a,b)=\frac{(a-1)!(b-1)!}{(a+b-1)!}
=\frac{(a-1)!(b-1)!}{m!}.
$$

另一方面，$G(0)=0$，而 $B&#95;0(a,b)/B(a,b)=0$。两个函数在 $0$ 处取值相同、导数也处处相同，所以

$$
G(y)=\frac{B&#95;y(a,b)}{B(a,b)}.
$$

其中

$$
\binom{m}{j}:=\frac{m!}{j!(m-j)!}
$$

是二项式系数，$m!:=1\cdot2\cdots m$。代回积分并交换有限求和与积分：

$$
\begin{aligned}
\mathbb P(Y>X)
&=\frac{1}{B(c,d)}
\sum&#95;{j=a}^{m}\binom{m}{j}
\int&#95;0^1
y^{c+j-1}(1-y)^{d+m-j-1}\,\mathrm dy\\\\
&=\frac{1}{B(c,d)}
\sum&#95;{j=a}^{m}\binom{m}{j}
B(c+j,d+m-j).
\end{aligned}
$$

广告例子中取

$$
a=101,\quad b=901,\quad c=2,\quad d=20,
$$

所以 $m=1001$，从而

$$
\mathbb P(\widetilde\theta&#95;B>\widetilde\theta&#95;A)
=\frac{1}{B(2,20)}
\sum&#95;{j=101}^{1001}
\binom{1001}{j}B(2+j,1021-j)
\approx0.3630417.
$$

四舍五入到三位小数就是正文中的 $0.363$。

## I. 遗憾分解的逐行证明

固定真实参数 $\theta$。第 $t$ 轮选择广告 $A&#95;t$ 的伪遗憾为

$$
\theta^\star-\theta&#95;{A&#95;t}.
$$

总伪遗憾定义为

$$
R&#95;T
:=\sum&#95;{t=1}^{T}
(\theta^\star-\theta&#95;{A&#95;t}).
$$

因为每一轮恰好选择一个广告，

$$
\theta^\star-\theta&#95;{A&#95;t}
=\sum&#95;{i=1}^{K}
\mathbf 1\lbrace A&#95;t=i\rbrace
(\theta^\star-\theta&#95;i).
$$

代回总和：

$$
\begin{aligned}
R&#95;T
&=\sum&#95;{t=1}^{T}
\sum&#95;{i=1}^{K}
\mathbf 1\lbrace A&#95;t=i\rbrace
(\theta^\star-\theta&#95;i)\\\\
&=\sum&#95;{i=1}^{K}
(\theta^\star-\theta&#95;i)
\sum&#95;{t=1}^{T}
\mathbf 1\lbrace A&#95;t=i\rbrace\\\\
&=\sum&#95;{i=1}^{K}\Delta&#95;i N&#95;i(T).
\end{aligned}
$$

对算法与奖励的随机性取期望，利用有限和的线性：

$$
\begin{aligned}
\mathbb E[R&#95;T]
&=\mathbb E
\left[\sum&#95;{i=1}^{K}
\Delta&#95;iN&#95;i(T)\right]\\\\
&=\sum&#95;{i=1}^{K}
\Delta&#95;i\mathbb E[N&#95;i(T)]\\\\
&=\sum&#95;{i:\Delta&#95;i>0}
\Delta&#95;i\mathbb E[N&#95;i(T)].
\end{aligned}
$$

最后一行删除了 $\Delta&#95;i=0$ 的最优广告项，因为这些项恒等于零。

## J. Beta 后验为什么按平方根尺度收缩

设某个广告已经获得 $n$ 个观察，其中有 $s$ 次点击。后验参数为

$$
a=\alpha+s,
\qquad
b=\beta+n-s.
$$

因此

$$
a+b=\alpha+\beta+n.
$$

由 $(a-b)^2\geq0$ 得

$$
a^2-2ab+b^2\geq0.
$$

在两边加上 $4ab$：

$$
a^2+2ab+b^2\geq4ab.
$$

即

$$
(a+b)^2\geq4ab,
\qquad
ab\leq\frac{(a+b)^2}{4}.
$$

将它代入 Beta 方差公式：

$$
\begin{aligned}
\operatorname{Var}(\theta\mid H)
&=\frac{ab}{(a+b)^2(a+b+1)}\\\\
&\leq
\frac{(a+b)^2/4}{(a+b)^2(a+b+1)}\\\\
&=\frac{1}{4(a+b+1)}\\\\
&=\frac{1}{4(\alpha+\beta+n+1)}.
\end{aligned}
$$

取平方根得到

$$
\operatorname{sd}(\theta\mid H)
\leq\frac{1}{2\sqrt{\alpha+\beta+n+1}}.
$$

因此当先验参数固定而 $n$ 增大时，后验标准差至多按 $n^{-1/2}$ 的尺度缩小。

## K. 渐近遗憾公式的逻辑分解

固定一个具有唯一最优臂的 Bernoulli 环境。对任意次优臂 $i$，Lai--Robbins 下界给出

$$
\liminf&#95;{T\to\infty}
\frac{\mathbb E[N&#95;i(T)]}{\log T}
\geq
\frac{1}{\operatorname{kl}(\theta&#95;i,\theta^\star)}.
$$

$\liminf$ 表示当 $T$ 趋于无穷时所有尾部下确界的极限。Kaufmann、Korda 与 Munos 对 Bernoulli Thompson Sampling 证明了相反方向的渐近上界：

$$
\limsup&#95;{T\to\infty}
\frac{\mathbb E[N&#95;i(T)]}{\log T}
\leq
\frac{1}{\operatorname{kl}(\theta&#95;i,\theta^\star)}.
$$

$\limsup$ 是相应的尾部上确界极限。一个数列的下极限与上极限被同一个常数夹住，因此极限存在并等于该常数：

$$
\lim&#95;{T\to\infty}
\frac{\mathbb E[N&#95;i(T)]}{\log T}
=
\frac{1}{\operatorname{kl}(\theta&#95;i,\theta^\star)}.
$$

由附录 I 的遗憾分解，

$$
\frac{\mathbb E[R&#95;T]}{\log T}
=\sum&#95;{i:\Delta&#95;i>0}
\Delta&#95;i
\frac{\mathbb E[N&#95;i(T)]}{\log T}.
$$

广告数 $K$ 有限，所以极限可以逐项进入有限和：

$$
\begin{aligned}
\lim&#95;{T\to\infty}
\frac{\mathbb E[R&#95;T]}{\log T}
&=\sum&#95;{i:\Delta&#95;i>0}
\Delta&#95;i
\lim&#95;{T\to\infty}
\frac{\mathbb E[N&#95;i(T)]}{\log T}\\\\
&=\sum&#95;{i:\Delta&#95;i>0}
\frac{\Delta&#95;i}
{\operatorname{kl}(\theta&#95;i,\theta^\star)}.
\end{aligned}
$$

## L. 贝叶斯遗憾上界的完整证明

### L.1 预生成奖励序列

对每个广告 $i$，预先生成一个独立奖励序列

$$
Y&#95;{i,1},Y&#95;{i,2},\ldots
\overset{\mathrm{i.i.d.}}{\sim}
\operatorname{Bernoulli}(\theta&#95;i).
$$

$\mathrm{i.i.d.}$ 表示独立同分布。当算法第 $n$ 次选择广告 $i$ 时，就观察 $Y&#95;{i,n}$。这种构造与逐轮生成奖励具有相同分布，同时让每个广告的前 $n$ 个潜在奖励可以明确定义为

$$
\widehat\theta&#95;{i,n}
:=\frac{1}{n}\sum&#95;{s=1}^{n}Y&#95;{i,s}.
$$

### L.2 同时成立的置信事件

假设 $T\geq2$。对 $1\leq n\leq T$，定义半径

$$
r&#95;n:=\sqrt{\frac{2\log T}{n}}.
$$

定义好事件

$$
\mathcal G
:=\bigcap&#95;{i=1}^{K}
\bigcap&#95;{n=1}^{T}
\left\lbrace
|\widehat\theta&#95;{i,n}-\theta&#95;i|
\leq r&#95;n
\right\rbrace.
$$

给定 $\theta&#95;i$ 后，Hoeffding 不等式给出

$$
\begin{aligned}
\mathbb P
\left(
|\widehat\theta&#95;{i,n}-\theta&#95;i|>r&#95;n
\mid\theta&#95;i
\right)
&\leq2\exp(-2nr&#95;n^2)\\\\
&=2\exp(-4\log T)\\\\
&=2T^{-4}.
\end{aligned}
$$

对 $K$ 个广告和至多 $T$ 个样本数使用 union bound，即事件并集的概率不超过各事件概率之和：

$$
\begin{aligned}
\mathbb P(\mathcal G^c\mid\theta)
&\leq\sum&#95;{i=1}^{K}
\sum&#95;{n=1}^{T}2T^{-4}\\\\
&=2KTT^{-4}\\\\
&=\frac{2K}{T^3}.
\end{aligned}
$$

右端不依赖 $\theta$，再对先验取期望仍有

$$
\mathbb P(\mathcal G^c)\leq\frac{2K}{T^3}.
$$

### L.3 一个只用于证明的上置信值

先让每个广告各展示一次。对 $t>K$，令

$$
U&#95;t(i)
:=\min\left\lbrace
1,
\widehat\theta&#95;{i,N&#95;i(t-1)}
+r&#95;{N&#95;i(t-1)}
\right\rbrace.
$$

$U&#95;t(i)$ 只是分析工具，Thompson Sampling 本身并不使用它选择动作。给定历史 $H&#95;{t-1}$ 后，所有 $U&#95;t(i)$ 都已经确定。由概率匹配，

$$
\mathbb P(A^\star=i\mid H&#95;{t-1})
=\mathbb P(A&#95;t=i\mid H&#95;{t-1}).
$$

因此

$$
\begin{aligned}
\mathbb E[U&#95;t(A^\star)\mid H&#95;{t-1}]
&=\sum&#95;{i=1}^{K}
U&#95;t(i)\mathbb P(A^\star=i\mid H&#95;{t-1})\\\\
&=\sum&#95;{i=1}^{K}
U&#95;t(i)\mathbb P(A&#95;t=i\mid H&#95;{t-1})\\\\
&=\mathbb E[U&#95;t(A&#95;t)\mid H&#95;{t-1}].
\end{aligned}
$$

再取无条件期望：

$$
\mathbb E[U&#95;t(A^\star)]
=\mathbb E[U&#95;t(A&#95;t)].
$$

### L.4 单轮贝叶斯遗憾分解

在单轮遗憾中加上并减去两个上置信值：

$$
\begin{aligned}
\theta&#95;{A^\star}-\theta&#95;{A&#95;t}
&=\theta&#95;{A^\star}-U&#95;t(A^\star)\\\\
&\quad+U&#95;t(A^\star)-U&#95;t(A&#95;t)\\\\
&\quad+U&#95;t(A&#95;t)-\theta&#95;{A&#95;t}.
\end{aligned}
$$

取期望后，中间一项因上一节的等式而消失：

$$
\begin{aligned}
\mathbb E[
\theta&#95;{A^\star}-\theta&#95;{A&#95;t}]
&=\mathbb E[
\theta&#95;{A^\star}-U&#95;t(A^\star)]\\\\
&\quad+\mathbb E[
U&#95;t(A&#95;t)-\theta&#95;{A&#95;t}].
\end{aligned}
$$

在好事件 $\mathcal G$ 上，每个广告都满足 $\theta&#95;i\leq U&#95;t(i)$，所以第一项不大于零。第二项满足

$$
\begin{aligned}
U&#95;t(A&#95;t)-\theta&#95;{A&#95;t}
&\leq
\widehat\theta&#95;{A&#95;t,N&#95;{A&#95;t}(t-1)}
+r&#95;{N&#95;{A&#95;t}(t-1)}
-\theta&#95;{A&#95;t}\\\\
&\leq2r&#95;{N&#95;{A&#95;t}(t-1)}.
\end{aligned}
$$

在坏事件 $\mathcal G^c$ 上，因为 $\theta&#95;i,U&#95;t(i)\in[0,1]$，两个差值各自至多为 $1$。因此

$$
\mathbb E[
\theta&#95;{A^\star}-\theta&#95;{A&#95;t}]
\leq
2\,\mathbb E
\left[r&#95;{N&#95;{A&#95;t}(t-1)}\right]
+2\mathbb P(\mathcal G^c).
$$

### L.5 对所有轮次求和

初始化的前 $K$ 轮每轮遗憾至多为 $1$，总计至多 $K$。其余轮次的置信半径之和为

$$
\begin{aligned}
\sum&#95;{t=K+1}^{T}
r&#95;{N&#95;{A&#95;t}(t-1)}
&=\sqrt{2\log T}
\sum&#95;{t=K+1}^{T}
\frac{1}{\sqrt{N&#95;{A&#95;t}(t-1)}}\\\\
&\leq\sqrt{2\log T}
\sum&#95;{i=1}^{K}
\sum&#95;{n=1}^{N&#95;i(T)}\frac{1}{\sqrt n}.
\end{aligned}
$$

对任意正整数 $m$，函数 $x^{-1/2}$ 单调递减，因此

$$
\begin{aligned}
\sum&#95;{n=1}^{m}\frac{1}{\sqrt n}
&\leq1+\int&#95;1^m x^{-1/2}\,\mathrm dx\\\\
&=1+2(\sqrt m-1)\\\\
&\leq2\sqrt m.
\end{aligned}
$$

于是

$$
\sum&#95;{t=K+1}^{T}
r&#95;{N&#95;{A&#95;t}(t-1)}
\leq2\sqrt{2\log T}
\sum&#95;{i=1}^{K}\sqrt{N&#95;i(T)}.
$$

由 Cauchy--Schwarz 不等式，

$$
\begin{aligned}
\sum&#95;{i=1}^{K}\sqrt{N&#95;i(T)}
&\leq
\sqrt{
\left(\sum&#95;{i=1}^{K}1^2\right)
\left(\sum&#95;{i=1}^{K}N&#95;i(T)\right)}\\\\
&=\sqrt{KT},
\end{aligned}
$$

因为每轮恰好选择一个广告，所以 $\sum&#95;iN&#95;i(T)=T$。综上，

$$
\sum&#95;{t=K+1}^{T}
r&#95;{N&#95;{A&#95;t}(t-1)}
\leq2\sqrt{2KT\log T}.
$$

把单轮界对 $t=K+1,\ldots,T$ 求和，并代入坏事件概率：

$$
\begin{aligned}
\operatorname{BR}&#95;T
&\leq K
+2\sum&#95;{t=K+1}^{T}
\mathbb E
\left[r&#95;{N&#95;{A&#95;t}(t-1)}\right]
+2T\mathbb P(\mathcal G^c)\\\\
&\leq K
+4\sqrt{2KT\log T}
+2T\frac{2K}{T^3}\\\\
&=K+4\sqrt{2KT\log T}+\frac{4K}{T^2}.
\end{aligned}
$$

这就证明了正文中的贝叶斯遗憾上界。

## M. 贝叶斯遗憾证明中使用的三个基础不等式

### M.1 Union bound

对任意事件 $E&#95;1,\ldots,E&#95;m$，指标变量满足逐点不等式

$$
\mathbf 1\left\lbrace
\bigcup&#95;{j=1}^{m}E&#95;j
\right\rbrace
\leq\sum&#95;{j=1}^{m}\mathbf 1\lbrace E&#95;j\rbrace.
$$

这是因为左端等于 $1$ 时，至少有一个 $E&#95;j$ 发生，右端至少为 $1$；左端等于 $0$ 时不等式自然成立。两边取期望，并使用 $\mathbb E[\mathbf 1\lbrace E\rbrace]=\mathbb P(E)$，得到

$$
\mathbb P\left(\bigcup&#95;{j=1}^{m}E&#95;j\right)
\leq\sum&#95;{j=1}^{m}\mathbb P(E&#95;j).
$$

### M.2 Hoeffding 引理与 Bernoulli 均值的尾界

先证明 Hoeffding 引理。若随机变量 $Z$ 满足 $\mathbb E[Z]=0$ 且 $Z\in[a,b]$，那么对任意实数 $\lambda$，

$$
\mathbb E[e^{\lambda Z}]
\leq\exp\left(\frac{\lambda^2(b-a)^2}{8}\right).
$$

令

$$
\psi(\lambda):=\log\mathbb E[e^{\lambda Z}].
$$

定义指数倾斜后的期望

$$
\mathbb E&#95;\lambda[g(Z)]
:=\frac{\mathbb E[g(Z)e^{\lambda Z}]}
{\mathbb E[e^{\lambda Z}]}.
$$

对 $\psi$ 求导：

$$
\psi'(\lambda)
=\frac{\mathbb E[Ze^{\lambda Z}]}
{\mathbb E[e^{\lambda Z}]}
=\mathbb E&#95;\lambda[Z].
$$

再次求导：

$$
\begin{aligned}
\psi''(\lambda)
&=\frac{\mathbb E[Z^2e^{\lambda Z}]}
{\mathbb E[e^{\lambda Z}]}
-\left(
\frac{\mathbb E[Ze^{\lambda Z}]}
{\mathbb E[e^{\lambda Z}]}
\right)^2\\\\
&=\operatorname{Var}&#95;\lambda(Z).
\end{aligned}
$$

任何取值位于 $[a,b]$ 的随机变量 $W$ 都满足

$$
\operatorname{Var}(W)\leq\frac{(b-a)^2}{4}.
$$

证明如下。令中点 $c:=(a+b)/2$，则 $|W-c|\leq(b-a)/2$。又因为均值使平方损失最小，

$$
\begin{aligned}
\operatorname{Var}(W)
&=\mathbb E[(W-\mathbb E[W])^2]\\\\
&\leq\mathbb E[(W-c)^2]\\\\
&\leq\frac{(b-a)^2}{4}.
\end{aligned}
$$

指数倾斜不会改变 $Z\in[a,b]$ 这一取值范围，所以

$$
\psi''(\lambda)\leq\frac{(b-a)^2}{4}.
$$

因为 $\psi(0)=0$ 且 $\psi'(0)=\mathbb E[Z]=0$，两次积分得到

$$
\begin{aligned}
\psi(\lambda)
&=\int&#95;0^\lambda(\lambda-s)\psi''(s)\,\mathrm ds\\\\
&\leq\frac{(b-a)^2}{4}
\int&#95;0^\lambda(\lambda-s)\,\mathrm ds\\\\
&=\frac{\lambda^2(b-a)^2}{8}.
\end{aligned}
$$

对 $\lambda<0$，把积分方向相应反转可得同一结论。两边取指数，Hoeffding 引理得证。

现在令 $Y&#95;1,\ldots,Y&#95;n$ 为相互独立的 Bernoulli 随机变量，均值均为 $\theta$，并定义

$$
\widehat\theta&#95;n:=\frac1n\sum&#95;{s=1}^{n}Y&#95;s.
$$

对任意 $\varepsilon>0$、$\lambda>0$，Markov 不等式给出

$$
\begin{aligned}
\mathbb P(\widehat\theta&#95;n-\theta\geq\varepsilon)
&=\mathbb P\left(
e^{\lambda\sum&#95;{s=1}^{n}(Y&#95;s-\theta)}
\geq e^{\lambda n\varepsilon}
\right)\\\\
&\leq e^{-\lambda n\varepsilon}
\mathbb E\left[
e^{\lambda\sum&#95;{s=1}^{n}(Y&#95;s-\theta)}
\right].
\end{aligned}
$$

这里使用的 Markov 不等式可以直接证明。对任意非负随机变量 $W$ 和常数 $c>0$，逐点有

$$
W\geq c\,\mathbf 1\lbrace W\geq c\rbrace.
$$

两边取期望：

$$
\mathbb E[W]
\geq c\,\mathbb P(W\geq c).
$$

因此

$$
\mathbb P(W\geq c)
\leq\frac{\mathbb E[W]}{c}.
$$

上面的尾界中取

$$
W=e^{\lambda\sum&#95;{s=1}^{n}(Y&#95;s-\theta)},
\qquad
c=e^{\lambda n\varepsilon},
$$

就得到所写的那一步。

独立性把矩母函数分解为乘积，Hoeffding 引理对 $Y&#95;s-\theta\in[-\theta,1-\theta]$ 给出

$$
\begin{aligned}
\mathbb E\left[
e^{\lambda\sum&#95;{s=1}^{n}(Y&#95;s-\theta)}
\right]
&=\prod&#95;{s=1}^{n}
\mathbb E[e^{\lambda(Y&#95;s-\theta)}]\\\\
&\leq\prod&#95;{s=1}^{n}e^{\lambda^2/8}\\\\
&=e^{n\lambda^2/8}.
\end{aligned}
$$

因此

$$
\mathbb P(\widehat\theta&#95;n-\theta\geq\varepsilon)
\leq\exp\left(-\lambda n\varepsilon+rac{n\lambda^2}{8}\right).
$$

右端指数关于 $\lambda$ 的导数为

$$
-n\varepsilon+\frac{n\lambda}{4}.
$$

令其等于零，得到最优选择 $\lambda=4\varepsilon$。代回：

$$
\mathbb P(\widehat\theta&#95;n-\theta\geq\varepsilon)
\leq e^{-2n\varepsilon^2}.
$$

对 $\theta-\widehat\theta&#95;n$ 使用同样论证：

$$
\mathbb P(\theta-\widehat\theta&#95;n\geq\varepsilon)
\leq e^{-2n\varepsilon^2}.
$$

最后使用 union bound：

$$
\boxed{
\mathbb P(
|\widehat\theta&#95;n-\theta|\geq\varepsilon)
\leq2e^{-2n\varepsilon^2}}.
$$

这就是附录 L 使用的 Bernoulli Hoeffding 不等式。

### M.3 Cauchy--Schwarz 不等式

对任意实数 $x&#95;1,\ldots,x&#95;K$ 和 $y&#95;1,\ldots,y&#95;K$，定义

$$
S:=\sum&#95;{i=1}^{K}x&#95;i^2,
\qquad
P:=\sum&#95;{i=1}^{K}x&#95;iy&#95;i.
$$

若 $S=0$，则所有 $x&#95;i=0$，不等式显然成立。若 $S>0$，对任意实数 $c$，

$$
0\leq\sum&#95;{i=1}^{K}(y&#95;i-cx&#95;i)^2
=\sum&#95;{i=1}^{K}y&#95;i^2-2cP+c^2S.
$$

取 $c=P/S$：

$$
0\leq\sum&#95;{i=1}^{K}y&#95;i^2-\frac{P^2}{S}.
$$

整理得到

$$
\left(\sum&#95;{i=1}^{K}x&#95;iy&#95;i\right)^2
\leq
\left(\sum&#95;{i=1}^{K}x&#95;i^2\right)
\left(\sum&#95;{i=1}^{K}y&#95;i^2\right).
$$

在附录 L 中取 $x&#95;i=1$、$y&#95;i=\sqrt{N&#95;i(T)}$，便得到

$$
\sum&#95;{i=1}^{K}\sqrt{N&#95;i(T)}
\leq\sqrt{KT}.
$$
