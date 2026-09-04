---
title: "后验为什么会把组合探索卡住：多项式 Thompson Sampling 与错配采样悖论"
date: 2026-09-04 16:30:00
categories:
  - 机器学习理论
tags:
  - 多臂老虎机
  - 组合 Bandit
  - Thompson Sampling
  - 后悔分析
  - Bayesian Learning
  - 在线学习
  - 数学证明
mathjax: true
toc: true
toc_number: false
comments: true
---

这篇文章解读 Raymond Zhang 与 Richard Combes 的论文 [*Thompson Sampling For Combinatorial Bandits: Polynomial Regret and Mismatched Sampling Paradox*](https://arxiv.org/abs/2410.05441)。论文发表于 NeurIPS 2024，站内也保存了 [论文 PDF](/files/papers/bandit/zhang-combinatorial-ts-2024.pdf)。

论文要解决的不是“Thompson Sampling 是否会探索”这样宽泛的问题，而是一个很具体的高维困难：**一次动作由许多基础臂组成时，逐坐标独立抽取后验，可能让最优组合的出现概率随动作大小指数衰减。** 论文提出带 Gaussian exploration boost 的 BG-CTS，把已知的指数级瞬态遗憾降为多项式依赖；更出人意料的是，在一个 Bernoulli 环境中，使用正确 Bernoulli 后验的算法反而可能比使用 Gaussian 错配后验的算法差很多。

这篇工作最值得追踪的地方，不是“Gaussian 一定比 Beta 好”，而是它暴露出一个更基础的结构问题：**后验的边缘分布决定单个坐标怎样摇摆，坐标之间的联合采样方式决定一个组合能否整体显得乐观。** 论文用 BG-CTS 研究了这种联合采样结构，但还没有给出完整刻画。后文将区分论文已经证明的结果与尚未解决的问题。

<!--more-->

## 1. 研究目的：把组合探索中的指数障碍拆开

### 1.1 从单臂到组合动作

设有 $d$ 个基础臂，编号集合写成

$$
[d]:=\{1,2,\ldots,d\}.
$$

算法每轮选择一个二进制向量

$$
A_t\in\mathcal A\subseteq\{0,1\}^d.
$$

若 $(A_t)_i=1$，表示第 $t$ 轮使用了基础臂 $i$；若 $(A_t)_i=0$，表示没有使用它。集合 $\mathcal A$ 是可行动作族。一个动作可以是一个广告集合、一条网络路径中的边集合，也可以是一个匹配或一个调度方案。

记

$$
m:=\max_{A\in\mathcal A}\|A\|_1,
$$

其中 $\lVert A\rVert_1=\sum_{i=1}^d A_i$ 是动作中包含的基础臂数量，$m$ 是最大动作大小。论文中的困难正随着 $m$ 增大而出现：一次决策需要同时判断越来越多的坐标。

第 $t$ 轮环境生成随机向量 $X(t)=(X_1(t),\ldots,X_d(t))$。算法选择 $A_t$ 后，只看到被选择坐标的结果

$$
Y(t)=A_t\odot X(t),
$$

其中 $\odot$ 表示逐坐标相乘。因此 $Y_i(t)=X_i(t)$ 当 $(A_t)_i=1$，而未选择的坐标在本轮没有观测。这个反馈叫 **semi-bandit feedback（半 Bandit 反馈）**：我们知道所选组合里的每一项，而不是只知道组合总和。

论文先研究线性组合收益

$$
r(A,\mu^\star)=A^\top\mu^\star,
$$

其中 $\mu^\star=\mathbb E[X(t)]$ 是未知均值向量，$A^\top\mu^\star=\sum_i A_i\mu_i^\star$。假设每个坐标的噪声是独立的 $\sigma^2$-次高斯随机变量。这个假设的含义是：对任意 $\lambda\in\mathbb R^d$，

$$
\mathbb E\exp\bigl(\lambda^\top(X(t)-\mu^\star)\bigr)
\leq
\exp\left(\frac{\sigma^2\|\lambda\|_2^2}{2}\right).
$$

它允许 Bernoulli、Gaussian 和一般有界噪声，核心是尾部不会比 Gaussian 更重。

### 1.2 遗憾在这里衡量什么

令唯一的最优动作为

$$
A^\star\in\operatorname*{arg\max}_{A\in\mathcal A}A^\top\mu^\star.
$$

每个动作的差距是

$$
\Delta_A:=A^{\star\top}\mu^\star-A^\top\mu^\star\geq0.
$$

到时域 $T$ 的期望伪遗憾为

$$
R(T,\mu^\star)
:=
\mathbb E\left[\sum_{t=1}^T\Delta_{A_t}\right].
$$

它不把奖励本身的随机波动算成算法错误，只计算“选择了次优组合”造成的期望损失。

这篇论文的目标可以准确地写成：在动作大小 $m$ 很大时，能否让一个每轮只调用一次线性 oracle 的 Thompson Sampling 算法，仍然具有多项式而不是指数级的有限时间遗憾？这里的 oracle 接收一个参数向量 $\theta$，返回

$$
\operatorname{ORACLE}(\theta)
\in
\operatorname*{arg\max}_{A\in\mathcal A}A^\top\theta.
$$

## 2. 旧瓶颈：独立后验如何制造指数小概率

### 2.1 两条互斥路径

Zhang 与 Combes 在 2021 年的论文 [*On the Suboptimality of Thompson Sampling in High Dimensions*](https://arxiv.org/abs/2102.05502) 构造了一个极简反例，站内保存了 [论文 PDF](/files/papers/bandit/zhang-combinatorial-ts-high-dimensions-2021.pdf)。把前 $m$ 个基础臂看作左侧路径，后 $m$ 个基础臂看作右侧路径。先记

$$
\mathbf 1_m=(1,\ldots,1),
\qquad
\mathbf 0_m=(0,\ldots,0),
$$

其中 $\mathbf 1_m$ 是长度为 $m$ 的全 1 向量，$\mathbf 0_m$ 是长度为 $m$ 的全 0 向量。两条动作便可简写成

$$
A^\star=(\mathbf 1_m,\mathbf 0_m),
\qquad
A^-=(\mathbf 0_m,\mathbf 1_m).
$$

这里的括号表示把两个长度为 $m$ 的向量首尾拼接，而不是做乘法。上标 $\star$ 表示“最优”，上标 $-$ 表示“次优”。因此 $A^\star$ 选择左侧的 $m$ 个臂，$A^-$ 选择右侧的 $m$ 个臂。

令最优路径上的均值为 $1$，次优路径上的均值为 $1-\Delta/m$。按照前文的收益记号，两条路径的总期望收益差为

$$
r(A^\star,\mu^\star)-r(A^-,\mu^\star)=\Delta.
$$

差距 $\Delta$ 可以固定为常数，但每条路径包含的坐标数 $m$ 可以不断增大。

### 2.2 B-CTS 的每个坐标都没有做错

在 Bernoulli 环境中，均值 $\mu_i$ 的均匀先验是 $\operatorname{Beta}(1,1)$。若到当前为止，第 $i$ 个臂得到过 $s_i$ 次成功、$f_i$ 次失败，那么后验是

$$
\mu_i\mid H_t
\sim
\operatorname{Beta}(1+s_i,1+f_i).
$$

这就是 B-CTS：每个坐标独立抽一个后验样本 $V_i(t)$，然后调用 oracle 选择 $V(t)$ 下收益最大的组合。

如果算法一直只选择 $A^-$，那么 $A^\star$ 中的坐标从未被观测，仍然服从均匀分布；$A^-$ 中的坐标则越来越集中在 $1-\Delta/m$ 附近。于是选择最优路径的事件是

$$
\sum_{i=1}^{m}V_i(t)
\geq
\sum_{j=m+1}^{2m}V_j(t).
$$

注意：单个最优坐标的后验样本并不悲观。问题发生在求和以后。左边是 $m$ 个近似均匀的样本之和，右边是 $m$ 个集中在略高于 $1/2$ 的样本之和。当 $m$ 增大时，两个和的波动规模只有 $O(\sqrt m)$，而均值差是 $\Delta$；把一个常数级的总差距分摊到 $m$ 个坐标后，联合比较就会变成一个高维集中事件。

### 2.3 把概率算出来

在“尚未选过 $A^\star$”的历史下，令

$$
L_t:=\sum_{i=1}^{m}V_i(t),
\qquad
R_t:=\sum_{j=m+1}^{2m}V_j(t).
$$

由后验均值可得，当 $t$ 足够大时，近似有

$$
\mathbb E[L_t\mid H_t]=\frac m2,
\qquad
\mathbb E[R_t\mid H_t]\approx m-\Delta.
$$

为了让 $L_t\geq R_t$，左边必须比自己的均值高，或者右边必须比自己的均值低。由于所有 $V_i(t)$ 都位于 $[0,1]$，Hoeffding 不等式给出一个形如

$$
\mathbb P(L_t\geq R_t\mid H_t)
\leq
\exp(-c m)
$$

的上界，其中 $c>0$ 与 $\Delta$ 和参数范围有关，但不随 $m$ 增大而消失。于是首次选到最优路径的等待时间满足

$$
\mathbb E[\tau_\star]
\gtrsim
\exp(c m),
\qquad
\tau_\star:=\inf\{t:A_t=A^\star\}.
$$

这不是“某个常数写得不够紧”。在这个历史上，算法的每一个坐标后验都按照 Bayes 规则更新，真正失败的是：**独立样本的坐标波动无法协同地把整条路径抬起来。**

### 2.4 固定的强制探索也没有从根上解决问题

如果先强制探索 $\ell$ 轮，再切换到 B-CTS，每个坐标都会获得一些初始样本。2021 年的结果说明，对任意固定的 $\ell$，都可以选择足够小的 $\Delta$，让后验在切换后仍然形成上面的高维集中，因而遗憾依旧呈指数增长。

强制探索只改变了起点，没改变后续联合采样的几何结构。2024 年论文正是在这一背景下考察 Gaussian boost 对组合采样概率的影响。

## 3. Zhang–Combes 2024：先把三个算法分清楚

### 3.1 B-CTS：正确模型上的自然后验

B-CTS 假设奖励确实是 Bernoulli，使用 Beta 先验和 Bernoulli likelihood。它的优点是统计模型匹配，后验更新简单；它的困难正是上节的高维联合集中。

### 3.2 G-CTS：故意使用 Gaussian 后验

令 $N_i(t-1)$ 是第 $t$ 轮开始前臂 $i$ 的观测次数，$\widehat\mu_i(t-1)$ 是经验均值。把所有臂至少初始化一次后，论文使用不适定均匀先验和 Gaussian likelihood，得到

$$
\theta_i(t)
\sim
\mathcal N\left(
\widehat\mu_i(t-1),
\frac{\sigma^2}{N_i(t-1)}
\right),
$$

并独立抽取各个坐标。

这里的 Gaussian 不是说作者认为 Bernoulli 奖励真的服从 Gaussian。它是一种探索分布：让采样值可以越过 $[0,1]$，并让上尾按照可计算的 Gaussian 尾概率衰减。

### 3.3 BG-CTS：给 Gaussian 方差加一个 boost

论文真正证明的是 BG-CTS。它令

$$
\theta(t)
=
\widehat\mu(t-1)
+
\sigma\sqrt{2g(t)}\,V_{t-1}^{1/2}Z(t),
$$

其中

$$
V_{t-1}:=\operatorname{diag}\left(
\frac1{N_1(t-1)},\ldots,
\frac1{N_d(t-1)}
\right),
\qquad
Z(t)\sim\mathcal N(0,I_d).
$$

$I_d$ 是 $d\times d$ 单位矩阵，$V_{t-1}^{1/2}$ 是逐坐标取平方根。于是第 $i$ 个坐标的条件方差是 $2g(t)\sigma^2/N_i(t-1)$。

论文取

$$
g(t)=\frac{f(t)}{\log t},
$$

$$
f(t)=(1+\lambda)
\left[
\log t+(m+2)\log\log t
+\frac m2\log\left(1+e^\lambda\right)
\right],
$$

其中 $\lambda>0$ 是探索参数。$g(t)$ 在有限时间内略大于 $1$，而当 $t$ 增大时趋近于 $1+\lambda$。这个表达式来自后面要控制的两个事件：经验均值不能偏得太远，以及最优动作的 Gaussian 样本要在足够多的轮次里高于真实均值。

### 3.4 三种算法其实共享同一套流程

前面三个公式容易让人误以为这是三套完全不同的算法。实际上，论文中的差别只发生在“怎样生成参数样本”这一步，后面的动作选择、反馈和更新完全相同。共同流程可以直接写成五步：

1. **初始化。**令每个基础臂的观测次数为 $N_i=0$，经验均值为 $\widehat\mu_i=0$。论文先让每个臂至少被观测一次，之后进入 Thompson Sampling 循环。
2. **生成样本。**在第 $t$ 轮，根据当前历史为每个坐标生成 $\theta_i(t)$。B-CTS、G-CTS 和 BG-CTS 只在这一步使用不同的分布。
3. **调用 oracle。**把样本向量 $\theta(t)$ 交给线性 oracle，选择 $A_t\in\operatorname*{arg\max}_{A\in\mathcal A}A^\top\theta(t)$。
4. **接收反馈。**只观察被选坐标的奖励 $X_i(t)$，其中 $i\in A_t$；没有被选中的坐标在这一轮没有观测。
5. **更新统计量。**对每个 $i\in A_t$，令 $N_i\leftarrow N_i+1$，并用新观测更新 $\widehat\mu_i$。下一轮重新从更新后的后验或探索分布中采样。

三种算法在第 2 步的具体分布如下。表中的 $S_i$ 和 $F_i$ 分别是坐标 $i$ 到当前为止的成功、失败次数；$N_i=S_i+F_i$。

<table class="algorithm-compare" aria-label="三种组合 Thompson Sampling 的采样分布对照">
  <thead>
    <tr>
      <th>算法</th>
      <th>第 $t$ 轮的坐标样本</th>
      <th>论文中扮演的角色</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>B-CTS</strong></td>
      <td>$\theta_i(t)\sim\operatorname{Beta}(1+S_i,1+F_i)$</td>
      <td>奖励模型为 Bernoulli 时的匹配后验。</td>
    </tr>
    <tr>
      <td><strong>G-CTS</strong></td>
      <td>$\theta_i(t)\sim\mathcal N(\widehat\mu_i,\sigma^2/N_i)$</td>
      <td>用 Gaussian 分布产生不受 $[0,1]$ 截断的探索样本。</td>
    </tr>
    <tr>
      <td><strong>BG-CTS</strong></td>
      <td>$\theta_i(t)\sim\mathcal N(\widehat\mu_i,2g(t)\sigma^2/N_i)$</td>
      <td>在 G-CTS 的方差上加入论文分析所需的 boost。</td>
    </tr>
  </tbody>
</table>

因此，论文比较的不是三种不同的 oracle，也不是三种不同的反馈模式，而是同一条“采样—优化—观测—更新”链条中，采样分布如何改变组合动作被选中的概率。

## 4. 论文的核心证明：先保证“干净运行”

BG-CTS 的证明不直接计算每一轮选中最优动作的概率。作者先构造一个高概率事件，在这个事件上证明算法会逐渐获得更多最优动作样本。

### 4.1 把 Thompson 样本拆成两种误差

写

$$
\theta(t)=\widehat\mu(t-1)
+\sigma\sqrt{2g(t)}V_{t-1}^{1/2}Z(t).
$$

对最优动作取内积，并加减 $\mu^\star$，得到

$$
\begin{aligned}
A^{\star\top}\theta(t)-A^{\star\top}\mu^\star
&=A^{\star\top}(\widehat\mu(t-1)-\mu^\star)\\
&\quad+\sigma\sqrt{2g(t)}A^{\star\top}V_{t-1}^{1/2}Z(t).
\end{aligned}
$$

再除以方向上的标准差

$$
\sigma\sqrt{A^{\star\top}V_{t-1}A^\star},
$$

定义

$$
U^\star(t):=
\frac{A^{\star\top}(\widehat\mu(t-1)-\mu^\star)}
{\sigma\sqrt{A^{\star\top}V_{t-1}A^\star}},
$$

$$
S^\star(t):=
\frac{\sigma\sqrt{2g(t)}A^{\star\top}V_{t-1}^{1/2}Z(t)}
{\sigma\sqrt{A^{\star\top}V_{t-1}A^\star}}.
$$

因此

$$
A^{\star\top}\theta(t)-A^{\star\top}\mu^\star
=
\bigl(U^\star(t)+S^\star(t)\bigr)
\sigma\sqrt{A^{\star\top}V_{t-1}A^\star}.
$$

$U^\star(t)$ 是历史数据造成的经验均值误差，$S^\star(t)$ 是本轮 Thompson 随机化造成的偏移。两者必须同时被控制：只控制经验均值，无法保证本轮样本真的乐观；只增加随机化，又会让次优动作获得过大的样本。

### 4.2 Clean run 的两个条件

论文把到时刻 $t$ 为止的运行称为 clean run，要求两个条件同时成立。

第一，对所有历史时刻 $s\leq t$ 和所有动作 $A\in\mathcal A$，样本偏离真实均值不能超过一个置信 bonus：

$$
\left|A^\top\theta(s)-A^\top\mu^\star\right|
\leq
C_1\sigma\sqrt m\log s\,\sqrt{A^\top V_{s-1}A}.
$$

第二，直到时刻 $t$，至少有 $C_2t^\alpha$ 个时刻满足

$$
A^{\star\top}\theta(s)\geq A^{\star\top}\mu^\star.
$$

这里 $C_1,C_2>0$ 是常数，论文取 $\alpha\approx0.131$。第二个条件看起来奇怪，但它是证明的关键：如果最优动作的样本经常高于真实值，那么 oracle 就不能无限期地只选择次优动作。

### 4.3 四个坏事件分别从哪里来

为了证明 clean run 高概率成立，作者把失败拆成四类。

**事件 B：经验均值偏离太远。** 对某个坐标 $i$ 和某个历史时刻 $s$，有

$$
\sqrt{N_i(s)}\,\lvert\widehat\mu_i(s)-\mu_i^\star\rvert > \sigma\sqrt{8\log t}.
$$

次高斯集中不等式和两次 union bound 给出 $\mathbb P(B_t)\lesssim d/t^2$。

**事件 C：Gaussian 随机化异常大。** 若某个 $Z_i(s)$ 超过 $\sqrt{6\log t}$，就可能破坏所有动作的 bonus 控制。Gaussian 尾界给出同样的 $O(d/t^2)$ 级别。

**事件 D：最优动作的经验均值方向偏离太远。** 这是事件 B 在 $A^\star$ 方向上的专门版本，概率可以压到 $O(1/(t(\log t)^2))$。

**事件 E：最优动作的乐观时刻太少。** 条件于过去，$S^\star(s)$ 服从方差为 $2g(s)$ 的标准化 Gaussian。因此

$$
\mathbb P\bigl(S^\star(s)\geq\sqrt{2f(t)}\mid H_s\bigr)
=Q\left(\sqrt{\frac{f(t)}{g(s)}}\right),
$$

其中 $Q(x)=\mathbb P(Z\geq x)$，$Z\sim\mathcal N(0,1)$。$g(t)$ 的设计保证这个概率虽然会随 $t$ 下降，但在前 $t$ 轮的总和仍然至少是常数倍的 $t$。于是用乘法型 Azuma--Hoeffding 界，可以得到

$$
\mathbb P(E_t)\leq\exp(-C_4t^\alpha).
$$

将四个坏事件相加，论文得到

$$
\mathbb P(\text{clean run fails by }t)
\leq
4dt^{-2}+t^{-1}(\log t)^{-2}+e^{-C_4t^\alpha}.
$$

这一步的思想是：不要求每一轮都抽到理想样本，只要异常轮次足够少，并且乐观样本累计出现得足够多，就能控制整个样本路径。

## 5. 为什么 clean run 会逼着算法重新看到最优动作

设 clean run 已经成立。考虑某一轮 $s$，如果当前动作 $A_s$ 不是最优动作，并且它的置信 bonus 已经小于最小正差距 $\Delta_{\min}$，那么第一条 clean-run 条件意味着

$$
A_s^\top\theta(s)
\leq
A_s^\top\mu^\star+\frac{\Delta_{\min}}{2}
\leq
A^{\star\top}\mu^\star-\frac{\Delta_{\min}}{2}.
$$

但第二条条件保证存在许多时刻 $s$ 使

$$
A^{\star\top}\theta(s)\geq A^{\star\top}\mu^\star.
$$

在这些时刻，oracle 不可能选择已经收敛得很紧的次优动作。因此，所有“最优动作样本乐观”但“当前动作 bonus 仍然很小”的时刻，都必须贡献给 $A^\star$ 的选择次数。

另一方面，一个坐标 $i$ 的 bonus 仍大于 $\Delta_{\min}/m$，意味着

$$
C_1\sigma\sqrt m\log t\frac1{\sqrt{N_i(s)}}
\gtrsim
\frac{\Delta_{\min}}m,
$$

从而

$$
N_i(s)
\lesssim
\frac{\sigma^2m^3(\log t)^2}{\Delta_{\min}^2}.
$$

每个坐标只有多项式数量的轮次可以处于“大 bonus”状态。除此以外，clean run 中最优动作乐观的时刻必须不断选择 $A^\star$。论文由此得到

$$
M_{A^\star}(t)\geq C_6t^\alpha,
$$

其中 $M_{A^\star}(t)$ 是前 $t$ 轮选择最优动作的次数。

这一步修复了 2021 年证明中的关键断裂。旧分析只知道“如果选到最优动作会怎样”，却没有证明最优动作会在足够多的时刻重新出现；BG-CTS 先用 Gaussian 上尾建立乐观时刻，再把这些时刻转成真实的最优动作样本。

由于 $A^\star$ 已经被选择了 $C_6t^\alpha$ 次，最优动作方向上的方差满足

$$
A^{\star\top}V_{t-1}A^\star
\leq
\frac{m^2}{C_6t^\alpha}.
$$

于是最优动作的 Thompson 样本与真实均值之间的差距至多为

$$
h(t)
\asymp
\sigma m\sqrt{\frac{m\log t}{C_6t^\alpha}},
$$

并且 $h(t)\to0$。当 $h(t)<\Delta_{\min}/4$ 后，只要当前动作的样本没有异常偏高，oracle 就只能选最优动作。

## 6. 论文的遗憾上界从哪里来

论文把次优选择事件进一步拆成四部分。令

$$
Z_t:=\{\Delta_{A_t}>0\}
$$

表示第 $t$ 轮选了次优动作。

### 6.1 不干净运行造成的遗憾

clean run 失败的概率可求和：

$$
\sum_{t=1}^{T}\Delta_{\max}\,\mathbb P(\text{clean run fails by }t)
$$

是有限的，贡献只进入与 $T$ 无关的常数项。这里 $\Delta_{\max}$ 是最大动作差距。

### 6.2 经验均值估计错误

如果选中的动作包含某个坐标 $i$，且 $\widehat\mu_i$ 偏离 $\mu_i^\star$ 超过 $\Delta_{\min}/(4m)$，就把这次选择记入事件 F。对每个坐标使用次高斯尾界并按观测次数求和，得到

$$
\mathbb E\left[\sum_{t=1}^{T}\mathbf 1\{F_t\}\right]
\lesssim
\frac{d m^2\sigma^2}{\Delta_{\min}^2}.
$$

这反映了“先把每个被选坐标估计准确”所需的样本成本。

### 6.3 Thompson 随机化异常大

若 Gaussian 随机项超过

$$
\sigma\sqrt{8\widetilde f(t)A_t^\top V_{t-1}A_t},
$$

记为事件 H。对 Gaussian 尾部做 union bound 后，$\mathbb P(H_t)\leq t^{-2}$，因此这一部分只贡献常数级遗憾。

### 6.4 一切都正常，但仍然选了次优动作

剩下的情况是：经验均值正常、Thompson 随机项正常、clean run 成立，但动作仍次优。此时必须有

$$
\Delta_{A_t}
\lesssim
\sigma\sqrt{\widetilde f(t)A_t^\top V_{t-1}A_t}.
$$

平方后，对每个坐标累计，使用组合半 Bandit 的椭圆势能型求和，可以得到论文主导项

$$
R(T,\mu^\star)
\leq
C\frac{\sigma^2d\log m}{\Delta_{\min}}\log T
+C_1\frac{\sigma^2d^2m\log m}{\Delta_{\min}}\log\log T
+P\left(m,d,\frac1{\Delta_{\min}},\Delta_{\max},\sigma\right).
$$

这里 $C,C_1$ 是普适常数，$P$ 是关于所列参数的多项式。论文明确给出了多项式的构造，但它的次数很高：不同参数方向上的次数最高可达约 30。这意味着论文证明了“不是指数级”，却还没有把有限时间常数压到一个可以直接拿来部署或比较的程度。

## 7. 错配采样悖论究竟悖论在哪里

在实验中，环境仍然使用 Bernoulli 奖励，最优路径每个坐标的均值为 $0.9$，次优路径每个坐标的均值为 $0.7$。B-CTS 使用正确的 Bernoulli likelihood 和 Beta 后验；BG-CTS 却使用 Gaussian likelihood。

结果是：当 $m$ 较大、时域不是极其长时，B-CTS 的遗憾几乎线性增长，BG-CTS 的遗憾保持在低得多的量级。

这件事不能简单解释成“Gaussian 比 Bernoulli 更好”。Gaussian 的作用有两个。

第一，它的样本没有被截断在 $[0,1]$ 内。即使经验均值已经很集中，也仍然存在可计算的上尾，让整个最优组合有机会同时变得乐观。

第二，BG-CTS 在有限时间内把方差乘以 $2g(t)$。这个 boost 使最优动作在足够多的轮次里跨过真实均值，从而触发第 5 节的自我修复机制。

所以“错配”真正暴露的是：**统计推断中的模型正确，不等于控制问题中的探索几何正确。** 后验是用来描述参数不确定性的，但 Thompson Sampling 还要把这个不确定性转换成动作；在组合动作中，转换过程本身就是算法设计的一部分。

## 8. 论文已经完成了什么，哪些还没有完成

下面把论文的结论和没有解决的问题分开。前四项是论文已经证明的内容，后面的部分只列出论文尚未回答的技术问题。

### 8.1 已经完成的结果

1. 在独立、次高斯坐标和线性组合收益下，BG-CTS 的有限时间遗憾不再含有关于动作大小 $m$ 的指数项。
2. clean-run 分析证明：最优动作会以 $\Omega(t^\alpha)$ 的次数被重新采样，最优动作方向上的不确定性因此逐渐收缩。
3. 在同一个 Bernoulli 环境中，正确的 Beta 后验可能比 Gaussian 错配后验产生更差的高维瞬态遗憾。
4. 算法每轮只需要一次线性 maximization oracle；只要离线线性优化可高效完成，计算复杂度仍是可控的。

### 8.2 未完成工作一：多项式到底能不能降下来

定理中的 $P(m,d,1/\Delta_{\min},\Delta_{\max},\sigma)$ 只被写成“某个多项式”。证明里它来自三个地方：clean run 首次成立前的等待时间、最优动作达到 $C_6t^\alpha$ 次采样所需的时间，以及 $h(t)<\Delta_{\min}/4$ 的阈值。

这三个地方都使用了比较粗的 union bound 和最坏情形计数。因此，定理中的 $m^3$、$d^2$ 以及高次 $1/\Delta_{\min}$ 因子，是否只是证明技术带来的松弛，仍然没有答案。

### 8.3 未完成工作二：BG-CTS 的 gap-free 或 minimax 遗憾

论文的主定理依赖 $\Delta_{\min}>0$，是 instance-dependent bound。它没有给出一个对所有差距都成立的 minimax 界，也没有证明 BG-CTS 达到组合半 Bandit 的已知最优量级。

因此，在同样的线性组合模型下，BG-CTS 是否也具有

$$
R_T=\widetilde O\bigl(\sigma\sqrt{d m T}\bigr)
$$

或更精确的动作族相关 minimax 界，仍是未解决的问题。也就是说，“去掉指数项”和“达到 minimax 最优”在这篇论文中并不是同一个已经完成的结论。

### 8.4 未完成工作三：Gaussian 是必要条件，还是只是一个代表

2024 年的论文证明了 Gaussian boost 有效，但没有给出“什么样的后验探索尾部足够”的充要条件。尚未回答的问题包括：

1. 每个坐标样本的边缘分布可以在线采样；
2. 在标准化尺度 $O(1)$ 上，上尾不会随 $m$ 额外塌缩；
3. 最优动作方向的乐观事件能在多项式数量的轮次中重复出现；
4. 次优动作的异常上偏仍可由可求和的尾界控制。

论文没有证明这四点是否足够，也没有给出满足其中部分条件却仍然指数失败的反例。

### 8.5 未完成工作四：独立坐标假设能否去掉

B-CTS 的后验按坐标分解，论文的集中分析也依赖坐标独立。现实中的组合动作经常存在相关性：同一条线路上的边有共同拥堵因素，同一组广告有共同用户偏好。

相关坐标会同时带来两种效果：一次观测可能提供更多信息，也可能使多个误差方向一起偏离。论文没有回答在这种相关结构下，额外信息与共同偏移哪一个占主导，也没有说明指数小概率是否仍会出现。

### 8.6 未完成工作五：近似 oracle 的代价如何进入定理

2018 年 Wang 与 Chen 的论文 [*Thompson Sampling for Combinatorial Semi-Bandits*](https://arxiv.org/abs/1803.04623) 讨论过近似 oracle 的问题，站内保存了 [论文 PDF](/files/papers/bandit/wang-chen-combinatorial-ts-2018.pdf)。对 UCB 类算法，近似 oracle 的误差通常可以直接转化为 approximation regret；但对 Thompson Sampling，oracle 误差会改变“哪个动作被认为乐观”的概率，不能只在最终遗憾式子里加一个常数。

目前还缺少一个统一结果来说明：若每轮 oracle 只保证

$$
A_t^\top\theta_t
\geq
\rho\max_{A\in\mathcal A}A^\top\theta_t-\varepsilon,
$$

那么 clean-run 证明中的“乐观时刻必然选择最优动作”这一环节是否仍然成立，以及 $\rho$ 和 $\varepsilon$ 的影响如何进入 regret bound，论文都没有给出结论。

### 8.7 未完成工作六：自然后验与错配后验之间的下界

目前我们有一个方向上的分离：某些实例上自然 Beta-CTS 指数差，而 BG-CTS 多项式；但还没有一个刻画两者差距的匹配下界。

论文也没有给出一个由后验联合分布直接刻画首次命中时间的匹配下界。因此目前只能看到方向性的分离，还不能判断错配后验在什么样的实例上改善探索，以及改善幅度究竟是多少。

## 附录 A：Beta 后验为什么还是 Beta

如下为正文附录补充。这里把正文中用到的两个计算分别写开：Beta 后验的更新，以及 Gaussian 后验的平方完成。

设一个 Bernoulli 臂的未知成功概率为 $\mu\in(0,1)$。均匀先验的密度是

$$
\pi_0(\mu)=1,\qquad 0<\mu<1.
$$

观察到 $s$ 次成功和 $f$ 次失败后，数据的 likelihood 为

$$
L(\mu)
=\mu^s(1-\mu)^f.
$$

Bayes 公式给出后验密度

$$
\pi(\mu\mid\text{data})
\propto
L(\mu)\pi_0(\mu)
=\mu^s(1-\mu)^f.
$$

而 $\operatorname{Beta}(a,b)$ 的密度正比于 $\mu^{a-1}(1-\mu)^{b-1}$。令 $a=s+1,b=f+1$，就得到

$$
\mu\mid\text{data}
\sim
\operatorname{Beta}(s+1,f+1).
$$

这解释了 B-CTS 的更新：每观察到一次成功，$a$ 加一；每观察到一次失败，$b$ 加一。这里的后验样本是算法为了决策生成的随机数，不是环境在本轮产生的 Bernoulli 奖励。

## 附录 B：Gaussian 后验的矩阵推导

为简化记号，设某个坐标在过去被观测了 $n$ 次，观测值为 $x_1,\ldots,x_n$，且

$$
x_s\mid\mu\sim\mathcal N(\mu,\sigma^2).
$$

使用密度为常数的不适定均匀先验。likelihood 为

$$
L(\mu)
\propto
\exp\left(-\frac1{2\sigma^2}\sum_{s=1}^{n}(x_s-\mu)^2\right).
$$

令 $\bar x=n^{-1}\sum_sx_s$。逐项展开平方和：

$$
\begin{aligned}
\sum_{s=1}^{n}(x_s-\mu)^2
&=\sum_{s=1}^{n}\bigl((x_s-\bar x)+(\bar x-\mu)\bigr)^2\\
&=\sum_{s=1}^{n}(x_s-\bar x)^2
+2(\bar x-\mu)\sum_{s=1}^{n}(x_s-\bar x)
+n(\bar x-\mu)^2\\
&=\sum_{s=1}^{n}(x_s-\bar x)^2+n(\mu-\bar x)^2.
\end{aligned}
$$

第二行中间项为零，因为 $\sum_s(x_s-\bar x)=0$。与 $\mu$ 无关的第一项可以吸收到归一化常数中，于是

$$
\pi(\mu\mid x_1,\ldots,x_n)
\propto
\exp\left(-\frac{n(\mu-\bar x)^2}{2\sigma^2}\right).
$$

这正是

$$
\mathcal N\left(\bar x,\frac{\sigma^2}{n}\right)
$$

的密度。因此，论文中的 Gaussian Thompson sample 可以写成

$$
\theta_i(t)
=\widehat\mu_i(t-1)
+\frac{\sigma}{\sqrt{N_i(t-1)}}Z_i(t),
\qquad Z_i(t)\sim\mathcal N(0,1),
$$

而 BG-CTS 只是在这个标准差上乘以 $\sqrt{2g(t)}$。

## 参考文献

1. Zhang, R. and Combes, R. (2024). [*Thompson Sampling For Combinatorial Bandits: Polynomial Regret and Mismatched Sampling Paradox*](https://arxiv.org/abs/2410.05441). NeurIPS 2024. [站内 PDF](/files/papers/bandit/zhang-combinatorial-ts-2024.pdf).
2. Zhang, R. and Combes, R. (2021). [*On the Suboptimality of Thompson Sampling in High Dimensions*](https://arxiv.org/abs/2102.05502). [站内 PDF](/files/papers/bandit/zhang-combinatorial-ts-high-dimensions-2021.pdf).
3. Wang, S. and Chen, W. (2018). [*Thompson Sampling for Combinatorial Semi-Bandits*](https://arxiv.org/abs/1803.04623). [站内 PDF](/files/papers/bandit/wang-chen-combinatorial-ts-2018.pdf).
4. 实验代码：[CTS-Mismatched-Paradox](https://github.com/RaymZhang/CTS-Mismatched-Paradox)。
