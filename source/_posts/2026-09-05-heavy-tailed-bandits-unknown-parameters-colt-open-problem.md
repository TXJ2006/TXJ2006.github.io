---
title: "当奖励有重尾而尾部未知：COLT 2025 的 Bandit 开放问题"
date: 2026-09-04 21:40:00
categories:
  - 机器学习理论
tags:
  - 多臂老虎机
  - 重尾 Bandit
  - 在线学习
  - COLT
  - 学习理论
  - 遗憾下界
  - 稳健统计
mathjax: true
toc: true
toc_number: false
comments: true
---

<p>这篇文章解读 Gianmarco Genalti 与 Alberto Maria Metelli 的论文 <a href="https://proceedings.mlr.press/v291/genalti25a.html" target="_blank" rel="noopener"><em>Open Problem: Regret Minimization in Heavy-Tailed Bandits with Unknown Distributional Parameters</em></a>。论文收录于 COLT 2025 的 <em>Proceedings of Machine Learning Research</em>，原始 PDF 也保存于站内：<a href="/files/papers/open-problems/genalti-metelli-heavy-tailed-bandits-2025.pdf">论文 PDF</a>。</p>
<p>论文研究一个看似只是“把参数估计出来”的问题：奖励分布只有有限的 $1+\varepsilon$ 阶绝对矩，且这个矩阶 $\varepsilon$ 和矩界 $u$ 都未知。参数已知时，重尾 Bandit 已经有近似匹配的上界与下界；参数未知时，论文指出，直接照搬这些界是不可能的，但最好的自适应遗憾率究竟是什么，仍然没有答案。</p>
<a id="more"></a>

<h2 id="1-先把重尾-Bandit-写清楚"><a href="#1-先把重尾-Bandit-写清楚" class="headerlink" title="1. 先把重尾 Bandit 写清楚"></a>1. 先把重尾 Bandit 写清楚</h2><h3 id="1-1-交互过程"><a href="#1-1-交互过程" class="headerlink" title="1.1 交互过程"></a>1.1 交互过程</h3><p>设有 $K\geq2$ 个臂，编号集合记作</p>
<div class="math-block">
\[
[K]:=\{1,2,\ldots,K\}.
\]
</div>

<p>$T\in\mathbb N$ 是交互轮数，$\mathbb N$ 表示正整数集合。第 $t$ 轮，算法根据过去的历史选择一个臂</p>
<div class="math-block">
\[
I_t\in[K].
\]
</div>

<p>臂 $i$ 对应一个未知奖励分布 $\nu_i$。选中 $i$ 后，环境产生</p>
<div class="math-block">
\[
X_t\sim\nu_i.
\]
</div>

<p>不同轮次的奖励在给定所选臂后独立产生；算法可以随机，也可以根据全部过去观测自适应地选择下一臂。臂 $i$ 的期望奖励是</p>
<div class="math-block">
\[
\mu_i:=\mathbb E_{X\sim\nu_i}[X].
\]
</div>

<p>令</p>
<div class="math-block">
\[
\mu^\star:=\max_{i\in[K]}\mu_i,
\qquad
i^\star\in\operatorname*{arg\,max}_{i\in[K]}\mu_i.
\]
</div>

<p>$\mathbb E$ 表示对奖励随机性和算法随机性一起取期望，$\operatorname*{arg\,max}$ 是取到最大值的所有臂的集合。</p>
<p>臂 $i$ 的差距（gap）定义为</p>
<div class="math-block">
\[
\Delta_i:=\mu^\star-\mu_i\geq0.
\]
</div>

<p>如果第 $T$ 轮结束前臂 $i$ 被选择了</p>
<div class="math-block">
\[
N_i(T):=\sum_{t=1}^{T}\mathbf 1\{I_t=i\}
\]
</div>

<p>次，那么 $N_i(T)$ 是一个随机变量，满足</p>
<div class="math-block">
\[
\sum_{i=1}^{K}N_i(T)=T.
\]
</div>

<p>这里 $\mathbf 1{I_t=i}$ 是示性函数：事件 $I_t=i$ 发生时取 $1$，否则取 $0$。</p>
<p>本文使用期望伪遗憾</p>
<div class="math-block">
\[
R_T:=\mathbb E\left[\sum_{t=1}^{T}(\mu^\star-\mu_{I_t})\right].
\]
</div>

<p>把每一轮按所选臂重新分组，就得到一个很重要的恒等式：</p>
<div class="math-block">
\[
\begin{aligned}
R_T
&=\mathbb E\left[\sum_{t=1}^{T}\sum_{i=1}^{K}
\mathbf 1\{I_t=i\}\Delta_i\right]\\
&=\sum_{i=1}^{K}\Delta_i\,\mathbb E[N_i(T)].
\end{aligned}
\]
</div>

<p>所以遗憾有两个来源：次优臂被选了多少次，以及每次选错的损失有多大。重尾问题首先改变的是“如何可靠地估计 $\mu_i$”，但最终仍然通过 $N_i(T)$ 进入遗憾。</p>
<h3 id="1-2-重尾条件到底限制了什么"><a href="#1-2-重尾条件到底限制了什么" class="headerlink" title="1.2 重尾条件到底限制了什么"></a>1.2 重尾条件到底限制了什么</h3><p>论文考虑的分布族由两个参数描述。对每个臂 $i$，要求存在</p>
<div class="math-block">
\[
\varepsilon\in(0,1],
\qquad
u>0,
\]
</div>

<p>使得</p>
<div class="math-block">
\[
\mathbb E_{X\sim\nu_i}\left[|X|^{1+\varepsilon}\right]\leq u.
\tag{1}
\]
</div>

<p>$|X|$ 是绝对值，$1+\varepsilon$ 是要求有限的矩的阶数，$u$ 是所有臂共用的上界。满足 (1) 的 $K$ 个分布组成的集合记作</p>
<div class="math-block">
\[
\mathcal E(\varepsilon,u).
\]
</div>

<p>这里的“重尾”不是指某一种唯一的分布。它允许很多不同的尾部形状，唯一的共同要求是：矩阶不超过 $1+\varepsilon$ 时，绝对矩有限并且不超过 $u$。</p>
<p>两个端点有不同含义。</p>
<ul>
<li>当 $\varepsilon=1$ 时，要求二阶绝对矩有限，通常可以把它看成有限方差量级的问题。</li>
<li>当 $\varepsilon$ 趋近于 $0$ 时，只剩下略高于一阶的矩约束，样本均值对极端观测会非常敏感。</li>
</ul>
<p>条件 (1) 不保证矩母函数</p>
<div class="math-block">
\[
\mathbb E[e^{\lambda X}]
\]
</div>

<p>在非零 $\lambda$ 处有限，因此标准的 Hoeffding 或 Chernoff 指数集中一般不能直接使用。一次很大的观测不一定违反 (1)，却可能在很长一段时间里主导经验均值。</p>
<h3 id="1-3-这在实际系统里是什么问题"><a href="#1-3-这在实际系统里是什么问题" class="headerlink" title="1.3 这在实际系统里是什么问题"></a>1.3 这在实际系统里是什么问题</h3><p>考虑网络路由。一次选择一条路径，奖励可以取为“基准时延减去实际时延”。平常时延变化不大，但偶尔会出现拥塞、故障或临时管制，产生远大于平常尺度的异常值。如果把这些异常值简单当作普通样本，算法可能因为一次事故就长期放弃一条其实平均表现更好的路径。</p>
<p>金融收益有同样的结构：大多数交易日的收益处在常规范围，少数极端波动却决定了样本均值和风险估计。这里的难点不是把极端值删掉，而是删掉多少、删掉以后会产生多大的偏差，都取决于尾部的形状。</p>
<p>在广告点击率那类有界问题中，样本均值可以用指数尾界控制；在重尾场景中，系统仍想比较长期平均奖励，却不能再假设“所有异常都不可能发生”。这正是截断、稳健估计与遗憾下界进入 Bandit 的地方。</p>
<h2 id="2-参数已知时，基线是什么"><a href="#2-参数已知时，基线是什么" class="headerlink" title="2. 参数已知时，基线是什么"></a>2. 参数已知时，基线是什么</h2><p>先假设算法知道 $\varepsilon$ 和 $u$。这不是论文要解决的最终设置，但它给出了所有自适应结果必须对照的基线。</p>
<h3 id="2-1-为什么要截断样本"><a href="#2-1-为什么要截断样本" class="headerlink" title="2.1 为什么要截断样本"></a>2.1 为什么要截断样本</h3><p>设同一条臂已经得到 $s$ 个独立奖励 $X_1,\ldots,X_s$。给定阈值 $M&gt;0$，截断均值定义为</p>
<div class="math-block">
\[
\widehat\mu_s(M)
:=\frac1s\sum_{j=1}^{s}X_j\mathbf 1\{|X_j|\leq M\}.
\tag{2}
\]
</div>

<p>它把绝对值超过 $M$ 的观测替换成 $0$，然后再取平均。这样做会引入偏差，但保留下来的变量都位于 $[-M,M]$，可以使用有界变量的集中工具。</p>
<p>截断的两个代价可以分别写出来。</p>
<p>第一，删掉尾部会造成偏差：</p>
<div class="math-block">
\[
\mu-\mathbb E[\widehat\mu_s(M)]
=\mathbb E\left[X\mathbf 1\{|X|>M\}\right].
\tag{3}
\]
</div>

<p>由 (1) 可得</p>
<div class="math-block">
\[
\left|\mathbb E\left[X\mathbf 1\{|X|>M\}\right]\right|
\leq \frac{u}{M^{\varepsilon}}.
\tag{4}
\]
</div>

<p>附录 B 会逐行证明这个不等式。它告诉我们：$M$ 越大，截断偏差越小。</p>
<p>第二，$M$ 越大，保留下来的变量范围越宽，随机波动越难控制。这里不能只看变量的范围，还要利用矩条件控制截断变量的二阶矩。因为在 $|X|\leq M$ 上有</p>
<div class="math-block">
\[
X^2\leq |X|^{1+\varepsilon}M^{1-\varepsilon},
\]
</div>

<p>所以截断变量的二阶矩至多为 $uM^{1-\varepsilon}$。Bernstein 型置信误差因此带有</p>
<div class="math-block">
\[
\sqrt{\frac{uM^{1-\varepsilon}\log(1/\delta)}s}
+\frac{M\log(1/\delta)}s
\]
</div>

<p>这一量级，其中 $\delta\in(0,1)$ 是允许的失败概率。$M$ 越小，随机误差越小，但 (4) 中的偏差越大。</p>
<p>因此，重尾估计的核心不是“把异常值删掉”，而是平衡两种误差：</p>
<div class="math-block">
\[
\text{尾部偏差}\quad \frac{u}{M^{\varepsilon}}
\qquad\text{与}\qquad
\text{样本波动}\quad M\sqrt{\frac{\log(1/\delta)}{s}}.
\]
</div>

<p>先平衡尾部偏差和 Bernstein 置信项中的平方根部分：</p>
<div class="math-block">
\[
\frac{u}{M^{\varepsilon}}
\asymp
\sqrt{\frac{uM^{1-\varepsilon}\log(1/\delta)}s}.
\]
</div>

<p>两边平方并约去一个 $u$，得到</p>
<div class="math-block">
\[
M^{1+\varepsilon}
\asymp
\frac{us}{\log(1/\delta)}.
\]
</div>

<p>因此阈值的基本尺度是</p>
<div class="math-block">
\[
M_s(\delta)\asymp
\left(\frac{us}{\log(1/\delta)}\right)^{\!1/(1+\varepsilon)}.
\tag{5}
\]
</div>

<p>这里 $a\asymp b$ 表示两者相差至多一个与 $s,u,\varepsilon,\delta$ 无关的常数因子。代回任意一项，置信误差的尺度为</p>
<div class="math-block">
\[
u^{1/(1+\varepsilon)}
\left(\frac{\log(1/\delta)}{s}\right)^{\!\varepsilon/(1+\varepsilon)}.
\tag{6}
\]
</div>

<p>当 $\varepsilon&lt;1$ 时，$s$ 的指数小于 $1/2$，这正是重尾下收敛慢于有限方差情形的地方。</p>
<h3 id="2-2-从置信误差到-instance-dependent-遗憾"><a href="#2-2-从置信误差到-instance-dependent-遗憾" class="headerlink" title="2.2 从置信误差到 instance-dependent 遗憾"></a>2.2 从置信误差到 instance-dependent 遗憾</h3><p>把 (6) 记成</p>
<div class="math-block">
\[
b_s:=c\,u^{1/(1+\varepsilon)}
\left(\frac{\log T}{s}\right)^{\!\varepsilon/(1+\varepsilon)},
\]
</div>

<p>其中 $c$ 是常数。假设算法在臂 $i$ 的置信半径降到 $\Delta_i/2$ 后，就不会因为估计误差再次频繁选择它。令 $b_s\leq\Delta_i/2$，逐步解出 $s$：</p>
<div class="math-block">
\[
\begin{aligned}
c\,u^{1/(1+\varepsilon)}
\left(\frac{\log T}{s}\right)^{\!\varepsilon/(1+\varepsilon)}
&\leq\frac{\Delta_i}{2},\\
\left(\frac{\log T}{s}\right)^{\!\varepsilon/(1+\varepsilon)}
&\leq c'\,\Delta_i u^{-1/(1+\varepsilon)},\\
\frac{\log T}{s}
&\leq c''\,\Delta_i^{(1+\varepsilon)/\varepsilon}u^{-1/\varepsilon},\\
s
&\geq c'''
\frac{u^{1/\varepsilon}\log T}
{\Delta_i^{(1+\varepsilon)/\varepsilon}}.
\end{aligned}
\]
</div>

<p>$c’,c’’,c’’’$ 只是重新吸收常数。乘以每次选择的损失 $\Delta_i$，臂 $i$ 的遗憾量级为</p>
<div class="math-block">
\[
\Delta_i s
\asymp
\left(\frac{u}{\Delta_i}\right)^{1/\varepsilon}\log T.
\]
</div>

<p>因此，已知 $\varepsilon,u$ 时的 instance-dependent 基线是</p>
<div class="math-block">
\[
R_T
\lesssim
\sum_{i:\Delta_i>0}
\left(\frac{u}{\Delta_i}\right)^{1/\varepsilon}\log T,
\tag{7}
\]
</div>

<p>其中 $\lesssim$ 表示小于一个常数倍，省略了算法初始化和低阶项。这里的 instance-dependent 意味着界依赖当前问题的差距向量 $\Delta=(\Delta_1,\ldots,\Delta_K)$。</p>
<h3 id="2-3-从-instance-dependent-到-worst-case"><a href="#2-3-从-instance-dependent-到-worst-case" class="headerlink" title="2.3 从 instance-dependent 到 worst-case"></a>2.3 从 instance-dependent 到 worst-case</h3><p>若不希望遗憾依赖某个未知的最小差距，可以把臂分成“大差距”和“小差距”两类。取一个阈值 $\gamma&gt;0$：</p>
<ul>
<li>差距不超过 $\gamma$ 的臂，即使一直选，累计损失至多为 $\gamma T$；</li>
<li>差距大于 $\gamma$ 的臂，用 (7) 求和，数量至多为 $K$。</li>
</ul>
<p>于是得到一个典型的平衡式</p>
<div class="math-block">
\[
R_T
\lesssim
\gamma T
+K u^{1/\varepsilon}\gamma^{-1/\varepsilon}\log T.
\tag{8}
\]
</div>

<p>忽略对数因子，对右侧关于 $\gamma$ 求平衡。令两项同阶：</p>
<div class="math-block">
\[
\gamma T
\asymp
K u^{1/\varepsilon}\gamma^{-1/\varepsilon}.
\]
</div>

<p>于是</p>
<div class="math-block">
\[
\gamma^{(1+\varepsilon)/\varepsilon}
\asymp
\frac{K u^{1/\varepsilon}}{T},
\]
</div>

<p>代回 (8)，得到</p>
<div class="math-block">
\[
R_T
\lesssim
K^{\varepsilon/(1+\varepsilon)}
(uT)^{1/(1+\varepsilon)}
\]
</div>

<p>的量级，通常写成</p>
<div class="math-block">
\[
\widetilde O\left(
K^{\varepsilon/(1+\varepsilon)}
(uT)^{1/(1+\varepsilon)}
\right).
\tag{9}
\]
</div>

<p>$\widetilde O$ 表示还可能含有对数因子。Bubeck、Cesa-Bianchi 与 Lugosi 2013 年的工作证明了，在 $\varepsilon,u$ 已知时，这个量级不能在一般情况下显著改善。</p>
<p>这就是自适应问题要追赶的目标：不是凭空要求一个漂亮的算法，而是要求在不知道控制尾部的两个参数时，尽量保留 (7) 和 (9) 的量级。</p>
<h2 id="3-未知参数为什么不是一个小修补"><a href="#3-未知参数为什么不是一个小修补" class="headerlink" title="3. 未知参数为什么不是一个小修补"></a>3. 未知参数为什么不是一个小修补</h2><h3 id="3-1-不知道-u：安静世界与异常世界"><a href="#3-1-不知道-u：安静世界与异常世界" class="headerlink" title="3.1 不知道 u：安静世界与异常世界"></a>3.1 不知道 u：安静世界与异常世界</h3><p>$u$ 控制极端奖励允许有多大。若算法把 $u$ 估得过小，它会把一次巨大观测当成“不可能事件”，置信界就不再覆盖真实均值；若把 $u$ 估得过大，所有臂的探索半径都会被放宽，遗憾变大。</p>
<p>关键在于：在有限时间内，两个世界可以共享几乎相同的普通观测。</p>
<ul>
<li>世界 $\nu$ 的尾部很轻，尺度为 $u$；</li>
<li>世界 $\nu’$ 只在极小概率下产生一个很大的奖励，尺度为 $u’\geq u$。</li>
</ul>
<p>在异常值还没有出现时，算法看不到两者的差别。如果为了防备 $\nu’$ 而持续探索，算法在 $\nu$ 中付出额外遗憾；如果不防备，异常值出现时又没有足够信息修正决策。</p>
<p>Genalti 等人在 2024 年的正式论文 <a href="https://proceedings.mlr.press/v247/genalti24a.html" target="_blank" rel="noopener"><em>(ε,u)-Adaptive Regret Minimization in Heavy-Tailed Bandits</em></a> 把这件事写成了负面结果。对固定 $\varepsilon$，他们证明任何 $u$-adaptive 算法都不可能保持已知 $u$ 时的统一归一化遗憾：对任意 $u’\geq u$，存在两个实例，使得下面的式子（该论文定理 2 的式 (6)）成立：</p>
<div class="math-block">
\[
\max\left\lbrace
\frac{R_T(\nu)}{u^{1/(1+\varepsilon)}},
\frac{R_T(\nu^{\prime})}{(u^{\prime})^{1/(1+\varepsilon)}}
\right\rbrace
\geq
c_1\left(\frac{u^{\prime}}u\right)^{\varepsilon/(1+\varepsilon)^2}
T^{1/(1+\varepsilon)}.
\tag{10}
\]
</div>

<p>$c_1&gt;0$ 是与 $u,u’,T$ 无关的常数。因为 $u’/u$ 可以任意大，这个比值不可能被一个统一常数控制。论文把这一结论概括为：不知道 $u$ 时，已知参数情形的 minimax 归一化界不能直接保留。</p>
<p>这里 minimax 指最坏分布上的遗憾；$u$-adaptive 指算法运行时不把 $u$ 作为输入。这个下界并不是说任何数据驱动方法都没有意义，而是说“对所有 $u$ 都不增加代价”这一要求过强。</p>
<h3 id="3-2-不知道-ε：尾部形状本身也会隐藏"><a href="#3-2-不知道-ε：尾部形状本身也会隐藏" class="headerlink" title="3.2 不知道 ε：尾部形状本身也会隐藏"></a>3.2 不知道 ε：尾部形状本身也会隐藏</h3><p>即使固定 $u=1$，未知 $\varepsilon$ 仍然造成另一种不可适应性。取两个阶数</p>
<div class="math-block">
\[
0 \lt \varepsilon'\leq\varepsilon\leq1.
\]
</div>

<p>$\varepsilon’$ 越小，允许的尾部越重；已知参数时，最坏情形遗憾中的时间指数 $1/(1+\varepsilon’)$ 越接近 $1$。问题在于，有限样本下，一个轻尾分布和一个极少发生极端事件的重尾分布也可能难以区分。</p>
<p>2024 年论文给出的结论是：存在两个分别属于 $\varepsilon$ 和 $\varepsilon’$ 类的实例，使得任意 $\varepsilon$-adaptive 算法满足下面的式子（该论文定理 3 的式 (8)）：</p>
<div class="math-block">
\[
\max\left\lbrace
\frac{R_T(\nu)}{T^{1/(1+\varepsilon)}},
\frac{R_T(\nu^{\prime})}{T^{1/(1+\varepsilon^{\prime})}}
\right\rbrace
\geq
c_2T^{\frac{\varepsilon^{\prime}(\varepsilon-\varepsilon^{\prime})}{(1+\varepsilon)(1+\varepsilon^{\prime})^2}}.
\tag{11}
\]
</div>

<p>$c_2&gt;0$ 是常数。让 $\varepsilon=1$、$\varepsilon’=1/3$，右侧可以得到 $T^{1/16}$ 的额外因子。因此，未知 $\varepsilon$ 时也不可能对所有重尾分布都保持已知参数的 worst-case 量级。</p>
<p>这两个负面结果的共同结构是：算法需要用观测判断自己处在哪一个尾部世界，但观测本身又必须通过选择臂来获得。参数估计不是决策之外的预处理阶段，它和探索成本绑在了一起。</p>
<h3 id="3-3-这和普通的未知方差有什么不同"><a href="#3-3-这和普通的未知方差有什么不同" class="headerlink" title="3.3 这和普通的未知方差有什么不同"></a>3.3 这和普通的未知方差有什么不同</h3><p>在 Gaussian 或有界奖励中，未知尺度通常可以从样本方差或极差中逐渐识别；更重要的是，异常观测不会把均值估计推到任意远。重尾条件只给出一个有限矩，既允许极端值很大，也允许它们出现得非常少。</p>
<p>因此，“先用一段数据估计 $u,\varepsilon$，再运行已知参数算法”并不是一个自动成立的两阶段方案。第一阶段若没有看到尾部，估计可能过于乐观；若为了保证不乐观而长期等待，等待本身就是 Bandit 遗憾。</p>
<h2 id="4-增加假设可以得到什么，又牺牲了什么"><a href="#4-增加假设可以得到什么，又牺牲了什么" class="headerlink" title="4. 增加假设可以得到什么，又牺牲了什么"></a>4. 增加假设可以得到什么，又牺牲了什么</h2><h3 id="4-1-截断非正性假设"><a href="#4-1-截断非正性假设" class="headerlink" title="4.1 截断非正性假设"></a>4.1 截断非正性假设</h3><p>为了在未知 $\varepsilon,u$ 时得到可用结果，文献引入过若干额外假设。Genalti 等人重点讨论一种不直接写出 $\varepsilon,u$ 数值的条件：截断非正性（truncated non-positivity）。设最优臂编号为 $1$，要求对任意 $M\geq0$，都有</p>
<div class="math-block">
\[
\mathbb E_{X\sim\nu_1}
\left[X\mathbf 1\{|X|>M\}\right]\leq0.
\tag{12}
\]
</div>

<p>它的作用是单向的。对最优臂做截断时，被删除的尾部期望不会为正，因此截断不会把最优臂的估计系统性地压低到一个无法控制的方向。这个假设并不限制尾部一定很轻，也不需要算法知道它的参数值。</p>
<p>它不是一个“免费”的技术条件。没有 (12) 时，最优臂可能恰好通过极少数的巨大正奖励获得更高均值；把这些奖励删掉以后，截断均值会产生向下偏差，而一个只看截断均值的乐观上界无法直接恢复这部分信息。</p>
<h3 id="4-2-AdaR-UCB-把阈值也从数据中找出来"><a href="#4-2-AdaR-UCB-把阈值也从数据中找出来" class="headerlink" title="4.2 AdaR-UCB 把阈值也从数据中找出来"></a>4.2 AdaR-UCB 把阈值也从数据中找出来</h3><p>2024 年论文构造了 Adaptive Robust UCB（AdaR-UCB）。它不把 $\varepsilon,u$ 输入算法，而是先用一批独立样本 $X’_1,\ldots,X’_s$ 求一个经验阈值 $\widehat M_s(\delta)$。阈值被定义为正方程的根：</p>
<div class="math-block">
\[
\frac1s\sum_{j=1}^{s}
\frac{\min\{(X'_j)^2,M^2\}}{M^2}
-\frac{c\log(1/\delta)}s
=0,
\tag{13}
\]
</div>

<p>其中 $c&gt;0$ 是常数，$M$ 是待求的变量。分子中的 $\min{(X’_j)^2,M^2}$ 把平方观测截在 $M^2$，所以左侧随 $M$ 的变化可以从数据中计算。</p>
<p>用另一批独立样本 $X_1,\ldots,X_s$ 计算截断均值和截断样本方差</p>
<div class="math-block">
\[
\widehat\mu_s(X;M)
:=\frac1s\sum_{j=1}^{s}X_j\mathbf 1\{|X_j|\leq M\},
\]
</div>

<div class="math-block">
\[
V_s(X;M)
:=\frac1{s-1}\sum_{j=1}^{s}
\left(X_j\mathbf 1\{|X_j|\leq M\}-\widehat\mu_s(X;M)\right)^2.
\]
</div>

<p>在截断非正性假设下，论文证明了一个不需要输入 $\varepsilon,u$ 的单侧置信界：以至少 $1-2\delta$ 的概率，</p>
<div class="math-block">
\[
\mu-\widehat\mu_s(X;M)
\leq
\sqrt{\frac{2V_s(X;M)\log(1/\delta)}s}
+\frac{10M\log(1/\delta)}s.
\tag{14}
\]
</div>

<p>算法用经验截断均值加上右侧置信项构造 UCB。这里的关键不是某个常数 $10$，而是置信项只依赖样本方差、阈值和失败概率，可以在不知道尾部参数时计算。</p>
<p>论文报告的 AdaR-UCB 结果是：在 (12) 下，它的 instance-dependent 遗憾与已知参数基线只相差额外的对数项和强制探索项；worst-case 遗憾也达到 (9) 的量级，允许对数因子。对臂 $i$，instance-dependent 结果中出现</p>
<div class="math-block">
\[
\left(\frac{u}{\Delta_i}\right)^{1/\varepsilon}
+\frac{\Delta_i}{\mathbb P_{\nu_i}(X\neq0)}
\]
</div>

<p>这样的两部分。第一部分是重尾估计本身的代价，第二部分来自为计算经验阈值而进行的额外观测。</p>
<p>这说明增加一个与参数数值无关的分布形状假设，确实可以把问题推进到接近已知参数的程度；但它并没有说明没有假设时的最优答案是什么。</p>
<h2 id="5-COLT-2025-正式留下的三个开放问题"><a href="#5-COLT-2025-正式留下的三个开放问题" class="headerlink" title="5. COLT 2025 正式留下的三个开放问题"></a>5. COLT 2025 正式留下的三个开放问题</h2><h3 id="5-1-开放问题一：无额外假设时，最坏情形遗憾的最佳速率是什么"><a href="#5-1-开放问题一：无额外假设时，最坏情形遗憾的最佳速率是什么" class="headerlink" title="5.1 开放问题一：无额外假设时，最坏情形遗憾的最佳速率是什么"></a>5.1 开放问题一：无额外假设时，最坏情形遗憾的最佳速率是什么</h3><p>已知 $\varepsilon,u$ 时，最坏情形基线是 (9)。未知参数且不加假设时，2024 年的负面结果说明这个基线不能原样复制，但它没有给出新的精确 minimax 速率。</p>
<p>因此第一个问题是：对每一组固定但算法事先不知道的 $(\varepsilon,u)$，在分布族</p>
<div class="math-block">
\[
\mathcal E(\varepsilon,u)
\]
</div>

<p>上，究竟能达到怎样的最坏情形遗憾率？这里不能把 $u&gt;0$ 直接取无界上确界后再声称一个有限的“关于 $u$ 的速率”：$u$ 是要出现在答案中的问题尺度，而不是要被消掉的变量。更准确地说，对一个不读取 $(\varepsilon,u)$ 的统一算法 $\pi$，定义</p>
<div class="math-block">
\[
\mathfrak R_T^\pi(\varepsilon,u)
:=\sup_{\nu\in\mathcal E(\varepsilon,u)}R_T^\pi(\nu),
\qquad
\mathfrak R_T^\pi(\varepsilon,u,\Delta)
:=\sup_{\nu\in\mathcal E(\varepsilon,u,\Delta)}R_T^\pi(\nu).
\]
</div>

<p>论文要寻找的是这两个量的匹配上下界：一个只依赖 $T,K$（以及在 instance-dependent 情形下的 $\Delta$）的增长函数，前面乘上由 $\varepsilon,u$ 决定的尺度；同时算法 $\pi$ 对所有固定的 $(\varepsilon,u)$ 都不改变。这里的关键不是找一个暂时可行的上界，而是同时给出：</p>
<ol>
<li>所有算法都必须承受的 minimax 下界；</li>
<li>一个不使用 $\varepsilon,u$ 的算法上界；</li>
<li>两者在主要参数上的匹配关系。</li>
</ol>
<p>论文把这称为“best attainable performance”，而没有把答案预先写成某个猜测公式。</p>
<h3 id="5-2-开放问题二：是否存在匹配这个速率的算法"><a href="#5-2-开放问题二：是否存在匹配这个速率的算法" class="headerlink" title="5.2 开放问题二：是否存在匹配这个速率的算法"></a>5.2 开放问题二：是否存在匹配这个速率的算法</h3><p>第一个问题确定目标后，第二个问题问能否真正达到它。形式上，需要找到一个算法，它在未知 $\varepsilon,u$ 时同时处理：</p>
<ul>
<li>不同的尾部阶数；</li>
<li>不同的尾部尺度；</li>
<li>instance-dependent 遗憾中的差距依赖；</li>
<li>worst-case 遗憾中的 $T,K$ 依赖。</li>
</ul>
<p>这四项不能由一个“先估计参数、再套公式”的口号自动得到。若阈值选得太小，极端奖励造成的偏差会漏进置信界；若阈值选得太大，样本波动会让探索次数膨胀。未知 $\varepsilon$ 还意味着算法不知道置信误差应当按 $s^{-\varepsilon/(1+\varepsilon)}$ 的哪一个指数衰减。</p>
<p>AdaR-UCB 解决的是带截断非正性假设的子问题。COLT 2025 的开放问题要求把这个假设拿掉，并且仍然确定最优代价，而不是只证明某一个算法在某个假设下有效。</p>
<h3 id="5-3-开放问题三：有没有比现有假设更自然、更弱的条件"><a href="#5-3-开放问题三：有没有比现有假设更自然、更弱的条件" class="headerlink" title="5.3 开放问题三：有没有比现有假设更自然、更弱的条件"></a>5.3 开放问题三：有没有比现有假设更自然、更弱的条件</h3><p>文献中的额外假设并不形成一条包含关系。</p>
<ul>
<li>截断非正性直接控制截断对最优臂的偏差方向；</li>
<li>对称性假设要求奖励分布围绕均值具有对称结构；</li>
<li>另一些方法要求时域 $T$ 大于一个依赖 $u,\varepsilon$ 的阈值。</li>
</ul>
<p>这些条件都能排除一部分困难分布，但没有一个被证明严格弱于其余所有条件。因此论文提出第三个问题：是否存在一个“更好”的假设，它包含所有能够无额外自适应代价的分布，同时又不把不必要的结构排除在外？</p>
<p>这不是在问哪个假设写起来更短，而是在问自适应代价究竟由哪一种尾部几何触发。若两个分布都允许未知重尾，但一个可以达到已知参数的速率、另一个不可以，那么区分它们的最小结构条件是什么，目前仍没有统一答案。</p>
<h2 id="6-论文真正留下的技术缺口"><a href="#6-论文真正留下的技术缺口" class="headerlink" title="6. 论文真正留下的技术缺口"></a>6. 论文真正留下的技术缺口</h2><h3 id="6-1-阈值、偏差和波动必须同时控制"><a href="#6-1-阈值、偏差和波动必须同时控制" class="headerlink" title="6.1 阈值、偏差和波动必须同时控制"></a>6.1 阈值、偏差和波动必须同时控制</h3><p>已知参数时，(4) 和 (5) 把阈值尺度算得很清楚；未知参数时，困难不是少一个代入步骤，而是阈值本身也是随机的。算法需要用数据决定 $M$，但决定 $M$ 的数据又可能没有包含真正重要的极端事件。</p>
<p>如果 $M$ 小于理想尺度，截断偏差可能大于两个臂之间的差距；如果 $M$ 远大于理想尺度，置信半径中与 $M$ 成正比的部分会主导。要达到匹配下界，必须在未知参数的情况下把这两种失败都纳入同一个时间均匀的分析。</p>
<h3 id="6-2-instance-dependent-与-worst-case-不能分开处理"><a href="#6-2-instance-dependent-与-worst-case-不能分开处理" class="headerlink" title="6.2 instance-dependent 与 worst-case 不能分开处理"></a>6.2 instance-dependent 与 worst-case 不能分开处理</h3><p>已知参数的 (7) 和 (9) 由同一个估计误差产生，但它们对小差距的要求不同。instance-dependent 界希望对每个 $\Delta_i$ 只付出必要的探索；worst-case 界则要在最坏差距配置下保持次线性。</p>
<p>在非重尾的许多 Bandit 模型中，可以通过同一个置信界分别推导两种结果。重尾自适应问题里，负面结果说明这条兼容性可能被未知参数破坏。论文没有回答：最优的自适应算法是否必须在两类界之间做不可避免的取舍，或者是否存在一种新的遗憾函数可以同时刻画它们。</p>
<h3 id="6-3-额外假设的边界还没有统一描述"><a href="#6-3-额外假设的边界还没有统一描述" class="headerlink" title="6.3 额外假设的边界还没有统一描述"></a>6.3 额外假设的边界还没有统一描述</h3><p>截断非正性让单侧 UCB 成为可能，但它对某些自然分布并不包容；例如论文讨论到，Pareto 型分布可能不满足该假设，却仍有办法控制截断偏差。另一方面，对称性又会排除一些实际的偏斜分布。</p>
<p>所以“增加假设”不是一个二元开关。真正的边界是：哪些尾部形状允许数据驱动的置信界保持方向正确，哪些尾部形状会迫使算法付出额外探索。COLT 2025 只把这条边界明确地列为问题，没有声称已经找到它。</p>
<h2 id="7-这篇开放问题和前面-Bandit-主线怎么接上"><a href="#7-这篇开放问题和前面-Bandit-主线怎么接上" class="headerlink" title="7. 这篇开放问题和前面 Bandit 主线怎么接上"></a>7. 这篇开放问题和前面 Bandit 主线怎么接上</h2><p>前面的组合 Thompson Sampling 文章关注的是：后验样本如何在高维组合动作中产生探索概率。这里换成重尾 Bandit，采样分布不再是唯一瓶颈，奖励估计本身的尾部也进入了遗憾。</p>
<p>两篇文章的共同骨架仍然是：</p>
<ol>
<li>先写清楚反馈能提供哪些观测；</li>
<li>再写清楚一个估计量的误差如何进入动作选择；</li>
<li>最后用下界判断这个误差是否只是算法实现问题，还是信息本身不允许更好。</li>
</ol>
<p>在组合 Thompson Sampling 中，正确的 Beta 后验可能因为联合采样结构而探索不足；在重尾 Bandit 中，即使估计器形式正确，未知尾部参数也可能让置信界无法同时适用于所有分布。前者是“后验到动作”的几何问题，后者是“观测到置信界”的适应问题。</p>
<p>这正是这篇 COLT 论文值得收录的原因：它没有给出一个孤立的技巧，而是把在线学习中一个很实际的目标写成了可检验的开放问题。我们目前可以确切地知道哪些参数已知时是最优的、哪些自适应要求已经被下界排除，以及还缺少哪一段匹配证明；但还不能把“未知重尾下的最优遗憾”写成已经解决的定理。</p>
<h2 id="附录-A：符号、分布与遗憾指标"><a href="#附录-A：符号、分布与遗憾指标" class="headerlink" title="附录 A：符号、分布与遗憾指标"></a>附录 A：符号、分布与遗憾指标</h2><p>如下为正文附录补充。这里集中解释正文出现过、但容易混淆的符号。</p>
<h3 id="A-1-绝对矩与重尾阶数"><a href="#A-1-绝对矩与重尾阶数" class="headerlink" title="A.1 绝对矩与重尾阶数"></a>A.1 绝对矩与重尾阶数</h3><p>对实随机变量 $X$ 和 $p&gt;0$，$p$ 阶绝对矩定义为</p>
<div class="math-block">
\[
\mathbb E|X|^p.
\]
</div>

<p>正文取 $p=1+\varepsilon$。因此</p>
<div class="math-block">
\[
\mathbb E|X|^{1+\varepsilon}\leq u
\]
</div>

<p>不是说 $X$ 的取值被限制在某个区间，而是说非常大的取值出现得足够少，使这个加权平均仍然有限。$\varepsilon$ 越小，允许的尾部越重。</p>
<h3 id="A-2-截断变量和截断均值"><a href="#A-2-截断变量和截断均值" class="headerlink" title="A.2 截断变量和截断均值"></a>A.2 截断变量和截断均值</h3><p>给定阈值 $M&gt;0$，定义</p>
<div class="math-block">
\[
Y_j:=X_j\mathbf 1\{|X_j|\leq M\}.
\]
</div>

<p>于是 $Y_j=X_j$ 当 $|X_j|\leq M$，否则 $Y_j=0$。截断均值就是</p>
<div class="math-block">
\[
\widehat\mu_s(M)=\frac1s\sum_{j=1}^{s}Y_j.
\]
</div>

<p>$Y_j$ 有界于 $[-M,M]$，但它的期望一般不再等于 $\mu=\mathbb E[X]$。两者的差正是尾部期望：</p>
<div class="math-block">
\[
\begin{aligned}
\mu-\mathbb E[Y_j]
&=\mathbb E[X]-\mathbb E[X\mathbf 1\{|X|\leq M\}]\\
&=\mathbb E\left[X\left(1-\mathbf 1\{|X|\leq M\}\right)\right]\\
&=\mathbb E\left[X\mathbf 1\{|X|>M\}\right].
\end{aligned}
\]
</div>

<h3 id="A-3-置信失败概率和渐近记号"><a href="#A-3-置信失败概率和渐近记号" class="headerlink" title="A.3 置信失败概率和渐近记号"></a>A.3 置信失败概率和渐近记号</h3><p>$\delta\in(0,1)$ 是置信失败概率。写成“以至少 $1-2\delta$ 的概率成立”，意思是对应事件的概率不小于 $1-2\delta$。</p>
<p>若 $a\lesssim b$，表示存在一个普适常数 $C&gt;0$ 使 $a\leq Cb$。若 $a\asymp b$，表示两边互相都被对方的常数倍控制。$\widetilde O(\cdot)$ 允许额外的 $\log T$ 或 $\log K$ 因子。</p>
<p>instance-dependent bound 依赖具体实例的差距 $\Delta_i$；worst-case 或 minimax bound 则对给定参数范围内的所有奖励分布取上确界，再研究这个最坏值随 $T,K$ 如何增长。</p>
<h2 id="附录-B：截断偏差与阈值尺度的逐步推导"><a href="#附录-B：截断偏差与阈值尺度的逐步推导" class="headerlink" title="附录 B：截断偏差与阈值尺度的逐步推导"></a>附录 B：截断偏差与阈值尺度的逐步推导</h2><h3 id="B-1-从有限矩得到尾部偏差"><a href="#B-1-从有限矩得到尾部偏差" class="headerlink" title="B.1 从有限矩得到尾部偏差"></a>B.1 从有限矩得到尾部偏差</h3><p>由正文的矩条件，</p>
<div class="math-block">
\[
\mathbb E|X|^{1+\varepsilon}\leq u.
\]
</div>

<p>在事件 $|X|&gt;M$ 上，有 $|X|^{\varepsilon}&gt;M^{\varepsilon}$，因此</p>
<div class="math-block">
\[
|X|=\frac{|X|^{1+\varepsilon}}{|X|^{\varepsilon}}
\leq\frac{|X|^{1+\varepsilon}}{M^{\varepsilon}}.
\]
</div>

<p>两边乘以示性函数 $\mathbf 1{|X|&gt;M}$，得到逐点不等式</p>
<div class="math-block">
\[
|X|\mathbf 1\{|X|>M\}
\leq
\frac{|X|^{1+\varepsilon}}{M^{\varepsilon}}.
\]
</div>

<p>取期望，并使用期望的单调性：</p>
<div class="math-block">
\[
\mathbb E\left[|X|\mathbf 1\{|X|>M\}\right]
\leq
\frac{\mathbb E|X|^{1+\varepsilon}}{M^{\varepsilon}}
\leq
\frac{u}{M^{\varepsilon}}.
\tag{B.1}
\]
</div>

<p>另一方面，</p>
<div class="math-block">
\[
\left|\mathbb E\left[X\mathbf 1\{|X|>M\}\right]\right|
\leq
\mathbb E\left[|X|\mathbf 1\{|X|>M\}\right].
\]
</div>

<p>结合 (B.1)，得到正文中的</p>
<div class="math-block">
\[
\left|\mu-\mathbb E[\widehat\mu_s(M)]\right|
\leq
\frac{u}{M^{\varepsilon}}.
\]
</div>

<h3 id="B-2-为什么阈值含有-1-1-ε"><a href="#B-2-为什么阈值含有-1-1-ε" class="headerlink" title="B.2 为什么阈值含有 1/(1+ε)"></a>B.2 为什么阈值含有 1/(1+ε)</h3><p>把尾部偏差写成</p>
<div class="math-block">
\[
B(M):=\frac{u}{M^{\varepsilon}}.
\]
</div>

<p>对截断变量 $Y=X\mathbf 1{|X|\leq M}$，在 $|X|\leq M$ 上有</p>
<div class="math-block">
\[
Y^2\leq |X|^{1+\varepsilon}M^{1-\varepsilon}.
\]
</div>

<p>取期望可得</p>
<div class="math-block">
\[
\mathbb E[Y^2]\leq uM^{1-\varepsilon}.
\]
</div>

<p>所以 Bernstein 型波动项的平方根部分为</p>
<div class="math-block">
\[
V(M):=\sqrt{\frac{uM^{1-\varepsilon}\log(1/\delta)}s}.
\]
</div>

<p>将尾部偏差与该项平衡：</p>
<div class="math-block">
\[
\frac{u}{M^{\varepsilon}}=V(M).
\]
</div>

<p>两边平方并约去一个 $u$：</p>
<div class="math-block">
\[
\frac{u}{M^{2\varepsilon}}
=\frac{M^{1-\varepsilon}\log(1/\delta)}s
\quad\Longrightarrow\quad
M^{1+\varepsilon}=\frac{us}{\log(1/\delta)}.
\]
</div>

<p>这就得到</p>
<div class="math-block">
\[
M\asymp\left(\frac{us}{\log(1/\delta)}\right)^{1/(1+\varepsilon)}.
\]
</div>

<p>Bernstein 项中的线性部分 $M\log(1/\delta)/s$ 在这个阈值下也具有与偏差项相同的阶，因此不会改变阈值的指数。</p>
<h3 id="B-3-用标准阈值检查置信误差的阶"><a href="#B-3-用标准阈值检查置信误差的阶" class="headerlink" title="B.3 用标准阈值检查置信误差的阶"></a>B.3 用标准阈值检查置信误差的阶</h3><p>取</p>
<div class="math-block">
\[
M_s(\delta)=
\left(\frac{us}{\log(1/\delta)}\right)^{1/(1+\varepsilon)}.
\]
</div>

<p>先计算偏差项：</p>
<div class="math-block">
\[
\begin{aligned}
\frac{u}{M_s(\delta)^{\varepsilon}}
&=u\left(\frac{us}{\log(1/\delta)}\right)^{-\varepsilon/(1+\varepsilon)}\\
&=u^{1/(1+\varepsilon)}
\left(\frac{\log(1/\delta)}s\right)^{\varepsilon/(1+\varepsilon)}.
\end{aligned}
\]
</div>

<p>再计算截断尺度乘以 $s^{-1}$ 的 Bernstein 型主项：</p>
<div class="math-block">
\[
\begin{aligned}
M_s(\delta)\frac{\log(1/\delta)}s
&=u^{1/(1+\varepsilon)}
\left(\frac{s}{\log(1/\delta)}\right)^{1/(1+\varepsilon)}
\frac{\log(1/\delta)}s\\
&=u^{1/(1+\varepsilon)}
\left(\frac{\log(1/\delta)}s\right)^{\varepsilon/(1+\varepsilon)}.
\end{aligned}
\]
</div>

<p>两项具有相同的 $s$、$u$、$\varepsilon$ 依赖，这就是 (6) 的来源。</p>
<h2 id="附录-C：自适应-Bandit-下界中的信息分解"><a href="#附录-C：自适应-Bandit-下界中的信息分解" class="headerlink" title="附录 C：自适应 Bandit 下界中的信息分解"></a>附录 C：自适应 Bandit 下界中的信息分解</h2><p>这里不复述 2024 年论文的完整下界证明，只说明“安静世界和异常世界为什么能同时出现”所依赖的通用信息恒等式。</p>
<p>设真实环境为 $\nu=(\nu_1,\ldots,\nu_K)$，另一个环境为 $\nu^{\prime}=(\nu^{\prime}_1,\ldots,\nu^{\prime}_K)$。完整历史写成</p>
<div class="math-block">
\[
H_T=(I_1,X_1,\ldots,I_T,X_T).
\]
</div>

<p>算法在历史 $h_{t-1}$ 下选择臂 $i$ 的概率记为 $\pi_t(i\mid h_{t-1})$。在两个环境中，算法规则相同，只有奖励分布不同。</p>
<p>历史密度可以按时间分解：</p>
<div class="math-block">
\[
P_\nu(H_T)
=\prod_{t=1}^{T}
\pi_t(I_t\mid H_{t-1})\,p_{I_t}(X_t),
\]
</div>

<div class="math-block">
\[
P_{\nu^{\prime}}(H_T)
=\prod_{t=1}^{T}
\pi_t(I_t\mid H_{t-1})\,p^{\prime}_{I_t}(X_t),
\]
</div>

<p>其中 $p_i,p^{\prime}_i$ 是 $\nu_i,\nu^{\prime}_i$ 的密度；离散分布时把密度换成概率质量即可。</p>
<p>两式相除，动作策略项逐项抵消：</p>
<div class="math-block">
\[
\frac{P_\nu(H_T)}{P_{\nu^{\prime}}(H_T)}
=\prod_{t=1}^{T}
\frac{p_{I_t}(X_t)}{p^{\prime}_{I_t}(X_t)}.
\]
</div>

<p>取对数并在环境 $\nu$ 下取期望：</p>
<div class="math-block">
\[
\begin{aligned}
\operatorname{KL}(P_\nu^T\Vert P_{\nu^{\prime}}^T)
&=\mathbb E_\nu\left[
\sum_{t=1}^{T}
\log\frac{p_{I_t}(X_t)}{p^{\prime}_{I_t}(X_t)}
\right]\\
&=\sum_{t=1}^{T}\sum_{i=1}^{K}
\mathbb P_\nu(I_t=i)\operatorname{KL}(\nu_i\Vert\nu^{\prime}_i)\\
&=\sum_{i=1}^{K}
\mathbb E_\nu[N_i(T)]
\operatorname{KL}(\nu_i\Vert\nu^{\prime}_i).
\end{aligned}
\tag{C.1}
\]
</div>

<p>最后一步使用</p>
<div class="math-block">
\[
\mathbb E_\nu[N_i(T)]
=\sum_{t=1}^{T}\mathbb P_\nu(I_t=i).
\]
</div>

<p>(C.1) 的含义很直接：算法在臂 $i$ 上观察多少次，就积累多少份区分 $\nu_i$ 与 $\nu^{\prime}_i$ 的信息。如果两个世界在臂 $i$ 上的单次 KL 散度很小，那么要让整个历史明显区分它们，就必须让 $N_i(T)$ 足够大。</p>
<p>重尾下的困难正是可以把差异放进极少发生的尾部，使单次观测的区分信息很小；但一旦尾部事件发生，它又可能带来很大的均值差。下界构造利用的就是“信息少、潜在后果大”这组张力。这个恒等式本身不是 COLT 2025 的新定理，却解释了为什么未知尾部参数会成为在线决策的下界问题，而不仅是估计器调参问题。</p>
<h2 id="参考文献"><a href="#参考文献" class="headerlink" title="参考文献"></a>参考文献</h2><ol>
<li>Genalti, G. and Metelli, A. M. (2025). <a href="https://proceedings.mlr.press/v291/genalti25a.html" target="_blank" rel="noopener"><em>Open Problem: Regret Minimization in Heavy-Tailed Bandits with Unknown Distributional Parameters</em></a>. <em>Proceedings of the Thirty Eighth Conference on Learning Theory</em>, PMLR 291:1-5. <a href="/files/papers/open-problems/genalti-metelli-heavy-tailed-bandits-2025.pdf">站内 PDF</a>。</li>
<li>Genalti, G., Marsigli, L., Gatti, N. and Metelli, A. M. (2024). <a href="https://proceedings.mlr.press/v247/genalti24a.html" target="_blank" rel="noopener"><em>(ε,u)-Adaptive Regret Minimization in Heavy-Tailed Bandits</em></a>. <em>Proceedings of the Thirty Seventh Conference on Learning Theory</em>, PMLR 247:1882-1915.</li>
<li>Bubeck, S., Cesa-Bianchi, N. and Lugosi, G. (2013). <a href="https://doi.org/10.1109/TIT.2013.2278454" target="_blank" rel="noopener"><em>Bandits with Heavy Tail</em></a>. <em>IEEE Transactions on Information Theory</em>, 59(11), 7711-7717.</li>
<li>Lattimore, T. and Szepesvári, C. (2020). <a href="https://tor-lattimore.com/downloads/book/book.pdf" target="_blank" rel="noopener"><em>Bandit Algorithms</em></a>. Cambridge University Press.</li>
</ol>

      
    </div>
