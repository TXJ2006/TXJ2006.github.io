---
title: "静态遗憾与动态遗憾能否同时最优：对手模型与 Blackwell 方法"
date: 2026-09-04 22:00:00
categories:
  - 机器学习理论
tags:
  - 多臂老虎机
  - 静态遗憾
  - 动态遗憾
  - 对抗性 Bandit
  - Blackwell Approachability
  - 在线学习
  - COLT 与学习理论
mathjax: true
toc: true
toc_number: false
comments: true
---

<p>这一篇解读 Jian Qian 与 Chen-Yu Wei 的论文 <a href="https://arxiv.org/abs/2602.07418" target="_blank" rel="noopener"><em>Achieving Optimal Static and Dynamic Regret Simultaneously in Bandits with Deterministic Losses</em></a>。论文的站内 PDF 已保存为 <a href="/files/papers/bandit/simultaneous-static-dynamic-regret-bandits-2026.pdf">simultaneous-static-dynamic-regret-bandits-2026.pdf</a>。</p>

<p>研究目的很明确：同一个 Bandit 算法，能不能在不知道环境究竟“基本稳定”还是“频繁变化”的情况下，同时获得静态遗憾的最优阶和动态遗憾的最优阶？论文给出的答案不是一个无条件的“可以”或“不可以”，而是揭示了对手模型的分界：</p>

<table>
<thead>
<tr><th>对手</th><th>损失</th><th>结论</th></tr>
</thead>
<tbody>
<tr><td>adaptive</td><td>确定性</td><td>同时最优仍不可能</td></tr>
<tr><td>oblivious</td><td>确定性，已知切换数 $S$</td><td>可以同时达到最优阶</td></tr>
</tbody>
</table>

<p>文章的技术核心有两层。第一层是一个两臂构造：如果对手能在看到学习者探测成功后立刻撤回“变好”的臂，那么探索和追踪之间存在不可消除的冲突。第二层是 Blackwell approachability：把“对每个固定臂的静态遗憾”和“对每一轮最佳臂的动态遗憾”放在同一个向量空间里，让一个元算法在两种基础策略之间调节，并利用探索产生的负静态遗憾抵消探索成本。</p>

<a id="more"></a>

<h2 id="1-问题从一条臂开始"><a href="#1-问题从一条臂开始" class="headerlink" title="1. 问题从一条臂开始"></a>1. 问题从一条臂开始</h2>

<h3 id="1-1-交互过程和符号"><a href="#1-1-交互过程和符号" class="headerlink" title="1.1 交互过程和符号"></a>1.1 交互过程和符号</h3>

<p>设有 $A\geq 2$ 个动作（也称为臂），编号集合写成</p>

<div class="math-block">
\[
[A]:=\{1,2,\ldots,A\}.
\]
</div>

<p>总交互轮数为 $T\in\mathbb N$，其中 $\mathbb N$ 表示正整数集合。第 $t$ 轮开始前，对手给每个动作指定损失，组成向量</p>

<div class="math-block">
\[
\mathbf c_t=(c_t(1),c_t(2),\ldots,c_t(A))\in[0,1]^A.
\]
</div>

<p>$c_t(a)$ 是第 $t$ 轮选择动作 $a$ 要付出的损失。学习者选择 $I_t\in[A]$，只能观察自己选择的那一项 $c_t(I_t)$，不能直接看到其他动作的损失。这就是 bandit feedback。</p>

<p>论文中的“确定性损失”有一个容易忽略的精确定义：一旦动作 $I_t$ 被选中，反馈就是确定的数值 $c_t(I_t)$，没有额外噪声。如果一般模型写作 $c_t(I_t)+w_t$，这里相当于 $w_t=0$。确定性并不意味着学习者能看到整个向量；反馈仍然是单坐标的。</p>

<h3 id="1-2-损失为什么会切换"><a href="#1-2-损失为什么会切换" class="headerlink" title="1.2 损失为什么会切换"></a>1.2 损失为什么会切换</h3>

<p>如果每个 $\mathbf c_t$ 都相同，问题接近普通的静态对抗性 Bandit；如果损失随时间变化，昨天最好的臂今天可能已经不再好。论文用切换数刻画变化次数：</p>

<div class="math-block">
\[
S:=1+\sum_{t=2}^{T}\mathbb I\{\mathbf c_t\neq \mathbf c_{t-1}\}.
\]
</div>

<p>$\mathbb I\{E\}$ 是示性函数：事件 $E$ 成立时取 $1$，否则取 $0$。因此 $S$ 是损失向量初始出现的那一段算作一次，再加上真正发生变化的次数。损失最多变化 $S-1$ 次。论文假设学习者知道 $S$，但不知道变化具体发生在哪些轮。</p>

<h3 id="1-3-两种对手模型"><a href="#1-3-两种对手模型" class="headerlink" title="1.3 两种对手模型"></a>1.3 两种对手模型</h3>

<p><strong>Oblivious adversary</strong>（预先固定的对手）在第 1 轮之前就选好整个序列 $\mathbf c_1,\ldots,\mathbf c_T$。学习者的随机选择不会改变这条序列。</p>

<p><strong>Adaptive adversary</strong>（自适应对手）可以根据此前的交互历史决定下一轮损失。它不需要预先承诺一条完整序列；学习者刚刚探测到某个变化，对手就可以针对这个动作改变后续损失。</p>

<p>这一区分在普通的单一遗憾指标下有时不显眼，但在同时控制两个指标时会直接改变可行性。全文的不可行性定理针对 adaptive 对手，可行性定理针对 oblivious 对手。</p>

<h2 id="2-两个遗憾基准"><a href="#2-两个遗憾基准" class="headerlink" title="2. 两个遗憾基准"></a>2. 两个遗憾基准</h2>

<h3 id="2-1-静态遗憾"><a href="#2-1-静态遗憾" class="headerlink" title="2.1 静态遗憾"></a>2.1 静态遗憾</h3>

<p>先固定一个动作 $a$，假设从头到尾都选择它。学习者的累计损失是 $\sum_{t=1}^{T}c_t(I_t)$，固定动作 $a$ 的累计损失是 $\sum_{t=1}^{T}c_t(a)$。相对于这个固定动作的静态遗憾定义为</p>

<div class="math-block">
\[
\operatorname{SReg}(a):=
\sum_{t=1}^{T}c_t(I_t)-\sum_{t=1}^{T}c_t(a).
\]
</div>

<p>通常还会取最坏的固定动作：</p>

<div class="math-block">
\[
\operatorname{SReg}:=\max_{a\in[A]}\operatorname{SReg}(a).
\]
</div>

<p>这个基准只允许比较者选一次动作。它适合这样的环境：损失序列总体上没有太多变化，或者我们关心的是“这段时间里一直坚持某个动作”能否被学习者击败。</p>

<h3 id="2-2-动态遗憾"><a href="#2-2-动态遗憾" class="headerlink" title="2.2 动态遗憾"></a>2.2 动态遗憾</h3>

<p>动态基准在每一轮都可以重新选择动作，因此第 $t$ 轮的比较损失是当轮最小值 $\min_{a\in[A]}c_t(a)$：</p>

<div class="math-block">
\[
\operatorname{DReg}:=
\sum_{t=1}^{T}c_t(I_t)-\sum_{t=1}^{T}\min_{a\in[A]}c_t(a).
\]
</div>

<p>动态基准不是一条事先给定的动作序列，而是每轮都取当时最好的动作。它因此比静态基准更强：如果固定动作 $a$ 在某些轮不是最优，动态基准会把这些轮的差距全部扣掉。</p>

<p>两种遗憾之间有一个之后反复使用的恒等式。对任意固定动作 $a$，将静态遗憾的定义加减 $\sum_t c_t(a)$，得到</p>

<div class="math-block">
\[
\begin{aligned}
\operatorname{DReg}
&=\sum_{t=1}^{T}c_t(I_t)-\sum_{t=1}^{T}c_t(a)
  +\sum_{t=1}^{T}\bigl(c_t(a)-\min_{b\in[A]}c_t(b)\bigr)\\
&=\operatorname{SReg}(a)+
  \sum_{t=1}^{T}\bigl(c_t(a)-\min_{b\in[A]}c_t(b)\bigr).
\end{aligned}
\]
</div>

<p>第二项永远非负，它衡量固定动作 $a$ 因为没有随时间切换而错过的机会。动态遗憾要小，既要学习一个好的固定动作，也要尽快发现“另一个动作刚刚变好了”。</p>

<h3 id="2-3-基准的最优量级"><a href="#2-3-基准的最优量级" class="headerlink" title="2.3 基准的最优量级"></a>2.3 基准的最优量级</h3>

<p>在 $A$ 个动作、$T$ 轮的对抗性 Bandit 中，忽略对数因子后，静态遗憾的标准最优量级是</p>

<div class="math-block">
\[
\widetilde O\bigl(\sqrt{AT}\bigr).
\]
</div>

<p>如果损失向量最多有 $S-1$ 次切换，动态遗憾的标准最优量级是</p>

<div class="math-block">
\[
\widetilde O\bigl(\sqrt{SAT}\bigr).
\]
</div>

<p>$\widetilde O(\cdot)$ 表示把对数因子暂时隐藏起来。例如 $\widetilde O(\sqrt{AT})$ 可能具体包含 $\sqrt{AT\log T}$ 或同阶的对数修正。这里真正要比较的是 $A,T,S$ 的幂次。</p>

<p>当 $S=1$ 时，动态问题退化到静态问题的量级；当 $S$ 增大，动态基准变化得更快，学习者必须付出额外的 $\sqrt S$ 因子。</p>

<h2 id="3-为什么一个参数不够"><a href="#3-为什么一个参数不够" class="headerlink" title="3. 为什么一个参数不够"></a>3. 为什么一个参数不够</h2>

<h3 id="3-1-探索和遗忘"><a href="#3-1-探索和遗忘" class="headerlink" title="3.1 探索和遗忘"></a>3.1 探索和遗忘</h3>

<p>静态遗憾算法倾向于逐渐集中到当前看起来最好的臂，不会频繁遗忘历史。这样可以减少探索成本，并在某个固定臂上快速积累优势。</p>

<p>动态遗憾算法必须保留重新探索的能力。一个很久没有被选择的臂可能在某个时刻变好；如果它几乎没有被抽样，学习者就要等很久才会发现。EXP3.S 等算法通过更大的探索和遗忘机制追踪变化。</p>

<p>论文把这个冲突写成学习率和探索率的尺度。对 EXP3.S 类型的算法，静态目标大致要求</p>

<div class="math-block">
\[
\eta_{\mathrm{stat}},\ \gamma_{\mathrm{stat}}
\asymp \sqrt{\frac{A}{T}},
\]
</div>

<p>而动态目标大致要求</p>

<div class="math-block">
\[
\eta_{\mathrm{dyn}},\ \gamma_{\mathrm{dyn}}
\asymp \sqrt{\frac{SA}{T}}.
\]
</div>

<p>$\eta$ 是更新步长，$\gamma$ 是向所有动作分配的探索量；符号 $x\asymp y$ 表示二者相差一个与 $A,S,T$ 无关的常数阶。因为 $S\geq1$，动态目标要求的探索率更大。</p>

<p>探索率太小，变化后的好臂很久得不到观测，动态遗憾变大；探索率太大，学习者在原本不好的臂上浪费太多轮，静态遗憾变大。问题不是简单地在两个常数之间取平均，而是要让算法根据已经观察到的结构选择不同的策略。</p>

<h3 id="3-2-同时最优的问题"><a href="#3-2-同时最优的问题" class="headerlink" title="3.2 同时最优的问题"></a>3.2 同时最优的问题</h3>

<p>论文研究的精确问题可以写成：</p>

<p>是否存在一个算法，使得</p>

<div class="math-block">
\[
\operatorname{SReg}=\widetilde O(\sqrt{AT}),\qquad
\operatorname{DReg}=\widetilde O(\sqrt{SAT})
\]
</div>

<p>并且这两个界在同一条运行轨迹上同时成立？</p>

<p>“同时”表示同一个算法、同一组反馈、同一个对手序列，而不是先知道环境类型再选择两套算法。此前的 bandit-over-bandit 方法可以在两种目标之间做模型选择，但静态遗憾会出现 $\widetilde O(A^{1/4}T^{3/4})$ 这样的次优项。论文的目标是判断这个额外的 $T^{1/4}$ 是否只是分析损失，还是问题本身的代价。</p>

<h2 id="4-自适应对手下的不可行性"><a href="#4-自适应对手下的不可行性" class="headerlink" title="4. 自适应对手下的不可行性"></a>4. 自适应对手下的不可行性</h2>

<h3 id="4-1-定理的说法"><a href="#4-1-定理的说法" class="headerlink" title="4.1 定理的说法"></a>4.1 定理的说法</h3>

<p>论文的 Theorem 2 给出一个乘积型下界。对任意 $S\geq2$，存在一个最多切换 $S$ 次的自适应对手，使任意算法都满足：对任意满足</p>

<div class="math-block">
\[
\max\{S^{\alpha},S^{1-\alpha}\}\leq\sqrt T
\]
</div>

<p>的实数 $\alpha$，要么</p>

<div class="math-block">
\[
\operatorname{SReg}\geq\Omega\bigl(S^{\alpha}\sqrt T\bigr)
\]
</div>

<p>或者，</p>

<div class="math-block">
\[
\operatorname{DReg}\geq\Omega\bigl(S^{1-\alpha}\sqrt T\bigr).
\]
</div>

<p>$\Omega(g)$ 表示至少有常数倍的 $g$ 这么大。这个定理不是说两种遗憾都会同时达到下界，而是说算法不可能把两者都压到各自希望的尺度。令 $\alpha=0$ 或 $1$ 附近，就能看见一边的目标越激进，另一边必须付出的代价越明显。</p>

<p>更紧凑的写法是</p>

<div class="math-block">
\[
\operatorname{SReg}\cdot\operatorname{DReg}
\geq\Omega(S\,T).
\]
</div>

<p>如果静态遗憾已经做到 $O(\sqrt T)$，乘积下界就迫使动态遗憾至少达到 $\Omega(S\sqrt T)$；如果动态遗憾做到 $O(\sqrt{ST})$，静态遗憾至少达到 $\Omega(\sqrt{ST})$。这正好说明了为什么不能在 adaptive 对手下同时达到两个标准量级。</p>

<h3 id="4-2-两臂分段构造"><a href="#4-2-两臂分段构造" class="headerlink" title="4.2 两臂分段构造"></a>4.2 两臂分段构造</h3>

<p>证明只需要两个动作。把动作记作 $1$ 和 $2$：</p>

<ul>
<li>动作 $1$ 的损失永远是 $1/2$；</li>
<li>动作 $2$ 的默认损失是 $1$，但在某些短区间内可以变成 $0$。</li>
</ul>

<p>把总时间划分成 $S/2$ 个 epoch，每个 epoch 长度为 $2T/S$。在第 $k$ 个 epoch 中，先看学习者在整个 epoch 里抽取动作 $2$ 的期望次数。</p>

<p>选两个稍后确定的整数 $N,M$。每个 epoch 有两个分支：</p>

<ol>
<li>如果动作 $2$ 的期望抽取次数至少为 $N$，对手让动作 $2$ 在整个 epoch 保持损失 $1$；</li>
<li>如果期望抽取次数小于 $N$，对手在该 epoch 内找一个长度为 $M$ 的窗口，把动作 $2$ 的损失改成 $0$。</li>
</ol>

<p>第二个分支的窗口选在“动作 $2$ 的期望抽取次数最少”的位置。到这里为止，对手只使用了学习者算法本身的分布，不需要看到真实的随机抽样，因此这部分仍然可以预先固定。</p>

<h3 id="4-3-高抽样分支带来静态遗憾"><a href="#4-3-高抽样分支带来静态遗憾" class="headerlink" title="4.3 高抽样分支带来静态遗憾"></a>4.3 高抽样分支带来静态遗憾</h3>

<p>在第一个分支中，动作 $2$ 的损失比动作 $1$ 高 $1/2$。每抽取一次动作 $2$，相对于固定动作 $1$ 就多付出 $1/2$。因为该 epoch 的期望抽取次数至少为 $N$，所以该 epoch 的期望静态遗憾至少为</p>

<div class="math-block">
\[
\mathbb E[\operatorname{SReg}_k(1)]\geq\frac N2.
\]
</div>

<p>这里的 $\operatorname{SReg}_k(1)$ 表示只计算第 $k$ 个 epoch，并以动作 $1$ 为固定基准。若有很多 epoch 都属于这个分支，静态遗憾就会累积起来。</p>

<h3 id="4-4-低抽样分支带来动态遗憾"><a href="#4-4-低抽样分支带来动态遗憾" class="headerlink" title="4.4 低抽样分支带来动态遗憾"></a>4.4 低抽样分支带来动态遗憾</h3>

<p>现在考虑第二个分支。整个 epoch 的长度是 $2T/S$，动作 $2$ 的期望抽取次数小于 $N$。为避免边界上的重复计数，先按论文证明取 $M$ 能整除 epoch 长度，把一个 epoch 划分成 $2T/(SM)$ 个互不重叠的窗口。于是至少有一个窗口的期望抽取次数不超过这些窗口的平均值：</p>

<p>记 $N_2(W)$ 为窗口 $W$ 内抽到动作 $2$ 的次数。则至少有一个窗口满足</p>

<div class="math-block">
\[
\mathbb E[N_2(W)]
\leq \frac{N}{2T/(SM)}=\frac{SMN}{2T}.
\]
</div>

<p>论文选择 $M,N$ 满足</p>

<div class="math-block">
\[
MN=\frac TS.
\]
</div>

<p>于是上式右侧为 $1/2$。由 Markov 不等式，窗口内至少抽到一次动作 $2$ 的概率不超过 $1/2$，因此完全没有抽到动作 $2$ 的概率至少为 $1/2$。</p>

<p>接下来才使用自适应性：如果学习者在窗口内抽到动作 $2$，对手立刻把动作 $2$ 的损失改回 $1$。所以在“没有抽到动作 $2$”的事件上，动作 $2$ 在整个窗口保持损失 $0$，而学习者一直选择动作 $1$，每轮比动态基准多付出 $1/2$。该 epoch 的动态遗憾期望至少为</p>

<div class="math-block">
\[
\mathbb E[\operatorname{DReg}_k]\geq
\frac12\times\frac M2=\frac M4.
\]
</div>

<p>前面的 $1/2$ 来自“窗口内完全没有探测”的概率，后面的 $M/2$ 来自窗口长度乘以每轮损失差。</p>

<h3 id="4-5-两种分支总有一种占多数"><a href="#4-5-两种分支总有一种占多数" class="headerlink" title="4.5 两种分支总有一种占多数"></a>4.5 两种分支总有一种占多数</h3>

<p>设 $E$ 是属于第一个分支的 epoch 数。总共有 $S/2$ 个 epoch：</p>

<p>若 $E>3S/8$，高抽样分支贡献至少 $EN/2$；低抽样分支至多带来每个 epoch $1/2$ 的负静态遗憾，而此时低抽样分支少于 $S/8$ 个。因此总静态遗憾仍满足
$\operatorname{SReg}\geq EN/2-(S/8)/2\geq SN/8$，其中最后一步使用定理条件推出 $N\geq1$。若 $E\leq3S/8$，第二个分支至少出现 $S/8$ 次，于是动态遗憾至少为常数倍的 $SM$。因此下面两种情况至少有一种成立：</p>

<div class="math-block">
\[
\operatorname{SReg}=\Omega(SN)
\]
</div>

<p>或者，</p>

<div class="math-block">
\[
\operatorname{DReg}=\Omega(SM).
\]
</div>

<p>最后取</p>

<div class="math-block">
\[
N=\frac{\sqrt T}{S^{1-\alpha}},
\qquad
M=\frac{\sqrt T}{S^{\alpha}}.
\]
</div>

<p>两者的乘积正好是 $T/S$。代入上一行便得到</p>

<div class="math-block">
\[
SN=S^{\alpha}\sqrt T,
\qquad
SM=S^{1-\alpha}\sqrt T.
\]
</div>

<p>这就完成了下界的尺度计算。构造的关键并不是动作数很多，而是对手可以在学习者“看见好机会”的瞬间撤销这个机会。学习者为了防止动态遗憾，必须探测；但一旦探测，负静态遗憾又被对手抹掉，只留下探索成本。</p>

<h2 id="5-oblivious-对手留下的缝隙"><a href="#5-oblivious-对手留下的缝隙" class="headerlink" title="5. oblivious 对手留下的缝隙"></a>5. oblivious 对手留下的缝隙</h2>

<h3 id="5-1-撤回动作为什么改变结论"><a href="#5-1-撤回动作为什么改变结论" class="headerlink" title="5.1 撤回动作为什么改变结论"></a>5.1 撤回动作为什么改变结论</h3>

<p>自适应构造中，动作 $2$ 变好后，学习者一旦抽到它，对手就把损失改回去。若对手必须在游戏开始前固定所有损失，就做不到这一步。动作 $2$ 在窗口中保持为 $0$，学习者一旦探测到它，就可以继续选择它并获得真实收益。</p>

<p>这会带来一个对静态目标有利的量：在动作 $2$ 变好但动作 $1$ 仍为 $1/2$ 的窗口里，每选择一次动作 $2$，相对于固定动作 $1$ 的静态遗憾减少 $1/2$。这部分是<strong>负静态遗憾</strong>。探索不再只是成本，也可能发现一个足够好的区间来抵消此前的探索成本。</p>

<h3 id="5-2-warm-up-环境"><a href="#5-2-warm-up-环境" class="headerlink" title="5.2 warm-up 环境"></a>5.2 warm-up 环境</h3>

<p>论文先在一个简化环境中展示这个抵消关系。令 $K=\sqrt T$，把时间划分成 $K$ 个 epoch，每个 epoch 的长度为 $\sqrt T$。第 $k$ 个 epoch 记为</p>

<div class="math-block">
\[
\mathcal I_k=[(k-1)\sqrt T+1,\,k\sqrt T].
\]
</div>

<p>对手在其中选一个子区间 $\mathcal J_k\subseteq\mathcal I_k$，满足 $|\mathcal J_k|\leq\sqrt T/2$，并设置</p>

<div class="math-block">
\[
\mathbf c_t=
\begin{cases}
(1/2,0),&t\in\mathcal J_k,\\
(1/2,1),&t\in\mathcal I_k\setminus\mathcal J_k.
\end{cases}
\]
</div>

<p>动作 $1$ 在所有轮的损失都是 $1/2$。由于好区间长度不超过整个 epoch 的一半，动作 $1$ 的总损失不大于动作 $2$ 的总损失，所以动作 $1$ 是全局最优固定动作。</p>

<h3 id="5-3-两种基础策略"><a href="#5-3-两种基础策略" class="headerlink" title="5.3 两种基础策略"></a>5.3 两种基础策略</h3>

<p>在每个 epoch 开始时，学习者在两种基础策略之间选择：</p>

<ol>
<li>$\mathsf X$ 策略：整个 epoch 始终选择动作 $1$；</li>
<li>$\mathsf O$ 策略：每轮以 $\epsilon=T^{-1/4}$ 的概率探索动作 $2$，以 $1-\epsilon$ 的概率选择动作 $1$。一旦观测到动作 $2$ 的损失为 $0$，就继续选择动作 $2$，直到它的损失恢复为 $1$。</li>
</ol>

<p>字母 $\mathsf X$ 和 $\mathsf O$ 只是两种策略的标签，不是乘法符号或集合运算。令 $L_k:=|\mathcal J_k|$，并令 $\widehat L_k$ 表示在好区间内实际选择动作 $2$ 的轮数。</p>

<p>若使用 $\mathsf X$，动作 $1$ 始终被选择，因此</p>

<div class="math-block">
\[
\operatorname{SReg}_{k,\mathsf X}=0,
\qquad
\operatorname{DReg}_{k,\mathsf X}=\frac12L_k.
\tag{1}
\]
</div>

<p>$\mathsf X$ 的静态遗憾为零，但在好区间中错过了动态基准，所以动态遗憾是 $L_k/2$。</p>

<p>若使用 $\mathsf O$，探索动作 $2$ 的期望次数约为 $\epsilon\sqrt T=T^{1/4}$。好区间中每次成功选择动作 $2$，相对于固定动作 $1$ 贡献 $-1/2$。论文的直接计算给出</p>

<div class="math-block">
\[
\begin{aligned}
\mathbb E[\operatorname{SReg}_{k,\mathsf O}]&\leq
\frac12T^{1/4}-\frac12\mathbb E[\widehat L_k]+\frac12,\\
\mathbb E[\operatorname{DReg}_{k,\mathsf O}]&\leq
\frac12T^{1/4}+\frac12\bigl(L_k-\mathbb E[\widehat L_k]+1\bigr).
\end{aligned}
\tag{2}
\]
</div>

<p>第一行的三项分别是探索成本、在好区间内抽到动作 $2$ 带来的负遗憾，以及切换时刻的常数项。第二行把动态遗憾拆成探索和“还没有发现好区间”的等待成本。</p>

<p>在好区间内，动作 $2$ 每轮以概率 $\epsilon$ 被探索。第一次成功前的失败次数服从几何等待过程，因此</p>

<div class="math-block">
\[
L_k-\mathbb E[\widehat L_k]+1\leq\epsilon^{-1}=T^{1/4}.
\]
</div>

<p>把它代回 (2)，得到更容易使用的形式：</p>

<div class="math-block">
\[
\mathbb E[\operatorname{SReg}_{k,\mathsf O}]\leq
T^{1/4}-\frac12L_k,
\qquad
\mathbb E[\operatorname{DReg}_{k,\mathsf O}]\leq T^{1/4}.
\tag{3}
\]
</div>

<p>这两条式子已经显示出策略之间的取舍：$\mathsf X$ 的动态成本随 $L_k$ 增大，而 $\mathsf O$ 在 $L_k$ 较大时得到更多负静态遗憾。</p>

<h3 id="5-4-把两个目标放进一个向量"><a href="#5-4-把两个目标放进一个向量" class="headerlink" title="5.4 把两个目标放进一个向量"></a>5.4 把两个目标放进一个向量</h3>

<p>两个遗憾的量纲不同：静态遗憾的目标尺度是 $\sqrt T$，warm-up 中动态遗憾的目标尺度是 $T^{3/4}$。论文先把两个坐标缩放到同一个 $O(T)$ 的总预算里：</p>

<div class="math-block">
\[
\ell_{k,\mathsf X}:=
\begin{bmatrix}
\sqrt T\cdot\overline{\operatorname{SReg}}_{k,\mathsf X}\\
T^{1/4}\cdot\overline{\operatorname{DReg}}_{k,\mathsf X}
\end{bmatrix}
=
\begin{bmatrix}
0\\
\frac12T^{1/4}L_k
\end{bmatrix},
\]
</div>

<div class="math-block">
\[
\ell_{k,\mathsf O}:=
\begin{bmatrix}
\sqrt T\cdot\overline{\operatorname{SReg}}_{k,\mathsf O}\\
T^{1/4}\cdot\overline{\operatorname{DReg}}_{k,\mathsf O}
\end{bmatrix}
=
\begin{bmatrix}
T^{3/4}-\frac12\sqrt T L_k\\
\sqrt T
\end{bmatrix}.
\tag{4}
\]
</div>

<p>上横线表示使用 (1) 和 (3) 得到的遗憾上界，而不是新的随机变量。第一个坐标乘 $\sqrt T$，因为最后要把它控制在 $O(T)$；第二个坐标乘 $T^{1/4}$，同样是为了把 $T^{3/4}$ 级别的动态遗憾变成 $T$ 级预算。</p>

<p>如果 $L_k$ 较小，选择 $\mathsf X$：动态坐标不大；如果 $L_k$ 较大，选择 $\mathsf O$：第一坐标中的负项 $-\frac12\sqrt T L_k$ 可以抵消 $T^{3/4}$。在知道 $L_k$ 的理想情况下，阈值选择</p>

<div class="math-block">
\[
(p_{k,\mathsf X},p_{k,\mathsf O})=
\bigl(\mathbb I\{L_k\leq2T^{1/4}\},\,\mathbb I\{L_k>2T^{1/4}\}\bigr)
\]
</div>

<p>可以让每个 epoch 的加权向量满足</p>

<div class="math-block">
\[
p_{k,\mathsf X}\ell_{k,\mathsf X}
+p_{k,\mathsf O}\ell_{k,\mathsf O}
\leq
\begin{bmatrix}0\\\sqrt T\end{bmatrix}.
\tag{5}
\]
</div>

<p>学习者当然不能在 epoch 开始前直接看到 $L_k$。因此问题变成：如何在不知道哪种策略更合适时，仍然实现类似 (5) 的长期平均效果？这正是 Blackwell approachability 出现的位置。</p>

<h2 id="6-Blackwell-approachability-如何调节两种策略"><a href="#6-Blackwell-approachability-如何调节两种策略" class="headerlink" title="6. Blackwell approachability 如何调节两种策略"></a>6. Blackwell approachability 如何调节两种策略</h2>

<h3 id="6-1-从标量遗憾到向量遗憾"><a href="#6-1-从标量遗憾到向量遗憾" class="headerlink" title="6.1 从标量遗憾到向量遗憾"></a>6.1 从标量遗憾到向量遗憾</h3>

<p>Blackwell approachability 处理的是向量收益。这里每个 epoch 产生一个二维向量：第一个坐标对应静态遗憾，第二个坐标对应动态遗憾。选择 $\mathsf X$ 或 $\mathsf O$ 就像在两个向量之间作一次选择。</p>

<p>令 $p_{k,\mathsf X}$ 和 $p_{k,\mathsf O}$ 是第 $k$ 个 epoch 选择两种策略的概率，满足</p>

<div class="math-block">
\[
p_{k,\mathsf X}+p_{k,\mathsf O}=1,
\qquad
p_{k,\mathsf X},p_{k,\mathsf O}\geq0.
\]
</div>

<p>记 $\Delta_2:=\{(p_1,p_2):p_1+p_2=1,\ p_1,p_2\geq0\}$ 为二维概率单纯形。理想目标是控制</p>

<div class="math-block">
\[
\sum_{k=1}^{K}
\bigl(p_{k,\mathsf X}\ell_{k,\mathsf X}
 +p_{k,\mathsf O}\ell_{k,\mathsf O}\bigr)
\]
</div>

<p>的两个坐标。关键困难在于 $\ell_{k,\mathsf X}$、$\ell_{k,\mathsf O}$ 取决于隐藏的 $L_k$，而 $L_k$ 只有执行探索策略并观察到损失后才能估计。</p>

<h3 id="6-2-用一个元学习问题表示目标"><a href="#6-2-用一个元学习问题表示目标" class="headerlink" title="6.2 用一个元学习问题表示目标"></a>6.2 用一个元学习问题表示目标</h3>

<p>在第 $k$ 个 epoch 开始时，元算法维护一个权重向量 $\theta_k\in\Delta_2$。它的两个坐标分别表示当前更关心 $\mathsf X$ 方向还是 $\mathsf O$ 方向。</p>

<p>假设我们能构造一个观测向量 $\widehat v_k\in\mathbb R^2$，并使用一个在线学习算法保证，对每个比较坐标 $a\in\{1,2\}$ 都有</p>

<div class="math-block">
\[
\sum_{k=1}^{K}
\bigl(\widehat v_k(a)-\theta_k^{\mathsf T}\widehat v_k\bigr)
\leq \mathcal R(a).
\tag{6}
\]
</div>

<p>这里 $\theta_k^{\mathsf T}\widehat v_k$ 是内积：</p>

<div class="math-block">
\[
\theta_k^{\mathsf T}\widehat v_k
=\theta_k(1)\widehat v_k(1)+\theta_k(2)\widehat v_k(2).
\]
</div>

<p>把 (6) 移项，就得到</p>

<div class="math-block">
\[
\sum_{k=1}^{K}\widehat v_k(a)
\leq
\mathcal R(a)+\sum_{k=1}^{K}\theta_k^{\mathsf T}\widehat v_k.
\tag{7}
\]
</div>

<p>所以只要控制两件事，两个遗憾目标就会被控制：一是元算法自己的比较遗憾 $\mathcal R(a)$，二是每轮内积 $\theta_k^{\mathsf T}\widehat v_k$ 的总和。后者正是策略概率需要解决的局部问题。</p>

<h3 id="6-3-让未知的好区间项相互抵消"><a href="#6-3-让未知的好区间项相互抵消" class="headerlink" title="6.3 让未知的好区间项相互抵消"></a>6.3 让未知的好区间项相互抵消</h3>

<p>为了展示取消机制，先回到 warm-up。令 $u\in(0,1/3]$ 是一个松弛常数。使用带估计的两个向量时，$L_k$ 的相关项在 $\theta_k^{\mathsf T}\widehat v_k$ 中具有如下系数：</p>

<div class="math-block">
\[
p_{k,\mathsf X}\theta_k(2)\frac12T^{1/4}
-p_{k,\mathsf O}\theta_k(1)\frac12u\sqrt T.
\]
</div>

<p>我们不需要知道 $L_k$ 的具体值，只要令这个系数为零：</p>

<div class="math-block">
\[
p_{k,\mathsf X}\theta_k(2)\frac12T^{1/4}
=p_{k,\mathsf O}\theta_k(1)\frac12u\sqrt T.
\]
</div>

<p>再加上 $p_{k,\mathsf X}+p_{k,\mathsf O}=1$，逐步解出概率：</p>

<div class="math-block">
\[
\boxed{
(p_{k,\mathsf X},p_{k,\mathsf O})
=
\frac{\bigl(u\theta_k(1)\sqrt T,\;\theta_k(2)T^{1/4}\bigr)}
{u\theta_k(1)\sqrt T+\theta_k(2)T^{1/4}}
}.
\tag{8}
\]
</div>

<p>分子第一项对应 $p_{k,\mathsf X}$，第二项对应 $p_{k,\mathsf O}$。这不是凭经验调一个探索率，而是让两个策略对未知区间长度 $L_k$ 的一阶影响精确抵消。</p>

<p>把 (8) 代回剩余项后，论文得到</p>

<div class="math-block">
\[
\theta_k^{\mathsf T}\widehat v_k
\leq
\frac{\mathbb I\{i_k=\mathsf O\}}{p_{k,\mathsf O}}
\left(\frac{\sqrt T}{u}+\sqrt T\right),
\tag{9}
\]
</div>

<p>其中 $i_k$ 是实际抽到的策略。因为 $i_k=\mathsf O$ 的概率正好是 $p_{k,\mathsf O}$，取期望后逆概率因子抵消：</p>

<div class="math-block">
\[
\mathbb E\left[
\frac{\mathbb I\{i_k=\mathsf O\}}{p_{k,\mathsf O}}
\right]=1.
\]
</div>

<p>这就是 inverse-probability weighting 在这里的作用：只在真的执行探索策略时观测 $L_k$，但在期望意义下恢复“如果每个 epoch 都能看到该向量”的贡献。</p>

<h3 id="6-4-元算法的流程"><a href="#6-4-元算法的流程" class="headerlink" title="6.4 元算法的流程"></a>6.4 元算法的流程</h3>

<p>把 warm-up 中的思想写成流程，核心步骤如下：</p>

<pre><code>初始化 theta_1 到概率单纯形 Delta_2 中
对每个 epoch k:
    由 theta_k 和式 (8) 计算 p_(k,X), p_(k,O)
    按这两个概率抽取策略 i_k
    若 i_k = X，整段执行静态策略
    若 i_k = O，执行探索和变化检测，得到 L_hat_k
    用逆概率权重构造 v_hat_k
    用 Broad-OMD 根据 v_hat_k 更新 theta_k 到 theta_(k+1)</code></pre>

<p>Broad-OMD 是论文选用的在线镜像下降变体。$\psi(\theta)=\sum_a\log(1/\theta(a))$ 是对数障碍正则项，它防止概率坐标落到边界；二阶修正项用来处理逆概率估计带来的大幅度向量。这里的元算法只负责在两种基础策略之间分配概率，基础策略本身负责收集 Bandit 反馈。</p>

<h2 id="7-推广到一般的-A-臂问题"><a href="#7-推广到一般的-A-臂问题" class="headerlink" title="7. 推广到一般的 A 臂问题"></a>7. 推广到一般的 $A$ 臂问题</h2>

<h3 id="7-1-为什么要分成多个epoch"><a href="#7-1-为什么要分成多个epoch" class="headerlink" title="7.1 为什么要分成多个 epoch"></a>7.1 为什么要分成多个 epoch</h3>

<p>在一般 $A$ 臂设置中，论文把总时间划分为</p>

<div class="math-block">
\[
K=\sqrt{\frac TA}
\]
</div>

<p>个 epoch，每个 epoch 长度约为 $\sqrt{AT}$。这个长度让“每个动作至少获得一次信息”和“变化检测的等待成本”处于同一个分析尺度。</p>

<p>现在不能只比较动作 $1$。静态遗憾要分别相对于 $A$ 个动作控制，动态遗憾再多一个目标，因此需要同时控制 $A+1$ 个坐标：</p>

<div class="math-block">
\[
\bigl(\operatorname{SReg}(1),\ldots,\operatorname{SReg}(A),\operatorname{DReg}\bigr).
\]
</div>

<p>论文为每个 epoch 仍然保留两种基础策略：$\mathsf X$ 侧重静态遗憾，使用 EXP3-IX 型估计器；$\mathsf O$ 侧重动态遗憾，运行 ExpEst 过程检测 epoch 内的损失变化，并估计每个动作在变化区间中的收益长度。</p>

<h3 id="7-2-一般算法的关键参数"><a href="#7-2-一般算法的关键参数" class="headerlink" title="7.2 一般算法的关键参数"></a>7.2 一般算法的关键参数</h3>

<p>令 $\delta\in(0,1)$ 为失败概率，论文记</p>

<div class="math-block">
\[
\iota:=\log\frac{AT}{\delta},
\qquad
\gamma:=\frac{4\iota}{\sqrt{AT}},
\qquad
\rho:=4\iota\sqrt{\frac{SA}{T}}.
\]
</div>

<p>$\gamma$ 控制 $\mathsf X$ 策略中的最低探索量，$\rho$ 控制 $\mathsf O$ 策略中的变化检测强度。算法还要求 $\rho\leq1/2$；当 $T$ 足够大时这是一个正常的参数范围条件。</p>

<p>元算法的权重位于 $A+1$ 维单纯形</p>

<div class="math-block">
\[
\Delta_{A+1}:=\left\{\theta\in\mathbb R^{A+1}:\theta(j)\geq0,\ \sum_{j=1}^{A+1}\theta(j)=1\right\}.
\]
</div>

<p>为了避免动态坐标吞掉全部概率，论文进一步限制可行集合：</p>

<div class="math-block">
\[
\Theta:=\left\{\theta\in\Delta_{A+1}:
1-\theta(A+1)=\sum_{a=1}^{A}\theta(a)\geq A\gamma\right\}.
\]
</div>

<p>前 $A$ 个坐标对应 $A$ 个静态比较者，第 $A+1$ 个坐标对应动态遗憾。这个约束保证前 $A$ 个动作总体仍有足够概率被抽样，逆概率估计不会失控。</p>

<h3 id="7-3-主定理"><a href="#7-3-主定理" class="headerlink" title="7.3 主定理"></a>7.3 主定理</h3>

<p>在确定性损失、oblivious 对手、损失向量最多变化 $S-1$ 次且 $S$ 已知的条件下，论文的 Theorem 4 给出：以至少 $1-O(\delta)$ 的概率，同时对所有 $a\in[A]$ 有</p>

<div class="math-block">
\[
\operatorname{SReg}(a)
\leq O\left(\sqrt{AT}\log\frac{T}{\delta}\right),
\]
</div>

<p>并且</p>

<div class="math-block">
\[
\operatorname{DReg}
\leq O\left(\sqrt{SAT}\log\frac{T}{\delta}\right).
\]
</div>

<p>这里“同时”体现在同一个高概率事件上：不是分别为静态目标和动态目标抽取两次随机性，而是一次运行同时满足所有静态比较者和动态比较者的界。</p>

<h3 id="7-4-确定性假设究竟用在哪里"><a href="#7-4-确定性假设究竟用在哪里" class="headerlink" title="7.4 确定性假设究竟用在哪里"></a>7.4 确定性假设究竟用在哪里</h3>

<p>确定性反馈不是为了让符号更简单，而是进入了静态遗憾的精确抵消。$\mathsf O$ 策略需要判断“动作的损失是否已经变好”，然后把探索切换为利用。如果反馈含有噪声，一次观测到的小损失可能只是噪声，变化检测就会产生额外延迟或误报。</p>

<p>论文指出，在带噪声的情况下，变化检测的额外成本会进入静态遗憾上界，破坏“正的探索成本”和“负的好区间收益”之间的精确平衡。因此当前定理并没有声称随机损失下也能保持同样的同时最优；确定性正是算法能够闭合证明的地方。</p>

<h2 id="8-这篇论文把问题推进到哪里"><a href="#8-这篇论文把问题推进到哪里" class="headerlink" title="8. 这篇论文把问题推进到哪里"></a>8. 这篇论文把问题推进到哪里</h2>

<h3 id="8-1-已经解决的边界"><a href="#8-1-已经解决的边界" class="headerlink" title="8.1 已经解决的边界"></a>8.1 已经解决的边界</h3>

<p>论文完成了一个清晰的边界刻画：</p>

<ul>
<li>对 adaptive 对手，即使损失是确定性的、切换数 $S$ 已知，同时达到静态和动态最优阶仍然不可能；</li>
<li>对 oblivious 对手，在确定性损失和已知 $S$ 的条件下，可以同时达到 $\widetilde O(\sqrt{AT})$ 与 $\widetilde O(\sqrt{SAT})$；</li>
<li>可行性的关键不是把两个算法机械拼接，而是利用探索发现好区间后产生的负静态遗憾，并用向量在线学习调节两种策略。</li>
</ul>

<h3 id="8-2-论文明确留下的方向"><a href="#8-2-论文明确留下的方向" class="headerlink" title="8.2 论文明确留下的方向"></a>8.2 论文明确留下的方向</h3>

<p>论文结尾把以下问题留作后续工作：</p>

<ol>
<li><strong>未知切换数。</strong> 当前算法需要提前知道 $S$ 才能设置 epoch 长度和检测强度。能否在不知道 $S$ 的情况下，仍然同时达到两个含 $S$ 的最优界？</li>
<li><strong>随机或带噪反馈。</strong> 当 $c_t(I_t)$ 还叠加观测噪声时，变化检测的误报和漏报会不会带来不可避免的额外静态遗憾？</li>
<li><strong>自适应对手下的替代目标。</strong> 同时最优被 Theorem 2 排除后，能否给出完整的 Pareto 前沿，描述静态遗憾和动态遗憾之间所有可达到的权衡？</li>
<li><strong>多个切换预算的 best-of-all-worlds。</strong> 论文只固定了一个 $S$。能否对不同切换次数的 switching benchmark 同时保持各自的最优遗憾，而不为每个 $S$ 单独运行一套算法？</li>
</ol>

<p>这些问题都保留了同一个核心：有限反馈下，学习者必须决定把多少预算花在发现变化上，又要保证长期固定比较者不会被探索拖垮。</p>

<h2 id="附录A-符号表与量级记号"><a href="#附录A-符号表与量级记号" class="headerlink" title="附录 A 符号表与量级记号"></a>附录 A：符号表与量级记号</h2>

<ul>
<li>$A$：动作（臂）的数量；$[A]=\{1,\ldots,A\}$。</li>
<li>$T$：交互轮数；$t$ 是轮编号。</li>
<li>$I_t$：第 $t$ 轮学习者实际选择的动作。</li>
<li>$c_t(a)$：第 $t$ 轮动作 $a$ 的损失；$\mathbf c_t$ 是完整损失向量。</li>
<li>$S$：损失向量段数，因此最多有 $S-1$ 次变化。</li>
<li>$\operatorname{SReg}(a)$：相对于固定动作 $a$ 的静态遗憾；$\operatorname{SReg}$ 是对所有固定动作取最大值。</li>
<li>$\operatorname{DReg}$：相对于每轮最佳动作的动态遗憾。</li>
<li>$\mathbb I\{E\}$：事件 $E$ 的示性函数。</li>
<li>$\mathbb E[\cdot]$：对学习者随机性取期望。</li>
<li>$\Omega(g)$：下界，表示至少为某个常数倍的 $g$；$O(g)$ 表示至多为某个常数倍的 $g$。</li>
<li>$\widetilde O(g)$：忽略对数因子后的上界。</li>
<li>$\Delta_d$：$d$ 维概率单纯形，坐标非负且坐标和为 $1$。</li>
<li>$K$：epoch 的总数；$H$：自适应下界构造中单个 epoch 的长度。</li>
<li>$M$：低抽样分支中“好窗口”的长度；$N$：判断高、低抽样分支的期望抽样次数阈值。</li>
<li>$\alpha$：下界权衡参数；$u$：混合概率计算中的松弛常数。</li>
<li>$\epsilon=T^{-1/4}$：warm-up 探索策略每轮尝试动作 $2$ 的概率；上横线 $\overline{\operatorname{SReg}}$、$\overline{\operatorname{DReg}}$ 表示采用前文上界后的量，不是新的随机变量。</li>
<li>$\eta$：元算法的更新步长；$\gamma$：静态侧的最低探索量；$\rho$：动态侧的变化检测强度；$\iota=\log(AT/\delta)$。</li>
<li>$\ell_{k,\mathsf X},\ell_{k,\mathsf O}$：第 $k$ 个 epoch 选择两种基础策略时的缩放遗憾向量；$\widehat v_k$：只能在实际执行策略后构造的向量估计。</li>
<li>$\mathcal R(a)$：元算法相对于比较坐标 $a$ 的在线学习遗憾；$\Theta$：元算法允许使用的受约束概率集合。</li>
<li>$\theta_k$：第 $k$ 个 epoch 的元算法权重；$\theta_k^{\mathsf T}v$ 是向量内积。</li>
<li>$p_{k,\mathsf X},p_{k,\mathsf O}$：第 $k$ 个 epoch 选择两种基础策略的概率。</li>
<li>$L_k$：warm-up 中第 $k$ 个好区间的长度；$\widehat L_k$：实际在好区间中选择好臂的次数。</li>
</ul>

<h2 id="附录B-自适应下界的完整计算"><a href="#附录B-自适应下界的完整计算" class="headerlink" title="附录 B 自适应下界的完整计算"></a>附录 B：自适应下界的完整计算</h2>

<h3 id="B-1-窗口的期望抽样次数"><a href="#B-1-窗口的期望抽样次数" class="headerlink" title="B.1 窗口的期望抽样次数"></a>B.1 窗口的期望抽样次数</h3>

<p>一个 epoch 长度为 $H:=2T/S$。按照论文证明，先假设 $H$ 能被 $M$ 整除。设该 epoch 中动作 $2$ 的总期望抽样次数小于 $N$，并把它划分为 $H/M$ 个互不重叠、长度均为 $M$ 的窗口 $W_1,\ldots,W_{H/M}$。窗口不重叠，所以它们的期望抽样次数之和不超过 $N$：</p>

<div class="math-block">
\[
\sum_{j=1}^{H/M}\mathbb E[N_2(W_j)]\lt N.
\]
</div>

<p>因此存在一个窗口 $W_\star$，满足</p>

<div class="math-block">
\[
\mathbb E[N_2(W_\star)]
\leq \frac{N}{H/M}
 =\frac{NM}{H}
 =\frac{SMN}{2T},
\]
</div>

<p>其中 $N_2(W_\star)$ 是窗口内抽到动作 $2$ 的次数。取 $MN=T/S$ 后，得到</p>

<div class="math-block">
\[
\mathbb E[N_2(W_\star)]\leq\frac12.
\]
</div>

<h3 id="B-2-从期望次数到不抽样事件"><a href="#B-2-从期望次数到不抽样事件" class="headerlink" title="B.2 从期望次数到不抽样事件"></a>B.2 从期望次数到不抽样事件</h3>

<p>令事件 $E_\star:=\{N_2(W_\star)\geq1\}$。因为 $N_2(W_\star)$ 是非负整数值随机变量，</p>

<div class="math-block">
\[
\mathbb E[N_2(W_\star)]
\geq 1\cdot\mathbb P(E_\star).
\]
</div>

<p>于是</p>

<div class="math-block">
\[
\mathbb P(E_\star)\leq\frac12,
\qquad
\mathbb P(N_2(W_\star)=0)\geq\frac12.
\]
</div>

<p>在 $N_2(W_\star)=0$ 的事件上，学习者每轮都选择动作 $1$，而对手把动作 $2$ 的损失固定为 $0$。动态基准每轮选择动作 $2$，所以窗口贡献 $M/2$。取事件概率后：</p>

<div class="math-block">
\[
\mathbb E[\operatorname{DReg}_k]
\geq\frac12\cdot\frac M2=\frac M4.
\]
</div>

<h3 id="B-3-两个分支的累积"><a href="#B-3-两个分支的累积" class="headerlink" title="B.3 两个分支的累积"></a>B.3 两个分支的累积</h3>

<p>共有 $S/2$ 个 epoch，设第一个分支的数量为 $E$。</p>

<p>若 $E>3S/8$，低抽样分支的数量为 $S/2-E\lt S/8$，每个这样的分支对静态遗憾的负贡献至多为 $1/2$。所以</p>

<div class="math-block">
\[
\mathbb E[\operatorname{SReg}]
\geq E\frac N2-\left(\frac S2-E\right)\frac12
>\frac{3SN}{16}-\frac S{16}
\geq\frac{SN}{8}
=\Omega(SN),
\qquad (N\geq1).
\]
</div>

<p>若 $E\leq3S/8$，第二个分支数量至少为 $S/2-3S/8=S/8$，则</p>

<div class="math-block">
\[
\mathbb E[\operatorname{DReg}]
\geq\frac S8\cdot\frac M4
=\frac{SM}{32}
=\Omega(SM).
\]
</div>

<p>取 $N=\sqrt T/S^{1-\alpha}$、$M=\sqrt T/S^\alpha$，便有 $MN=T/S$，并且</p>

<div class="math-block">
\[
SN=S^\alpha\sqrt T,
\qquad
SM=S^{1-\alpha}\sqrt T.
\]
</div>

<p>因此对任意满足定理条件的 $\alpha$，算法至少在两个坐标中的一个上承担相应下界。</p>

<h2 id="附录C-warm-up中的逐项推导"><a href="#附录C-warm-up中的逐项推导" class="headerlink" title="附录 C warm-up 中的逐项推导"></a>附录 C：warm-up 中的逐项推导</h2>

<h3 id="C-1-策略-X"><a href="#C-1-策略-X" class="headerlink" title="C.1 策略 X"></a>C.1 策略 $\mathsf X$</h3>

<p>策略 $\mathsf X$ 每轮选动作 $1$。由于动作 $1$ 同时也是静态基准，</p>

<div class="math-block">
\[
\operatorname{SReg}_{k,\mathsf X}
=\sum_{t\in\mathcal I_k}c_t(1)-\sum_{t\in\mathcal I_k}c_t(1)=0.
\]
</div>

<p>动态基准只在 $\mathcal J_k$ 中选择动作 $2$。这些轮动作 $1$ 的损失是 $1/2$，动态基准的损失是 $0$，因此</p>

<div class="math-block">
\[
\operatorname{DReg}_{k,\mathsf X}
=\sum_{t\in\mathcal J_k}\left(\frac12-0\right)
=\frac12L_k.
\]
</div>

<h3 id="C-2-策略-O的静态遗憾"><a href="#C-2-策略-O的静态遗憾" class="headerlink" title="C.2 策略 O 的静态遗憾"></a>C.2 策略 $\mathsf O$ 的静态遗憾</h3>

<p>先忽略切换时刻的常数项。在 epoch 长度 $\sqrt T$ 内，探索动作 $2$ 的期望次数是</p>

<div class="math-block">
\[
\epsilon\sqrt T=T^{-1/4}\sqrt T=T^{1/4}.
\]
</div>

<p>在好区间内，每一次选择动作 $2$，动作损失从动作 $1$ 的 $1/2$ 变为 $0$，所以静态遗憾减少 $1/2$。如果好区间内选择动作 $2$ 的次数为 $\widehat L_k$，则对应贡献为 $-\widehat L_k/2$。加上探索成本和至多 $1/2$ 的切换常数：</p>

<div class="math-block">
\[
\mathbb E[\operatorname{SReg}_{k,\mathsf O}]
\leq\frac12T^{1/4}-\frac12\mathbb E[\widehat L_k]+\frac12.
\]
</div>

<h3 id="C-3-策略-O的动态遗憾"><a href="#C-3-策略-O的动态遗憾" class="headerlink" title="C.3 策略 O 的动态遗憾"></a>C.3 策略 $\mathsf O$ 的动态遗憾</h3>

<p>动态遗憾与静态遗憾的差异，在 $\mathcal J_k$ 的每个时刻都是 $1/2$。因此</p>

<div class="math-block">
\[
\operatorname{DReg}_{k,\mathsf O}
=\operatorname{SReg}_{k,\mathsf O}+\frac12L_k.
\]
</div>

<p>将上一节的静态上界代入，得到</p>

<div class="math-block">
\[
\mathbb E[\operatorname{DReg}_{k,\mathsf O}]
\leq\frac12T^{1/4}
+\frac12\bigl(L_k-\mathbb E[\widehat L_k]+1\bigr).
\]
</div>

<p>在好区间开始前，动作 $2$ 每轮以概率 $\epsilon$ 被抽到。直到第一次成功为止，失败次数的期望不超过 $(1-\epsilon)/\epsilon\lt\epsilon^{-1}$。所以</p>

<div class="math-block">
\[
L_k-\mathbb E[\widehat L_k]+1\leq\epsilon^{-1}=T^{1/4}.
\]
</div>

<p>最终得到</p>

<div class="math-block">
\[
\mathbb E[\operatorname{DReg}_{k,\mathsf O}]\leq T^{1/4},
\qquad
\mathbb E[\operatorname{SReg}_{k,\mathsf O}]\leq T^{1/4}-\frac12L_k.
\]
</div>

<h2 id="附录D-概率公式的代数验证"><a href="#附录D-概率公式的代数验证" class="headerlink" title="附录 D 概率公式的代数验证"></a>附录 D：概率公式的代数验证</h2>

<p>从 (8) 出发，记</p>

<div class="math-block">
\[
D_k:=u\theta_k(1)\sqrt T+\theta_k(2)T^{1/4}.
\]
</div>

<p>则</p>

<div class="math-block">
\[
p_{k,\mathsf X}=\frac{u\theta_k(1)\sqrt T}{D_k},
\qquad
p_{k,\mathsf O}=\frac{\theta_k(2)T^{1/4}}{D_k}.
\]
</div>

<p>两者相加为</p>

<div class="math-block">
\[
p_{k,\mathsf X}+p_{k,\mathsf O}
=\frac{u\theta_k(1)\sqrt T+\theta_k(2)T^{1/4}}{D_k}=1.
\]
</div>

<p>再检查 $L_k$ 的系数：</p>

<div class="math-block">
\[
\begin{aligned}
p_{k,\mathsf X}\theta_k(2)\frac12T^{1/4}
&=\frac{u\theta_k(1)\theta_k(2)\sqrt T\,T^{1/4}}{2D_k},\\
p_{k,\mathsf O}\theta_k(1)\frac12u\sqrt T
&=\frac{u\theta_k(1)\theta_k(2)T^{1/4}\sqrt T}{2D_k}.
\end{aligned}
\]
</div>

<p>两项完全相同，因此在相减时为零。这一步是整个混合策略的代数核心：$L_k$ 可以未知，但它出现的系数能够在策略概率层面被消掉。</p>

<h2 id="附录E-参考文献"><a href="#附录E-参考文献" class="headerlink" title="附录 E 参考文献"></a>附录 E：参考文献</h2>

<ol>
<li>Qian, J. and Wei, C.-Y. (2026). <a href="https://arxiv.org/abs/2602.07418" target="_blank" rel="noopener"><em>Achieving Optimal Static and Dynamic Regret Simultaneously in Bandits with Deterministic Losses</em></a>. arXiv:2602.07418. <a href="/files/papers/bandit/simultaneous-static-dynamic-regret-bandits-2026.pdf">站内 PDF</a>。</li>
<li>Auer, P., Cesa-Bianchi, N., Freund, Y. and Schapire, R. E. (2002). <a href="https://doi.org/10.1007/s00453-002-0099-6" target="_blank" rel="noopener"><em>The Nonstochastic Multiarmed Bandit Problem</em></a>。</li>
<li>Blackwell, D. (1956). <a href="https://projecteuclid.org/journals/pacific-journal-of-mathematics/volume-6/issue-1/An-analog-of-the-minimax-theorem-for-vector-payoffs/pjm/1103044261.full" target="_blank" rel="noopener"><em>An Analog of the Minimax Theorem for Vector Payoffs</em></a>。</li>
<li>Wei, C.-Y., Hong, Y.-T. and Lu, C.-J. (2016). <em>Tracking the Best Expert in Adversarial Bandits</em>。</li>
</ol>
