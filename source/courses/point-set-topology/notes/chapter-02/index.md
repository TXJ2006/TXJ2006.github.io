---
title: 第 2 篇：内部、闭包与连续性｜阅读笔记
layout: page
permalink: /courses/point-set-topology/notes/chapter-02/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/point-set-topology/">← 课程目录</a> · <a href="/courses/point-set-topology/notes/chapter-01/">上一篇</a> · <a href="/courses/point-set-topology/notes/chapter-03/">下一篇 →</a></p>
<div class="note-lead"><p>固定 \(A=[0,1)\subset\mathbb R\)。它的内部是 \((0,1)\)，闭包是 \([0,1]\)，边界是 \(\{0,1\}\)。先把这三个集合算清，再看它们怎样刻画连续映射。</p></div>
<nav class="reading-links" aria-label="参考讲义"><a href="/files/math-resources/archive-2026-11-08-59.pdf">林剑峰讲义</a><a href="/files/math-resources/archive-2026-11-08-45.pdf">Yi Li 讲义</a></nav>
<figure class="note-figure"><svg viewBox="0 0 360 188" role="img" aria-labelledby="set-interval-title set-interval-desc"><title id="set-interval-title">区间的内部、闭包与边界</title><desc id="set-interval-desc">四行数轴：A 从闭端点 0 到开端点 1；内部两端均开；闭包两端均闭；边界只含端点 0 和 1。</desc><path d="M140 34H335 M140 74H335 M140 114H335 M140 154H335" stroke="#bac2bd" stroke-width="2"/><path d="M180 34H295 M180 74H295 M180 114H295" stroke="#557f79" stroke-width="5"/><text x="14" y="40">A = [0, 1)</text><text x="14" y="80">内部 (0, 1)</text><text x="14" y="120">闭包 [0, 1]</text><text x="14" y="160">边界 {0, 1}</text><circle cx="180" cy="34" r="6" fill="#557f79"/><circle cx="295" cy="34" r="6" fill="var(--surface)" stroke="#557f79" stroke-width="2"/><circle cx="180" cy="74" r="6" fill="var(--surface)" stroke="#557f79" stroke-width="2"/><circle cx="295" cy="74" r="6" fill="var(--surface)" stroke="#557f79" stroke-width="2"/><circle cx="180" cy="114" r="6" fill="#557f79"/><circle cx="295" cy="114" r="6" fill="#557f79"/><circle cx="180" cy="154" r="6" fill="#bb6a4b"/><circle cx="295" cy="154" r="6" fill="#bb6a4b"/><text x="175" y="181">0</text><text x="290" y="181">1</text></svg><figcaption>实心点属于集合，空心点不属于集合；边界两点可以一内一外。</figcaption></figure>

<h2 id="section-7">一、内部：哪些点稳稳留在集合里？</h2>
<p>点 \(x\) 是 \(A\) 的内部点，意思是存在开集 \(U\) 满足 \(x\in U\subseteq A\)。记内部为 \(A^\circ\)。对于 \(A=[0,1)\)，每个 \(0\lt x\lt1\) 都有一段小区间留在 \(A\) 内；\(0\) 的任何实数开邻域都有负数，所以 \(0\) 不是内部点。因此 \(A^\circ=(0,1)\)。</p>
<p>把所有包含于 \(A\) 的开集并起来，恰好得到 \(A^\circ\)。所以它是 \(A\) 内最大的开集，\(A\) 本身开放当且仅当 \(A=A^\circ\)。这个“最大的”不是长度最大，而是按集合包含关系最大。</p>

<h2 id="section-8">二、闭包：哪些点无法与集合隔开？</h2>
<p>点 \(x\) 属于 \(A\) 的闭包 \(\overline A\)，当且仅当它的<strong>每一个</strong>开邻域都与 \(A\) 相交。对 \([0,1)\) 来说，\(1\notin A\)，但 \(1\) 附近任何开区间都含小于 1 的点，所以 \(1\in\overline A\)；\(2\) 有邻域 \((3/2,5/2)\) 与 \(A\) 不交，所以 \(2\notin\overline A\)。故 \(\overline A=[0,1]\)。</p>
<p>闭包也等于所有包含 \(A\) 的闭集的交。因此 \(\overline A\) 是包含 \(A\) 的最小闭集，\(A\) 闭当且仅当 \(A=\overline A\)。为什么两种定义相同？若 \(x\notin\overline A\)，存在与 \(A\) 不交的开邻域 \(U\ni x\)，于是闭集 \(X\setminus U\) 含 \(A\) 却不含 \(x\)；反过来从这种闭集取补集即可。</p>
<p>在度量空间中可以把“每个邻域都相交”写成 \(\inf_{a\in A}d(x,a)=0\)（\(A\ne\varnothing\)）。请留意：\(x\) 不一定属于 \(A\)；\(x=1\) 就是我们的例子。</p>

<h2 id="section-9">三、极限点和孤立点不是同一个概念</h2>
<p>如果 \(x\) 的每个开邻域都碰到 \(A\setminus\{x\}\)，称 \(x\) 是 \(A\) 的极限点。与闭包的区别只在于：即使 \(x\in A\)，也必须找到<strong>另一个</strong>属于 \(A\) 的点。若 \(x\in A\) 却不是极限点，则 \(x\) 是 \(A\) 的孤立点。记极限点集为 \(A'\)，总有 \(\overline A=A\cup A'\)。</p>
<p>取 \(B=\{1,1/2,1/3,\ldots\}\subset\mathbb R\)。每个 \(1/n\) 都能用一个足够小的开区间与其余点隔开，所以它们是 \(B\) 的孤立点；\(0\) 不在 \(B\) 中，却是极限点。于是 \(B'=\{0\}\)、\(\overline B=B\cup\{0\}\)。这个例子说明“点属于闭包”与“点是极限点”不能混用。</p>

<h2 id="section-10">四、边界：每个邻域都看到两边</h2>
<p>定义 \(\partial A=\overline A\setminus A^\circ\)。利用补集得到更有用的判据：</p>
<div class="equation">\[\partial A=\overline A\cap\overline{X\setminus A}.\]</div>
<p>因此边界点的每个开邻域同时碰到 \(A\) 和 \(X\setminus A\)。对 \([0,1)\) 来说，\(0\) 在集合内、\(1\) 不在集合内，但两者都是边界点；边界并不要求属于原集合。</p>
<p>公式可从补集关系直接推出：\(x\notin A^\circ\) 意味着没有完全落在 \(A\) 内的开邻域，等价于每个邻域都碰到 \(X\setminus A\)，即 \(x\in\overline{X\setminus A}\)。</p>

<h2 id="section-11">五、换了环境，答案可能改变</h2>
<p>令 \(Y=[0,1]\) 取 \(\mathbb R\) 的子空间拓扑。集合 \(A=[0,1)\) 在 \(Y\) 中是开集，因为 \(A=Y\cap(-1,1)\)；所以 \(\operatorname{Int}_Y(A)=A\)，虽然 \(\operatorname{Int}_{\mathbb R}(A)=(0,1)\)。这里 0 不再有“左边”的点可供检验。</p>
<p>闭包则有精确的相对公式。只要 \(A\subseteq Y\subseteq X\)，就有 \(\overline A^{\,Y}=Y\cap\overline A^{\,X}\)。因为 \(Y\) 中包含 \(y\) 的基本邻域形如 \(U\cap Y\)，而 \(A\subseteq Y\)，所以 \((U\cap Y)\cap A=U\cap A\)。此处的上标表示<strong>在哪个空间求闭包</strong>。</p>

<h2 id="section-12">六、连续映射为什么检查原像？</h2>
<p>映射 \(f:X\to Y\) 连续，定义为每个 \(Y\) 中开集 \(V\) 的原像 \(f^{-1}(V)\) 都在 \(X\) 中开。直观地说：若 \(f(x)\in V\)，连续性保证 \(x\) 的一个完整邻域都映入 \(V\)。把目标中的“安全范围”拉回定义域，才能检查 \(x\) 附近的行为。</p>
<p>在 \(\mathbb R\) 上取 \(f(x)=x^2\)。目标开区间 \((1,4)\) 的原像是 \((-2,-1)\cup(1,2)\)，它是开集；原像不必是一个区间。对任意目标开集都成立才是全局连续。复合映射连续则来自 \((g\circ f)^{-1}(W)=f^{-1}(g^{-1}(W))\)。</p>
<p>连续性也能用闭集表达：每个目标闭集的原像闭。其证明只需 \(f^{-1}(Y\setminus F)=X\setminus f^{-1}(F)\)。另一个等价形式是对所有 \(A\subseteq X\)，\(f(\overline A)\subseteq\overline{f(A)}\)：连续映射会把 \(A\) 的逼近点送到 \(f(A)\) 的闭包内。反向证明时令 \(A=f^{-1}(F)\)，其中 \(F\) 是目标闭集；这时 \(f(A)\subseteq F\)，故 \(f(\overline A)\subseteq\overline{f(A)}\subseteq F\)，从而 \(\overline A\subseteq f^{-1}(F)=A\)，即 \(A\) 闭。</p>

<h2 id="section-13">七、连续双射为什么还不是同胚？</h2>
<p>令 \(X\) 是至少有两个点的集合。同一个恒等函数 \(\mathrm{id}:X_{\mathrm{discrete}}\to X_{\mathrm{indiscrete}}\) 是连续双射，因为目标仅有 \(\varnothing,X\) 两个开集；反向却不连续，因为目标离散空间中的单点开集拉回后，在平凡拓扑中不是开集。因此要说两个拓扑空间“相同”，还必须要求逆映射连续。</p>
<p>一个更贴近几何的例子是 \([0,2\pi)\to S^1\)、\(t\mapsto(\cos t,\sin t)\)。它是连续双射，但逆映射在 \((1,0)\) 处会把相近的圆周点分别送到靠近 0 与靠近 \(2\pi\) 的参数，无法连续。这解释了为什么“画成一圈”不能仅靠连续双射证明同胚。</p>

<h2 id="section-14">八、自己算一遍</h2>
<ol><li>在 \(\mathbb R\) 中计算 \(A=\mathbb Q\) 的内部、闭包和边界，并解释任意开区间为何同时含有有理数与无理数。</li><li>在 \(Y=[0,1]\) 中重新计算 \(A=[0,1)\) 的内部、闭包和边界；与 \(\mathbb R\) 中的答案比较。</li><li>直接证明：若 \(f:X\to Y\) 连续且 \(F\subseteq Y\) 闭，则 \(f^{-1}(F)\) 闭。</li></ol>
<p class="note-footer">依据林剑峰 / 陈晓《拓扑学讲义》第 2 章及 Yi Li <em>Topology I</em> 第 1.2 节整理。</p>
</section>
