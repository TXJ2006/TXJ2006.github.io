---
title: 第 4 篇：可数性、序列与网｜阅读笔记
layout: page
permalink: /courses/point-set-topology/notes/chapter-04/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/point-set-topology/">← 课程目录</a> · <a href="/courses/point-set-topology/notes/chapter-03/">上一篇</a> · <a href="/courses/point-set-topology/notes/chapter-05/">下一篇 →</a></p>
<div class="note-lead"><p>实分析常用序列寻找极限点，但在任意拓扑空间中，序列未必看得见所有闭包点。可数性公理说明什么时候序列足够；网把序列推广为适用于一般空间的收敛工具。</p></div>
<nav class="reading-links" aria-label="参考讲义"><a href="/files/math-resources/archive-2026-11-08-59.pdf">林剑峰讲义</a><a href="/files/math-resources/archive-2026-11-08-45.pdf">Yi Li 讲义</a><a href="/files/math-resources/archive-2026-11-08-43.pdf">Munkres 教材</a></nav>

<h2 id="section-21">一、收敛只需要邻域</h2>
<p>序列 \((x_n)\) 收敛到 \(x\)，写作 \(x_n\to x\)，意思是：对 \(x\) 的每个开邻域 \(U\)，存在 \(N\)，使 \(n\ge N\) 时 \(x_n\in U\)。在度量空间中，取 \(U=B(x,\varepsilon)\) 就回到 \(d(x_n,x)\to0\)。这里没有要求极限唯一；唯一性需要额外的 Hausdorff 条件。</p>
<p>“\(x\in\overline A\)”只说 \(x\) 的每个邻域与 \(A\) 相交；要从中构造序列，还需要能用一个可数的邻域表逐级逼近 \(x\)。这正是第一可数性的作用。</p>

<h2 id="section-22">二、第一可数与第二可数</h2>
<p>若对每个 \(x\in X\)，都存在可数族 \(U_1,U_2,\ldots\) 的 \(x\)-邻域，使任何 \(x\)-邻域都包含某个 \(U_n\)，称 \(X\) <strong>第一可数</strong>。度量空间第一可数：取 \(B(x,1/n)\) 即可。若整个拓扑有一个可数基，称 \(X\) <strong>第二可数</strong>；它自动第一可数，因为只需选取基中含 \(x\) 的那些成员。</p>
<p>若 \(X\) 第一可数且 \(x\in\overline A\)，把局部基改成递减的 \(V_n=U_1\cap\cdots\cap U_n\)，从每个 \(V_n\cap A\) 选一点 \(a_n\)。任意 \(x\)-邻域最终包含所有 \(V_n\)，故 \(a_n\to x\)。因此在第一可数空间中，闭包可由序列刻画；在一般空间中，这一步可能失败。</p>
<p>第二可数还推出 Lindelöf 性质：任意开覆盖有可数子覆盖。对覆盖中的每个基元素，若它被某个覆盖成员包含，就选一个这样的覆盖成员；可数基保证至多选可数多个。第二可数空间也有可数稠密子集：从每个非空基元素各选一点。反向在一般拓扑中不成立；在度量空间中成立。</p>

<h2 id="section-23">三、序列失效的具体空间</h2>
<p>取不可数集合 \(X\) 并赋予<strong>余可数拓扑</strong>：空集以及补集可数的集合开。固定 \(x\in X\)，令 \(A=X\setminus\{x\}\)。每个包含 \(x\) 的非空开集都与 \(A\) 相交，所以 \(x\in\overline A\)。然而任何由 \(A\) 中点组成的序列都不能收敛到 \(x\)：它的值域 \(C=\{a_n:n\ge1\}\) 可数，故 \(X\setminus C\) 是 \(x\) 的开邻域，却不含序列的任何项。</p>
<p>这个例子指出错误的推论：“闭包点一定能由序列逼近”。问题不在于证明技巧，而在于序列只有可数多个位置，空间可能同时要求躲开任意给定的可数集合。</p>

<h2 id="section-24">四、网怎样修复这个问题？</h2>
<p><strong>有向集</strong> \((D,\preceq)\) 要求任意 \(d_1,d_2\) 都有共同上界。以 \(D\) 为索引的点族 \((x_d)_{d\in D}\) 称网。网收敛到 \(x\)，若对每个 \(x\)-邻域 \(U\)，存在 \(d_0\) 使 \(d\succeq d_0\) 时 \(x_d\in U\)。自然数按通常顺序是有向集，所以序列是网的特例。</p>
<p>若 \(x\in\overline A\)，把 \(x\) 的所有邻域按“越小越靠后”排序。对每个邻域 \(U\) 选 \(a_U\in A\cap U\)。给定一个固定邻域 \(V\)，所有充分靠后的 \(U\) 都包含于 \(V\)，故 \(a_U\in V\)；这个网收敛到 \(x\)。于是任意拓扑空间都满足：\(x\in\overline A\) 当且仅当有取值于 \(A\) 的网收敛到 \(x\)。</p>

<h2 id="section-25">五、可数性怎样影响连续性？</h2>
<p>连续映射总保持网的极限：若 \(x_d\to x\) 且 \(f\) 连续，则 \(f(x_d)\to f(x)\)，因为目标邻域的原像是源邻域。反向也成立：若 \(f\) 不连续，某个开集原像不是开集，可在一个非内部点周围用上节的邻域索引构造反例网。</p>
<p>如果定义域第一可数，可以把上面的“网”换成“序列”。所以在实分析和度量空间里，逐点检验所有收敛序列就够；离开第一可数的世界后，这个检验可能漏掉不连续点。</p>

<h2 id="section-26">六、理解检查</h2>
<ol><li>证明度量空间中 \(B(x,1/n)\) 构成 \(x\) 的可数邻域基。</li><li>在不可数余可数空间中，说明为何每个常值以外的收敛序列最终必须等于其极限。</li><li>证明第二可数蕴含 Lindelöf；指出证明中“可数”用在哪里。</li><li>把邻域构造的网写出索引顺序，逐字检查它满足网的收敛定义。</li></ol>
<p class="note-footer">依据林剑峰 / 陈晓《拓扑学讲义》第 2–3 章及 Yi Li <em>Topology I</em> 第 1.2 节整理；网用于补足序列在一般拓扑中的局限。</p>
</section>
