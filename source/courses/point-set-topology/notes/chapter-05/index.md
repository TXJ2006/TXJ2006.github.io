---
title: 第 5 篇：分离公理与连续函数｜阅读笔记
layout: page
permalink: /courses/point-set-topology/notes/chapter-05/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/point-set-topology/">← 课程目录</a> · <a href="/courses/point-set-topology/notes/chapter-04/">上一篇</a> · <a href="/courses/point-set-topology/notes/chapter-06/">下一篇 →</a></p>
<div class="note-lead"><p>开集不一定能区分两个点。分离公理逐级要求我们能用开集分开“点与点”“点与闭集”“两个闭集”，从而控制极限是否唯一、商空间是否好用、能否构造连续函数。</p></div>
<nav class="reading-links" aria-label="参考讲义"><a href="/files/math-resources/archive-2026-11-08-59.pdf">林剑峰讲义：2.3、2.4、8</a><a href="/files/math-resources/archive-2026-11-08-34.pdf">Körner 讲义：8</a></nav>

<h2 id="section-27">一、先比较四种“分开”</h2>
<p><strong>\(T_1\)</strong> 要求每个单点集闭。等价地，对不同的 \(x,y\)，可找一个包含 \(x\) 却不含 \(y\) 的开集，也可反过来找一个包含 \(y\) 却不含 \(x\) 的开集。<strong>Hausdorff（\(T_2\)）</strong> 更强：可找互不相交的开集分别包含 \(x,y\)。</p>
<p>在本系列约定，<strong>正则（\(T_3\)）</strong> 指 \(T_1\) 且每个点 \(x\) 与不含它的闭集 \(F\) 有不相交的开邻域；<strong>正规（\(T_4\)）</strong> 指 \(T_1\) 且任意两个不交闭集有不相交的开邻域。这样有 \(T_4\Rightarrow T_3\Rightarrow T_2\Rightarrow T_1\)。不同教材对 \(T_3,T_4\) 是否默认含 \(T_1\) 约定不一，判断定理时应看定义而非编号。</p>
<p>无限集合上的余有限拓扑是 \(T_1\)，却不是 Hausdorff：单点的补集开，但任意两个非空开集相交。它证明“每个点都闭”仍不足以保证极限唯一。</p>

<h2 id="section-28">二、Hausdorff 条件到底保证什么？</h2>
<p>若序列或网同时收敛到不同点 \(x,y\)，取不交的开邻域 \(U\ni x,V\ni y\)。充分靠后的项必须同时在 \(U\) 和 \(V\)，矛盾。故 Hausdorff 空间的收敛极限唯一。反过来，在特殊空间中极限唯一未必保证完整的 Hausdorff 分离条件，因此这里使用的是充分条件。</p>
<p>还有一个实用判据：\(X\) Hausdorff 当且仅当对角线 \(\Delta_X=\{(x,x):x\in X\}\) 在 \(X\times X\) 中闭。若 \((x,y)\notin\Delta_X\)，不交邻域 \(U,V\) 给出不碰对角线的积开集 \(U\times V\)；反过来从这种积邻域可读出 \(U\cap V=\varnothing\)。</p>

<h2 id="section-29">三、度量空间为何正规？</h2>
<p>取度量空间中不交的闭集 \(A,B\)。定义到集合的距离 \(d(x,A)=\inf_{a\in A}d(x,a)\)。三角不等式给出 \(|d(x,A)-d(y,A)|\le d(x,y)\)，所以它连续；闭集条件使 \(d(x,A)=0\) 当且仅当 \(x\in A\)。</p>
<p>若 \(A,B\) 都非空，则 \(d(x,A)+d(x,B)>0\)。因此</p>
<div class="equation">\[h(x)=\frac{d(x,A)}{d(x,A)+d(x,B)}\]</div>
<p>是连续函数，且 \(h(A)=0,h(B)=1\)。取 \(U=h^{-1}([0,1/3))\)、\(V=h^{-1}((2/3,1])\)，其中区间按 \([0,1]\) 的子空间拓扑理解，就得到不交开邻域。若某个闭集为空，分离显然。这个构造说明度量空间不只是能分开点，还能用连续实值函数分开闭集。</p>

<h2 id="section-30">四、开集怎样升级成 Urysohn 函数？</h2>
<p><strong>Urysohn lemma：</strong>若 \(X\) 正规、\(A,B\subseteq X\) 不交且闭，则存在连续 \(f:X\to[0,1]\)，使 \(f(A)=0,f(B)=1\)。度量情形刚才给了显式公式；一般正规空间没有距离，必须用开集逐级逼近。</p>
<p>证明的核心引理是：若闭集 \(F\subseteq U\) 且 \(U\) 开，可找开集 \(V\) 使 \(F\subseteq V\subseteq\overline V\subseteq U\)。这是把 \(F\) 与 \(X\setminus U\) 分开。随后对 \([0,1]\) 的有理数 \(r\) 构造嵌套开集 \(U_r\)，使 \(r\lt s\) 时 \(\overline{U_r}\subseteq U_s\)，最后令 \(f(x)=\inf\{r:x\in U_r\}\)（空集时取 1）。嵌套的“闭包留有余地”保证 \(f\) 的上下水平集都开，因此 \(f\) 连续。完整的逐项归纳见原讲义第 8 章。</p>

<h2 id="section-31">五、从分离到延拓</h2>
<p><strong>Tietze extension theorem：</strong>若 \(A\) 是正规空间 \(X\) 的闭子集，连续实值函数 \(g:A\to\mathbb R\) 可延拓为连续 \(G:X\to\mathbb R\)。闭集假设很重要；若 \(A=(0,1)\subset\mathbb R\)、\(g(x)=1/x\)，它在 \(A\) 上连续，却不能连续延拓到 0。</p>
<p>对有界 \(g\)，可反复用 Urysohn 函数逼近剩余误差，使误差按几何级数收缩，再取一致收敛的函数级数；一般实值函数先用 \(\arctan\) 压入有界区间，延拓后再逆变换。这里的逻辑是“正规性 \(\Rightarrow\) 可用连续函数分开闭集 \(\Rightarrow\) 可延拓闭集上的函数”。</p>

<h2 id="section-32">六、理解检查</h2>
<ol><li>证明 \(T_1\) 与“每个单点集闭”等价。</li><li>用对角线判据说明 \(\mathbb R\) 是 Hausdorff。</li><li>把 \(A=\{0\},B=\{2\}\subset\mathbb R\) 代入上面的距离公式，算出 \(h(1)\)。</li><li>指出 Tietze 定理中为什么要求 \(A\) 闭；尝试给出另一个不能延拓的例子。</li></ol>
<p class="note-footer">依据林剑峰 / 陈晓《拓扑学讲义》第 2.3、2.4、8 章及 Körner <em>Metric and Topological Spaces</em> 第 8 节整理。</p>
</section>
