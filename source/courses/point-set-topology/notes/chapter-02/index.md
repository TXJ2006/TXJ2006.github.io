---
title: 第 2 篇：内部、闭包、边界与连续映射｜阅读笔记
layout: page
permalink: /courses/point-set-topology/notes/chapter-02/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/point-set-topology/">← 课程目录</a> · <a href="/courses/point-set-topology/notes/chapter-01/">上一篇</a></p>
<div class="note-lead"><p>给定任意子集 \(A\subseteq X\)，它通常既不全开也不全闭。本篇用内部、闭包、边界和极限点把 \(A\) 在环境空间中的位置精确拆开，再说明连续性如何用这些对象重新表达。</p></div>
<nav class="reading-links" aria-label="参考资料"><a href="/files/math-resources/archive-2026-11-08-59.pdf">拓扑学讲义 PDF</a><a href="/files/math-resources/archive-2026-11-08-45.pdf">Topology I PDF</a></nav>

<h2 id="section-10">一、内部是集合中真正的开部分</h2>
<p>设 \(A\subseteq X\)。点 \(x\in A\) 若存在开集 \(U\) 使 \(x\in U\subseteq A\)，称为 \(A\) 的内部点。所有内部点组成内部 \(A^\circ\)。等价地，</p>
<div class="equation">\[A^\circ=\bigcup\{U\subseteq A:U\text{ 在 }X\text{ 中开}\}.\]</div>
<p>因此 \(A^\circ\) 是包含在 \(A\) 中的最大开集：它本身开，且任何包含于 \(A\) 的开集都包含在 \(A^\circ\) 中。由此立即得到 \(A\) 开当且仅当 \(A=A^\circ\)。内部是一个“向内修剪”的操作，删去所有没有完整开邻域的点。</p>

<h2 id="section-11">二、闭包是包含集合的最小闭集</h2>
<p>闭包记作 \(\overline A\)，定义为所有包含 \(A\) 的闭集的交：
\[\overline A=\bigcap\{F\subseteq X:A\subseteq F,\ F\text{ 闭}\}.\]
由于任意闭集交仍闭，\(\overline A\) 是包含 \(A\) 的最小闭集。因此 \(A\) 闭当且仅当 \(A=\overline A\)。</p>
<p>更有用的是邻域判据：\(x\in\overline A\) 当且仅当 \(x\) 的每个开邻域都与 \(A\) 相交。证明的关键是取补集：若某个开邻域 \(U\ni x\) 与 \(A\) 不交，则 \(X\setminus U\) 是包含 \(A\) 的闭集，却不含 \(x\)；反过来也同理。</p>
<p>在度量空间中，邻域判据变成 \(\inf_{a\in A}d(x,a)=0\)。但“距离为零”只是度量空间中的表达，闭包本身只依赖拓扑。</p>

<h2 id="section-12">三、极限点与孤立点</h2>
<p>点 \(x\in X\) 称为 \(A\) 的极限点，如果每个包含 \(x\) 的开邻域都与 \(A\setminus\{x\}\) 相交。记所有极限点组成的集合为 \(A'\)。闭包分解为
\[\overline A=A\cup A'.\]
所以一个集合闭，当且仅当它包含自己的所有极限点。</p>
<p>若 \(x\in A\) 但不是极限点，则称 \(x\) 是孤立点：存在开邻域 \(U\) 使 \(U\cap A=\{x\}\)。例如 \(A=\{1/n:n\ge1\}\subset\mathbb R\) 的每个点都是孤立点，但 \(0\) 是唯一的极限点，因此 \(\overline A=A\cup\{0\}\)。离散拓扑中每个点都有单点开邻域，所以每个子集都没有新的极限点。</p>

<h2 id="section-13">四、边界同时接触集合和补集</h2>
<p>边界定义为
\[\partial A=\overline A\setminus A^\circ=\overline A\cap\overline{X\setminus A}.\]
第二个等式解释了边界的几何意义：\(x\in\partial A\) 当且仅当每个邻域既碰到 \(A\)，也碰到补集 \(X\setminus A\)。边界点不一定属于 \(A\)：\(\partial(0,1)=\{0,1\}\)，而 \(\partial[0,1]\) 也是同一个集合。</p>
<p>三种情况要区分：开集满足 \(A\cap\partial A=\varnothing\)；闭集满足 \(\partial A\subseteq A\)；既开又闭的集合边界为空。边界不是“集合的边缘图像”，而是由开集结构定义的集合，因此在不同拓扑下会改变。</p>

<h2 id="section-14">五、子空间中的内部与闭包</h2>
<p>若 \(Y\subseteq X\) 取子空间拓扑，集合 \(A\subseteq Y\) 在 \(Y\) 中的闭包满足
\[\overline A^{\,Y}=Y\cap\overline A^{\,X}.\]
原因是：\(Y\) 中的邻域正是 \(X\) 中开集与 \(Y\) 的交，邻域是否与 \(A\) 相交可以直接转写为环境空间的邻域条件。内部则只有包含在 \(Y\) 中的环境开集才能使用，所以不能简单地把 \(A^\circ_X\) 与 \(Y\) 相交后就当作内部。</p>
<p>例如 \([0,1]\) 在 \(\mathbb R\) 中的内部是 \((0,1)\)，但在子空间 \([0,1]\) 中自身是开集，因此它的内部是 \([0,1]\)。内部取决于环境空间，闭包的相对公式则给出了两者之间的精确联系。</p>

<h2 id="section-15">六、连续性的三个等价版本</h2>
<p>映射 \(f:X\to Y\) 连续的定义是：对每个开集 \(V\subseteq Y\)，原像 \(f^{-1}(V)\) 在 \(X\) 中开。取补集可得完全等价的闭集版本：对每个闭集 \(F\subseteq Y\)，\(f^{-1}(F)\) 在 \(X\) 中闭。</p>
<p>第三个版本直接使用闭包：\(f\) 连续，当且仅当对每个 \(A\subseteq X\)，
\[f(\overline A)\subseteq\overline{f(A)}.\]
证明方向是：连续性使 \(f^{-1}(\overline{f(A)})\) 成为包含 \(A\) 的闭集，所以包含 \(\overline A\)；反向则把 \(A=f^{-1}(F)\) 代入闭集判据。这个版本说明连续映射不会把集合的极限关系送到“超出目标闭包”的位置。</p>
<p>在度量空间中，连续性还可写成邻域形式：对每个 \(x\in X\) 和每个 \(f(x)\) 的开邻域 \(V\)，存在 \(x\) 的开邻域 \(U\) 使 \(f(U)\subseteq V\)。当空间有度量时，这才进一步化为熟悉的 \(\varepsilon\)-\(\delta\) 语言。</p>

<h2 id="section-16">七、连续、开、闭不是同一个词</h2>
<p>连续映射检查原像；开映射要求开集的像开；闭映射要求闭集的像闭。连续双射不必是同胚，因为它的逆映射可能不连续。若一个双射同时是开映射或闭映射，则逆映射自动连续，因而是同胚。</p>
<p>一个基本例子是恒等映射。若 \(\mathcal T_1\subseteq\mathcal T_2\) 是同一集合上的两个拓扑，则恒等映射 \((X,\mathcal T_2)\to(X,\mathcal T_1)\) 连续，因为较少的目标开集原像仍是 \(\mathcal T_2\)-开；反方向通常不连续。这说明拓扑的粗细会直接影响连续性。</p>

<h2 id="section-17">八、理解检查</h2>
<ol><li>证明 \(X\setminus A^\circ=\overline{X\setminus A}\)，再推出边界的两个表达式相等。</li><li>在通常拓扑、离散拓扑和余有限拓扑中分别求 \(A=(0,1)\) 的闭包。</li><li>证明子空间闭包公式 \(\overline A^{\,Y}=Y\cap\overline A^{\,X}\)。</li><li>构造一个连续双射但不是同胚的映射，并直接找出其逆映射不连续的开集原像。</li><li>用闭集原像判据证明多项式映射 \(\mathbb R\to\mathbb R\) 连续。</li></ol>
<p class="note-footer">主要依据《拓扑学讲义》第 2 章与 Yi Li, <em>Topology I: Elementary Topology</em> 第 1.2 节整理；闭包、边界和连续映射的等价刻画与 Munkres 对照。</p>
</section>
