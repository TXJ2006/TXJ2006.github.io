---
title: 第 1 篇：拓扑空间、基与子空间｜阅读笔记
layout: page
permalink: /courses/point-set-topology/notes/chapter-01/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/point-set-topology/">← 课程目录</a></p>
<div class="note-lead"><p>拓扑的第一步不是定义“距离很近”，而是直接规定哪些集合算作开集。只要开集满足三条公理，就可以在没有坐标和距离的集合上讨论连续性。</p></div>
<nav class="reading-links" aria-label="参考资料"><a href="/files/math-resources/archive-2026-11-08-59.pdf">拓扑学讲义 PDF</a><a href="/files/math-resources/archive-2026-11-08-45.pdf">Topology I PDF</a><a href="/files/math-resources/archive-2026-11-08-34.pdf">Metric and Topological Spaces PDF</a></nav>

<h2 id="section-1">一、拓扑要解决什么问题？</h2>
<p>在 \(\mathbb R^n\) 中，连续性的通常定义依赖距离：点 \(x\) 附近的点在映射下仍应落在 \(f(x)\) 附近。可是许多重要对象没有天然的欧氏距离，例如连续函数空间、无穷维空间、矩阵群、商空间和代数几何中的点集。若把连续性写成“开集的原像仍是开集”，就能把距离从定义中抽掉。</p>
<p>因此拓扑学研究的基本对象是一个集合 \(X\) 和一批被指定为“开”的子集。这个选择必须足以表达局部性，又不能依赖某个具体坐标系统。</p>

<h2 id="section-2">二、拓扑空间的严格定义</h2>
<p>设 \(X\) 是集合。一个拓扑是幂集 \(\mathcal P(X)\) 的子集 \(\mathcal T\)，满足：</p>
<ol><li>\(\varnothing\in\mathcal T\)，且 \(X\in\mathcal T\)；</li><li>若 \(U,V\in\mathcal T\)，则 \(U\cap V\in\mathcal T\)；更一般地，任意有限个开集的交仍开；</li><li>对任意指标集 \(I\)，若每个 \(U_i\in\mathcal T\)，则 \(\bigcup_{i\in I}U_i\in\mathcal T\)。</li></ol>
<p>二元组 \((X,\mathcal T)\) 称拓扑空间，\(\mathcal T\) 中的集合称开集。这里“有限交”和“任意并”不是对称的：无限交不必开，例如 \(\bigcap_{n\ge1}(-1/n,1/n)=\{0\}\) 在通常拓扑的 \(\mathbb R\) 中不是开集。</p>
<p>闭集不是另一种独立结构：集合 \(F\subseteq X\) 称闭集，当且仅当 \(X\setminus F\) 开。由德摩根律，闭集满足对任意交封闭、对有限并封闭。这种开闭对偶会反复用于闭包、紧性和连续映射的证明。</p>

<h2 id="section-3">三、同一个集合可以承载不同拓扑</h2>
<p>拓扑不是集合本身的属性，而是集合上的额外结构。常见例子如下。</p>
<table><thead><tr><th>拓扑</th><th>开集</th><th>局部性</th></tr></thead><tbody><tr><td>平凡拓扑</td><td>\(\varnothing,X\)</td><td>最少，几乎无法区分点</td></tr><tr><td>离散拓扑</td><td>\(\mathcal P(X)\)</td><td>最多，每个单点集都开</td></tr><tr><td>余有限拓扑</td><td>空集或补集有限的集合</td><td>无限集合中任意两个非空开集相交</td></tr><tr><td>度量拓扑</td><td>每个点都含一个开球的集合</td><td>由距离产生，但不只限于欧氏空间</td></tr></tbody></table>
<p>拓扑越细，开集越多；连续性条件通常越容易从定义域满足、越难从目标空间满足。不能只说“这个集合是开集”，必须说明是在什么拓扑下。</p>

<h2 id="section-4">四、度量怎样产生拓扑？</h2>
<p>度量是函数 \(d:X\times X\to[0,\infty)\)，满足正定性、对称性和三角不等式。给定 \(x\in X\) 与 \(r>0\)，开球为 \(B_d(x,r)=\{y:d(x,y)\lt r\}\)。定义 \(U\subseteq X\) 开，当且仅当每个 \(x\in U\) 都存在 \(r>0\) 使 \(B_d(x,r)\subseteq U\)。这些开球生成的拓扑称度量拓扑。</p>
<p>关键点是：度量不是拓扑本身。不同度量可能产生同一个拓扑；而有些拓扑根本不能由任何度量产生。拓扑学保留的是开集、连续性和收敛等结构，舍弃具体距离的数值。</p>

<h2 id="section-5">五、基：如何少写很多开集？</h2>
<p>直接列出所有开集通常不现实。设 \(\mathcal B\subseteq\mathcal P(X)\)。称 \(\mathcal B\) 为一个基，如果：</p>
<ol><li>每个点 \(x\in X\) 至少属于某个 \(B\in\mathcal B\)；</li><li>若 \(x\in B_1\cap B_2\)，其中 \(B_1,B_2\in\mathcal B\)，则存在 \(B_3\in\mathcal B\)，满足 \(x\in B_3\subseteq B_1\cap B_2\)。</li></ol>
<p>由基生成的拓扑规定：开集恰好是基元素的任意并。第二条不是要求基元素本身对交封闭，而是要求交集中的每个点附近都能找到更小的基元素。它正是“基元素能够描述局部邻域”的条件。</p>
<p>证明生成的确是拓扑时，任意并自动满足；有限交的证明使用第二条：若 \(x\in U\cap V\)，分别取包含 \(x\) 的 \(B_1\subseteq U\)、\(B_2\subseteq V\)，再取 \(B_3\subseteq B_1\cap B_2\)，于是 \(x\) 落在交集内的基元素中。</p>
<p>\(\mathbb R^n\) 的开球是基；也可以只取中心和半径为有理数的开球，仍得到同一个通常拓扑。后一个例子说明：基是描述拓扑的工具，不必包含所有可能的开集。</p>

<h2 id="section-6">六、子空间拓扑：子集继承什么结构？</h2>
<p>设 \((X,\mathcal T)\) 是拓扑空间，\(Y\subseteq X\)。子空间拓扑定义为
\[\mathcal T_Y=\{U\cap Y:U\in\mathcal T\}.\]
因此 \(Y\) 中的开集不一定是 \(X\) 中的开集，而是某个环境开集与 \(Y\) 的交。例如 \([0,1]\) 在 \(\mathbb R\) 的子空间拓扑中是开集，因为 \([0,1]=(-1,2)\cap[0,1]\)，但它不是 \(\mathbb R\) 中的开集。</p>
<p>子空间拓扑是使包含映射 \(\iota:Y\hookrightarrow X\) 连续的最自然拓扑：若 \(U\) 在 \(X\) 中开，则 \(\iota^{-1}(U)=U\cap Y\) 在 \(Y\) 中开。更强的事实是，对任意拓扑空间 \(Z\)，映射 \(f:Z\to Y\) 连续，当且仅当复合 \(\iota\circ f:Z\to X\) 连续。这是子空间的泛性质。</p>

<h2 id="section-7">七、连续映射与同胚</h2>
<p>拓扑空间之间的映射 \(f:(X,\mathcal T_X)\to(Y,\mathcal T_Y)\) 连续，当且仅当对每个目标开集 \(V\in\mathcal T_Y\)，原像 \(f^{-1}(V)\) 在 \(X\) 中开。注意方向：连续性检查的是<strong>原像</strong>，而不是像。这个方向使连续映射可以复合，也使子空间和商空间的定义自然成立。</p>
<p>若 \(f\) 是双射且 \(f\) 与 \(f^{-1}\) 都连续，则称 \(f\) 为同胚。拓扑学把同胚空间视为同一个空间，因为开集结构、闭集结构和所有由它们定义的性质都被保留。仅有连续双射还不够：逆映射可能不连续。</p>

<h2 id="section-8">八、这一篇的逻辑链</h2>
<ol><li>先在集合上指定开集，得到拓扑空间；</li><li>用度量的开球或抽象的基生成拓扑；</li><li>用原像定义连续性，用连续双射及连续逆映射定义同胚；</li><li>用交集定义子空间，使子集继承环境空间的局部结构。</li></ol>
<p>后续的内部、闭包、边界、紧性和连通性都建立在这条链上。学习时遇到新拓扑，先回答三个问题：哪些集合开？它由什么基生成？一个映射的开集原像是什么？</p>

<h2 id="section-9">理解检查</h2>
<ol><li>证明余有限拓扑满足拓扑的三条公理，特别说明有限个余有限集的交为何仍余有限。</li><li>在离散拓扑和余有限拓扑中分别求单点集的闭包。</li><li>证明有理数开区间 \((a,b)\)（\(a,b\in\mathbb Q\)）构成 \(\mathbb R\) 通常拓扑的一个基。</li><li>说明 \([0,1)\) 为什么在 \([0,1]\) 的子空间拓扑中开，却不在 \(\mathbb R\) 中开。</li><li>给出一个连续双射但逆映射不连续的例子，并指出失败发生在哪里。</li></ol>
<p class="note-footer">主要依据《拓扑学讲义》第 1 章与 Yi Li, <em>Topology I: Elementary Topology</em> 第 1 章整理；定义与基的判据与 Munkres 的标准表述对照。</p>
</section>
