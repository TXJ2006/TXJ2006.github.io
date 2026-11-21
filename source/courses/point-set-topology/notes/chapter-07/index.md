---
title: 第 7 篇：连通、道路与分支｜阅读笔记
layout: page
permalink: /courses/point-set-topology/notes/chapter-07/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/point-set-topology/">← 课程目录</a> · <a href="/courses/point-set-topology/notes/chapter-06/">上一篇</a> · <a href="/courses/point-set-topology/notes/chapter-08/">下一篇 →</a></p>
<div class="note-lead"><p>一条线段无法被拆成两个彼此隔开的非空开块，这是连通性；线段上任意两点还能用一条连续路径相连，这是道路连通。两者在熟悉的空间中常同时出现，在一般空间中却不同。</p></div>
<nav class="reading-links" aria-label="参考讲义"><a href="/files/math-resources/archive-2026-11-08-59.pdf">林剑峰讲义：第 5 章</a><a href="/files/math-resources/archive-2026-11-08-34.pdf">Körner 讲义：11</a></nav>

<h2 id="section-39">一、连通的定义与三种等价说法</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/07-disconnected.svg" alt="两个相隔的非空区域 U 和 V 将空间分成两块" loading="lazy"><figcaption>两个非空、互不相交的相对开块若盖满空间，就给出“不连通”的证据。</figcaption></figure>
<p>空间 \(X\) <strong>不连通</strong>，若可写成 \(X=U\cup V\)，其中 \(U,V\) 非空、互不相交，并且都在 \(X\) 中开。否则称 \(X\) 连通。等价地，连通空间只有 \(\varnothing,X\) 两个既开又闭（clopen）的集合：若 \(U\) 既开又闭，\(U\) 和 \(X\setminus U\) 就构成这样的拆分。</p>
<p>这个定义必须使用<strong>子空间拓扑</strong>。例如 \((0,1)\cup(2,3)\) 作为 \(\mathbb R\) 的子空间不连通，因为两段分别在子空间中开；\([0,1]\) 作为实数子空间连通，虽然其中很多集合在 \(\mathbb R\) 中不是开集。</p>

<h2 id="section-40">二、为什么实数区间连通？</h2>
<p>设区间 \(I\subseteq\mathbb R\) 被不交的非空相对开集 \(U,V\) 覆盖。取 \(u\in U,v\in V\)，不妨令 \(u\lt v\)。在 \([u,v]\subseteq I\) 中，令 \(c=\sup(U\cap[u,v])\)。若 \(c\in U\)，相对开放性使 \(c\) 右侧还有 \(U\) 的点，矛盾；若 \(c\in V\)，相对开放性使 \(c\) 左侧有一段区间落在 \(V\)，与 \(U\) 的点可任意逼近 \(c\) 矛盾。因此不存在拆分。</p>
<p>反向也成立：若 \(A\subseteq\mathbb R\) 不是区间，存在 \(a,c\in A\)、\(b\notin A\) 且 \(a\lt b\lt c\)。则 \(A\cap(-\infty,b)\) 与 \(A\cap(b,\infty)\) 是非空、不交的相对开集，拆分 \(A\)。所以实数轴的连通子集恰是区间，包括单点、开闭半开及无界区间。</p>

<h2 id="section-41">三、中间值定理是连通像的结论</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/07-intermediate-value.svg" alt="连续曲线从低高度上升到高高度并穿过给定中间高度 t" loading="lazy"><figcaption>区间的连续像仍是区间，所以从低值走到高值的曲线必须取到中间高度 \(t\)。</figcaption></figure>
<p>连续映射把连通空间送到连通空间。若 \(f(X)\) 可被两个不交非空相对开集拆开，把它们拉回 \(X\)，就得到 \(X\) 的拆分，矛盾。于是连续 \(f:[a,b]\to\mathbb R\) 的像是区间；当 \(f(a)\lt t\lt f(b)\) 时，\(t\in f([a,b])\)。这正是中间值定理，不需要在证明中重新分析每个函数的公式。</p>
<p>还有一个常用粘合法：若一族连通子集 \(\{C_i\}\) 有共同点，则 \(\bigcup_iC_i\) 连通。若整个并集被两个开块拆开，每个 \(C_i\) 必全部落在其中一块；共同点迫使所有 \(C_i\) 落在同一块，另一块为空。</p>

<h2 id="section-42">四、道路连通比连通更强</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/07-sine-curve.svg" alt="向竖直极限线无限振荡的正弦曲线闭包" loading="lazy"><figcaption>正弦曲线的闭包连通；左侧竖线与曲线之间却不能画出一条连续路径。</figcaption></figure>
<p>若对任意 \(x,y\in X\)，存在连续路径 \(\gamma:[0,1]\to X\)，满足 \(\gamma(0)=x,\gamma(1)=y\)，称 \(X\) 道路连通。因为 \([0,1]\) 连通，任一路径的像连通；固定一点 \(x_0\)，把从 \(x_0\) 到各点的路径像并起来，这些像有共同点且覆盖 \(X\)，故道路连通推出连通。</p>
<p>逆命题不成立。<strong>拓扑学家的正弦曲线</strong>是 \(S=\{(x,\sin(1/x)):0\lt x\le1\}\) 的闭包。\(S\) 连通：原曲线是区间 \((0,1]\) 的连续像，闭包保持连通。但竖线 \(\{0\}\times[-1,1]\) 上的点无法通过 \(S\) 中路径走到原曲线。若存在路径 \(\gamma\)，取其第一坐标最后一次为 0 的时刻 \(t_0\)。在 \(t_0\) 之后第一坐标为正，且从 0 连续增加到某个正数，所以在任意靠近 \(t_0\) 的时间段内，它取到使 \(\sin(1/x)=1\) 和 \(-1\) 的值。路径的第二坐标因此不可能在 \(t_0\) 连续。这个反例揭示“整体没被拆开”与“能走一条路”之间的差别。</p>

<h2 id="section-43">五、分支与局部道路连通</h2>
<p>点 \(x\) 的<strong>连通分支</strong>是包含 \(x\) 的最大连通子集：取所有含 \(x\) 的连通子集的并即可。分支彼此相同或不交，并且每个连通分支闭，因为连通集的闭包仍连通；但分支不一定开。<strong>道路分支</strong>则把“连通”换成“可用路径相连”。</p>
<p>若每点的每个邻域都含一个更小的道路连通开邻域，称空间局部道路连通。在这种空间中，道路分支是开集：对分支中的点取道路连通小邻域，它全落在该分支内；其他道路分支的并也是开集，所以每个道路分支也闭。连通分支因此恰好等于道路分支。开子集的 \(\mathbb R^n\) 局部道路连通，正弦曲线闭包在振荡竖线附近不满足这个条件。</p>

<h2 id="section-44">六、理解检查</h2>
<ol><li>从“只有平凡 clopen 集”推出连通定义，反方向也证明。</li><li>用连续像保持连通证明多项式在变号区间中有零点。</li><li>证明连通集 \(C\) 的闭包 \(\overline C\) 仍连通。</li><li>给出不连通但每个连通分支都是单点的空间。</li></ol>
<p class="note-footer">依据林剑峰 / 陈晓《拓扑学讲义》第 5 章及 Körner <em>Metric and Topological Spaces</em> 第 11 节整理。</p>
</section>
