---
title: 第 1 篇：从距离到拓扑、基与子空间｜阅读笔记
layout: page
permalink: /courses/point-set-topology/notes/chapter-01/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/point-set-topology/">← 课程目录</a> · <a href="/courses/point-set-topology/notes/chapter-02/">下一篇 →</a></p>
<div class="note-lead"><p>先看一个熟悉的事实：\((0,1)\) 在实数轴上是开集，因为其中每一点都能容下一个更小的开区间。拓扑把“每一点周围还能活动一点”抽象为开集，不再要求先给距离。</p></div>
<nav class="reading-links" aria-label="参考讲义"><a href="/files/math-resources/archive-2026-11-08-59.pdf">林剑峰讲义</a><a href="/files/math-resources/archive-2026-11-08-45.pdf">Yi Li 讲义</a><a href="/files/math-resources/archive-2026-11-08-34.pdf">Körner 讲义</a></nav>

<h2 id="section-1">一、为什么从开集入手？</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/01-open-ball.svg" alt="透视平面中的开区域 U 包含点 x 周围的整个小邻域" loading="lazy"><figcaption>开集的条件逐点检查：\(x\) 周围的小邻域完全留在 \(U\) 内；贴近边界的点要缩小邻域。</figcaption></figure>
<p>在度量空间 \((X,d)\) 中，点 \(x\) 的半径 \(r\) 开球是 \(B(x,r)=\{y:d(x,y)\lt r\}\)。集合 \(U\subseteq X\) 开，意思是每个 \(x\in U\) 都有某个 \(r>0\)，使 \(B(x,r)\subseteq U\)。这不是要求全体点共用一个半径；靠近区间端点的点，可以选择更小的半径。</p>
<p>例如 \(x\in(0,1)\) 时，取 \(r=\frac12\min\{x,1-x\}\)，便有 \(B(x,r)\subset(0,1)\)。而 \([0,1)\) 在 \(\mathbb R\) 中不开放，因为 \(0\) 的每个开球都含负数。这个区别完全由开球如何覆盖点决定。</p>
<p>我们关心的不只是数值距离，还关心哪些映射连续、哪些点可被开集区分。不同度量有时给出完全相同的开集，例如 \(d(x,y)=|x-y|\) 和 \(d'(x,y)=\min\{1,|x-y|\}\) 在 \(\mathbb R\) 上给出同一个拓扑。由此可把开集结构本身作为研究对象。</p>

<h2 id="section-2">二、开集公理从哪里来？</h2>
<p>度量空间的开集有三条共同性质：空集和全空间开；任意多个开集的并开；有限多个开集的交开。交集只保证有限个，是因为点在两个开集内时可取两个半径的最小值，而无限多个半径的下确界可能为零。</p>
<p>于是给定集合 \(X\)，一个<strong>拓扑</strong>就是子集族 \(\mathcal T\subseteq\mathcal P(X)\)，满足上面三条性质。\((X,\mathcal T)\) 称拓扑空间，\(U\in\mathcal T\) 称开集。\(\mathcal P(X)\) 是 \(X\) 的所有子集的集合；写成 \(\mathcal T\subseteq\mathcal P(X)\)，是说我们从所有子集中选出哪些算开。</p>
<p>这组公理恰好允许连续性的开集原像定义与复合运算配合。无限交不能随意加入：在 \(\mathbb R\) 中，\(\bigcap_{n\ge1}(-1/n,1/n)=\{0\}\)，而 \(\{0\}\) 不是通常拓扑的开集。</p>

<h2 id="section-3">三、同一个集合有不同的“靠近”方式</h2>
<p>令 \(X\) 是无限集合。\(\{\varnothing,X\}\) 是<strong>平凡拓扑</strong>，几乎无法用开集区分点。\(\mathcal P(X)\) 是<strong>离散拓扑</strong>，每个单点集都开。<strong>余有限拓扑</strong>则把空集和所有“补集有限”的集合规定为开集。检验任意并时，看补集变成有限集合的交；检验有限交时，看补集变成有限集合的并。</p>
<p>这些结构不同：在无限余有限空间中，两个非空开集一定相交；在离散空间中，不同点的单点开集不相交。也因此，“\(A\) 是开集”必须连同环境空间及其拓扑一起说。</p>
<p>若 \(\mathcal T_1\subseteq\mathcal T_2\)，称 \(\mathcal T_2\) 比 \(\mathcal T_1\) <strong>更细</strong>。同一个集合的恒等映射 \((X,\mathcal T_2)\to(X,\mathcal T_1)\) 连续，因为目标的每个开集在定义域中仍开；反方向未必连续。这是判断粗细方向的可靠办法。</p>

<h2 id="section-4">四、基为什么能够代替全部开集？</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/01-basis-overlap.svg" alt="两个基元素在点 x 处相交，交内有更小的基元素 B3" loading="lazy"><figcaption>基条件只要求交集中的每个点都能找到更小的基元素 \(B_3\)，并不要求整个交集本身属于基。</figcaption></figure>
<p>实数轴上不必列出全部开集，只需给出开区间：每个开集都是若干开区间的并。一般地，子集族 \(\mathcal B\) 是 \(X\) 上的<strong>基</strong>，若它覆盖 \(X\)，而且当 \(x\in B_1\cap B_2\)（\(B_1,B_2\in\mathcal B\)）时，能找到 \(B_3\in\mathcal B\) 使 \(x\in B_3\subseteq B_1\cap B_2\)。由 \(\mathcal B\) 生成的开集就是它的成员的任意并，空并给出空集。</p>
<p>第二条条件只要求在交集的<strong>每个点附近</strong>找到更小的基元素，并不要求整个交集本身属于基。例如开球的交一般不是一个开球，但交集内每一点都能容下一个小开球。这样，基元素的任意并既对任意并封闭，也对有限交封闭，确实形成拓扑。</p>
<p>一个常用判据是：\(\mathcal B\) 是已有拓扑 \(\mathcal T\) 的基，当且仅当每个 \(x\in U\in\mathcal T\) 都能找到 \(B\in\mathcal B\)，使 \(x\in B\subseteq U\)。取端点为有理数的开区间就给 \(\mathbb R\) 一个可数基：在实数 \(x\) 周围的任意开区间内，都可用有理数选出更小的区间。</p>
<p>若只给出<strong>子基</strong> \(\mathcal S\)，先取子基元素的所有有限交（包括空交 \(X\)）形成基，再取任意并。例如 \(\mathbb R\) 的所有射线 \((a,\infty)\) 与 \((-\infty,b)\) 是通常拓扑的子基，因为二者的交给出开区间。</p>

<h2 id="section-5">五、子空间的“开”是相对于自身的</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/01-subspace.svg" alt="透视平面 X 中的带状子空间 Y 与环境开区域 U 相交" loading="lazy"><figcaption>窄带表示子空间 \(Y\)；其中被环境开集 \(U\) 截出的部分 \(U\cap Y\)，在 \(Y\) 中开放。</figcaption></figure>
<p>若 \(Y\subseteq X\)，定义 \(Y\) 中的开集为 \(U\cap Y\)，其中 \(U\) 在 \(X\) 中开。这给出<strong>子空间拓扑</strong> \(\mathcal T_Y=\{U\cap Y:U\in\mathcal T_X\}\)。例如令 \(Y=[0,1]\subset\mathbb R\)。\([0,1/2)\) 在 \(Y\) 中开，因为它等于 \((-1,1/2)\cap Y\)；在 \(\mathbb R\) 中却不开放，因为 \(0\) 没有留在该集合中的实数开邻域。</p>
<p>如果 \(\mathcal B\) 是 \(X\) 的基，那么 \(\{B\cap Y:B\in\mathcal B\}\) 是 \(Y\) 的基。理由是任意相对开集 \(U\cap Y\) 可把 \(U\) 写成基元素的并，再与 \(Y\) 取交。这个事实让我们计算子空间时无需重新从公理开始。</p>
<p>包含映射 \(i:Y\hookrightarrow X\) 自动连续，因为 \(i^{-1}(U)=U\cap Y\)。反过来，对任何 \(f:Z\to Y\)，\(f\) 连续当且仅当 \(i\circ f:Z\to X\) 连续；这把“在 \(Y\) 中连续”化成“在环境空间中连续”。</p>

<h2 id="section-6">六、读完能做什么？</h2>
<ol><li>给定一个“开集”候选族，逐一检查空集、全体、任意并、有限交。</li><li>给定基，只需用“每个点附近能找到更小的基元素”检验开集。</li><li>遇到子集先说明环境空间：\([0,1/2)\) 在 \([0,1]\) 中开，在 \(\mathbb R\) 中不开。</li></ol>
<p>试着证明：\(\mathbb R\) 中开球与开区间产生同一拓扑；若 \(A\subseteq Y\subseteq X\)，\(A\) 在 \(Y\) 中开且 \(Y\) 在 \(X\) 中开，则 \(A\) 在 \(X\) 中开。第二篇将把这些开集用于计算内部、闭包和连续性。</p>
<p class="note-footer">依据林剑峰 / 陈晓《拓扑学讲义》第 1 章、Yi Li <em>Topology I</em> 第 1.1–1.2 节及 Körner <em>Metric and Topological Spaces</em> 第 2–5 节整理。</p>
</section>
