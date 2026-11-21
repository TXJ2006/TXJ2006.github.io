---
title: 第 9 篇：可度量化与函数空间｜阅读笔记
layout: page
permalink: /courses/point-set-topology/notes/chapter-09/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/point-set-topology/">← 课程目录</a> · <a href="/courses/point-set-topology/notes/chapter-08/">上一篇</a> · <a href="/courses/point-set-topology/notes/chapter-10/">下一篇 →</a></p>
<div class="note-lead"><p>拓扑不必来自距离，但何时能重新找到一个产生同样开集的度量？Urysohn metrization theorem 给出可检查的充分条件。另一个方向是把连续映射本身组成空间，讨论“函数之间怎样接近”。</p></div>
<nav class="reading-links" aria-label="参考讲义"><a href="/files/math-resources/archive-2026-11-08-59.pdf">林剑峰讲义：第 8 章</a><a href="/files/math-resources/archive-2026-11-08-45.pdf">Yi Li 讲义：1.2–1.3</a><a href="/files/math-resources/archive-2026-11-08-43.pdf">Munkres 教材</a></nav>

<h2 id="section-51">一、什么叫可度量化？</h2>
<p>空间 \((X,\mathcal T)\) <strong>可度量化</strong>，若存在某个度量 \(d\) 使它的开球恰好生成 \(\mathcal T\)。这与“已经指定了一个完备度量”不同：可度量化只关心拓扑是否能由某种距离描述，不要求那种距离完备。第 8 篇的 \(\mathbb R\) 例子说明同一拓扑可由一个完备度量和一个不完备度量产生。</p>
<p>度量空间必 Hausdorff、第一可数、正规。因而余有限拓扑（无限集合）不可度量化，因为它不是 Hausdorff；不可数集合的余可数拓扑也不可度量化，因为它不是第一可数。这些是快速排除法，不是可度量化的完整刻画。</p>

<h2 id="section-52">二、Urysohn 可度量化定理的构造</h2>
<p><strong>Urysohn metrization theorem：</strong>若 \(X\) 是第二可数且正则的 \(T_1\) 空间，则 \(X\) 可度量化。第二可数提供可数多个局部测试，正则性使一个点及其邻域可以用连续实值函数区分。</p>
<p>证明的主线如下。先由第二可数与正则性得到正规性；再选可数个基对 \((U_n,V_n)\)，使 \(\overline{U_n}\subseteq V_n\)。Urysohn lemma 为每对构造连续函数 \(f_n:X\to[0,1]\)，在 \(\overline{U_n}\) 上为 0，在 \(X\setminus V_n\) 上为 1。基与正则性保证：给定 \(x\) 及其开邻域 \(W\)，有某个 \(n\) 使 \(x\in U_n\subseteq\overline{U_n}\subseteq V_n\subseteq W\)。</p>
<p>把这些函数组成 \(F:X\to[0,1]^{\mathbb N}\)、\(F(x)=(f_1(x),f_2(x),\ldots)\)。它连续，因为各坐标连续；上面的基对还保证 \(F\) 区分点，并且目标中的坐标开集能拉回成 \(x\) 的任意小邻域，所以 \(F\) 是拓扑嵌入。目标积空间可由
\[d(a,b)=\sum_{n=1}^{\infty}2^{-n}|a_n-b_n|\]
度量化，限制到 \(F(X)\) 就给 \(X\) 所需的度量。这里的核心不是凭空猜一个距离，而是先用可数个连续函数记录拓扑信息，再从函数值构造距离。</p>

<h2 id="section-53">三、函数空间为什么也需要拓扑？</h2>
<p>写 \(C(X,Y)\) 表示从 \(X\) 到 \(Y\) 的连续映射集合。若想讨论一族函数是否连续地变化，必须在 \(C(X,Y)\) 上指定拓扑。最常用的<strong>紧开拓扑</strong>以集合
\[[K,U]=\{f\in C(X,Y):f(K)\subseteq U\}\]
为子基，其中 \(K\subseteq X\) 紧、\(U\subseteq Y\) 开。它表示“整个紧集 \(K\) 的像都落入目标开集 \(U\)”这一有限范围的控制。</p>
<p>如果 \(X\) 紧、\(Y\) 有度量，紧开拓扑通常对应一致收敛拓扑：用 \(\sup_{x\in X}d_Y(f(x),g(x))\) 衡量函数距离。若 \(X\) 不紧，紧开拓扑对应的是在每个紧子集上一致收敛，而不是对全体 \(X\) 一致收敛。例如 \(f_n(x)=x/n\) 在 \(\mathbb R\) 的每个有界闭区间上一致收敛到零，却不在整个 \(\mathbb R\) 上一致收敛。</p>

<h2 id="section-54">四、评价映射与指数律</h2>
<p>评价映射把函数和点送到函数值：\(\mathrm{ev}:C(X,Y)\times X\to Y\)、\((f,x)\mapsto f(x)\)。若 \(X\) 局部紧且 Hausdorff，紧开拓扑使评价映射连续。关键在于 \(x\) 附近能找到闭包紧的小邻域 \(V\)；若 \(f(x)\in U\)，可缩小 \(V\) 使 \(f(\overline V)\subseteq U\)，于是 \([\overline V,U]\times V\) 是映入 \(U\) 的邻域。</p>
<p>同一假设下，\(F:Z\times X\to Y\) 连续，等价于它的“转置” \(\widehat F:Z\to C(X,Y)\)、\(\widehat F(z)(x)=F(z,x)\) 连续。这就是紧开拓扑的指数律。它解释为何同伦可以看成到函数空间的一条路径，也说明函数空间不是为了多引入一个集合，而是为了把“连续变化的映射”纳入拓扑学。</p>

<h2 id="section-55">五、理解检查</h2>
<ol><li>说明 \(d(a,b)=\sum2^{-n}|a_n-b_n|\) 为什么收敛，并写出它的基本开球怎样控制前面有限个坐标。</li><li>证明 \(f_n(x)=x/n\) 在每个紧区间上一致收敛，却不在 \(\mathbb R\) 上一致收敛。</li><li>在 \(C([0,1],\mathbb R)\) 中解释 \([K,U]\) 是怎样约束一整段函数图像的。</li></ol>
<p class="note-footer">Urysohn 定理的条件与构造依据林剑峰 / 陈晓《拓扑学讲义》第 8 章；函数空间部分参照标准紧开拓扑的定义，局部紧性将在下一篇展开。</p>
</section>
