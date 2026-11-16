---
title: 第三单元 6：线性映射与矩阵表示｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-3-session-06/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit3">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-3-session-05/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-3-session-07/">下一讲 →</a></p>
<div class="note-lead"><p>前面的投影、旋转、求导、SVD 都在把输入送到输出。矩阵是这些作用在选定基下的数字记录；作用本身叫线性映射。本讲把“映射”“坐标”“矩阵”三个层次分开，解释为什么只知道基向量的去向就够了。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-3-session-6-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-3-session-6-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-3-session-6-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="linearity">一、线性的要求究竟有多强？</h2>
<p>映射 \(T:V\to W\) 从向量空间 \(V\) 接受输入，在向量空间 \(W\) 给出输出。它<strong>线性</strong>是指对任意 \(v,w\in V\) 和标量 \(a,b\)，都有 \(T(av+bw)=aT(v)+bT(w)\)。这同时包含加法与数乘的保持，并立刻推出 \(T(0)=0\)。</p>
<p>例如把平面上的点绕原点旋转 \(45^\circ\) 是线性的：两个箭头的和整体旋转，仍等于分别旋转后相加。沿过原点直线的正交投影也线性。相反，\(T(v)=v+v_0\) 若 \(v_0\ne0\)，则 \(T(0)=v_0\)；它是平移，不线性。长度函数 \(v\mapsto\|v\|\) 也不线性，因为 \(\|-v\|=\|v\|\) 却应等于 \(-\|v\|\)。仅有 \(T(0)=0\) 也不足以保证线性，例如 \(T(s,t)=(s^2,t)\)。</p>

<h2 id="basis-columns">二、为什么矩阵的列是“基向量的像”？</h2>
<p>选输入空间的一组基 \(v_1,\ldots,v_n\)，再选输出空间的一组基 \(w_1,\ldots,w_m\)。对每个 \(j\)，把 \(T(v_j)\) 按输出基展开：\(T(v_j)=\sum_{i=1}^m a_{ij}w_i\)。令 \(A=(a_{ij})\)，则其第 \(j\) 列就是 \(T(v_j)\) 的<strong>输出坐标</strong>，而非不加说明的几何箭头本身。</p>
<p>任意输入 \(v=\sum_jc_jv_j\) 的输入坐标为 \([v]_V=(c_1,\ldots,c_n)^T\)。线性性使</p>
<div class="equation">\[[T(v)]_W=A[v]_V.\]</div>
<p>左边是输出相对于 \(w_i\) 的坐标，右边是矩阵乘输入坐标。只有两边都选标准基，才可把坐标列直接视为通常的向量列。这个区分能防止把不同基下的数字硬相乘。</p>

<h2 id="examples">三、同一个作用在不同基下长什么样？</h2>
<p>投影到直线 \(y=x\)。在标准基下，投影矩阵 \(P=\frac12\begin{pmatrix}1&1\\1&1\end{pmatrix}\)。若输入输出都改用标准正交基 \(q_1=(1,1)^T/\sqrt2\)、\(q_2=(1,-1)^T/\sqrt2\)，便有 \(T(q_1)=q_1\)、\(T(q_2)=0\)，矩阵变成 \(\operatorname{diag}(1,0)\)。投影本身没有改变，变的是描述它的坐标。</p>
<p>再看求导 \(D:\mathcal P_2\to\mathcal P_1\)，其中 \(\mathcal P_k\) 是次数不超过 \(k\) 的实多项式空间。输入基取 \((1,x,x^2)\)，输出基取 \((1,x)\)。因为 \(D(1)=0,D(x)=1,D(x^2)=2x\)，所以</p>
<div class="equation">\[[D]=\begin{pmatrix}0&1&0\\0&0&2\end{pmatrix},\qquad [D(c_0+c_1x+c_2x^2)]_{(1,x)}=\begin{pmatrix}c_1\\2c_2\end{pmatrix}.\]</div>
<p>这里 \([D]\) 是 \(2\times3\)：求导不是同一二维或三维空间上的方阵作用，因而不应无条件谈它的特征值。</p>

<h2 id="composition">四、矩阵相乘从哪里来？</h2>
<p>若 \(T:V\to W\) 的矩阵是 \(A\)，\(S:W\to Z\) 的矩阵是 \(B\)，并且 \(T\) 的输出基与 \(S\) 的输入基<strong>一致</strong>，则 \([S(T(v))]=B(A[v])=(BA)[v]\)。所以复合映射对应 \(BA\)：右边的 \(A\) 先作用。若中间两次选用不同基，必须先插入换基矩阵，不能省略。</p>
<h2 id="check">理解检查</h2>
<ol><li>习题 30.1 的极坐标规则 \((r,\theta)\mapsto(2r,\theta)\) 对应笛卡尔坐标下哪个矩阵？为什么不能因为表达式出现 \(\theta\) 就断定非线性？</li><li>写出 \(D(x^2+3x+5)\) 的输出坐标，与矩阵乘法核对。</li><li>若把上面投影的输入改回标准基、输出仍用 \((q_1,q_2)\)，矩阵的第一列应怎样求？</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit III, Session 6 <em>Linear Transformations and Their Matrices</em> 整理。明确区分抽象向量、坐标列和矩阵。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
