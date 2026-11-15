---
title: 第二单元 1：正交向量与正交子空间｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-2-session-01/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit2">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-12/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-02/">下一讲 →</a></p>
<div class="note-lead"><p>第一单元已经用秩数出了四个基本子空间的维数。本讲把其中两对空间的<strong>位置关系</strong>说清：零空间不是偶然与行空间维数互补，它恰好包含所有垂直于行空间的向量。这个事实将使“无解”转化为下一讲的“找最近的可解目标”。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-2-session-1-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-2-session-1-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-2-session-1-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="orthogonal-vectors">一、垂直为什么写成内积为零？</h2>
<p>在 \(\mathbb R^n\) 中，对同样有 \(n\) 个分量的向量 \(u,v\)，定义内积 \(u^Tv=\sum_{i=1}^n u_iv_i\)，长度为 \(\|u\|=\sqrt{u^Tu}\)。展开平方可得</p>
<div class="equation">\[\|u+v\|^2=\|u\|^2+2u^Tv+\|v\|^2.\]</div>
<p>若 \(u,v\) 是两条互相垂直的非零边，勾股定理说上式的中间项必须为零。因此在任意维度，用 \(u^Tv=0\) 定义<strong>正交</strong>，写作 \(u\perp v\)。例如 \((1,2,-1)^T\) 与 \((2,-1,0)^T\) 的内积是 \(2-2=0\)。零向量同所有向量的内积都是零，也被称为正交；但零向量没有方向，不能说它与谁“成 \(90^\circ\)”。</p>
<p>两个子空间 \(S,T\subseteq\mathbb R^n\) 正交，意思是<strong>任意</strong> \(s\in S\) 与<strong>任意</strong> \(t\in T\) 都满足 \(s^Tt=0\)。只找到一对垂直向量远远不够。若 \(S\) 由 \(s_1,\ldots,s_r\) 张成、\(T\) 由 \(t_1,\ldots,t_k\) 张成，检查全部 \(s_i^Tt_j=0\) 就够了，因为内积对线性组合是线性的。</p>

<h2 id="four-spaces">二、四个空间如何成对正交？</h2>
<p>设 \(A\) 是 \(m\times n\) 实矩阵。它的行空间 \(C(A^T)\) 与零空间 \(N(A)\) 都在 \(\mathbb R^n\)；列空间 \(C(A)\) 与左零空间 \(N(A^T)\) 都在 \(\mathbb R^m\)。只有先把所在空间配对，谈垂直才有意义。</p>
<p>若 \(x\in N(A)\)，则 \(Ax=0\)。这个列向量的第 \(i\) 个分量，就是 \(A\) 的第 \(i\) 行与 \(x\) 的内积，所以 \(x\) 与每一行都垂直，进而与所有行的线性组合垂直。反过来，若 \(x\) 与整个行空间垂直，它就与每一行垂直，故 \(Ax=0\)。因此</p>
<div class="equation">\[N(A)=C(A^T)^\perp\quad\text{在 }\mathbb R^n;\qquad N(A^T)=C(A)^\perp\quad\text{在 }\mathbb R^m.\]</div>
<p>这里 \(S^\perp\) 是与 \(S\) 中<strong>所有</strong>向量正交的向量集合。第二个等式把第一个等式应用于 \(A^T\) 即得。它们不只是“某些向量碰巧垂直”，而是<strong>正交补</strong>的等式：已经把所有垂直向量找全了。第一对维数为 \(r\) 与 \(n-r\)，第二对为 \(r\) 与 \(m-r\)，各自恰好填满所在空间。</p>
<p>例如 \(A=\begin{pmatrix}1&2&5\\2&4&10\end{pmatrix}\)。它的两行同向，所以行空间只是 \(\operatorname{span}\{(1,2,5)^T\}\)；零空间是平面 \(x_1+2x_2+5x_3=0\)。向量 \((-2,1,0)^T\) 落在这个平面中，因为 \(-2+2=0\)。它也与行空间的生成向量内积为零。这是 \(\mathbb R^3\) 中一条直线和与它垂直的一个过原点平面。</p>

<h2 id="inconsistency">三、无解时，左零空间能指出矛盾</h2>
<p>方程 \(Ax=b\) 有解，当且仅当 \(b\in C(A)\)。由刚才的正交补等式，这又等价于 \(y^Tb=0\) 对每个 \(y\in N(A^T)\) 成立。理由很直接：若 \(b=Ax\)，则 \(y^Tb=y^TAx=(A^Ty)^Tx=0\)；反过来，若 \(b\) 与整个 \(N(A^T)\) 正交，它就属于 \(C(A)\)。</p>
<p>配套习题 16.1 的三条方程是 \(x_1-x_2=1,\ x_2-x_3=1,\ x_1-x_3=1\)。前两式相加，再减第三式，左端完全抵消，右端却成为 \(1+1-1=1\)，即 \(0=1\)。系数 \(y=(1,1,-1)^T\) 满足 \(A^Ty=0\)，但对右端 \(b=(1,1,1)^T\) 有 \(y^Tb=1\)。这个 \(y\) 就是一份可直接核对的<strong>无解证据</strong>。</p>
<div class="note-aside"><p>对任意 \(A,b\)，恰有一个成立：\(Ax=b\) 有解；或者存在 \(y\) 使 \(A^Ty=0\) 且 \(y^Tb\ne0\)。把后一种 \(y\) 除以 \(y^Tb\)，就可规定 \(y^Tb=1\)。这正是本讲习题中的 Fredholm's Alternative 在有限维情形的写法。</p></div>

<h2 id="ata">四、为什么下一讲总出现 \(A^TA\)？</h2>
<p>给定 \(A\in\mathbb R^{m\times n}\)，\(A^TA\) 是 \(n\times n\) 对称矩阵。关键恒等式是 \(x^TA^TAx=\|Ax\|^2\)。右侧非负，且只在 \(Ax=0\) 时为零。因此</p>
<div class="equation">\[A^TAx=0\ \Longrightarrow\ \|Ax\|^2=0\ \Longrightarrow\ Ax=0;\qquad Ax=0\ \Longrightarrow\ A^TAx=0.\]</div>
<p>于是 \(N(A^TA)=N(A)\)，秩也相同。特别地，\(A^TA\) 可逆，当且仅当 \(A\) 的列线性无关。这里不能因为 \(A^TA\) 是方阵就直接断言它可逆：若 \(A\) 有重复列，\(A^TA\) 仍有非零零空间。下一讲要用它把不可解的 \(Ax=b\) 改写为一个可解的投影问题。</p>
<h2 id="check">理解检查</h2>
<ol><li>为什么 \(N(A)\) 的向量有 \(n\) 个分量，却不能拿它与 \(C(A)\subseteq\mathbb R^m\) 直接配成一对正交补？</li><li>重新写出习题 16.1 的矩阵 \(A\)，验证 \(A^T(1,1,-1)^T=0\) 与 \(y^Tb=1\)。</li><li>若 \(A\) 的列相关，给出一个非零 \(x\) 同时属于 \(N(A)\) 与 \(N(A^TA)\)。</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit II, Session 1 <em>Orthogonal Vectors and Subspaces</em> 及习题 16.1 整理。正交补证明与无解证据均逐步展开；下讲进入正交投影。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
