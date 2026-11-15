---
title: 第二单元 7：Cramer 法则、逆矩阵与体积｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-2-session-07/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit2">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-06/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-08/">下一讲 →</a></p>
<div class="note-lead"><p>余子式不只是一种展开行列式的办法。把所有余子式按位置排好、再转置，就能构造逆矩阵；用列线性还能推出 Cramer 法则。最后回到几何：行列式的绝对值到底测量什么？</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-2-session-7-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-2-session-7-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-2-session-7-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="adjugate">一、余子式矩阵为什么要转置？</h2>
<p>设 \(A\) 是 \(n\times n\) 方阵，沿用上讲的 \(C_{ij}=(-1)^{i+j}\det M_{ij}\)。把所有 \(C_{ij}\) 按原位置排成矩阵 \(\mathcal C=(C_{ij})\)，称余子式矩阵。要理解公式，先看乘积 \(A\mathcal C^T\) 的第 \(i,k\) 个数：</p>
<div class="equation">\[(A\mathcal C^T)_{ik}=\sum_{j=1}^{n}a_{ij}C_{kj}.\]</div>
<p>若 \(i=k\)，这正是沿第 \(i\) 行展开 \(A\) 的行列式，结果为 \(\det A\)。若 \(i\ne k\)，把 \(A\) 的第 \(k\) 行换成第 \(i\) 行，再沿第 \(k\) 行展开，得到的也正是这个和；但新矩阵有两行相同，所以和为零。于是</p>
<div class="equation">\[A\mathcal C^T=(\det A)I_n.\]</div>
<p>乘积中的第 \(k\) 列需要的是<strong>原矩阵第 \(k\) 行</strong>的余子式，因此必须转置。这是最容易漏掉的一步。若 \(\det A\ne0\)，除以它便得到</p>
<div class="equation">\[A^{-1}=\frac{\mathcal C^T}{\det A}.\]</div>
<p>例如 \(A=\begin{pmatrix}2&1\\1&1\end{pmatrix}\)，\(\det A=1\)，余子式矩阵为 \(\mathcal C=\begin{pmatrix}1&-1\\-1&2\end{pmatrix}\)，故 \(A^{-1}=\mathcal C^T=\mathcal C\)。直接相乘得单位矩阵。一般矩阵的 \(\mathcal C\) 未必对称；这个例子中转置后不变只是巧合。</p>

<h2 id="cramer">二、Cramer 法则为何是“替换一列”？</h2>
<p>设 \(Ax=b\) 且 \(\det A\ne0\)。把 \(A\) 的列写成 \(a_1,\ldots,a_n\)，则 \(b=x_1a_1+\cdots+x_na_n\)。令 \(B_j\) 是把 \(A\) 的<strong>第 \(j\) 列</strong>换成 \(b\) 得到的矩阵。行列式对这一列线性，所以 \(\det B_j\) 可按 \(b\) 的上述组合拆成 \(n\) 项；除第 \(j\) 项外，每项都与原矩阵已有某列重复，因此为零。留下</p>
<div class="equation">\[\det B_j=x_j\det A,\qquad x_j=\frac{\det B_j}{\det A}.\]</div>
<p>这就是 Cramer 法则，条件是 \(A\) 为可逆方阵；它没有声称对长方形或奇异矩阵也能这样除。上面的两阶例子若 \(b=(3,2)^T\)，则</p>
<div class="equation">\[\det\begin{pmatrix}3&1\\2&1\end{pmatrix}=1,\qquad
\det\begin{pmatrix}2&3\\1&2\end{pmatrix}=1.\]</div>
<p>故 \(x_1=x_2=1\)，核对 \(A(1,1)^T=(3,2)^T\)。Cramer 法则说明每个未知数与“替换一列后的体积”有关，但对大矩阵逐列计算行列式通常比消元费力；它更适于证明和理解，而非日常数值求解。</p>

<h2 id="volume">三、行列式为什么是有向面积与体积？</h2>
<p>把二维矩阵 \(A=[u\ v]\) 的两列当成从原点出发的两条边。它们围成平行四边形；面积是 \(|\det A|\)。绝对值去掉朝向，正负号则区分两条边的排列方向。若列相关，图形压成一条线，面积与行列式都为零。</p>
<figure class="note-figure"><svg viewBox="0 0 560 320" role="img" aria-labelledby="volume-title volume-desc"><title id="volume-title">由两列向量围成的平行四边形</title><desc id="volume-desc">从原点出发的向量 u=(2,1) 和 v=(1,1) 围成面积为 1 的平行四边形，另外两个顶点为 u+v 与原点。</desc><path d="M65 270H475M95 287V28" fill="none" stroke="#87958f" stroke-width="2"/><polygon points="95,260 255,180 335,100 175,180" fill="#b0c8bb" opacity=".42" stroke="#557f79" stroke-width="2"/><path d="M95 260L255 180M95 260L175 180" stroke="#557f79" stroke-width="3"/><g fill="#557f79"><circle cx="95" cy="260" r="4"/><circle cx="255" cy="180" r="5"/><circle cx="175" cy="180" r="5"/><circle cx="335" cy="100" r="5"/></g><text x="72" y="283">O</text><text x="265" y="193">u=(2,1)</text><text x="108" y="173">v=(1,1)</text><text x="345" y="100">u+v</text><text x="212" y="142">面积 1</text></svg><figcaption>两列 \(u=(2,1)^T,\ v=(1,1)^T\) 张成平行四边形；\(\det[u\ v]=2\cdot1-1\cdot1=1\)。</figcaption></figure>
<p>为什么这一面积服从前几讲的行列式规则？单位矩阵的两列围成单位正方形，面积为 1；交换两列只改变朝向；一列乘 \(t\) 使有向面积乘 \(t\)；给一列加另一列的倍数，相当于剪切，底和高不变。因而它与行列式的基本性质一致。三维时 \(|\det A|\) 是三列围成的平行六面体体积；\(n\) 维可理解为对应的体积尺度。</p>
<p>讲义习题 20.2 把这个观点用于球坐标。坐标变换在一点附近近似一个线性映射，其导数矩阵的行列式给局部体积的放大倍数；球坐标的绝对值为 \(\rho^2\sin\phi\)。这不是说曲面上的小块严格就是一个直角盒子，而是在尺度趋于零时由线性近似给出的体积因子。</p>
<h2 id="check">理解检查</h2>
<ol><li>对习题 20.1 的三阶矩阵，先求余子式矩阵第一行，再直接乘 \(A\mathcal C^T\) 的第一行，核对非对角项为什么为零。</li><li>为什么把 \(b\) 替换进第 \(j\) 列以后，\(x_i a_i\) 在 \(i\ne j\) 时一定产生重复列？</li><li>把图中的第二列换成 \(u+v\)，平行四边形形状变了，面积与行列式为什么不变？</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit II, Session 7 <em>Cramer's Rule, Inverse Matrix and Volume</em> 整理。逆矩阵转置位置、替列证明与体积图均逐步核对。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
