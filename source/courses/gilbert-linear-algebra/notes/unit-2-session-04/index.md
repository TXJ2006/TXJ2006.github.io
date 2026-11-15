---
title: 第二单元 4：正交矩阵、Gram–Schmidt 与 QR｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-2-session-04/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit2">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-03/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-05/">下一讲 →</a></p>
<div class="note-lead"><p>投影的公式 \(A(A^TA)^{-1}A^T\) 看起来复杂，是因为原来的列向量不一定互相垂直。若把同一列空间改用<strong>标准正交基</strong>表示，\(A^TA\) 会变成单位矩阵。Gram–Schmidt 做的是这次换基，QR 分解则把它写成矩阵等式。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-2-session-4-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-2-session-4-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-2-session-4-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="orthonormal">一、标准正交为什么让坐标容易求？</h2>
<p>一组向量 \(q_1,\ldots,q_r\in\mathbb R^m\) 称为<strong>标准正交</strong>，若 \(q_i^Tq_j=0\)（\(i\ne j\)），且每个 \(q_i^Tq_i=1\)。也就是两两垂直、每个长度为 1。把它们排成 \(m\times r\) 矩阵 \(Q=[q_1\ \cdots\ q_r]\)，这些等式合并为</p>
<div class="equation">\[Q^TQ=I_r.\]</div>
<p>这里 \(I_r\) 是 \(r\times r\) 单位矩阵。若 \(p\in C(Q)\)，写成 \(p=Qc=\sum_i c_iq_i\)，左乘 \(q_j^T\) 后，其余项因垂直而消失，只剩 \(c_j=q_j^Tp\)。矩阵写法就是 \(c=Q^Tp\)。对任意 \(b\in\mathbb R^m\)，\(Q^Tb\) 给出投到 \(C(Q)\) 后的系数，投影本身为 \(QQ^Tb\)。</p>
<p>当 \(m>r\)，\(Q\) 是长方形，虽然 \(Q^TQ=I_r\)，但 \(QQ^T\) 通常<strong>不是</strong> \(I_m\)；它只保留 \(C(Q)\) 中的分量。只有 \(Q\) 是方阵且列标准正交时，才有 \(Q^{-1}=Q^T\) 和 \(QQ^T=I\)，这时称 \(Q\) 为<strong>正交矩阵</strong>。例如二维旋转矩阵保持长度与角度；长方形 \(Q\) 不能当作可逆方阵使用。</p>

<h2 id="gram-schmidt">二、Gram–Schmidt 每一步在减掉什么？</h2>
<p>从无关的 \(a_1,\ldots,a_r\) 出发，先把第一列变成单位向量 \(q_1=a_1/\|a_1\|\)。第二列若直接归一化，仍会与 \(q_1\) 不垂直，所以先减去它沿 \(q_1\) 的投影：</p>
<div class="equation">\[u_2=a_2-(q_1^Ta_2)q_1,\qquad q_2=\frac{u_2}{\|u_2\|}.\]</div>
<p>因为 \(q_1^Tu_2=q_1^Ta_2-(q_1^Ta_2)q_1^Tq_1=0\)，新向量确实垂直于 \(q_1\)。而 \(u_2=a_2-\text{旧方向的一部分}\)，所以 \(a_1,a_2\) 与 \(q_1,q_2\) 张成<strong>同一个平面</strong>。第三列则须同时减掉它在前两个方向上的分量。一般地，</p>
<div class="equation">\[u_j=a_j-\sum_{i=1}^{j-1}(q_i^Ta_j)q_i,\qquad q_j=\frac{u_j}{\|u_j\|}.\]</div>
<p>如果某一步 \(u_j=0\)，原来的 \(a_j\) 已在前面向量的张成空间内，不能再归一化。这正是输入列相关时算法会暴露的问题；本讲先假定原列无关。</p>

<h2 id="worked-example">三、把讲义的两列真正做完</h2>
<p>取 \(a_1=(1,1,1)^T,\ a_2=(1,0,2)^T\)。第一列的长度是 \(\sqrt3\)，所以 \(q_1=(1,1,1)^T/\sqrt3\)。先不急于归一化第二列：\(a_1^Ta_2=3,\ a_1^Ta_1=3\)，故 \(a_2\) 在 \(a_1\) 方向的投影恰为 \(a_1\)。减掉后</p>
<div class="equation">\[u_2=a_2-a_1=\begin{pmatrix}0\\-1\\1\end{pmatrix},\qquad q_2=\frac1{\sqrt2}\begin{pmatrix}0\\-1\\1\end{pmatrix}.\]</div>
<p>核对 \(q_1^Tq_2=(0-1+1)/\sqrt6=0\)，两者长度都为 1。于是</p>
<div class="equation">\[Q=\begin{pmatrix}1/\sqrt3&0\\1/\sqrt3&-1/\sqrt2\\1/\sqrt3&1/\sqrt2\end{pmatrix},\qquad Q^TQ=I_2.\]</div>
<p>这两列与原来的 \(a_1,a_2\) 张成同一平面。若把 \(b\) 投到这个平面，先算两个内积 \(q_1^Tb,q_2^Tb\)，再组成 \(p=(q_1^Tb)q_1+(q_2^Tb)q_2=QQ^Tb\)，不再需要求 \((A^TA)^{-1}\)。</p>

<h2 id="qr">四、QR 分解保存了哪些信息？</h2>
<p>设原矩阵 \(A=[a_1\ \cdots\ a_r]\)，正交化得到 \(Q=[q_1\ \cdots\ q_r]\)。每个 \(a_j\) 都由前 \(j\) 个 \(q_i\) 组合而成，所以有 \(A=QR\)，其中 \(R\) 是 \(r\times r\) 上三角矩阵。由 \(Q^TQ=I\)，左乘 \(Q^T\) 得 \(R=Q^TA\)。\(R\) 的第 \(i,j\) 个数 \(q_i^Ta_j\)，记录原来的第 \(j\) 列在新方向 \(q_i\) 上有多少；当 \(i>j\) 时为零。</p>
<p>在刚才的例子中，\(a_1=\sqrt3\,q_1\)，\(a_2=\sqrt3\,q_1+\sqrt2\,q_2\)，所以</p>
<div class="equation">\[A=QR,\qquad R=\begin{pmatrix}\sqrt3&\sqrt3\\0&\sqrt2\end{pmatrix}.\]</div>
<p>这与第一单元的 LU 分解不同：LU 通过行消元得到三角结构；QR 保持列空间，用正交方向表达原来的列。实际计算最小二乘时，若 \(A=QR\) 且列无关，正规方程可化为 \(R\widehat x=Q^Tb\)，只需解一个上三角方程。</p>
<details class="note-extension"><summary>为什么最后一步是 \(R\widehat x=Q^Tb\)？</summary><p>最近点为 \(p=QQ^Tb\)，同时又是 \(A\widehat x=QR\widehat x\)。于是 \(QR\widehat x=QQ^Tb\)。左乘 \(Q^T\)，利用 \(Q^TQ=I\)，得到 \(R\widehat x=Q^Tb\)。这不是把长方形 \(Q\) 当方阵求逆。</p></details>
<h2 id="check">理解检查</h2>
<ol><li>用讲义习题 17.1 的思路，若 \(Qx=0\)，左乘 \(Q^T\) 能得到什么？由此证明标准正交列一定无关。</li><li>在上面的 \(A=QR\) 中，直接乘回去，核对第二列仍是 \((1,0,2)^T\)。</li><li>习题 17.2 给出 \(\mathbb R^4\) 中三个相邻差向量。它们为什么都与 \((1,1,1,1)^T\) 垂直？正交化后张成空间会改变吗？</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit II, Session 4 <em>Orthogonal Matrices and Gram-Schmidt</em> 整理。讲义中投影矩阵一处乘法顺序排版有误；本文使用长方形 \(Q\) 的正确公式 \(P=QQ^T\)。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
