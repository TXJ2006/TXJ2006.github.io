---
title: 第三单元 8：左右逆与伪逆｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-3-session-08/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit3">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-3-session-07/">上一讲</a></p>
<div class="note-lead"><p>方阵可逆时，\(A^{-1}\) 把输出完全还原。长方形或秩不足时，信息可能被压掉，也可能某些目标根本到不了。本讲先分清左逆和右逆各能保证什么，再用 SVD 定义对任意矩阵都适用的伪逆。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-3-session-8-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-3-session-8-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-3-session-8-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="one-sided">一、左逆保证唯一，右逆保证可达</h2>
<p>设 \(A\) 为 \(m\times n\)。若其 \(n\) 列无关，即秩 \(r=n\)，则 \(L=(A^TA)^{-1}A^T\) 满足 \(LA=I_n\)，称<strong>左逆</strong>。因此 \(Ax=b\) 一旦有解，左乘 \(L\) 就得唯一的 \(x=Lb\)。但 \(AL=A(A^TA)^{-1}A^T\) 通常不是 \(I_m\)，而是把目标 \(b\) 投到列空间；列空间以外的目标仍到不了。</p>
<p>若 \(A\) 的 \(m\) 行无关，即秩 \(r=m\)，则 \(R=A^T(AA^T)^{-1}\) 满足 \(AR=I_m\)，称<strong>右逆</strong>。每个 \(b\in\mathbb R^m\) 都可由 \(x=Rb\) 达到，但若 \(n\gt m\)，零空间非零，解不唯一。此时 \(RA\) 是到行空间 \(C(A^T)\) 的正交投影。只有 \(m=n=r\) 时，左右逆合为普通双侧逆。</p>

<h2 id="example">二、同一个方程组为什么有无穷多解？</h2>
<p>习题 32.1 的 \(A=\begin{pmatrix}1&0&1\\0&1&0\end{pmatrix}\) 把 \(x=(x_1,x_2,x_3)^T\) 送到 \((x_1+x_3,x_2)^T\)。它有满行秩，所以右逆可简单选 \(R=\begin{pmatrix}1&0\\0&1\\0&0\end{pmatrix}\)；对 \(b=(2,3)^T\) 给出解 \((2,3,0)^T\)。但全部解是 \((1+t,3,1-t)^T\)，因为只固定了第一、三分量的和。</p>
<p>上面那个简单右逆偏向把第三分量置零。若要在所有解中找<strong>长度最小</strong>的一个，应让第一、三分量平分总和：\((1,3,1)^T\)。确实其长度平方为 \((1+t)^2+9+(1-t)^2=11+2t^2\)，在 \(t=0\) 最小。由公式 \(A^T(AA^T)^{-1}\) 得到的右逆恰是 \(\begin{pmatrix}1/2&0\\0&1\\1/2&0\end{pmatrix}\)，它选择这个最短解。</p>

<h2 id="pseudo">三、秩不足时，伪逆先投影再还原</h2>
<p>一般矩阵可能同时有非零的零空间与左零空间。关键是把输入限制在行空间 \(C(A^T)=N(A)^\perp\)：若其中的 \(x\) 还满足 \(Ax=0\)，它同时属于行空间和零空间，只能是零向量。反过来，每个可达输出 \(b=Ax\) 都能去掉 \(x\) 的零空间分量，找到唯一一个行空间中的输入。因此 \(A\) 在“行空间到列空间”之间确实可逆，伪逆就是把这个反向对应扩展到所有目标 \(b\)。</p>
<p>SVD 写成 \(A=U\Sigma V^T\)，其中正奇异值为 \(\sigma_1,\ldots,\sigma_r\)。构造 \(n\times m\) 的 \(\Sigma^+\)：把前 \(r\) 个正奇异值换成倒数 \(1/\sigma_i\)，其余位置保留零。定义 <strong>Moore–Penrose 伪逆</strong></p>
<div class="equation">\[A^+=V\Sigma^+U^T.\]</div>
<p>它把到不了的输出分量先舍去，把到得了的分量沿 \(u_i\) 取坐标、除以 \(\sigma_i\)，再沿 \(v_i\) 重建。矩阵 \(AA^+\) 是到 \(C(A)\) 的正交投影；\(A^+A\) 是到 \(C(A^T)\) 的正交投影。它们一般不等于各自的单位矩阵，这精确表达了“只能还原没有丢失的部分”。</p>
<p>例如第 5 讲的秩一矩阵 \(B=\begin{pmatrix}4&3\\8&6\end{pmatrix}=(1,2)^T(4,3)\)。这里 \(u=(1,2)^T/\sqrt5\)、\(v=(4,3)^T/5\)、\(\sigma=5\sqrt5\)，所以 \(B^+=v\sigma^{-1}u^T=\frac1{125}\begin{pmatrix}4&8\\3&6\end{pmatrix}\)。直接算 \(BB^+\) 与 \(B^+B\)，分别是到 \(\operatorname{span}\{(1,2)^T\}\) 和 \(\operatorname{span}\{(4,3)^T\}\) 的投影。</p>

<h2 id="least-squares">四、把无解与多解一次处理完</h2>
<p>给定任意目标 \(b\)，令 \(x_*=A^+b\)。其输出 \(Ax_*=AA^+b\) 是列空间里离 \(b\) 最近的点，因此 \(x_*\) 使 \(\|Ax-b\|\) 最小。在所有得到同一个最近输出的 \(x\) 中，\(x_*\) 又位于行空间，与零空间正交，因此长度最小。其他最小二乘解恰为 \(x_*+z\)，其中 \(z\in N(A)\)，由勾股定理 \(\|x_*+z\|^2=\|x_*\|^2+\|z\|^2\)。</p>
<p>满列秩时，\(A^+=(A^TA)^{-1}A^T\)，回到正规方程；满行秩时，\(A^+=A^T(AA^T)^{-1}\)，回到最短精确解；方阵可逆时，\(A^+=A^{-1}\)。这三种旧公式是同一个 SVD 构造在不同秩条件下的特例。</p>
<h2 id="check">理解检查</h2>
<ol><li>对上面的 \(A\)，分别乘出 \(AR\)、\(AA^+\)、\(A^+A\)，说明哪些是单位矩阵，余下一个投到哪里。</li><li>习题 32.2：秩一矩阵 \(B\) 有左逆或右逆吗？为什么仍有上面给出的伪逆？</li><li>若 \(b\perp C(A)\)，则 \(A^+b\) 应是多少？从 SVD 和最小二乘目标分别解释。</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit III, Session 8 <em>Left and Right Inverses; Pseudoinverse</em> 整理。用“最近输出、最短输入”概括伪逆的两个最优化性质。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
