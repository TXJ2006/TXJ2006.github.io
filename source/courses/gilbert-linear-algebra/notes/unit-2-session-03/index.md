---
title: 第二单元 3：投影矩阵与最小二乘｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-2-session-03/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit2">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-02/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-04/">下一讲 →</a></p>
<div class="note-lead"><p>上一讲找到 \(b\) 在列空间中的最近点 \(p\)。本讲回答两个更具体的问题：怎样把“最近”写成一个可检验的优化条件？三组互不完全吻合的数据，怎样算出一条真正的最小二乘直线？</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-2-session-3-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-2-session-3-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-2-session-3-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="what-is-minimized">一、“最小二乘”究竟最小化什么？</h2>
<p>给定 \(A\in\mathbb R^{m\times n}\) 和 \(b\in\mathbb R^m\)，我们未必能解 \(Ax=b\)，因为 \(b\) 可能不在列空间 \(C(A)\) 中。仍可选择 \(x\)，使输出 \(Ax\) 尽量靠近 \(b\)。这里的误差是 \(e(x)=b-Ax\)，目标是使<strong>误差长度的平方</strong>最小：</p>
<div class="equation">\[F(x)=\|b-Ax\|^2=\sum_{i=1}^{m}\bigl(b_i-(Ax)_i\bigr)^2.\]</div>
<p>这不是要求每条方程都尽量精确到同一程度，而是把所有残差平方后相加。设最近的输出为 \(p=A\widehat x\)，剩余误差为 \(e=b-p\)。上一讲已证明 \(e\perp C(A)\)，所以 \(A^Te=0\)。这给出正规方程 \(A^TA\widehat x=A^Tb\)。</p>
<p>不借助微积分也能看出它为什么给出最小值。对任意其他 \(x\)，</p>
<div class="equation">\[b-Ax=e+A(\widehat x-x),\qquad e\perp A(\widehat x-x).\]</div>
<p>于是 \(\|b-Ax\|^2=\|e\|^2+\|A(\widehat x-x)\|^2\ge\|e\|^2\)。若 \(A\) 的列无关，等号只在 \(x=\widehat x\) 时成立；若列相关，最小的<strong>输出</strong>仍唯一，达到它的系数 \(x\) 却可能不唯一。</p>

<h2 id="line-fit">二、把三组数据写成一个方程组</h2>
<p>讲义的三个观测点是 \((t,b)=(1,1),(2,2),(3,2)\)。想用直线 \(b\approx C+Dt\) 描述它们：\(C\) 是截距，\(D\) 是斜率，这里的字母 \(C\) 只是一个实数，不是列空间 \(C(A)\)。若直线经过所有点，应同时满足 \(C+D=1,\ C+2D=2,\ C+3D=2\)；前两式给 \(C=0,D=1\)，却使第三式左边成为 3 而非 2，因此无精确解。</p>
<p>把未知数放进 \(x=(C,D)^T\)，三次观测放进 \(b=(1,2,2)^T\)，就得到</p>
<div class="equation">\[A=\begin{pmatrix}1&1\\1&2\\1&3\end{pmatrix},\qquad Ax=\begin{pmatrix}C+D\\C+2D\\C+3D\end{pmatrix}.\]</div>
<p>矩阵第一列表示“每个预测值都加同一个截距”，第二列表示“斜率按 \(t=1,2,3\) 改变预测值”。我们求的是 \(b\) 在这两个列方向张成的平面中的投影，而不是强迫 \(Ax=b\)。</p>
<figure class="note-figure"><svg viewBox="0 0 560 325" role="img" aria-labelledby="fit-title fit-desc"><title id="fit-title">三组数据的最小二乘直线</title><desc id="fit-desc">点为 1,1 与 2,2 与 3,2；拟合直线为 2/3 加 t/2；三条竖线显示观测值与直线预测值的残差。</desc><path d="M74 268H505M74 268V27" fill="none" stroke="#87958f" stroke-width="2"/><path d="M90 190L484 86" fill="none" stroke="#557f79" stroke-width="3"/><g stroke="#bb6a4b" stroke-width="3"><path d="M150 188V174"/><path d="M290 112V137"/><path d="M430 112V100"/></g><g fill="#bb6a4b"><circle cx="150" cy="188" r="5"/><circle cx="290" cy="112" r="5"/><circle cx="430" cy="112" r="5"/></g><g fill="#557f79"><circle cx="150" cy="174" r="4"/><circle cx="290" cy="137" r="4"/><circle cx="430" cy="100" r="4"/></g><text x="145" y="291">1</text><text x="285" y="291">2</text><text x="425" y="291">3</text><text x="497" y="286">t</text><text x="47" y="191">1</text><text x="47" y="116">2</text><text x="350" y="63">拟合直线</text><text x="97" y="39">红点：观测值</text></svg><figcaption>橙色竖段表示观测值与预测值的差。这里最小化的是<strong>竖直残差平方和</strong>，不是点到平面图中直线的垂直距离平方和。</figcaption></figure>

<h2 id="normal-equations">三、逐步解出直线与误差</h2>
<p>先做两个小矩阵乘法，\(A^TA=\begin{pmatrix}3&6\\6&14\end{pmatrix}\)，\(A^Tb=(5,11)^T\)。正规方程就是</p>
<div class="equation">\[3C+6D=5,\qquad 6C+14D=11.\]</div>
<p>第二式减去第一式的两倍，得到 \(2D=1\)，所以 \(D=\frac12\)；代回得 \(C=\frac23\)。最小二乘直线为 \(\widehat b(t)=\frac23+\frac12t\)。在三个时间点，它给出的预测值与残差分别是</p>
<div class="equation">\[p=A\widehat x=\begin{pmatrix}7/6\\5/3\\13/6\end{pmatrix},\qquad e=b-p=\begin{pmatrix}-1/6\\1/3\\-1/6\end{pmatrix}.\]</div>
<p>可直接验算 \(A^Te=0\)：第一列给三项残差之和 \(-1/6+1/3-1/6=0\)；第二列给按时间加权的和 \(1(-1/6)+2(1/3)+3(-1/6)=0\)。这两条等式不是巧合，正是“误差垂直于 \(A\) 的两个列方向”的具体含义。误差平方和为 \(1/36+1/9+1/36=1/6\)。</p>

<h2 id="two-projections">四、投影矩阵与余下的误差</h2>
<p>列无关时 \(P=A(A^TA)^{-1}A^T\)，因此 \(p=Pb\)、\(e=(I-P)b\)。\(P\) 把向量投到 \(C(A)\)；\(I-P\) 把向量投到正交补 \(N(A^T)\)。两者相加等于 \(I\)，输出又互相垂直。故每个 \(b\in\mathbb R^m\) 都可唯一拆为</p>
<div class="equation">\[b=\underbrace{Pb}_{\text{可由 }A\text{ 的列组成}}+\underbrace{(I-P)b}_{\text{与列空间正交}}.\]</div>
<p>特别地，若 \(b\in C(A)\)，则 \(Pb=b\)，残差为零；若 \(b\in N(A^T)\)，则 \(Pb=0\)，最佳预测只能是零向量。习题 16.2–16.5 正是用一组数据检验这两种极端情形，以及 \(p,e,\widehat x\) 分别属于哪个空间。注意 \(p,e\in\mathbb R^m\)，而 \(\widehat x\in\mathbb R^n\)，不能因它们都出现在同一道题里就混作同一种向量。</p>
<h2 id="check">理解检查</h2>
<ol><li>为什么图中的残差要沿竖直方向画？如果改为最小化观测点到直线的几何垂距，目标函数还是 \(\|b-Ax\|^2\) 吗？</li><li>不计算 \(P\) 的全部九个元素，只利用 \(p,e\) 的具体数值检验 \(Pp=p\)、\(Pe=0\)。</li><li>习题 16.1 改用三点 \((-1,7),(1,7),(2,21)\)。先写出它的设计矩阵两列各代表什么，再求最近直线。</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit II, Session 3 <em>Projection Matrices and Least Squares</em> 整理。数值计算沿用讲义数据，图中特别区分竖直残差与点到直线的几何垂距。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
