---
title: 第三单元 5：奇异值分解｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-3-session-05/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit3">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-3-session-04/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-3-session-06/">下一讲 →</a></p>
<div class="note-lead"><p>对角化要求方阵有足够多特征向量；一般长方形矩阵连特征值都无从定义。奇异值分解（SVD）换一个问题：能否在输入和输出空间分别选标准正交坐标，让映射只沿对应方向缩放？答案对任意实矩阵都成立。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-3-session-5-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-3-session-5-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-3-session-5-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="construction">一、为何先研究 \(A^TA\)？</h2>
<p>设 \(A\) 是 \(m\times n\) 实矩阵，秩为 \(r\)。虽然 \(A\) 可能不对称，\(A^TA\) 一定是 \(n\times n\) 对称半正定矩阵。由谱定理取它的标准正交特征基 \(v_1,\ldots,v_n\)，并把正特征值排在前面：\(A^TAv_i=\sigma_i^2v_i\)（\(1\leq i\leq r\)），其中 \(\sigma_i\gt0\) 是正平方根，叫<strong>奇异值</strong>。</p>
<p>为什么这组输入方向的输出会互相垂直？若 \(i\ne j\)，则 \((Av_i)^T(Av_j)=v_i^TA^TAv_j=\sigma_j^2v_i^Tv_j=0\)；而 \(\|Av_i\|=\sigma_i\)。于是令 \(u_i=Av_i/\sigma_i\)，就得到标准正交的输出方向，并有 \(Av_i=\sigma_i u_i\)。对剩余 \(v_{r+1},\ldots,v_n\)，\(A^TAv_i=0\) 意味着 \(Av_i=0\)，所以它们构成零空间的一组标准正交基。</p>
<p>把 \(u_1,\ldots,u_r\) 补全为 \(\mathbb R^m\) 的标准正交基，令 \(U\) 的列为这些<strong>输出</strong>方向；令 \(V\) 的列为 \(v_i\)，即<strong>输入</strong>方向。令 \(\Sigma\) 是 \(m\times n\) 矩阵，前 \(r\) 个对角数为 \(\sigma_i\)、其余为零，就有</p>
<div class="equation">\[AV=U\Sigma,\qquad A=U\Sigma V^T.\]</div>
<p>读式子时从右往左：\(V^T\) 把输入写成合适坐标，\(\Sigma\) 缩放或抹去各分量，\(U\) 把输出坐标转回去。非零 \(v_i\) 张成行空间 \(C(A^T)\)，非零 \(u_i\) 张成列空间 \(C(A)\)；补上的方向分别属于 \(N(A)\) 与 \(N(A^T)\)。原讲义有一处把 \(U,V\) 对应的空间写反；按维数 \(U\in\mathbb R^{m\times m}\)、\(V\in\mathbb R^{n\times n}\) 也能发现问题。</p>

<h2 id="example">二、逐列核对一个 SVD</h2>
<p>取讲义的 \(A=\begin{pmatrix}4&4\\-3&3\end{pmatrix}\)。计算 \(A^TA=\begin{pmatrix}25&7\\7&25\end{pmatrix}\)，所以 \(v_1=(1,1)^T/\sqrt2\)、\(v_2=(1,-1)^T/\sqrt2\)，对应特征值 32 和 18。由 \(u_i=Av_i/\sigma_i\) 得 \(u_1=(1,0)^T\)、\(u_2=(0,-1)^T\)。因此</p>
<div class="equation">\[\begin{aligned}
U&=\begin{pmatrix}1&0\\0&-1\end{pmatrix},\quad
\Sigma=\begin{pmatrix}4\sqrt2&0\\0&3\sqrt2\end{pmatrix},\\
V&=\frac1{\sqrt2}\begin{pmatrix}1&1\\1&-1\end{pmatrix},\quad A=U\Sigma V^T.
\end{aligned}\]</div>
<p>直接检查第一列关系 \(Av_1=4\sqrt2\,u_1\)，第二列关系 \(Av_2=3\sqrt2\,u_2\)。仅从 \(AA^T\) 找特征向量时，\(u_2\) 的正负号无法确定；必须用 \(Av_2=\sigma_2u_2\) 与右奇异向量配对。这也是讲义特别提醒的符号问题。</p>

<h2 id="geometry">三、圆为什么变成椭圆？</h2>
<p>单位圆上的输入 \(x\) 先由 \(V^T\) 旋转或反射，仍在单位圆上；再由 \(\Sigma\) 沿两个新坐标轴分别拉伸 \(4\sqrt2\) 和 \(3\sqrt2\)，变成椭圆；最后 \(U\) 改变椭圆朝向。故奇异值是映射后椭圆的半轴长度，\(v_i\) 是输入中的主方向，\(u_i\) 是输出中的主方向。若某个奇异值为零，该方向被压扁，图形降维。</p>
<figure class="note-figure"><svg viewBox="0 0 560 245" role="img" aria-labelledby="svd-title svd-desc"><title id="svd-title">SVD 把圆映成椭圆</title><desc id="svd-desc">左边单位圆经输入坐标变化，再按两个奇异值缩放为右边椭圆；两个方向上的缩放不同。</desc><circle cx="115" cy="120" r="64" fill="none" stroke="#557f79" stroke-width="2"/><path d="M32 120H198M115 37V203" stroke="#8e9992"/><path d="M220 120H300" stroke="#bb6a4b" stroke-width="2"/><path d="M290 113L300 120L290 127" fill="none" stroke="#bb6a4b" stroke-width="2"/><ellipse cx="415" cy="120" rx="112" ry="72" fill="none" stroke="#557f79" stroke-width="2"/><path d="M295 120H535M415 35V205" stroke="#8e9992"/><text x="84" y="226">输入单位圆</text><text x="380" y="226">输出椭圆</text><text x="334" y="107">σ₁</text><text x="422" y="58">σ₂</text></svg><figcaption>示意两条主轴的不同伸缩；图上长度未按本例 \(4\sqrt2:3\sqrt2\) 精确比例绘制。</figcaption></figure>
<p>还可写成 \(A=\sum_{i=1}^r\sigma_i u_iv_i^T\)。每一项都是秩一映射：先从输入取 \(v_i\) 方向的坐标，再沿 \(u_i\) 输出。这个式子是下一讲理解低秩近似与压缩的基础。</p>
<h2 id="check">理解检查</h2>
<ol><li>习题 29.1 的 Fibonacci 矩阵对称却有一个负特征值。为什么它的奇异值要取特征值的<strong>绝对值</strong>？</li><li>对 \(A=\begin{pmatrix}4&3\\8&6\end{pmatrix}\)，写成一个列向量乘一个行向量，求唯一非零奇异值 \(5\sqrt5\)。</li><li>为什么 \(A^TA\) 的零特征向量正是 \(N(A)\) 中的向量？</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit III, Session 5 <em>Singular Value Decomposition</em> 整理。明确左右奇异向量所属空间与符号配对。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
