---
title: 第二单元 8：特征值与特征向量｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-2-session-08/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit2">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-07/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-09/">下一讲 →</a></p>
<div class="note-lead"><p>行列式告诉我们一个方阵会不会压扁整个空间；特征向量则寻找它作用时<strong>方向不变</strong>的特殊输入。找到这些方向后，后几讲的反复乘矩阵与微分方程都会变成按方向分别计算。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-2-session-8-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-2-session-8-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-2-session-8-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="definition">一、什么输入会保留自己的方向？</h2>
<p>给定 \(n\times n\) 方阵 \(A\)，若有<strong>非零</strong>向量 \(v\) 和数 \(\lambda\) 满足 \(Av=\lambda v\)，就称 \(v\) 是对应 \(\lambda\) 的特征向量，\(\lambda\) 是特征值。排除零向量很重要：\(A0=\lambda0\) 对任何 \(\lambda\) 都成立，零向量无法挑出矩阵的特殊方向。若 \(\lambda<0\)，输出还会反向；\(\lambda=0\) 则把这个方向压到零。</p>
<p>例如对平面上某条过原点直线的正交投影 \(P\)，直线上的非零向量满足 \(Pv=v\)，特征值为 1；垂直直线的非零向量满足 \(Pv=0\)，特征值为 0。第一个方向被保留，第二个方向被抹去。于是<strong>零是 \(A\) 的特征值，当且仅当 \(N(A)\ne\{0\}\)，也当且仅当 \(A\) 不可逆。</strong></p>
<figure class="note-figure"><svg viewBox="0 0 560 300" role="img" aria-labelledby="eig-title eig-desc"><title id="eig-title">两个保持方向的特征向量</title><desc id="eig-desc">两条特征方向都经过原点。向量 v1 经过作用后仍沿右上但变长；v2 仍沿右下但变长较少。</desc><defs><marker id="u2-eig-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#557f79"/></marker></defs><path d="M60 150H500M280 25V275" stroke="#8e9992" stroke-width="1.5"/><path d="M155 275L405 25M155 25L405 275" stroke="#a2aaa3" stroke-width="1.5" stroke-dasharray="5 5"/><path d="M280 150L325 105M280 150L405 25" stroke="#557f79" stroke-width="2.5" marker-end="url(#u2-eig-arrow)"/><path d="M280 150L325 195M280 150L370 240" stroke="#bb6a4b" stroke-width="2.5" marker-end="url(#u2-eig-arrow)"/><text x="333" y="110">v₁</text><text x="410" y="34">Av₁</text><text x="333" y="192">v₂</text><text x="381" y="248">Av₂</text><text x="264" y="170">O</text></svg><figcaption>示意 \(A=\begin{pmatrix}3&1\\1&3\end{pmatrix}\) 的两个特征方向：\(v_1=(1,1)^T\) 放大 4 倍，\(v_2=(1,-1)^T\) 放大 2 倍。箭头长度未按 4:2 精确比例绘制，只表示同向放大。</figcaption></figure>

<h2 id="characteristic">二、为什么先解一个行列式方程？</h2>
<p>\(Av=\lambda v\) 等价于 \((A-\lambda I)v=0\)。我们要的是非零解，因此 \(A-\lambda I\) 必须奇异；上几讲的判据立即给出</p>
<div class="equation">\[\det(A-\lambda I)=0.\]</div>
<p>这叫<strong>特征方程</strong>。先求使行列式为零的 \(\lambda\)，再对每个值解 \((A-\lambda I)v=0\)，得到对应的<strong>特征空间</strong> \(N(A-\lambda I)\)；其中除零向量外才是特征向量。两步缺一不可：只找到特征值，还不知道具体不变方向。</p>
<p>例如 \(A=\begin{pmatrix}3&1\\1&3\end{pmatrix}\)。由 \((3-\lambda)^2-1=0\) 得 \(\lambda=4,2\)。当 \(\lambda=4\)，两条方程都化为 \(-v_1+v_2=0\)，可取 \(v=(1,1)^T\)；当 \(\lambda=2\)，方程化为 \(v_1+v_2=0\)，可取 \(v=(1,-1)^T\)。直接验算 \(A(1,1)^T=4(1,1)^T\)、\(A(1,-1)^T=2(1,-1)^T\)。两个方向正交，与 \(A=A^T\) 相符；对称矩阵的完整正交特征基将在后面证明。</p>

<h2 id="trace-det">三、迹、行列式和特征值的关系有多强？</h2>
<p>对两阶矩阵 \(A=\begin{pmatrix}a&b\\c&d\end{pmatrix}\)，展开得到</p>
<div class="equation">\[\det(\lambda I-A)=\lambda^2-(a+d)\lambda+(ad-bc).\]</div>
<p>所以两根之和是 \(\operatorname{tr}A=a+d\)（迹，即对角线之和），乘积是 \(\det A\)。一般 \(n\) 阶方阵也如此：在复数范围内按<strong>代数重数</strong>计入全部 \(n\) 个根，特征值之和等于迹，乘积等于行列式。这些关系给校验方法，却不能单靠迹与行列式确定三阶及更高阶矩阵的所有特征值。</p>
<p>三角矩阵的特征值就是对角线上的数，因为 \(\det(A-\lambda I)\) 仍是三角行列式。重根并不保证有足够多的特征向量：\(\begin{pmatrix}3&1\\0&3\end{pmatrix}\) 只有特征值 3，却只有 \((1,0)^T\) 这一个独立特征方向；单位矩阵虽然也全是重根，却有任意一组基作为特征向量。是否能对角化，要看<strong>独立特征向量的数目</strong>。</p>

<h2 id="complex">四、实矩阵未必有实特征方向</h2>
<p>平面旋转 \(90^\circ\) 的矩阵 \(R=\begin{pmatrix}0&-1\\1&0\end{pmatrix}\) 把每个非零实向量转到垂直方向，所以没有非零实特征向量。但 \(\det(R-\lambda I)=\lambda^2+1\)，在复数范围的根是 \(i,-i\)。这不矛盾：把工作空间扩展到 \(\mathbb C^2\)，才会出现相应的复特征向量。实矩阵的非实特征值成共轭对出现，因为特征多项式的系数是实数。</p>
<p>习题 21.1 的一个好处是暴露“哪些量不能仅靠特征值决定”：若 \(B\) 的特征值为 \(0,1,2\)，则可知秩为 2、\(\det(B^TB)=0\)，以及 \((B^2+I)^{-1}\) 的特征值；却不能仅凭这三个数求出 \(B^TB\) 的全部特征值，因为它还依赖特征向量之间的几何关系。</p>
<h2 id="check">理解检查</h2>
<ol><li>对上面的 \(A\)，把它换成 \(A+5I\)：特征向量变不变？两个特征值怎样变？用等式直接证明。</li><li>为什么 \(0\) 必须排除在“特征向量”的定义之外，却允许作为特征值？</li><li>求三角矩阵 \(\begin{pmatrix}3&1\\0&3\end{pmatrix}\) 的特征空间，说明它为什么无法给出一组二维基。</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit II, Session 8 <em>Eigenvalues and Eigenvectors</em> 整理。对根的数量、重数及 \(B^TB\) 可确定信息作了必要限定。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
