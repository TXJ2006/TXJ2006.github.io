---
title: 第二单元 2：正交投影｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-2-session-02/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit2">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-01/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-03/">下一讲 →</a></p>
<div class="note-lead"><p>上一讲说明 \(C(A)\) 与 \(N(A^T)\) 互为正交补。当 \(Ax=b\) 无解时，\(b\) 不在列空间里；正交投影先找列空间中距离 \(b\) 最近的 \(p\)，让 \(Ax=p\) 变成可解。这一讲先从投到一条直线算起，再推广到平面和一般列空间。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-2-session-2-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-2-session-2-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-2-session-2-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="line">一、投到直线：先找最近的点</h2>
<p>取非零向量 \(a\in\mathbb R^m\)，它张成过原点的直线 \(L=\{ta:t\in\mathbb R\}\)。给定 \(b\)，在这条直线上找离 \(b\) 最近的点 \(p\)。因为 \(p\in L\)，可写成 \(p=\widehat x a\)，其中 \(\widehat x\) 是待求的一个数。把误差向量记为 \(e=b-p\)。最近点的关键条件是 \(e\perp a\)：若还沿着直线有误差分量，就可以继续移动 \(p\) 使距离更短。</p>
<figure class="note-figure"><svg viewBox="0 0 560 315" role="img" aria-labelledby="projection-title projection-desc"><title id="projection-title">向量投到一条直线</title><desc id="projection-desc">从原点 O 沿向量 a 到投影 p，向量 b 的终点在 p 的左上方，误差 e 从 p 指向 b，并与过原点的直线垂直。</desc><defs><marker id="u2-proj-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto"><path d="M0 0L10 5L0 10Z" fill="#557f79"/></marker></defs><path d="M75 265L468 68" stroke="#8d9992" stroke-width="2"/><path d="M75 265L315 145" stroke="#557f79" stroke-width="3" marker-end="url(#u2-proj-arrow)"/><path d="M75 265L255 25" stroke="#557f79" stroke-width="2" stroke-dasharray="5 5" marker-end="url(#u2-proj-arrow)"/><path d="M315 145L255 25" stroke="#bb6a4b" stroke-width="3" marker-end="url(#u2-proj-arrow)"/><path d="M308 132L295 139L302 151" fill="none" stroke="#77847d" stroke-width="1.6"/><circle cx="75" cy="265" r="4" fill="#557f79"/><circle cx="315" cy="145" r="5" fill="#557f79"/><circle cx="255" cy="25" r="5" fill="#bb6a4b"/><text x="54" y="285">O</text><text x="326" y="156">p</text><text x="263" y="22">b</text><text x="174" y="192">a 的方向</text><text x="271" y="100">e=b-p</text></svg><figcaption>投影点在直线上，误差垂直于直线；图中的位置只表示关系，不对应下面例子的具体坐标。</figcaption></figure>
<p>将 \(a^Te=0\) 与 \(e=b-\widehat x a\) 合起来，便有 \(a^Tb-\widehat x a^Ta=0\)。因为 \(a\ne0\)，分母 \(a^Ta=\|a\|^2\gt0\)，所以</p>
<div class="equation">\[\widehat x=\frac{a^Tb}{a^Ta},\qquad p=a\frac{a^Tb}{a^Ta},\qquad e=b-p.\]</div>
<p>例如 \(a=(2,1)^T,\ b=(1,3)^T\)。这里 \(a^Tb=5,\ a^Ta=5\)，于是 \(\widehat x=1,\ p=(2,1)^T,\ e=(-1,2)^T\)；验算 \(a^Te=-2+2=0\)。</p>
<p>垂直为什么保证<strong>全局最近</strong>？对直线上任意另一点 \(q=ta\)，有 \(b-q=e+(p-q)\)，而 \(e\perp(p-q)\)。由勾股定理，\(\|b-q\|^2=\|e\|^2+\|p-q\|^2\ge\|e\|^2\)；等号只在 \(q=p\) 时成立。</p>

<h2 id="projection-matrix">二、把“投影”做成一个矩阵</h2>
<p>固定直线方向 \(a\)，让目标 \(b\) 改变。上面的式子对每个 \(b\) 都适用，而且对 \(b\) 是线性的，因此可写 \(p=Pb\)，其中</p>
<div class="equation">\[P=\frac{aa^T}{a^Ta}.\]</div>
<p>\(a^Ta\) 是正实数；\(aa^T\) 却是 \(m\times m\) 矩阵。顺序不能交换：\(a^Ta\) 算长度平方，\(aa^T\) 把一个输入送到 \(a\) 的方向。对本讲的 \(a=(2,1)^T\)，\(P=\frac15\begin{pmatrix}4&2\\2&1\end{pmatrix}\)，乘 \(b=(1,3)^T\) 确实得到 \((2,1)^T\)。</p>
<p>投影矩阵有两个特征：\(P^T=P\)（对称）和 \(P^2=P\)（再投一次不变）。其像空间是 \(L\)，零空间是 \(L^\perp\)：已经在直线上的向量保持原样，垂直直线的向量被送到零。注意只有 \(P^2=P\) 还不足以保证“垂直投影”；另需 \(P^T=P\)。</p>

<h2 id="subspace">三、从直线推广到列空间</h2>
<p>现在让 \(A=[a_1\ \cdots\ a_n]\) 是 \(m\times n\) 矩阵，列线性无关。目标 \(b\in\mathbb R^m\)，要找 \(C(A)\) 中的最近点 \(p=A\widehat x\)，其中 \(\widehat x\in\mathbb R^n\) 是这点按列向量写成的系数。误差 \(e=b-A\widehat x\) 必须与<strong>每一列</strong>正交，即 \(a_j^Te=0\) 对每个 \(j\) 成立。把这 \(n\) 条等式合并，才得到</p>
<div class="equation">\[A^T(b-A\widehat x)=0\qquad\Longleftrightarrow\qquad A^TA\widehat x=A^Tb.\]</div>
<p>这叫<strong>正规方程</strong>。上一讲证明了列无关时 \(A^TA\) 可逆，于是</p>
<div class="equation">\[\widehat x=(A^TA)^{-1}A^Tb,\qquad p=Pb,\qquad P=A(A^TA)^{-1}A^T.\]</div>
<p>\(A^T\) 把 \(m\) 个目标分量变成 \(n\) 个方程右端；\((A^TA)^{-1}\) 求出 \(n\) 个系数；左乘 \(A\) 再把它们拼成 \(m\) 维的投影点。这解释了乘法顺序。若列相关，\(A^TA\) 不可逆，上式不能直接用；但最近点 \(p\) 仍唯一，可先选列空间的一组基来计算。</p>
<p>误差 \(e=b-p\in N(A^T)\)，而 \(p\in C(A)\)，所以 \(b=p+e\) 是前一讲的正交补在计算中的具体样子。任意 \(q\in C(A)\) 都满足 \(\|b-q\|^2=\|e\|^2+\|p-q\|^2\)，因此 \(p\) 的确最近。</p>
<h2 id="check">理解检查</h2>
<ol><li>习题 15.1 的 \(A\) 是四阶单位矩阵去掉最后一列。它的列空间是什么？对 \(b=(1,2,3,4)^T\)，投影与误差分别是什么？</li><li>写出对应的 \(P\)，核对 \(P^T=P,\ P^2=P\)；再说明 \(I-P\) 把向量投到哪个空间。</li><li>把 \(a\) 换成 \(2a\) 后，为什么直线不变、投影点也不变？从公式逐步验证。</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit II, Session 2 <em>Projections onto Subspaces</em> 整理。图和数值例子用来解释最近点、误差与投影矩阵；下一讲把这一过程用于实际数据。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
