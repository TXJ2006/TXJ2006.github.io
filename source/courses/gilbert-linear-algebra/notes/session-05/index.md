---
title: 第五节：转置、置换与向量空间｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/session-05/
mathjax: true
toc_number: false
comments: false
---

<style>
.post-block:has(.la-notes--session05) .post-header { margin-bottom: 24px; }
.post-block:has(.la-notes--session05) .post-title { font-size: 26px; line-height: 1.45; color: #253c3c; }
.post-block:has(.la-notes--session05) .breadcrumb { display: none; }
.post-body .la-notes--session05 { max-width: 660px; margin: 0 auto; font-size: 17px; line-height: 1.85; color: #303737; }
.post-body .la-notes--session05 p { margin: 0 0 .9em; text-indent: 0; }
.post-body .la-notes--session05 h2 { margin: 2.4em 0 1em; padding: 0 0 .45em; border-bottom: 1px solid #dce5e3; color: #285d5d; font-size: 22px; line-height: 1.5; }
.post-body .la-notes--session05 h3 { margin: 1.7em 0 .75em; color: #303737; font-size: 18px; line-height: 1.6; }
.post-body .la-notes--session05 [id] { scroll-margin-top: 24px; }
.post-body .la-notes--session05 .note-nav { font-size: 14px; color: #687474; margin-bottom: 18px; }
.post-body .la-notes--session05 .note-nav a { border-color: #ccd8d5; }
.post-body .la-notes--session05 .note-lead { margin: 0 0 1.4em; padding: .1em 0 .1em 14px; border-left: 3px solid #729692; }
.post-body .la-notes--session05 .note-lead p { margin: 0; }
.post-body .la-notes--session05 .reading-links { padding: 14px 0; margin-bottom: 1.6em; border-top: 1px solid #e2e8e6; border-bottom: 1px solid #e2e8e6; font-size: 14px; line-height: 1.8; }
.post-body .la-notes--session05 .reading-links p { margin: 0; }
.post-body .la-notes--session05 .reading-links p + p { margin-top: 8px; }
.post-body .la-notes--session05 .link-row { display: flex; flex-wrap: wrap; column-gap: 18px; row-gap: 6px; }
.post-body .la-notes--session05 .equation { margin: 1em 0 1.15em; padding: 8px 2px; overflow-x: auto; overflow-y: hidden; }
.post-body .la-notes--session05 .MathJax_Display { font-size: 100%; margin: .25em 0 !important; padding: 3px 0; }
.post-body .la-notes--session05 ul, .post-body .la-notes--session05 ol { margin: .65em 0 1.2em; padding-left: 1.35em; }
.post-body .la-notes--session05 li { margin-bottom: .55em; }
.post-body .la-notes--session05 .note-extension { margin: 1.2em 0; padding: 10px 14px; border-left: 2px solid #ccd8d5; font-size: 16px; }
.post-body .la-notes--session05 .note-extension summary { cursor: pointer; color: #285d5d; }
.post-body .la-notes--session05 .note-extension[open] summary { margin-bottom: 1em; }
.post-body .la-notes--session05 .note-footer { margin-top: 1.8em; padding-top: 1em; border-top: 1px solid #e2e8e6; font-size: 14px; }
@media (max-width: 600px) {
  .post-block:has(.la-notes--session05) .post-title { font-size: 23px; }
  .post-body .la-notes--session05 { font-size: 16px; line-height: 1.85; }
  .post-body .la-notes--session05 h2 { font-size: 20px; margin-top: 2em; }
  .post-body .la-notes--session05 h3 { font-size: 17px; }
}
</style>

<section class="la-notes la-notes--session05">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-04/">第四节</a></p>
<div class="note-lead"><p><strong>定义向量空间与子空间，证明 span 的最小性，再将列空间识别为线性映射的像。</strong>转置与置换的作用见后文。</p></div>
<div class="reading-links">
<p class="link-row"><a href="/files/mit-18-06sc/unit-1-session-5-summary.pdf">讲义摘要</a> <a href="/files/mit-18-06sc/unit-1-session-5-problems.pdf">习题</a> <a href="/files/mit-18-06sc/unit-1-session-5-solutions.pdf">参考解答</a></p>
<p class="link-row"><a href="#outputs">向量空间</a> <a href="#spaces">子空间判据</a> <a href="#span">Span 与列空间</a> <a href="#transpose">转置的作用</a> <a href="#permutation">置换的作用</a> <a href="#matrix-spaces">用习题检验理解</a></p>
</div>

<h2 id="outputs">一、向量空间的定义</h2>
<p><strong>定义 1（向量空间）。</strong>固定一个域 \(\mathbb F\)。设 \(V\) 是非空集合，配有两个运算：</p>
<div class="equation">\[
\begin{aligned}
+\colon V\times V&amp;\longrightarrow V,
&amp;(\mathbf u,\mathbf v)&amp;\longmapsto\mathbf u+\mathbf v,\\
\cdot\colon\mathbb F\times V&amp;\longrightarrow V,
&amp;(a,\mathbf v)&amp;\longmapsto a\mathbf v.
\end{aligned}
\]</div>
<p>若对任意 \(\mathbf u,\mathbf v,\mathbf w\in V\) 及 \(a,b\in\mathbb F\)，满足以下公理，则称 \(V\) 为<strong>域 \(\mathbb F\) 上的向量空间</strong>。\(V\) 的元素称为向量，\(\mathbb F\) 的元素称为标量。</p>
<ol>
<li><strong>加法交换律：</strong>\(\mathbf u+\mathbf v=\mathbf v+\mathbf u\)。</li>
<li><strong>加法结合律：</strong>\((\mathbf u+\mathbf v)+\mathbf w=\mathbf u+(\mathbf v+\mathbf w)\)。</li>
<li><strong>零向量：</strong>存在 \(\mathbf0\in V\)，使每个 \(\mathbf v\in V\) 满足 \(\mathbf v+\mathbf0=\mathbf v\)。</li>
<li><strong>加法逆元：</strong>每个 \(\mathbf v\in V\) 都存在 \(-\mathbf v\in V\)，使 \(\mathbf v+(-\mathbf v)=\mathbf0\)。</li>
<li><strong>数乘结合律：</strong>\(a(b\mathbf v)=(ab)\mathbf v\)。</li>
<li><strong>数乘单位元：</strong>\(1\mathbf v=\mathbf v\)，其中 \(1\) 是域中的乘法单位元。</li>
<li><strong>对向量加法的分配律：</strong>\(a(\mathbf u+\mathbf v)=a\mathbf u+a\mathbf v\)。</li>
<li><strong>对标量加法的分配律：</strong>\((a+b)\mathbf v=a\mathbf v+b\mathbf v\)。</li>
</ol>
<p>“域”是允许加、减、乘及除以非零元素的标量体系，例如 \(\mathbb R\) 与 \(\mathbb C\)。<strong>本节以下均取 \(\mathbb F=\mathbb R\)</strong>。上述运算的目标集合都是 \(V\)，已经包含了加法、数乘的封闭性。向量空间的结构由集合、标量域及这两种运算共同确定。</p>
<p><strong>例 1。</strong>\(\mathbb R^n\) 在逐分量加法与数乘下是实向量空间。这里 \(\mathbb R^n\) 表示所有 \(n\) 分量实列向量的集合。向量也可以是矩阵、函数等对象，只要指定的运算满足上述公理。</p>

<h2 id="spaces">二、子空间的定义与判据</h2>
<p id="subspace"><strong>定义 2（子空间）。</strong>设 \(V\) 是实向量空间，\(W\subseteq V\)。若 \(W\) 在 \(V\) 的加法、数乘的限制下也是实向量空间，则称 \(W\) 为 \(V\) 的<strong>线性子空间</strong>。这里“限制”指使用原来的运算，只让输入取自 \(W\)。</p>
<p id="closure"><strong>命题 1（子空间判据）。</strong>\(W\subseteq V\) 是子空间，当且仅当 \(W\ne\varnothing\)，且</p>
<div class="equation">\[
\forall\,\mathbf u,\mathbf v\in W,\quad
\forall\,a,b\in\mathbb R,\qquad
 a\mathbf u+b\mathbf v\in W.
\]</div>
<p><strong>证明。</strong>必要性来自向量空间的加法、数乘封闭性。反过来，设上述条件成立。任取 \(\mathbf w\in W\)，令系数均为零，得 \(\mathbf0\in W\)；令 \(a=-1,b=0\)，得每个向量的相反向量都在 \(W\) 中。取 \(a=b=1\) 得加法封闭，取 \(b=0\) 得数乘封闭。其余公理是 \(V\) 中已成立的等式，在 \(W\) 上仍成立。因此 \(W\) 是向量空间。\(\square\)</p>
<p><strong>例 2。</strong>设 \(\mathbf v=(1,2)^T\)，则 \(W=\{t\mathbf v:t\in\mathbb R\}\) 是 \(\mathbb R^2\) 的子空间：它含零向量，且</p>
<div class="equation">\[
a(s\mathbf v)+b(t\mathbf v)=(as+bt)\mathbf v\in W.
\]</div>
<p>几何上 \(W\) 是直线 \(y=2x\)。\(\mathbf v\) 是其中一个向量，\(W\) 是整个集合；\(W\) 既是实向量空间，也是 \(\mathbb R^2\) 的子空间。</p>
<p id="counterexamples"><strong>反例。</strong>直线 \(y=2x+1\) 不含零向量，故不是子空间。两条坐标轴的并集虽然含零向量，却因 \((1,0)^T+(0,1)^T=(1,1)^T\) 不在其中而不封闭，也不是子空间。\(\{\mathbf0\}\) 则是子空间。</p>

<h2 id="span">三、张成空间与列空间</h2>
<h3 id="column-space">张成空间</h3>
<p><strong>定义 3（Span）。</strong>设 \(\mathbf a_1,\ldots,\mathbf a_n\in V\)，其中 \(n\ge1\)。它们的<strong>张成空间</strong>定义为</p>
<div class="equation">\[
\operatorname{span}_{\mathbb R}\{\mathbf a_1,\ldots,\mathbf a_n\}
=\left\{\sum_{j=1}^n x_j\mathbf a_j:
(x_1,\ldots,x_n)\in\mathbb R^n\right\}.
\]</div>
<p>其中每个和式称为这些向量的<strong>线性组合</strong>。以下省略下标 \(\mathbb R\)。</p>
<p><strong>命题 2。</strong>该 span 是包含 \(\mathbf a_1,\ldots,\mathbf a_n\) 的最小子空间。</p>
<p><strong>证明。</strong>记这个集合为 \(S\)。系数全取零，得 \(\mathbf0\in S\)。若 \(\mathbf u=\sum_j x_j\mathbf a_j\)、\(\mathbf v=\sum_j y_j\mathbf a_j\)，则</p>
<div class="equation">\[
a\mathbf u+b\mathbf v
=\sum_{j=1}^n(ax_j+by_j)\mathbf a_j\in S.
\]</div>
<p>由子空间判据，\(S\) 是子空间。取第 \(j\) 个系数为 1、其余为零，得 \(\mathbf a_j\in S\)；而任何包含这些向量的子空间，因对加法、数乘封闭，必包含它们的所有线性组合，即包含 \(S\)。这就是“最小”的含义。\(\square\)</p>
<p><strong>例 3。</strong>\(\operatorname{span}\{(1,2)^T,(2,4)^T\}=W\)，因为第二个向量是第一个的两倍。若换成 \((0,1)^T\)，则</p>
<div class="equation">\[
\begin{pmatrix}p\\q\end{pmatrix}
=p\begin{pmatrix}1\\2\end{pmatrix}
+(q-2p)\begin{pmatrix}0\\1\end{pmatrix}
\qquad(p,q\in\mathbb R),
\]</div>
<p>所以这两个向量张成整个 \(\mathbb R^2\)。</p>

<h3 id="reachable">列空间：矩阵各列的张成空间</h3>
<p><strong>定义 4（列空间）。</strong>设 \(A\in\mathbb R^{m\times n}\)，即 \(A\) 为 \(m\) 行 \(n\) 列的实矩阵，\(m,n\ge1\)。将其各列记为 \(\mathbf a_1,\ldots,\mathbf a_n\in\mathbb R^m\)。\(A\) 的<strong>列空间</strong>定义为</p>
<div class="equation">\[
\boxed{C(A)=\operatorname{span}\{\mathbf a_1,\ldots,\mathbf a_n\}
\subseteq\mathbb R^m.}
\]</div>
<p>由命题 2，\(C(A)\) 是 \(\mathbb R^m\) 的子空间，所以它自身也是实向量空间。<strong>向量空间是一般的代数结构；列空间是由给定矩阵确定的一个具体向量空间。</strong></p>
<p>也可以用映射表述。令 \(T_A\colon\mathbb R^n\to\mathbb R^m\)，\(T_A(\mathbf x)=A\mathbf x\)。对任意 \(\mathbf x,\mathbf y\in\mathbb R^n\) 与 \(a,b\in\mathbb R\)，有</p>
<div class="equation">\[
T_A(a\mathbf x+b\mathbf y)=aT_A(\mathbf x)+bT_A(\mathbf y),
\]</div>
<p>\(T_A\) 是<strong>线性映射</strong>，即保持线性组合的映射。由矩阵乘法 \(A\mathbf x=\sum_j x_j\mathbf a_j\)，得</p>
<div class="equation">\[
C(A)=\operatorname{im}T_A
=\{A\mathbf x:\mathbf x\in\mathbb R^n\}.
\]</div>
<p>\(\operatorname{im}T_A\) 称为映射的<strong>像</strong>，就是全部输出组成的集合。\(\mathbb R^m\) 是映射的目标空间，\(C(A)\) 是实际像，两者未必相等。</p>
<p><strong>命题 3（可解性）。</strong>对任意 \(\mathbf b\in\mathbb R^m\)，</p>
<div class="equation">\[
A\mathbf x=\mathbf b\text{ 有解}
\quad\Longleftrightarrow\quad
\mathbf b\in C(A).
\]</div>
<p><strong>证明。</strong>有解等价于存在 \(\mathbf x\in\mathbb R^n\) 使 \(\mathbf b=A\mathbf x\)；按像的定义，这恰好等价于 \(\mathbf b\in C(A)\)。\(\square\)</p>
<p><strong>例 4。</strong>取</p>
<div class="equation">\[
A=\begin{pmatrix}1&amp;0&amp;1\\0&amp;1&amp;1\\1&amp;1&amp;2\end{pmatrix},\qquad
A\begin{pmatrix}s\\t\\r\end{pmatrix}
=\begin{pmatrix}s+r\\t+r\\s+t+2r\end{pmatrix}.
\]</div>
<p>每个输出的第三分量等于前两分量之和，故</p>
<div class="equation">\[
C(A)\subseteq
H:=\left\{\begin{pmatrix}p\\q\\p+q\end{pmatrix}:p,q\in\mathbb R\right\}.
\]</div>
<p>反过来，对任意 \((p,q,p+q)^T\in H\)，取输入 \((p,q,0)^T\)，即可得到它，故 \(H\subseteq C(A)\)。因此 \(C(A)=H\)，是平面 \(z=x+y\)，并非整个目标空间 \(\mathbb R^3\)。由命题 3，\((2,3,5)^T\) 对应的方程有解，\((2,3,6)^T\) 对应的方程无解。</p>
<details class="note-extension">
<summary>延伸：零空间怎样解释多解？</summary>
<h3 id="nullspace">零空间：哪些输入被送到零？</h3>
<p>知道哪些 \(\mathbf b\) 能解，还没有回答解是否唯一。为此看<strong>零空间</strong>：</p>
<div class="equation">\[
N(A)=\{\mathbf x\in\mathbb R^n:A\mathbf x=\mathbf0\}.
\]</div>
<p>它收集的是<strong>全部被矩阵送到零的输入</strong>，不是一个特殊输入，也不是所有可能的输出。它在<strong>输入空间</strong>中，列空间在输出空间中。它也是子空间，因为 \(A\mathbf u=A\mathbf v=\mathbf0\) 时，\(A(\alpha\mathbf u+\beta\mathbf v)=\mathbf0\)。</p>
<p>在本节例子里，\(A\mathbf x=\mathbf0\) 要求 \(s+r=0,t+r=0\)，第三个方程随之成立。所以</p>
<div class="equation">\[
N(A)=\left\{r\begin{pmatrix}-1\\-1\\1\end{pmatrix}:r\in\mathbb R\right\}.
\]</div>
<p>这个方向表示“前两列各减一份、第三列加一份”，输出正好抵消。因而输入沿这个方向改变，输出不变。例如：</p>
<div class="equation">\[
A\mathbf x=\begin{pmatrix}2\\3\\5\end{pmatrix}
\quad\Longleftrightarrow\quad
\mathbf x=\begin{pmatrix}2\\3\\0\end{pmatrix}
+r\begin{pmatrix}-1\\-1\\1\end{pmatrix},\quad r\in\mathbb R.
\]</div>
<p>这不只是凑出一批解。若 \(\mathbf x_p\) 是一个解，则另一个解 \(\mathbf x\) 必须满足 \(A(\mathbf x-\mathbf x_p)=\mathbf0\)；反过来，给 \(\mathbf x_p\) 加上任意零空间向量，仍是解。因此，一般地：</p>
<div class="equation">\[
\boxed{\text{全部解}=\mathbf x_p+N(A)}.
\]</div>
<p><strong>列空间决定是否有解；零空间决定已有的解能否变化。</strong>若 \(N(A)=\{\mathbf0\}\)，有解时就是唯一解；若零空间有非零向量，有解时就有无穷多个解。</p>
<p>特别注意：\(A\mathbf x=\mathbf0\) 的解集是子空间；\(A\mathbf x=\mathbf b\) 在 \(\mathbf b\ne\mathbf0\) 时，即使有解也不含零向量，因此不是子空间，而是零空间的平移，叫作<strong>仿射子空间</strong>。本例的全部解是一条不过原点的直线。</p>

</details>

<h2 id="transpose">四、转置：怎样连接行、列和内积？</h2>
<h3 id="inner-product">从“按行检查”到“按列组合”</h3>
<p>若 \(A\) 为 \(m\times n\)，转置 \(A^T\) 为 \(n\times m\)，定义为 \((A^T)_{ij}=A_{ji}\)。它把行与列互换。例如：</p>
<div class="equation">\[
B=\begin{pmatrix}1&amp;2\\3&amp;4\\0&amp;1\end{pmatrix},\qquad
B^T=\begin{pmatrix}1&amp;3&amp;0\\2&amp;4&amp;1\end{pmatrix}.
\]</div>
<p>前几节把 \(B\mathbf x\) 看作列的组合；按行看，每个输出分量又是某一行与 \(\mathbf x\) 的内积。对实列向量，内积就是 \(\mathbf u^T\mathbf v=\sum_i u_i v_i\)。转置使这两种读法能写在同一套乘法中。</p>
<p>它还有一个更有用的联系：若 \(\mathbf x\in\mathbb R^n,\mathbf y\in\mathbb R^m\)，则</p>
<div class="equation">\[
\boxed{(A\mathbf x)^T\mathbf y=\mathbf x^T(A^T\mathbf y)}.
\]</div>
<p>左边在输出空间算内积，右边在输入空间算内积；两者给出同一个数。这说明转置能把“输出与 \(\mathbf y\) 的比较”转回输入一侧。<strong>它通常不撤销 \(A\) 的作用，因而一般不是逆矩阵。</strong></p>
<p>本节三阶例子的输出都满足 \(z=x+y\)，所以与 \(\mathbf y=(-1,-1,1)^T\) 的内积总为零；这正是 \(A^T\mathbf y=\mathbf0\)。一般地，\(A^T\mathbf y=\mathbf0\) 当且仅当 \(\mathbf y\) 与 \(A\) 的每一列、从而与整个列空间都垂直。转置由此连接了<strong>可到达的输出与输出必须满足的线性约束</strong>。</p>
<h3 id="gram">乘积为什么反序？为什么会出现对称矩阵？</h3>
<p>设 \(A\) 为 \(m\times n\)、\(B\) 为 \(n\times p\)。逐项比较可得：</p>
<div class="equation">\[
\bigl((AB)^T\bigr)_{ij}
=(AB)_{ji}=\sum_k A_{jk}B_{ki}
=\sum_k(B^T)_{ik}(A^T)_{kj}.
\]</div>
<p>所以 \((AB)^T=B^TA^T\)。转置后中间指标的连接顺序反过来，矩阵尺寸也要求 \(p\times n\) 乘 \(n\times m\)。</p>
<p>若 \(S^T=S\)，称 \(S\) 为对称矩阵。任意实矩阵 \(B\) 都能产生对称矩阵 \(B^TB\)：它的第 \((i,j)\) 项是第 \(i\) 列与第 \(j\) 列的内积，交换两列的角色，数值不变。上面的长方形矩阵给出：</p>
<div class="equation">\[
B^TB=\begin{pmatrix}10&amp;14\\14&amp;21\end{pmatrix}.
\]</div>
<p>\(10=1^2+3^2+0^2\)，\(21=2^2+4^2+1^2\) 是两列的长度平方；\(14=1\cdot2+3\cdot4+0\cdot1\) 是两列的内积。于是 \(B^TB\) 把列之间的几何关系集中到一个矩阵里。</p>
<details class="note-extension">
<summary>后续联系：转置为什么出现在最小二乘中？</summary>
<p>这也解释它以后为什么出现在最小二乘中：当 \(A\mathbf x=\mathbf b\) 无解时，要使误差 \(\mathbf e=\mathbf b-A\mathbf x\) 的长度最小，最佳误差必须垂直于列空间，所以</p>
<div class="equation">\[
A^T(\mathbf b-A\mathbf x)=\mathbf0
\quad\Longrightarrow\quad
A^TA\mathbf x=A^T\mathbf b.
\]</div>
<p>若误差还沿某个列空间方向有分量，就可以沿那个方向调整输出来缩短它；若已垂直，任何非零的输出调整都会增加长度平方。这是后续投影与最小二乘的入口。\(A^TA\) 总是对称，但对称不保证可逆：因为 \(\mathbf x^TA^TA\mathbf x=(A\mathbf x)^T(A\mathbf x)\)，右边是输出的长度平方，所以 \(A^TA\) 与 \(A\) 有相同的零空间；恰在 \(N(A)=\{\mathbf0\}\) 时，方阵 \(A^TA\) 才可逆。本节三阶例子就不满足。</p>

</details>

<h2 id="permutation">五、置换：换行保留了什么？</h2>
<p>置换矩阵是把单位矩阵的行重新排列得到的矩阵，每行、每列恰有一个 1，其余为零。左乘时，每行从原矩阵中选取一行，因此 \(PA\) 重排 \(A\) 的行；右乘 \(AP\) 则重排列。两者改变的对象不同。</p>
<p>例如，主元位置为零时先换行：</p>
<div class="equation">\[
A=\begin{pmatrix}0&amp;2\\1&amp;3\end{pmatrix},\quad
\mathbf b=\begin{pmatrix}4\\7\end{pmatrix},\quad
P=\begin{pmatrix}0&amp;1\\1&amp;0\end{pmatrix},
\qquad
PA=\begin{pmatrix}1&amp;3\\0&amp;2\end{pmatrix},\quad
P\mathbf b=\begin{pmatrix}7\\4\end{pmatrix}.
\]</div>
<p>原方程是 \(2x_2=4,x_1+3x_2=7\)。换行只重排这两个条件，解仍为 \((1,2)^T\)。<strong>必须同时换右端：</strong>\(A\mathbf x=\mathbf b\iff PA\mathbf x=P\mathbf b\)。换列则是在重排变量对应的系数，也要相应重排未知量。</p>
<p>所以第四节的分解扩展为 \(PA=LU\)，求解时使用 \(L\mathbf y=P\mathbf b\)，再解 \(U\mathbf x=\mathbf y\)。上面例子换行后已经是上三角矩阵，恰有 \(L=I,U=PA\)；一般还要继续消元。实际数值计算也会主动选取绝对值较大的主元，减少除以过小数值带来的舍入误差放大。</p>
<h3 id="orthogonal-permutations">为什么逆矩阵等于转置？</h3>
<p>\(P\) 的各列是顺序不同的单位向量：每列与自身内积为 1，与别列内积为 0。因此</p>
<div class="equation">\[
P^TP=I,\qquad P^{-1}=P^T.
\]</div>
<p>这也意味着 \((P\mathbf u)^T(P\mathbf v)=\mathbf u^T\mathbf v\)：重排坐标保留内积、长度与夹角。置换矩阵因此是后面<strong>正交矩阵</strong>的一类具体例子；一般消元矩阵虽然可逆，却未必保留这些几何量。</p>
<p>但 \(P^{-1}\) 不一定等于 \(P\)。习题 5.1 的三循环就是反例：</p>
<div class="equation">\[
P=\begin{pmatrix}0&amp;0&amp;1\\1&amp;0&amp;0\\0&amp;1&amp;0\end{pmatrix},\qquad
P\begin{pmatrix}a\\b\\c\end{pmatrix}=\begin{pmatrix}c\\a\\b\end{pmatrix}.
\]</div>
<p>做三次回到原顺序，所以 \(P^3=I\)，撤销一次要再做两次，故 \(P^{-1}=P^2=P^T\ne P\)。四阶置换也不必做四次才还原：固定一个坐标，循环其余三个，就有 \(P'^3=I\)，因而 \(P'^4=P'\ne I\)。置换还原所需的次数取决于循环结构。</p>
<p>另一个必须分清的联系：可逆行操作保持方程的解，却不一定保持列空间原来的位置。若消元矩阵为 \(E\)，则 \(N(EA)=N(A)\)，而 \(C(EA)=E\,C(A)\)。前者因为 \(EA\mathbf x=\mathbf0\iff A\mathbf x=\mathbf0\)；这里 \(E\,C(A)\) 指把列空间中每个向量左乘 \(E\) 后得到的集合。解非齐次方程时，右端也必须一起改写。</p>

<h2 id="matrix-spaces">六、习题里的矩阵，为什么也能组成向量空间？</h2>
<p>“向量”不限定为一列数字。所有 \(n\times n\) 实矩阵组成的集合 \(M_n(\mathbb R)\)，用逐项加法和数乘，也满足向量空间的运算法则。这里考察的封闭性是<strong>加法和数乘</strong>，不要求矩阵乘法封闭。</p>
<p>习题 5.3 就是在这个空间里判断子空间。若 \(S^T=S,T^T=T\)，则</p>
<div class="equation">\[
(\alpha S+\beta T)^T=\alpha S^T+\beta T^T=\alpha S+\beta T.
\]</div>
<p>对称矩阵的线性组合仍对称，且包含零矩阵，因此它们形成子空间。同样，若 \(K^T=-K,H^T=-H\)，则 \((\alpha K+\beta H)^T=-(\alpha K+\beta H)\)，所以反对称矩阵也形成子空间。</p>
<p>“所有不对称矩阵”却不形成子空间：它不含零矩阵，而且一个不对称矩阵与它的相反数都在集合中，两者相加就离开集合。<strong>满足某种性质的对象组成集合，不保证这个集合能承受任意线性组合。</strong></p>
<p>习题 5.2 问四阶矩阵有多少项能自由选择。对称矩阵可自由选 4 个对角项与上三角的 6 项，下三角随之确定，共 10 项；反对称矩阵的对角项满足 \(a_{ii}=-a_{ii}\)，只能为零，其余只需选上三角 6 项，共 6 项。</p>
<p>这两类空间还有一个具体用途：任意实方阵都可以<strong>唯一</strong>拆成对称部分和反对称部分：</p>
<div class="equation">\[
S=\frac{A+A^T}{2},\qquad K=\frac{A-A^T}{2},\qquad A=S+K.
\]</div>
<p>两部分相加还原 \(A\)，而 \(S^T=S,K^T=-K\)，转置后分别保持不变与变号。唯一性来自：若一个矩阵既对称又反对称，就有 \(M=-M\)，只能是零矩阵。这展示了向量空间的另一种作用：把复杂对象分解到具有明确性质的子空间中，分别研究。</p>

<h2 id="self-check">理解检查</h2>
<p>为什么一个向量与它的 span 不是同一个对象？为什么两条过原点的坐标轴合在一起却不是子空间？一个集合是向量空间，与它是另一个空间的子空间，有什么关系？能否不消元就判断本节例子的 \((2,3,6)^T\) 为什么不在列空间？为什么全部解中的自由变化属于零空间，而非列空间？为什么 \(A^T\mathbf y=\mathbf0\) 能表达输出的约束？为什么换行要同时换 \(\mathbf b\)，却不需要改 \(\mathbf x\)？</p>
<h2 id="sources">资料与衔接</h2>
<p>依据 MIT OCW 18.06SC 第五讲 <em>Transposes, Permutations, Vector Spaces</em>、讲义摘要与习题 5.1–5.3 整理。零空间、仿射解集与最小二乘的段落是为说明概念用途而补充的衔接；下一讲将正式展开列空间与零空间。<a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/">课程来源</a>；笔记按 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 分享。</p>
<p class="note-footer"><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-04/">上一节：A = LU 分解</a> · <a href="/courses/gilbert-linear-algebra/notes/session-06/">下一节：列空间与零空间</a></p>
</section>
