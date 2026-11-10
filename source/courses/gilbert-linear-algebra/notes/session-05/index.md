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
.post-body .la-notes--session05 ul { margin: .65em 0 1.2em; padding-left: 1.35em; }
.post-body .la-notes--session05 li { margin-bottom: .55em; }
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
<div class="note-lead"><p><strong>先看一条直线为什么能独立进行线性组合，再理解向量空间及其在方程中的作用。</strong>向量空间使这些集合可以统一研究；转置连接行、列与内积，置换把换行接入矩阵运算。</p></div>
<div class="reading-links">
<p class="link-row"><a href="/files/mit-18-06sc/unit-1-session-5-summary.pdf">讲义摘要</a> <a href="/files/mit-18-06sc/unit-1-session-5-problems.pdf">习题</a> <a href="/files/mit-18-06sc/unit-1-session-5-solutions.pdf">参考解答</a></p>
<p class="link-row"><a href="#outputs">从一个向量到整个集合</a> <a href="#spaces">空间与子空间</a> <a href="#span">Span、列空间与解集</a> <a href="#transpose">转置的作用</a> <a href="#permutation">置换的作用</a> <a href="#matrix-spaces">用习题检验理解</a></p>
</div>

<h2 id="outputs">一、先把一个向量和一个向量空间分开</h2>
<p>从一个具体向量 \(\mathbf v=(1,2)^T\) 出发。它是一支从原点指向 \((1,2)\) 的箭头。如果允许把它乘以任意实数，就得到</p>
<div class="equation">\[
2\mathbf v=\begin{pmatrix}2\\4\end{pmatrix},\quad
-\mathbf v=\begin{pmatrix}-1\\-2\end{pmatrix},\quad
\tfrac12\mathbf v=\begin{pmatrix}1/2\\1\end{pmatrix},\quad
0\mathbf v=\begin{pmatrix}0\\0\end{pmatrix}.
\]</div>
<p>把所有结果收集起来，得到集合</p>
<div class="equation">\[
W=\left\{\begin{pmatrix}t\\2t\end{pmatrix}:t\in\mathbb R\right\}.
\]</div>
<p>花括号表示集合，冒号后面说明参数的取值：\(t\) 遍历全部实数。几何上，箭头的终点恰好填满直线 \(y=2x\)。所以 \(\mathbf v\) 是<strong>一个向量</strong>，\(W\) 是<strong>一整条直线上所有向量组成的集合</strong>。这里把箭头的起点固定在原点，才能用终点表示向量。</p>
<p>这条直线为什么值得作为一个整体研究？因为无论从中取哪两个向量，相加以后仍在直线上；无论把其中哪个向量乘以什么实数，也仍在直线上：</p>
<div class="equation">\[
\begin{pmatrix}s\\2s\end{pmatrix}
+\begin{pmatrix}t\\2t\end{pmatrix}
=\begin{pmatrix}s+t\\2(s+t)\end{pmatrix},\qquad
c\begin{pmatrix}t\\2t\end{pmatrix}
=\begin{pmatrix}ct\\2ct\end{pmatrix}.
\]</div>
<p>\(s+t\) 和 \(ct\) 仍是实数，所以结果仍具有 \((\text{某个数},\text{它的两倍})^T\) 的形式。<strong>只使用这条直线上的向量，就能一直做加法、数乘和任意有限次线性组合，无须补进直线外的向量。</strong>这正是这个例子成为向量空间的关键。</p>
<p>虽然这些向量写着两个分量，但选定 \(t\) 后两个分量就都确定了，只能独立选择一个实数。这解释了它为什么是一条直线。\(\mathbb R^2\) 中的向量 \((x,y)^T\) 则允许两个分量独立选择：直线是平面中的一个空间，但并没有填满平面。</p>
<p>“空间”因此不一定是整个三维世界，也不要求有面积或体积。一条过原点的直线可以是空间；整个平面也可以；只含零向量的集合也可以。这里关心的是集合中的向量能怎样运算。</p>

<h2 id="spaces">二、为什么把这些条件叫作向量空间？</h2>
<h3 id="closure">定义研究对象：集合，以及它的两种运算</h3>
<p>一个<strong>实向量空间</strong>由一个集合 \(V\)，以及集合上的<strong>加法、实数数乘</strong>组成。两种运算首先必须始终有结果留在 \(V\) 中：</p>
<div class="equation">\[
\mathbf u,\mathbf v\in V\ \Longrightarrow\ \mathbf u+\mathbf v\in V,
\qquad
c\in\mathbb R,\ \mathbf v\in V\ \Longrightarrow\ c\mathbf v\in V.
\]</div>
<p>\(\in\) 表示“属于”。这叫<strong>封闭性</strong>：“封闭”只说运算结果不会离开集合，不是说集合在几何上有边界。条件必须对任意成员、任意实系数成立，不能只试几个例子。</p>
<p>此外，加法必须可交换、可结合；有零向量，使 \(\mathbf v+\mathbf0=\mathbf v\)；每个向量有相反向量，使 \(\mathbf v+(-\mathbf v)=\mathbf0\)。数乘满足熟悉的规则：</p>
<div class="equation">\[
\begin{gathered}
c(\mathbf u+\mathbf v)=c\mathbf u+c\mathbf v,\qquad
(c+d)\mathbf v=c\mathbf v+d\mathbf v,\\
c(d\mathbf v)=(cd)\mathbf v,\qquad 1\mathbf v=\mathbf v.
\end{gathered}
\]</div>
<p>这些要求保证展开括号、合并系数、移项等推理仍有效。<strong>定义并不是断言任意向量集合都会满足这些性质，而是在选出能按同一套线性规则运算的集合。</strong>满足定义后，许多关于线性组合的结论就可以统一使用。</p>
<p>这也解释为什么特别选加法与数乘：前面解方程一直在研究 \(x_1\mathbf a_1+\cdots+x_n\mathbf a_n\)。若研究的集合能承受这两种运算，就能承受所有这样的线性组合。若组合后会离开集合，它就不足以独立承载这些运算。</p>
<h3 id="subspace">向量空间与子空间是什么关系？</h3>
<p>\(\mathbb R^2\) 表示所有二分量实向量的集合，用逐分量加法和数乘，它本身就是向量空间；\(\mathbb R^n\) 同理。现在从 \(\mathbb R^2\) 中取出前面的直线 \(W\)，<strong>沿用原来的运算</strong>。只要 \(W\) 非空，并且加法、数乘封闭，它也就成为向量空间：交换律、分配律等从 \(\mathbb R^2\) 继承，不必再证明。</p>
<p>这种“包含在一个向量空间内，沿用运算而自身也成为向量空间”的集合，叫作它的<strong>子空间</strong>。因此 \(W\) 既是一个向量空间，也是 \(\mathbb R^2\) 的子空间；前者说它自身的性质，后者说它与所在空间的关系。</p>
<p>判断子空间时，常把两项封闭性合写为：</p>
<div class="equation">\[
\boxed{W\ne\varnothing,\quad
\mathbf u,\mathbf v\in W,\ \alpha,\beta\in\mathbb R
\ \Longrightarrow\ \alpha\mathbf u+\beta\mathbf v\in W.}
\]</div>
<p>\(\varnothing\) 是空集。非空很必要：任取 \(\mathbf v\in W\)，再令系数为零，才能推出 \(\mathbf0\in W\)；令系数为 \(-1\)，得到相反向量也在里面。<strong>这条简洁判据适用于已有向量空间中的子集；抽象地定义一个新空间时，还必须指定运算并满足前面的规则。</strong></p>
<h3 id="counterexamples">过原点为什么必要？为什么又不充分？</h3>
<p>看平移后的直线 \(y=2x+1\)。向量 \((0,1)^T\) 在直线上，但乘以 2 得到 \((0,2)^T\)，已经不在直线上；两个 \((0,1)^T\) 相加也会离开。原因可以直接从方程看出：两个向量各自带着常数项 1，相加后常数项变成了 2。它也不包含零向量。</p>
<p>所以过原点是子空间的必要条件，却不是充分条件。两条坐标轴的并集含原点，也允许任意数乘，但 \((1,0)^T+(0,1)^T=(1,1)^T\) 不在其中，故不是子空间。只允许沿一条轴活动、又禁止把两个方向组合起来，无法满足加法封闭。</p>
<p>\(\{\mathbf0\}\) 则确实是子空间：里面所有加法和数乘都得到零。一个集合“大不大”“能不能画成一整块”，都不是判据；要检查的是运算。</p>

<h2 id="span">三、向量空间怎样接回矩阵与方程？</h2>
<h3 id="column-space">Span：从几个向量生成一个空间</h3>
<p>向量空间的定义告诉我们怎样判断一个集合；另一个问题是：<strong>给出几个向量，如何构造包含它们的最小子空间？</strong>把它们的全部线性组合收集起来即可。这个集合叫它们的 <strong>span（张成空间）</strong>。</p>
<p>例如，\(\mathbf v=(1,2)^T\) 的 span 就是第一部分的直线。若再给一个同向向量 \(\mathbf w=(2,4)^T=2\mathbf v\)，则 \(s\mathbf v+t\mathbf w=(s+2t)\mathbf v\)，仍只得到原来的直线。<strong>增加向量的数量，不一定增加能生成的方向。</strong></p>
<p>若改取 \(\mathbf w=(0,1)^T\)，则</p>
<div class="equation">\[
s\begin{pmatrix}1\\2\end{pmatrix}
+t\begin{pmatrix}0\\1\end{pmatrix}
=\begin{pmatrix}s\\2s+t\end{pmatrix}.
\]</div>
<p>任给目标 \((p,q)^T\)，取 \(s=p,t=q-2p\) 就能得到，所以这两个向量的 span 是整个 \(\mathbb R^2\)。它不只是两支箭头，也不只是它们的某一次相加，而是任意实系数产生的全部结果。</p>
<p>一般记为</p>
<div class="equation">\[
\operatorname{span}\{\mathbf a_1,\ldots,\mathbf a_n\}
=\left\{x_1\mathbf a_1+\cdots+x_n\mathbf a_n:x_1,\ldots,x_n\in\mathbb R\right\}.
\]</div>
<p>为什么这一定是子空间？两个结果相加时，只需把各自的组合系数相加；数乘时，把所有系数同乘那个数；系数全为零时得到零向量。为什么是最小的？任何子空间只要包含原来的向量，按封闭性就必须包含这些组合，不能再少。</p>
<p>也要区分“哪些向量生成空间”和“空间本身”：\((1,2)^T\) 与 \((2,4)^T\) 各自都能生成同一条直线。空间记录全部可用的向量，不唯一指定生成它的那组向量。</p>
<h3 id="reachable">列空间：一次说明哪些右端有解</h3>
<p>现在回到矩阵。若 \(A\) 的列是 \(\mathbf a_1,\ldots,\mathbf a_n\)，那么 \(A\mathbf x=x_1\mathbf a_1+\cdots+x_n\mathbf a_n\)。因此所有可能的输出，正好是各列的 span，叫作<strong>列空间</strong> \(C(A)\)。</p>
<p>取一个三阶例子：</p>
<div class="equation">\[
A=\begin{pmatrix}1&amp;0&amp;1\\0&amp;1&amp;1\\1&amp;1&amp;2\end{pmatrix},\qquad
A\begin{pmatrix}s\\t\\r\end{pmatrix}
=\begin{pmatrix}s+r\\t+r\\s+t+2r\end{pmatrix}.
\]</div>
<p>每个输出都满足“第三个分量等于前两个之和”。反过来，任意 \((p,q,p+q)^T\) 都可以由输入 \((p,q,0)^T\) 得到。因此</p>
<div class="equation">\[
C(A)=\left\{\begin{pmatrix}p\\q\\p+q\end{pmatrix}:p,q\in\mathbb R\right\}.
\]</div>
<p>这就是平面 \(z=x+y\)。第三列等于前两列之和，所以虽然列向量有三个，也只生成这个平面。平面上两个向量相加、任意数乘后，第三分量仍等于前两者之和，因而它确实是子空间。</p>
<p>方程 \(A\mathbf x=\mathbf b\) 是在问：有没有一组系数把列组合成 \(\mathbf b\)？所以</p>
<div class="equation">\[
\boxed{A\mathbf x=\mathbf b\text{ 有解}\iff\mathbf b\in C(A)}.
\]</div>
<p>本例 \((2,3,5)^T\) 能得到，\((2,3,6)^T\) 得不到。我们用一个平面描述了<strong>全部可解的右端</strong>，无须逐个重新消元。此前研究的单个方程，由此连接到了整个输出集合。</p>
<p>一般地，\(m\times n\) 矩阵接收 \(\mathbb R^n\) 中的输入，产生 \(\mathbb R^m\) 中的输出，\(C(A)\) 是后者的子空间。输入有几个分量、输出有几个分量，与输出实际能覆盖多大的空间，是不同的问题。</p>
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
<p>这也解释它以后为什么出现在最小二乘中：当 \(A\mathbf x=\mathbf b\) 无解时，要使误差 \(\mathbf e=\mathbf b-A\mathbf x\) 的长度最小，最佳误差必须垂直于列空间，所以</p>
<div class="equation">\[
A^T(\mathbf b-A\mathbf x)=\mathbf0
\quad\Longrightarrow\quad
A^TA\mathbf x=A^T\mathbf b.
\]</div>
<p>若误差还沿某个列空间方向有分量，就可以沿那个方向调整输出来缩短它；若已垂直，任何非零的输出调整都会增加长度平方。这是后续投影与最小二乘的入口。\(A^TA\) 总是对称，但对称不保证可逆：因为 \(\mathbf x^TA^TA\mathbf x=(A\mathbf x)^T(A\mathbf x)\)，右边是输出的长度平方，所以 \(A^TA\) 与 \(A\) 有相同的零空间；恰在 \(N(A)=\{\mathbf0\}\) 时，方阵 \(A^TA\) 才可逆。本节三阶例子就不满足。</p>

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
<p class="note-footer"><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-04/">上一节：A = LU 分解</a></p>
</section>
