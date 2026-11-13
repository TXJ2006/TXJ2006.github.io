---
title: 第八节：求解 Ax = b：有解条件与全部解｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/session-08/
mathjax: true
toc_number: false
comments: false
---

<style>
.post-block:has(.la-notes--session08) .post-header { margin-bottom: 24px; }
.post-block:has(.la-notes--session08) .post-title { font-size: 26px; line-height: 1.45; color: #253c3c; }
.post-block:has(.la-notes--session08) .breadcrumb { display: none; }
.post-body .la-notes--session08 { max-width: 660px; margin: 0 auto; font-size: 17px; line-height: 1.85; color: #303737; }
.post-body .la-notes--session08 p { margin: 0 0 .9em; text-indent: 0; }
.post-body .la-notes--session08 h2 { margin: 2.4em 0 1em; padding: 0 0 .45em; border-bottom: 1px solid #dce5e3; color: #285d5d; font-size: 22px; line-height: 1.5; }
.post-body .la-notes--session08 h3 { margin: 1.7em 0 .75em; color: #303737; font-size: 18px; line-height: 1.6; }
.post-body .la-notes--session08 [id] { scroll-margin-top: 24px; }
.post-body .la-notes--session08 .note-nav { font-size: 14px; color: #687474; margin-bottom: 18px; }
.post-body .la-notes--session08 .note-nav a { border-color: #ccd8d5; }
.post-body .la-notes--session08 .note-lead { margin: 0 0 1.4em; padding: .1em 0 .1em 14px; border-left: 3px solid #729692; }
.post-body .la-notes--session08 .note-lead p { margin: 0; }
.post-body .la-notes--session08 .reading-links { padding: 14px 0; margin-bottom: 1.6em; border-top: 1px solid #e2e8e6; border-bottom: 1px solid #e2e8e6; font-size: 14px; line-height: 1.8; }
.post-body .la-notes--session08 .reading-links p { margin: 0; }
.post-body .la-notes--session08 .reading-links p + p { margin-top: 8px; }
.post-body .la-notes--session08 .link-row { display: flex; flex-wrap: wrap; column-gap: 18px; row-gap: 6px; }
.post-body .la-notes--session08 .equation { margin: 1em 0 1.15em; padding: 8px 2px; overflow-x: auto; overflow-y: hidden; }
.post-body .la-notes--session08 .MathJax_Display { font-size: 100%; margin: .25em 0 !important; padding: 3px 0; }
.post-body .la-notes--session08 ul, .post-body .la-notes--session08 ol { margin: .65em 0 1.2em; padding-left: 1.35em; }
.post-body .la-notes--session08 li { margin-bottom: .55em; }
.post-body .la-notes--session08 .note-extension { margin: 1.2em 0; padding: 10px 14px; border-left: 2px solid #ccd8d5; font-size: 16px; }
.post-body .la-notes--session08 .note-extension summary { cursor: pointer; color: #285d5d; }
.post-body .la-notes--session08 .note-extension[open] summary { margin-bottom: 1em; }
.post-body .la-notes--session08 .note-table { overflow-x: auto; margin: 1em 0; }
.post-body .la-notes--session08 .note-table table { width: 100%; font-size: 15px; line-height: 1.7; }
.post-body .la-notes--session08 .note-table th, .post-body .la-notes--session08 .note-table td { padding: 10px 12px; vertical-align: top; }
.post-body .la-notes--session08 .note-footer { margin-top: 1.8em; padding-top: 1em; border-top: 1px solid #e2e8e6; font-size: 14px; }
@media (max-width: 600px) {
  .post-block:has(.la-notes--session08) .post-title { font-size: 23px; }
  .post-body .la-notes--session08 { font-size: 16px; line-height: 1.85; }
  .post-body .la-notes--session08 h2 { font-size: 20px; margin-top: 2em; }
  .post-body .la-notes--session08 h3 { font-size: 17px; }
}
</style>

<section class="la-notes la-notes--session08">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-07/">第七节</a></p>
<div class="note-lead"><p>第七节求出了 \(A\mathbf x=\mathbf0\) 的全部解。本节保留同一个矩阵，把右端换成给定的 \(\mathbf b\)。新增的问题是“有没有解”；一旦有解，第七节的零空间就能描述所有解之间的差别。</p></div>
<div class="reading-links">
<p class="link-row"><a href="/files/mit-18-06sc/unit-1-session-8-summary.pdf">讲义摘要</a> <a href="/files/mit-18-06sc/unit-1-session-8-problems.pdf">习题</a> <a href="/files/mit-18-06sc/unit-1-session-8-solutions.pdf">参考解答</a></p>
<p class="link-row"><a href="#solvability">何时有解</a> <a href="#particular">找一个特解</a> <a href="#complete">为什么得到全部解</a> <a href="#geometry">解集的几何</a> <a href="#rank">秩与解的个数</a></p>
</div>

<h2 id="solvability">一、消元时，右端为什么必须一起变？</h2>
<p>设 \(A\in\mathbb R^{m\times n}\)，\(\mathbf b\in\mathbb R^m\) 是给定的输出，\(\mathbf x\in\mathbb R^n\) 是待求的输入。上一节右端为零，每次行操作后仍为零；现在若把第二条方程减去第一条的两倍，右端也必须作相同运算，否则就不再是等价方程。</p>
<p>因此把 \(A\) 和 \(\mathbf b\) 放在一起，写成<strong>增广矩阵</strong> \([A\mid\mathbf b]\)。竖线只用于分隔未知数的系数与右端，最后一列不对应新的未知数。若可逆行操作的乘积为 \(E\)，则</p>
<div class="equation">\[
A\mathbf x=\mathbf b
\iff EA\mathbf x=E\mathbf b.
\]</div>
<p>若这些行操作把 \(A\) 化为行最简形 \(R=EA\)，将变换后的右端记为 \(\mathbf c=E\mathbf b\)，则增广矩阵变为 \([R\mid\mathbf c]\)，方程变为 \(R\mathbf x=\mathbf c\)。\(\mathbf c\) 是计算所得的右端，不是另一个未知向量；未知数仍是 \(\mathbf x\)，分量顺序也没有改变。</p>
<p>讲义继续使用第七节的矩阵：</p>
<div class="equation">\[
A=\begin{pmatrix}
1&amp;2&amp;2&amp;2\\
2&amp;4&amp;6&amp;8\\
3&amp;6&amp;8&amp;10
\end{pmatrix},\qquad
\mathbf b=\begin{pmatrix}b_1\\b_2\\b_3\end{pmatrix}.
\]</div>
<p>先看行之间的关系：第 3 行等于前两行之和。因此，若前三条方程同时成立，必有 \(b_3=b_1+b_2\)。这是有解的<strong>必要条件</strong>；要确认没有别的限制，还需消元。</p>
<p>依次作第 2 行减第 1 行的 2 倍、第 3 行减第 1 行的 3 倍，再作第 3 行减去更新后的第 2 行，得到</p>
<div class="equation">\[
[A\mid\mathbf b]\longrightarrow
\left(\begin{array}{cccc|c}
1&amp;2&amp;2&amp;2&amp;b_1\\
0&amp;0&amp;2&amp;4&amp;b_2-2b_1\\
0&amp;0&amp;0&amp;0&amp;b_3-b_2-b_1
\end{array}\right).
\]</div>
<p>最后一行表示 \(0=b_3-b_2-b_1\)，它只检查右端，不再约束未知数。若 \(b_3\ne b_1+b_2\)，这就是矛盾；若相等，最后一行成为 \(0=0\)。此时例如取 \(x_2=x_4=0\)，前两行便能依次解出</p>
<div class="equation">\[
x_3=\frac{b_2-2b_1}{2},\qquad
x_1=b_1-2x_3=3b_1-b_2.
\]</div>
<p>这构造出了一个解，说明上述条件也<strong>充分</strong>。因此，本例有解的充要条件是</p>
<div class="equation">\[
b_3=b_1+b_2.
\]</div>
<p><strong>一般地，消元后每个系数零行的右端都必须为零；全部满足时，其余主元行就能回代求解。</strong>满足这些条件，称右端与方程组相容。注意查看的是系数部分：像 \([0\ 0\ 0\ 0\mid1]\) 这样的行在增广矩阵中并非零行，却代表无解。</p>
<p>列空间给出同一件事的另一种说法：\(A\mathbf x=\sum_jx_j\mathbf a_j\)，其中 \(\mathbf a_j\) 是 \(A\) 的第 \(j\) 列，所以</p>
<div class="equation">\[
A\mathbf x=\mathbf b\text{ 有解}
\iff\mathbf b\in C(A).
\]</div>
<p>本例具体为 \(C(A)=\{(b_1,b_2,b_3)^T:b_3=b_1+b_2\}\)。消元把“\(\mathbf b\) 在列空间中”转化成了可以直接检查的等式。</p>

<h2 id="particular">二、为什么把自由变量取零，就能找一个特解？</h2>
<p>按讲义取 \(\mathbf b=(1,5,6)^T\)，它满足 \(6=1+5\)。继续把主元上方消成零、主元化为 1：</p>
<div class="equation">\[
\left(\begin{array}{cccc|c}
1&amp;2&amp;2&amp;2&amp;1\\
0&amp;0&amp;2&amp;4&amp;3\\
0&amp;0&amp;0&amp;0&amp;0
\end{array}\right)
\longrightarrow
\left(\begin{array}{cccc|c}
1&amp;2&amp;0&amp;-2&amp;-2\\
0&amp;0&amp;1&amp;2&amp;3/2\\
0&amp;0&amp;0&amp;0&amp;0
\end{array}\right).
\]</div>
<p>这里先作第 1 行减第 2 行，再把第 2 行除以 2。左侧仍是第七节的 \(R\)，右端已相应改变。方程为</p>
<div class="equation">\[
\begin{cases}
x_1+2x_2-2x_4=-2,\\
x_3+2x_4=\dfrac32.
\end{cases}
\]</div>
<p>主元列仍是第 1、3 列，故 \(x_2,x_4\) 仍是自由变量。有解条件已满足，它们任取实数都能求出 \(x_1,x_3\)。<strong>取零只是一种方便的选择</strong>：令 \(x_2=x_4=0\)，两条方程分别给出 \(x_1=-2\)、\(x_3=3/2\)，得到</p>
<div class="equation">\[
\mathbf x_p=\begin{pmatrix}-2\\0\\3/2\\0\end{pmatrix},
\qquad A\mathbf x_p=\begin{pmatrix}1\\5\\6\end{pmatrix}.
\]</div>
<p>\(\mathbf x_p\) 称为一个<strong>特解</strong>（particular solution）；下标 \(p\) 取自 particular，标记一个固定的解。非齐次特解满足 \(A\mathbf x_p=\mathbf b\)；第七节的齐次特殊解满足 \(A\mathbf s_i=\mathbf0\)，两者作用不同。取自由变量为零能找到特解的前提，是方程组相容；若已出现矛盾行，怎样取值都无法补救。</p>

<h2 id="complete">三、全部解为什么等于“特解加零空间”？</h2>
<p>直接令 \(x_2=s,x_4=t\)，其中 \(s,t\) 可独立取任意实数。由刚才的两条方程，</p>
<div class="equation">\[
\mathbf x=\begin{pmatrix}
-2-2s+2t\\s\\3/2-2t\\t
\end{pmatrix}.
\]</div>
<p>按常数项、\(s\) 的系数、\(t\) 的系数分别收集，就得到</p>
<div class="equation">\[
\mathbf x=
\underbrace{\begin{pmatrix}-2\\0\\3/2\\0\end{pmatrix}}_{\mathbf x_p}
+s\underbrace{\begin{pmatrix}-2\\1\\0\\0\end{pmatrix}}_{\mathbf s_1}
+t\underbrace{\begin{pmatrix}2\\0\\-2\\1\end{pmatrix}}_{\mathbf s_2}.
\]</div>
<p>例如第一分量 \(-2-2s+2t\) 分别贡献 \(-2,-2,2\)；第二分量 \(s=0+1s+0t\) 分别贡献 \(0,1,0\)。两个固定向量 \(\mathbf s_1,\mathbf s_2\) 正是第七节的齐次特殊解，满足 \(A\mathbf s_i=\mathbf0\)。右端改变了特解，但 \(A\) 不变，零空间就不变。</p>
<p>这个结构适用于任意有解的线性方程组。固定一个特解 \(\mathbf x_p\)，用 \(\mathbf z\) 表示零空间中任取的向量，即 \(A\mathbf z=\mathbf0\)。证明需要两个方向：</p>
<ul>
<li><strong>相加总能得到解：</strong>\(A(\mathbf x_p+\mathbf z)=A\mathbf x_p+A\mathbf z=\mathbf b+\mathbf0=\mathbf b\)。所以加上任意齐次解，都不会改变输出。</li>
<li><strong>任意解都能这样写：</strong>若 \(A\mathbf x=\mathbf b\)，则 \(A(\mathbf x-\mathbf x_p)=\mathbf b-\mathbf b=\mathbf0\)，故差向量 \(\mathbf x-\mathbf x_p\) 必在零空间中。</li>
</ul>
<div class="equation">\[
\{\mathbf x:A\mathbf x=\mathbf b\}
=\{\mathbf x_p+\mathbf z:\mathbf z\in N(A)\}
=\mathbf x_p+N(A).
\]</div>
<p>最后的“向量加集合”是前一个集合的简写：把 \(\mathbf x_p\) 加到零空间的每个向量上。第一个方向保证没有添解，第二个方向保证没有漏解。<strong>两个非齐次解的差一定是齐次解；两个非齐次解的和一般不是原方程的解。</strong>因为后者的输出是 \(2\mathbf b\)。</p>

<h2 id="geometry">四、两个空间分别在回答什么？</h2>
<p>\(C(A)\subseteq\mathbb R^m\) 收集能够达到的<strong>输出</strong>，回答给定 \(\mathbf b\) 是否有解。\(N(A)\subseteq\mathbb R^n\) 收集不改变输出的<strong>输入改变量</strong>：若 \(\mathbf x\) 已是一个解，则沿 \(\mathbf z\in N(A)\) 改成 \(\mathbf x+\mathbf z\)，输出仍为 \(\mathbf b\)。</p>
<p>本例的列空间是 \(\mathbb R^3\) 中的二维平面 \(b_3=b_1+b_2\)；零空间是 \(\mathbb R^4\) 中过原点的二维平面。对 \(\mathbf b=(1,5,6)^T\)，解集是后一个平面平移到经过 \(\mathbf x_p\) 的位置，方向仍由 \(\mathbf s_1,\mathbf s_2\) 决定。</p>
<p>这种子空间的平移称为<strong>仿射子空间</strong>。当 \(\mathbf b\ne\mathbf0\) 时，零向量不满足方程，所以解集不含原点，也就不是向量子空间；当 \(\mathbf b=\mathbf0\) 时，解集就是 \(N(A)\)。有解时，平移不改变自由参数的数目，解集的仿射维数就是 \(\dim N(A)=n-r\)；本例为 2。</p>
<p>特解也可以换：若 \(\mathbf x_q\) 是另一个特解，\(\mathbf x_q-\mathbf x_p\in N(A)\)，因此 \(\mathbf x_q+N(A)=\mathbf x_p+N(A)\)。换特解只是换了平面上的参照点，并未改变解集。</p>

<h2 id="rank">五、秩怎样同时控制有解性与唯一性？</h2>
<p>设 \(A\) 有 \(m\) 行、\(n\) 列：原方程组有 \(m\) 条方程、\(n\) 个未知数。这里 \(r=\operatorname{rank}(A)\) 是<strong>系数矩阵</strong>的秩，等于消元所得主元数。每个主元占一个不同的行和列，因此 \(r\le m\)、\(r\le n\)。同一个 \(r\)，从行与列两个方向看，分别回答不同的问题。</p>

<h3 id="rank-conditions">为什么有 m − r 个右端条件？</h3>
<p>将 \([A\mid\mathbf b]\) 化为 \([R\mid\mathbf c]\)，其中 \(R=EA\)、\(\mathbf c=E\mathbf b\)，\(E\) 是实施这些可逆行操作的矩阵。在行最简形 \(R\) 中，每个非零行恰有一个主元，故有 \(r\) 个非零行；总共 \(m\) 行，余下恰为 \(m-r\) 个零行。</p>
<p>这 \(m-r\) 行在方程中分别写成</p>
<div class="equation">\[
0=c_{r+1},\quad\ldots,\quad0=c_m.
\]</div>
<p>\(c_i\) 是变换后右端 \(\mathbf c\) 的第 \(i\) 个分量，由给定的 \(\mathbf b\) 算出。这些等式中已经没有未知数：它们要求右端满足相容条件。任一等式失败，方程组便无解；全部通过，剩下 \(r\) 个主元行就能求解。</p>
<p><strong>条件约束的是 \(E\mathbf b\) 的分量。</strong>写回原来的 \(b_i\) 时，通常是若干分量之间的线性关系。又因为 \(E\) 可逆，\(\mathbf b\) 遍历 \(\mathbb R^m\) 时，\(\mathbf c\) 也遍历 \(\mathbb R^m\)；将其中 \(m-r\) 个不同分量置零，确实给出 \(m-r\) 个独立条件。</p>

<h3 id="rank-freedom">为什么有 n − r 个自由变量？</h3>
<p>\(n\) 列对应 \(n\) 个未知数，其中 \(r\) 个主元列对应主元变量，其余 \(n-r\) 列对应自由变量。行最简形的主元为 1，且其所在列的其余元素均为零，所以一条主元行中不含别的主元变量。把自由变量项移到右端，这条方程就唯一确定了自己的主元变量。</p>
<p>相容条件通过后，可以<strong>独立任取 \(n-r\) 个自由变量，再唯一确定全部主元变量</strong>。不同的自由变量取值会给出不同的解，因为它们本身就是解向量的分量；任意解也都能由自己的自由分量这样恢复。因此，有解时恰有 \(n-r\) 个自由参数。</p>
<p>右端为零时，这一过程得到 \(N(A)\)，所以 \(\dim N(A)=n-r\)；一般右端相容时，解集是 \(\mathbf x_p+N(A)\)，自由参数数目保持不变。若 \(n-r=0\)，有解时只剩一个解；若 \(n-r>0\)，自由参数可以连续变化，便有无穷多个解。<strong>自由变量的数目只在相容之后决定解的多少。</strong></p>

<h3 id="rank-example">在本讲的矩阵中，把这两个数分别数出来</h3>
<p>本例 \(m=3,n=4,r=2\)。对一般右端完成消元后，\(R\mathbf x=\mathbf c\) 是</p>
<div class="equation">\[
\begin{cases}
x_1+2x_2-2x_4=3b_1-b_2,\\
x_3+2x_4=\dfrac{b_2-2b_1}{2},\\
0=b_3-b_1-b_2.
\end{cases}
\]</div>
<p><strong>按行数：</strong>三行中两行有主元，剩下一行检查 \(b_3=b_1+b_2\)，正好一个条件，即 \(m-r=1\)。<strong>按列数：</strong>四列中第 1、3 列是主元列，\(x_2,x_4\) 自由，正好两个自由变量，即 \(n-r=2\)。</p>
<p>若 \(\mathbf b=(1,5,6)^T\)，第三行通过，任取 \(x_2=s,x_4=t\) 就能由前两行算出 \(x_1,x_3\)，得到第三部分的全部解。若改成 \((1,5,7)^T\)，第三行变成 \(0=1\)，无解。两次的矩阵、秩和自由列完全相同，差别在于右端是否相容。</p>

<h3 id="full-row-rank">满行秩：每个右端都有解</h3>
<p>若 \(r=m\)，称 \(A\) <strong>满行秩</strong>。此时 \(m-r=0\)，消元后没有零行，因而没有要额外检查的右端条件；不论 \(\mathbf b\) 怎样给定，都能用主元求解。反过来，若 \(r&lt;m\)，就有零行，而可逆变换 \(\mathbf c=E\mathbf b\) 能取得任意 \(\mathbf c\)，必然存在使零行对应分量非零的右端，所以有些 \(\mathbf b\) 无解。因此，<strong>每个右端都有解，当且仅当满行秩</strong>；从列空间看，这等价于全部输出空间都能达到：</p>
<div class="equation">\[
r=m\iff C(A)=\mathbb R^m.
\]</div>
<p>满行秩说明所有输出都能达到，但若 \(n>m\)，仍有 \(n-m\) 个自由变量，每个右端都对应无穷多个输入。例如 \(A=\begin{pmatrix}1&amp;2\end{pmatrix}\) 的 \(m=r=1,n=2\)：对任意实数 \(b\)，\(x_1+2x_2=b\) 都有解，全部解为 \((x_1,x_2)=(b-2t,t)\)，其中 \(t\in\mathbb R\)。</p>

<h3 id="full-column-rank">满列秩：有解时唯一</h3>
<p>若 \(r=n\)，称 \(A\) <strong>满列秩</strong>。此时 \(n-r=0\)，没有自由变量，齐次方程只有零解，\(N(A)=\{\mathbf0\}\)。一个右端只要相容，主元行就把所有未知数唯一确定。反过来，若 \(r&lt;n\)，自由变量至少一个，任何相容右端都会有无穷多个解。因此，<strong>有解时唯一，当且仅当满列秩</strong>；从零空间看，这等价于没有非零的齐次解：</p>
<div class="equation">\[
r=n\iff N(A)=\{\mathbf0\}.
\]</div>
<p>满列秩说明同一个输出不会对应两个不同输入，但若 \(m>n\)，仍有 \(m-n\) 个右端条件。例如 \(A=(1,2)^T\) 的 \(m=2,n=r=1\)，方程是 \(x=b_1,\ 2x=b_2\)。消元给出条件 \(b_2=2b_1\)：\((1,2)^T\) 对应唯一解 \(x=1\)，\((1,3)^T\) 则无解。</p>
<p>“满行秩”与“满列秩”中的“满”，分别表示每行、每列都能有主元。因此满行秩要求 \(m\le n\)，满列秩要求 \(n\le m\)；这些大小关系只是必要条件，是否满秩仍要检查主元。以下按两种条件是否同时满足分类：</p>
<div class="note-table"><table>
<thead><tr><th>秩与大小</th><th>右端是否受限</th><th>自由变量数</th><th>解的个数</th></tr></thead>
<tbody>
<tr><td>\(r=m=n\)</td><td>任意 \(\mathbf b\) 都可</td><td>0</td><td>每个右端恰有 1 解</td></tr>
<tr><td>\(r=n&lt;m\)</td><td>须 \(\mathbf b\in C(A)\)</td><td>0</td><td>0 或 1</td></tr>
<tr><td>\(r=m&lt;n\)</td><td>任意 \(\mathbf b\) 都可</td><td>\(n-m&gt;0\)</td><td>每个右端无穷多解</td></tr>
<tr><td>\(r&lt;m,\ r&lt;n\)</td><td>须 \(\mathbf b\in C(A)\)</td><td>\(n-r&gt;0\)</td><td>0 或无穷多</td></tr>
</tbody></table></div>
<p>两种满秩同时成立时，\(m=n=r\)，\(A\) 是可逆方阵，\(R=I_n\)（\(n\) 阶单位矩阵），每个右端都有唯一解 \(\mathbf x=A^{-1}\mathbf b\)。本讲的 \(3\times4\) 矩阵秩为 2，属于最后一类：\(m-r=1\) 个零行给出条件 \(b_3=b_1+b_2\)，满足后留下 \(n-r=2\) 个自由参数。</p>
<details class="note-extension"><summary>讲义的分块式：R 中的 I、F 和零块分别做什么？</summary>
<p>讲义汇总表把主元列排在自由列之前。如果主元列原本不连续，需要重排列，同时按相同顺序重排未知向量的分量。用波浪号标记这种排列，则一般形式为</p>
<div class="equation">\[
\widetilde R=\begin{pmatrix}I_r&amp;F\\0&amp;0\end{pmatrix},
\qquad
\widetilde{\mathbf x}=\begin{pmatrix}\mathbf x_P\\\mathbf x_F\end{pmatrix},
\qquad
\mathbf c=\begin{pmatrix}\mathbf c_P\\\mathbf c_0\end{pmatrix}.
\]</div>
<p>\(I_r\) 是 \(r\) 阶单位矩阵；\(F\) 有 \(r\) 行、\(n-r\) 列，收集自由变量的系数。\(\mathbf x_P,\mathbf x_F\) 分别收集主元变量和自由变量，下标 \(P,F\) 用来区分两组。\(\mathbf c_P\) 是变换后右端的前 \(r\) 个分量，\(\mathbf c_0\) 是最后 \(m-r\) 个分量；下标 0 只是标记零行对应的部分，不表示它已等于零。</p>
<p>下方零块有 \(m-r\) 行，左右分别有 \(r\) 列和 \(n-r\) 列。因此 \(\widetilde R\widetilde{\mathbf x}=\mathbf c\) 恰好表示</p>
<div class="equation">\[
\mathbf x_P+F\mathbf x_F=\mathbf c_P,
\qquad \mathbf0=\mathbf c_0.
\]</div>
<p>第二式检查有解条件；通过后，第一式给出 \(\mathbf x_P=\mathbf c_P-F\mathbf x_F\)。取 \(\mathbf x_F=\mathbf0\) 得到特解，其余项给出齐次解：</p>
<div class="equation">\[
\widetilde{\mathbf x}
=\begin{pmatrix}\mathbf c_P\\\mathbf0\end{pmatrix}
+\begin{pmatrix}-F\\I_{n-r}\end{pmatrix}\mathbf x_F.
\]</div>
<p>若 \(r=m\)，下方零行消失，不再限制右端；若 \(r=n\)，自由变量组和 \(F\) 消失，相容时解唯一。这就是汇总表中各个矩阵形状的来源。求解后必须把分量恢复到原顺序；本例主元列是第 1、3 列，不能未经重排就把原来的 \(R\) 写成左上角为单位矩阵的形式。若 \(r=0\)，没有主元方程，只有右端必须为零的条件；相容时所有未知数都自由。</p>
</details>
<p>这里讨论实数上的方程组。只要有解且存在非零齐次解 \(\mathbf z\)，\(\mathbf x_p+t\mathbf z\) 随不同实数 \(t\) 就给出不同的解。因此解的个数只能是 0、1 或无穷多，不会恰好有两个解。</p>

<h2 id="self-check">理解检查</h2>
<p>把本例右端改成 \((1,5,7)^T\)，矛盾出现在哪一行？为什么 \(\mathbf x_p+\mathbf s_1\) 仍是一个特解？一个 \(4\times3\)、秩为 2 的矩阵，在行最简形中有几个零行、几个自由变量；它对相容右端有多少解？为什么这两个数分别由 \(m-r\)、\(n-r\) 决定？</p>
<h2 id="sources">资料与衔接</h2>
<p>依据 MIT OCW 18.06SC 第八讲 <em>Solving Ax = b: Row Reduced Form R</em> 及<a href="/files/mit-18-06sc/unit-1-session-8-summary.pdf">讲义摘要</a>整理。矩阵、右端 \((1,5,6)^T\)、特解、四种秩的分类及分块形状来自讲义；有解条件的充分性、全部解的反向证明、分块公式与换列说明作了展开，小矩阵例子为补充。下一节将系统整理线性无关、基与维数。<a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/">课程来源</a>；笔记按 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 分享。</p>
<p class="note-footer"><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-07/">上一节：主元变量与特殊解</a> · <a href="/courses/gilbert-linear-algebra/notes/session-09/">下一节：线性无关、基与维数</a></p>
</section>
