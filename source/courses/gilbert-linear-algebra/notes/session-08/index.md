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
<div class="note-lead"><p>第七节求出了 \(A\mathbf x=\mathbf0\) 的全部解。本节把右端换成给定的 \(\mathbf b\)：<strong>先判断能否到达这个输出，再找一个解，最后用零空间写出全部解。</strong></p></div>
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
<p>所以将 \(A\) 化为行最简形 \(R\) 后，右端一般已变为 \(E\mathbf b\)，不能直接写成 \(R\mathbf x=\mathbf b\)。未知向量的分量顺序则没有改变。</p>
<p>讲义继续使用第七节的矩阵：</p>
<div class="equation">\[
A=\begin{pmatrix}
1&amp;2&amp;2&amp;2\\
2&amp;4&amp;6&amp;8\\
3&amp;6&amp;8&amp;10
\end{pmatrix},\qquad
\mathbf b=\begin{pmatrix}b_1\\b_2\\b_3\end{pmatrix}.
\]</div>
<p>依次作第 2 行减第 1 行的 2 倍、第 3 行减第 1 行的 3 倍，再作第 3 行减去更新后的第 2 行，得到</p>
<div class="equation">\[
[A\mid\mathbf b]\longrightarrow
\left(\begin{array}{cccc|c}
1&amp;2&amp;2&amp;2&amp;b_1\\
0&amp;0&amp;2&amp;4&amp;b_2-2b_1\\
0&amp;0&amp;0&amp;0&amp;b_3-b_2-b_1
\end{array}\right).
\]</div>
<p>最后一行表示 \(0=b_3-b_2-b_1\)。若右端非零，就是矛盾；若等于零，这一行成为 \(0=0\)，剩下两行都能用主元回代。因此本例有解的<strong>充要条件</strong>是</p>
<div class="equation">\[
b_3=b_1+b_2.
\]</div>
<p>这也能从原方程看出：\(A\) 的第 3 行是前两行之和，所以右端必须具有相同关系。<strong>一般地，系数矩阵消成零行时，对应的右端也必须为零。</strong>这些条件全部满足后，自由变量可任取，主元变量能逐个求出，故条件也充分。</p>
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
<p>主元列仍是第 1、3 列，故 \(x_2,x_4\) 仍是自由变量。有解条件已满足，它们任取实数都能求出 \(x_1,x_3\)。<strong>取零只是一种方便的选择</strong>：令 \(x_2=x_4=0\)，得到</p>
<div class="equation">\[
\mathbf x_p=\begin{pmatrix}-2\\0\\3/2\\0\end{pmatrix},
\qquad A\mathbf x_p=\begin{pmatrix}1\\5\\6\end{pmatrix}.
\]</div>
<p>\(\mathbf x_p\) 称为一个<strong>特解</strong>（particular solution）；下标 \(p\) 取自 particular，只用于标记这个固定的解。它不是新的未知数，也不表示全部解。先判断有解、再取自由变量为零，这个顺序不能省：若消元已出现矛盾行，怎样取自由变量都无法补救。</p>

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
<p>这种子空间的平移称为<strong>仿射子空间</strong>。当 \(\mathbf b\ne\mathbf0\) 时，零向量不满足方程，所以解集不含原点，也就不是向量子空间；当 \(\mathbf b=\mathbf0\) 时，解集就是 \(N(A)\)。本例解集仍由两个参数唯一描述，其仿射维数为 2。</p>
<p>特解也可以换：若 \(\mathbf x_q\) 是另一个特解，\(\mathbf x_q-\mathbf x_p\in N(A)\)，因此 \(\mathbf x_q+N(A)=\mathbf x_p+N(A)\)。换特解只是换了平面上的参照点，并未改变解集。</p>

<h2 id="rank">五、满行秩管有解，满列秩管唯一</h2>
<p>设 \(A\) 有 \(m\) 行、\(n\) 列，秩为 \(r\)。主元在不同的行、列，所以 \(r\le m\)、\(r\le n\)。以下两个问题必须区分：<strong>每个右端是否都有解？一个有解的右端是否只有一个解？</strong></p>
<h3>满列秩：没有改变输入却保持输出的余地</h3>
<p>若 \(r=n\)，称 \(A\) <strong>满列秩</strong>：每列都是主元列，各列线性无关，没有自由变量，\(N(A)=\{\mathbf0\}\)。若已有特解，全部解公式便只剩 \(\mathbf x=\mathbf x_p\)。所以<strong>有解时唯一</strong>；但若 \(\mathbf b\notin C(A)\)，仍然无解。</p>
<p>例如 \(A=(1,2)^T\) 是 \(2\times1\) 的满列秩矩阵。对 \(\mathbf b=(1,2)^T\)，唯一解为 \(x=1\)；对 \(\mathbf b=(1,3)^T\)，两条方程要求 \(x=1\) 且 \(2x=3\)，所以无解。</p>
<h3>满行秩：每个给定的输出都能达到</h3>
<p>若 \(r=m\)，称 \(A\) <strong>满行秩</strong>：每行都有主元。消元后没有系数为零的行，也就没有额外的右端相容条件，任何 \(\mathbf b\in\mathbb R^m\) 都能回代求解。等价地，\(\dim C(A)=m\)，所以 \(C(A)=\mathbb R^m\)。</p>
<p>若同时 \(n>m\)，仍有 \(n-m\) 个自由变量，每个右端都有无穷多个解。例如 \(A=\begin{pmatrix}1&amp;2\end{pmatrix}\)，对任意实数 \(b\)，方程 \(x_1+2x_2=b\) 的全部解是 \((x_1,x_2)=(b-2t,t)\)，其中 \(t\in\mathbb R\)。</p>
<p>因此，下面的分类来自同一条求解逻辑：先检查右端是否相容；相容后，再看有没有自由变量。</p>
<div class="note-table"><table>
<thead><tr><th>秩与大小</th><th>右端是否受限</th><th>自由变量数</th><th>解的个数</th></tr></thead>
<tbody>
<tr><td>\(r=m=n\)</td><td>任意 \(\mathbf b\) 都可</td><td>0</td><td>每个右端恰有 1 解</td></tr>
<tr><td>\(r=n&lt;m\)</td><td>须 \(\mathbf b\in C(A)\)</td><td>0</td><td>0 或 1</td></tr>
<tr><td>\(r=m&lt;n\)</td><td>任意 \(\mathbf b\) 都可</td><td>\(n-m&gt;0\)</td><td>每个右端无穷多解</td></tr>
<tr><td>\(r&lt;m,\ r&lt;n\)</td><td>须 \(\mathbf b\in C(A)\)</td><td>\(n-r&gt;0\)</td><td>0 或无穷多</td></tr>
</tbody></table></div>
<p>两种满秩同时成立时，\(m=n=r\)，\(A\) 是可逆方阵，\(R=I_n\)，每个右端都有唯一解 \(\mathbf x=A^{-1}\mathbf b\)。本讲的 \(3\times4\) 矩阵秩为 2，属于最后一类：先满足 \(b_3=b_1+b_2\)，才有由两个自由参数描述的全部解。</p>
<p>这里讨论实数上的方程组。只要有解且存在非零齐次解 \(\mathbf z\)，\(\mathbf x_p+t\mathbf z\) 随不同实数 \(t\) 就给出不同的解。因此解的个数只能是 0、1 或无穷多，不会恰好有两个解。</p>

<h2 id="self-check">理解检查</h2>
<p>把本例右端改成 \((1,5,7)^T\)，矛盾出现在哪一行？为什么 \(\mathbf x_p+\mathbf s_1\) 仍是一个特解？满列秩为什么只能保证“有解时唯一”？若 \(A\) 有两行三列，能否只凭大小断言每个右端都有解？</p>
<h2 id="sources">资料与衔接</h2>
<p>依据 MIT OCW 18.06SC 第八讲 <em>Solving Ax = b: Row Reduced Form R</em> 及<a href="/files/mit-18-06sc/unit-1-session-8-summary.pdf">讲义摘要</a>整理。矩阵、右端 \((1,5,6)^T\)、特解和四种秩的分类来自讲义；全部解的反向证明、符号说明及小矩阵例子作了补充。下一节将系统整理线性无关、基与维数。<a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/">课程来源</a>；笔记按 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 分享。</p>
<p class="note-footer"><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-07/">上一节：主元变量与特殊解</a></p>
</section>
