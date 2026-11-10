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
<div class="note-lead"><p><strong>向量空间规定线性组合的运算规则；span 生成这样的空间；列空间把它用于判断方程是否有解。</strong>转置连接行、列与内积，置换连接换行与消元。</p></div>
<div class="reading-links">
<p class="link-row"><a href="/files/mit-18-06sc/unit-1-session-5-summary.pdf">讲义摘要</a> <a href="/files/mit-18-06sc/unit-1-session-5-problems.pdf">习题</a> <a href="/files/mit-18-06sc/unit-1-session-5-solutions.pdf">参考解答</a></p>
<p class="link-row"><a href="#outputs">向量空间</a> <a href="#spaces">子空间判据</a> <a href="#span">Span 与列空间</a> <a href="#transpose">转置的作用</a> <a href="#permutation">置换的作用</a> <a href="#matrix-spaces">用习题检验理解</a></p>
</div>

<h2 id="outputs">一、向量空间：一个集合，以及加法和数乘</h2>
<p><strong>实向量空间是一个配有加法与实数数乘的非空集合；这两种运算对集合封闭，并满足通常的运算法则。</strong>封闭指：集合中的向量相加、乘以任意实数，结果仍在集合中。运算法则包括加法的交换律、结合律、零向量与相反向量，以及数乘的分配律、结合律和 \(1\mathbf v=\mathbf v\)。</p>
<p>例如，\(\mathbb R^2\) 是所有二分量实向量的集合，按分量相加、数乘，就是向量空间。平面内的直线</p>
<div class="equation">\[
W=\{t\mathbf v:t\in\mathbb R\},\qquad
\mathbf v=\begin{pmatrix}1\\2\end{pmatrix}
\]</div>
<p>也能成为向量空间。这里 \(\mathbf v\) 是一个向量，\(W\) 是它的全部实数倍组成的集合，即直线 \(y=2x\) 上的所有向量。任取 \(s\mathbf v,t\mathbf v\in W\)，有</p>
<div class="equation">\[
s\mathbf v+t\mathbf v=(s+t)\mathbf v\in W,\qquad
c(t\mathbf v)=(ct)\mathbf v\in W.
\]</div>
<p>所以加法、数乘都留在直线上；其余运算法则沿用 \(\mathbb R^2\) 中的规则。<strong>向量空间研究的是能进行线性组合的一整个集合。</strong>“空间”可以是一条直线；向量写着两个分量，也不代表这两个分量能独立变化。</p>

<h2 id="spaces">二、子空间：怎样判断一个子集仍是向量空间？</h2>
<p id="subspace">若 \(W\) 包含在向量空间 \(V\) 中，沿用 \(V\) 的加法和数乘，且自身也是向量空间，就称 \(W\) 为 \(V\) 的<strong>子空间</strong>。例如，上面的直线既是向量空间，也是 \(\mathbb R^2\) 的子空间。</p>
<p id="closure">由于运算法则已从 \(V\) 继承，只需检查<strong>非空与封闭性</strong>：</p>
<div class="equation">\[
W\ne\varnothing,\qquad
\mathbf u,\mathbf v\in W,\ \alpha,\beta\in\mathbb R
\ \Longrightarrow\ \alpha\mathbf u+\beta\mathbf v\in W.
\]</div>
<p>\(\in\) 表示“属于”，\(\varnothing\) 表示空集。取系数为零，推出 \(\mathbf0\in W\)；取 \(-1\)，得到相反向量也在 \(W\) 中。这个判据要求<strong>任意</strong>向量、任意实系数都成立。</p>
<p id="counterexamples">两个反例说明每项要求的作用：</p>
<ul>
<li>直线 \(y=2x+1\)：\((0,1)^T\) 在其中，但 \(2(0,1)^T=(0,2)^T\) 不在，数乘不封闭。它也不含零向量。</li>
<li>两条坐标轴的并集：含零向量，数乘也封闭，但 \((1,0)^T+(0,1)^T=(1,1)^T\) 离开集合，加法不封闭。</li>
</ul>
<p>所以<strong>过原点必要，但不足以保证是子空间</strong>。\(\{\mathbf0\}\) 则满足全部条件，也是子空间。</p>

<h2 id="span">三、Span 与列空间：收集全部线性组合</h2>
<h3 id="column-space">从给定向量生成子空间</h3>
<p>给出 \(\mathbf a_1,\ldots,\mathbf a_n\)，它们的全部线性组合组成的集合叫作 <strong>span（张成空间）</strong>：</p>
<div class="equation">\[
\operatorname{span}\{\mathbf a_1,\ldots,\mathbf a_n\}
=\left\{\sum_{j=1}^n x_j\mathbf a_j:x_j\in\mathbb R\right\}.
\]</div>
<p>两次组合相加，只需把系数相加；数乘，只需把系数同乘一个数。因此 span 是子空间，而且是<strong>包含这些向量的最小子空间</strong>：凡包含这些向量的子空间，按封闭性就必须包含它们的全部组合。</p>
<p>例如，\(\mathbf v=(1,2)^T\) 生成前面的直线。再加入 \(2\mathbf v\)，因为 \(s\mathbf v+t(2\mathbf v)=(s+2t)\mathbf v\)，空间仍是原来的直线；若加入 \((0,1)^T\)，则</p>
<div class="equation">\[
s\begin{pmatrix}1\\2\end{pmatrix}
+t\begin{pmatrix}0\\1\end{pmatrix}
=\begin{pmatrix}s\\2s+t\end{pmatrix}.
\]</div>
<p>任给 \((p,q)^T\)，取 \(s=p,t=q-2p\) 就能得到，所以这次生成整个 \(\mathbb R^2\)。<strong>决定空间大小的是能生成多少独立方向，单纯增加向量数量未必有用。</strong></p>
<h3 id="reachable">矩阵的列空间决定方程是否有解</h3>
<p>矩阵乘法 \(A\mathbf x=x_1\mathbf a_1+\cdots+x_n\mathbf a_n\) 正是列的线性组合。因此各列的 span 叫作<strong>列空间</strong> \(C(A)\)，它收集 \(A\mathbf x\) 的全部可能输出。看下面的例子：</p>
<div class="equation">\[
A=\begin{pmatrix}1&amp;0&amp;1\\0&amp;1&amp;1\\1&amp;1&amp;2\end{pmatrix},\qquad
A\begin{pmatrix}s\\t\\r\end{pmatrix}
=\begin{pmatrix}s+r\\t+r\\s+t+2r\end{pmatrix}.
\]</div>
<p>所有输出的第三分量都等于前两者之和；反过来，任意 \((p,q,p+q)^T\) 都能由输入 \((p,q,0)^T\) 得到。因此</p>
<div class="equation">\[
C(A)=\left\{\begin{pmatrix}p\\q\\p+q\end{pmatrix}:p,q\in\mathbb R\right\},
\]</div>
<p>即平面 \(z=x+y\)。第三列等于前两列之和，没有增加新方向。求解方程就是寻找组合系数，所以</p>
<div class="equation">\[
\boxed{A\mathbf x=\mathbf b\text{ 有解}\iff\mathbf b\in C(A)}.
\]</div>
<p>本例 \((2,3,5)^T\) 在列空间中，有解；\((2,3,6)^T\) 不在，无解。<strong>列空间把全部可解的右端统一描述出来。</strong>对于 \(m\times n\) 矩阵，输入属于 \(\mathbb R^n\)，列空间则是输出空间 \(\mathbb R^m\) 的子空间。</p>
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
<p class="note-footer"><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-04/">上一节：A = LU 分解</a></p>
</section>
