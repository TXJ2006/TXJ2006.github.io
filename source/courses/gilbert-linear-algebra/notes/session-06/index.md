---
title: 第六节：列空间与零空间｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/session-06/
mathjax: true
toc_number: false
comments: false
---

<style>
.post-block:has(.la-notes--session06) .post-header { margin-bottom: 24px; }
.post-block:has(.la-notes--session06) .post-title { font-size: 26px; line-height: 1.45; color: #253c3c; }
.post-block:has(.la-notes--session06) .breadcrumb { display: none; }
.post-body .la-notes--session06 { max-width: 660px; margin: 0 auto; font-size: 17px; line-height: 1.85; color: #303737; }
.post-body .la-notes--session06 p { margin: 0 0 .9em; text-indent: 0; }
.post-body .la-notes--session06 h2 { margin: 2.4em 0 1em; padding: 0 0 .45em; border-bottom: 1px solid #dce5e3; color: #285d5d; font-size: 22px; line-height: 1.5; }
.post-body .la-notes--session06 h3 { margin: 1.7em 0 .75em; color: #303737; font-size: 18px; line-height: 1.6; }
.post-body .la-notes--session06 [id] { scroll-margin-top: 24px; }
.post-body .la-notes--session06 .note-nav { font-size: 14px; color: #687474; margin-bottom: 18px; }
.post-body .la-notes--session06 .note-nav a { border-color: #ccd8d5; }
.post-body .la-notes--session06 .note-lead { margin: 0 0 1.4em; padding: .1em 0 .1em 14px; border-left: 3px solid #729692; }
.post-body .la-notes--session06 .note-lead p { margin: 0; }
.post-body .la-notes--session06 .reading-links { padding: 14px 0; margin-bottom: 1.6em; border-top: 1px solid #e2e8e6; border-bottom: 1px solid #e2e8e6; font-size: 14px; line-height: 1.8; }
.post-body .la-notes--session06 .reading-links p { margin: 0; }
.post-body .la-notes--session06 .reading-links p + p { margin-top: 8px; }
.post-body .la-notes--session06 .link-row { display: flex; flex-wrap: wrap; column-gap: 18px; row-gap: 6px; }
.post-body .la-notes--session06 .equation { margin: 1em 0 1.15em; padding: 8px 2px; overflow-x: auto; overflow-y: hidden; }
.post-body .la-notes--session06 .MathJax_Display { font-size: 100%; margin: .25em 0 !important; padding: 3px 0; }
.post-body .la-notes--session06 ul, .post-body .la-notes--session06 ol { margin: .65em 0 1.2em; padding-left: 1.35em; }
.post-body .la-notes--session06 li { margin-bottom: .55em; }
.post-body .la-notes--session06 .note-extension { margin: 1.2em 0; padding: 10px 14px; border-left: 2px solid #ccd8d5; font-size: 16px; }
.post-body .la-notes--session06 .note-extension summary { cursor: pointer; color: #285d5d; }
.post-body .la-notes--session06 .note-extension[open] summary { margin-bottom: 1em; }
.post-body .la-notes--session06 .note-footer { margin-top: 1.8em; padding-top: 1em; border-top: 1px solid #e2e8e6; font-size: 14px; }
@media (max-width: 600px) {
  .post-block:has(.la-notes--session06) .post-title { font-size: 23px; }
  .post-body .la-notes--session06 { font-size: 16px; line-height: 1.85; }
  .post-body .la-notes--session06 h2 { font-size: 20px; margin-top: 2em; }
  .post-body .la-notes--session06 h3 { font-size: 17px; }
}
</style>

<section class="la-notes la-notes--session06">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-05/">第五节</a></p>
<div class="note-lead"><p><strong>列空间描述全部可能的输出；零空间描述输出为零的全部输入。</strong>同一个列关系，既限制输出范围，也使不同输入可能产生相同输出。</p></div>
<div class="reading-links">
<p class="link-row"><a href="/files/mit-18-06sc/unit-1-session-6-summary.pdf">讲义摘要</a> <a href="/files/mit-18-06sc/unit-1-session-6-problems.pdf">习题</a> <a href="/files/mit-18-06sc/unit-1-session-6-solutions.pdf">参考解答</a></p>
<p class="link-row"><a href="#subspaces">交、并与子空间</a> <a href="#definitions">列空间与零空间</a> <a href="#column-space">哪些右端有解</a> <a href="#nullspace">求出全部齐次解</a> <a href="#solution-set">非齐次解集</a></p>
</div>

<h2 id="subspaces">一、子空间的交与并</h2>
<p>以下均讨论实向量空间，沿用第五节的子空间判据：非空，且对任意实线性组合封闭。</p>
<p><strong>命题 1。</strong>设 \(S,T\) 是向量空间 \(V\) 的子空间，则交集 \(S\cap T\) 也是子空间。</p>
<p><strong>证明。</strong>两者都包含零向量，故交集非空。任取 \(\mathbf u,\mathbf v\in S\cap T\) 及 \(a,b\in\mathbb R\)。由于两者都对线性组合封闭，\(a\mathbf u+b\mathbf v\) 同时属于 \(S,T\)，因此属于交集。\(\square\)</p>
<p>并集 \(S\cup T\) 则不一定是子空间。例如，令 \(S\) 为 \(\mathbb R^3\) 中的平面 \(z=0\)，\(T\) 为 \(z\) 轴。\((1,0,0)^T\in S\)、\((0,0,1)^T\in T\)，但它们的和 \((1,0,1)^T\) 不在并集中。</p>
<p>若要得到包含两个子空间的最小子空间，应取<strong>和空间</strong>：</p>
<div class="equation">\[
S+T:=\{\mathbf s+\mathbf t:\mathbf s\in S,\ \mathbf t\in T\}
=\operatorname{span}(S\cup T).
\]</div>
<p>这里 \(\operatorname{span}(S\cup T)\) 指从并集中取有限个向量作全部线性组合。将来自 \(S\) 与 \(T\) 的项分别合并，任何组合都可写成 \(\mathbf s+\mathbf t\)；反过来，每个这样的和都是并集中两个向量的线性组合。它对加法、数乘封闭，且包含两个子空间。上述平面与轴的并集不是整个 \(\mathbb R^3\)，和空间却是整个 \(\mathbb R^3\)。</p>

<h2 id="definitions">二、两个空间的定义：先区分输入与输出</h2>
<p>设 \(A\in\mathbb R^{m\times n}\)，各列为 \(\mathbf a_1,\ldots,\mathbf a_n\in\mathbb R^m\)。矩阵确定线性映射</p>
<div class="equation">\[
T_A\colon\mathbb R^n\longrightarrow\mathbb R^m,\qquad
\mathbf x\longmapsto A\mathbf x.
\]</div>
<p><strong>定义 1（列空间与零空间）。</strong></p>
<div class="equation">\[
\begin{aligned}
C(A)&amp;:=\operatorname{span}\{\mathbf a_1,\ldots,\mathbf a_n\}\\
&amp;=\{A\mathbf x:\mathbf x\in\mathbb R^n\}\subseteq\mathbb R^m,\\[5pt]
N(A)&amp;:=\{\mathbf x\in\mathbb R^n:A\mathbf x=\mathbf0\}
\subseteq\mathbb R^n.
\end{aligned}
\]</div>
<p>\(C(A)\) 是映射的像，元素是<strong>输出向量</strong>；\(N(A)\) 是映射的核，元素是<strong>输入向量</strong>。这里 \(A\mathbf x=\mathbf0\) 中的零向量属于 \(\mathbb R^m\)，而 \(N(A)\) 所包含的零输入属于 \(\mathbb R^n\)。</p>
<p><strong>命题 2。</strong>\(C(A)\) 与 \(N(A)\) 分别是 \(\mathbb R^m\)、\(\mathbb R^n\) 的子空间。</p>
<p><strong>证明。</strong>列空间是张成空间，结论由第五节已证的命题得到。零空间含零向量；若 \(\mathbf u,\mathbf v\in N(A)\)，则对任意 \(a,b\in\mathbb R\)，</p>
<div class="equation">\[
A(a\mathbf u+b\mathbf v)
=aA\mathbf u+bA\mathbf v=\mathbf0.
\]</div>
<p>故 \(a\mathbf u+b\mathbf v\in N(A)\)，由子空间判据得证。\(\square\)</p>

<h2 id="column-space">三、讲义的矩阵：哪些右端能够得到？</h2>
<p>本节始终使用讲义中的矩阵</p>
<div class="equation">\[
A=\begin{pmatrix}
1&amp;1&amp;2\\2&amp;1&amp;3\\3&amp;1&amp;4\\4&amp;1&amp;5
\end{pmatrix}
=\begin{pmatrix}\mathbf a_1&amp;\mathbf a_2&amp;\mathbf a_3\end{pmatrix}.
\]</div>
<p>因为 \(\mathbf a_3=\mathbf a_1+\mathbf a_2\)，所以</p>
<div class="equation">\[
A\mathbf x
=(x_1+x_3)\mathbf a_1+(x_2+x_3)\mathbf a_2.
\]</div>
<p>第三列的系数可以合并到前两列的系数中，因此</p>
<div class="equation">\[
C(A)=\operatorname{span}\{\mathbf a_1,\mathbf a_2\}
=\left\{\begin{pmatrix}p+q\\2p+q\\3p+q\\4p+q\end{pmatrix}:p,q\in\mathbb R\right\}.
\]</div>
<p>为确定这个空间的维数，还要证明前两列线性无关。若 \(p\mathbf a_1+q\mathbf a_2=\mathbf0\)，前两分量给出 \(p+q=0,2p+q=0\)，相减得 \(p=0\)，继而 \(q=0\)。所以 \(\mathbf a_1,\mathbf a_2\) 构成列空间的一组<strong>基</strong>：它们张成该空间且线性无关，故列空间的<strong>维数</strong>为 2。</p>
<p>“\(\mathbb R^4\) 中的二维平面”在这里有精确含义：每个向量有四个分量，但只需两个独立系数便能唯一表示。第三列没有增加维数。</p>
<p><strong>命题 3（可解性）。</strong>对 \(\mathbf b\in\mathbb R^4\)，\(A\mathbf x=\mathbf b\) 有解，当且仅当 \(\mathbf b\in C(A)\)。这是列空间定义与 \(A\mathbf x=\sum_jx_j\mathbf a_j\) 的直接结果。</p>
<p>本例还能把“属于列空间”改写为可检查的条件：若 \(\mathbf b=(b_1,b_2,b_3,b_4)^T\in C(A)\)，则它的四个分量构成等差数列，即</p>
<div class="equation">\[
\boxed{b_2-b_1=b_3-b_2=b_4-b_3.}
\]</div>
<p>反过来，若三个差都等于 \(p\)，令 \(q=b_1-p\)，就有 \(b_i=ip+q\)（\(i=1,2,3,4\)），故 \(\mathbf b=p\mathbf a_1+q\mathbf a_2\in C(A)\)。因此这也是有解的充要条件。例如 \((1,2,3,4)^T\) 对应有解，\((1,2,3,5)^T\) 对应无解。</p>

<h2 id="nullspace">四、同一个列关系，怎样确定全部齐次解？</h2>
<p>列关系 \(\mathbf a_1+\mathbf a_2-\mathbf a_3=\mathbf0\) 给出一个非零输入：</p>
<div class="equation">\[
A\begin{pmatrix}1\\1\\-1\end{pmatrix}=\mathbf0.
\]</div>
<p>这只找到了一个零空间向量。要求出<strong>全部</strong>零空间，利用前面已经证明的线性无关性：</p>
<div class="equation">\[
\begin{aligned}
A\mathbf x=\mathbf0
&amp;\iff (x_1+x_3)\mathbf a_1+(x_2+x_3)\mathbf a_2=\mathbf0\\
&amp;\iff x_1+x_3=0,\quad x_2+x_3=0.
\end{aligned}
\]</div>
<p>令 \(t=-x_3\)，便得到</p>
<div class="equation">\[
\boxed{N(A)=\left\{t\begin{pmatrix}1\\1\\-1\end{pmatrix}:t\in\mathbb R\right\}.}
\]</div>
<p>因此零空间是 \(\mathbb R^3\) 中的一条直线，维数为 1。列空间中的二维平面位于 \(\mathbb R^4\)，这条直线位于 \(\mathbb R^3\)；它们分别描述输出与输入。</p>
<p>这个非零方向也解释了为什么输入不能由输出唯一恢复。对任意 \(\mathbf x\in\mathbb R^3\)，</p>
<div class="equation">\[
A\left(\mathbf x+t\begin{pmatrix}1\\1\\-1\end{pmatrix}\right)=A\mathbf x.
\]</div>
<p>增加前两列各 \(t\) 份，同时减少第三列 \(t\) 份，输出完全抵消。因此<strong>两次输入产生同一输出，当且仅当它们的差属于零空间</strong>：\(A\mathbf x=A\mathbf y\iff\mathbf x-\mathbf y\in N(A)\)。</p>

<h2 id="solution-set">五、非齐次解集：零空间的平移</h2>
<p><strong>命题 4（全部解的结构）。</strong>设 \(A\in\mathbb R^{m\times n}\)，且 \(A\mathbf x=\mathbf b\) 至少有一个解 \(\mathbf x_p\)。则它的解集为</p>
<div class="equation">\[
\boxed{\{\mathbf x:A\mathbf x=\mathbf b\}
=\mathbf x_p+N(A)
:=\{\mathbf x_p+\mathbf z:\mathbf z\in N(A)\}.}
\]</div>
<p><strong>证明。</strong>若 \(\mathbf x\) 是解，则 \(A(\mathbf x-\mathbf x_p)=\mathbf b-\mathbf b=\mathbf0\)，故 \(\mathbf x-\mathbf x_p\in N(A)\)。反过来，若 \(\mathbf z\in N(A)\)，则 \(A(\mathbf x_p+\mathbf z)=\mathbf b\)，仍是解。两边相互包含，结论成立。\(\square\)</p>
<p>讲义取 \(\mathbf b=\mathbf a_1=(1,2,3,4)^T\)。一个特解是 \(\mathbf x_p=(1,0,0)^T\)，因此全部解为</p>
<div class="equation">\[
\mathbf x=\begin{pmatrix}1\\0\\0\end{pmatrix}
+t\begin{pmatrix}1\\1\\-1\end{pmatrix},\qquad t\in\mathbb R.
\]</div>
<p>它是一条与 \(N(A)\) 平行的直线，经过 \((1,0,0)^T\)；取 \(t=-1\)，还经过讲义中的 \((0,-1,1)^T\)。它不经过原点，因为 \(A\mathbf0=\mathbf0\ne\mathbf b\)，故不是线性子空间。这种子空间的平移称为<strong>仿射子空间</strong>。</p>
<p>特解的选择不影响整个解集：若 \(\mathbf x'_p\) 是另一特解，两者之差属于 \(N(A)\)，所以 \(\mathbf x'_p+N(A)=\mathbf x_p+N(A)\)。</p>
<p>由此可分别判断：\(\mathbf b\notin C(A)\) 时无解；\(\mathbf b\in C(A)\) 时有解，且恰在 \(N(A)=\{\mathbf0\}\) 时唯一。若零空间包含非零向量，实数参数的不同取值便给出无穷多个解。</p>

<h2 id="self-check">理解检查</h2>
<p>能否证明等差条件既必要又充分？为什么只找到 \((1,1,-1)^T\) 一个齐次解，还不足以声称已求出零空间？为什么非齐次解集平行于零空间，却不一定等于零空间？</p>
<h2 id="sources">资料与衔接</h2>
<p>依据 MIT OCW 18.06SC 第六讲 <em>Column Space and Nullspace</em> 与讲义摘要整理。和空间与习题 6.1 衔接；右端的等差条件、完整证明及基与维数的简要说明为补充。后续课程将用消元、主元变量与自由变量系统求解 \(A\mathbf x=\mathbf0\)。<a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/">课程来源</a>；笔记按 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 分享。</p>
<p class="note-footer"><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-05/">上一节：转置、置换与向量空间</a></p>
</section>
