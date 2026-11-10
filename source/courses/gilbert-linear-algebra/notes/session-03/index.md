---
title: 第三节：矩阵乘法与逆矩阵｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/session-03/
mathjax: true
toc_number: false
comments: false
---

<style>
.post-block:has(.la-notes--session03) .post-header { margin-bottom: 24px; }
.post-block:has(.la-notes--session03) .post-title { font-size: 26px; line-height: 1.45; color: #253c3c; }
.post-block:has(.la-notes--session03) .breadcrumb { display: none; }
.post-body .la-notes--session03 { max-width: 660px; margin: 0 auto; font-size: 17px; line-height: 1.85; color: #303737; }
.post-body .la-notes--session03 p { margin: 0 0 .9em; text-indent: 0; }
.post-body .la-notes--session03 h2 { margin: 2.4em 0 1em; padding: 0 0 .45em; border-bottom: 1px solid #dce5e3; color: #285d5d; font-size: 22px; line-height: 1.5; }
.post-body .la-notes--session03 h3 { margin: 1.7em 0 .75em; color: #303737; font-size: 18px; line-height: 1.6; }
.post-body .la-notes--session03 [id] { scroll-margin-top: 24px; }
.post-body .la-notes--session03 .note-nav { font-size: 14px; color: #687474; margin-bottom: 18px; }
.post-body .la-notes--session03 .note-nav a { border-color: #ccd8d5; }
.post-body .la-notes--session03 .note-lead { margin: 0 0 1.4em; padding: .1em 0 .1em 14px; border-left: 3px solid #729692; }
.post-body .la-notes--session03 .note-lead p { margin: 0; }
.post-body .la-notes--session03 .reading-links { padding: 14px 0; margin-bottom: 1.6em; border-top: 1px solid #e2e8e6; border-bottom: 1px solid #e2e8e6; font-size: 14px; line-height: 1.8; }
.post-body .la-notes--session03 .reading-links p { margin: 0; }
.post-body .la-notes--session03 .reading-links p + p { margin-top: 8px; }
.post-body .la-notes--session03 .link-row { display: flex; flex-wrap: wrap; column-gap: 18px; row-gap: 6px; }
.post-body .la-notes--session03 .equation { margin: 1em 0 1.15em; padding: 8px 2px; overflow-x: auto; overflow-y: hidden; }
.post-body .la-notes--session03 .MathJax_Display { font-size: 100%; margin: .25em 0 !important; padding: 3px 0; }
.post-body .la-notes--session03 ul { margin: .65em 0 1.2em; padding-left: 1.35em; }
.post-body .la-notes--session03 li { margin-bottom: .55em; }
.post-body .la-notes--session03 .note-footer { margin-top: 1.8em; padding-top: 1em; border-top: 1px solid #e2e8e6; font-size: 14px; }
@media (max-width: 600px) {
  .post-block:has(.la-notes--session03) .post-title { font-size: 23px; }
  .post-body .la-notes--session03 { font-size: 16px; line-height: 1.85; }
  .post-body .la-notes--session03 h2 { font-size: 20px; margin-top: 2em; }
  .post-body .la-notes--session03 h3 { font-size: 17px; }
}
</style>

<section class="la-notes la-notes--session03">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-02/">第二节</a></p>
<div class="note-lead"><p><strong>矩阵乘法描述线性变换的连续作用；逆矩阵负责把这种作用撤销。</strong></p></div>
<div class="reading-links">
<p class="link-row"><a href="/files/mit-18-06sc/unit-1-session-3-summary.pdf">讲义摘要</a> <a href="/files/mit-18-06sc/unit-1-session-3-problems.pdf">习题</a> <a href="/files/mit-18-06sc/unit-1-session-3-solutions.pdf">参考解答</a></p>
<p class="link-row"><a href="#product">矩阵乘法</a> <a href="#four-views">四种理解</a> <a href="#order">作用顺序</a> <a href="#inverse">逆矩阵</a> <a href="#gauss-jordan">Gauss–Jordan elimination</a></p>
</div>

<h2 id="product">一、矩阵乘法先看尺寸</h2>
<p>若 \(A\) 是 \(m\times n\) 矩阵，\(B\) 是 \(n\times p\) 矩阵，则 \(AB\) 是 \(m\times p\) 矩阵：</p>
<div class="equation">\[
A_{m\times n}B_{n\times p}=(AB)_{m\times p}.
\]</div>
<p>中间的 \(n\) 必须相同，因为 \(B\) 的输出必须正好成为 \(A\) 的输入。结果保留左边矩阵的行数和右边矩阵的列数。</p>

<h2 id="four-views">二、同一个乘积的四种理解</h2>
<p>下面始终使用同一例子：</p>
<div class="equation">\[
A=\begin{pmatrix}1&amp;3\\2&amp;7\end{pmatrix},\quad
B=\begin{pmatrix}2&amp;1\\0&amp;1\end{pmatrix},\quad
AB=\begin{pmatrix}2&amp;4\\4&amp;9\end{pmatrix}.
\]</div>
<h3 id="entrywise">1. 行乘列：计算单个元素</h3>
<p>设 \(C=AB\)。\(c_{ij}\) 表示结果的第 \(i\) 行、第 \(j\) 列元素：取 \(A\) 的第 \(i\) 行与 \(B\) 的第 \(j\) 列，对应相乘再相加，这叫作<strong>点积</strong>。</p>
<div class="equation">\[
c_{ij}=\sum_{k=1}^{n}a_{ik}b_{kj}.
\]</div>
<p>例如 \(c_{12}=1\cdot1+3\cdot1=4\)。下标 \(k\) 表示当前配对的是 \(A\) 的第 \(k\) 列和 \(B\) 的第 \(k\) 行。</p>

<h3 id="by-columns">2. 按列看：\(B\) 的每一列分别经过 \(A\)</h3>
<p>把 \(B\) 的第 \(j\) 列记为 \(\mathbf b_j\)。固定 \(j\)，对 \(A\) 的每一行做点积，得到的正是 \(A\mathbf b_j\)：</p>
<div class="equation">\[
AB=[\,A\mathbf b_1\ A\mathbf b_2\ \cdots\ A\mathbf b_p\,].
\]</div>
<p>记 \(A\) 的列为 \(\mathbf a_1,\mathbf a_2\)。本例中 \(B\) 的两列分别是 \((2,0)^{\mathsf T}\)、\((1,1)^{\mathsf T}\)，所以结果的两列是 \(2\mathbf a_1=(2,4)^{\mathsf T}\)、\(\mathbf a_1+\mathbf a_2=(4,9)^{\mathsf T}\)。上标 \(\mathsf T\) 表示把横排坐标转为列。</p>

<h3 id="by-rows">3. 按行看：\(A\) 的每一行分别作用于 \(B\)</h3>
<p>同理，固定第 \(i\) 行，让 \(A\) 的这一行 \(\mathbf r_i\) 依次乘 \(B\) 的各列，就得到 \(AB\) 的整行 \(\mathbf r_iB\)。</p>
<p>记 \(B\) 的两行为 \(\mathbf s_1=(2,1)\)、\(\mathbf s_2=(0,1)\)。结果的第一行是 \(1\mathbf s_1+3\mathbf s_2=(2,4)\)，第二行是 \(2\mathbf s_1+7\mathbf s_2=(4,9)\)。<strong>左边矩阵的每一行规定一组行组合系数</strong>，这正是消元矩阵的原理。</p>

<h3 id="outer-products">4. 列乘行：把乘积拆成简单矩阵的和</h3>
<p>记 \(\mathbf a_k\) 为 \(A\) 的第 \(k\) 列，\(\mathbf s_k\) 为 \(B\) 的第 \(k\) 行。列乘行的尺寸是 \((m\times1)(1\times p)=m\times p\)，所以它产生一个矩阵。例如：</p>
<div class="equation">\[
\begin{aligned}
AB
&amp;=\begin{pmatrix}1\\2\end{pmatrix}\begin{pmatrix}2&amp;1\end{pmatrix}
+\begin{pmatrix}3\\7\end{pmatrix}\begin{pmatrix}0&amp;1\end{pmatrix}\\[6pt]
&amp;=\begin{pmatrix}2&amp;1\\4&amp;2\end{pmatrix}
+\begin{pmatrix}0&amp;3\\0&amp;7\end{pmatrix}.
\end{aligned}
\]</div>
<p>一般地，\(AB=\sum_{k=1}^{n}\mathbf a_k\mathbf s_k\)：<strong>左边第 \(k\) 列乘右边第 \(k\) 行，再把这些矩阵相加。</strong>第 \(i,j\) 个位置的和仍是 \(\sum_k a_{ik}b_{kj}\)，因此四种视角计算的是同一个乘积。每项的列都是 \(\mathbf a_k\) 的倍数；两向量都非零时，它只沿一个方向产生输出。</p>
<p>讲义还介绍<strong>分块乘法</strong>：把矩阵切成尺寸匹配的矩形块，仍按“行乘列”相乘相加。例如 \(C=AB\) 的左上块是 \(C_{11}=A_{11}B_{11}+A_{12}B_{21}\)。这里每个符号代表一整块矩阵，块内乘法的顺序仍不能交换。</p>

<h2 id="order">三、为什么乘法顺序不能随便交换</h2>
<p>把 \(A,B\) 看成两个固定的线性变换。先做 \(B\)，得到中间向量 \(B\mathbf x\)；再把<strong>这个结果</strong>交给 \(A\)，得到 \(A(B\mathbf x)\)。前面已经算出，\(AB\) 的第 \(j\) 列就是 \(A\mathbf b_j\)，其中 \(\mathbf b_j\) 是 \(B\) 的第 \(j\) 列。因此，对任意输入 \(\mathbf x=(x_1,\ldots,x_p)^{\mathsf T}\)，</p>
<div class="equation">\[
\begin{aligned}
A(B\mathbf x)
&amp;=A\left(\sum_{j=1}^{p}x_j\mathbf b_j\right)\\
&amp;=\sum_{j=1}^{p}x_jA\mathbf b_j
=(AB)\mathbf x.
\end{aligned}
\]</div>
<p>第二步使用了 \(A\) 保持线性组合的性质。这个等式说明：<strong>矩阵乘积 \(AB\) 恰好记录“先 \(B\)、后 \(A\)”的复合变换。</strong>顺序从右向左读，是因为输入先交给最靠近它的 \(B\)。</p>
<p>交换成 \(BA\) 后，中间向量变为 \(A\mathbf x\)，后一步也变为 \(B\)，得到的是 \(B(A\mathbf x)\)。线性性质并不保证这与原结果相同。用本节的矩阵，取同一个输入 \(\mathbf e_1=(1,0)^{\mathsf T}\)：</p>
<div class="equation">\[
\begin{aligned}
A(B\mathbf e_1)&amp;=A\begin{pmatrix}2\\0\end{pmatrix}
=\begin{pmatrix}2\\4\end{pmatrix},\\[6pt]
B(A\mathbf e_1)&amp;=B\begin{pmatrix}1\\2\end{pmatrix}
=\begin{pmatrix}4\\2\end{pmatrix}.
\end{aligned}
\]</div>
<p>同一个输入产生不同输出，就足以说明 \(AB\ne BA\)。某些矩阵确实满足 \(AB=BA\)，但需要具体证明。结合律 \((AB)C=A(BC)\) 则始终成立（尺寸匹配时）：两边都按“先 \(C\)、再 \(B\)、最后 \(A\)”作用；<strong>改变括号只改变计算的分组，交换矩阵才会改变作用顺序。</strong></p>

<h2 id="inverse">四、逆矩阵：撤销一个可逆变换</h2>
<p>单位矩阵 \(I\) 不改变向量。若方阵 \(A\) 有一个矩阵 \(A^{-1}\) 满足</p>
<div class="equation">\[
AA^{-1}=A^{-1}A=I,
\]</div>
<p>就称 \(A\) 可逆。若 \(\mathbf b=A\mathbf x\)，左乘 \(A^{-1}\) 得 \(\mathbf x=A^{-1}\mathbf b\)：逆矩阵把输出还原为输入。</p>
<p>若有非零向量 \(\mathbf z\) 满足 \(A\mathbf z=\mathbf0\)，则 \(A(\mathbf x+\mathbf z)=A\mathbf x\)：两个不同输入得到同一输出，无法唯一还原，\(A\) 就不可逆。例如：</p>
<div class="equation">\[
\begin{pmatrix}1&amp;3\\2&amp;6\end{pmatrix}
\begin{pmatrix}3\\-1\end{pmatrix}
=\begin{pmatrix}0\\0\end{pmatrix}.
\]</div>
<p>反过来，若 \(A\mathbf x=\mathbf0\) 只有零解，消元就没有自由变量。对方阵，这意味着每行每列都有主元，所以任意右端都能唯一求解，也能用下面的方法求逆。这与第一节的列观点一致：<strong>列向量线性无关，就是只有零系数才能组合出零向量。</strong></p>
<p>若 \(A,B\) 都可逆，撤销“先 \(B\) 后 \(A\)”时，要先撤销 \(A\)，再撤销 \(B\)，所以 \((AB)^{-1}=B^{-1}A^{-1}\)。</p>

<h2 id="gauss-jordan">五、Gauss–Jordan elimination</h2>
<p>仍用前面的 \(A=\begin{pmatrix}1&amp;3\\2&amp;7\end{pmatrix}\)。把单位矩阵接在右边，对整个增广矩阵做行操作：</p>
<div class="equation">\[
\begin{aligned}
&amp;\left[\begin{array}{cc|cc}1&amp;3&amp;1&amp;0\\2&amp;7&amp;0&amp;1\end{array}\right]\\[8pt]
\xrightarrow{\ R_2\leftarrow R_2-2R_1\ }&amp;
\left[\begin{array}{cc|cc}1&amp;3&amp;1&amp;0\\0&amp;1&amp;-2&amp;1\end{array}\right]\\[8pt]
\xrightarrow{\ R_1\leftarrow R_1-3R_2\ }&amp;
\left[\begin{array}{cc|cc}1&amp;0&amp;7&amp;-3\\0&amp;1&amp;-2&amp;1\end{array}\right].
\end{aligned}
\]</div>
<p>普通消元先消掉主元下方的数，得到上三角；<strong>Gauss–Jordan elimination 继续消掉主元上方的数，并把主元归一为 1</strong>。左半边成为 \(I\)，右半边就是：</p>
<div class="equation">\[
A^{-1}=
\begin{pmatrix}7&amp;-3\\-2&amp;1\end{pmatrix}.
\]</div>
<p>为什么这样能求逆？\(AA^{-1}=I\) 按列看，就是同时解两个方程：</p>
<div class="equation">\[
A\mathbf x_1=\mathbf e_1,\qquad A\mathbf x_2=\mathbf e_2,
\]</div>
<p>这里 \(\mathbf e_1=(1,0)^{\mathsf T}\)、\(\mathbf e_2=(0,1)^{\mathsf T}\) 是单位矩阵的两列。把对应解 \(\mathbf x_1=(7,-2)^{\mathsf T}\)、\(\mathbf x_2=(-3,1)^{\mathsf T}\) 排成列，就得到 \(A^{-1}\)。右侧两列始终接受相同的行操作，因此是在同时解这两个方程组。</p>
<p>从上一节的语言看，若所有消元矩阵的乘积为 \(E\)，那么</p>
<div class="equation">\[
E[A\mid I]=[EA\mid E]=[I\mid E].
\]</div>
<p>每步可逆，所以 \(E\) 可逆；由 \(EA=I\) 得 \(A=E^{-1}\)，因此 \(E=A^{-1}\)。<strong>若左半边无法化为 \(I\)，则 \(A\) 不可逆；只解某个 \(A\mathbf x=\mathbf b\) 时，直接消元即可，不必先求逆。</strong></p>

<h2 id="self-check">理解检查</h2>
<p>你应该能回答：为什么 \(AB\) 的列是 \(A\) 作用在 \(B\) 的列上？为什么 \(AB\mathbf x\) 要先算 \(B\mathbf x\)？为什么 \([A\mid I]\to[I\mid A^{-1}]\) 能同时得到所有右端向量的解法？</p>
<h2 id="sources">资料</h2>
<p>依据 MIT OCW 18.06SC 第三讲 <em>Multiplication and Inverse Matrices</em> 及其讲义摘要整理；习题链接供练习使用。<a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/">课程来源</a>；笔记按 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 分享。</p>
<p class="note-footer"><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-02/">上一节：矩阵消元</a></p>
</section>
