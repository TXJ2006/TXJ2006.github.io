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
.post-body .la-notes--session03 figure { margin: 1.5em 0 1.8em; }
.post-body .la-notes--session03 figure img { display: block; width: 100%; max-width: 660px; height: auto; margin: 0 auto; }
.post-body .la-notes--session03 figcaption { margin-top: .7em; color: #506565; font-size: 14px; line-height: 1.7; }
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
<p class="link-row"><a href="#product">矩阵乘法</a> <a href="#four-views">四种理解</a> <a href="#order">作用顺序</a> <a href="#inverse">逆矩阵</a> <a href="#gauss-jordan">高斯—约旦求逆</a></p>
</div>

<h2 id="product">一、矩阵乘法先看尺寸</h2>
<p>若 \(A\) 是 \(m\times n\) 矩阵，\(B\) 是 \(n\times p\) 矩阵，则 \(AB\) 是 \(m\times p\) 矩阵：</p>
<div class="equation">\[
A_{m\times n}B_{n\times p}=(AB)_{m\times p}.
\]</div>
<p>中间的 \(n\) 必须相同，因为 \(B\) 的输出必须正好成为 \(A\) 的输入。结果保留左边矩阵的行数和右边矩阵的列数。</p>

<h2 id="four-views">二、同一个乘积的四种理解</h2>
<h3 id="entrywise">1. 行乘列：计算单个元素</h3>
<p>设 \(C=AB\)，则 \(C\) 的第 \(i,j\) 个元素为 \(A\) 的第 \(i\) 行与 \(B\) 的第 \(j\) 列的点积：</p>
<div class="equation">\[
c_{ij}=\sum_{k=1}^{n}a_{ik}b_{kj}.
\]</div>
<p>这是实际计算公式，但它没有单独说明乘法的整体结构。</p>

<h3 id="by-columns">2. 按列看：\(B\) 的每一列分别经过 \(A\)</h3>
<p>把 \(B\) 写成列向量：</p>
<div class="equation">\[
B=[\,\mathbf b_1\ \mathbf b_2\ \cdots\ \mathbf b_p\,].
\]</div>
<p>那么</p>
<div class="equation">\[
AB=[\,A\mathbf b_1\ A\mathbf b_2\ \cdots\ A\mathbf b_p\,].
\]</div>
<p>所以矩阵乘矩阵，就是把右边矩阵的每一列交给左边矩阵处理。结合第一节，\(A\mathbf b_j\) 又是 \(A\) 的列向量的线性组合。</p>

<h3 id="by-rows">3. 按行看：\(A\) 的每一行分别作用于 \(B\)</h3>
<p>如果 \(\mathbf r_i\) 是 \(A\) 的第 \(i\) 行，那么 \(AB\) 的第 \(i\) 行就是</p>
<div class="equation">\[
\mathbf r_iB.
\]</div>
<p>也就是说，左乘矩阵的行决定如何组合右边矩阵的行。这正是上一节用左乘矩阵表示行操作的原因。</p>

<h3 id="outer-products">4. 列乘行：把乘积拆成简单矩阵的和</h3>
<p>把 \(A\) 的列记为 \(\mathbf a_1,\ldots,\mathbf a_n\)，把 \(B\) 的行记为 \(\mathbf s_1^{\mathsf T},\ldots,\mathbf s_n^{\mathsf T}\)，则</p>
<div class="equation">\[
AB=\mathbf a_1\mathbf s_1^{\mathsf T}
+\mathbf a_2\mathbf s_2^{\mathsf T}
+\cdots+
\mathbf a_n\mathbf s_n^{\mathsf T}.
\]</div>
<p>每一项“列乘行”都是一个矩阵；整个乘积是这些矩阵的和。这个观点揭示了 \(AB\) 是由 \(A\) 的列和 \(B\) 的行共同构成的。</p>

<h2 id="order">三、为什么乘法顺序不能随便交换</h2>
<figure>
<img src="/images/courses/gilbert-linear-algebra/session-03/matrix-composition.svg" alt="输入向量先经过矩阵 B，再经过矩阵 A，结果是 ABx。" loading="lazy">
<figcaption>图 1：\(AB\mathbf x=A(B\mathbf x)\)。右边的矩阵先作用。</figcaption>
</figure>
<p>矩阵乘法表示变换的连续作用：</p>
<div class="equation">\[
(AB)\mathbf x=A(B\mathbf x).
\]</div>
<p>因此 \(AB\) 表示先做 \(B\)，再做 \(A\)；\(BA\) 则是相反的顺序。两个变换通常不能交换顺序，就像“先旋转再投影”和“先投影再旋转”一般不同。</p>

<h2 id="inverse">四、逆矩阵：撤销一个可逆变换</h2>
<p>单位矩阵 \(I\) 不改变向量：</p>
<div class="equation">\[
I\mathbf x=\mathbf x.
\]</div>
<p>对方阵 \(A\)，若存在 \(A^{-1}\) 使</p>
<div class="equation">\[
AA^{-1}=A^{-1}A=I,
\]</div>
<p>就称 \(A\) 可逆。若 \(\mathbf b=A\mathbf x\)，则</p>
<div class="equation">\[
\mathbf x=A^{-1}\mathbf b.
\]</div>
<p>逆矩阵不是把每个元素分别取倒数，而是撤销整个矩阵代表的变换。</p>
<p>方阵不可逆，意味着至少有一个非零向量 \(\mathbf z\) 满足 \(A\mathbf z=\mathbf0\)。于是不同输入可能有相同输出，不能从输出唯一恢复输入。对方阵而言，下面几件事是等价的：</p>
<ul>
<li>\(A\) 可逆；</li>
<li>\(A\mathbf x=\mathbf0\) 只有零解；</li>
<li>\(A\) 的列向量线性无关；</li>
<li>对每个 \(\mathbf b\)，\(A\mathbf x=\mathbf b\) 都有唯一解。</li>
</ul>

<h2 id="gauss-jordan">五、用高斯—约旦消元求逆</h2>
<p>把单位矩阵接在 \(A\) 的右边，同时对左右两部分做同样的行操作：</p>
<div class="equation">\[
\left[\begin{array}{cc|cc}
1&3&1&0\\
2&7&0&1
\end{array}\right]
\longrightarrow
\left[\begin{array}{cc|cc}
1&0&7&-3\\
0&1&-2&1
\end{array}\right].
\]</div>
<p>左半边从 \(A\) 变成 \(I\)，右半边就变成 \(A^{-1}\)：</p>
<div class="equation">\[
A^{-1}=
\begin{pmatrix}7&-3\\-2&1\end{pmatrix}.
\]</div>
<p>为什么右半边就是逆矩阵？每一列分别满足</p>
<div class="equation">\[
A\mathbf x_1=\mathbf e_1,\qquad A\mathbf x_2=\mathbf e_2,
\]</div>
<p>而把 \(\mathbf x_1,\mathbf x_2\) 作为列排起来，正是满足 \(AA^{-1}=I\) 的矩阵。</p>
<p>从上一节的语言看，若所有消元矩阵的乘积为 \(E\)，那么</p>
<div class="equation">\[
E[A\mid I]=[EA\mid E]=[I\mid E].
\]</div>
<p>因为 \(EA=I\)，所以 \(E=A^{-1}\)。求逆就是把撤销 \(A\) 所需的全部行操作集中记录下来。</p>

<h2 id="self-check">理解检查</h2>
<p>你应该能回答：为什么 \(AB\) 的列是 \(A\) 作用在 \(B\) 的列上？为什么 \(AB\mathbf x\) 要先算 \(B\mathbf x\)？为什么 \([A\mid I]\to[I\mid A^{-1}]\) 能同时得到所有右端向量的解法？</p>
<h2 id="sources">资料</h2>
<p>依据 MIT OCW 18.06SC 第三讲 <em>Multiplication and Inverse Matrices</em> 及其习题整理。<a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/">课程来源</a>；笔记与原创图示按 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 分享。</p>
<p class="note-footer"><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-02/">上一节：矩阵消元</a></p>
</section>
