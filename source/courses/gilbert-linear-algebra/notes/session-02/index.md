---
title: 第二节：矩阵消元｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/session-02/
mathjax: true
toc_number: false
comments: false
---

<style>
.post-block:has(.la-notes--session02) .post-header { margin-bottom: 24px; }
.post-block:has(.la-notes--session02) .post-title { font-size: 26px; line-height: 1.45; color: #253c3c; }
.post-block:has(.la-notes--session02) .breadcrumb { display: none; }
.post-body .la-notes--session02 { max-width: 660px; margin: 0 auto; font-size: 17px; line-height: 1.85; color: #303737; }
.post-body .la-notes--session02 p { margin: 0 0 .9em; text-indent: 0; }
.post-body .la-notes--session02 h2 { margin: 2.4em 0 1em; padding: 0 0 .45em; border-top: 0; border-bottom: 1px solid #dce5e3; color: #285d5d; font-size: 22px; line-height: 1.5; }
.post-body .la-notes--session02 h3 { margin: 1.8em 0 .75em; padding: 0; border: 0; color: #303737; font-size: 18px; line-height: 1.6; }
.post-body .la-notes--session02 [id] { scroll-margin-top: 24px; }
.post-body .la-notes--session02 .note-nav { font-size: 14px; color: #687474; margin-bottom: 18px; }
.post-body .la-notes--session02 .note-nav a { border-color: #ccd8d5; }
.post-body .la-notes--session02 .note-lead { margin: 0 0 1.4em; padding: .1em 0 .1em 14px; border-left: 3px solid #729692; }
.post-body .la-notes--session02 .note-lead p { margin: 0; }
.post-body .la-notes--session02 .reading-links { padding: 14px 0; margin-bottom: 1.6em; border-top: 1px solid #e2e8e6; border-bottom: 1px solid #e2e8e6; background: none; font-size: 14px; line-height: 1.8; }
.post-body .la-notes--session02 .reading-links p { margin: 0; }
.post-body .la-notes--session02 .reading-links p + p { margin-top: 8px; }
.post-body .la-notes--session02 .link-row { display: flex; flex-wrap: wrap; column-gap: 18px; row-gap: 6px; }
.post-body .la-notes--session02 .equation { margin: 1em 0 1.15em; padding: 8px 2px; overflow-x: auto; overflow-y: hidden; }
.post-body .la-notes--session02 .MathJax_Display { font-size: 100%; margin: .25em 0 !important; padding: 3px 0; }
.post-body .la-notes--session02 ul { margin: .65em 0 1.2em; padding-left: 1.35em; }
.post-body .la-notes--session02 li { margin-bottom: .6em; }
.post-body .la-notes--session02 .note-footer { margin-top: 1.8em; padding-top: 1em; border-top: 1px solid #e2e8e6; font-size: 14px; }
@media (max-width: 600px) {
  .post-block:has(.la-notes--session02) .post-title { font-size: 23px; }
  .post-body .la-notes--session02 { font-size: 16px; line-height: 1.85; }
  .post-body .la-notes--session02 h2 { font-size: 20px; margin-top: 2em; }
  .post-body .la-notes--session02 h3 { font-size: 17px; }
  .post-body .la-notes--session02 .MathJax_Display { font-size: 96%; }
}
</style>

<section class="la-notes la-notes--session02">
<p class="note-nav" id="session-02"><a href="/courses/gilbert-linear-algebra/notes/">← 返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-01/">第一节</a></p>
<div class="note-lead"><p><strong>用可逆的行操作，把方程组化成上三角形式，再回代求解。</strong>消元改变方程的表达，保持未知数的含义与整个解集不变。</p></div>
<div class="reading-links">
<p class="link-row"><a href="/files/mit-18-06sc/unit-1-session-2-summary.pdf">讲义摘要</a> <a href="/files/mit-18-06sc/unit-1-session-2-problems.pdf">习题</a> <a href="/files/mit-18-06sc/unit-1-session-2-solutions.pdf">参考解答</a></p>
<p class="link-row"><a href="#elimination">消元与回代</a> <a href="#left-multiplication">消元矩阵</a> <a href="#inverse">逆矩阵与解集不变</a> <a href="#pivots">主元与换行</a> <a href="#order">多步操作的顺序</a></p>
</div>

<h2 id="elimination">一、消元与回代</h2>
<p>沿用讨论中的例子：</p>
<div class="equation">\[
\begin{cases}2x+y=5,\\4x+3y=11.\end{cases}
\]</div>
<p>用第一式消掉第二式的 \(x\)。第一行的系数 \(2\) 是<strong>主元</strong>，要消去的系数为 \(4\)，因此消元倍数是 \(4/2=2\)。记 \(R_i\) 为第 \(i\) 行，执行 \(R_2\leftarrow R_2-2R_1\)。</p>
<p>把常数也放在右侧一列，得到<strong>增广矩阵</strong>。这样可以同时记录等号两边的操作：</p>
<div class="equation">\[
\left[\begin{array}{cc|c}2&amp;1&amp;5\\4&amp;3&amp;11\end{array}\right]
\xrightarrow{\ R_2-2R_1\ }
\left[\begin{array}{cc|c}2&amp;1&amp;5\\0&amp;1&amp;1\end{array}\right].
\]</div>
<p>左边的系数矩阵变成<strong>上三角矩阵</strong>，即主对角线下方为零。第二式给出 \(y=1\)，代入第一式得 \(x=2\)。从最后一个方程向上求未知数，叫作<strong>回代</strong>。这里两个主元分别为 \(2,1\)。</p>

<h2 id="left-multiplication">二、消元矩阵：把行操作写成乘法</h2>
<p>第一节已经说明：左侧行向量的系数组合矩阵的行。因此：</p>
<div class="equation">\[
EA=U,\qquad
\begin{pmatrix}1&amp;0\\-2&amp;1\end{pmatrix}
\begin{pmatrix}2&amp;1\\4&amp;3\end{pmatrix}
=\begin{pmatrix}2&amp;1\\0&amp;1\end{pmatrix}.
\]</div>
<p>\(E\) 的第一行 \((1,0)\) 表示保留原第一行；第二行 \((-2,1)\) 表示原第二行减去原第一行的两倍。\(E\) 叫作<strong>消元矩阵</strong>，\(U\) 表示所得的上三角矩阵。</p>
<p>常数向量 \(\mathbf b\) 必须接受同样的操作：</p>
<div class="equation">\[
E\mathbf b=
\begin{pmatrix}1&amp;0\\-2&amp;1\end{pmatrix}
\begin{pmatrix}5\\11\end{pmatrix}
=\begin{pmatrix}5\\1\end{pmatrix}=\mathbf c.
\]</div>
<p>整个过程就是 \(A\mathbf x=\mathbf b\) 变成 \(U\mathbf x=\mathbf c\)。只改变 \(A\) 而不改变 \(\mathbf b\)，通常会得到另一个问题。</p>

<h2 id="inverse">三、什么是逆矩阵，怎样得到它？</h2>
<h3 id="inverse-definition">1. 逆矩阵负责撤销原来的运算</h3>
<p>数乘 2 可以用乘 \(1/2\) 撤销，因为 \((1/2)\cdot2=1\)。矩阵也有类似概念，但这里用<strong>单位矩阵</strong>代替数字 1：</p>
<div class="equation">\[
I=\begin{pmatrix}1&amp;0\\0&amp;1\end{pmatrix},
\qquad I\mathbf x=\mathbf x.
\]</div>
<p>对于方阵 \(A\)，如果存在矩阵 \(B\)，满足 \(BA=AB=I\)，就称 \(A\) <strong>可逆</strong>，并把 \(B\) 记为 \(A^{-1}\)。它能把 \(A\) 的输出还原为输入：</p>
<div class="equation">\[
\begin{aligned}
\mathbf b&amp;=A\mathbf x,\\
\mathbf x&amp;=A^{-1}\mathbf b.
\end{aligned}
\]</div>
<p>上标 \(-1\) 表示逆运算，<strong>不是把矩阵里的每个数取倒数</strong>。逆矩阵也不一定存在：若不同输入经 \(A\) 得到相同输出，就无法唯一还原。</p>
<h3 id="inverse-elimination">2. 消元矩阵的逆，从反向操作得到</h3>
<p>前面的 \(E\) 将任意列向量变成：</p>
<div class="equation">\[
E\begin{pmatrix}u\\v\end{pmatrix}
=\begin{pmatrix}u\\v-2u\end{pmatrix}.
\]</div>
<p>要恢复 \(v\)，只需把第一坐标的两倍加回第二坐标。这给出了逆矩阵，而不只是猜一个公式：</p>
<div class="equation">\[
\begin{aligned}
E^{-1}&amp;=\begin{pmatrix}1&amp;0\\2&amp;1\end{pmatrix},\\[8pt]
E^{-1}\begin{pmatrix}u\\v-2u\end{pmatrix}
&amp;=\begin{pmatrix}u\\2u+(v-2u)\end{pmatrix}
=\begin{pmatrix}u\\v\end{pmatrix}.
\end{aligned}
\]</div>
<p>直接相乘可检查 \(E^{-1}E=EE^{-1}=I\)。所以，原方程左乘 \(E\) 得到新方程，新方程左乘 \(E^{-1}\) 又恢复原方程：</p>
<div class="equation">\[
A\mathbf x=\mathbf b
\quad\Longleftrightarrow\quad
EA\mathbf x=E\mathbf b.
\]</div>
<p><strong>能撤销每一步，才保证解集不变。</strong>\(E^{-1}\) 只撤销这一次消元；\(A^{-1}\) 要撤销整个 \(A\) 的运算，二者不同。</p>
<h3 id="inverse-original">3. 原矩阵的逆，可以通过解方程求出来</h3>
<p>仍取前面的矩阵 \(A\)，把右端换成任意的 \((p,q)\)。同样消元、回代：</p>
<div class="equation">\[
\begin{cases}2x+y=p,\\4x+3y=q.\end{cases}
\]</div>
<div class="equation">\[
\begin{aligned}
y&amp;=q-2p,\\
x&amp;=\frac{p-y}{2}=\frac{3p-q}{2}.
\end{aligned}
\]</div>
<p>把“从输出 \((p,q)\) 求回输入”的公式写成矩阵：</p>
<div class="equation">\[
\begin{aligned}
A^{-1}&amp;=\begin{pmatrix}\frac32&amp;-\frac12\\-2&amp;1\end{pmatrix},\\[8pt]
\begin{pmatrix}x\\y\end{pmatrix}
&amp;=A^{-1}\begin{pmatrix}p\\q\end{pmatrix}.
\end{aligned}
\]</div>
<p>这就是 \(A^{-1}\) 的来源。代入 \((p,q)=(5,11)\)，得到 \((x,y)=(2,1)\)。<strong>求逆是在一次推导中求出所有右端的解法；只解某一个方程组，直接消元即可。</strong></p>

<h2 id="pivots">四、主元位置为零：先看能不能换行</h2>
<h3 id="pivot-swap">1. 下方有非零数：交换方程的顺序</h3>
<p>例如：</p>
<div class="equation">\[
\begin{cases}0x+y=1,\\2x+3y=5.\end{cases}
\]</div>
<p>左上角的 0 不能作为除数，但第二行同一列有 2。把两条方程交换，条件本身没有改变：</p>
<div class="equation">\[
\left[\begin{array}{cc|c}0&amp;1&amp;1\\2&amp;3&amp;5\end{array}\right]
\xrightarrow{\ R_1\leftrightarrow R_2\ }
\left[\begin{array}{cc|c}2&amp;3&amp;5\\0&amp;1&amp;1\end{array}\right].
\]</div>
<p>现在两个主元是 2、1，回代得到 \(y=1,x=1\)。因此，<strong>某个位置是零，不等于矩阵不可逆。</strong></p>
<p>换行也能写成左乘矩阵：\(P\) 的第一行 \((0,1)\) 取原第二行，第二行 \((1,0)\) 取原第一行：</p>
<div class="equation">\[
\begin{aligned}
P&amp;=\begin{pmatrix}0&amp;1\\1&amp;0\end{pmatrix},\\[8pt]
P\begin{pmatrix}u\\v\end{pmatrix}&amp;=\begin{pmatrix}v\\u\end{pmatrix}.
\end{aligned}
\]</div>
<p>交换两次恢复原状，即 \(P^2=I\)，所以<strong>这个交换两行的矩阵</strong>满足 \(P^{-1}=P\)。系数和常数必须整行一起交换。</p>
<h3 id="pivot-zero-row">2. 消元后整行系数为零：检查常数项</h3>
<p>把例子第二行的系数改为第一行的两倍：</p>
<div class="equation">\[
\begin{cases}2x+y=5,\\4x+2y=d.\end{cases}
\]</div>
<p>执行 \(R_2\leftarrow R_2-2R_1\)，得到：</p>
<div class="equation">\[
\begin{cases}2x+y=5,\\0x+0y=d-10.\end{cases}
\]</div>
<p>最后一行不再能确定一个未知数，要看右端：</p>
<ul>
<li>\(d=11\)：得到 \(0=1\)，矛盾，因此<strong>无解</strong>。</li>
<li>\(d=10\)：得到 \(0=0\)，没有新限制。只剩 \(2x+y=5\)，可以任取 \(y=t\)，再令 \(x=(5-t)/2\)，因此<strong>有无穷多个解</strong>。这个可以任取的 \(y\) 叫作自由变量。</li>
</ul>
<p>两种情况的系数矩阵相同：</p>
<div class="equation">\[
C=\begin{pmatrix}2&amp;1\\4&amp;2\end{pmatrix}.
\]</div>
<p>它只能输出 \((2x+y,\,2(2x+y))\)，无法区分所有输入，所以不可逆。<strong>不可逆描述矩阵本身；无解还是无穷多解，还取决于右端。</strong></p>

<h2 id="order">五、多步操作：后做的矩阵放左边</h2>
<p>先用 \(E_1\) 消元，再用 \(E_2\)，便有：</p>
<div class="equation">\[
\begin{aligned}
A&amp;\xrightarrow{E_1}E_1A\xrightarrow{E_2}E_2E_1A=U,\\[6pt]
\mathbf c&amp;=E_2E_1\mathbf b.
\end{aligned}
\]</div>
<p>撤销时先撤销最后一步，所以：</p>
<div class="equation">\[
(E_2E_1)^{-1}=E_1^{-1}E_2^{-1}.
\]</div>
<p>乘法按右边先作用来读。这个顺序以后会用于理解 \(A=LU\) 分解。</p>

<h2 id="self-check">理解检查</h2>
<p>消元倍数为什么是“待消系数 ÷ 主元”？为什么必须同时操作常数列？为什么零主元不一定意味着矩阵不可逆？能解释这三点，就抓住了本节的基础。</p>
<h2 id="sources">资料</h2>
<p>依据 MIT OCW 18.06SC 第二讲 <em>Elimination with Matrices</em> 及其讲义摘要整理，例子沿用我们的讨论。<a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/">课程来源</a>；笔记按 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 分享。</p>
<p class="note-footer"><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-01/">上一节：线性方程的几何</a></p>
</section>
