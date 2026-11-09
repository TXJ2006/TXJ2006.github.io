---
title: 第二节：矩阵消元｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/session-02/
mathjax: true
toc_number: false
comments: false
---

<style>
.la-notes { line-height: 1.95; }
.la-notes h2 { margin-top: 2em; }
.la-notes .equation { overflow-x: auto; padding: .4em 0; }
.la-notes .reading-links { padding: .8em 1em; border-left: 3px solid #496987; background: #f7f9fa; }
</style>

<section class="la-notes">
<p><a href="/courses/gilbert-linear-algebra/notes/">← 返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-01/">第一节</a></p>
<h1 id="session-02">第二节：矩阵消元</h1>
<p><strong>用可逆的行操作，把方程组化成上三角形式，再回代求解。</strong>消元改变方程的表达，保持未知数的含义与整个解集不变。</p>
<div class="reading-links">
<p><a href="/files/mit-18-06sc/unit-1-session-2-summary.pdf">讲义摘要</a> · <a href="/files/mit-18-06sc/unit-1-session-2-problems.pdf">习题</a> · <a href="/files/mit-18-06sc/unit-1-session-2-solutions.pdf">参考解答</a></p>
<p><a href="#elimination">消元与回代</a> · <a href="#left-multiplication">消元矩阵</a> · <a href="#inverse">为什么解不变</a> · <a href="#pivots">主元与换行</a> · <a href="#order">多步操作的顺序</a></p>
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
\underbrace{\begin{pmatrix}1&amp;0\\-2&amp;1\end{pmatrix}}_E
\underbrace{\begin{pmatrix}2&amp;1\\4&amp;3\end{pmatrix}}_A
=\underbrace{\begin{pmatrix}2&amp;1\\0&amp;1\end{pmatrix}}_U.
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

<h2 id="inverse">三、为什么消元不改变解？</h2>
<p>“第二行减去第一行的两倍”可以用“第二行加回第一行的两倍”撤销：</p>
<div class="equation">\[
E^{-1}=\begin{pmatrix}1&amp;0\\2&amp;1\end{pmatrix},
\qquad
E^{-1}E=EE^{-1}=\underbrace{\begin{pmatrix}1&amp;0\\0&amp;1\end{pmatrix}}_I.
\]</div>
<p>\(I\) 是单位矩阵，乘上它不会改变向量。若原方程成立，左乘 \(E\) 就得到新方程；若新方程成立，左乘 \(E^{-1}\) 就恢复原方程：</p>
<div class="equation">\[
A\mathbf x=\mathbf b
\quad\Longleftrightarrow\quad
EA\mathbf x=E\mathbf b.
\]</div>
<p><strong>两边都成立的双向关系，才说明解集完全相同。</strong>\(E^{-1}\) 撤销一次行操作，\(A^{-1}\) 则要撤销整个映射 \(\mathbf x\mapsto A\mathbf x\)。消元矩阵可逆，不代表原矩阵一定可逆。</p>

<h2 id="pivots">四、主元为零怎么办？</h2>
<p>不能除以零。当前主元位置是零时，若同一列下方有非零数，就交换两行。例如：</p>
<div class="equation">\[
\underbrace{\begin{pmatrix}0&amp;1\\1&amp;0\end{pmatrix}}_P
\begin{pmatrix}0&amp;1\\2&amp;3\end{pmatrix}
=\begin{pmatrix}2&amp;3\\0&amp;1\end{pmatrix}.
\]</div>
<p>置换矩阵 \(P\) 交换两行；\(\mathbf b\) 也要交换对应分量。再交换一次就恢复原状，所以 \(P^{-1}=P\)。<strong>碰到零主元不一定失败，可能只需换行。</strong></p>
<p>若处理方阵时，当前列从主元位置向下全为零，就无法得到每个对角位置都非零的三角形式，原矩阵不可逆。仍可继续整理方程：出现 \(0=d\)、\(d\ne0\) 时无解；若条件相容且有自由变量，就有无穷多个解。例如同一系数行 \((0,0)\)，右端为 1 表示矛盾，右端为 0 则只表示一条重复条件。</p>

<h2 id="order">五、多步操作：后做的矩阵放左边</h2>
<p>先用 \(E_1\) 消元，再用 \(E_2\)，便有：</p>
<div class="equation">\[
A\xrightarrow{E_1}E_1A\xrightarrow{E_2}E_2E_1A=U,
\qquad \mathbf c=E_2E_1\mathbf b.
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
<p><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-01/">上一节：线性方程的几何</a></p>
</section>
