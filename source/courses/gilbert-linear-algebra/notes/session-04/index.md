---
title: 第四节：A = LU 分解｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/session-04/
mathjax: true
toc_number: false
comments: false
---

<style>
.post-block:has(.la-notes--session04) .post-header { margin-bottom: 24px; }
.post-block:has(.la-notes--session04) .post-title { font-size: 26px; line-height: 1.45; color: #253c3c; }
.post-block:has(.la-notes--session04) .breadcrumb { display: none; }
.post-body .la-notes--session04 { max-width: 660px; margin: 0 auto; font-size: 17px; line-height: 1.85; color: #303737; }
.post-body .la-notes--session04 p { margin: 0 0 .9em; text-indent: 0; }
.post-body .la-notes--session04 h2 { margin: 2.4em 0 1em; padding: 0 0 .45em; border-bottom: 1px solid #dce5e3; color: #285d5d; font-size: 22px; line-height: 1.5; }
.post-body .la-notes--session04 h3 { margin: 1.7em 0 .75em; color: #303737; font-size: 18px; line-height: 1.6; }
.post-body .la-notes--session04 [id] { scroll-margin-top: 24px; }
.post-body .la-notes--session04 .note-nav { font-size: 14px; color: #687474; margin-bottom: 18px; }
.post-body .la-notes--session04 .note-nav a { border-color: #ccd8d5; }
.post-body .la-notes--session04 .note-lead { margin: 0 0 1.4em; padding: .1em 0 .1em 14px; border-left: 3px solid #729692; }
.post-body .la-notes--session04 .note-lead p { margin: 0; }
.post-body .la-notes--session04 .reading-links { padding: 14px 0; margin-bottom: 1.6em; border-top: 1px solid #e2e8e6; border-bottom: 1px solid #e2e8e6; font-size: 14px; line-height: 1.8; }
.post-body .la-notes--session04 .reading-links p { margin: 0; }
.post-body .la-notes--session04 .reading-links p + p { margin-top: 8px; }
.post-body .la-notes--session04 .link-row { display: flex; flex-wrap: wrap; column-gap: 18px; row-gap: 6px; }
.post-body .la-notes--session04 .equation { margin: 1em 0 1.15em; padding: 8px 2px; overflow-x: auto; overflow-y: hidden; }
.post-body .la-notes--session04 .MathJax_Display { font-size: 100%; margin: .25em 0 !important; padding: 3px 0; }
.post-body .la-notes--session04 ul { margin: .65em 0 1.2em; padding-left: 1.35em; }
.post-body .la-notes--session04 li { margin-bottom: .55em; }
.post-body .la-notes--session04 .note-footer { margin-top: 1.8em; padding-top: 1em; border-top: 1px solid #e2e8e6; font-size: 14px; }
@media (max-width: 600px) {
  .post-block:has(.la-notes--session04) .post-title { font-size: 23px; }
  .post-body .la-notes--session04 { font-size: 16px; line-height: 1.85; }
  .post-body .la-notes--session04 h2 { font-size: 20px; margin-top: 2em; }
  .post-body .la-notes--session04 h3 { font-size: 17px; }
}
</style>

<section class="la-notes la-notes--session04">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-03/">第三节</a></p>
<div class="note-lead"><p><strong>消元得到上三角矩阵；把消元步骤倒过来，就得到原矩阵的分解。</strong>先理解恢复过程，再引入符号。</p></div>
<div class="reading-links">
<p class="link-row"><a href="/files/mit-18-06sc/unit-1-session-4-summary.pdf">讲义摘要</a> <a href="/files/mit-18-06sc/unit-1-session-4-problems.pdf">习题</a> <a href="/files/mit-18-06sc/unit-1-session-4-solutions.pdf">参考解答</a></p>
<p class="link-row"><a href="#restore">从消元到分解</a> <a href="#multipliers">多步消元</a> <a href="#solve">用分解解方程</a> <a href="#pivoting">需要换行时</a> <a href="#extensions">讲义补充</a></p>
</div>

<h2 id="restore">一、从第二节接着问：怎样恢复原矩阵？</h2>
<p>第二节中，第二行减去第一行的两倍，把原矩阵变成上三角矩阵：</p>
<div class="equation">\[
A=\begin{pmatrix}2&amp;1\\4&amp;3\end{pmatrix}
\quad\xrightarrow{\ R_2\leftarrow R_2-2R_1\ }\quad
U=\begin{pmatrix}2&amp;1\\0&amp;1\end{pmatrix}.
\]</div>
<p>要恢复原矩阵，把两倍第一行加回第二行即可。按前两节的行组合规则：</p>
<div class="equation">\[
\underbrace{\begin{pmatrix}2&amp;1\\4&amp;3\end{pmatrix}}_{\text{原矩阵}}
=
\underbrace{\begin{pmatrix}1&amp;0\\2&amp;1\end{pmatrix}}_{\text{恢复操作}}
\underbrace{\begin{pmatrix}2&amp;1\\0&amp;1\end{pmatrix}}_{\text{消元结果}}.
\]</div>
<p>中间矩阵的第二行 \((2,1)\) 表示“取两倍 \(U\) 的第一行，加上一倍第二行”：\(2(2,1)+(0,1)=(4,3)\)。它的对角线上方全为零，叫作<strong>下三角矩阵</strong>，记为 \(L\)（lower triangular）；\(U\) 的名称来自 upper triangular。上面的等式便写成：</p>
<div class="equation">\[
\boxed{A=LU}.
\]</div>
<p>这与第三节的逆矩阵直接相接：第二节写成 \(EA=U\)，撤销消元得到 \(A=E^{-1}U\)，所以本例 \(L=E^{-1}\)。<strong>\(L\) 记录恢复操作，\(U\) 保留消元结果。</strong></p>

<h2 id="multipliers">二、多步消元：倍数为什么直接进入 L？</h2>
<p>把刚才的例子扩为三阶，下面重新用 \(A\) 表示这个三阶矩阵：</p>
<div class="equation">\[
\begin{aligned}
A&amp;=\begin{pmatrix}2&amp;1&amp;1\\4&amp;3&amp;3\\0&amp;5&amp;7\end{pmatrix}\\[6pt]
\xrightarrow{\ R_2\leftarrow R_2-2R_1\ }&amp;
\begin{pmatrix}2&amp;1&amp;1\\0&amp;1&amp;1\\0&amp;5&amp;7\end{pmatrix}\\[6pt]
\xrightarrow{\ R_3\leftarrow R_3-5R_2\ }&amp;
\begin{pmatrix}2&amp;1&amp;1\\0&amp;1&amp;1\\0&amp;0&amp;2\end{pmatrix}=U.
\end{aligned}
\]</div>
<p>第二步使用<strong>更新后的第二行</strong>。消元倍数依次是 \(4/2=2\)、\(5/1=5\)；第三行第一列本来为零，不需要消。</p>
<p>恢复时顺序反过来：先把五倍 \(U\) 的第二行加回第三行，得到 \((0,5,7)\)；再把两倍第一行加回第二行，得到 \((4,3,3)\)。因此，原矩阵的行可以用 \(U\) 的行表示：</p>
<div class="equation">\[
\begin{aligned}
A\text{ 的第 1 行}&amp;=U\text{ 的第 1 行},\\
A\text{ 的第 2 行}&amp;=2\,(U\text{ 的第 1 行})+U\text{ 的第 2 行},\\
A\text{ 的第 3 行}&amp;=5\,(U\text{ 的第 2 行})+U\text{ 的第 3 行}.
\end{aligned}
\]</div>
<p>把三组组合系数排成行，就得到：</p>
<div class="equation">\[
L=\begin{pmatrix}1&amp;0&amp;0\\2&amp;1&amp;0\\0&amp;5&amp;1\end{pmatrix},
\qquad A=LU.
\]</div>
<p>\(L_{21}=2\) 对应“第二行减去第一行的两倍”，\(L_{32}=5\) 对应“第三行减去第二行的五倍”。<strong>记录的是被减去的倍数，所以这里存 \(2,5\)，消元矩阵里才是 \(-2,-5\)。</strong>倍数本身也可能为负；对角线的 1 表示保留当前行的一份。</p>
<p>把两次消元矩阵依次记为 \(E_1,E_2\)。第三节已经说明，后做的操作放在左边，撤销时顺序反过来：</p>
<div class="equation">\[
E_2E_1A=U
\quad\Longrightarrow\quad
A=\underbrace{E_1^{-1}E_2^{-1}}_{L}U.
\]</div>
<p>右边先作用 \(E_2^{-1}\)，即先恢复第三行。若先恢复第二行，再用它更新第三行，就会额外加进 \(5\cdot2=10\) 倍第一行，无法恢复原矩阵。</p>
<p>一般地，按列向下消元，且<strong>不换行、不缩放主元行</strong>时，\(L\) 的对角线为 1，\(L_{ij}\) 记录操作 \(R_i\leftarrow R_i-\ell_{ij}R_j\) 中的倍数 \(\ell_{ij}\)。倒序恢复时，主元行尚未撤销更早的消元，因此可以直接记录这些倍数。由于总是用前面的行更新后面的行，记录都在对角线下方。</p>

<h2 id="solve">三、为什么分解后更容易解方程？</h2>
<p>原方程成为 \(L(U\mathbf x)=\mathbf b\)。我们已经认识 \(U\mathbf x\)，但还不知道它的值；现在把这个<strong>中间向量</strong>记为 \(\mathbf y\)，分两步求：</p>
<div class="equation">\[
L\mathbf y=\mathbf b,\qquad U\mathbf x=\mathbf y.
\]</div>
<p>\(L\) 是下三角，从第一行向下逐个求坐标，叫作<strong>前代</strong>；\(U\) 是上三角，从最后一行向上<strong>回代</strong>。\(\mathbf y\) 是辅助量，原未知数仍是 \(\mathbf x\)。</p>
<p>用刚才的三阶分解，取 \(\mathbf b=(7,19,31)^{\mathsf T}\)。先解：</p>
<div class="equation">\[
\begin{cases}y_1=7,\\2y_1+y_2=19,\\5y_2+y_3=31\end{cases}
\quad\Longrightarrow\quad
\mathbf y=\begin{pmatrix}7\\5\\6\end{pmatrix}.
\]</div>
<p>再解：</p>
<div class="equation">\[
\begin{cases}2x_1+x_2+x_3=7,\\x_2+x_3=5,\\2x_3=6\end{cases}
\quad\Longrightarrow\quad
\mathbf x=\begin{pmatrix}1\\2\\3\end{pmatrix}.
\]</div>
<p>\(\mathbf y\) 正是消元后的右端：\(L^{-1}\mathbf b=E_2E_1\mathbf b\)。所以这两步仍是第二节的“同时变换系数和常数，再回代”。</p>
<p><strong>同一个 \(A\) 对应许多个不同的 \(\mathbf b\) 时，分解只需做一次。</strong>对于一般稠密的 \(n\times n\) 矩阵，分解的运算量约随 \(n^3\) 增长；之后每个右端的前代、回代合计约随 \(n^2\) 增长。这样不必对每个右端重新消元，也不必显式求逆。</p>

<h2 id="pivoting">四、需要换行时：PA = LU</h2>
<p>下面用 \(A\) 表示第二节见过的矩阵 \(\begin{pmatrix}0&amp;1\\2&amp;3\end{pmatrix}\)。它可逆，但左上角的 0 不能作主元。先用交换矩阵 \(P\) 换行：</p>
<div class="equation">\[
P=\begin{pmatrix}0&amp;1\\1&amp;0\end{pmatrix},\qquad
PA=\begin{pmatrix}2&amp;3\\0&amp;1\end{pmatrix}.
\]</div>
<p>换行后分解，写成 \(PA=LU\)；本例 \(L=I\)，\(U=PA\)。\(P\) 记录总的换行次序。求解时常数也要同步换行：</p>
<div class="equation">\[
L\mathbf y=P\mathbf b,\qquad U\mathbf x=\mathbf y.
\]</div>
<p>若在消元中途换行，构造 \(L\) 时还要交换相关两行中<strong>先前已记录的倍数</strong>，才能保持等式成立。前面“倍数直接填入 \(L\)”的简单规则，前提是不换行。</p>
<p>本节的求解过程假设 \(A\) 可逆，从而有足够的非零主元。不可逆矩阵也可能有 \(LU\) 分解，但 \(U\) 会缺少主元，不能保证任意右端都有唯一解。</p>

<h2 id="extensions">五、讲义中的两个补充</h2>
<p><strong>把主元单独保存。</strong>将刚才 \(U\) 的主元排成对角矩阵 \(D\)，再把 \(U\) 的各行除以对应主元，得到对角线为 1 的 \(\widehat U\)：</p>
<div class="equation">\[
D=\begin{pmatrix}2&amp;0&amp;0\\0&amp;1&amp;0\\0&amp;0&amp;2\end{pmatrix},\quad
\widehat U=\begin{pmatrix}1&amp;\frac12&amp;\frac12\\0&amp;1&amp;1\\0&amp;0&amp;1\end{pmatrix}.
\]</div>
<p>于是 \(U=D\widehat U\)，分解也可写成 \(A=LD\widehat U\)。</p>
<p><strong>转置也反转乘积顺序。</strong>\(A^{\mathsf T}\) 表示交换行与列，即 \((A^{\mathsf T})_{ij}=A_{ji}\)。把“左边的行乘右边的列”转置后，成为“右边转置后的行乘左边转置后的列”，因此：</p>
<div class="equation">\[
(AB)^{\mathsf T}=B^{\mathsf T}A^{\mathsf T},\qquad
(A^{\mathsf T})^{-1}=(A^{-1})^{\mathsf T}.
\]</div>
<p>第二式要求 \(A\) 可逆；把 \(AA^{-1}=A^{-1}A=I\) 转置即可验证。对换行矩阵，\(P^{-1}=P^{\mathsf T}\)：转置后的排列恰好把各行放回原位。</p>

<h2 id="self-check">理解检查</h2>
<p>为什么 \(L\) 里存 \(2,5\)，而消元矩阵里是 \(-2,-5\)？三阶例子为什么必须先恢复第三行？\(\mathbf y\) 与第二节消元后的右端有什么关系？</p>
<h2 id="sources">资料</h2>
<p>依据 MIT OCW 18.06SC 第四讲 <em>Factorization into A = LU</em> 及其讲义摘要整理，例子从第二节延伸。<a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/">课程来源</a>；笔记按 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 分享。</p>
<p class="note-footer"><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-03/">上一节：矩阵乘法与逆矩阵</a></p>
</section>
