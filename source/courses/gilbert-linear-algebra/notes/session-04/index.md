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
<p class="link-row"><a href="#restore">从消元到分解</a> <a href="#multipliers">倍数如何进入 L</a> <a href="#exercise-41">习题 4.1：用行向量读出 L 与 E</a> <a href="#solve">用分解解方程</a> <a href="#cost">运算量与存储</a> <a href="#pivoting">换行与置换</a> <a href="#extensions">主元与转置</a></p>
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
<p>这个例子还说明：<strong>\(L\) 不是把合并后的消元矩阵逐项变号。</strong>直接相乘得到：</p>
<div class="equation">\[
E_2E_1=
\begin{pmatrix}1&amp;0&amp;0\\-2&amp;1&amp;0\\10&amp;-5&amp;1\end{pmatrix},
\qquad
(E_2E_1)^{-1}=
\begin{pmatrix}1&amp;0&amp;0\\2&amp;1&amp;0\\0&amp;5&amp;1\end{pmatrix}=L.
\]</div>
<p>左下角的 \(10\) 来自连续减行产生的交叉项；取逆会改变操作顺序，不能只改变符号。</p>
<p>为什么一般情况下也能直接记录倍数？固定原矩阵的第 \(i\) 行。按列消元时，它依次减去第 1、2、……、\(i-1\) 个主元行的若干倍。<strong>第 \(j\) 个主元行一旦用来消下面的行，之后就不再被修改，因此它正是最终 \(U\) 的第 \(j\) 行。</strong>记原矩阵第 \(i\) 行为 \(\mathbf r_i\)，\(U\) 的第 \(j\) 行为 \(\mathbf u_j\)，实际减去的倍数为 \(\ell_{ij}\)，则消元的全过程就是：</p>
<div class="equation">\[
\mathbf u_i=\mathbf r_i-\sum_{j=1}^{i-1}\ell_{ij}\mathbf u_j
\quad\Longrightarrow\quad
\mathbf r_i=\sum_{j=1}^{i-1}\ell_{ij}\mathbf u_j+\mathbf u_i.
\]</div>
<p>右边恰好是矩阵乘法 \(LU\) 的第 \(i\) 行。这就同时推出：\(L_{ij}=\ell_{ij}\)（\(j&lt;i\)），对角线为 1，上方为零。<strong>下三角形状与每个数字的位置，都由实际做过的行操作决定。</strong>这里假定按列向下消元、不换行、不缩放主元行；换行的情况见后文。</p>

<h3 id="exercise-41">习题 4.1：用行向量读出 L 与 E</h3>
<p>题目给出下面的 \(A\)，要求求出 \(EA=U\) 中的 \(E\)，再得到 \(A=LU\)。这里的矩阵与前面的示例不同：</p>
<div class="equation">\[
A=\begin{pmatrix}1&amp;3&amp;0\\2&amp;4&amp;0\\2&amp;0&amp;1\end{pmatrix}.
\]</div>
<p>把<strong>原矩阵的三行</strong>记为 \(\mathbf r_1=(1,3,0)\)、\(\mathbf r_2=(2,4,0)\)、\(\mathbf r_3=(2,0,1)\)。这些名字始终指原行，不随消元变化；把<strong>最终 \(U\) 的三行</strong>记为 \(\mathbf u_1,\mathbf u_2,\mathbf u_3\)。它们都是整行向量，不是单个元素。</p>
<p>第一行不变，所以 \(\mathbf u_1=\mathbf r_1\)。第二行减去两倍第一行，得到 \(\mathbf u_2=\mathbf r_2-2\mathbf u_1=(0,-2,0)\)。第三行先减去两倍第一行，成为 \((0,-6,1)\)；再用<strong>当前第二行 \(\mathbf u_2\)</strong>消去 \(-6\)，倍数为 \((-6)/(-2)=3\)。因此：</p>
<div class="equation">\[
\begin{aligned}
\mathbf u_1&amp;=\mathbf r_1=(1,3,0),\\
\mathbf u_2&amp;=\mathbf r_2-2\mathbf u_1=(0,-2,0),\\
\mathbf u_3&amp;=\mathbf r_3-2\mathbf u_1-3\mathbf u_2=(0,0,1).
\end{aligned}
\]</div>
<p>将这三行排起来就是 \(U\)。要读出 \(L\)，把等式移项，<strong>用 \(U\) 的行表示原矩阵的行</strong>：</p>
<div class="equation">\[
\begin{aligned}
\mathbf r_1&amp;=1\mathbf u_1+0\mathbf u_2+0\mathbf u_3,\\
\mathbf r_2&amp;=2\mathbf u_1+1\mathbf u_2+0\mathbf u_3,\\
\mathbf r_3&amp;=2\mathbf u_1+3\mathbf u_2+1\mathbf u_3.
\end{aligned}
\]</div>
<p>矩阵左乘正是组合右边矩阵的行，所以每条等式中的三个系数，就是 \(L\) 的对应行：</p>
<div class="equation">\[
L=\begin{pmatrix}1&amp;0&amp;0\\2&amp;1&amp;0\\2&amp;3&amp;1\end{pmatrix},
\qquad
U=\begin{pmatrix}1&amp;3&amp;0\\0&amp;-2&amp;0\\0&amp;0&amp;1\end{pmatrix}.
\]</div>
<p>例如第三行 \((2,3,1)\) 表示取 \(2\mathbf u_1+3\mathbf u_2+\mathbf u_3\)，实际结果为 \((2,6,0)+(0,-6,0)+(0,0,1)=(2,0,1)=\mathbf r_3\)。这说明了第三行每一个数的含义，也直接验证了 \(A=LU\)。</p>
<p>题目中的 \(E\) 方向相反：它要<strong>用原矩阵的行表示 \(U\) 的行</strong>。把刚才的 \(\mathbf u_1=\mathbf r_1\)、\(\mathbf u_2=\mathbf r_2-2\mathbf r_1\) 代入第三行：</p>
<div class="equation">\[
\begin{aligned}
\mathbf u_1&amp;=\mathbf r_1,\\
\mathbf u_2&amp;=-2\mathbf r_1+\mathbf r_2,\\
\mathbf u_3&amp;=\mathbf r_3-2\mathbf r_1-3(\mathbf r_2-2\mathbf r_1)\\
&amp;=4\mathbf r_1-3\mathbf r_2+\mathbf r_3.
\end{aligned}
\]</div>
<p>再次按行读取组合系数，得到：</p>
<div class="equation">\[
E=\begin{pmatrix}1&amp;0&amp;0\\-2&amp;1&amp;0\\4&amp;-3&amp;1\end{pmatrix},
\qquad EA=U,\quad E^{-1}=L.
\]</div>
<p><strong>\(L\) 组合的是 \(\mathbf u_1,\mathbf u_2,\mathbf u_3\)，\(E\) 组合的是 \(\mathbf r_1,\mathbf r_2,\mathbf r_3\)。</strong>两者使用的行不同，所以 \(E\) 的左下角是 \(4=-2+3\cdot2\)，而 \(L\) 的左下角是 \(2\)；不能逐项变号来求逆。</p>

<h2 id="solve">三、为什么分解后更容易解方程？</h2>
<p>原方程成为 \(L(U\mathbf x)=\mathbf b\)。已知最终输出 \(\mathbf b\)，先求“经过 \(L\) 后会得到 \(\mathbf b\) 的向量”；这个向量就是 \(U\mathbf x\)，现在才给它一个名字 \(\mathbf y\)。所以分成：</p>
<div class="equation">\[
L\mathbf y=\mathbf b,\qquad U\mathbf x=\mathbf y.
\]</div>
<p>\(L\) 是下三角，第 \(i\) 行只涉及 \(y_1,\ldots,y_i\)，而对角线为 1。因此，前面的坐标求出后，就有 \(y_i=b_i-\sum_{j&lt;i}L_{ij}y_j\)。从上往下逐个求解叫作<strong>前代</strong>。同理，\(U\) 的最后一行先确定 \(x_n\)，随后向上<strong>回代</strong>；每一步需要除以对应的非零主元。\(\mathbf y\) 是辅助量，原未知数仍是 \(\mathbf x\)。</p>
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
<p><strong>同一个 \(A\) 对应许多个不同的 \(\mathbf b\) 时，分解只需做一次。</strong>因为消元倍数只由系数矩阵决定，改变右端不会改变 \(L,U\)。究竟节省多少计算，需要从每一步数起。</p>

<h2 id="cost">四、消元究竟需要多少计算与存储？</h2>
<p>考虑一般稠密的 \(n\times n\) 矩阵。第 \(k\) 列消元时，前 \(k-1\) 列已经处理完。主元下有 \(n-k\) 行，每行需要更新右侧 \(n-k\) 个数。下面的 \(a_{ij}\) 表示<strong>当前工作矩阵</strong>中的数，每个数都按同一规则更新：</p>
<div class="equation">\[
\ell_{ik}=\frac{a_{ik}}{a_{kk}},\qquad
a_{ij}\leftarrow a_{ij}-\ell_{ik}a_{kj}
\quad(i,j&gt;k).
\]</div>
<p>除法求出“这一行要减去主元行的多少倍”；后一个式子就是行操作在单个坐标上的写法。共有 \((n-k)^2\) 次这样的<strong>乘后减更新</strong>，所以全部消元的更新次数为：</p>
<div class="equation">\[
\begin{aligned}
\sum_{k=1}^{n-1}(n-k)^2
&amp;=1^2+2^2+\cdots+(n-1)^2\\
&amp;=\frac{n(n-1)(2n-1)}6
=\frac13n^3-\frac12n^2+\frac16n.
\end{aligned}
\]</div>
<p>讲义用 \(\sum_{i=1}^{n}i^2\approx\int_0^n x^2\,dx=\frac13n^3\) 估计同一个主导项。端点差异只影响较低次项。<strong>三次方来自：逐列消元，每列要更新一个边长逐渐缩短的方块。</strong>矩阵阶数翻倍，主导计算量约为原来的八倍。</p>
<p>计数单位要说清：讲义把“乘后减”作为一组来估计；若一次乘法、一次减法各算一次浮点运算，主导项就是 \(\frac23n^3\)。求消元倍数还需要 \(n(n-1)/2\) 次除法，属于二次项。两种计数都说明同一个事实：一般稠密矩阵的消元是三次量级。</p>
<p>右端为何只需二次量级？同一次行操作只需更新一个常数 \(b_i\leftarrow b_i-\ell_{ik}b_k\)，不是一整行系数。因此全部右端更新只有：</p>
<div class="equation">\[
\sum_{k=1}^{n-1}(n-k)=\frac{n(n-1)}2.
\]</div>
<p>用已经保存的 \(L\) 做前代，也是这些更新；回代再做约同样多次乘后减和 \(n\) 次除法。因此，每个新右端只增加二次量级工作。若有 \(r\) 个右端，重复消元的主要成本约为 \(r\) 次三次量级工作；保存分解后，则是<strong>一次三次量级工作，加 \(r\) 次二次量级工作</strong>。这里比较的是运算次数，实际耗时还受存储访问、硬件和矩阵结构影响。</p>
<p><strong>存储也不需要把所有消元矩阵留下来。</strong>主元下方的位置被消成零后，后续步骤不再需要这些系数；可以用这些位置保存刚算出的倍数。对角线和上方保存 \(U\)，下方保存 \(L\) 的倍数，而 \(L\) 的对角线 1 无须存储。刚才的三阶例子可以在一个数组中保存为：</p>
<div class="equation">\[
\begin{pmatrix}
2&amp;1&amp;1\\
\boxed{2}&amp;1&amp;1\\
\boxed{0}&amp;\boxed{5}&amp;2
\end{pmatrix}.
\]</div>
<p>框内属于 \(L\)，其余属于 \(U\)；这个数组<strong>本身并不是 \(A\)、\(L\) 或 \(U\)</strong>。计算时已处理的下方位置只作记录，不再当作待消系数。这样用约 \(n^2\) 个数就能保存分解，可以覆盖原矩阵；若之后还需要原矩阵本身，才额外保留副本。换行次序通常另用一个长度为 \(n\) 的整数数组保存。</p>

<h2 id="pivoting">五、需要换行时：置换矩阵从哪里来？</h2>
<p>下面用 \(A\) 表示第二节见过的矩阵 \(\begin{pmatrix}0&amp;1\\2&amp;3\end{pmatrix}\)。它可逆，但左上角的 0 不能作主元。先用交换矩阵 \(P\) 换行：</p>
<div class="equation">\[
P=\begin{pmatrix}0&amp;1\\1&amp;0\end{pmatrix},\qquad
PA=\begin{pmatrix}2&amp;3\\0&amp;1\end{pmatrix}.
\]</div>
<p>这里 \(P\) 的第一行 \((0,1)\) 选出原第二行，第二行 \((1,0)\) 选出原第一行。一般的<strong>置换矩阵（permutation matrix）</strong>由单位矩阵重排行得到：每行每列恰有一个 1，其余为零。每行的 1 选择一条原行，每列恰有一个 1 保证既不重复选取，也不遗漏。</p>
<p>多次换行仍是一次整体重排，把它记为 \(P\)，再分解重排后的矩阵，就得到 \(PA=LU\)；本例 \(L=I\)，\(U=PA\)。求解时常数也要同步换行：</p>
<div class="equation">\[
L\mathbf y=P\mathbf b,\qquad U\mathbf x=\mathbf y.
\]</div>
<p>若在消元中途换行，已记录的倍数也必须跟着对应的行移动。例如第一列处理后，第二、三行分别记录了倍数 \(\ell_{21},\ell_{31}\)；再交换这两行时，这两个数必须一同交换，因为它们记录的是各自那条方程已经减去过什么。只交换工作矩阵而不交换这些记录，\(L\) 就无法恢复 \(PA\)。</p>
<p>本节的求解过程假设 \(A\) 可逆，从而有足够的非零主元。不可逆矩阵也可能有 \(LU\) 分解，但 \(U\) 会缺少主元，不能保证任意右端都有唯一解。</p>
<p><strong>为什么恰有 \(n!\) 个置换矩阵？</strong>第一行可选 \(n\) 条原行之一，第二行只剩 \(n-1\) 种选择，依次到最后一行只剩一种。因此总数为 \(n(n-1)\cdots1=n!\)，包含保持原顺序的单位矩阵。</p>
<p><strong>为什么它们在乘法下组成一个群？</strong>连续两次重排仍是重排，故乘法封闭；结合律来自矩阵乘法；\(I\) 是不改变顺序的单位元；每次重排都能逆向放回，故每个元素都有逆。这四条正是群的要求。这个群叫作 <em>symmetric group</em>，记为 \(S_n\)。有了这个观点，多次换行就能作为一个可逆操作 \(P\) 统一处理。</p>
<p>逆矩阵为何恰为转置？\(P\) 的各行是互不相同的标准单位向量，每行与自身点积为 1，与其他行点积为 0。按“行乘列”计算，\(PP^{\mathsf T}=I\)；同样，\(P^{\mathsf T}P=I\)，所以 \(P^{-1}=P^{\mathsf T}\)。单次交换两行还满足 \(P^{-1}=P\)，但一般重排未必如此。</p>

<h2 id="extensions">六、把主元分离，以及转置的乘法规则</h2>
<p><strong>把主元单独保存。</strong>将刚才 \(U\) 的主元排成对角矩阵 \(D\)，再把 \(U\) 的各行除以对应主元，得到对角线为 1 的 \(\widehat U\)：</p>
<div class="equation">\[
D=\begin{pmatrix}2&amp;0&amp;0\\0&amp;1&amp;0\\0&amp;0&amp;2\end{pmatrix},\quad
\widehat U=\begin{pmatrix}1&amp;\frac12&amp;\frac12\\0&amp;1&amp;1\\0&amp;0&amp;1\end{pmatrix}.
\]</div>
<p>为什么左乘 \(D\)？因为对角矩阵左乘时，第 \(i\) 行只取原第 \(i\) 行的 \(D_{ii}\) 倍，正好把除掉的主元乘回来。因此 \(U=D\widehat U\)，分解写成 \(A=LD\widehat U\)：\(L\) 保存消元倍数，\(D\) 保存主元，\(\widehat U\) 保存每行除以主元后的系数。这里需要各主元非零。</p>
<p><strong>转置也反转乘积顺序。</strong>\(A^{\mathsf T}\) 表示交换行与列，即 \((A^{\mathsf T})_{ij}=A_{ji}\)。固定转置后第 \(i,j\) 个元素，用第三节的乘法公式：</p>
<div class="equation">\[
\begin{aligned}
((AB)^{\mathsf T})_{ij}
&amp;=(AB)_{ji}=\sum_k a_{jk}b_{ki}\\
&amp;=\sum_k(B^{\mathsf T})_{ik}(A^{\mathsf T})_{kj}
=(B^{\mathsf T}A^{\mathsf T})_{ij}.
\end{aligned}
\]</div>
<p>中间只交换了两个实数因子的次序，再用乘法定义识别结果，没有交换矩阵。所以每个位置都相等，从而：</p>
<div class="equation">\[
(AB)^{\mathsf T}=B^{\mathsf T}A^{\mathsf T},\qquad
(A^{\mathsf T})^{-1}=(A^{-1})^{\mathsf T}.
\]</div>
<p>第二式要求 \(A\) 可逆：将 \(AA^{-1}=I\) 转置，得到 \((A^{-1})^{\mathsf T}A^{\mathsf T}=I\)；再将 \(A^{-1}A=I\) 转置，得到反向乘积也为 \(I\)。因此 \((A^{-1})^{\mathsf T}\) 符合 \(A^{\mathsf T}\) 的逆矩阵定义。</p>

<h2 id="self-check">理解检查</h2>
<p>能否从一行经历过的消元推导 \(L\) 的对应行？为什么系数更新要数平方和，而右端只需数一次和？一个数组怎样同时存下 \(L,U\)？为什么置换矩阵的逆一定是转置，却未必是自身？</p>
<h2 id="sources">资料</h2>
<p>依据 MIT OCW 18.06SC 第四讲 <em>Factorization into A = LU</em>、讲义摘要与习题 4.1 整理，正文示例从第二节延伸。<a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/">课程来源</a>；笔记按 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 分享。</p>
<p class="note-footer"><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-03/">上一节：矩阵乘法与逆矩阵</a></p>
</section>
