---
title: 第二单元 5：行列式的性质与消元｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-2-session-05/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit2">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-04/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-06/">下一讲 →</a></p>
<div class="note-lead"><p>前几讲研究长方形矩阵的列空间与投影。现在回到方阵，想用<strong>一个数</strong>判断它是否可逆、如何改变有向面积或体积。行列式并非先背一个庞大展开式；本讲从行操作对这个数的影响出发，推到实用的消元算法。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-2-session-5-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-2-session-5-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-2-session-5-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="first-properties">一、为什么行交换会变号？</h2>
<p>二维时，若 \(A=\begin{pmatrix}a&b\\c&d\end{pmatrix}\)，行列式为 \(\det A=ad-bc\)。例如把两行交换，得到 \(\begin{pmatrix}c&d\\a&b\end{pmatrix}\)，其行列式是 \(cb-da=-(ad-bc)\)。几何上，交换两条有向边使面积的朝向反转，而普通面积大小不变。</p>
<p>讲义用三条性质刻画任意阶行列式：</p>
<ol><li><strong>单位矩阵的标准值：</strong>\(\det I_n=1\)。</li><li><strong>交替性：</strong>交换两行，行列式变号。</li><li><strong>逐行线性：</strong>其他行固定时，一行乘 \(t\)，行列式也乘 \(t\)；该行若是两行向量之和，行列式等于分别代入所得两项之和。</li></ol>
<p>“逐行线性”不是说 \(\det(A+B)=\det A+\det B\)：后一个式子会同时改动所有行，展开时还有交叉项。比如 \(\det I_2=1\)，但 \(\det(I_2+I_2)=\det(2I_2)=4\)，不是 2。</p>

<h2 id="elimination-rules">二、三个常见行操作怎样影响行列式？</h2>
<p>若两行相同，交换它们不会改变矩阵，却必须使行列式变号，故行列式为零。若有零行，从该行提取因子 0，也得零。利用这两个结论可以证明最重要的计算规则：<strong>把第 \(i\) 行的 \(t\) 倍加到另一行，不改变行列式。</strong></p>
<p>固定其余行，把新行 \(r_j+tr_i\) 按逐行线性拆开。第一项是原行列式，第二项的第 \(j\) 行变成 \(tr_i\)，与第 \(i\) 行重复，所以为零。这样便知道三类操作的完整效果：交换两行乘 \(-1\)，一行乘 \(t\) 也乘 \(t\)，一行加另一行的倍数则不变。</p>
<p>消元把方阵化为上三角矩阵 \(U\)。三角矩阵的行列式是对角线元素的乘积；可以通过再消去非对角元素和逐行线性证明，也可由下一讲的置换展开式直接看出。若消元只用了“加倍数”与 \(s\) 次行交换，</p>
<div class="equation">\[\det A=(-1)^s\prod_{i=1}^{n}u_{ii}.\]</div>
<p>这里 \(u_{ii}\) 是消元后 \(U\) 的第 \(i\) 个对角数。若某一步把行额外放大了 \(t\)，还要除回这个因子；不能无条件只乘最后的对角线。</p>

<h2 id="worked-example">三、亲手用消元算一次</h2>
<p>取 \(A=\begin{pmatrix}2&1&0\\4&3&1\\0&2&1\end{pmatrix}\)。先做 \(R_2\leftarrow R_2-2R_1\)，得到第二行 \((0,1,1)\)；再做 \(R_3\leftarrow R_3-2R_2\)，得到第三行 \((0,0,-1)\)。两次都是“加另一行的倍数”，不改变行列式，所以</p>
<div class="equation">\[\det A=\det\begin{pmatrix}2&1&0\\0&1&1\\0&0&-1\end{pmatrix}=2\cdot1\cdot(-1)=-2.\]</div>
<p>这一计算同时说明：三条主元都非零，\(A\) 可逆；负号还记录其有向体积的朝向反转。若消元产生零主元但可通过交换行继续，必须把交换的负号计入；若最终出现零行，则秩不足 3、行列式为零。一般地，\(\det A=0\) 当且仅当方阵奇异，也当且仅当它的列线性相关。</p>

<h2 id="products">四、乘积、转置与尺度</h2>
<p>一个线性变换 \(B\) 先改变体积，再由 \(A\) 改变一次；总体有向尺度相乘。这对应 \(\det(AB)=\det A\,\det B\)。严格代数证明可先对三类基本行操作验证，随后把可逆矩阵分解为基本矩阵；奇异情形则由乘积不可能可逆处理。由此 \(\det(A^{-1})=1/\det A\)（仅 \(A\) 可逆时），而 \(\det(2A)=2^n\det A\)，因为 \(n\) 行各被乘了 2。</p>
<p>还成立 \(\det(A^T)=\det A\)。因此前面所有关于“行”的规则，也能改写为“列”的规则：交换两列变号，给一列加另一列的倍数不变。下一讲会从一个每行、每列恰选一项的展开式看见这个对称性。</p>
<h2 id="check">理解检查</h2>
<ol><li>讲义习题 18.1：若方阵每行的数之和为零，哪个具体非零向量在 \(N(A)\) 中？为什么这立刻给 \(\det A=0\)？</li><li>若每行的数之和为 1，为什么 \(\det(A-I)=0\)？这是否迫使 \(\det A=1\)？试给一个 \(2\times2\) 反例。</li><li>习题 18.2 的 Vandermonde 矩阵可先用后两行减第一行，再从两行提出 \(b-a,c-a\)。剩下的因子为什么是 \(c-b\)？</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit II, Session 5 <em>Properties of Determinants</em> 整理。数值例子与行操作的逐步说明为本笔记补充。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
