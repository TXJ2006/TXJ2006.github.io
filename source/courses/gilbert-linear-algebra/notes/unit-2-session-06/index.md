---
title: 第二单元 6：行列式展开与余子式｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-2-session-06/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit2">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-05/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-07/">下一讲 →</a></p>
<div class="note-lead"><p>上讲用消元计算行列式，本讲追问：不做消元，行列式的各项究竟从哪里来？先观察两阶和三阶矩阵中“每行、每列选一个数”的规则，再把项按某一行分组，才得到余子式展开。大公式放在理解这些选择之后。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-2-session-6-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-2-session-6-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-2-session-6-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="one-per-row-column">一、为什么每项必须“每行每列各取一个”？</h2>
<p>从两阶矩阵 \(A=\begin{pmatrix}a&b\\c&d\end{pmatrix}\) 看起：\(ad\) 选的是第一行第一列、第二行第二列；\(-bc\) 选的是第一行第二列、第二行第一列。两项都从每行取一个数，而且不能重复取同一列。原因是把每一行按坐标方向拆开后，若两行都选同一列，拆出的矩阵就有两条平行的行，行列式为零。</p>
<p>三阶时，第一行先有三种列可选；第二行只剩两种，第三行只剩一种，因此恰有 \(3\cdot2\cdot1=6\) 个可能非零项。每项的正负由把所选列顺序排回 \(1,2,3\) 所需交换次数的奇偶决定。比如选 \((2,3,1)\) 时，项 \(a_{12}a_{23}a_{31}\) 为正；选 \((2,1,3)\) 时，\(a_{12}a_{21}a_{33}\) 为负。</p>
<p>对具体矩阵 \(B=\begin{pmatrix}1&2&0\\0&3&4\\5&0&6\end{pmatrix}\)，多数选择因含 0 消失，只剩 \(1\cdot3\cdot6\) 和 \(2\cdot4\cdot5\)，两项都为正，所以 \(\det B=18+40=58\)。这比盲背三阶展开式容易核对。</p>
<details class="note-extension"><summary>理解之后的一般置换公式</summary><p>对 \(n\times n\) 矩阵，令 \(\sigma\) 遍历 \(1,\ldots,n\) 的所有排列；\(\sigma(i)\) 指第 \(i\) 行选的列。只有各列互不重复的选法能留下，故有 \(\det A=\sum_{\sigma}\operatorname{sgn}(\sigma)\prod_{i=1}^n a_{i,\sigma(i)}\)。\(\operatorname{sgn}(\sigma)\) 为偶数次交换时的 \(1\)，奇数次时的 \(-1\)。为什么“奇偶”不依赖具体交换路径？可数出排列中的<strong>逆序对</strong>：满足 \(i\lt j\) 却有 \(\sigma(i)\gt\sigma(j)\) 的一对位置。每做一次相邻交换，逆序对数的奇偶就翻转；因此交换路径的奇偶由排列本身决定。共有 \(n!\) 个候选项；实际计算大矩阵通常仍用消元。</p></details>

<h2 id="cofactor">二、余子式是怎样从分组中出现的？</h2>
<p>先固定第一行的选择。若这一行选 \(a_{11}\)，余下各行只能在删去第一列后的矩阵中各选一列，贡献合起来是 \(a_{11}\) 乘一个 \((n-1)\) 阶行列式。若选 \(a_{12}\)，要把第二列移到第一列的位置，产生一个负号；依次选下去，符号交替。</p>
<p>把“删去第 \(i\) 行、第 \(j\) 列后得到的矩阵”记为 \(M_{ij}\)。它是 \((n-1)\times(n-1)\) 方阵。定义相应的<strong>代数余子式</strong> \(C_{ij}=(-1)^{i+j}\det M_{ij}\)。下标 \(i,j\) 指被删去的行与列，符号形成棋盘格：第一行依次为 \(+,-,+,\ldots\)。沿第一行展开就是</p>
<div class="equation">\[\det A=a_{11}C_{11}+a_{12}C_{12}+\cdots+a_{1n}C_{1n}.\]</div>
<p>也可以沿其他任何一行或一列展开；选择零较多的一行或一列通常最省事。把上讲的矩阵 \(A=\begin{pmatrix}2&1&0\\4&3&1\\0&2&1\end{pmatrix}\) 沿第一行展开：第一项删去首行首列，得 \(2(3\cdot1-1\cdot2)=2\)；第二项带负号，得 \(-1(4\cdot1-1\cdot0)=-4\)；第三项为零。总和 \(-2\)，与消元结果一致。</p>
<p>为什么第二项一定带负号？不是“交叉乘法”的记忆技巧，而是把第一行所选的第二列交换到第一列，恰好交换一次。把第 \(i,j\) 处的元素移到左上角，要越过 \(i-1\) 行与 \(j-1\) 列，总共 \(i+j-2\) 次，奇偶性等同于 \(i+j\)。</p>

<h2 id="tridiagonal">三、一个递推比反复展开更有用</h2>
<p>令 \(T_n\) 为主对角线和紧邻上下两条对角线全为 1、其余为 0 的 \(n\times n\) 矩阵。记 \(D_n=\det T_n\)，并约定 \(D_0=1\)（空矩阵的行列式），则 \(D_1=1,\ D_2=\det\begin{pmatrix}1&1\\1&1\end{pmatrix}=0\)。</p>
<p>沿 \(T_n\) 的第一行展开，只有头两个数非零。第一个余子式是 \(T_{n-1}\)；第二项带负号，继续沿其首列展开后是 \(T_{n-2}\)。所以</p>
<div class="equation">\[D_n=D_{n-1}-D_{n-2}\qquad(n\ge2).\]</div>
<p>逐项得到 \(D_0=1,D_1=1,D_2=0,D_3=-1,D_4=-1,D_5=0,D_6=1\)，之后每六项重复。讲义摘要在 \(D_4\) 的一行写出了与此递推不一致的数；直接展开或按递推验算，正确值是 \(-1\)。这也说明公式不仅用于求一个数，还能揭示一族矩阵的结构。</p>
<h2 id="check">理解检查</h2>
<ol><li>习题 19.1 的四阶矩阵每行、每列恰有一个 1。选中的列排列是什么？只靠交换次数判断其行列式的正负。</li><li>在上面的 \(B\) 中沿第一列展开，能否再次得到 58？</li><li>对 \(T_4\) 先写出完整矩阵，再用两次余子式展开核对 \(D_4=-1\)。</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit II, Session 6 <em>Determinant Formulas and Cofactors</em> 整理。三对角递推逐项验算，并更正摘要中 \(D_4\) 的不一致数值。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
