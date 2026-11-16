---
title: 第三单元 4：相似矩阵与 Jordan 形｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-3-session-04/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit3">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-3-session-03/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-3-session-05/">下一讲 →</a></p>
<div class="note-lead"><p>对角化不是让矩阵本身突然变成对角阵，而是换一组坐标。本讲先把“同一变换的不同坐标写法”精确地叫作相似，再问：特征向量不够时，矩阵还能简化到什么程度？</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-3-session-4-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-3-session-4-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-3-session-4-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="similarity">一、相似为什么保留特征值？</h2>
<p>两个同阶方阵 \(A,B\) 若满足 \(B=M^{-1}AM\)，其中 \(M\) 可逆，就称<strong>相似</strong>。把新坐标 \(c\) 先由 \(M\) 变成旧坐标 \(Mc\)，作用 \(A\)，最后用 \(M^{-1}\) 返回新坐标，恰得到 \(Bc\)。若 \(Av=\lambda v\)，则 \(B(M^{-1}v)=\lambda(M^{-1}v)\)，所以对应特征空间的维数相同；又因 \(\det(tI-B)=\det(M^{-1}(tI-A)M)=\det(tI-A)\)，特征值连同重数也相同。</p>
<p>更强地，\(B^k=M^{-1}A^kM\)，所以 \(\operatorname{rank}(B^k)=\operatorname{rank}(A^k)\)。若 \(A\) 可对角化，选特征向量作 \(M\) 的列，\(B\) 就是对角矩阵。这解释了第二单元的 \(A=S\Lambda S^{-1}\)。但“特征值相同”不够推出相似，例如 \(4I_2\) 与 \(\begin{pmatrix}4&1\\0&4\end{pmatrix}\) 都有两个特征值 4，前者有二维特征空间，后者只有一维。</p>

<h2 id="jordan-block">二、缺失的特征向量留下什么痕迹？</h2>
<p>令 \(J=\begin{pmatrix}4&1\\0&4\end{pmatrix}=4I+N\)，其中 \(N=\begin{pmatrix}0&1\\0&0\end{pmatrix}\) 满足 \(N^2=0\)。只有 \(e_1=(1,0)^T\) 是独立特征向量；但 \((J-4I)e_2=e_1\)，所以 \(e_2\) 是一个<strong>广义特征向量</strong>，作用一次后落到特征向量上。这个 1 正是无法对角化留下的耦合。</p>
<p>因为 \(4I\) 与 \(N\) 可交换，二项式展开立即给</p>
<div class="equation">\[J^k=4^kI+k4^{k-1}N\quad(k\geq1),\qquad e^{Jt}=e^{4t}(I+tN).\]</div>
<p>这说明只写“特征值为 4，因此变化是 \(4^k\) 或 \(e^{4t}\)”会漏掉多项式因子。Jordan 块就是把这种特征方向链条排在一起：对角线是同一个 \(\lambda\)，紧贴其上的超对角线是 1。一般 Jordan 标准形存在于<strong>复数域</strong>；实矩阵若有非实特征值，要先扩展到复数域，不能无条件宣称有实 Jordan 形。</p>

<h2 id="blocks">三、同样的根与特征向量数，也未必相似</h2>
<p>看两个四阶零特征值矩阵：\(J_{3}\oplus J_1\) 有一条长度 3 的 Jordan 链加一条长度 1 的链；\(J_2\oplus J_2\) 有两条长度 2 的链。两者特征值都是四个 0，零空间维数都是 2，秩也都是 2。但前者平方非零且秩为 1，后者平方就是零矩阵。因此它们不能相似，因为相似必保留各次幂的秩。</p>
<p>一般而言，固定特征值 \(\lambda\) 后，\(\dim\ker(A-\lambda I)^k\) 随 \(k\) 的增长记录 Jordan 链的长度；它比只数特征值与普通特征向量更细。这也是本讲习题 28.1 不能只凭“都有两个特征向量”作结论的原因。</p>

<h2 id="ata">四、与正定矩阵的联系</h2>
<p>正定矩阵可以正交对角化，所以不会有非平凡 Jordan 块。它们还有两条直接可证的性质：若 \(P\) 正定，\(P^{-1}\) 的特征值是原来正特征值的倒数，故仍正定；若 \(P,Q\) 都正定，则 \(x^T(P+Q)x=x^TPx+x^TQx\gt0\) 对非零 \(x\) 成立，故 \(P+Q\) 也正定。</p>
<p>对任意实矩阵 \(C\)，\(C^TC\) 总是对称半正定，因为 \(x^TC^TCx=\|Cx\|^2\geq0\)；只有 \(C\) 列无关时才正定。即使 \(C\) 自身不可对角化，\(C^TC\) 仍能找到标准正交特征基。下一讲的 SVD 就从这个事实出发。</p>
<h2 id="check">理解检查</h2>
<ol><li>核算 \(J^2,J^3\)，找出上式里系数 \(k\) 的来源。</li><li>写出 \(J_3\oplus J_1\) 与 \(J_2\oplus J_2\) 的平方，并比较秩。</li><li>习题 28.2(e)：若同时交换方阵的第 1、2 行和第 1、2 列，写出对应置换矩阵 \(P\)，说明结果为什么是 \(P^{-1}AP\)。</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit III, Session 4 <em>Similar Matrices and Jordan Form</em> 整理。用广义特征向量和幂的秩解释 Jordan 块的必要性。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
