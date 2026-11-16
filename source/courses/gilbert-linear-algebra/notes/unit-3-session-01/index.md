---
title: 第三单元 1：实对称矩阵与正定性｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-3-session-01/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit3">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-11/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-3-session-02/">下一讲 →</a></p>
<div class="note-lead"><p>第二单元找到特征方向；但普通方阵可能没有实特征方向，也可能没有足够多的特征向量。实对称矩阵消除了这两个困难。本讲从一组可核算的方向出发，把谱、二次型、主元与正定性接起来。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-3-session-1-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-3-session-1-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-3-session-1-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="spectral">一、对称性带来了什么？</h2>
<p>实方阵满足 \(A^T=A\) 时称为<strong>实对称矩阵</strong>。先看 \(A=\begin{pmatrix}3&1\\1&3\end{pmatrix}\)：单位方向 \(q_1=(1,1)^T/\sqrt2\)、\(q_2=(1,-1)^T/\sqrt2\) 满足 \(Aq_1=4q_1\)、\(Aq_2=2q_2\)，而且彼此正交。令 \(Q=[q_1\ q_2]\)、\(\Lambda=\operatorname{diag}(4,2)\)，就有 \(Q^TQ=I\) 以及 \(A=Q\Lambda Q^T\)。</p>
<p>一般实对称矩阵也能如此写：\(Q\) 的列是一组标准正交特征向量，\(\Lambda\) 的对角数是实特征值。这是<strong>谱定理</strong>。为什么特征值必为实数？允许特征向量 \(z\) 暂时属于 \(\mathbb C^n\)，用共轭转置 \(z^*=(\overline z)^T\) 左乘 \(Az=\lambda z\)，得 \(z^*Az=\lambda z^*z\)。因为 \(A^*=A\)，数 \(z^*Az\) 等于自身的共轭，故为实数；\(z^*z=\sum_j|z_j|^2\gt0\)，于是 \(\lambda\) 为实数。这里不能把复向量的 \(z^Tz\) 当长度平方，例如 \(z=(1,i)^T\) 时 \(z^Tz=1+i^2=0\)。</p>
<p>对不同特征值的向量 \(u,v\)，对称性还给 \(\lambda u^Tv=(Au)^Tv=u^TAv=\mu u^Tv\)，故 \(u^Tv=0\)。若特征值重复，可在同一特征空间中另选正交基；第二单元第 8 讲的展开说明了完整基为何存在。</p>

<h2 id="projectors">二、谱分解为什么是一组投影？</h2>
<p>把 \(Q\Lambda Q^T\) 按列展开，得到 \(A=\sum_{j=1}^n\lambda_j q_jq_j^T\)。矩阵 \(q_jq_j^T\) 将输入投到直线 \(\operatorname{span}\{q_j\}\)，所以作用 \(A\) 等于先取每个正交方向上的分量，再各乘 \(\lambda_j\)。对上面的二维例子，输入 \(x=(2,0)^T=q_1\sqrt2+q_2\sqrt2\)，两个方向分别放大 4 倍和 2 倍后得到 \(Ax=(6,2)^T\)。</p>
<p>这个拆解也给出二次型的关键公式。若 \(x=Qc\)，其中 \(c=Q^Tx\) 是特征坐标，则</p>
<div class="equation">\[x^TAx=c^T\Lambda c=\sum_{j=1}^n\lambda_jc_j^2.\]</div>
<p>因此对实对称矩阵，<strong>所有特征值为正</strong>，恰好等价于<strong>每个非零实向量都满足 \(x^TAx\gt0\)</strong>。这就是正定。若特征值只要求非负，称半正定；这时可能有非零向量使二次型为零。注意 \(\det A\gt0\) 单独不足以判定正定：\(\operatorname{diag}(-1,-3)\) 行列式为 3，却在每个非零方向上给出负值。</p>

<h2 id="pivots">三、为什么前导子式和主元也能判定？</h2>
<p>讲义的 \(B=\begin{pmatrix}5&2\\2&3\end{pmatrix}\) 有第一主元 5；做 \(R_2\leftarrow R_2-(2/5)R_1\) 后第二主元为 \(3-4/5=11/5\)。两个主元都正。也可配方：</p>
<div class="equation">\[x^TBx=5\left(x_1+\frac25x_2\right)^2+\frac{11}{5}x_2^2.\]</div>
<p>右侧对非零 \(x\) 必正，所以 \(B\) 正定。对应特征方程 \(\lambda^2-8\lambda+11=0\)，根 \(4\pm\sqrt5\) 也都正。三种检查读的是同一事实。</p>
<p>对称矩阵若能在不交换行且主元均非零的条件下逐步消元，就可写成 \(A=LDL^T\)，其中 \(L\) 是对角线全为 1 的下三角矩阵，\(D=\operatorname{diag}(d_1,\ldots,d_n)\) 收集主元。于是 \(x^TAx=(L^Tx)^TD(L^Tx)\)；因 \(L^T\) 可逆，所有主元 \(d_j\gt0\) 与正定等价。正定矩阵必有这样的分解。左上角 \(k\times k\) 子矩阵的行列式 \(\Delta_k=d_1\cdots d_k\)，故也等价于每个<strong>顺序前导子式</strong> \(\Delta_k\gt0\)（Sylvester 判据）。这里要求的是从左上角逐级扩大的那 \(n\) 个子式，并非任取的 \(n\) 个子式。</p>
<p>主元还能提供比“是否正定”更多的信息：在上述分解可行时，正主元和负主元的个数，分别等于正特征值和负特征值的个数。这是二次型的<strong>惯性定律</strong>：可逆的坐标替换 \(y=L^Tx\) 不改变正、负方向各自的最大维数。因此无需解特征方程，也能判断谱的符号分布；对 \(A-bI\) 做同样的检查，还可数出 \(A\) 有多少特征值大于 \(b\)。若消元遇到零主元，不能直接沿用这一无换行分解。</p>
<h2 id="check">理解检查</h2>
<ol><li>习题 25.1 用旋转矩阵的特征向量 \((i,1)^T\) 检查“\(x^TAx/x^Tx\) 一定为实数”的错误证明：分母究竟是多少？</li><li>把 \(B\) 的配方展开，并从两个主元求出 \(\Delta_1,\Delta_2\)。</li><li>为什么两矩阵分别正定，乘积却未必对称？取两个不交换的对称矩阵试一试。</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit III, Session 1 <em>Symmetric Matrices and Positive Definiteness</em> 整理。复特征向量的证明明确使用共轭转置，正定判据注明对称性前提。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
