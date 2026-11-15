---
title: 第二单元 11：马尔可夫矩阵与傅里叶级数｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-2-session-11/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit2">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-10/">上一讲</a></p>
<div class="note-lead"><p>前两讲用特征向量拆解矩阵的反复作用。这一讲先看一个概率模型：总量不变时，状态会怎样趋向稳态？随后把向量的正交投影搬到函数空间：傅里叶系数为什么由积分给出？两部分都在寻找复杂对象的合适“坐标”。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-2-session-11-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-2-session-11-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-2-session-11-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="markov">一、每一列为什么必须加起来等于 1？</h2>
<p>设状态 \(u_k=(C_k,M_k)^T\) 表示第 \(k\) 步在两地的人数，第一项为加州、第二项为马萨诸塞州。讲义给出的转移矩阵是</p>
<div class="equation">\[u_{k+1}=Au_k,\qquad A=\begin{pmatrix}0.9&0.2\\0.1&0.8\end{pmatrix}.\]</div>
<p>第一列说明：原在加州的人有 \(90\%\) 留下、\(10\%\) 转到另一地；第二列说明：原在马萨诸塞州的人有 \(20\%\) 转到加州、\(80\%\) 留下。每列描述<strong>同一出发地</strong>的人如何分配，因此必须和为 1。元素非负且每列和为 1 的矩阵，在这里称为<strong>列随机矩阵</strong>（讲义称 Markov matrix）。</p>
<p>例如 \(u_0=(0,1000)^T\) 给出 \(u_1=(200,800)^T\)。令 \(\boldsymbol 1^T=(1,1)\)，则 \(\boldsymbol 1^TA=\boldsymbol 1^T\)，因而 \(\boldsymbol 1^Tu_{k+1}=\boldsymbol 1^Tu_k\)：两地总人数始终为 1000。这里 \(\boldsymbol 1^T\) 是左特征向量；它表达总量守恒，而人口比例的稳态要找右特征向量。</p>

<h2 id="steady-state">二、稳态从哪里来，何时真的会趋近它？</h2>
<p>稳态是不再改变的状态 \(u_*\)，满足 \(Au_*=u_*\)，即 \((A-I)u_*=0\)。本例两行都化成 \(-0.1C+0.2M=0\)，所以 \(C=2M\)。总数为 1000 时，稳态是 \(u_*=(2000/3,1000/3)^T\)。列和为 1 保证 \(A-I\) 的各列和为零，因此其行线性相关、行列式为零；这解释了为什么 1 一定是特征值，却没有单凭这一点保证收敛。</p>
<p>本例另一个特征值是 \(0.9+0.8-1=0.7\)（迹等于特征值之和），相应特征向量可取 \((-1,1)^T\)：它让一地增加、另一地等量减少，总人数仍不变。将初值拆成稳态与偏差：</p>
<div class="equation">\[\begin{aligned}
u_0&=\frac{1000}{3}\begin{pmatrix}2\\1\end{pmatrix}
+\frac{2000}{3}\begin{pmatrix}-1\\1\end{pmatrix},\\
u_k&=\frac{1000}{3}\begin{pmatrix}2\\1\end{pmatrix}
+\frac{2000}{3}(0.7)^k\begin{pmatrix}-1\\1\end{pmatrix}.
\end{aligned}\]</div>
<p>第一项始终不动，第二项每一步缩为原来的 \(0.7\) 倍，所以趋于零。这才是此例趋向稳态的理由。下面的图画出前几步第一分量 \(C_k\) 逐渐接近 \(2000/3\)，而第二分量总是 \(1000-C_k\)。</p>
<figure class="note-figure"><svg viewBox="0 0 560 270" role="img" aria-labelledby="markov-title markov-desc"><title id="markov-title">加州人数逐步趋向稳态</title><desc id="markov-desc">第零到第五步的加州人数依次为 0、200、340、438、506.6、554.62，逐渐靠近 666.67。</desc><path d="M65 215H510M65 215V25" fill="none" stroke="#87958f" stroke-width="2"/><path d="M65 45H510" stroke="#bb6a4b" stroke-width="2" stroke-dasharray="5 5"/><polyline points="65,215 145,164 225,128 305,103 385,86 465,74" fill="none" stroke="#557f79" stroke-width="3"/><g fill="#557f79"><circle cx="65" cy="215" r="4"/><circle cx="145" cy="164" r="4"/><circle cx="225" cy="128" r="4"/><circle cx="305" cy="103" r="4"/><circle cx="385" cy="86" r="4"/><circle cx="465" cy="74" r="4"/></g><text x="44" y="237">0</text><text x="140" y="237">1</text><text x="220" y="237">2</text><text x="300" y="237">3</text><text x="380" y="237">4</text><text x="460" y="237">5</text><text x="500" y="239">k</text><text x="11" y="47">667</text><text x="349" y="35">稳态 2000/3</text><text x="10" y="19">Cₖ</text></svg><figcaption>人数按递推通项计算，图中纵向位置近似按比例绘制；虚线为稳态值 \(2000/3\)。</figcaption></figure>
<p>一般列随机矩阵的所有特征值满足 \(|\lambda|\leq1\)：它把向量的 \(1\) 范数至多保持原值（对任意可能带负号的向量也成立）。但不能说除 1 以外的特征值都严格小于 1 的模。例如 \(\begin{pmatrix}0&1\\1&0\end{pmatrix}\) 的另一个特征值是 \(-1\)，状态会在两地之间来回交换。若 1 是唯一的单位圆上特征值且稳态方向唯一，有限维情形下其余模式衰减；常见的严格正转移矩阵满足这样的收敛条件。若初始总量为 \(M\)，本例的极限改成 \((2M/3,M/3)^T\)，并非不看初值就有一个固定人数。</p>

<h2 id="function-inner-product">三、函数怎样像向量一样“垂直”？</h2>
<p>在 \(\mathbb R^n\) 中，两向量的内积是 \(v^Tw=\sum_jv_jw_j\)。对区间 \([0,2\pi]\) 上的实函数，把连续的取值相乘再累加，得到函数内积</p>
<div class="equation">\[\langle f,g\rangle=\int_0^{2\pi}f(x)g(x)\,dx.\]</div>
<p>当内积为零，我们就说两个函数在这个区间正交。例如 \(\langle\sin x,\cos x\rangle=\int_0^{2\pi}\sin x\cos x\,dx=0\)。更一般地，常数函数 1、\(\cos nx\)、\(\sin nx\)（\(n=1,2,\ldots\)）两两正交；可以用积化和差公式验算。它们并非都已归一化：\(\langle1,1\rangle=2\pi\)，而 \(\langle\cos nx,\cos nx\rangle=\langle\sin nx,\sin nx\rangle=\pi\)。这个长度差异决定了下面分母为何不同。</p>

<h2 id="fourier-coefficients">四、傅里叶系数就是沿各方向的投影坐标</h2>
<p>先取有限个三角函数，令 \(S_N(x)=a_0+\sum_{n=1}^N(a_n\cos nx+b_n\sin nx)\)。目标是让 \(f-S_N\) 与所选的每个函数正交；这和第二、三讲把误差垂直投到列空间完全相同。比如对 \(\cos mx\) 取内积，除 \(a_m\cos mx\) 外其余项都为零，于是 \(\langle f,\cos mx\rangle=\pi a_m\)。同理得到</p>
<div class="equation">\[\begin{aligned}
a_0&=\frac1{2\pi}\int_0^{2\pi}f(x)\,dx,\\
a_n&=\frac1\pi\int_0^{2\pi}f(x)\cos(nx)\,dx,\\
b_n&=\frac1\pi\int_0^{2\pi}f(x)\sin(nx)\,dx.
\end{aligned}\]</div>
<p>这里 \(a_0\) 是平均值；若采用另一种写法 \(a_0/2+\sum_{n\ge1}(a_n\cos nx+b_n\sin nx)\)，常数系数的公式便改为 \(a_0=\pi^{-1}\int f\)。两种记号不要混用。例如 \(f(x)=1+2\cos x+\sin2x\)，正交性直接给出常数坐标 1、\(\cos x\) 坐标 2、\(\sin2x\) 坐标 1，其余坐标为零。</p>
<p>对平方可积函数，按这些公式构成的 \(S_N\) 是 \(f\) 在有限维空间 \(\operatorname{span}\{1,\cos x,\sin x,\ldots,\cos Nx,\sin Nx\}\) 中的<strong>正交投影</strong>：对这个空间里的任意 \(g\)，都有 \(\langle f-S_N,g\rangle=0\)，所以 \(\|f-g\|_2^2=\|f-S_N\|_2^2+\|S_N-g\|_2^2\)。这说明它使积分平方误差最小。令 \(N\to\infty\) 时能否逐点收敛到 \(f(x)\)，还要看函数的条件；有限投影的结论本身不需要先假定逐点收敛。</p>

<h2 id="check">理解检查</h2>
<ol><li>用矩阵乘法算 \(u_2\)，再用上面的通项式核对；说明为何每一步总人数为 1000。</li><li>对交换矩阵与初值 \((1,0)^T\)，写出前四步，解释为何“1 是特征值”不足以推出极限存在。</li><li>计算 \(\int_0^{2\pi}\cos^2x\,dx\) 与 \(\int_0^{2\pi}\cos x\sin2x\,dx\)，再说出 \(f=3+4\cos x-2\sin2x\) 的非零傅里叶系数。</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit II, Session 11 <em>Markov Matrices; Fourier Series</em> 整理。补充了周期链的反例与有限投影的精确适用范围。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
