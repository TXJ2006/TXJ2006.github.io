---
title: 第三单元 2：复矩阵与快速傅里叶变换｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-3-session-02/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit3">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-3-session-01/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-3-session-03/">下一讲 →</a></p>
<div class="note-lead"><p>上一讲为处理实矩阵的复特征向量引入共轭转置。本讲把它用到底：复数向量怎样定义长度，离散傅里叶变换怎样把数据拆成频率，以及 FFT 为什么能省掉大量重复计算。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-3-session-2-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-3-session-2-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-3-session-2-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="complex-inner-product">一、复向量为何不能沿用转置？</h2>
<p>向量 \(z=(1,i)^T\) 非零，但 \(z^Tz=1+i^2=0\)，所以它不能表示长度平方。令 \(z^*=(\overline z)^T\) 为共轭转置，定义 \(\langle y,z\rangle=y^*z\)，则 \(\|z\|^2=z^*z=\sum_j|z_j|^2\)，本例为 2。这个内积对第二个位置线性，对第一个位置共轭线性；\(\langle z,y\rangle=\overline{\langle y,z\rangle}\)。</p>
<p>复方阵满足 \(H^*=H\) 时称<strong>Hermitian 矩阵</strong>，这是实对称矩阵在复数域的对应物。其特征值为实数，且可选标准正交特征基，写成 \(H=U\Lambda U^*\)。方阵 \(U\) 满足 \(U^*U=I\) 时称<strong>酉矩阵</strong>，它保持长度：\(\|Ux\|^2=x^*U^*Ux=\|x\|^2\)。在实数域，酉矩阵就是正交矩阵。两者角色不要混淆：Hermitian 关心矩阵与自身共轭转置相等；酉矩阵关心共轭转置是逆矩阵。</p>

<h2 id="fourier">二、四点数据怎样变成四个频率坐标？</h2>
<p>记 \(\omega_q=e^{2\pi i/q}\)，它绕单位圆走 \(q\) 等份，满足 \(\omega_q^q=1\)。令行列下标从 0 到 \(n-1\)，定义 \(n\times n\) 傅里叶矩阵 \((F_n)_{jk}=\omega_n^{jk}\)。对输入 \(x=(x_0,\ldots,x_{n-1})^T\)，输出 \(y=F_nx\) 的第 \(j\) 项是 \(y_j=\sum_{k=0}^{n-1}x_k\omega_n^{jk}\)。这是<strong>此讲义采用的正指数约定</strong>；有些资料把正负号反过来，使用时要一起改逆变换。</p>
<p>当 \(n=4\) 时 \(\omega_4=i\)，于是</p>
<div class="equation">\[F_4=\begin{pmatrix}1&1&1&1\\1&i&-1&-i\\1&-1&1&-1\\1&-i&-1&i\end{pmatrix}.\]</div>
<p>输入 \(x=(1,0,0,0)^T\) 只在第 0 个位置有一个脉冲，却输出 \((1,1,1,1)^T\)：每个频率坐标都参与重建这个尖峰。傅里叶列向量正交的原因是有限等比和：若 \(j\ne k\)，\(\sum_{r=0}^{n-1}\omega_n^{(k-j)r}=0\)；同一列与自身的内积则是 \(n\)。因此 \(F_n^*F_n=nI\)，归一化矩阵 \(F_n/\sqrt n\) 才是酉矩阵，正确逆变换为 \(x=F_n^*y/n\)。对一般数据，再乘 \(F_n\) 并不等于求逆；脉冲例子恰好返回原方向只是特殊对称性。</p>

<h2 id="fft">三、FFT 省下的是哪部分运算？</h2>
<p>设长度为 \(2m\)，把输入分为偶下标 \((x_0,x_2,\ldots)\) 和奇下标 \((x_1,x_3,\ldots)\)。各自做一个长度 \(m\) 的变换：\(E_j=\sum_{r=0}^{m-1}x_{2r}\omega_m^{jr}\)，\(O_j=\sum_{r=0}^{m-1}x_{2r+1}\omega_m^{jr}\)。原和式中，偶下标项含 \(\omega_{2m}^{2jr}=\omega_m^{jr}\)；奇下标项只多出因子 \(\omega_{2m}^{j}\)。因此</p>
<div class="equation">\[\begin{aligned}
y_j&=E_j+\omega_{2m}^{j}O_j,\\
y_{j+m}&=E_j-\omega_{2m}^{j}O_j\qquad(0\leq j\lt m).
\end{aligned}\]</div>
<p>第二式来自 \(\omega_{2m}^{j+m}=-\omega_{2m}^j\)；这两个输出共享同一对较短变换结果。若 \(n=2^s\)，不断对半拆分，运算量满足 \(T(n)=2T(n/2)+O(n)\)，所以为 \(O(n\log_2 n)\)，而逐项矩阵乘法要 \(O(n^2)\)。这里的效率来自单位根的重复结构，<strong>正交性保证可逆</strong>，却不是单独带来 FFT 速度的原因。</p>
<p>在四点情形，\(E_0=x_0+x_2\)、\(E_1=x_0-x_2\)，\(O_0=x_1+x_3\)、\(O_1=x_1-x_3\)；代入上式即可得到 \(F_4x\) 的四个分量。这一步把抽象的“拆成两半”落实为两次 \(F_2\) 运算。</p>
<p>习题 26.2 的 \(F_4\) 分解正是一次这样的拆分：先把下标排成 \((0,2,1,3)\)，做两个 \(F_2\)，再乘旋转因子 \(1,i\) 并合并上下两半。把这三步分别写出来，比直接记块矩阵更容易理解。</p>
<h2 id="check">理解检查</h2>
<ol><li>求 \(F_2\)，直接检查 \(F_2^*F_2=2I\)。</li><li>对 \(x=(1,2,0,0)^T\)，先算 \(F_4x\)，再用 \(F_4^*/4\) 恢复原输入。</li><li>在四点情形写出 \(E_0,E_1,O_0,O_1\)，逐项验证上面的两个合并公式。</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit III, Session 2 <em>Complex Matrices; Fast Fourier Transform</em> 整理。明确区分正指数约定、共轭逆变换与 FFT 的结构性加速。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
