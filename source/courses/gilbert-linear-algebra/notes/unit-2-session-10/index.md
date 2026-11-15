---
title: 第二单元 10：微分方程与矩阵指数｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-2-session-10/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit2">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-09/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-11/">下一讲 →</a></p>
<div class="note-lead"><p>上一讲的状态按整步更新：\(u_{k+1}=Au_k\)，所以出现 \(\lambda^k\)。现在状态在连续时间变化，满足 \(u'(t)=Au(t)\)。同一特征方向上的变化率与当前值成正比，因而 \(\lambda^k\) 换成 \(e^{\lambda t}\)；把所有方向合起来就是矩阵指数 \(e^{At}\)。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-2-session-10-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-2-session-10-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-2-session-10-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="modes">一、为什么特征向量给出“纯模式”？</h2>
<p>若 \(Av=\lambda v\)，试取 \(u(t)=e^{\lambda t}v\)。它的导数是 \(\lambda e^{\lambda t}v\)，而 \(Au(t)=e^{\lambda t}Av=\lambda e^{\lambda t}v\)，所以确实满足 \(u'=Au\)。这不是猜完就停：已经把两边逐项代入验证。</p>
<p>如果 \(A\) 有一组特征基 \(v_1,\ldots,v_n\)，初值 \(u(0)=u_0\) 可唯一写成 \(\sum_jc_jv_j\)。线性性给出整个解：</p>
<div class="equation">\[u(t)=\sum_{j=1}^{n}c_je^{\lambda_jt}v_j.\]</div>
<p>\(c_j\) 由初值决定，\(e^{\lambda_jt}\) 决定该模式随时间的变化。若 \(\lambda_j<0\)，该模式衰减；\(\lambda_j=0\) 时保持；\(\lambda_j>0\) 时增长。复特征值还可能带来振荡，不能把所有情况都理解成单调变化。</p>

<h2 id="worked-system">二、把讲义的两变量系统从头解完</h2>
<p>讲义考虑 \(u_1'=-u_1+2u_2,\ u_2'=u_1-2u_2\)，即 \(u'=Au\)，其中</p>
<div class="equation">\[A=\begin{pmatrix}-1&2\\1&-2\end{pmatrix},\qquad u(0)=\begin{pmatrix}1\\0\end{pmatrix}.\]</div>
<p>\(\det A=0\)、\(\operatorname{tr}A=-3\)，故两个特征值是 \(0,-3\)。直接验证 \(v_0=(2,1)^T\) 满足 \(Av_0=0\)，\(v_{-3}=(1,-1)^T\) 满足 \(Av_{-3}=-3v_{-3}\)。把初值拆开：\((1,0)^T=\frac13(2,1)^T+\frac13(1,-1)^T\)。因此</p>
<div class="equation">\[u(t)=\frac13\begin{pmatrix}2\\1\end{pmatrix}+\frac{e^{-3t}}3\begin{pmatrix}1\\-1\end{pmatrix}.\]</div>
<p>两个分量分别是 \(u_1(t)=(2+e^{-3t})/3\)、\(u_2(t)=(1-e^{-3t})/3\)。当 \(t=0\) 得 \((1,0)^T\)，当 \(t\to\infty\) 得稳态 \((2/3,1/3)^T\)。还有一个不用求特征值就能先发现的守恒量：两条微分方程相加得到 \((u_1+u_2)'=0\)，所以总量始终为 1。代数上这是 \((1,1)A=0\)，即 \((1,1)^T\in N(A^T)\)。</p>

<h2 id="exponential">三、\(e^{At}\) 是怎样定义与计算的？</h2>
<p>实数指数有幂级数 \(e^z=1+z+z^2/2!+\cdots\)。把数 \(z\) 换成方阵 \(At\)，并把常数 1 换成单位矩阵，定义</p>
<div class="equation">\[e^{At}=I+At+\frac{A^2t^2}{2!}+\frac{A^3t^3}{3!}+\cdots.\]</div>
<p>这个级数对每个有限矩阵、有限时间都收敛。逐项求导可验 \(\frac{d}{dt}e^{At}=Ae^{At}\)，且 \(e^{A0}=I\)，所以 \(u(t)=e^{At}u_0\) 满足微分方程和初值。这个定义不要求 \(A\) 可对角化。</p>
<p>若 \(A=S\Lambda S^{-1}\)，代入每一个幂 \(A^k=S\Lambda^kS^{-1}\)，得到 \(e^{At}=Se^{\Lambda t}S^{-1}\)，其中 \(e^{\Lambda t}\) 只需把对角线上的 \(\lambda_j\) 换成 \(e^{\lambda_jt}\)。这正好重现上一节的“逐特征方向求解”。</p>
<p>对讲义的具体 \(A\)，还可以不写 \(S^{-1}\) 的全部元素，而把两种模式分开：</p>
<div class="equation">\[e^{At}=\frac13\begin{pmatrix}2&2\\1&1\end{pmatrix}
+\frac{e^{-3t}}3\begin{pmatrix}1&-2\\-1&2\end{pmatrix}.\]</div>
<p>取 \(t=0\)，两矩阵之和是 \(I\)；在 \(t=0\) 求导，结果是 \(A\)。长期极限是第一个矩阵，它把任意初值投到满足 \(Av=0\) 的稳态方向，但这一步还保留初值总量：若 \(u_0\) 的分量和为 \(M\)，极限就是 \((2M/3,M/3)^T\)。</p>

<h2 id="stability">四、稳定性的判断要说明初值与前提</h2>
<p>对可对角化矩阵，所有特征值实部都负时，每个模式都衰减，故每个初值的解趋零。若有实部为正的特征值，沿相应特征方向的初值会增长；但初值若恰好没有那个分量，不能说它也一定增长。若某个特征值为零，可能出现非零稳态；要让<strong>每个</strong>解都收敛到稳态，还须其余模式衰减，并排除零特征值的非平凡 Jordan 块。仅看迹为负不够，例如 \(\operatorname{diag}(-2,1)\) 的迹为 \(-1\)，仍有增长方向。</p>
<details class="note-extension"><summary>二阶方程怎样成为一阶矩阵方程？</summary><p>对 \(y''+by'+ky=0\)，令 \(u=(y,y')^T\)，那么 \(u'=(y',y'')^T=\begin{pmatrix}0&1\\-k&-b\end{pmatrix}u\)。这样就可用矩阵特征值研究衰减、振荡与增长。原来的一个二阶方程，需要两个初始数 \(y(0),y'(0)\)，恰好对应二维状态的两个分量。</p></details>
<h2 id="check">理解检查</h2>
<ol><li>从具体解验证 \(u_1(t)+u_2(t)=1\) 与 \(u'(t)=Au(t)\)。</li><li>习题 23.1 的 \(A^T=-A\)。对 \(u'=Au\)，计算 \(\frac{d}{dt}\|u(t)\|^2\)，说明为什么长度保持不变。</li><li>对上面的 \(e^{At}\)，直接代 \(t=0\) 并求一次导数，分别核对 \(I\) 与 \(A\)。</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit II, Session 10 <em>Differential Equations and exp(At)</em> 整理。对稳定性的初值依赖及零特征值情形补充了适用条件。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
