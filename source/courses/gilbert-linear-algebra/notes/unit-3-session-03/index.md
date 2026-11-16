---
title: 第三单元 3：正定矩阵、二次型与极小值｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-3-session-03/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit3">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-3-session-02/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-3-session-04/">下一讲 →</a></p>
<div class="note-lead"><p>第 1 讲给出了正定的等价判据。本讲把判据落实到可见的函数：为什么有的二次曲面像碗，有的像鞍？配方、主元、Hessian 和椭球主轴说的是同一件事。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-3-session-3-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-3-session-3-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-3-session-3-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="quadratic">一、矩阵怎样变成一个标量函数？</h2>
<p>对实对称矩阵 \(A=\begin{pmatrix}a&b\\b&c\end{pmatrix}\) 和向量 \(x=(s,t)^T\)，定义<strong>二次型</strong> \(q(x)=x^TAx=as^2+2bst+ct^2\)。交叉项前的 2 来自矩阵两个对称位置各贡献一次 \(bst\)。若 \(a\gt0\)，配方给出</p>
<div class="equation">\[q(s,t)=a\left(s+\frac ba t\right)^2+\left(c-\frac{b^2}{a}\right)t^2.\]</div>
<p>所以正定的两个条件是 \(a\gt0\) 和 \(c-b^2/a\gt0\)，后者等价于 \(\det A=ac-b^2\gt0\)。它们正是消元的两个主元与两个顺序前导子式的判据。若 \(a\) 非正，不能从行列式为正直接断定正定。</p>

<h2 id="three-cases">二、同一交叉项，三个不同的几何形状</h2>
<p>考虑讲义的一族矩阵 \(A_y=\begin{pmatrix}2&6\\6&y\end{pmatrix}\)。沿第一行消元，第二主元是 \(y-18\)；对应二次型为 \(q_y(s,t)=2(s+3t)^2+(y-18)t^2\)。于是：</p>
<ol><li>\(y\gt18\)：两项平方的系数都正，原点是严格极小点；每条正高度的等值线是椭圆。</li><li>\(y=18\)：\(q_y=2(s+3t)^2\) 半正定，沿 \((s,t)=(-3t,t)\) 的整条直线都等于零；原点不是唯一极小点。</li><li>\(y\lt18\)：沿 \(s=-3t\) 得负值，沿 \(t=0\) 得正值；原点附近同时有上升和下降方向，是鞍点。</li></ol>
<p>例如 \(y=20\) 时 \(q=2(s+3t)^2+2t^2\)；\(y=7\) 时 \(q=2(s+3t)^2-11t^2\)。这里“沿某个方向为零”与“沿某个方向为负”是半正定和不定的本质区别。</p>
<figure class="note-figure"><svg viewBox="0 0 600 250" role="img" aria-labelledby="quad-title quad-desc"><title id="quad-title">同一二次型在三个参数值下的几何</title><desc id="quad-desc">横轴为 s，纵轴为 t。y 等于 20 时，一条细长椭圆的长轴沿斜向右下；y 等于 18 时，零值直线沿同一斜向；y 等于 7 时，水平线给正值，斜向直线给负值。</desc><path d="M198 25V220M402 25V220" stroke="#8e9992"/><g stroke="#87958f" fill="none"><path d="M24 126H172M96 50V200M228 126H372M300 50V200M432 126H578M505 50V200"/></g><ellipse cx="96" cy="126" rx="72" ry="7" transform="rotate(17 96 126)" fill="none" stroke="#557f79" stroke-width="2"/><path d="M230 103L370 149" stroke="#557f79" stroke-width="3"/><path d="M435 126H575" stroke="#557f79" stroke-width="3"/><path d="M435 103L575 149" stroke="#bb6a4b" stroke-width="3"/><g text-anchor="middle"><text x="96" y="38">y = 20</text><text x="300" y="38">y = 18</text><text x="505" y="38">y = 7</text><text x="96" y="230">q = 1：椭圆</text><text x="300" y="230">q = 0：直线</text><text x="505" y="230">绿为正，橙为负</text></g></svg><figcaption>横轴为 \(s\)，纵轴为 \(t\)。图中斜向直线满足 \(s=-3t\)；\(y=20\) 的椭圆形状按主轴方向与约 11:1 的轴长比绘制。</figcaption></figure>

<h2 id="hessian">三、Hessian 何时能判定极小值？</h2>
<p>对二次函数 \(f(x)=x^TAx+\ell^Tx+c\)，其中 \(A=A^T\)、\(\ell\) 是给定向量，梯度是 \(\nabla f(x)=2Ax+\ell\)，二阶导数组成的矩阵叫 <strong>Hessian</strong>，这里恒为 \(H=2A\)。若 \(H\) 正定，驻点 \(x_*\) 满足 \(2Ax_*+\ell=0\)，且</p>
<div class="equation">\[f(x_*+h)-f(x_*)=h^TAh\gt0\quad(h\ne0).\]</div>
<p>所以它是<strong>唯一的全局极小点</strong>。对一般二阶连续可导函数，驻点附近只有展开式 \(f(x_*+h)-f(x_*)=\tfrac12h^TH(x_*)h+o(\|h\|^2)\)：Hessian 正定时主导项为正，保证严格<strong>局部</strong>极小，却不保证全局极小。Hessian 半正定时还需看更高阶项；例如 \(t^4\) 与 \(-t^4\) 在原点的 Hessian 都为零，但一个是极小、一个是极大。</p>

<h2 id="higher-dimensional">四、三维的“椭球轴长”怎么算？</h2>
<p>讲义给 \(T=\begin{pmatrix}2&-1&0\\-1&2&-1\\0&-1&2\end{pmatrix}\)。三个顺序前导子式是 \(2,3,4\)，主元是 \(2,3/2,4/3\)，故正定。讲义摘要有一处把第二个子式写成 5；直接算 \(\det\begin{pmatrix}2&-1\\-1&2\end{pmatrix}=3\)，与后续主元一致。</p>
<p>谱定理给 \(T=Q\Lambda Q^T\)，其特征值是 \(2-\sqrt2,2,2+\sqrt2\)。令 \(z=Q^Tx\)，等值面 \(x^TTx=1\) 就变成 \(\lambda_1z_1^2+\lambda_2z_2^2+\lambda_3z_3^2=1\)：主轴方向是 \(Q\) 的三列，半轴长度分别为 \(1/\sqrt{\lambda_1},1/\sqrt{\lambda_2},1/\sqrt{\lambda_3}\)。特征值越大，该方向的二次型增长越快、等值椭球的半轴反而越短。</p>
<h2 id="check">理解检查</h2>
<ol><li>把 \(A_y\) 中的 \(y\) 取为 18，找出全部零值向量；再取 17，给一个正值方向和一个负值方向。</li><li>习题 27.2 的矩阵 \(\begin{pmatrix}1&5\\7&9\end{pmatrix}\) 不对称。先算 \(x^TAx\)，再找出决定它的对称矩阵 \((A+A^T)/2\)。</li><li>核对 \(T\) 的三个前导子式，解释为什么第二、第三主元分别是 \(3/2,4/3\)。</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit III, Session 3 <em>Positive Definite Matrices and Minima</em> 整理。更正三阶例子的第二前导子式，并区分局部与全局极小。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
