---
title: 第二单元 9：对角化、矩阵幂与递推｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-2-session-09/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit2">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-08/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-2-session-10/">下一讲 →</a></p>
<div class="note-lead"><p>上讲找到矩阵作用时不会转向的向量。本讲问：若这种方向足够多，能否把<strong>每个输入</strong>都拆成这些方向，再分别乘上特征值？对角化正是这样一次换坐标，它把 \(A^k\) 与递推序列变得可计算。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-2-session-9-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-2-session-9-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-2-session-9-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="diagonalization">一、矩阵等式从每一列怎样长出来？</h2>
<p>设 \(A\) 为 \(n\times n\) 方阵，找到了 \(n\) 个线性无关的特征向量 \(v_1,\ldots,v_n\)，满足 \(Av_j=\lambda_jv_j\)。把它们按列排成 \(S=[v_1\ \cdots\ v_n]\)，把对应的特征值放在对角线组成 \(\Lambda=\operatorname{diag}(\lambda_1,\ldots,\lambda_n)\)。逐列看，\(AS\) 的第 \(j\) 列是 \(Av_j=\lambda_jv_j\)，而 \(S\Lambda\) 的第 \(j\) 列也是 \(\lambda_jv_j\)，所以 \(AS=S\Lambda\)。</p>
<p>因为列无关，\(S\) 可逆，故</p>
<div class="equation">\[S^{-1}AS=\Lambda,\qquad A=S\Lambda S^{-1}.\]</div>
<p>\(S^{-1}\) 先把原坐标的输入拆成特征向量的系数，\(\Lambda\) 分别缩放这些系数，\(S\) 再把结果拼回原坐标。这不是把原矩阵“直接变成”一个对角矩阵；换了坐标系，才看见这种简单形式。</p>
<p>不同特征值对应的特征向量必线性无关，所以 \(n\) 个互异特征值足以保证可对角化；它不是必要条件。单位矩阵只有特征值 1，却有一整组特征基。相反，上讲的 \(\begin{pmatrix}3&1\\0&3\end{pmatrix}\) 只有一个独立特征方向，无法组成可逆的 \(S\)。</p>

<h2 id="powers">二、反复作用时，为什么只需给特征值乘方？</h2>
<p>将 \(A=S\Lambda S^{-1}\) 连乘两次，中间的 \(S^{-1}S=I\) 抵消，得 \(A^2=S\Lambda^2S^{-1}\)；重复可得 \(A^k=S\Lambda^kS^{-1}\)。由于 \(\Lambda\) 是对角矩阵，\(\Lambda^k\) 只需把各 \(\lambda_j\) 分别变成 \(\lambda_j^k\)。</p>
<p>用上一讲 \(A=\begin{pmatrix}3&1\\1&3\end{pmatrix}\) 验算：取 \(S=\begin{pmatrix}1&1\\1&-1\end{pmatrix}\)，\(S^{-1}=\frac12S\)，\(\Lambda=\operatorname{diag}(4,2)\)。因此</p>
<div class="equation">\[A^k=\frac12\begin{pmatrix}4^k+2^k&4^k-2^k\\4^k-2^k&4^k+2^k\end{pmatrix}.\]</div>
<p>代 \(k=1\) 得原矩阵，代 \(k=0\) 得单位矩阵。如果所有 \(|\lambda_j|<1\)，每个 \(\lambda_j^k\to0\)，故 \(A^k\to0\)；反过来，只要有某个特征值的模不小于 1，沿其特征向量的输入就不会趋零。这里是在可对角化的前提下直接由公式证明；一般矩阵也有相同的最终判据，但需处理不可对角化情形。</p>
<p>要注意：\(A\) 的特征向量仍是 \(A^k\) 的特征向量，但 \(A^k\) 可能出现<strong>更多</strong>特征向量。例如 \(A=\operatorname{diag}(1,-1)\) 只有两条特征直线，而 \(A^2=I\) 的任意非零向量都是特征向量。所以“幂的特征向量完全不变”说得过强。</p>

<h2 id="difference-equation">三、把递推式写成矩阵幂</h2>
<p>若状态按 \(u_{k+1}=Au_k\) 更新，给定初始列向量 \(u_0\)，则 \(u_k=A^ku_0\)。若 \(A\) 可对角化，把初始状态唯一写成 \(u_0=c_1v_1+\cdots+c_nv_n\)，便得到</p>
<div class="equation">\[u_k=c_1\lambda_1^kv_1+\cdots+c_n\lambda_n^kv_n.\]</div>
<p>每个特征方向是一种独立的“模式”。\(|\lambda_j|<1\) 的模式衰减；\(\lambda_j=1\) 的模式保持；\(\lambda_j=-1\) 的模式每步翻向；\(|\lambda_j|>1\) 的模式增长。实际初始状态中若某一模式系数 \(c_j=0\)，那个模式就不会出现，因此不能只看矩阵的最大特征值就断言<strong>每条</strong>轨迹的行为。</p>

<h2 id="fibonacci">四、Fibonacci 为何也是一个二维系统？</h2>
<p>数列 \(F_0=0,F_1=1,F_{k+2}=F_{k+1}+F_k\) 看似是二阶递推；把相邻两项放进状态 \(u_k=(F_{k+1},F_k)^T\)，则</p>
<div class="equation">\[u_{k+1}=\begin{pmatrix}1&1\\1&0\end{pmatrix}u_k,\qquad u_0=\begin{pmatrix}1\\0\end{pmatrix}.\]</div>
<p>记这个矩阵为 \(B\)。特征方程 \(\lambda^2-\lambda-1=0\) 给出 \(\phi=(1+\sqrt5)/2\)、\(\psi=(1-\sqrt5)/2\)，对应的特征向量可取 \((\phi,1)^T,(\psi,1)^T\)。解 \(u_0=c_1(\phi,1)^T+c_2(\psi,1)^T\)：第二分量给 \(c_2=-c_1\)，第一分量给 \(c_1(\phi-\psi)=1\)，故 \(c_1=1/\sqrt5\)。于是状态的第二分量恰为</p>
<div class="equation">\[F_k=\frac{\phi^k-\psi^k}{\sqrt5}.\]</div>
<p>这不是凭空猜出的通项：两个指数分别来自矩阵的两个特征值，系数来自初值。由于 \(|\psi|<1<\phi\)，\(F_k\) 的长期增长由 \(\phi^k/\sqrt5\) 主导；但精确等式仍须保留第二项。</p>
<h2 id="check">理解检查</h2>
<ol><li>在上面的 \(A^k\) 公式中取 \(k=2\)，与直接相乘 \(A^2\) 核对。</li><li>习题 22.1 的 \(A=\begin{pmatrix}4&0\\1&2\end{pmatrix}\)。先找两条特征直线，再说明改变特征向量的非零倍数会怎样改变 \(S\)，而 \(A\) 是否改变。</li><li>从 \(F_k\) 的通项直接验证 \(F_0=0,F_1=1\)，再解释为什么 \(\phi,\psi\) 都满足同一个递推关系。</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit II, Session 9 <em>Diagonalization and Powers of A</em> 整理。对“幂的特征向量不变”及长期行为的适用条件作了限定。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
