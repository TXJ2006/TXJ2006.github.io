---
title: 第十一节：矩阵空间、秩一矩阵与图｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/session-11/
mathjax: true
toc_number: false
comments: false
---

<style>
.post-body .la-notes--session11 { max-width: 700px; margin: 0 auto; font-size: 17px; line-height: 1.85; }
.post-body .la-notes--session11 p { margin: 0 0 1em; text-indent: 0; }
.post-body .la-notes--session11 h2 { margin: 2.3em 0 .8em; padding-bottom: .35em; border-bottom: 1px solid var(--line, #dce5e3); font-size: 25px; }
.post-body .la-notes--session11 h3 { margin: 1.6em 0 .6em; font-size: 20px; }
.post-body .la-notes--session11 [id] { scroll-margin-top: 25px; }
.post-body .la-notes--session11 .note-nav, .post-body .la-notes--session11 .reading-links, .post-body .la-notes--session11 .note-footer { font: 14px/1.8 var(--sans, sans-serif); }
.post-body .la-notes--session11 .note-lead { border-left: 3px solid var(--accent, #729692); padding-left: 16px; margin: 1.5em 0; }
.post-body .la-notes--session11 .reading-links { display: flex; flex-wrap: wrap; gap: 8px 20px; border-block: 1px solid var(--line, #dce5e3); padding: 13px 0; margin: 1.5em 0 2em; }
.post-body .la-notes--session11 .equation { overflow-x: auto; overflow-y: hidden; padding: 7px 0; margin: .8em 0 1.1em; }
.post-body .la-notes--session11 .equation .MathJax_Display { margin: .25em 0 !important; }
.post-body .la-notes--session11 ul, .post-body .la-notes--session11 ol { padding-left: 1.4em; }
.post-body .la-notes--session11 li { margin-bottom: .6em; }
.post-body .la-notes--session11 .graph-figure { margin: 1.5em 0; }
.post-body .la-notes--session11 .graph-figure svg { width: 100%; height: auto; display: block; background: var(--surface, #f4f6f4); border: 1px solid var(--line, #dce5e3); border-radius: 5px; }
.post-body .la-notes--session11 .graph-figure figcaption { margin-top: 8px; color: var(--muted, #687474); font: 13px/1.6 var(--sans, sans-serif); }
.post-body .la-notes--session11 .graph-edge { stroke: var(--accent, #557c76); stroke-width: 2.5; }
.post-body .la-notes--session11 .graph-node { fill: var(--paper, #fff); stroke: var(--ink, #253c3c); stroke-width: 2; }
.post-body .la-notes--session11 .graph-label { fill: var(--ink, #253c3c); font: 18px var(--sans, sans-serif); text-anchor: middle; dominant-baseline: middle; }
.post-body .la-notes--session11 .note-footer { border-top: 1px solid var(--line, #dce5e3); padding-top: 18px; margin-top: 3em; color: var(--muted, #687474); }
@media (max-width: 600px) { .post-body .la-notes--session11 { font-size: 16px; } .post-body .la-notes--session11 h2 { font-size: 22px; } .post-body .la-notes--session11 h3 { font-size: 18px; } }
</style>

<section class="la-notes la-notes--session11">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-10/">第十节</a></p>
<div class="note-lead"><p>第十节把 \(3\times3\) 矩阵本身看成向量，求出了上三角、对称、对角矩阵空间的基。现在继续问：<strong>两个子空间怎样合成一个更大的空间？固定秩的矩阵能否组成子空间？矩阵又怎样记录一个图？</strong>三个问题都要求我们先认清矩阵在当下扮演的是“空间中的元素”，还是“作用于其他对象的线性映射”。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-1-session-11-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-1-session-11-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-1-session-11-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="matrix-space">一、矩阵空间：对象变了，向量空间的定义没变</h2>
<p>记 \(M_{m,n}(\mathbb R)\) 为全体 \(m\times n\) 实矩阵。逐项相加、逐项数乘以及零矩阵满足向量空间公理。这里“向量”指空间中的元素；它可以是一张矩阵，不必是一列数。令 \(E_{ij}\) 仅在第 \(i\) 行、第 \(j\) 列为 1，其余位置为 0，则</p>
<div class="equation">\[X=(x_{ij})=\sum_{i=1}^{m}\sum_{j=1}^{n}x_{ij}E_{ij}.\]</div>
<p>逐个位置比较可知表示唯一。因此 \(mn\) 个 \(E_{ij}\) 构成基，\(\dim M_{m,n}(\mathbb R)=mn\)。这解释了它为何与 \(\mathbb R^{mn}\) <em>线性同构</em>：把矩阵按固定顺序摊成一列数，会得到保持加法、数乘的双向对应。两者的元素写法不同，线性结构相同；<strong>矩阵乘法并不是这个向量空间结构的一部分</strong>。</p>
<p>对 \(M=M_{3,3}(\mathbb R)\)，第十节定义了对称矩阵空间 \(S=\{X:X^T=X\}\)、上三角矩阵空间 \(U=\{X:x_{ij}=0\text{ 当 }i>j\}\) 与对角矩阵空间 \(D=S\cap U\)。它们都是子空间，且 \(\dim S=6,\dim U=6,\dim D=3\)。例如对称矩阵的基可以取三个 \(E_{ii}\) 和三个 \(E_{ij}+E_{ji}\ (i<j)\)；上三角矩阵的基则取六个 \(E_{ij}\ (i\le j)\)。两个空间同为六维，却不是同一个空间。</p>

<h3 id="sum-and-union">为什么要取“和”，而不是“并”？</h3>
<p>并集 \(S\cup U\) 只允许矩阵属于其中一个空间，通常不对加法封闭。取 \(F=E_{12}+E_{21}\in S\) 和 \(G=E_{23}\in U\)：\(F+G\) 有下三角位置 \((2,1)\) 的非零元，故不在 \(U\)；又缺少与 \((2,3)\) 对称的 \((3,2)\) 元，故不在 \(S\)。这就给出一个可核验的反例。</p>
<p>要把二者的元素自由相加，定义<strong>子空间的和</strong> \(S+U=\{F+G:F\in S,\ G\in U\}\)。它在加法和数乘下封闭。这里甚至有 \(S+U=M\)：任给矩阵 \(X=(x_{ij})\)，先取一个对称矩阵 \(F\)，令其下三角及对角线与 \(X\) 相同，并把下三角镜像到上三角；于是 \(G=X-F\) 在下三角及对角线上全为零，必属 \(U\)。因此 \(X=F+G\)。</p>
<p>这种分解并不唯一：任意 \(H\in S\cap U=D\) 都能从 \(F\) 移到 \(G\)，因为 \((F+H)+(G-H)=X\)。重复计算的正是交空间的三个自由度，所以</p>
<div class="equation">\[\dim(S+U)=\dim S+\dim U-\dim(S\cap U)=6+6-3=9.\]</div>
<p>更一般地，把 \((F,G)\) 送到 \(F+G\) 是线性映射；它的核恰为 \(\{(H,-H):H\in S\cap U\}\)。对这张映射用前面学过的秩与零度关系，就得到上面的维数公式。<strong>讲义末行把 \(\dim(S\cup U)\) 写进公式，应为 \(\dim(S+U)\)</strong>；并集不是向量空间时，根本没有这里所说的维数。</p>

<h2 id="functions">二、函数也是向量：线性方程的解空间</h2>
<p>矩阵不是唯一“长得不像箭头”的向量。定义 \(V=C^2(\mathbb R,\mathbb R)\)，元素是二阶连续可导的实函数。对微分算子 \(L:V\to C(\mathbb R,\mathbb R)\)，令 \(L(y)=y''+y\)。求方程 \(y''+y=0\)，就是求线性映射 \(L\) 的零空间。</p>
<p>因为求导是线性的，\(L(ay+bz)=aL(y)+bL(z)\)。\(\sin x\) 与 \(\cos x\) 都在核中；若 \(a\cos x+b\sin x=0\)，代入 \(x=0\) 并在求导后再代入 \(x=0\)，依次得到 \(a=b=0\)，故它们无关。还要证明没有别的解：给任意解 \(y\)，令 \(z=y-y(0)\cos x-y'(0)\sin x\)，则 \(z''+z=0\) 且 \(z(0)=z'(0)=0\)。此时 \((z'^2+z^2)'=2z'(z''+z)=0\)，其初值为零，故 \(z'^2+z^2\) 处处为零，\(z=0\)。因此</p>
<div class="equation">\[N(L)=\{c_1\cos x+c_2\sin x:c_1,c_2\in\mathbb R\},\qquad \dim N(L)=2.\]</div>
<p>讲义还列出 \(e^{ix}=\cos x+i\sin x\)。它是<strong>复值</strong>解；若讨论实函数空间，就不属于上述 \(V\)。若改在复值函数空间工作，则 \(\sin x,\cos x\) 仍可作复数域上的一组基。写清底层数域，才知道“线性组合”的系数允许取什么数。</p>

<h2 id="rank-one">三、秩一矩阵：所有列共用一个方向</h2>
<p>设 \(A\in M_{m,n}(\mathbb R)\) 非零。秩为 1 的定义是 \(\dim C(A)=1\)：它的每一列都沿同一条过原点的直线。选一个非零列 \(u\in\mathbb R^m\)，把各列相对于 \(u\) 的倍数记为 \(v_1,\ldots,v_n\)，再令 \(v=(v_1,\ldots,v_n)^T\in\mathbb R^n\)，便得到</p>
<div class="equation">\[A=[v_1u\ \cdots\ v_nu]=uv^T,\qquad (uv^T)_{ij}=u_i v_j.\]</div>
<p>反过来，若 \(u\ne0\)、\(v\ne0\)，则 \(uv^T\) 至少有一列非零，且每列都是 \(u\) 的倍数，故秩恰为 1。这证明了“非零矩阵秩一”与“可以写为两个非零列向量的外积”等价。讲义中的例子是</p>
<div class="equation">\[\begin{pmatrix}1&4&5\\2&8&10\end{pmatrix}=\begin{pmatrix}1\\2\end{pmatrix}\begin{pmatrix}1&4&5\end{pmatrix}.\]</div>
<p>作为线性映射，它把 \(x\in\mathbb R^n\) 送到 \(u(v^Tx)\)：先用 \(v^Tx\) 把输入压成一个数，再沿 \(u\) 的方向输出。因此像空间是 \(\operatorname{span}(u)\)，零空间是 \(\{x:v^Tx=0\}\)，维数分别为 1 与 \(n-1\)。这是“秩一”的几何含义，而不只是把表格拆成两块。</p>
<p>外积分解有比例自由度：对 \(t\ne0\)，\((tu)(v/t)^T=uv^T\)。又不能把“秩一矩阵的集合”当作一个一维向量空间：它不含零矩阵；即使加上零，\(E_{11}+E_{22}\) 也是两个秩一矩阵之和，却秩为 2。<strong>矩阵的秩衡量它的列空间；矩阵空间的维数衡量描述整个集合需要几个基元素。</strong></p>
<p>秩一矩阵仍是有用的积木。若 \(A\) 的秩为 \(r\)，从它的列空间选基 \(u_1,\ldots,u_r\)，把每列在这组基下的系数排成矩阵，就有 \(A=UV^T=\sum_{k=1}^r u_kv_k^T\)，其中 \(U\) 的列为 \(u_k\)，\(V^T\) 的行为 \(v_k^T\)。少于 \(r\) 个秩一项不可能组成 \(A\)，因为这些项的列空间至多由相应的 \(u_k\) 张成，维数不超过项数。</p>

<h2 id="matrix-map">四、习题 11.2：当矩阵成为线性映射的输入</h2>
<p>讲义习题固定 \(A\in M_{3,3}(\mathbb R)\)，研究 \(T:M_{3,3}(\mathbb R)\to M_{3,3}(\mathbb R),\ T(X)=AX\)，其中</p>
<div class="equation">\[A=\begin{pmatrix}1&0&-1\\-1&1&0\\0&-1&1\end{pmatrix}.\]</div>
<p>这里有两个不同层次：\(A\) 本身是秩为 2 的 \(3\times3\) 矩阵；\(T\) 是以<strong>整个矩阵</strong>为输入、输出的线性映射。令 \(X=[x_1\ x_2\ x_3]\)，则 \(AX=[Ax_1\ Ax_2\ Ax_3]\)。矩阵乘法把同一个 \(A\) 分别作用在 \(X\) 的三列上，这使前几节的列空间与零空间可以直接复用。</p>
<p>由 \(A(1,1,1)^T=0\)，且前两列无关，得 \(N(A)=\operatorname{span}((1,1,1)^T)\)。故 \(AX=0\) 当且仅当 \(X\) 的每列都是这个向量的倍数：</p>
<div class="equation">\[N(T)=\left\{\begin{pmatrix}a&b&c\\a&b&c\\a&b&c\end{pmatrix}:a,b,c\in\mathbb R\right\},\qquad \dim N(T)=3.\]</div>
<p>每列有一个自由系数，共三列，因此是 3 维；这不是说 \(\dim N(A)=3\)，后者只有 1 维。再看输出。\((1,1,1)A=0\)，故 \(Ax_j\) 的三个分量之和总是零。反过来，对任意 \((p,q,-p-q)^T\)，直接取 \(x=(p,p+q,0)^T\)，就有 \(Ax=(p,q,-p-q)^T\)。所以 \(C(A)\) 恰是分量和为零的平面，\(T\) 的像为</p>
<div class="equation">\[\operatorname{im}T=\left\{\begin{pmatrix}a&b&c\\d&e&f\\-a-d&-b-e&-c-f\end{pmatrix}:a,b,c,d,e,f\in\mathbb R\right\},\qquad \dim\operatorname{im}T=6.\]</div>
<p>输入空间 \(M_{3,3}\) 有 9 维，于是 \(\dim N(T)+\dim\operatorname{im}T=3+6=9\)。从“每列独立受作用”看，正是把 \(A\) 的零度 \(3-2=1\) 和秩 2 各重复三次。<strong>不要把 \(T\) 的秩 6 与 \(A\) 的秩 2 混为一谈。</strong></p>

<h2 id="graphs">五、图：先给出关系，再用矩阵编码</h2>
<p>图由顶点集合 \(V\) 与边集合 \(E\) 组成。下面是按照讲义五顶点、六边的示意图重画的编号版本；连线表示有边，线段的物理长短不表示图论距离。</p>
<figure class="graph-figure"><svg viewBox="0 0 560 270" role="img" aria-labelledby="graph-title graph-desc"><title id="graph-title">五个顶点、六条边的图</title><desc id="graph-desc">边为 1-2、1-3、2-3、2-4、3-4、4-5。顶点 1 到顶点 5 的最短路长度为 3。</desc><g class="graph-edge" fill="none"><path d="M105 195L180 55L335 82L305 205L105 195M105 195L335 82M305 205L455 130"/></g><g class="graph-node"><circle cx="180" cy="55" r="22"/><circle cx="335" cy="82" r="22"/><circle cx="105" cy="195" r="22"/><circle cx="305" cy="205" r="22"/><circle cx="455" cy="130" r="22"/></g><g class="graph-label"><text x="180" y="55">1</text><text x="335" y="82">2</text><text x="105" y="195">3</text><text x="305" y="205">4</text><text x="455" y="130">5</text></g></svg><figcaption>顶点编号仅为把图写成矩阵；图形改绘自本讲讲义的示意图。</figcaption></figure>
<p>将边 \(\{1,2\},\{1,3\},\{2,3\},\{2,4\},\{3,4\},\{4,5\}\) 记入<strong>邻接矩阵</strong> \(B\)：有边时 \(b_{ij}=1\)，没有边时 \(b_{ij}=0\)。无向图使 \(B=B^T\)；没有自环使对角线为零。对上图，</p>
<div class="equation">\[B=\begin{pmatrix}0&1&1&0&0\\1&0&1&1&0\\1&1&0&1&0\\0&1&1&0&1\\0&0&0&1&0\end{pmatrix}.\]</div>
<p>图上的距离 \(d(i,j)\) 是从 \(i\) 走到 \(j\) 所需的最少边数，不是画面上的欧氏距离。上图从 1 到 5 没有一步或两步路径，但有 \(1\to2\to4\to5\) 和 \(1\to3\to4\to5\)，所以 \(d(1,5)=3\)。讲义在此只提出用矩阵研究距离；下面这一点是把该思路展开：矩阵乘法给出</p>
<div class="equation">\[(B^2)_{ij}=\sum_{\ell=1}^{5}b_{i\ell}b_{\ell j}.\]</div>
<p>每个非零乘积对应一个中间顶点 \(\ell\)，即一条两步行走。继续相乘时，\((B^k)_{ij}=\sum_{\ell}(B^{k-1})_{i\ell}b_{\ell j}\) 把每条 \(k-1\) 步行走接上一条末边，所以它数出从 \(i\) 到 \(j\) 的 \(k\) 步行走。于是 \((B^2)_{1,4}=2\)，对应共同邻居 2、3；\((B^3)_{1,5}=2\)，对应上面两条三步路。因元素非负且图连通，\(d(i,j)=\min\{k\ge1:(B^k)_{ij}>0\}\)（\(i\ne j\)）。行走可以重复顶点或边；最先出现的正数仍正确给出最短路。所谓 small world 关心的是网络中远端顶点之间的最短路是否也很短，而非图纸上是否画得紧凑。</p>

<h2 id="check">理解检查</h2>
<ol><li>为什么 \(S\cup U\) 不是子空间，\(S+U\) 却是？在 \(S+U\) 的表示中，为什么交空间会导致不唯一？</li><li>若 \(A=uv^T\ne0\)，给出它作为映射的像空间与零空间，并解释为什么乘以非零常数不会改变秩。</li><li>习题 11.2 中，\(A\) 的秩是 2，\(T:X\mapsto AX\) 的秩是 6。分别说出这两个数在数什么。</li><li>从图中的顶点 1 到 5 为什么最少走三条边？\((B^2)_{1,4}\) 为什么恰等于 2？</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC 第 1 单元第 11 讲 <em>Matrix Spaces; Rank 1; Small World Graphs</em> 及其习题整理。矩阵子空间、微分方程、秩一矩阵与五顶点图来自讲义；并集反例、习题 11.2 的逐列推导和邻接矩阵的行走计数为本笔记的展开。<a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a>。</p>
</section>
