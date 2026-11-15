---
title: 第十二节：图、网络与关联矩阵｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/session-12/
mathjax: true
toc_number: false
comments: false
---

<style>
.post-body .la-notes--session12 { max-width: 700px; margin: 0 auto; font-size: 17px; line-height: 1.85; }
.post-body .la-notes--session12 p { margin: 0 0 1em; text-indent: 0; }
.post-body .la-notes--session12 h2 { margin: 2.2em 0 .8em; padding-bottom: .35em; border-bottom: 1px solid var(--line, #dce5e3); font-size: 25px; }
.post-body .la-notes--session12 h3 { margin: 1.6em 0 .6em; font-size: 20px; }
.post-body .la-notes--session12 [id] { scroll-margin-top: 25px; }
.post-body .la-notes--session12 .note-nav, .post-body .la-notes--session12 .reading-links, .post-body .la-notes--session12 .note-footer { font: 14px/1.8 var(--sans, sans-serif); }
.post-body .la-notes--session12 .note-lead { border-left: 3px solid var(--accent, #729692); padding-left: 16px; margin: 1.5em 0; }
.post-body .la-notes--session12 .reading-links { display: flex; flex-wrap: wrap; gap: 8px 20px; border-block: 1px solid var(--line, #dce5e3); padding: 13px 0; margin: 1.5em 0 2em; }
.post-body .la-notes--session12 .equation { overflow-x: auto; overflow-y: hidden; padding: 7px 0; margin: .8em 0 1.1em; }
.post-body .la-notes--session12 .equation .MathJax_Display { margin: .25em 0 !important; }
.post-body .la-notes--session12 ol { padding-left: 1.4em; }
.post-body .la-notes--session12 li { margin-bottom: .65em; }
.post-body .la-notes--session12 .network-figure { margin: 1.5em 0; }
.post-body .la-notes--session12 .network-figure svg { display: block; width: 100%; height: auto; background: var(--surface, #f4f6f4); border: 1px solid var(--line, #dce5e3); border-radius: 5px; }
.post-body .la-notes--session12 .network-figure figcaption { margin-top: 8px; color: var(--muted, #687474); font: 13px/1.6 var(--sans, sans-serif); }
.post-body .la-notes--session12 .network-edge { fill: none; stroke: var(--accent, #527e78); stroke-width: 2.8; }
.post-body .la-notes--session12 .network-node { fill: var(--paper, #fff); stroke: var(--ink, #253c3c); stroke-width: 2; }
.post-body .la-notes--session12 .network-node-label { fill: var(--ink, #253c3c); font: 19px var(--sans, sans-serif); text-anchor: middle; dominant-baseline: middle; }
.post-body .la-notes--session12 .network-edge-label { fill: var(--ink, #253c3c); font: 15px var(--sans, sans-serif); text-anchor: middle; paint-order: stroke; stroke: var(--surface, #f4f6f4); stroke-width: 6px; stroke-linejoin: round; }
.post-body .la-notes--session12 .note-aside { margin: 1.4em 0; padding: 14px 18px; border-left: 2px solid var(--accent, #729692); background: var(--surface, #f4f6f4); }
.post-body .la-notes--session12 .note-aside p:last-child { margin-bottom: 0; }
.post-body .la-notes--session12 details.note-extension { margin: 1.2em 0; padding: 12px 17px; border: 1px solid var(--line, #dce5e3); }
.post-body .la-notes--session12 details.note-extension summary { cursor: pointer; font-weight: 600; }
.post-body .la-notes--session12 details.note-extension[open] summary { margin-bottom: 1em; }
.post-body .la-notes--session12 .note-footer { border-top: 1px solid var(--line, #dce5e3); padding-top: 18px; margin-top: 3em; color: var(--muted, #687474); }
@media (max-width: 600px) { .post-body .la-notes--session12 { font-size: 16px; } .post-body .la-notes--session12 h2 { font-size: 22px; } }
</style>

<section class="la-notes la-notes--session12">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-11/">第十一节</a></p>
<div class="note-lead"><p>上一节用<strong>邻接矩阵</strong>回答“两个顶点是否直接相连”，并用它的幂数行走。本节换一个问题：如果每条边有方向和流量，哪些边上的数能由节点的电势差产生？哪些流量能在每个节点守恒？关联矩阵把这两个问题连到零空间、秩和左零空间。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-1-session-12-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-1-session-12-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-1-session-12-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="graph-to-matrix">一、从边的方向构造矩阵</h2>
<p>沿用讲义的四个节点、五条边，给每条边任选一个<strong>参考方向</strong>。箭头只规定正负号；若电流算出负值，实际方向就与箭头相反。下图把五条边依次编号。</p>
<figure class="network-figure"><svg viewBox="0 0 640 320" role="img" aria-labelledby="network-title network-desc"><title id="network-title">四节点五条有向边的网络</title><desc id="network-desc">边 1 从 1 到 2，边 2 从 2 到 3，边 3 从 1 到 3，边 4 从 1 到 4，边 5 从 3 到 4。节点 1 在上方，节点 2、3、4 从左到右排列。</desc><defs><marker id="session12-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0L10 5L0 10Z" fill="#527e78"/></marker></defs><g class="network-edge" marker-end="url(#session12-arrow)"><path d="M300 72L116 209"/><path d="M128 230L292 230"/><path d="M320 82L320 202"/><path d="M340 72L524 209"/><path d="M348 230L512 230"/></g><g class="network-edge-label"><text x="180" y="126">边 1</text><text x="210" y="214">边 2</text><text x="346" y="150">边 3</text><text x="460" y="126">边 4</text><text x="430" y="214">边 5</text></g><g class="network-node"><circle cx="320" cy="55" r="25"/><circle cx="100" cy="230" r="25"/><circle cx="320" cy="230" r="25"/><circle cx="540" cy="230" r="25"/></g><g class="network-node-label"><text x="320" y="55">1</text><text x="100" y="230">2</text><text x="320" y="230">3</text><text x="540" y="230">4</text></g></svg><figcaption>按讲义的关联矩阵重画：边 1 为 \(1\to2\)，边 2 为 \(2\to3\)，边 3 为 \(1\to3\)，边 4 为 \(1\to4\)，边 5 为 \(3\to4\)。</figcaption></figure>
<p>我们需要一个矩阵把<strong>节点上的数</strong>变成<strong>边上的差</strong>。输入有四个位置，输出有五个位置，因此关联矩阵 \(A\) 是 \(5\times4\)：第 \(k\) 行对应第 \(k\) 条边，四列对应节点 1 到 4。若一条边从节点 \(i\) 指向节点 \(j\)，就在该行第 \(i\) 列填 \(-1\)、第 \(j\) 列填 \(1\)，其余填 0。该行乘节点数列时，就会得到“终点减起点”。</p>
<p>例如边 1 从节点 1 指向节点 2，给出第一行 \((-1,1,0,0)\)；边 5 从节点 3 指向节点 4，给出第五行 \((0,0,-1,1)\)。逐边填写得到</p>
<div class="equation">\[A=\begin{pmatrix}-1&1&0&0\\0&-1&1&0\\-1&0&1&0\\-1&0&0&1\\0&0&-1&1\end{pmatrix}.\]</div>
<p>这与上一节的邻接矩阵不同：邻接矩阵按“节点与节点”排，是方阵；关联矩阵按“边与节点”排，可以不是方阵，而且记录了参考方向。把某条边的箭头翻转，只会使 \(A\) 中相应的一行乘以 \(-1\)。</p>

<h2 id="potential-differences">二、\(Ax\) 在每条边上算出了什么？</h2>
<p>令 \(x=(x_1,x_2,x_3,x_4)^T\)，其中 \(x_i\) 是节点 \(i\) 的电势。第一行乘 \(x\) 给出 \(x_2-x_1\)，第五行给出 \(x_4-x_3\)。五行一起算，</p>
<div class="equation">\[Ax=\begin{pmatrix}x_2-x_1\\x_3-x_2\\x_3-x_1\\x_4-x_1\\x_4-x_3\end{pmatrix}=:e.\]</div>
<p>\(e\) 是<strong>五条边上的端点差组成的列向量</strong>，\(e_k\) 是第 \(k\) 条边上的差值。例如 \(x=(0,2,5,1)^T\) 时，\(e=(2,3,5,1,-4)^T\)。最后的 \(-4\) 表示边 5 的终点电势比起点低 4。</p>
<p>五个差不能任意指定：沿 \(1\to2\to3\) 走，总差必须与直接沿 \(1\to3\) 走相同，故 \(e_1+e_2=e_3\)。同理，沿 \(1\to3\to4\) 走，得到 \(e_3+e_5=e_4\)。上面的数满足 \(2+3=5\) 和 \(5+(-4)=1\)。<strong>回路上的端点差按方向相加必为零</strong>，因为沿途的中间电势两两抵消。</p>
<p>反过来，若五个给定的数满足这两条关系，就可以令 \(x_1=0,\ x_2=e_1,\ x_3=e_3,\ x_4=e_4\)；另外两条边的差会自动吻合。因此列空间 \(C(A)\) 恰好由满足这两条回路关系的边差组成，而不只是包含于其中。</p>

<h2 id="nullspace">三、为什么常数电势构成零空间？</h2>
<p>\(Ax=0\) 要求每条边两端的电势相同。由于图连通，沿边从节点 1 走到任何节点，等式会一路传递，得到 \(x_1=x_2=x_3=x_4\)。因此</p>
<div class="equation">\[N(A)=\operatorname{span}\{(1,1,1,1)^T\},\qquad \dim N(A)=1.\]</div>
<p>仅从电势差无法确定电势的<strong>共同零点</strong>：给所有节点同时加 7，五条边的差完全不变。规定某一节点电势为 0，称为选定参考点或接地，才消除这一个自由度。由秩与零度公式，\(A\) 有四列、零度为 1，所以 \(\operatorname{rank}(A)=4-1=3\)。</p>
<p>图上也能看见这三个独立方向：只保留边 1、边 2、边 4，四个节点仍连通，而且没有回路；这三条边构成一棵<strong>生成树</strong>。如果已知 \(x_1\)，沿这棵树上的三个边差便能依次恢复其余电势。加回边 3 和边 5 时，只会产生两条必须满足的回路关系。</p>
<details class="note-extension"><summary>如果图不连通会怎样？</summary><p>每个连通分量内仍须取相同电势，不同分量却可以各选一个常数。若有 \(c\) 个连通分量、\(n\) 个节点，则 \(\dim N(A)=c\)、\(\operatorname{rank}(A)=n-c\)。本讲的图连通，所以 \(c=1\)。</p></details>

<h2 id="cycles">四、左零空间为什么装着回路？</h2>
<p>第一条回路关系 \(e_1+e_2-e_3=0\) 对<strong>每一个</strong> \(e=Ax\) 都成立。把三个系数排成五维列 \(z_1=(1,1,-1,0,0)^T\)，就是 \(z_1^TAx=0\) 对所有 \(x\) 成立，因而 \(A^Tz_1=0\)。直接看 \(A\) 的行也一样：第三行是第一行加第二行。</p>
<p>第二个回路 \(1\to3\to4\to1\) 给出 \(e_3-e_4+e_5=0\)，对应 \(z_2=(0,0,1,-1,1)^T\)。所以</p>
<div class="equation">\[N(A^T)=\operatorname{span}\left\{\begin{pmatrix}1\\1\\-1\\0\\0\end{pmatrix},\begin{pmatrix}0\\0\\1\\-1\\1\end{pmatrix}\right\}.\]</div>
<p>为什么这两列就够？\(A^T\) 有五列，秩为 3，因此零度是 \(5-3=2\)；\(z_1,z_2\) 又线性无关。外侧大回路 \(1\to2\to3\to4\to1\) 也给一个关系，但它是前两条关系的和。因此图上有三个简单回路，<strong>独立回路却只有两个</strong>。这也把第十节的正交关系变得具体：\(C(A)\) 中的边差与每个回路向量正交。</p>

<h2 id="current">五、转置矩阵怎样表达节点守恒？</h2>
<p>现在令 \(y=(y_1,\ldots,y_5)^T\) 表示五条边上的<strong>带符号电流</strong>，正号表示沿图中箭头流。\(A^Ty\) 的四个分量是四个节点从边得到的净流入量：边起点在 \(A\) 中为 \(-1\)，终点为 \(1\)。逐个节点计算，</p>
<div class="equation">\[A^Ty=\begin{pmatrix}-y_1-y_3-y_4\\y_1-y_2\\y_2+y_3-y_5\\y_4+y_5\end{pmatrix}.\]</div>
<p>例如第二个分量 \(y_1-y_2\) 表示流入节点 2 的 \(y_1\)，减去从节点 2 流出的 \(y_2\)。没有外部注入或抽取时，每个节点净流入为零，即 Kirchhoff's Current Law：\(A^Ty=0\)。所以<strong>同一个左零空间</strong>也可理解为满足节点守恒的环流。\(z_1\) 表示沿 \(1\to2\to3\to1\) 流一单位；第三条边上的 \(-1\) 表示逆着它的参考箭头流。</p>
<p>不过，满足守恒的环流未必是某个实际电路在给定条件下的电流。还需要边上的欧姆定律。若第 \(k\) 条边的电导 \(c_k>0\)，令 \(C=\operatorname{diag}(c_1,\ldots,c_5)\)。本页把 \(e=Ax\) 定为“终点电势减起点电势”，正电流取箭头方向，所以 \(y=-Ce\)。若 \(s\) 表示各节点<strong>从外部注入的净电流</strong>（注入为正），节点平衡是 \(A^Ty=-s\)。于是</p>
<div class="equation">\[e=Ax,\qquad y=-Ce,\qquad A^Ty=-s\quad\Longrightarrow\quad A^TCAx=s.\]</div>
<p>讲义和习题把右端记作 \(f\)。不同资料可能给“电势差”或“流入”选相反的正向；只要从头到尾使用同一约定，最终电路关系一致。翻转一条边的参考箭头时，\(A\) 中相应一行、该边的 \(e_k\) 和 \(y_k\) 都换号，实际电路不变。</p>
<div class="note-aside"><p>这里有两个不同的条件：\(e=Ax\) 使<strong>电势差沿回路之和</strong>为零；\(A^Ty=0\) 使<strong>电流在每个节点</strong>守恒。前者说哪些边差可能出现，后者说哪些流量可以闭合。</p></div>

<h2 id="structure">六、维数公式里藏着什么结构？</h2>
<p>这张图有 \(n=4\) 个节点、\(m=5\) 条边。连通性给 \(\operatorname{rank}(A)=n-1=3\)，左零空间维数是 \(m-(n-1)=2\)，也就是独立回路数。因此</p>
<div class="equation">\[\underbrace{n}_{\text{节点数}}-\underbrace{m}_{\text{边数}}+\underbrace{\dim N(A^T)}_{\text{独立回路数}}=1.\]</div>
<p>这就是讲义所说的连通图的欧拉关系。右边的 1 来自<strong>只有一个连通分量</strong>；若有 \(c\) 个分量，右边应为 \(c\)。它数的是相互独立的回路，不是画面上肉眼可数的简单回路总数。</p>
<p>电路方程中的 \(L=A^TCA\) 叫加权图拉普拉斯矩阵。对任意电势列 \(x\)，</p>
<div class="equation">\[x^TLx=(Ax)^TC(Ax)=\sum_{k=1}^{5}c_k e_k^2\ge0.\]</div>
<p>每一项都是一条边的电势差平方乘正电导。若 \(Lx=0\)，则上式为零；因 \(c_k>0\)，所有边差都为零，图又连通，所以 \(x\) 只能是常数列。这再次解释电势解为何只确定到一个共同常数。还有 \(A(1,1,1,1)^T=0\)，所以 \(Lx=s\) 必须满足 \(s_1+s_2+s_3+s_4=0\)：总注入等于总抽出。反过来，\(L\) 的核是一维、秩为 3；它的像空间恰是四维空间中“分量和为零”的三维子空间，因此这个条件也足以保证有解。</p>

<h2 id="check">理解检查与习题入口</h2>
<ol><li>从图上读出边 4（\(1\to4\)）对应 \(A\) 的哪一行，并算出它作用于 \(x=(0,2,5,1)^T\) 的结果。</li><li>若把边 3 的箭头反过来，\(A\) 的第三行、差值 \(e_3\) 和两条回路关系怎样变化？哪些结论不变？</li><li>不用消元，说明为什么 \((1,0,0,0)^T\) 不在 \(A\) 的行空间：\(A\) 的每行元素之和是多少？这也是讲义习题 12.1 的关键。</li><li>习题 12.2 换用四边形网络和给定电导，要求算 \(A^TCA\) 及电势、电流。做题时先核对边的方向，再核对每个节点的平衡式。</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC 第 1 单元第 12 讲 <em>Graphs, Networks, Incidence Matrices</em> 及其习题整理。四节点五边网络、关联矩阵、零空间、回路与电流守恒来自讲义；具体数值例子、列空间的逆向构造、符号约定与图拉普拉斯矩阵的能量推导为本笔记补充。<a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a>。</p>
</section>
