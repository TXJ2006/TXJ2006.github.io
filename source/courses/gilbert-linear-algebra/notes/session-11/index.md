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
.post-body .la-notes--session11 .note-aside { margin: 1.4em 0; padding: 15px 18px; border-left: 2px solid var(--accent, #729692); background: var(--surface, #f4f6f4); }
.post-body .la-notes--session11 .note-aside p:last-child { margin-bottom: 0; }
.post-body .la-notes--session11 details.note-extension { margin: 1.2em 0; padding: 12px 17px; border: 1px solid var(--line, #dce5e3); }
.post-body .la-notes--session11 details.note-extension summary { cursor: pointer; font-weight: 600; }
.post-body .la-notes--session11 details.note-extension[open] summary { margin-bottom: 1em; }
.post-body .la-notes--session11 .note-footer { border-top: 1px solid var(--line, #dce5e3); padding-top: 18px; margin-top: 3em; color: var(--muted, #687474); }
@media (max-width: 600px) { .post-body .la-notes--session11 { font-size: 16px; } .post-body .la-notes--session11 h2 { font-size: 22px; } .post-body .la-notes--session11 h3 { font-size: 18px; } }
</style>

<section class="la-notes la-notes--session11">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-10/">第十节</a></p>
<div class="note-lead"><p>第十节第一次把<strong>矩阵本身</strong>当作向量，求出了上三角、对称和对角矩阵空间的基。本讲接着问：这些子空间怎样组合？秩一矩阵为何能作更复杂矩阵的积木？最后，怎样用矩阵记录一张图？读的时候先留意每个式子的输入和输出：同一张矩阵可以是空间中的一个元素，也可以表示一个线性映射。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-1-session-11-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-1-session-11-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-1-session-11-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="matrix-space">一、把矩阵当作向量，究竟改变了什么？</h2>
<p>先看四个实数排成的矩阵：</p>
<div class="equation">\[X=\begin{pmatrix}a&b\\c&d\end{pmatrix}\quad\longleftrightarrow\quad(a,b,c,d)^T.\]</div>
<p>右边只是把四个位置按顺序列出来。两个矩阵逐项相加，就对应两个列表相加；矩阵乘一个实数，也对应列表乘同一个数。因此“向量”在这里指<strong>能相加、能数乘的对象</strong>，不要求外形是一根箭头或一列数。</p>
<p>先在 \(2\times2\) 的小矩阵上看“用矩阵作基”是什么意思。把只有第一行第一列为 1、其余位置为 0 的矩阵记作 \(E_{11}\)；同理，\(E_{12}\) 的 1 在第一行第二列。下标的第一个数指<strong>行</strong>，第二个数指<strong>列</strong>：</p>
<div class="equation">\[E_{11}=\begin{pmatrix}1&0\\0&0\end{pmatrix},\quad E_{12}=\begin{pmatrix}0&1\\0&0\end{pmatrix},\quad E_{21}=\begin{pmatrix}0&0\\1&0\end{pmatrix},\quad E_{22}=\begin{pmatrix}0&0\\0&1\end{pmatrix}.\]</div>
<p>现在取一张具体矩阵。每个数字只负责填自己的位置，所以</p>
<div class="equation">\[\begin{pmatrix}2&-1\\3&4\end{pmatrix}=2E_{11}-E_{12}+3E_{21}+4E_{22}.\]</div>
<p>例如 \(-E_{12}\) 只把第一行第二列填成 \(-1\)，不会改变别的位置。四个位置可分别填写，故任意 \(2\times2\) 矩阵都能这样拆；要拼出零矩阵，四个位置的系数也只能全是零。这正好证明四个 \(E\) 矩阵既能生成整个空间，又线性无关，因此是一组基。</p>
<p>回到全体 \(3\times3\) 实矩阵组成的空间 \(M=M_{3,3}(\mathbb R)\)：同一做法要用九个位置各自的 \(E_{ij}\)。任意矩阵在第 \(i\) 行、第 \(j\) 列的数，通常记作 \(x_{ij}\)；它就是 \(E_{ij}\) 前面的系数。九个位置可以独立填写，所以 \(\dim M=9\)。这里的 9 数的是<strong>描述一张任意矩阵所需的独立数字</strong>；一张 \(3\times3\) 矩阵的秩却最多为 3，数的是它有几个独立的列方向。</p>
<p>第十节已经找到三个子空间：对称矩阵 \(S=\{X:X^T=X\}\)，上三角矩阵 \(U=\{X:x_{ij}=0\text{ 当 }i>j\}\)，以及既对称又上三角的对角矩阵 \(D=S\cap U\)。请把它们当作<strong>矩阵组成的集合</strong>；在每个集合中，矩阵才是“向量”。对称矩阵可自由选对角线三个位置和上方三个位置，下方由对称性确定，所以 \(\dim S=6\)；上三角矩阵可自由选对角线三个位置及上方三个位置，所以 \(\dim U=6\)；对角矩阵只有三个自由位置，所以 \(\dim D=3\)。</p>

<h2 id="sum-and-union">二、两个子空间合起来，为什么要用“和”？</h2>
<p>\(S\cup U\) 的意思是“属于 \(S\) <strong>或</strong>属于 \(U\)”；它没有允许把两种矩阵相加后得到的新矩阵收进来。例如 \(F=E_{12}+E_{21}\) 对称，\(G=E_{23}\) 上三角，而 \(F+G\) 既不对称，也不上三角：\((2,3)\) 位置是 1，\((3,2)\) 是 0；\((2,1)\) 又是 1。并集不对加法封闭，因此不是子空间。</p>
<p>如果确实要允许这种相加，就定义<strong>子空间的和</strong>：</p>
<div class="equation">\[S+U=\{F+G:F\in S,\ G\in U\}.\]</div>
<p>它是子空间，因为 \((F_1+G_1)+(F_2+G_2)=(F_1+F_2)+(G_1+G_2)\) 仍有同样形式，数乘也一样。这里 \(S+U\) 甚至等于整个 \(M\)。与其只看九个维数，不如亲手拆一次。对任意</p>
<div class="equation">\[X=\begin{pmatrix}a&b&c\\d&e&f\\g&h&i\end{pmatrix}.\]</div>
<p>先保留 \(X\) 的对角线和下三角，再把下三角镜像到上方，得到对称矩阵；相减后的余项只能在上三角非零：</p>
<div class="equation">\[F=\begin{pmatrix}a&d&g\\d&e&h\\g&h&i\end{pmatrix}.\]</div>
<div class="equation">\[G=X-F=\begin{pmatrix}0&b-d&c-g\\0&0&f-h\\0&0&0\end{pmatrix}.\]</div>
<p>\(F\) 确实对称，\(G\) 确实上三角。于是每个 \(X\in M\) 都落在 \(S+U\) 中，反向包含本来就成立，故 \(S+U=M\)。</p>
<p>不过，拆法不是唯一的。若 \(H\in D=S\cap U\)，则 \((F+H)+(G-H)=X\)：同一个对角矩阵可以算进 \(F\)，也可以算进 \(G\)。分别数 \(S\) 和 \(U\) 的六个自由度时，把这三个对角自由度数了两次，所以要减去一次：</p>
<div class="equation">\[\dim(S+U)=\dim S+\dim U-\dim(S\cap U)=6+6-3=9.\]</div>
<details class="note-extension"><summary>为什么一般的维数公式也成立？</summary><p>在 \(S\times U\) 中，把一对矩阵 \((F,G)\) 映为 \(F+G\)。像正是 \(S+U\)。核由满足 \(F+G=0\) 的对组成，也就是 \((H,-H)\)，其中 \(H\in S\cap U\)。输入空间有 \(\dim S+\dim U\) 维，核有 \(\dim(S\cap U)\) 维；用秩与零度公式便得到上式。讲义把公式中的 \(\dim(S+U)\) 印成 \(\dim(S\cup U)\)，这里以“和”为准：并集通常不是空间，不能对它使用这个维数公式。</p></details>

<details class="note-extension" id="functions"><summary>讲义中的另一个向量空间：微分方程的解</summary><p>函数也能相加、数乘。设 \(V=C^2(\mathbb R,\mathbb R)\) 是二阶连续可导实函数的空间，\(L(y)=y''+y\)。因为 \(L(ay+bz)=aL(y)+bL(z)\)，方程 \(y''+y=0\) 的解集是 \(L\) 的零空间。\(\cos x,\sin x\) 都是解，而且在 \(x=0\) 比较函数值和导数可证它们无关。</p><p>它们还给出全部解：对任意解 \(y\)，令 \(z=y-y(0)\cos x-y'(0)\sin x\)，则 \(z''+z=0\) 且 \(z(0)=z'(0)=0\)。由 \((z'^2+z^2)'=2z'(z''+z)=0\)，知 \(z'^2+z^2\) 恒为零，故 \(z=0\)。所以解空间恰为 \(\operatorname{span}\{\cos x,\sin x\}\)，维数为 2。讲义还写了 \(e^{ix}\)；那是复值解，不属于这里选定的实函数空间。</p></details>

<h2 id="rank-one">三、秩一矩阵：从三列重复的方向看起</h2>
<p>讲义给出矩阵</p>
<div class="equation">\[A=\begin{pmatrix}1&4&5\\2&8&10\end{pmatrix}.\]</div>
<p>它的三列依次是 \(u,4u,5u\)，其中 \(u=(1,2)^T\)。所以列空间只有 \(\operatorname{span}(u)\) 这一条直线，\(\operatorname{rank}(A)=1\)。把“每列是 \(u\) 的几倍”写成一行，就得到</p>
<div class="equation">\[A=\underbrace{\begin{pmatrix}1\\2\end{pmatrix}}_{u}
\underbrace{\begin{pmatrix}1&4&5\end{pmatrix}}_{v^T}=uv^T.\]</div>
<p>这里 \(u\) 是 \(2\times1\) 列，\(v^T\) 是 \(1\times3\) 行，乘积为 \(2\times3\) 矩阵。右边不是把两个矩阵并排放置：它的第 \(j\) 列实际等于 \(v_j u\)。</p>
<p>同样地，非零 \(m\times n\) 矩阵秩为 1，表示所有列都是某个非零列 \(u\in\mathbb R^m\) 的倍数。把倍数收集成 \(v=(v_1,\ldots,v_n)^T\in\mathbb R^n\)，就有 \(A=uv^T\)。反过来，\(u,v\) 都非零时，\(uv^T\) 至少有一列非零，所有列又都沿着 \(u\)，故秩必为 1。这就是秩一分解的充要条件。</p>
<p>再把这张矩阵当成映射 \(A:\mathbb R^n\to\mathbb R^m\)。对任意输入 \(x\)，</p>
<div class="equation">\[Ax=(uv^T)x=u(v^Tx).\]</div>
<p>\(v^Tx\) 先算出<strong>一个实数</strong>，再用它乘 \(u\)。所以输出只能落在 \(u\) 的方向上；若 \(v^Tx=0\)，输出就是零。于是 \(C(A)=\operatorname{span}(u)\)，\(N(A)=\{x:v^Tx=0\}\)。在讲义的例子中，\((x_1,x_2,x_3)^T\) 被送到 \((x_1+4x_2+5x_3)(1,2)^T\)；零空间由方程 \(x_1+4x_2+5x_3=0\) 描述。</p>
<p>这里还容易误会“秩一”与“一维空间”。固定的 \(A\) 的<strong>列空间</strong>是一维；但所有秩一矩阵并不组成一维向量空间：它们甚至不含零矩阵，而且 \(E_{11}\) 与 \(E_{22}\) 都秩一，其和却秩为 2。先前的 \(\dim M=9\) 数的是整个矩阵空间的一组基有几个元素；\(\operatorname{rank}(A)=1\) 数的是这一个矩阵的列空间有几个基向量。</p>
<details class="note-extension"><summary>为什么说秩一矩阵是“积木”？</summary><p>若 \(A\) 的秩为 \(r\)，先从其列空间选基 \(u_1,\ldots,u_r\)，把 \(A\) 的每一列按这组基表示。对固定的第 \(k\) 个基向量，把它在各列中的系数排为行向量 \(v_k^T\)。逐列核对即可得到 \(A=\sum_{k=1}^r u_kv_k^T\)。每一项最多提供一个列方向，所以少于 \(r\) 项无法拼出秩为 \(r\) 的矩阵。分解本身并不唯一：例如 \((tu)(v/t)^T=uv^T\)，只要 \(t\ne0\)。</p></details>

<h2 id="matrix-map">四、习题 11.2：输入从一列数换成一张矩阵</h2>
<p>前面把 \(A\) 作用在列向量 \(x\) 上，求 \(Ax\)。习题 11.2 固定同一张 \(3\times3\) 矩阵 \(A\)，让输入换成任意 \(3\times3\) 矩阵 \(X\)，求 \(AX\)。这就定义了一个新映射 \(T:M_{3,3}(\mathbb R)\to M_{3,3}(\mathbb R)\)，\(T(X)=AX\)，其中</p>
<div class="equation">\[A=\begin{pmatrix}1&0&-1\\-1&1&0\\0&-1&1\end{pmatrix}.\]</div>
<p>新映射确实线性，因为 \(A(X+Y)=AX+AY\)、\(A(cX)=cAX\)。关键是把输入按列写成 \(X=[x_1\ x_2\ x_3]\)，其中每个 \(x_j\in\mathbb R^3\)。矩阵乘法的列规则给出</p>
<div class="equation">\[T(X)=AX=[Ax_1\ Ax_2\ Ax_3].\]</div>
<p>因此只需分别研究 \(A\) 如何作用于三个列向量。<strong>\(N(A)\) 的元素是一列数；\(N(T)\) 的元素则是一整张矩阵。</strong>两种零空间的名字相近，输入对象却不同。</p>
<h3 id="matrix-map-kernel">哪些输入会得到零矩阵？</h3>
<p>先求原矩阵的零空间。\(A(1,1,1)^T=0\)，而前两列 \((1,-1,0)^T,(0,1,-1)^T\) 无关，所以 \(A\) 的秩为 2、零度为 1，\(N(A)=\operatorname{span}\{(1,1,1)^T\}\)。由上面的列规则，\(AX=0\) 当且仅当三个 \(Ax_j\) 都为零；每个 \(x_j\) 都可以独立选一个倍数。故</p>
<div class="equation">\[N(T)=\left\{\begin{pmatrix}a&b&c\\a&b&c\\a&b&c\end{pmatrix}:a,b,c\in\mathbb R\right\},\qquad \dim N(T)=3.\]</div>
<p>参数 \(a,b,c\) 分别控制三列，彼此不受约束，所以 \(N(T)\) 是 3 维。原来的 \(N(A)\) 只有 1 维；输入从一列变成三列后，同样的一维自由度出现了三次。</p>
<h3 id="matrix-map-image">哪些矩阵能作为输出？</h3>
<p>设 \(B=AX=[b_1\ b_2\ b_3]\)。每个输出列 \(b_j=Ax_j\) 必须在 \(C(A)\) 中。注意 \((1,1,1)A=0\)，所以 \(C(A)\) 的每个向量三个分量之和为零。反过来，给任意 \((p,q,-p-q)^T\)，取 \(x=(p,p+q,0)^T\) 就有 \(Ax=(p,q,-p-q)^T\)。因此 \(C(A)\) <strong>恰好</strong>是“分量和为零”的平面，而不是仅包含在它里面。三列可以分别选，所以所有可能的输出矩阵是</p>
<div class="equation">\[\operatorname{im}T=\left\{\begin{pmatrix}a&b&c\\d&e&f\\-a-d&-b-e&-c-f\end{pmatrix}:a,b,c,d,e,f\in\mathbb R\right\},\qquad \dim\operatorname{im}T=6.\]</div>
<p>上式中每列的前两个数可自由选，第三个由“列和为零”确定；三列共六个自由数，所以 \(\dim\operatorname{im}T=6\)。输入空间 \(M_{3,3}\) 是 9 维，于是 \(\dim N(T)+\dim\operatorname{im}T=3+6=9\)。这正是秩与零度公式，只是这次的输入不是 \(\mathbb R^3\)，而是 \(M_{3,3}\)。\(\operatorname{rank}(A)=2\) 数的是输出列向量的独立方向；\(\operatorname{rank}(T)=6\) 数的是输出<strong>矩阵</strong>的独立方向。</p>

<h2 id="graphs">五、图：先给出关系，再用矩阵编码</h2>
<p>图的顶点可以表示人或网页，边表示“是朋友”或“有链接”这类关系。我们关心两点之间最少要沿几条边走到，而画在纸上的远近没有意义。下面沿用讲义五个顶点、六条边的图，并给顶点编号。</p>
<figure class="graph-figure"><svg viewBox="0 0 560 270" role="img" aria-labelledby="graph-title graph-desc"><title id="graph-title">五个顶点、六条边的图</title><desc id="graph-desc">边为 1-2、1-3、2-3、2-4、3-4、4-5。顶点 1 到顶点 5 的最短路长度为 3。</desc><g class="graph-edge" fill="none"><path d="M105 195L180 55L335 82L305 205L105 195M105 195L335 82M305 205L455 130"/></g><g class="graph-node"><circle cx="180" cy="55" r="22"/><circle cx="335" cy="82" r="22"/><circle cx="105" cy="195" r="22"/><circle cx="305" cy="205" r="22"/><circle cx="455" cy="130" r="22"/></g><g class="graph-label"><text x="180" y="55">1</text><text x="335" y="82">2</text><text x="105" y="195">3</text><text x="305" y="205">4</text><text x="455" y="130">5</text></g></svg><figcaption>顶点编号仅为把图写成矩阵；图形改绘自本讲讲义的示意图。</figcaption></figure>
<p>怎样把一张图变成矩阵？让第 \(i\) 行、第 \(j\) 列的数 \(b_{ij}\) 回答一个问题：“\(i\) 与 \(j\) 之间是否直接有边？”有边填 1，无边填 0。例如 \(b_{12}=1\)，而 \(b_{15}=0\)。这张表叫<strong>邻接矩阵</strong>。由于边没有方向，\(b_{ij}=b_{ji}\)；由于没有顶点连向自己，\(b_{ii}=0\)。上图给出</p>
<div class="equation">\[B=\begin{pmatrix}0&1&1&0&0\\1&0&1&1&0\\1&1&0&1&0\\0&1&1&0&1\\0&0&0&1&0\end{pmatrix}.\]</div>
<p>矩阵乘法为什么有用？看看从 1 到 4 恰好走两条边的办法。第一步必须经过某个中间顶点 \(\ell\)；两段边同时存在时，乘积 \(b_{1\ell}b_{\ell4}\) 才为 1。把所有可能的 \(\ell\) 加起来，就是</p>
<div class="equation">\[(B^2)_{1,4}=\sum_{\ell=1}^{5}b_{1\ell}b_{\ell4}
=b_{12}b_{24}+b_{13}b_{34}=1+1=2.\]</div>
<p>这两个 1 分别对应 \(1\to2\to4\) 和 \(1\to3\to4\)。顶点 5 只连接顶点 4，因此从 1 到 5 恰好走三步的办法也有两种：把上述两步路径各接上 \(4\to5\)，故 \((B^3)_{1,5}=2\)。但 \(b_{15}=0\)，\((B^2)_{1,5}=0\)，所以从 1 到 5 的<strong>最短距离</strong>是 3，而不是“有两条路”中的 2。</p>
<p>一般地，\((B^k)_{ij}\) 数的是从 \(i\) 到 \(j\) 恰好走 \(k\) 条边的<strong>行走</strong>数；可以重复顶点或边。理由是每多乘一次 \(B\)，就把已有行走接上一条边。于是对连通图中不同的顶点，最先使 \((B^k)_{ij}>0\) 的 \(k\) 才是最短距离。讲义在此只提出用矩阵研究距离；幂次与行走数的推导是本笔记的补充。“Small world”关心大网络中这样的最短距离是否通常较小；不能仅凭这一张小图就断言整个网络具有该性质。</p>

<h2 id="check">理解检查</h2>
<ol><li>在上面的 \(X=F+G\) 中，取 \(H=E_{11}\)。写出另一组 \(F',G'\) 使 \(X=F'+G'\)。为什么 \(S\cup U\) 无法保证容纳这些和？</li><li>对讲义的 \(2\times3\) 秩一矩阵，输入 \(x=(1,1,-1)^T\) 时，先算 \(v^Tx\)，再算 \(Ax\)。由此说出它的像空间和零空间。</li><li>习题 11.2 中，为什么 \(N(A)\) 是一维，\(N(T)\) 却是三维？它们的元素分别长什么样？</li><li>直接在图上找出从 1 到 4 的两条两步行走，核对 \((B^2)_{1,4}=2\)；再说明为什么 \(d(1,5)=3\)。</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC 第 1 单元第 11 讲 <em>Matrix Spaces; Rank 1; Small World Graphs</em> 及其习题整理。矩阵子空间、微分方程、秩一矩阵与五顶点图来自讲义；并集反例、习题 11.2 的逐列推导和邻接矩阵的行走计数为本笔记的展开。<a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a>。</p>
</section>
