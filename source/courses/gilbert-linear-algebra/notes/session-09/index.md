---
title: 第九节：线性无关、基与维数｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/session-09/
mathjax: true
toc_number: false
comments: false
---

<style>
.post-block:has(.la-notes--session09) .post-header { margin-bottom: 24px; }
.post-block:has(.la-notes--session09) .post-title { font-size: 26px; line-height: 1.45; color: #253c3c; }
.post-block:has(.la-notes--session09) .breadcrumb { display: none; }
.post-body .la-notes--session09 { max-width: 660px; margin: 0 auto; font-size: 17px; line-height: 1.85; color: #303737; }
.post-body .la-notes--session09 p { margin: 0 0 .9em; text-indent: 0; }
.post-body .la-notes--session09 h2 { margin: 2.4em 0 1em; padding: 0 0 .45em; border-bottom: 1px solid #dce5e3; color: #285d5d; font-size: 22px; line-height: 1.5; }
.post-body .la-notes--session09 h3 { margin: 1.7em 0 .75em; color: #303737; font-size: 18px; line-height: 1.6; }
.post-body .la-notes--session09 [id] { scroll-margin-top: 24px; }
.post-body .la-notes--session09 .note-nav { font-size: 14px; color: #687474; margin-bottom: 18px; }
.post-body .la-notes--session09 .note-nav a { border-color: #ccd8d5; }
.post-body .la-notes--session09 .note-lead { margin: 0 0 1.4em; padding: .1em 0 .1em 14px; border-left: 3px solid #729692; }
.post-body .la-notes--session09 .note-lead p { margin: 0; }
.post-body .la-notes--session09 .reading-links { padding: 14px 0; margin-bottom: 1.6em; border-top: 1px solid #e2e8e6; border-bottom: 1px solid #e2e8e6; font-size: 14px; line-height: 1.8; }
.post-body .la-notes--session09 .reading-links p { margin: 0; }
.post-body .la-notes--session09 .reading-links p + p { margin-top: 8px; }
.post-body .la-notes--session09 .link-row { display: flex; flex-wrap: wrap; column-gap: 18px; row-gap: 6px; }
.post-body .la-notes--session09 .equation { margin: 1em 0 1.15em; padding: 8px 2px; overflow-x: auto; overflow-y: hidden; }
.post-body .la-notes--session09 .MathJax_Display { font-size: 100%; margin: .25em 0 !important; padding: 3px 0; }
.post-body .la-notes--session09 ul, .post-body .la-notes--session09 ol { margin: .65em 0 1.2em; padding-left: 1.35em; }
.post-body .la-notes--session09 li { margin-bottom: .55em; }
.post-body .la-notes--session09 .note-extension { margin: 1.2em 0; padding: 10px 14px; border-left: 2px solid #ccd8d5; font-size: 16px; }
.post-body .la-notes--session09 .note-extension summary { cursor: pointer; color: #285d5d; }
.post-body .la-notes--session09 .note-extension[open] summary { margin-bottom: 1em; }
.post-body .la-notes--session09 .note-table { overflow-x: auto; margin: 1em 0; }
.post-body .la-notes--session09 .note-table table { width: 100%; font-size: 15px; line-height: 1.7; }
.post-body .la-notes--session09 .note-table th, .post-body .la-notes--session09 .note-table td { padding: 10px 12px; vertical-align: top; }
.post-body .la-notes--session09 .note-footer { margin-top: 1.8em; padding-top: 1em; border-top: 1px solid #e2e8e6; font-size: 14px; }
@media (max-width: 600px) {
  .post-block:has(.la-notes--session09) .post-title { font-size: 23px; }
  .post-body .la-notes--session09 { font-size: 16px; line-height: 1.85; }
  .post-body .la-notes--session09 h2 { font-size: 20px; margin-top: 2em; }
  .post-body .la-notes--session09 h3 { font-size: 17px; }
}
</style>

<section class="la-notes la-notes--session09">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-08/">第八节</a></p>
<div class="note-lead"><p><a href="/courses/gilbert-linear-algebra/notes/session-06/#column-space">第六节</a>已给出线性无关、基与维数的定义，<a href="/courses/gilbert-linear-algebra/notes/session-07/">第七节</a>、<a href="/courses/gilbert-linear-algebra/notes/session-08/">第八节</a>又用主元列和特殊解描述两个空间。本节沿讲义的顺序补齐其中的理由：<strong>张成保证表示存在，无关保证表示唯一，基同时满足两者；维数记录一组基需要多少个向量。</strong></p></div>
<div class="reading-links">
<p class="link-row"><a href="/files/mit-18-06sc/unit-1-session-9-summary.pdf">讲义摘要</a> <a href="/files/mit-18-06sc/unit-1-session-9-problems.pdf">习题</a> <a href="/files/mit-18-06sc/unit-1-session-9-solutions.pdf">参考解答</a></p>
<p class="link-row"><a href="#independence">线性无关与唯一性</a> <a href="#span">张成与存在性</a> <a href="#basis">基与坐标</a> <a href="#dimension">维数为什么确定</a> <a href="#column-basis">列空间的基</a> <a href="#null-basis">零空间的基</a></p>
</div>

<h2 id="independence">一、线性无关为什么决定表示的唯一性？</h2>
<p>上一节的方程 \(A\mathbf x=\mathbf b\)，按列展开就是</p>
<div class="equation">\[
x_1\mathbf a_1+\cdots+x_n\mathbf a_n=\mathbf b.
\]</div>
<p>这里 \(A\in\mathbb R^{m\times n}\)，\(\mathbf a_j\in\mathbb R^m\) 是第 \(j\) 列，\(\mathbf b\in\mathbb R^m\) 是给定的目标向量，\(\mathbf x=(x_1,\ldots,x_n)^T\in\mathbb R^n\) 收集待求的组合系数。<strong>求方程的解，就是求目标向量的一组表示系数。</strong>第八节的“有没有解、解是否唯一”，因而变成“能否表示、表示是否唯一”。</p>
<p>先研究唯一性。本节都在实数范围内讨论；把矩阵的列推广为同一个实向量空间中的任意一组向量 \(\mathbf v_1,\ldots,\mathbf v_k\)，其中 \(k\ge1\)。</p>
<p><strong>定义：</strong>若对任意实系数 \(c_1,\ldots,c_k\)，都有</p>
<div class="equation">\[
c_1\mathbf v_1+\cdots+c_k\mathbf v_k=\mathbf0
\quad\Longrightarrow\quad c_1=\cdots=c_k=0,
\]</div>
<p>则称这组向量<strong>线性无关</strong>（linearly independent）。否则称<strong>线性相关</strong>：存在一组<strong>不全为零</strong>的系数，使组合为零。“不全为零”只要求至少一个系数非零，不要求每个都非零。全零系数给出<strong>平凡关系</strong>；有非零系数的关系称为<strong>非平凡关系</strong>。线性无关要求不存在非平凡关系。</p>
<p>这个定义为什么只检查零向量，却能判断所有表示的唯一性？假设同一个 \(\mathbf b\) 有两种表示：</p>
<div class="equation">\[
\mathbf b=\sum_{i=1}^k c_i\mathbf v_i
=\sum_{i=1}^k d_i\mathbf v_i.
\]</div>
<p>这里 \(i\) 是向量的编号，\(c_i,d_i\) 是两组系数。两式相减，得到 \(\sum_{i=1}^k(c_i-d_i)\mathbf v_i=\mathbf0\)。若向量线性无关，便有每个 \(c_i-d_i=0\)，两组系数完全相同。反过来，若存在非零的系数组合成零，零向量就已有两种表示：全零系数，以及这组非零系数。因此，</p>
<p><strong>线性无关，等价于每个能被这组向量表示的向量，都只有一组表示系数。</strong>这里尚未保证所有目标向量都能表示。</p>
<p>线性相关还等价于<strong>其中至少一个向量能由其余向量组合出来</strong>。因为在非零关系中选取 \(c_j\ne0\)，便可解出</p>
<div class="equation">\[
\mathbf v_j=-\sum_{i\ne j}\frac{c_i}{c_j}\mathbf v_i.
\]</div>
<p>反过来，把这种表示移到同一边，就得到非平凡关系。因此可以删去这个 \(\mathbf v_j\)，而不改变张成空间。不过，<strong>“相关”不表示任意一个向量都能删去</strong>：在 \((1,0)^T,(2,0)^T,(0,1)^T\) 中，前两个可以互相替代，第三个却不能由它们表示。</p>
<p>在二维、三维的几何中，两个向量无关，等价于它们不落在同一条<strong>过原点的直线</strong>中；在 \(\mathbb R^3\) 中，三个向量无关，要求它们不全落在同一个<strong>过原点的平面</strong>中。仅仅两两不平行还不够，例如 \((1,0,0)^T,(0,1,0)^T,(1,1,0)^T\) 两两不平行，第三个却是前两个之和。任何含零向量的向量组也一定相关，因为给零向量的系数取 1，其余取 0，就有非零关系。</p>

<h3 id="independence-elimination">把定义接回消元</h3>
<p>把 \(\mathbf v_1,\ldots,\mathbf v_n\in\mathbb R^m\) 按列排成矩阵 \(A\)，把系数收集为 \(\mathbf x=(x_1,\ldots,x_n)^T\)，则 \(A\mathbf x=\sum_jx_j\mathbf v_j\)。<strong>列向量组无关，当且仅当齐次方程只有零解，也当且仅当每列都有主元、没有自由变量：</strong></p>
<div class="equation">\[
N(A)=\{\mathbf0\}\iff r=n.
\]</div>
<p>其中 \(N(A)=\{\mathbf x:A\mathbf x=\mathbf0\}\)，\(r\) 仍按前几节定义为主元数。中间的依据是：有自由变量时，令其中一个为 1，就产生非零齐次解；没有自由变量时，齐次主元方程只能给出全零解。特别地，若 \(n&gt;m\)，主元数至多为 \(m\)，必有自由变量，所以 \(\mathbb R^m\) 中超过 \(m\) 个向量一定相关。</p>
<p>若列向量相关，取非零 \(\mathbf z\in N(A)\)。对任意已有的解 \(\mathbf x_p\)，\(\mathbf x_p+t\mathbf z\) 都给出同一输出，且不同实数 \(t\) 给出不同系数。因此，相关时<strong>每个能表示的目标都有无穷多组表示</strong>；无法表示的目标仍然无解。这正是第八节“特解加零空间”的结论。</p>

<h2 id="span">二、张成回答的是：到底能表示哪些向量？</h2>
<p><strong>定义：</strong>这组向量的<strong>张成空间</strong>（span）是它们的所有线性组合组成的集合：</p>
<div class="equation">\[
\operatorname{span}(\mathbf v_1,\ldots,\mathbf v_k)
=\left\{\sum_{i=1}^k c_i\mathbf v_i:
 c_1,\ldots,c_k\in\mathbb R\right\}.
\]</div>
<p>按第五节的子空间判据，所有组合形成子空间：零系数给出零向量，加法和数乘只改变组合系数。它也是<strong>包含这些向量的最小子空间</strong>，因为任何包含它们的子空间都必须包含它们的全部线性组合。</p>
<p>给定子空间 \(S\)，若上述集合恰好等于 \(S\)，就说这组向量<strong>张成 \(S\)</strong>。验证时先确认每个 \(\mathbf v_i\in S\)，这样所有组合都在 \(S\) 中；再证明每个 \(\mathbf b\in S\) 都能表示，这样没有遗漏。</p>
<p>矩阵的全部列按定义张成列空间 \(C(A)\)，但它们可能相关。张成只保证表示<strong>存在</strong>；无关只保证表示<strong>至多一种</strong>。必须说明要张成哪个空间：同一组向量可能足以张成一个平面，却不足以张成整个 \(\mathbb R^3\)。</p>

<h2 id="basis">三、基把存在性与唯一性合在一起</h2>
<p><strong>定义：</strong>子空间 \(S\) 的一组<strong>基</strong>（basis），是属于 \(S\)、线性无关且张成 \(S\) 的一组向量 \(\mathbf v_1,\ldots,\mathbf v_d\)。等价地，\(S\) 中每个向量都能<strong>唯一</strong>写成它们的线性组合。</p>
<p>张成给出“至少一组系数”，无关给出“至多一组系数”，合起来就是“恰好一组”。反过来，若每个向量都有唯一表示，便有张成性；又因零向量只能用全零系数表示，便有无关性。基向量不必互相垂直，也不必长度为 1。</p>
<p>例如 \(\mathbb R^3\) 的标准基为 \(\mathbf e_1=(1,0,0)^T\)、\(\mathbf e_2=(0,1,0)^T\)、\(\mathbf e_3=(0,0,1)^T\)。任意 \((a,b,c)^T\) 恰好等于 \(a\mathbf e_1+b\mathbf e_2+c\mathbf e_3\)，而三个分量分别强制三个系数为 \(a,b,c\)。</p>
<p>讲义还讨论了另一组向量：</p>
<div class="equation">\[
\mathbf v_1=\begin{pmatrix}1\\1\\2\end{pmatrix},\qquad
\mathbf v_2=\begin{pmatrix}2\\2\\5\end{pmatrix},\qquad
\mathbf v_3=\begin{pmatrix}3\\3\\8\end{pmatrix}.
\]</div>
<p>因为 \(\mathbf v_3=-\mathbf v_1+2\mathbf v_2\)，三者相关，不能构成基。去掉 \(\mathbf v_3\) 后，前两个向量构成的是哪个空间的基？</p>
<p>先证无关：若 \(\alpha\mathbf v_1+\beta\mathbf v_2=\mathbf0\)，第一、第三分量分别给出 \(\alpha+2\beta=0\)、\(2\alpha+5\beta=0\)。后式减前式的两倍，得到 \(\beta=0\)，再得 \(\alpha=0\)。</p>
<p>再证张成：两个向量的前两分量都相同，故每个组合都在平面 \(S=\{(a,a,c)^T:a,c\in\mathbb R\}\) 中。反过来，任取这个平面中的 \((a,a,c)^T\)，求解</p>
<div class="equation">\[
\alpha+2\beta=a,\qquad 2\alpha+5\beta=c
\quad\Longrightarrow\quad
\beta=c-2a,\quad\alpha=5a-2c.
\]</div>
<p>所以每个平面中的向量都能表示。\(\mathbf v_1,\mathbf v_2\) 因而是平面 \(S\) 的基，却不是 \(\mathbb R^3\) 的基，例如 \((1,0,0)^T\) 就无法表示。</p>
<p>固定基的排列顺序，记 \(B=(\mathbf v_1,\mathbf v_2)\)。按顺序把这两个系数排成列，所得向量称为原向量<strong>相对于基 \(B\) 的坐标向量</strong>：</p>
<div class="equation">\[
[(a,a,c)^T]_B=\begin{pmatrix}5a-2c\\c-2a\end{pmatrix}.
\]</div>
<p>下标 \(B\) 指定基及其顺序。例如 \((1,1,1)^T=3\mathbf v_1-\mathbf v_2\)，所以它在 \(B\) 下的坐标是 \((3,-1)^T\)。若改用同一平面的基 \(B'=((1,1,0)^T,(0,0,1)^T)\)，同一向量的坐标就变为 \((1,1)^T\)。<strong>向量没有变，记录它所用的系数变了。</strong>左侧原向量有三个分量，坐标向量只有两个分量；唯一性指基及其顺序固定以后的唯一。</p>

<h2 id="dimension">四、为什么换一组基，向量个数也不会变？</h2>
<p>同一个平面可以选不同的基。为什么不能有一组基含两个向量，另一组却含三个？第六节引用了“任意两组基的向量个数相同”，现在证明它。本节只讨论能由有限个向量张成的空间。</p>
<h3 id="independent-vs-spanning">先证明一个比较结论</h3>
<p><strong>若 \(p\) 个线性无关向量都在 \(q\) 个向量的张成空间中，则 \(p\le q\)。</strong>注意后面的 \(q\) 个向量不必无关。</p>
<p>设前一组是 \(\mathbf u_1,\ldots,\mathbf u_p\)，后一组是 \(\mathbf w_1,\ldots,\mathbf w_q\)。因为每个 \(\mathbf u_i\) 都能由后一组表示，可选系数 \(d_{ji}\) 使</p>
<div class="equation">\[
\mathbf u_i=\sum_{j=1}^q d_{ji}\mathbf w_j,
\qquad D=(d_{ji})\in\mathbb R^{q\times p}.
\]</div>
<p>\(d_{ji}\) 中，\(i\) 标记正在表示的 \(\mathbf u_i\)，\(j\) 标记使用的 \(\mathbf w_j\)；矩阵 \(D\) 的第 \(i\) 列就是表示 \(\mathbf u_i\) 的一组系数。若 \(p&gt;q\)，齐次方程 \(D\mathbf c=\mathbf0\) 的未知数比方程多，由消元可知存在非零解 \(\mathbf c=(c_1,\ldots,c_p)^T\)。于是</p>
<div class="equation">\[
\sum_{i=1}^p c_i\mathbf u_i
=\sum_{j=1}^q\left(\sum_{i=1}^p d_{ji}c_i\right)\mathbf w_j
=\mathbf0.
\]</div>
<p>括号中的系数正是 \(D\mathbf c\) 的第 \(j\) 个分量，全都为零。这就得到 \(\mathbf u_i\) 的非零系数关系，与无关性矛盾，故 \(p\le q\)。</p>
<h3 id="equal-basis-size">再比较任意两组基</h3>
<p>若 \(B\) 是有 \(p\) 个向量的基，\(B'\) 是有 \(q\) 个向量的基，则 \(B\) 无关而 \(B'\) 张成，刚才的结论给出 \(p\le q\)。交换角色，又得 \(q\le p\)，所以 \(p=q\)。</p>
<p><strong>定义：</strong>有限维空间 \(S\) 的<strong>维数</strong> \(\dim S\)，就是其任意一组基的向量个数。现在这个数才有了与选基无关的意义。标准基有 \(n\) 个向量，所以 \(\dim\mathbb R^n=n\)；上一部分的平面有一组含两个向量的基，所以维数为 2。</p>
<p>基也一定能找到：从有限张成组出发，逐次删去冗余向量，张成空间不变；有限次后得到无关的张成组。对只含零向量的空间 \(\{\mathbf0\}\)，空向量组就是基，空组合约定等于零，其维数为 0；含一个零向量的组 \( (\mathbf0) \) 则相关，不能作为基。</p>
<p><strong>不能只按式子里出现的参数个数判断维数。</strong>例如 \(\mathbf x=(s+t)\mathbf v\)，其中 \(\mathbf v\ne\mathbf0\)，虽然写了两个任取的实数 \(s,t\)，却只生成一条直线；\((s,t)=(1,0)\) 与 \((0,1)\) 给出同一向量。基坐标要求不同系数组合对应不同向量。维数数的是唯一表示所需的系数个数。</p>
<details class="note-extension">
<summary>知道维数以后，怎样判断一组向量是否足够？</summary>
<p>设 \(\dim S=d\)。比较结论说明：\(S\) 中的无关向量组至多有 \(d\) 个，张成组至少有 \(d\) 个。但<strong>仅有 \(d\) 个向量，不能保证是基</strong>，例如在二维平面中取两个相同向量。</p>
<p>若已有 \(d\) 个无关向量，它们必张成 \(S\)：否则选一个在其张成空间外的向量加入。新组仍无关，因为若新向量的系数非零，就能把它表示成旧向量的组合，产生矛盾；若该系数为零，则原组无关迫使其余系数也为零。这样得到 \(d+1\) 个无关向量，与比较结论矛盾。</p>
<p>若已有 \(d\) 个向量张成 \(S\)，它们必无关：否则删去冗余向量，就用少于 \(d\) 个向量张成，与比较结论矛盾。因此“恰有 \(d\) 个”配合“无关”或“张成”中的任一个，才足以判断是基。</p>
</details>

<h2 id="column-basis">五、为什么原矩阵的主元列是列空间的基？</h2>
<p>用这些定义重新检查求基的方法。讲义本节采用一个新矩阵，与第七、八节不同；记四列为 \(\mathbf a_1,\ldots,\mathbf a_4\)：</p>
<div class="equation">\[
A=\begin{pmatrix}
1&amp;2&amp;3&amp;1\\
1&amp;1&amp;2&amp;1\\
1&amp;2&amp;3&amp;1
\end{pmatrix}
\longrightarrow
R=\begin{pmatrix}
1&amp;0&amp;1&amp;1\\
0&amp;1&amp;1&amp;0\\
0&amp;0&amp;0&amp;0
\end{pmatrix}.
\]</div>
<p>这里依次作第 2 行减第 1 行、第 3 行减第 1 行，再将第 2 行乘以 −1，最后作第 1 行减第 2 行的 2 倍，得到行最简形。主元在第 1、2 列。</p>
<p>原矩阵满足 \(\mathbf a_3=\mathbf a_1+\mathbf a_2\)、\(\mathbf a_4=\mathbf a_1\)，所以前两列已张成整个 \(C(A)\)。它们也无关：\(\alpha\mathbf a_1+\beta\mathbf a_2=\mathbf0\) 的前两行给出 \(\alpha+2\beta=0\)、\(\alpha+\beta=0\)，只能有 \(\alpha=\beta=0\)。故</p>
<div class="equation">\[
B_C=\left(
\begin{pmatrix}1\\1\\1\end{pmatrix},
\begin{pmatrix}2\\1\\2\end{pmatrix}
\right),\qquad \dim C(A)=2.
\]</div>
<p>也可把空间写清楚：任意组合为 \((\alpha+2\beta,\alpha+\beta,\alpha+2\beta)^T\)，必满足第三分量等于第一分量。反过来，任给 \((a,b,a)^T\)，取 \(\beta=a-b\)、\(\alpha=2b-a\) 即可表示。因此</p>
<div class="equation">\[
C(A)=\{(a,b,a)^T:a,b\in\mathbb R\}.
\]</div>
<h3 id="pivot-basis-proof">一般情形为什么也成立？</h3>
<p>若 \(R=EA\)，其中 \(E\) 是可逆行操作的乘积，则对任意系数向量 \(\mathbf c\)，</p>
<div class="equation">\[
A\mathbf c=\mathbf0\iff R\mathbf c=\mathbf0.
\]</div>
<p>所以行操作保留了<strong>各列之间的所有线性关系及其系数</strong>。在行最简形中，主元列是在前 \(r\) 个位置上的标准单位向量，彼此无关；每个非主元列只有前 \(r\) 个分量可能非零，恰好可用这些单位向量及相应分量作系数组合出来。把这些关系传回 \(A\)，便知原矩阵中对应的主元列同样无关，并张成所有列。</p>
<p>这里必须取<strong>原矩阵中的列</strong>。本例 \(R\) 的主元列是 \((1,0,0)^T,(0,1,0)^T\)，张成平面 \(z=0\)，而原列空间是平面 \(z=x\)。行操作保持列之间的关系，却一般会改变列空间在环境空间中的位置。</p>
<p>由此，前面按主元数定义的秩有了空间解释：</p>
<div class="equation">\[
\operatorname{rank}(A)=r=\dim C(A).
\]</div>
<p>因此，秩既是主元数，也是列空间中能选出的无关向量的最大数目。主元列给出一种基，其他列也可能组成基；选哪一组可以不同，但个数总为 \(r\)。</p>

<h2 id="null-basis">六、为什么特殊解恰好构成零空间的基？</h2>
<p>先直接看刚才的两条列关系：</p>
<div class="equation">\[
-\mathbf a_1-\mathbf a_2+\mathbf a_3=\mathbf0,
\qquad -\mathbf a_1+\mathbf a_4=\mathbf0.
\]</div>
<p>按四列的顺序记录系数，分别得到 \((-1,-1,1,0)^T\)、\((-1,0,0,1)^T\)。它们属于 \(\mathbb R^4\)，不是原来的列向量，而是<strong>描述列之间关系的系数向量</strong>。由于 \(A\mathbf x\) 正是按系数 \(\mathbf x\) 组合各列，组合为零便等价于 \(\mathbf x\in N(A)\)。所以<strong>零空间收集了列向量组的全部线性关系</strong>。</p>
<p>这给出了两个齐次解，还需检查它们是否生成全部解。从 \(R\mathbf x=\mathbf0\) 读出</p>
<div class="equation">\[
x_1+x_3+x_4=0,\qquad x_2+x_3=0.
\]</div>
<p>第 3、4 列无主元，令自由变量 \(x_3=s,x_4=t\)，其中 \(s,t\) 独立取任意实数。主元变量随之为 \(x_1=-s-t,x_2=-s\)，所以</p>
<div class="equation">\[
\mathbf x=
\begin{pmatrix}-s-t\\-s\\s\\t\end{pmatrix}
=s\underbrace{\begin{pmatrix}-1\\-1\\1\\0\end{pmatrix}}_{\mathbf s_1}
+t\underbrace{\begin{pmatrix}-1\\0\\0\\1\end{pmatrix}}_{\mathbf s_2}.
\]</div>
<p>\(\mathbf s_1\) 来自自由分量 \((x_3,x_4)=(1,0)\)，\(\mathbf s_2\) 来自 \((0,1)\)。要说它们是基，还需明确验证两件事：</p>
<ul>
<li><strong>张成：</strong>任意零空间向量都满足刚才的主元方程，以它自己的第三、第四分量作为 \(s,t\)，就恰好得到上述组合；反过来，每个这样的组合都满足方程。因此组合给出的集合恰好是 \(N(A)\)。</li>
<li><strong>无关：</strong>若 \(s\mathbf s_1+t\mathbf s_2=\mathbf0\)，只看第三、第四分量，就分别得到 \(s=0,t=0\)。这里不需再次消元。</li>
</ul>
<p>所以 \(B_N=(\mathbf s_1,\mathbf s_2)\) 是零空间的一组基，\(\dim N(A)=2\)。两个特殊解既生成所有解，又使系数唯一；这才证明维数为 2。只验证它们代入后等于零，还不足以证明它们是基。</p>
<p><strong>讲义勘误：</strong>原摘要第 3 页第二个特殊解的末分量印成了 \(-1\)，应为 \(+1\)。代入原矩阵检查：\(A\mathbf s_2=-\mathbf a_1+\mathbf a_4=\mathbf0\)；若末分量用 \(-1\)，结果为 \(-\mathbf a_1-\mathbf a_4=-2\mathbf a_1\ne\mathbf0\)。这里采用已核验的正号。</p>
<p>一般地，有 \(n-r\) 个自由变量时，逐个把一个自由变量取 1、其余取 0，得到 \(n-r\) 个特殊解。它们张成全部齐次解，因为任意自由变量取值都能按这些特殊解组合；它们无关，因为在自由分量上形成单位矩阵，组合为零便迫使所有系数为零。因此</p>
<div class="equation">\[
\dim N(A)=n-r,\qquad
\dim C(A)+\dim N(A)=n.
\]</div>
<p>这就是 <strong>Rank–Nullity Theorem</strong> 在矩阵情形下的内容。它把 \(n\) 列分成 \(r\) 个主元列与 \(n-r\) 个自由列：前者构成输出空间 \(C(A)\) 的基；每个后者与主元列之间的一条关系，产生一个零空间基向量。两种基的个数之和因此等于输入空间 \(\mathbb R^n\) 的维数。</p>
<p>本例 \(C(A)\subseteq\mathbb R^3\)、\(N(A)\subseteq\mathbb R^4\)，维数分别为 2、2，满足 \(2+2=4\)。列空间的基向量有三个分量，零空间的基向量有四个分量；两个空间的维数相同，并不表示它们是同一个空间。上式是维数的等式，不是在把两个空间直接相加。</p>
<p>最后把结论接回求解：列空间的基描述<strong>哪些输出能达到</strong>；零空间的基描述<strong>输入怎样变化而不改变输出</strong>。若 \(A\mathbf x=\mathbf b\) 有解，后者便给出全部解中的自由部分。对于 \(n\times n\) 方阵，以下说法相互等价：</p>
<ul>
<li>\(A\) 可逆；</li>
<li>\(r=n\)，即每行、每列都有主元；</li>
<li>\(A\) 的各列构成 \(\mathbb R^n\) 的一组基；</li>
<li>每个 \(\mathbf b\in\mathbb R^n\) 都能唯一写成列向量的组合，即方程总有唯一解。</li>
</ul>
<p>满列秩保证无关，满行秩保证张成 \(\mathbb R^n\)；对于方阵，两者同时由 \(r=n\) 给出。此时行最简形为 \(I_n\)，若 \(EA=I_n\)，则可逆的 \(E\) 给出 \(A=E^{-1}\)。这把第二、八节的可逆性与本节的基联系起来。对于非方阵，满列秩与满行秩必须分别判断。</p>

<h2 id="self-check">理解检查</h2>
<ol>
<li>三个两两不平行的 \(\mathbb R^3\) 向量，为什么仍可能相关？请写出一个非零的系数关系。</li>
<li>平面 \(S=\{(a,a,c)^T\}\) 中，每个向量有三个分量，为什么维数却是 2？它在基 \((\mathbf v_1,\mathbf v_2)\) 下的两个坐标与原来的三个分量有什么区别？</li>
<li>本例的四个列向量为什么张成列空间，却不能一起成为基？能否用 \((\mathbf a_2,\mathbf a_3)\) 换一组基？请分别验证无关与张成。</li>
<li>\(\mathbf x=(s+t)\mathbf v\) 为什么不能说明空间是二维？对本例两个特殊解，为什么第三、第四分量却能保证参数的唯一性？怎样把一条列关系转换成一个零空间向量？</li>
</ol>
<h2 id="sources">资料与衔接</h2>
<p>依据 MIT OCW 18.06SC 第九讲 <em>Independence, Basis and Dimension</em> 及<a href="/files/mit-18-06sc/unit-1-session-9-summary.pdf">讲义摘要</a>整理。定义、两组矩阵例子、列空间与零空间的基来自摘要；唯一表示、基的个数不变、主元列与特殊解成基的证明作了展开；坐标比较、参数冗余与列关系的说明用于衔接第五至八节，并纠正摘要中的符号笔误。摘要注明三向量例子在 Lecture 10 开头讨论，本页用它补充辨析“平面的基”与“整个空间的基”。<a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/">课程来源</a>；笔记按 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 分享。</p>
<p class="note-footer"><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-08/">上一节：有解条件与全部解</a></p>
</section>
