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
<div class="note-lead"><p>前几节已用主元列描述列空间、用特殊解描述零空间，也用“维数”计算自由程度。本节把这些说法的依据补齐：怎样选取一组向量，使空间中的每个向量都能被它们<strong>唯一表示</strong>？</p></div>
<div class="reading-links">
<p class="link-row"><a href="/files/mit-18-06sc/unit-1-session-9-summary.pdf">讲义摘要</a> <a href="/files/mit-18-06sc/unit-1-session-9-problems.pdf">习题</a> <a href="/files/mit-18-06sc/unit-1-session-9-solutions.pdf">参考解答</a></p>
<p class="link-row"><a href="#independence">线性无关与唯一性</a> <a href="#span">张成与存在性</a> <a href="#basis">基与坐标</a> <a href="#dimension">维数为什么确定</a> <a href="#column-basis">列空间的基</a> <a href="#null-basis">零空间的基</a></p>
</div>

<h2 id="independence">一、线性无关为什么决定表示的唯一性？</h2>
<p>本节都在实数范围内讨论。设 \(\mathbf v_1,\ldots,\mathbf v_k\) 是同一个向量空间中的向量，\(c_1,\ldots,c_k\) 是实数系数。所谓线性组合，就是 \(c_1\mathbf v_1+\cdots+c_k\mathbf v_k\)。对于给定向量 \(\mathbf b\)，我们要问：这样的组合能否等于 \(\mathbf b\)？若能，系数是否只有一组？</p>
<p><strong>定义：</strong>若</p>
<div class="equation">\[
c_1\mathbf v_1+\cdots+c_k\mathbf v_k=\mathbf0
\quad\Longrightarrow\quad c_1=\cdots=c_k=0,
\]</div>
<p>则称这组向量<strong>线性无关</strong>（linearly independent）。否则称<strong>线性相关</strong>：存在一组<strong>不全为零</strong>的系数，使组合为零。“不全为零”只要求至少一个系数非零，不要求每个都非零。全零系数总能组合成零，所以要检查的是有没有除此以外的关系。</p>
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
<p>反过来，把这种表示移到同一边，就得到一个非零的系数关系。因此“相关”表示有冗余：去掉这个 \(\mathbf v_j\)，不会失去任何原来能表示的向量。</p>
<p>几何上，两个向量无关，要求它们不落在同一条<strong>过原点的直线</strong>中；在 \(\mathbb R^3\) 中，三个向量无关，要求它们不全落在同一个<strong>过原点的平面</strong>中。仅仅两两不平行还不够，例如 \((1,0,0)^T,(0,1,0)^T,(1,1,0)^T\) 两两不平行，第三个却是前两个之和。任何含零向量的向量组也一定相关，因为给零向量的系数取 1，其余取 0，就有非零关系。</p>

<h3 id="independence-elimination">把定义接回消元</h3>
<p>把 \(\mathbf v_1,\ldots,\mathbf v_n\in\mathbb R^m\) 按列排成矩阵 \(A\)，把系数收集为 \(\mathbf x=(x_1,\ldots,x_n)^T\)，则 \(A\mathbf x=\sum_jx_j\mathbf v_j\)。因此</p>
<div class="equation">\[
\begin{aligned}
\text{columns of }A\text{ are independent}
&\iff N(A)=\{\mathbf0\}\\
&\iff \text{no free variables}\\
&\iff r=n.
\end{aligned}
\]</div>
<p>其中 \(N(A)=\{\mathbf x:A\mathbf x=\mathbf0\}\)，\(r\) 仍按前几节定义为主元数。中间的依据是：有自由变量时，令其中一个为 1，就产生非零齐次解；没有自由变量时，齐次主元方程只能给出全零解。特别地，若 \(n&gt;m\)，主元数至多为 \(m\)，必有自由变量，所以 \(\mathbb R^m\) 中超过 \(m\) 个向量一定相关。这里还没有用到维数定理。</p>

<h2 id="span">二、张成回答的是：到底能表示哪些向量？</h2>
<p><strong>定义：</strong>这组向量的<strong>张成空间</strong>（span）是它们的所有线性组合组成的集合：</p>
<div class="equation">\[
\operatorname{span}(\mathbf v_1,\ldots,\mathbf v_k)
=\left\{\sum_{i=1}^k c_i\mathbf v_i:
 c_1,\ldots,c_k\in\mathbb R\right\}.
\]</div>
<p>这确实是子空间：全零系数给出零向量；两个组合相加，只是把对应系数相加；组合乘一个实数，只是把每个系数乘这个数。因此它对加法和数乘封闭。它也是<strong>包含这些向量的最小子空间</strong>：任何包含它们的子空间，都因封闭性而必须包含它们的每个线性组合。</p>
<p>给定子空间 \(S\)，若上述集合恰好等于 \(S\)，就说这组向量<strong>张成 \(S\)</strong>。这要求既不遗漏 \(S\) 中的向量，也不产生 \(S\) 以外的向量；通常先确认每个 \(\mathbf v_i\in S\)，再证明每个 \(\mathbf b\in S\) 都能表示。</p>
<p>矩阵的全部列按定义张成列空间 \(C(A)\)，但它们可能相关。张成只保证表示<strong>存在</strong>；无关只保证表示<strong>至多一种</strong>。必须说明要张成哪个空间：同一组向量可能足以张成一个平面，却不足以张成整个 \(\mathbb R^3\)。</p>

<h2 id="basis">三、基把存在性与唯一性合在一起</h2>
<p><strong>定义：</strong>子空间 \(S\) 的一组<strong>基</strong>（basis），是属于 \(S\)、线性无关且张成 \(S\) 的一组向量 \(\mathbf v_1,\ldots,\mathbf v_d\)。等价地，\(S\) 中每个向量都能<strong>唯一</strong>写成它们的线性组合。</p>
<p>前两个部分已经证明“基 ⇒ 唯一表示”。反方向也成立：每个向量都能表示，说明张成；零向量的表示唯一，说明线性无关。这两个要求缺一不可。</p>
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
<p>固定基的排列顺序，记 \(B=(\mathbf v_1,\mathbf v_2)\)。这两个系数按顺序组成向量<strong>相对于基 \(B\) 的坐标</strong>：</p>
<div class="equation">\[
[(a,a,c)^T]_B=\begin{pmatrix}5a-2c\\c-2a\end{pmatrix}.
\]</div>
<p>下标 \(B\) 表明采用哪组有序基。左边实际向量有三个分量，右边用两个系数就能确定它，因为它受限于这个平面。换基会改变坐标，向量本身不变；“唯一”指的是<strong>基及其顺序固定以后</strong>的唯一。</p>

<h2 id="dimension">四、为什么换一组基，向量个数也不会变？</h2>
<p>不能把“维数是基的向量个数”当成不需证明的计数规则：一个空间可能有许多组基，必须先证明它们的个数相同。本节只讨论能由有限个向量张成的空间。</p>
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
<p>括号中的系数正是 \(D\mathbf c\) 的第 \(j\) 个分量，全都为零。这就得到 \(\mathbf u_i\) 的非零系数关系，与无关性矛盾，故 \(p\le q\)。证明只使用消元与无关的定义，没有预先使用维数。</p>
<h3 id="equal-basis-size">再比较任意两组基</h3>
<p>若 \(B\) 是有 \(p\) 个向量的基，\(B'\) 是有 \(q\) 个向量的基，则 \(B\) 无关而 \(B'\) 张成，刚才的结论给出 \(p\le q\)。交换角色，又得 \(q\le p\)，所以 \(p=q\)。</p>
<p><strong>定义：</strong>有限维空间 \(S\) 的<strong>维数</strong> \(\dim S\)，就是其任意一组基的向量个数。现在这个数才有了与选基无关的意义。标准基有 \(n\) 个向量，所以 \(\dim\mathbb R^n=n\)；上一部分的平面有一组含两个向量的基，所以维数为 2。</p>
<p>基为什么一定能找到？从一个有限的张成向量组出发，只要相关，就删去一个能被其余向量表示的向量；张成空间不变。有限次后得到无关的张成组，即一组基。只含零向量的空间 \(\{\mathbf0\}\) 的基约定为空向量组，维数为 0；\(\{\mathbf0\}\) 本身作为一个含零向量的向量组是相关的，不能作为基。</p>
<details class="note-extension">
<summary>知道维数以后，怎样判断一组向量是否足够？</summary>
<p>设 \(\dim S=d\)。比较结论说明：\(S\) 中的无关向量组至多有 \(d\) 个，张成组至少有 \(d\) 个。但<strong>仅有 \(d\) 个向量，不能保证是基</strong>，例如在二维平面中取两个相同向量。</p>
<p>若已有 \(d\) 个无关向量，它们必张成 \(S\)：否则选一个在其张成空间外的向量加入。新组仍无关，因为若新向量的系数非零，就能把它表示成旧向量的组合，产生矛盾；若该系数为零，则原组无关迫使其余系数也为零。这样得到 \(d+1\) 个无关向量，与比较结论矛盾。</p>
<p>若已有 \(d\) 个向量张成 \(S\)，它们必无关：否则删去冗余向量，就用少于 \(d\) 个向量张成，与比较结论矛盾。因此“恰有 \(d\) 个”配合“无关”或“张成”中的任一个，才足以判断是基。</p>
</details>

<h2 id="column-basis">五、为什么原矩阵的主元列是列空间的基？</h2>
<p>回到讲义本节的新矩阵。它与第七、八节的矩阵不同；记四列为 \(\mathbf a_1,\ldots,\mathbf a_4\)：</p>
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
<p>主元在第 1、2 列。原矩阵满足 \(\mathbf a_3=\mathbf a_1+\mathbf a_2\)、\(\mathbf a_4=\mathbf a_1\)，所以前两列已张成整个 \(C(A)\)。它们也无关：\(\alpha\mathbf a_1+\beta\mathbf a_2=\mathbf0\) 的前两行给出 \(\alpha+2\beta=0\)、\(\alpha+\beta=0\)，只能有 \(\alpha=\beta=0\)。故</p>
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
<p>所以行操作保留了<strong>各列之间的所有线性关系及其系数</strong>。在行最简形中，主元列是在前 \(r\) 个位置上的标准单位向量，彼此无关；每个非主元列都是它们的线性组合，因为末尾零行没有其他分量。把这些关系传回 \(A\)，便知原矩阵中对应的主元列同样无关，并张成所有列。</p>
<p>这里必须取<strong>原矩阵中的列</strong>。本例 \(R\) 的主元列是 \((1,0,0)^T,(0,1,0)^T\)，张成平面 \(z=0\)，而原列空间是平面 \(z=x\)。行操作保持列之间的关系，却一般会改变列空间在环境空间中的位置。</p>
<p>由此，前面按主元数定义的秩有了空间解释：</p>
<div class="equation">\[
\operatorname{rank}(A)=r=\dim C(A).
\]</div>
<p>秩数的是列空间需要多少个基向量，而不是原矩阵一共写了多少列。</p>

<h2 id="null-basis">六、为什么特殊解恰好构成零空间的基？</h2>
<p>对同一个矩阵，从 \(R\mathbf x=\mathbf0\) 读出</p>
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
<p>所以 \(B_N=(\mathbf s_1,\mathbf s_2)\) 是零空间的一组基，\(\dim N(A)=2\)。先前说“两个自由参数意味着二维”，现在有了完整理由：两个特殊解不仅生成所有解，而且表示系数唯一。</p>
<p><strong>讲义勘误：</strong>原摘要第 3 页第二个特殊解的末分量印成了 \(-1\)，应为 \(+1\)。代入原矩阵检查：\(A\mathbf s_2=-\mathbf a_1+\mathbf a_4=\mathbf0\)；若末分量用 \(-1\)，结果为 \(-\mathbf a_1-\mathbf a_4=-2\mathbf a_1\ne\mathbf0\)。这里采用已核验的正号。</p>
<p>一般地，有 \(n-r\) 个自由变量时，逐个把一个自由变量取 1、其余取 0，得到 \(n-r\) 个特殊解。它们张成全部齐次解，因为任意自由变量取值都能按这些特殊解组合；它们无关，因为在自由分量上形成单位矩阵，组合为零便迫使所有系数为零。因此</p>
<div class="equation">\[
\dim N(A)=n-r,\qquad
\dim C(A)+\dim N(A)=n.
\]</div>
<p>这就是 <strong>Rank–Nullity Theorem</strong> 在矩阵情形下的内容。它的右端是输入空间 \(\mathbb R^n\) 的维数。本例 \(A\) 有三行四列，\(C(A)\subseteq\mathbb R^3\)，\(N(A)\subseteq\mathbb R^4\)，两者恰好都为二维；它们所在的空间及向量含义不同，不能当作同一个平面，也不能把它们直接相加成 \(\mathbb R^4\)。</p>
<p>最后把结论接回求解：列空间的基描述<strong>哪些输出能达到</strong>；零空间的基描述<strong>输入怎样变化而不改变输出</strong>。若 \(A\mathbf x=\mathbf b\) 有解，后者便给出全部解中的自由部分。对于 \(n\times n\) 方阵，以下说法相互等价：</p>
<div class="equation">\[
\begin{aligned}
A\text{ is invertible}
&\iff r=n\\
&\iff \text{columns of }A\text{ form a basis of }\mathbb R^n\\
&\iff \text{every }\mathbf b\in\mathbb R^n\text{ has exactly one solution.}
\end{aligned}
\]</div>
<p>因为此时每列有主元保证唯一，每行有主元保证所有右端都相容；行最简形就是 \(I_n\)，所以消元可逆地把 \(A\) 化为单位矩阵。逆矩阵、解的唯一性与基，至此成为同一结构的不同表述。</p>

<h2 id="self-check">理解检查</h2>
<ol>
<li>三个两两不平行的 \(\mathbb R^3\) 向量，为什么仍可能相关？请写出一个非零的系数关系。</li>
<li>平面 \(S=\{(a,a,c)^T\}\) 中，每个向量有三个分量，为什么维数却是 2？它在基 \((\mathbf v_1,\mathbf v_2)\) 下的两个坐标与原来的三个分量有什么区别？</li>
<li>本例的四个列向量为什么张成列空间，却不能一起成为基？能否用 \((\mathbf a_2,\mathbf a_3)\) 换一组基？请分别验证无关与张成。</li>
<li>对零空间的两个特殊解，为什么看第三、第四分量就能证明无关？若只知道两个向量都在零空间中，还缺少哪些依据才能说它们是基？</li>
</ol>
<h2 id="sources">资料与衔接</h2>
<p>依据 MIT OCW 18.06SC 第九讲 <em>Independence, Basis and Dimension</em> 及<a href="/files/mit-18-06sc/unit-1-session-9-summary.pdf">讲义摘要</a>整理。定义、两组矩阵例子、列空间与零空间的基来自摘要；唯一表示的等价证明、基的个数不变的证明、主元列规则与特殊解成基的证明作了展开，并纠正摘要中的符号笔误。摘要注明三向量例子在 Lecture 10 开头讨论，本页用它补充辨析“平面的基”与“整个空间的基”。<a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/">课程来源</a>；笔记按 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 分享。</p>
<p class="note-footer"><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-08/">上一节：有解条件与全部解</a></p>
</section>
