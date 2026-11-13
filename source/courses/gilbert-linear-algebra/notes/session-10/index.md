---
title: 第十节：四个基本子空间｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/session-10/
mathjax: true
toc_number: false
comments: false
---

<style>
.post-block:has(.la-notes--session10) .post-header { margin-bottom: 24px; }
.post-block:has(.la-notes--session10) .post-title { font-size: 26px; line-height: 1.45; color: #253c3c; }
.post-block:has(.la-notes--session10) .breadcrumb { display: none; }
.post-body .la-notes--session10 { max-width: 660px; margin: 0 auto; font-size: 17px; line-height: 1.85; color: #303737; }
.post-body .la-notes--session10 p { margin: 0 0 .9em; text-indent: 0; }
.post-body .la-notes--session10 h2 { margin: 2.4em 0 1em; padding: 0 0 .45em; border-bottom: 1px solid #dce5e3; color: #285d5d; font-size: 22px; line-height: 1.5; }
.post-body .la-notes--session10 h3 { margin: 1.7em 0 .75em; color: #303737; font-size: 18px; line-height: 1.6; }
.post-body .la-notes--session10 [id] { scroll-margin-top: 24px; }
.post-body .la-notes--session10 .note-nav { font-size: 14px; color: #687474; margin-bottom: 18px; }
.post-body .la-notes--session10 .note-nav a { border-color: #ccd8d5; }
.post-body .la-notes--session10 .note-lead { margin: 0 0 1.4em; padding: .1em 0 .1em 14px; border-left: 3px solid #729692; }
.post-body .la-notes--session10 .note-lead p { margin: 0; }
.post-body .la-notes--session10 .reading-links { padding: 14px 0; margin-bottom: 1.6em; border-top: 1px solid #e2e8e6; border-bottom: 1px solid #e2e8e6; font-size: 14px; line-height: 1.8; }
.post-body .la-notes--session10 .reading-links p { margin: 0; }
.post-body .la-notes--session10 .reading-links p + p { margin-top: 8px; }
.post-body .la-notes--session10 .link-row { display: flex; flex-wrap: wrap; column-gap: 18px; row-gap: 6px; }
.post-body .la-notes--session10 .equation { margin: 1em 0 1.15em; padding: 8px 2px; overflow-x: auto; overflow-y: hidden; }
.post-body .la-notes--session10 .MathJax_Display { font-size: 100%; margin: .25em 0 !important; padding: 3px 0; }
.post-body .la-notes--session10 ul, .post-body .la-notes--session10 ol { margin: .65em 0 1.2em; padding-left: 1.35em; }
.post-body .la-notes--session10 li { margin-bottom: .55em; }
.post-body .la-notes--session10 .note-extension { margin: 1.2em 0; padding: 10px 14px; border-left: 2px solid #ccd8d5; font-size: 16px; }
.post-body .la-notes--session10 .note-extension summary { cursor: pointer; color: #285d5d; }
.post-body .la-notes--session10 .note-extension[open] summary { margin-bottom: 1em; }
.post-body .la-notes--session10 .note-table { overflow-x: auto; margin: 1em 0; }
.post-body .la-notes--session10 .note-table table { width: 100%; font-size: 15px; line-height: 1.7; }
.post-body .la-notes--session10 .note-table th, .post-body .la-notes--session10 .note-table td { padding: 10px 12px; vertical-align: top; }
.post-body .la-notes--session10 .note-footer { margin-top: 1.8em; padding-top: 1em; border-top: 1px solid #e2e8e6; font-size: 14px; }
@media (max-width: 600px) {
  .post-block:has(.la-notes--session10) .post-title { font-size: 23px; }
  .post-body .la-notes--session10 { font-size: 16px; line-height: 1.85; }
  .post-body .la-notes--session10 h2 { font-size: 20px; margin-top: 2em; }
  .post-body .la-notes--session10 h3 { font-size: 17px; }
}

.post-body .la-notes--session10 .space-map { display: grid; grid-template-columns: 1fr 54px 1fr; gap: 8px; align-items: center; margin: 1.4em 0; }
.post-body .la-notes--session10 .space-box { border: 1px solid #d5e1de; border-top: 3px solid #729692; padding: 14px; background: #f8faf9; font-size: 15px; line-height: 1.8; }
.post-body .la-notes--session10 .space-box p { margin-bottom: .5em; }
.post-body .la-notes--session10 .space-box p:last-child { margin: 0; }
.post-body .la-notes--session10 .space-arrow { text-align: center; color: #285d5d; font-size: 22px; }
.post-body .la-notes--session10 .space-arrow::after { content: "→"; display: block; }
.post-body .la-notes--session10 .map-caption { font-size: 14px; color: #687474; }
@media (max-width: 520px) {
 .post-body .la-notes--session10 .space-map { grid-template-columns: 1fr; }
 .post-body .la-notes--session10 .space-arrow::after { content: "↓"; }
}
</style>

<section class="la-notes la-notes--session10">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-09/">第九节</a></p>
<div class="note-lead"><p>第九节解释了怎样为列空间、零空间选基。本节沿讲义加入行空间与左零空间：<strong>列空间记录可能的输出，零空间记录列之间的关系；行空间记录方程左端能组合出的系数，左零空间记录行之间的关系。</strong>最后一类关系，正是第八节有解条件的来源。</p></div>
<div class="reading-links">
<p class="link-row"><a href="/files/mit-18-06sc/unit-1-session-10-summary.pdf">讲义摘要</a> <a href="/files/mit-18-06sc/unit-1-session-10-problems.pdf">习题</a> <a href="/files/mit-18-06sc/unit-1-session-10-solutions.pdf">参考解答</a></p>
<p class="link-row"><a href="#four-spaces">四个空间的定义</a> <a href="#example">沿用上一节的矩阵</a> <a href="#row-space">行空间与行秩</a> <a href="#left-nullspace">左零空间</a> <a href="#elimination">消元矩阵怎样给出基</a> <a href="#compatibility">有解条件</a> <a href="#relations">空间之间的联系</a> <a href="#matrix-space">矩阵也是向量</a></p>
</div>

<h2 id="four-spaces">一、先分清：向量在哪里，系数又在哪里？</h2>
<p>设 \(A\in\mathbb R^{m\times n}\)，即有 \(m\) 行、\(n\) 列；\(r\) 是其主元数。映射 \(\mathbf x\mapsto A\mathbf x\) 把 \(n\) 个输入分量变成 \(m\) 个输出分量：</p>
<div class="equation">\[
A:\mathbb R^n\longrightarrow\mathbb R^m,
\qquad\mathbf x\longmapsto\mathbf b=A\mathbf x.
\]</div>
<p>把列记为 \(\mathbf a_1,\ldots,\mathbf a_n\in\mathbb R^m\)。第 \(i\) 行有 \(n\) 个数，把它竖着写，记为 \(\boldsymbol\rho_i=(a_{i1},\ldots,a_{in})^T\in\mathbb R^n\)；所以矩阵的第 \(i\) 行是 \(\boldsymbol\rho_i^T\)。这里 \(T\) 表示转置。全文把空间中的向量统一写成列；谈“行向量的空间”时，用转置把行识别为相应的列向量。</p>
<p>已有的两个空间是</p>
<div class="equation">\[
\begin{aligned}
C(A)&=\left\{\sum_{j=1}^n x_j\mathbf a_j:
\mathbf x\in\mathbb R^n\right\}\subseteq\mathbb R^m,\\
N(A)&=\{\mathbf x\in\mathbb R^n:A\mathbf x=\mathbf0\}.
\end{aligned}
\]</div>
<p>转置后，\(A^T\) 的列恰好是 \(\boldsymbol\rho_1,\ldots,\boldsymbol\rho_m\)。令 \(\mathbf y=(y_1,\ldots,y_m)^T\in\mathbb R^m\)，就有</p>
<div class="equation">\[
A^T\mathbf y=\sum_{i=1}^m y_i\boldsymbol\rho_i
\in\mathbb R^n.
\]</div>
<p>若原方程的第 \(i\) 条为 \(\boldsymbol\rho_i^T\mathbf x=b_i\)，按系数 \(y_i\) 相加，就得到 \((A^T\mathbf y)^T\mathbf x=\mathbf y^T\mathbf b\)。因此，行空间中的向量正是<strong>组合原方程时，左端可能出现的系数向量</strong>；左零空间则挑出那些让左端全部抵消的组合。</p>
<p>因此再定义两个空间：</p>
<ul>
<li><strong>行空间</strong>（row space）是 \(C(A^T)=\operatorname{span}(\boldsymbol\rho_1,\ldots,\boldsymbol\rho_m)\subseteq\mathbb R^n\)，即原矩阵各行的全部线性组合。</li>
<li><strong>左零空间</strong>（left nullspace）是 \(N(A^T)=\{\mathbf y\in\mathbb R^m:A^T\mathbf y=\mathbf0\}\)，即让各行组合为零的全部系数向量。</li>
</ul>
<p><strong>行空间的元素是组合结果，有 \(n\) 个分量；左零空间的元素是组合系数，有 \(m\) 个分量。</strong>正如列向量在 \(\mathbb R^m\) 中，而组合列所用的系数在 \(\mathbb R^n\) 中，这两者不能混淆。四者都是子空间：列空间、行空间是张成空间；两个零空间都是齐次线性方程的解集，封闭性已在第六节证明。</p>
<div class="space-map" role="group" aria-label="四个基本子空间所在的两个环境空间">
<div class="space-box"><p><strong>输入空间 \(\mathbb R^n\)</strong></p><p>行空间 \(C(A^T)\)</p><p>零空间 \(N(A)\)</p></div>
<div class="space-arrow" aria-hidden="true">A</div>
<div class="space-box"><p><strong>输出空间 \(\mathbb R^m\)</strong></p><p>列空间 \(C(A)\)</p><p>左零空间 \(N(A^T)\)</p></div>
</div>
<p class="map-caption">图中按向量的分量数放置各空间；箭头表示原映射 \(A\) 的方向。转置映射 \(A^T\) 的方向相反。两个零空间分别属于两个不同映射。</p>

<h2 id="example">二、同一个矩阵，先保留已经求出的两个基</h2>
<p>讲义沿用第九节的矩阵及其行最简形：</p>
<div class="equation">\[
A=\begin{pmatrix}
1&amp;2&amp;3&amp;1\\
1&amp;1&amp;2&amp;1\\
1&amp;2&amp;3&amp;1
\end{pmatrix},\qquad
R=\begin{pmatrix}
1&amp;0&amp;1&amp;1\\
0&amp;1&amp;1&amp;0\\
0&amp;0&amp;0&amp;0
\end{pmatrix}.
\]</div>
<p>这里 \(m=3,n=4,r=2\)。原矩阵的第 1、2 列构成列空间的基；\(R\mathbf x=\mathbf0\) 的自由变量为 \(x_3,x_4\)，给出零空间的两个基向量：</p>
<div class="equation">\[
\begin{aligned}
B_C&=\big((1,1,1)^T,\ (2,1,2)^T\big),\\
B_N&=\big((-1,-1,1,0)^T,\ (-1,0,0,1)^T\big).
\end{aligned}
\]</div>
<p>第九节已分别验证它们无关且张成，所以 \(\dim C(A)=2\)、\(\dim N(A)=2\)。以下用同一次消元求另外两个空间，并说明为什么它们的维数分别是 \(r\) 与 \(m-r\)。</p>

<h2 id="row-space">三、行操作为什么保持行空间，且行秩等于列秩？</h2>
<h3 id="same-row-space">先证明空间相同，再找基</h3>
<p>一次行操作把某一行换成旧行的线性组合，故新矩阵的每一行都在旧行空间中，新行空间包含于旧行空间。又因行操作可逆，旧行也能由新行组合出来，反向包含也成立。逐次消元，便得到</p>
<div class="equation">\[
C(R^T)=C(A^T).
\]</div>
<p>这说明的是两个集合<strong>完全相等</strong>，不只是维数相同。它不与列空间的变化矛盾：左乘消元矩阵是在重组各行，却是对每一列施加同一个变换。</p>
<p>\(R\) 的非零行张成其行空间，因为其余行全为零。它们也无关：设前 \(r\) 个非零行的组合为零，在第 \(i\) 个主元所在列检查，只有第 \(i\) 行提供一个 1，其余主元行在该列都是零，所以这行的组合系数必须为零。对所有主元列检查，所有系数都为零。</p>
<p>因此，<strong>行最简形的 \(r\) 个非零行，转置后构成原行空间的一组基</strong>。本例为</p>
<div class="equation">\[
B_{\rm row}=\big((1,0,1,1)^T,\ (0,1,1,0)^T\big).
\]</div>
<p>记这两个基向量为 \(\mathbf u_1,\mathbf u_2\)。具体的组合关系也可核验：</p>
<div class="equation">\[
\begin{aligned}
\mathbf u_1&=-\boldsymbol\rho_1+2\boldsymbol\rho_2,
&\mathbf u_2&=\boldsymbol\rho_1-\boldsymbol\rho_2,\\
\boldsymbol\rho_1&=\mathbf u_1+2\mathbf u_2,
&\boldsymbol\rho_2&=\mathbf u_1+\mathbf u_2.
\end{aligned}
\]</div>
<p>又有 \(\boldsymbol\rho_3=\boldsymbol\rho_1\)，所以原来的三行与现在的两条非零行，确实张成同一个空间。本例原来的前两行也能作基，但不能据此制定“一律取原矩阵前 \(r\) 行”的规则：例如三行依次为 \((1,0),(2,0),(0,1)\) 时，秩为 2，前两行却相关。取 \(R\) 的非零行才是上述证明保证的方法。</p>
<h3 id="row-column-rank">为什么两个秩必相同？</h3>
<p>行空间有一组包含 \(r\) 个向量的基，所以</p>
<div class="equation">\[
\dim C(A^T)=r=\dim C(A).
\]</div>
<p>左边称行秩，右边称列秩。这是从消元得到的结论，并非定义时强行令它们相等。再把第九节“秩等于列空间维数”用于 \(A^T\)，就得到 \(\operatorname{rank}(A^T)=\operatorname{rank}(A)=r\)。</p>
<p><strong>相同的是维数，通常不是空间。</strong>本例行空间在 \(\mathbb R^4\) 中，列空间在 \(\mathbb R^3\) 中；一个基向量有四个分量，另一个只有三个，不能相互替用。</p>

<h2 id="left-nullspace">四、左零空间为什么记录行关系？</h2>
<p>取 \(\mathbf y\in\mathbb R^m\)。转置等式给出</p>
<div class="equation">\[
A^T\mathbf y=\mathbf0
\iff \mathbf y^TA=\mathbf0^T
\iff\sum_{i=1}^m y_i\boldsymbol\rho_i=\mathbf0.
\]</div>
<p>中间的 \(\mathbf y^T\) 写在 \(A\) 左侧，所以叫“左零空间”。\(A^T\mathbf y\) 是 \(n\) 分量列向量，\(\mathbf y^TA\) 是 \(n\) 分量行向量；二者互为转置。<strong>“左”指乘法位置，空间中的元素仍记为列向量 \(\mathbf y\)。</strong></p>
<p>本例第三行等于第一行，故 \(-\boldsymbol\rho_1+\boldsymbol\rho_3=\mathbf0\)。按三行的顺序记录系数，得到 \((-1,0,1)^T\in N(A^T)\)。这和第九节“按列关系的系数得到零空间向量”完全对应。</p>
<p>还需证明没有漏掉其他行关系。若 \(\mathbf y=(y_1,y_2,y_3)^T\) 满足 \(A^T\mathbf y=\mathbf0\)，则由 \(\boldsymbol\rho_3=\boldsymbol\rho_1\) 可写成</p>
<div class="equation">\[
(y_1+y_3)\boldsymbol\rho_1+y_2\boldsymbol\rho_2=\mathbf0.
\]</div>
<p>前两行无关，所以 \(y_1+y_3=0\)、\(y_2=0\)。反过来，满足这两式就能组合为零。因此</p>
<div class="equation">\[
N(A^T)=\{t(-1,0,1)^T:t\in\mathbb R\},
\qquad B_L=\big((-1,0,1)^T\big).
\]</div>
<p>基向量非零，且生成全部行关系，所以本例左零空间是一维的。一般地，\(A^T\) 有 \(m\) 列，秩刚已证明为 \(r\)；把第九节的零空间维数公式用于 \(A^T\)，得到</p>
<div class="equation">\[
\dim N(A^T)=m-r.
\]</div>
<p>至此四个维数都有依据：列空间、行空间各为 \(r\)；零空间为 \(n-r\)；左零空间为 \(m-r\)。若某个数为零，对应空间便只有零向量，基为空向量组。</p>

<h2 id="elimination">五、怎样从一次消元中直接读出左零空间的基？</h2>
<p>当然可以单独求解 \(A^T\mathbf y=\mathbf0\)。讲义给出另一种方法：在消元 \(A\) 时，把每条新行如何由原行组合出来也记录下来。</p>
<h3 id="record-operations">为什么在右侧放单位矩阵？</h3>
<p>把 \(m\) 阶单位矩阵 \(I_m\) 放在 \(A\) 右边，作增广矩阵 \([A\mid I_m]\)。对整个增广矩阵实施相同的行操作，把左半块化为 \(R\)，右半块就变成消元矩阵 \(E\)：</p>
<div class="equation">\[
[A\mid I_m]\longrightarrow[R\mid E],
\qquad EA=R.
\]</div>
<p>理由是：若所有行操作的乘积为 \(E\)，左半块变为 \(EA\)，右半块变为 \(EI_m=E\)。这里的目标是把<strong>左半块</strong>化为行最简形，并记录同样的操作；不继续为了右半块的主元改变左半块。\(I_m,E\) 均为 \(m\times m\)，而 \(A,R\) 为 \(m\times n\)。</p>
<p>本例使用与上一节相同的操作：第 2 行减第 1 行，第 3 行减第 1 行，将第 2 行乘以 −1，再作第 1 行减第 2 行的 2 倍。得到</p>
<div class="equation">\[
E=\begin{pmatrix}
-1&amp;2&amp;0\\
1&amp;-1&amp;0\\
-1&amp;0&amp;1
\end{pmatrix},\qquad EA=R.
\]</div>
<p>例如 \(E\) 的末行 \((-1,0,1)\) 表示“取原来第 1 行的 −1 倍、第 2 行的 0 倍、第 3 行的 1 倍相加”。其结果是 \(R\) 的末行，恰为零。因此这行转置后就是左零空间中的向量。前两行也分别记录了上一部分 \(\mathbf u_1,\mathbf u_2\) 的组合系数。</p>
<p>\(E\) 由可逆行操作相乘而成，所以自身可逆，即使 \(A\) 不是方阵也成立。只有当 \(A\) 为可逆方阵、\(R=I_m\) 时，才有 \(E=A^{-1}\)；一般不能把消元矩阵当成原矩阵的逆。</p>
<h3 id="left-basis-proof">末 m − r 行为什么不仅“属于”，而且“构成基”？</h3>
<p>把 \(E\) 的第 \(i\) 行记为 \(\boldsymbol\varepsilon_i^T\)。由 \(EA=R\)，当 \(i&gt;r\) 时，对应的 \(R\) 行为零，所以 \(\boldsymbol\varepsilon_i^TA=\mathbf0^T\)。因此这 \(m-r\) 行的转置都属于 \(N(A^T)\)。</p>
<p><strong>无关性：</strong>若这些行的某个线性组合为零，在前 \(r\) 个位置补零，把其系数排成 \(\mathbf c\in\mathbb R^m\)，就有 \(\mathbf c^TE=\mathbf0^T\)。右乘 \(E^{-1}\) 得 \(\mathbf c=\mathbf0\)，所以所有组合系数都为零。</p>
<p><strong>张成性：</strong>任取 \(\mathbf y\in N(A^T)\)，设 \(\mathbf z^T=\mathbf y^TE^{-1}\)。这只是把行向量 \(\mathbf y^T\) 改写为 \(\mathbf z^TE\)，于是</p>
<div class="equation">\[
\mathbf z^TR=\mathbf z^TEA=\mathbf y^TA=\mathbf0^T.
\]</div>
<p>\(R\) 的第 \(i\) 个主元列是第 \(i\) 个标准单位向量，在这一列检查上式，得到 \(z_i=0\)，对 \(i=1,\ldots,r\) 均成立。因此</p>
<div class="equation">\[
\mathbf y^T=\mathbf z^TE
=\sum_{i=r+1}^{m}z_i\boldsymbol\varepsilon_i^T.
\]</div>
<p>任意左零空间向量都能由这些行的转置表示，张成性得证。结合无关性，<strong>\(E\) 的末 \(m-r\) 行转置后，构成原矩阵左零空间的一组基。</strong>这也独立验证了维数为 \(m-r\)。消元路径可能改变基的具体取法，但不会改变左零空间本身。</p>

<h2 id="compatibility">六、左零空间为什么给出全部有解条件？</h2>
<p>第八节通过消元得到有解条件，现在可以说明它们在原方程中表示什么。若 \(A\mathbf x=\mathbf b\) 有解，那么任取 \(\mathbf y\in N(A^T)\)，都有</p>
<div class="equation">\[
\mathbf y^T\mathbf b
=\mathbf y^TA\mathbf x=0.
\]</div>
<p>\(\mathbf y^T\mathbf b=\sum_{i=1}^m y_i b_i\) 是一个实数。它要求：<strong>原矩阵各行满足的任何线性关系，右端各分量也必须满足。</strong>否则把方程按这些系数组合，就会在左边得到零，右边却得到非零数，产生矛盾。</p>
<p>这不只是必要条件。若对所有 \(\mathbf y\in N(A^T)\) 都有 \(\mathbf y^T\mathbf b=0\)，特别地，上一部分每个基向量 \(\boldsymbol\varepsilon_i\) 都满足 \(\boldsymbol\varepsilon_i^T\mathbf b=0\)，其中 \(i&gt;r\)。而这些数正是 \(E\mathbf b\) 的最后 \(m-r\) 个分量。对增广矩阵消元，</p>
<div class="equation">\[
[A\mid\mathbf b]\longrightarrow[R\mid E\mathbf b].
\]</div>
<p>所有系数零行的右端因此全为零；其余主元方程可以求解。这证明条件也充分。所以</p>
<div class="equation">\[
\mathbf b\in C(A)\iff\bigl(\forall\mathbf y\in N(A^T),\ \mathbf y^T\mathbf b=0\bigr).
\]</div>
<p>实际检查时只需使用左零空间的一组基：若对基向量都为零，线性组合后对整个空间也为零。这个基有 \(m-r\) 个向量，恰好给出第八节的 \(m-r\) 个独立相容条件。其系数行线性无关，因而条件之间没有冗余。</p>
<h3 id="compatibility-example">在本例中，把条件直接算出来</h3>
<p>这里左零空间由 \(\mathbf y=(-1,0,1)^T\) 张成，因此条件只有</p>
<div class="equation">\[
\mathbf y^T\mathbf b=-b_1+b_3=0.
\]</div>
<p>也就是第三条方程与第一条左端相同，右端必须相同。计算变换后的右端则有</p>
<div class="equation">\[
E\mathbf b=\begin{pmatrix}
-b_1+2b_2\\b_1-b_2\\-b_1+b_3
\end{pmatrix}.
\]</div>
<p>相容时，取自由变量 \(x_3=x_4=0\)，便得特解 \(\mathbf x_p=(-b_1+2b_2,b_1-b_2,0,0)^T\)。例如 \(\mathbf b=(1,5,1)^T\) 时，\(\mathbf x_p=(9,-4,0,0)^T\) 确实给出该输出；若把右端改为 \((1,5,2)^T\)，同一个 \(\mathbf y\) 给出 \(\mathbf y^T\mathbf b=1\)，消元末行成为 \(0=1\)，无解。</p>
<p>因此，一个使 \(\mathbf y^TA=0\) 却使 \(\mathbf y^T\mathbf b\ne0\) 的向量，就是可直接核验的<strong>无解依据</strong>。反过来，若方程无解，消元必有某个零行对应非零右端，\(E\) 的那一行恰好给出这样的 \(\mathbf y\)。</p>

<h2 id="relations">七、把四个空间放回同一个结构中</h2>
<div class="note-table">
<table>
<thead><tr><th>空间</th><th>所在空间</th><th>一组基怎样求</th><th>维数</th></tr></thead>
<tbody>
<tr><td>\(C(A)\)</td><td>\(\mathbb R^m\)</td><td>原矩阵对应主元位置的列</td><td>\(r\)</td></tr>
<tr><td>\(N(A)\)</td><td>\(\mathbb R^n\)</td><td>每个自由变量产生的特殊解</td><td>\(n-r\)</td></tr>
<tr><td>\(C(A^T)\)</td><td>\(\mathbb R^n\)</td><td>\(R\) 的非零行的转置</td><td>\(r\)</td></tr>
<tr><td>\(N(A^T)\)</td><td>\(\mathbb R^m\)</td><td>\(E\) 的末 \(m-r\) 行的转置</td><td>\(m-r\)</td></tr>
</tbody>
</table>
</div>
<p>要区分消元对四个空间的作用：<strong>行空间与零空间保持原集合不变</strong>，即 \(C(R^T)=C(A^T)\)、\(N(R)=N(A)\)；列空间一般变为 \(C(R)=E\,C(A)\)，左零空间也一般改变。表格末行求的是原来 \(A\) 的左零空间，不能拿 \(R\) 的零行本身当基。零向量不能作基向量，基来自记录行关系的 \(E\)。</p>
<details class="note-extension">
<summary>为什么左零空间不会像零空间那样直接保留？</summary>
<p>\(R^T=A^TE^T\)，所以 \(R^T\mathbf z=\mathbf0\) 等价于 \(E^T\mathbf z\in N(A^T)\)。即</p>
<div class="equation">\[
N(R^T)=E^{-T}N(A^T),\qquad E^{-T}:=(E^T)^{-1}.
\]</div>
<p>这里矩阵乘集合表示对其中每个向量施加该矩阵。本例 \(N(R^T)\) 由 \((0,0,1)^T\) 张成，而原来的 \(N(A^T)\) 由 \((-1,0,1)^T\) 张成。把前者左乘 \(E^T\)，恰好得到后者。这与 \(E\) 末行转置后给出原左零空间基的规则一致。</p>
</details>
<h3 id="orthogonality-bridge">同一环境空间中的两者，为什么互相垂直？</h3>
<p>下面从定义补充一个联系，供后续正交与投影使用。对 \(\mathbf u,\mathbf v\in\mathbb R^k\)，定义内积 \(\mathbf u^T\mathbf v=\sum_{i=1}^k u_iv_i\)；内积为零就称两向量正交。</p>
<p>\(A\mathbf x=\mathbf0\) 的第 \(i\) 条方程是 \(\boldsymbol\rho_i^T\mathbf x=0\)。所以 \(\mathbf x\) 在零空间中，当且仅当它与每一行、从而与行空间的每个向量正交。同样，\(A^T\mathbf y=\mathbf0\) 当且仅当 \(\mathbf y\) 与列空间的每个向量正交。若用 \(S^\perp\) 表示与子空间 \(S\) 的所有向量正交的向量集合，便有</p>
<div class="equation">\[
\begin{aligned}
N(A)&=C(A^T)^\perp\quad\text{in }\mathbb R^n,\\
N(A^T)&=C(A)^\perp\quad\text{in }\mathbb R^m.
\end{aligned}
\]</div>
<p>“每一行”到“整个行空间”的一步来自线性：若 \(\mathbf x\) 与各 \(\boldsymbol\rho_i\) 的内积都为零，则与任意组合 \(\sum_i c_i\boldsymbol\rho_i\) 的内积也为零；反过来，各行本身就在行空间中。列空间的论证相同。这里的正交关系由方程直接证明，不能只凭维数相加推断。</p>
<p>两对维数分别满足 \(r+(n-r)=n\)、\(r+(m-r)=m\)。它们与求解的联系是：\(N(A)\) 决定已有解能怎样变化，\(N(A^T)\) 决定给定右端是否相容。满列秩 \(r=n\) 时，前者只有零向量，有解就唯一；满行秩 \(r=m\) 时，后者只有零向量，所有右端都有解。讲义习题正是从这两个方向检验理解。</p>

<h2 id="matrix-space">八、讲义最后为什么转向“矩阵组成的空间”？</h2>
<p>前面的基向量都写成一列数，但向量空间的定义只要求加法、数乘及其公理，并不要求元素必须是列向量。所有 \(3\times3\) 实矩阵组成的集合 \(M_3(\mathbb R)\)，用逐项相加与数乘，也构成向量空间；零向量就是零矩阵。</p>
<p>用 \(E_{ij}\) 表示第 \(i\) 行、第 \(j\) 列为 1，其余位置为 0 的矩阵。它与前面的消元矩阵 \(E\) 是不同对象；下标用于指定唯一的非零位置。任意矩阵 \(B=(b_{ij})\) 都能唯一写成</p>
<div class="equation">\[
B=\sum_{i=1}^3\sum_{j=1}^3 b_{ij}E_{ij}.
\]</div>
<p>张成性由逐项展开得到；若组合为零，逐个位置检查，所有系数就都为零，故这 9 个矩阵线性无关。因此 \(\dim M_3(\mathbb R)=9\)。这里讨论的是矩阵作为<strong>空间中的元素</strong>；它自身乘法的性质不参与向量空间的定义。</p>
<p>讲义列出三个子空间：上三角矩阵空间 \(U\)、对称矩阵空间 \(S\)、对角矩阵空间 \(D\)。这些形状条件都在加法、数乘下保持，并且包含零矩阵，故都是子空间。它们的基也能逐项构造：</p>
<ul>
<li><strong>上三角：</strong>允许非零的位置满足 \(i\le j\)，取这 6 个位置上的 \(E_{ij}\) 作基。每个允许位置的值就是唯一系数，所以 \(\dim U=6\)。</li>
<li><strong>对称：</strong>对角线上取 \(E_{11},E_{22},E_{33}\)，再取 \(E_{12}+E_{21},E_{13}+E_{31},E_{23}+E_{32}\)。每对对称位置共用一个系数，表示存在且唯一，所以 \(\dim S=6\)。</li>
<li><strong>对角：</strong>取 \(E_{11},E_{22},E_{33}\) 作基，任意 \(\operatorname{diag}(a,b,c)=aE_{11}+bE_{22}+cE_{33}\)，故 \(\dim D=3\)。\(\operatorname{diag}(a,b,c)\) 表示对角线为 \(a,b,c\)、其余位置为零的矩阵。</li>
</ul>
<p>而且 \(D=U\cap S\)：若矩阵既上三角又对称，下三角位置全零，对称位置的上三角元素便也全零，只剩对角线；反过来，对角矩阵显然同时满足两种条件。\(U,S\) 虽然维数相同，却不是同一空间。</p>
<p>讲义给出的对角空间基并非全用标准矩阵，而是</p>
<div class="equation">\[
G_1=E_{11},\qquad
G_2=E_{11}+3E_{22},\qquad G_3=7E_{33}.
\]</div>
<p>它们仍是基，因为任意对角矩阵都能唯一写成</p>
<div class="equation">\[
\operatorname{diag}(a,b,c)
=\left(a-\frac b3\right)G_1
+\frac b3G_2+\frac c7G_3.
\]</div>
<p>所以“基”取决于无关与张成，既不要求基元素形状最简单，也不要求它们的非零位置完全分开。</p>
<details class="note-extension">
<summary>矩阵的秩，与矩阵空间的维数，有什么区别？</summary>
<p>三阶单位矩阵 \(I_3\) 的秩为 3，意思是其列空间有三个基向量；它作为矩阵空间中的一个元素，所在的 \(M_3(\mathbb R)\) 的维数却为 9。前者是给定矩阵所确定的列空间维数，后者是整个矩阵集合的维数。也不能说“所有秩为 1 的矩阵组成一维空间”：它们不包含零矩阵，而且相加后秩可以变为 2，例如 \(E_{11}+E_{22}\)。</p>
</details>

<h2 id="self-check">理解检查</h2>
<ol>
<li>为什么行空间与列空间的维数相同，却通常不能拿同一组向量作基？本例两类基向量各有几个分量？</li>
<li>\(R\) 末行为零，为什么不能取这个零行作左零空间的基？\(E\) 末行又记录了什么？</li>
<li>若存在一个右端使 \(A\mathbf x=\mathbf b\) 无解，四个空间中哪个空间必含非零向量？这个向量怎样解释矛盾？</li>
<li>把方程换成 \(A^T\mathbf y=\mathbf d\)，先判断 \(\mathbf d\) 属于哪个环境空间；再分别说明有解性、已有解的唯一性由哪两个空间决定。</li>
<li>上三角空间与对称空间都为六维，为什么它们的交却只有三维？请用允许非零的位置与基解释。</li>
</ol>
<h2 id="sources">资料与衔接</h2>
<p>依据 MIT OCW 18.06SC 第十讲 <em>The Four Fundamental Subspaces</em> 及<a href="/files/mit-18-06sc/unit-1-session-10-summary.pdf">讲义摘要</a>整理。四个空间、贯穿矩阵、消元矩阵及矩阵空间的例子来自讲义；行秩等于列秩、消元矩阵末行成基、有解条件的充分性与正交关系作了展开，矩阵子空间的基与维数补上逐项解释。正交关系用于连接后续课程，讲义本节的主体仍是定义、基与维数。</p>
<p>两处记号校正：摘要第 2 页增广式中的 \(I,E\) 应为 \(m\times m\)，并非 \(m\times n\)；参考解答 10.1(a) 提及列空间未覆盖整个空间时，该环境空间应为 \(\mathbb R^m\)，不是 \(\mathbb R^n\)。原 PDF 保留供对照，本页按矩阵大小使用正确记号。<a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/">课程来源</a>；笔记按 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 分享。</p>
<p class="note-footer"><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-09/">上一节：线性无关、基与维数</a></p>
</section>
