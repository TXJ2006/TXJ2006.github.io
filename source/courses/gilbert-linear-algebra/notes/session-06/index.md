---
title: 第六节：列空间与零空间｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/session-06/
mathjax: true
toc_number: false
comments: false
---

<style>
.post-block:has(.la-notes--session06) .post-header { margin-bottom: 24px; }
.post-block:has(.la-notes--session06) .post-title { font-size: 26px; line-height: 1.45; color: #253c3c; }
.post-block:has(.la-notes--session06) .breadcrumb { display: none; }
.post-body .la-notes--session06 { max-width: 660px; margin: 0 auto; font-size: 17px; line-height: 1.85; color: #303737; }
.post-body .la-notes--session06 p { margin: 0 0 .9em; text-indent: 0; }
.post-body .la-notes--session06 h2 { margin: 2.4em 0 1em; padding: 0 0 .45em; border-bottom: 1px solid #dce5e3; color: #285d5d; font-size: 22px; line-height: 1.5; }
.post-body .la-notes--session06 h3 { margin: 1.7em 0 .75em; color: #303737; font-size: 18px; line-height: 1.6; }
.post-body .la-notes--session06 [id] { scroll-margin-top: 24px; }
.post-body .la-notes--session06 .note-nav { font-size: 14px; color: #687474; margin-bottom: 18px; }
.post-body .la-notes--session06 .note-nav a { border-color: #ccd8d5; }
.post-body .la-notes--session06 .note-lead { margin: 0 0 1.4em; padding: .1em 0 .1em 14px; border-left: 3px solid #729692; }
.post-body .la-notes--session06 .note-lead p { margin: 0; }
.post-body .la-notes--session06 .reading-links { padding: 14px 0; margin-bottom: 1.6em; border-top: 1px solid #e2e8e6; border-bottom: 1px solid #e2e8e6; font-size: 14px; line-height: 1.8; }
.post-body .la-notes--session06 .reading-links p { margin: 0; }
.post-body .la-notes--session06 .reading-links p + p { margin-top: 8px; }
.post-body .la-notes--session06 .link-row { display: flex; flex-wrap: wrap; column-gap: 18px; row-gap: 6px; }
.post-body .la-notes--session06 .equation { margin: 1em 0 1.15em; padding: 8px 2px; overflow-x: auto; overflow-y: hidden; }
.post-body .la-notes--session06 .MathJax_Display { font-size: 100%; margin: .25em 0 !important; padding: 3px 0; }
.post-body .la-notes--session06 ul, .post-body .la-notes--session06 ol { margin: .65em 0 1.2em; padding-left: 1.35em; }
.post-body .la-notes--session06 li { margin-bottom: .55em; }
.post-body .la-notes--session06 .note-extension { margin: 1.2em 0; padding: 10px 14px; border-left: 2px solid #ccd8d5; font-size: 16px; }
.post-body .la-notes--session06 .note-extension summary { cursor: pointer; color: #285d5d; }
.post-body .la-notes--session06 .note-extension[open] summary { margin-bottom: 1em; }
.post-body .la-notes--session06 .note-footer { margin-top: 1.8em; padding-top: 1em; border-top: 1px solid #e2e8e6; font-size: 14px; }
@media (max-width: 600px) {
  .post-block:has(.la-notes--session06) .post-title { font-size: 23px; }
  .post-body .la-notes--session06 { font-size: 16px; line-height: 1.85; }
  .post-body .la-notes--session06 h2 { font-size: 20px; margin-top: 2em; }
  .post-body .la-notes--session06 h3 { font-size: 17px; }
}
</style>

<section class="la-notes la-notes--session06">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-05/">第五节</a></p>
<div class="note-lead"><p>固定矩阵 \(A\)，研究 \(A\mathbf x=\mathbf b\) 有两个问题：<strong>哪些 \(\mathbf b\) 有解？有解时，解是否唯一？</strong>列空间与零空间分别从这两个问题中引出。</p></div>
<div class="reading-links">
<p class="link-row"><a href="/files/mit-18-06sc/unit-1-session-6-summary.pdf">讲义摘要</a> <a href="/files/mit-18-06sc/unit-1-session-6-problems.pdf">习题</a> <a href="/files/mit-18-06sc/unit-1-session-6-solutions.pdf">参考解答</a></p>
<p class="link-row"><a href="#definitions">为什么引入两个空间</a> <a href="#column-space">哪些右端有解</a> <a href="#nullspace">求出全部齐次解</a> <a href="#solution-set">非齐次解集</a> <a href="#subspaces">补充：子空间的交与并</a></p>
</div>

<h2 id="definitions">一、从方程的有解性与唯一性出发</h2>
<p>先明确 \(A\mathbf x=\mathbf b\) 中三个对象的角色：\(A\) 是已知的 \(m\times n\) 矩阵，\(\mathbf b\in\mathbb R^m\) 是给定的目标向量，而</p>
<div class="equation">\[
\mathbf x=\begin{pmatrix}x_1\\\vdots\\x_n\end{pmatrix}\in\mathbb R^n
\]</div>
<p>是把 \(n\) 个未知数排成一列得到的<strong>未知向量</strong>。求解就是找出它的各个分量。把矩阵乘法看作输入到输出的规则时，\(\mathbf x\) 称为输入，\(A\mathbf x\) 称为输出；解方程则要求这个输出恰好等于 \(\mathbf b\)。</p>
<p>前几节用消元求解一个给定的方程组。现在固定 \(A\in\mathbb R^{m\times n}\)，让右端 \(\mathbf b\in\mathbb R^m\) 变化：我们希望知道<strong>哪些右端有解，以及有解时能否唯一确定 \(\mathbf x\)</strong>。以下均讨论实向量空间。</p>
<p>先看有解性。记 \(A\) 的列为 \(\mathbf a_1,\ldots,\mathbf a_n\in\mathbb R^m\)。这只是为每一列取名；列空间还包括它们的任意实数倍与任意线性组合。由第一节的列观点，</p>
<div class="equation">\[
A\mathbf x=x_1\mathbf a_1+\cdots+x_n\mathbf a_n.
\]</div>
<p>其中 \(x_j\) 是一个数，是第 \(j\) 列 \(\mathbf a_j\) 的系数；\(\mathbf x\) 把这些系数收集成一个向量。解 \(A\mathbf x=\mathbf b\)，就是寻找一组列向量的系数，使其组合等于 \(\mathbf b\)。因此，方程有解，恰好意味着 \(\mathbf b\) 属于<strong>全部列向量的线性组合所成的集合</strong>。第五节已经把这样的集合称为张成空间；在这里，它就是矩阵的列空间。</p>
<p><strong>定义 1（列空间）。</strong>矩阵 \(A\) 的列空间为</p>
<div class="equation">\[
C(A):=\operatorname{span}\{\mathbf a_1,\ldots,\mathbf a_n\}
=\{A\mathbf x:\mathbf x\in\mathbb R^n\}\subseteq\mathbb R^m.
\]</div>
<h3 id="same-output">已有一个解后，怎样判断还有没有别的解？</h3>
<p>假设已经找到一个解，把它记作 \(\mathbf x_p\)，因此 \(A\mathbf x_p=\mathbf b\)。下标 \(p\) 来自英文 <em>particular solution</em>（特解），是标明“已经选定的一个具体解”的标签；\(\mathbf x_p\) 整体仍是一个向量，\(p\) 不是乘数，也不是分量编号。</p>
<p>现在考察另一个可能的输入 \(\mathbf x\)，把两次输入的差记为 \(\mathbf z:=\mathbf x-\mathbf x_p\)。于是 \(\mathbf x=\mathbf x_p+\mathbf z\)：\(\mathbf z\in\mathbb R^n\) 表示在已有解上<strong>增加的输入改变量</strong>。任何输入都能这样写，但它是否仍是解，需要检查。</p>
<p>利用矩阵乘法的分配律，再代入已知的 \(A\mathbf x_p=\mathbf b\)，改变后的输出为</p>
<div class="equation">\[
\begin{aligned}
A(\mathbf x_p+\mathbf z)
&amp;=A\mathbf x_p+A\mathbf z\\
&amp;=\mathbf b+A\mathbf z.
\end{aligned}
\]</div>
<p>这里 \(A\mathbf z\) 就是<strong>输出的改变量</strong>。要让新输入仍得到原来的目标 \(\mathbf b\)，输出的改变量必须为零。因此</p>
<div class="equation">\[
A(\mathbf x_p+\mathbf z)=\mathbf b
\iff \mathbf b+A\mathbf z=\mathbf b
\iff A\mathbf z=\mathbf0.
\]</div>
<p>符号 \(\iff\) 表示“当且仅当”：这三个条件彼此等价。整行表达的是<strong>给已有解加上 \(\mathbf z\) 后仍是解，当且仅当 \(\mathbf z\) 被矩阵送到零向量</strong>。若这样的 \(\mathbf z\) 可以非零，就得到另一个解；若只能取零向量，原来的解就唯一。</p>
<p>因此，研究右端为零的方程 \(A\mathbf z=\mathbf0\)，就是找出所有<strong>不改变输出的输入改变量</strong>。这正是下面零空间定义的用途。</p>
<p><strong>定义 2（零空间）。</strong>矩阵 \(A\) 的零空间为</p>
<div class="equation">\[
N(A):=\{\mathbf z\in\mathbb R^n:A\mathbf z=\mathbf0\}
\subseteq\mathbb R^n.
\]</div>
<p>若 \(N(A)=\{\mathbf0\}\)，已有的解无法改变，因而唯一；若其中存在非零向量，就可把它加到已有的解上，得到另一个解。这里“零”限定的是<strong>输出</strong>，零空间中的输入可以非零。</p>
<h3 id="small-example">例：哪些目标可达，输入又能怎样改变？</h3>
<p>先用一个小矩阵算清两个集合。这里有两个未知数 \(u,v\)，把它们合写为 \(\mathbf x=(u,v)^T\)，令</p>
<div class="equation">\[
B=\begin{pmatrix}1&amp;2\\2&amp;4\end{pmatrix},\qquad
B\begin{pmatrix}u\\v\end{pmatrix}
=\begin{pmatrix}u+2v\\2(u+2v)\end{pmatrix}.
\]</div>
<p><strong>列空间看输出。</strong>两列是 \((1,2)^T\)、\((2,4)^T\)，但列空间还包含 \((3,6)^T\)、\((-1,-2)^T\) 等所有组合。令 \(s=u+2v\)；它可以取任意实数，因此</p>
<div class="equation">\[
C(B)=\left\{\begin{pmatrix}s\\2s\end{pmatrix}:s\in\mathbb R\right\}.
\]</div>
<p>这条直线包含全部可能输出。于是 \(B\mathbf x=(3,6)^T\) 有解，而 \(B\mathbf x=(3,7)^T\) 无解：任何输出的第二分量都必须是第一分量的两倍。<strong>列空间给出对右端的限制条件。</strong></p>
<p><strong>零空间看输入。</strong>要使上面的输出为零，只需 \(u+2v=0\)。令 \(v=t\)，便得到</p>
<div class="equation">\[
N(B)=\left\{t\begin{pmatrix}-2\\1\end{pmatrix}:t\in\mathbb R\right\}.
\]</div>
<p>例如输入 \((-2,1)^T\) 的两个分量都非零，却满足 \(B(-2,1)^T=\mathbf0\)。两个列贡献相互抵消。这正是零空间可以包含非零输入的原因。</p>
<p>对目标 \(\mathbf b=(3,6)^T\)，解方程就是求出所有满足 \(u+2v=3\)、\(2u+4v=6\) 的数对 \((u,v)\)。第二式是第一式的两倍，所以只需满足 \(u+2v=3\)。例如 \((u,v)=(3,0)\) 是一组解；令 \(v=t\)，则 \(u=3-2t\)，全部未知向量可写成</p>
<div class="equation">\[
\mathbf x=\begin{pmatrix}3\\0\end{pmatrix}
+t\begin{pmatrix}-2\\1\end{pmatrix},\qquad t\in\mathbb R.
\]</div>
<p>在这个式子中，特解是 \(\mathbf x_p=(3,0)^T\)，输入改变量是 \(\mathbf z=t(-2,1)^T\)。沿 \((-2,1)^T\) 改变输入，输出始终是 \((3,6)^T\)。<strong>零空间给出保持同一输出的全部输入增量。</strong>固定 \(B\) 后，无论换成哪个可达目标，这些增量都相同。</p>
<details class="note-extension"><summary>线性映射中的名称：像与核</summary>
<p>从线性映射 \(T_A\colon\mathbb R^n\to\mathbb R^m,\ \mathbf x\mapsto A\mathbf x\) 看，\(C(A)\) 是映射的像，\(N(A)\) 是映射的核。两者的元素角色不同：列空间收集可能的<strong>输出</strong>，位于 \(\mathbb R^m\)；零空间收集输出为零的<strong>输入</strong>，位于 \(\mathbb R^n\)。方程 \(A\mathbf z=\mathbf0\) 中的零向量属于 \(\mathbb R^m\)。</p>
</details>
<p><strong>命题 1。</strong>\(C(A)\) 与 \(N(A)\) 分别是 \(\mathbb R^m\)、\(\mathbb R^n\) 的子空间。</p>
<p><strong>证明。</strong>列空间是张成空间，结论由第五节已证的命题得到。零空间含零向量；若 \(\mathbf u,\mathbf v\in N(A)\)，则对任意 \(a,b\in\mathbb R\)，</p>
<div class="equation">\[
A(a\mathbf u+b\mathbf v)
=aA\mathbf u+bA\mathbf v=\mathbf0.
\]</div>
<p>故 \(a\mathbf u+b\mathbf v\in N(A)\)，由子空间判据得证。\(\square\)</p>

<h2 id="column-space">二、讲义的矩阵：哪些右端能够得到？</h2>
<p>下面回到讲义中的矩阵</p>
<div class="equation">\[
A=\begin{pmatrix}
1&amp;1&amp;2\\2&amp;1&amp;3\\3&amp;1&amp;4\\4&amp;1&amp;5
\end{pmatrix}
=\begin{pmatrix}\mathbf a_1&amp;\mathbf a_2&amp;\mathbf a_3\end{pmatrix}.
\]</div>
<p>先直接观察列之间的关系：\(\mathbf a_3=\mathbf a_1+\mathbf a_2\)。于是</p>
<div class="equation">\[
A\mathbf x
=(x_1+x_3)\mathbf a_1+(x_2+x_3)\mathbf a_2.
\]</div>
<p>第三列的系数可以合并到前两列的系数中，因此</p>
<div class="equation">\[
C(A)=\operatorname{span}\{\mathbf a_1,\mathbf a_2\}
=\left\{\begin{pmatrix}p+q\\2p+q\\3p+q\\4p+q\end{pmatrix}:p,q\in\mathbb R\right\}.
\]</div>
<p>这里已经知道前两列能组合出整个列空间，还要问：能否再删去其中一列而保持张成空间不变？这需要把第一节简短提过的判据写清楚。</p>
<p><strong>定义 3（线性无关与线性相关）。</strong>设 \(\mathbf v_1,\ldots,\mathbf v_r\) 属于同一个实向量空间。若对任意实系数 \(c_1,\ldots,c_r\) 都有</p>
<div class="equation">\[
c_1\mathbf v_1+\cdots+c_r\mathbf v_r=\mathbf0
\quad\Longrightarrow\quad c_1=\cdots=c_r=0,
\]</div>
<p>则称这组向量<strong>线性无关</strong>；若存在一组<strong>不全为零</strong>的实系数使该组合等于零，则称它们<strong>线性相关</strong>。“不全为零”只要求至少一个系数非零。此时可以把对应向量移到等式一边并除以其系数，将它表示为其余向量的线性组合；因此删除它不改变张成空间。</p>
<p>本例中，\(\mathbf a_1+\mathbf a_2-\mathbf a_3=\mathbf0\)，系数 \((1,1,-1)\) 不全为零，所以<strong>三列线性相关</strong>，第三列可以删去。再检查前两列：若 \(p\mathbf a_1+q\mathbf a_2=\mathbf0\)，前两分量给出 \(p+q=0,2p+q=0\)，相减得 \(p=0\)，继而 \(q=0\)。所以<strong>前两列线性无关</strong>。</p>
<p>这还保证表示的唯一性：若 \(p\mathbf a_1+q\mathbf a_2=p'\mathbf a_1+q'\mathbf a_2\)，相减得到零组合，故 \(p=p'\)、\(q=q'\)。因此，列空间的每个向量都能由这两列表示，而且系数唯一。</p>
<p><strong>定义 4（基与维数，本例所需的有限维情形）。</strong>若一组向量既张成空间 \(V\)，又线性无关，就称为 \(V\) 的一组<strong>基</strong>。若 \(V\) 有一组有限基，就称它是有限维的。有限维空间的任意两组基所含向量数相同；这个数称为 \(V\) 的<strong>维数</strong>，记作 \(\dim V\)。基的向量数与选择无关这一结论，将在后续专讲基与维数时证明。</p>
<p>本例的 \(\mathbf a_1,\mathbf a_2\) 满足上述两个条件，因此构成 \(C(A)\) 的一组基，\(\dim C(A)=2\)。这就是“\(\mathbb R^4\) 中的二维平面”的含义：向量有四个分量，但在这组基下由两个可任意取值的系数唯一确定。<strong>向量的分量数与所在子空间的维数要区分。</strong></p>
<p><strong>命题 2（可解性）。</strong>对 \(\mathbf b\in\mathbb R^4\)，\(A\mathbf x=\mathbf b\) 有解，当且仅当 \(\mathbf b\in C(A)\)。这是列空间定义与 \(A\mathbf x=\sum_jx_j\mathbf a_j\) 的直接结果。</p>
<p>本例还能把“属于列空间”改写为可检查的条件：若 \(\mathbf b=(b_1,b_2,b_3,b_4)^T\in C(A)\)，则它的四个分量构成等差数列，即</p>
<div class="equation">\[
\boxed{b_2-b_1=b_3-b_2=b_4-b_3.}
\]</div>
<p>反过来，若三个差都等于 \(p\)，令 \(q=b_1-p\)，就有 \(b_i=ip+q\)（\(i=1,2,3,4\)），故 \(\mathbf b=p\mathbf a_1+q\mathbf a_2\in C(A)\)。因此这也是有解的充要条件。例如 \((1,2,3,4)^T\) 对应有解，\((1,2,3,5)^T\) 对应无解。</p>

<h2 id="nullspace">三、同一个列关系，怎样确定全部齐次解？</h2>
<p><strong>齐次方程</strong>指右端为零向量的线性方程 \(A\mathbf x=\mathbf0\)，它的解称为齐次解；全部齐次解组成 \(N(A)\)。本例的未知向量是 \(\mathbf x=(x_1,x_2,x_3)^T\in\mathbb R^3\)，右端则是 \(\mathbb R^4\) 中的零向量。</p>
<h3 id="one-homogeneous-solution">先找到一个解，再确定有没有其他方向的解</h3>
<p>矩阵乘法把输入的三个分量分别作为三列的系数。由于 \(\mathbf a_3=\mathbf a_1+\mathbf a_2\)，取系数 \((1,1,-1)\) 就有</p>
<div class="equation">\[
A\begin{pmatrix}1\\1\\-1\end{pmatrix}
=1\mathbf a_1+1\mathbf a_2-1\mathbf a_3
=\mathbf0.
\]</div>
<p>这说明 \((1,1,-1)^T\) 是一个非零齐次解。它的任意实数倍也都是解，但此时还没有证明：是否存在不在这些倍数中的其他解？所以要从<strong>任意未知向量</strong>出发求解。</p>
<h3 id="all-homogeneous-solutions">展开任意输入，求出三个分量之间的约束</h3>
<p>将第三列替换成前两列之和，再合并同类项：</p>
<div class="equation">\[
\begin{aligned}
A\mathbf x
&amp;=x_1\mathbf a_1+x_2\mathbf a_2+x_3\mathbf a_3\\
&amp;=x_1\mathbf a_1+x_2\mathbf a_2+x_3(\mathbf a_1+\mathbf a_2)\\
&amp;=(x_1+x_3)\mathbf a_1+(x_2+x_3)\mathbf a_2.
\end{aligned}
\]</div>
<p>前一节已证明 \(\mathbf a_1,\mathbf a_2\) 线性无关。因此，它们的组合为零时，<strong>各自的系数必须为零</strong>。这里两个系数分别是 \(x_1+x_3\)、\(x_2+x_3\)，所以</p>
<div class="equation">\[
A\mathbf x=\mathbf0
\iff
\begin{cases}
x_1+x_3=0,\\
x_2+x_3=0
\end{cases}
\iff
\begin{cases}
x_1=-x_3,\\
x_2=-x_3.
\end{cases}
\]</div>
<p>这两条等式并没有限制 \(x_3\) 必须取哪个数。它可以任取实数；一旦选定 \(x_3\)，\(x_1,x_2\) 就随之确定。为把结果写成前面那个向量的倍数，<strong>令 \(t=-x_3\)</strong>，于是 \(x_3=-t\)、\(x_1=x_2=t\)，得到</p>
<div class="equation">\[
\mathbf x=\begin{pmatrix}t\\t\\-t\end{pmatrix}
=t\begin{pmatrix}1\\1\\-1\end{pmatrix},\qquad t\in\mathbb R.
\]</div>
<p>\(t\) 是一个可任意取值的实数参数。取 \(t=0\) 得零解，取 \(t=1\) 得 \((1,1,-1)^T\)，取 \(t=-2\) 得 \((-2,-2,2)^T\)。也可以令 \(s=x_3\)，写成 \(s(-1,-1,1)^T\)；因为 \(s\) 遍历全部实数，得到的是同一个解集。</p>
<p>上述推导说明<strong>每个齐次解都必须有这个形式</strong>；反过来，任取 \(t\in\mathbb R\)，代回有 \(A\mathbf x=t(\mathbf a_1+\mathbf a_2-\mathbf a_3)=\mathbf0\)，说明<strong>每个这种形式的向量确实都是解</strong>。两方面合起来，才确定没有漏解：</p>
<div class="equation">\[
\boxed{N(A)=\left\{t\begin{pmatrix}1\\1\\-1\end{pmatrix}:t\in\mathbb R\right\}.}
\]</div>
<p>花括号表示一个集合，冒号后的 \(t\in\mathbb R\) 指明参数取值范围：把所有实数 \(t\) 代入，收集得到的全部向量。这个集合是 \(\mathbb R^3\) 中经过原点、方向为 \((1,1,-1)^T\) 的直线。</p>
<p>按前面的基定义，\((1,1,-1)^T\) 张成整个 \(N(A)\)；且若 \(c(1,1,-1)^T=\mathbf0\)，第一分量就给出 \(c=0\)，所以这一个向量组成的向量组线性无关。它因而构成零空间的一组基，\(\dim N(A)=1\)。三个分量由一个参数确定；前面列空间的二维平面则位于输出空间 \(\mathbb R^4\) 中。</p>
<h3 id="nullspace-use">这一结果有什么用：找出保持输出不变的全部输入变化</h3>
<p>将任意输入 \(\mathbf x=(x_1,x_2,x_3)^T\) 改为</p>
<div class="equation">\[
\mathbf x'=\mathbf x+t\begin{pmatrix}1\\1\\-1\end{pmatrix}
=\begin{pmatrix}x_1+t\\x_2+t\\x_3-t\end{pmatrix}.
\]</div>
<p>这里改变的是<strong>列的组合系数</strong>：前两个系数各增加 \(t\)，第三个系数减少 \(t\)，矩阵的列本身保持固定。对应的输出变化为</p>
<div class="equation">\[
A\mathbf x'-A\mathbf x
=A(\mathbf x'-\mathbf x)
=t(\mathbf a_1+\mathbf a_2-\mathbf a_3)
=\mathbf0.
\]</div>
<p>因此 \(A\mathbf x'=A\mathbf x\)。例如，输入 \((1,0,0)^T\) 取 \(t=-1\) 后变成 \((0,-1,1)^T\)，输出仍是 \(\mathbf a_1\)，因为 \(-\mathbf a_2+\mathbf a_3=\mathbf a_1\)。</p>
<p>反过来，若两个输入 \(\mathbf x,\mathbf y\) 产生同一输出，则 \(A(\mathbf y-\mathbf x)=A\mathbf y-A\mathbf x=\mathbf0\)，所以它们的差属于刚求出的 \(N(A)\)。即<strong>本例中，保持输出不变的全部输入变化，恰好就是沿 \((1,1,-1)^T\) 的任意实数倍变化</strong>。下一节据此写出非齐次方程的全部解。</p>

<h2 id="solution-set">四、非齐次解集：零空间的平移</h2>
<p><strong>命题 3（全部解的结构）。</strong>设 \(A\in\mathbb R^{m\times n}\)，且 \(A\mathbf x=\mathbf b\) 至少有一个解 \(\mathbf x_p\)。则它的解集为</p>
<div class="equation">\[
\boxed{\{\mathbf x:A\mathbf x=\mathbf b\}
=\mathbf x_p+N(A)
:=\{\mathbf x_p+\mathbf z:\mathbf z\in N(A)\}.}
\]</div>
<p><strong>证明。</strong>若 \(\mathbf x\) 是解，则 \(A(\mathbf x-\mathbf x_p)=\mathbf b-\mathbf b=\mathbf0\)，故 \(\mathbf x-\mathbf x_p\in N(A)\)。反过来，若 \(\mathbf z\in N(A)\)，则 \(A(\mathbf x_p+\mathbf z)=\mathbf b\)，仍是解。两边相互包含，结论成立。\(\square\)</p>
<p>讲义取 \(\mathbf b=\mathbf a_1=(1,2,3,4)^T\)。一个特解是 \(\mathbf x_p=(1,0,0)^T\)，因此全部解为</p>
<div class="equation">\[
\mathbf x=\begin{pmatrix}1\\0\\0\end{pmatrix}
+t\begin{pmatrix}1\\1\\-1\end{pmatrix},\qquad t\in\mathbb R.
\]</div>
<p>它是一条与 \(N(A)\) 平行的直线，经过 \((1,0,0)^T\)；取 \(t=-1\)，还经过讲义中的 \((0,-1,1)^T\)。它不经过原点，因为 \(A\mathbf0=\mathbf0\ne\mathbf b\)，故不是线性子空间。这种子空间的平移称为<strong>仿射子空间</strong>。</p>
<p>特解的选择不影响整个解集：若 \(\mathbf x'_p\) 是另一特解，两者之差属于 \(N(A)\)，所以 \(\mathbf x'_p+N(A)=\mathbf x_p+N(A)\)。</p>
<p>由此可分别判断：\(\mathbf b\notin C(A)\) 时无解；\(\mathbf b\in C(A)\) 时有解，且恰在 \(N(A)=\{\mathbf0\}\) 时唯一。若零空间包含非零向量，实数参数的不同取值便给出无穷多个解。</p>

<h2 id="subspaces">补充：子空间的交与并</h2>
<p>以下均讨论实向量空间，沿用第五节的子空间判据：非空，且对任意实线性组合封闭。</p>
<p><strong>命题 4。</strong>设 \(S,T\) 是向量空间 \(V\) 的子空间，则交集 \(S\cap T\) 也是子空间。</p>
<p><strong>证明。</strong>两者都包含零向量，故交集非空。任取 \(\mathbf u,\mathbf v\in S\cap T\) 及 \(a,b\in\mathbb R\)。由于两者都对线性组合封闭，\(a\mathbf u+b\mathbf v\) 同时属于 \(S,T\)，因此属于交集。\(\square\)</p>
<p>并集 \(S\cup T\) 则不一定是子空间。例如，令 \(S\) 为 \(\mathbb R^3\) 中的平面 \(z=0\)，\(T\) 为 \(z\) 轴。\((1,0,0)^T\in S\)、\((0,0,1)^T\in T\)，但它们的和 \((1,0,1)^T\) 不在并集中。</p>
<p>若要得到包含两个子空间的最小子空间，应取<strong>和空间</strong>：</p>
<div class="equation">\[
S+T:=\{\mathbf s+\mathbf t:\mathbf s\in S,\ \mathbf t\in T\}
=\operatorname{span}(S\cup T).
\]</div>
<p>这里 \(\operatorname{span}(S\cup T)\) 指从并集中取有限个向量作全部线性组合。将来自 \(S\) 与 \(T\) 的项分别合并，任何组合都可写成 \(\mathbf s+\mathbf t\)；反过来，每个这样的和都是并集中两个向量的线性组合。它对加法、数乘封闭，且包含两个子空间。上述平面与轴的并集不是整个 \(\mathbb R^3\)，和空间却是整个 \(\mathbb R^3\)。</p>

<h2 id="self-check">理解检查</h2>
<p>能否证明等差条件既必要又充分？为什么只找到 \((1,1,-1)^T\) 一个齐次解，还不足以声称已求出零空间？为什么非齐次解集平行于零空间，却不一定等于零空间？</p>
<h2 id="sources">资料与衔接</h2>
<p>依据 MIT OCW 18.06SC 第六讲 <em>Column Space and Nullspace</em> 与讲义摘要整理。二阶矩阵例子用于说明两个空间的用途；子空间的交、并及和空间见文末补充，与习题 6.1 衔接；右端的等差条件、完整证明及线性相关、基与维数的定义为补充。后续课程将用消元、主元变量与自由变量系统求解 \(A\mathbf x=\mathbf0\)。<a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/">课程来源</a>；笔记按 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 分享。</p>
<p class="note-footer"><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-05/">上一节：转置、置换与向量空间</a> · <a href="/courses/gilbert-linear-algebra/notes/session-07/">下一节：求解 Ax = 0</a></p>
</section>
