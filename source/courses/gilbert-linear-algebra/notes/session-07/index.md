---
title: 第七节：求解 Ax = 0：主元变量与特殊解｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/session-07/
mathjax: true
toc_number: false
comments: false
---

<style>
.post-block:has(.la-notes--session07) .post-header { margin-bottom: 24px; }
.post-block:has(.la-notes--session07) .post-title { font-size: 26px; line-height: 1.45; color: #253c3c; }
.post-block:has(.la-notes--session07) .breadcrumb { display: none; }
.post-body .la-notes--session07 { max-width: 660px; margin: 0 auto; font-size: 17px; line-height: 1.85; color: #303737; }
.post-body .la-notes--session07 p { margin: 0 0 .9em; text-indent: 0; }
.post-body .la-notes--session07 h2 { margin: 2.4em 0 1em; padding: 0 0 .45em; border-bottom: 1px solid #dce5e3; color: #285d5d; font-size: 22px; line-height: 1.5; }
.post-body .la-notes--session07 h3 { margin: 1.7em 0 .75em; color: #303737; font-size: 18px; line-height: 1.6; }
.post-body .la-notes--session07 [id] { scroll-margin-top: 24px; }
.post-body .la-notes--session07 .note-nav { font-size: 14px; color: #687474; margin-bottom: 18px; }
.post-body .la-notes--session07 .note-nav a { border-color: #ccd8d5; }
.post-body .la-notes--session07 .note-lead { margin: 0 0 1.4em; padding: .1em 0 .1em 14px; border-left: 3px solid #729692; }
.post-body .la-notes--session07 .note-lead p { margin: 0; }
.post-body .la-notes--session07 .reading-links { padding: 14px 0; margin-bottom: 1.6em; border-top: 1px solid #e2e8e6; border-bottom: 1px solid #e2e8e6; font-size: 14px; line-height: 1.8; }
.post-body .la-notes--session07 .reading-links p { margin: 0; }
.post-body .la-notes--session07 .reading-links p + p { margin-top: 8px; }
.post-body .la-notes--session07 .link-row { display: flex; flex-wrap: wrap; column-gap: 18px; row-gap: 6px; }
.post-body .la-notes--session07 .equation { margin: 1em 0 1.15em; padding: 8px 2px; overflow-x: auto; overflow-y: hidden; }
.post-body .la-notes--session07 .MathJax_Display { font-size: 100%; margin: .25em 0 !important; padding: 3px 0; }
.post-body .la-notes--session07 ul, .post-body .la-notes--session07 ol { margin: .65em 0 1.2em; padding-left: 1.35em; }
.post-body .la-notes--session07 li { margin-bottom: .55em; }
.post-body .la-notes--session07 .note-extension { margin: 1.2em 0; padding: 10px 14px; border-left: 2px solid #ccd8d5; font-size: 16px; }
.post-body .la-notes--session07 .note-extension summary { cursor: pointer; color: #285d5d; }
.post-body .la-notes--session07 .note-extension[open] summary { margin-bottom: 1em; }
.post-body .la-notes--session07 .note-footer { margin-top: 1.8em; padding-top: 1em; border-top: 1px solid #e2e8e6; font-size: 14px; }
@media (max-width: 600px) {
  .post-block:has(.la-notes--session07) .post-title { font-size: 23px; }
  .post-body .la-notes--session07 { font-size: 16px; line-height: 1.85; }
  .post-body .la-notes--session07 h2 { font-size: 20px; margin-top: 2em; }
  .post-body .la-notes--session07 h3 { font-size: 17px; }
}
</style>

<section class="la-notes la-notes--session07">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-06/">第六节</a></p>
<div class="note-lead"><p>第六节说明了零空间的用途；本节用消元系统求出它：<strong>找主元，选自由变量，再回代得到全部齐次解。</strong></p></div>
<div class="reading-links">
<p class="link-row"><a href="/files/mit-18-06sc/unit-1-session-7-summary.pdf">讲义摘要</a> <a href="/files/mit-18-06sc/unit-1-session-7-problems.pdf">习题</a> <a href="/files/mit-18-06sc/unit-1-session-7-solutions.pdf">参考解答</a></p>
<p class="link-row"><a href="#elimination">消元为何保留零空间</a> <a href="#pivots">主元与自由变量</a> <a href="#special-solutions">特殊解与全部解</a> <a href="#rref">行最简形</a> <a href="#column-space">与列空间的联系</a></p>
</div>

<h2 id="elimination">一、从“观察列关系”到“消元找全部解”</h2>
<p>设 \(A\in\mathbb R^{m\times n}\)，未知向量 \(\mathbf x=(x_1,\ldots,x_n)^T\in\mathbb R^n\)。我们要求出齐次方程 \(A\mathbf x=\mathbf0\) 的<strong>所有</strong>解，即零空间 \(N(A)\)。右端是 \(\mathbb R^m\) 中的零向量；未知向量本身可以非零。</p>
<p>第六节的例子能直接看出列关系。遇到较复杂的矩阵，可以对方程作行消元。每次初等行操作都可逆，若消元结果记作 \(U=EA\)，则 \(E\) 是可逆矩阵，故</p>
<div class="equation">\[
A\mathbf x=\mathbf0
\iff EA\mathbf x=\mathbf0
\iff U\mathbf x=\mathbf0.
\]</div>
<p>正向由左乘 \(E\) 得到，反向由左乘 \(E^{-1}\) 得到。因此 \(N(A)=N(U)\)，既没有丢解，也没有添解。右端为零，所有行操作后仍是零，所以可以省去增广矩阵中那一列零。<strong>消元只改变方程的写法，未知数的顺序保持不变。</strong></p>
<p>这里不要求 \(A\) 可逆或是方阵；需要可逆的是实施行操作的 \(E\)。讲义使用</p>
<div class="equation">\[
A=\begin{pmatrix}
1&amp;2&amp;2&amp;2\\
2&amp;4&amp;6&amp;8\\
3&amp;6&amp;8&amp;10
\end{pmatrix}.
\]</div>
<p>它有三条方程、四个未知数。用“第 \(i\) 行”表示行操作，先作第 2 行减第 1 行的 2 倍、第 3 行减第 1 行的 3 倍：</p>
<div class="equation">\[
A\longrightarrow
\begin{pmatrix}
1&amp;2&amp;2&amp;2\\
0&amp;0&amp;2&amp;4\\
0&amp;0&amp;2&amp;4
\end{pmatrix}.
\]</div>
<p>再作第 3 行减第 2 行，得到</p>
<div class="equation">\[
U=\begin{pmatrix}
1&amp;2&amp;2&amp;2\\
0&amp;0&amp;2&amp;4\\
0&amp;0&amp;0&amp;0
\end{pmatrix}.
\]</div>
<p>第三行现在只表示 \(0=0\)，不再施加约束。原矩阵的第 3 行恰好是前两行之和，因此第三条方程原本就能由前两条推出。</p>

<h2 id="pivots">二、主元如何决定哪些未知数可以自由选择？</h2>
<p>\(U\) 是<strong>行阶梯形矩阵</strong>：零行在底部，每个非零行的第一个非零数位于上一非零行相应位置的右侧。这些作为消元支点的首个非零数称为<strong>主元</strong>，所在列称为主元列；其余列称为自由列。</p>
<p>本例第一个主元是第 1 列的 \(1\)。往下一行找新主元时，第 2 列的剩余部分全为零，无法选取非零支点；因此跳到第 3 列，选 \(2\) 为第二个主元。<strong>主元不必落在主对角线上。</strong></p>
<p>把 \(U\mathbf x=\mathbf0\) 写回方程：</p>
<div class="equation">\[
\begin{cases}
x_1+2x_2+2x_3+2x_4=0,\\
2x_3+4x_4=0.
\end{cases}
\]</div>
<p>从下向上回代，第二式先给出 \(x_3=-2x_4\)；代入第一式，得到 \(x_1=-2x_2+2x_4\)。因此</p>
<div class="equation">\[
\begin{cases}
x_1=-2x_2+2x_4,\\
x_3=-2x_4.
\end{cases}
\]</div>
<p>对应主元列的 \(x_1,x_3\) 称为<strong>主元变量</strong>；对应自由列的 \(x_2,x_4\) 称为<strong>自由变量</strong>。这里“自由”有具体含义：\(x_2,x_4\) 可以各自任取实数，随后 \(x_1,x_3\) 由上式唯一确定。主元变量也是未知数，但它们的值受已选自由变量约束。</p>
<h3 id="rank">秩究竟在数什么？</h3>
<p><strong>矩阵的秩定义为列空间的维数</strong>，记作 \(\operatorname{rank}(A)=\dim C(A)\)。也就是说，要用多少个线性无关的列向量，才能组合出它的所有列。这里“线性无关”指这些向量的线性组合为零时，系数只能全为零；一组既线性无关、又能生成整个空间的向量称为基，基中向量的个数就是维数。等价地，秩是从矩阵各列中最多能选出的线性无关向量数。</p>
<p>本例虽然有四列，但第 2 列是第 1 列的 2 倍，第 4 列是第 1 列的 −2 倍加第 3 列的 2 倍；而第 1、3 列线性无关（证明见第五部分）。因此四列只需要两个独立的向量就能生成，\(\operatorname{rank}(A)=2\)。几何上，所有输出 \(A\mathbf x\) 恰好铺满 \(\mathbb R^3\) 中一个过原点的平面；秩数的是这个输出空间的维数。</p>
<p><strong>为什么消元能算秩？</strong>行最简形中，每个主元列都在自己的主元行取 1，在其余行取 0，因此这些列线性无关；其余列都能由主元列组合出来。所以主元列构成该矩阵列空间的一组基。可逆行操作又保留列之间的全部线性关系，因此原矩阵的对应列也构成基，故</p>
<div class="equation">\[
\boxed{\operatorname{rank}(A)=\dim C(A)=\text{主元数}=r.}
\]</div>
<p>这也解释了为什么换一种消元顺序不会改变秩：最终数出的都是同一个列空间的维数。对齐次方程而言，每个主元行确定一个主元变量，共有 \(r\) 个这样的约束；剩下 \(n-r\) 个未知数可以自由选择。本例四个未知数、两个主元，便留下两个自由变量。它们与零空间维数的关系接下来证明。</p>

<h2 id="special-solutions">三、为什么只需构造两个特殊解？</h2>
<p>令 \(x_2=s\)、\(x_4=t\)，其中 \(s,t\) 是可以独立取任意实数的参数。回代结果给出</p>
<div class="equation">\[
\mathbf x=
\begin{pmatrix}-2s+2t\\s\\-2t\\t\end{pmatrix}
=\begin{pmatrix}-2s\\s\\0\\0\end{pmatrix}
+\begin{pmatrix}2t\\0\\-2t\\t\end{pmatrix}.
\]</div>
<p>这一步只是按分量相加的规则，把含 \(s\) 和含 \(t\) 的部分分别收集起来；再从两个向量中分别提出公因子 \(s,t\)，得到</p>
<div class="equation">\[
\mathbf x=s\begin{pmatrix}-2\\1\\0\\0\end{pmatrix}
+t\begin{pmatrix}2\\0\\-2\\1\end{pmatrix}.
\]</div>
<p><strong>两个固定向量中的数，依次是各分量里 \(s\) 与 \(t\) 的系数。</strong>例如 \(x_1=-2s+2t\)，所以第一个位置分别填 \(-2,2\)；\(x_2=s=1s+0t\)，所以第二个位置分别填 \(1,0\)。其余分量同理。这是对全部解的表达式作代数整理，并没有增加新的条件。</p>
<p>将两个固定向量分别记为 \(\mathbf s_1,\mathbf s_2\)。下标只是区分这两个向量。它们由如下选择得到：</p>
<ul>
<li>取 \((x_2,x_4)=(1,0)\)，回代得到 \(\mathbf s_1=(-2,1,0,0)^T\)。</li>
<li>取 \((x_2,x_4)=(0,1)\)，回代得到 \(\mathbf s_2=(2,0,-2,1)^T\)。</li>
</ul>
<p>这种“依次将一个自由变量取为 1，其余自由变量取为 0，再回代”的解，讲义称为 <strong>special solutions（特殊解）</strong>。它们不同于第六节非齐次方程中的 particular solution（特解）：这里每个特殊解都满足 \(A\mathbf s_i=\mathbf0\)，用来生成零空间。</p>
<p><strong>为什么没有漏解？</strong>任一解必须满足前面的回代关系，因此把它的第 2、4 分量分别记为 \(s,t\)，就必定等于 \(s\mathbf s_1+t\mathbf s_2\)。反过来，任意 \(s,t\) 给出的向量都满足两条方程，因而也是原方程的解。所以</p>
<div class="equation">\[
\boxed{N(A)=\{s\mathbf s_1+t\mathbf s_2:s,t\in\mathbb R\}.}
\]</div>
<p>例如 \(s=t=1\) 给出 \((0,1,-2,1)^T\)，也在零空间中。特殊解不是全部解的逐个列举，而是用来组合出全部解的向量。</p>
<p><strong>为什么它们构成基？</strong>上式已经证明它们张成零空间。若 \(c_1\mathbf s_1+c_2\mathbf s_2=\mathbf0\)，比较第 2 分量得到 \(c_1=0\)，比较第 4 分量得到 \(c_2=0\)。所以它们线性无关，构成 \(N(A)\) 的一组基，\(\dim N(A)=2\)。零空间的元素有四个分量，但在这组基下由两个系数唯一确定。</p>
<p>这个论证适用于一般的 \(m\times n\) 矩阵：若有 \(r\) 个主元，就有 \(n-r\) 个自由变量。阶梯形方程允许先任取自由变量，再逐个回代求出主元变量。按上述方法构造的 \(n-r\) 个特殊解张成全部解；比较自由变量对应的分量，又能证明它们线性无关。因此</p>
<div class="equation">\[
\boxed{\dim N(A)=n-r.}
\]</div>
<p>这里 \(n\) 是列数，即未知数个数。若没有自由变量，齐次方程只有零解，零空间维数为 0；本例有两个自由变量，故有无穷多个解。</p>

<h2 id="rref">四、继续消元：把回代关系直接写在矩阵里</h2>
<p>\(U\) 已足以回代求解。讲义再把主元上方消成零，并把主元缩放为 1：先作第 1 行减第 2 行，再把第 2 行除以 2，得到</p>
<div class="equation">\[
U\longrightarrow
\begin{pmatrix}
1&amp;2&amp;0&amp;-2\\
0&amp;0&amp;2&amp;4\\
0&amp;0&amp;0&amp;0
\end{pmatrix}
\longrightarrow
R=\begin{pmatrix}
1&amp;2&amp;0&amp;-2\\
0&amp;0&amp;1&amp;2\\
0&amp;0&amp;0&amp;0
\end{pmatrix}.
\]</div>
<p>这里 \(R\) 是新得到的矩阵名称；\(\mathbb R\) 则表示实数集。\(R\) 称为<strong>行最简形</strong>（reduced row echelon form，rref）：它是行阶梯形，每个主元为 1，且主元所在列的其余元素全为零。</p>
<p>\(R\mathbf x=\mathbf0\) 直接写出 \(x_1+2x_2-2x_4=0\)、\(x_3+2x_4=0\)，也就是前面的回代结果。所有操作仍为行操作，所以 \(N(A)=N(U)=N(R)\)。</p>
<details class="note-extension"><summary>讲义中的分块式：为什么出现 −F 和单位矩阵？</summary>
<p>为把主元列放在一起，本例按第 1、3、2、4 列重新排列，同时把未知向量的分量按同一顺序写成 \(\widetilde{\mathbf x}=(x_1,x_3,x_2,x_4)^T\)。波浪号仅表示采用了新的排列。</p>
<div class="equation">\[
\widetilde R=
\begin{pmatrix}
1&amp;0&amp;2&amp;-2\\
0&amp;1&amp;0&amp;2\\
0&amp;0&amp;0&amp;0
\end{pmatrix}
=\begin{pmatrix}I_2&amp;F\\0&amp;0\end{pmatrix},\qquad
F=\begin{pmatrix}2&amp;-2\\0&amp;2\end{pmatrix}.
\]</div>
<p>\(I_2\) 是二阶单位矩阵；分块式中的 \(0\) 表示相应大小的零矩阵。\(F\) 收集自由变量在两条非零方程中的系数。令 \(\mathbf x_P=(x_1,x_3)^T\)、\(\mathbf x_F=(x_2,x_4)^T\)，下标 \(P,F\) 分别标记主元与自由变量组，则方程成为</p>
<div class="equation">\[
\mathbf x_P+F\mathbf x_F=\mathbf0,
\qquad\mathbf x_P=-F\mathbf x_F.
\]</div>
<p>故按新顺序排列的全部解为</p>
<div class="equation">\[
\widetilde{\mathbf x}
=\begin{pmatrix}-F\\I_2\end{pmatrix}\mathbf x_F,
\qquad
\widetilde N:=\begin{pmatrix}-F\\I_2\end{pmatrix}
=\begin{pmatrix}-2&amp;2\\0&amp;-2\\1&amp;0\\0&amp;1\end{pmatrix}.
\]</div>
<p>上半块 \(-F\) 求出主元变量，下半块 \(I_2\) 原样保留自由变量。因此 \(\widetilde R\widetilde N=0\)，\(\widetilde N\) 的列就是新坐标顺序下的特殊解。恢复原来的第 1、2、3、4 分量顺序，得到</p>
<div class="equation">\[
N=\begin{pmatrix}-2&amp;2\\1&amp;0\\0&amp;-2\\0&amp;1\end{pmatrix}
=\begin{pmatrix}\mathbf s_1&amp;\mathbf s_2\end{pmatrix},
\qquad AN=0.
\]</div>
<p>这里 \(N\) 是由基向量组成的矩阵，\(N(A)\) 是由所有齐次解组成的集合，两者要区分。<strong>交换列时必须同步调整未知数的顺序</strong>；否则便改变了原未知向量所满足的方程。本例原来的 \(R\) 中，主元列并未挨在一起，不能直接照搬左上角为单位矩阵的分块式。</p>
</details>

<h2 id="column-space">五、零空间保持不变，列空间通常会改变</h2>
<p>行操作给出 \(U=EA\)，所以 \(U\) 的各列是 \(A\) 的各列经过 \(E\) 作用后的结果，通常已经不在原来的位置。本例 \(U\) 的全部列都满足第三分量为零，而 \(A\) 的第一列 \((1,2,3)^T\) 不满足，故 \(C(U)\ne C(A)\)。求原矩阵的列空间，必须使用<strong>原矩阵的列</strong>。</p>
<p>不过，\(E\) 可逆，列之间的组合关系保持不变。本例从 \(R\) 的列可读出第 2 列是第 1 列的 2 倍，第 4 列是第 1 列的 −2 倍加第 3 列的 2 倍。相同关系对 \(A\) 的列也成立：</p>
<div class="equation">\[
\mathbf a_2=2\mathbf a_1,\qquad
\mathbf a_4=-2\mathbf a_1+2\mathbf a_3.
\]</div>
<p>原矩阵的第 1、3 列因此张成 \(C(A)\)；它们又线性无关：若 \(c_1\mathbf a_1+c_3\mathbf a_3=\mathbf0\)，施加相同的行操作变到 \(R\)，就得到 \((c_1,c_3,0)^T=\mathbf0\)，故系数均为零。所以 \(C(A)\) 的维数为 2，与主元数相同。</p>
<p>一般地，对任意系数 \(c_j\)，可逆性给出 \(\sum_j c_j\mathbf a_j=\mathbf0\iff\sum_j c_jE\mathbf a_j=\mathbf0\)。所以行操作前后，哪些列线性无关、哪些列由其余列生成，都保持一致。这证明了上文计算秩的依据：<strong>列空间本身可能改变，但其维数不变。</strong></p>
<p>两个特殊解也正是上述两条列关系的系数向量：\(-2\mathbf a_1+\mathbf a_2=\mathbf0\)、\(2\mathbf a_1-2\mathbf a_3+\mathbf a_4=\mathbf0\)。消元所做的，就是系统找出这些关系，并证明它们生成所有使列组合为零的系数。</p>

<h2 id="self-check">理解检查</h2>
<p>为什么第 2 列没有主元，却不要求 \(x_2=0\)？若取 \(x_2=3,x_4=-1\)，能否回代得到一个解？为什么构造两个特殊解后，还需证明它们能组合出全部解？交换列时，为什么必须同步调整未知数顺序？</p>
<h2 id="sources">资料与衔接</h2>
<p>依据 MIT OCW 18.06SC 第七讲 <em>Solving Ax = 0: Pivot Variables, Special Solutions</em> 及讲义摘要整理。矩阵、消元过程、特殊解与分块形式来自本讲；零空间基的完整性论证及列空间说明作了展开。下一节将用行最简形处理一般右端 \(A\mathbf x=\mathbf b\)，第九节再系统讨论线性无关、基与维数。<a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/">课程来源</a>；笔记按 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 分享。</p>
<p class="note-footer"><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/session-06/">上一节：列空间与零空间</a></p>
</section>
