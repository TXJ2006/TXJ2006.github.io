---
title: 第一节：线性方程的几何｜阅读笔记与习题分析
layout: page
permalink: /courses/gilbert-linear-algebra/notes/session-01/
mathjax: true
toc_number: false
comments: false
---

<style>
.la-notes { line-height: 1.95; }
.la-notes h2 { margin-top: 2.2em; }
.la-notes h3 { margin-top: 1.7em; }
.la-notes .equation { overflow-x: auto; padding: .4em 0; }
.la-notes figure { margin: 1.8em 0 2em; }
.la-notes figure img { display: block; width: 100%; max-width: 560px; height: auto; margin: 0 auto; border: 0; padding: 0; }
.la-notes .image-caption { display: none; }
.la-notes figcaption { font-size: .9em; line-height: 1.7; margin: .8em auto 0; max-width: 600px; color: #444; text-align: left; }
.la-notes .reading-links { padding: .9em 1.1em; border-left: 3px solid #496987; background: #f7f9fa; }
.la-notes .table-wrap { overflow-x: auto; }
.la-notes details { border-left: 2px solid #bbb; padding: .3em 1em; margin: 1em 0; }
.la-notes summary { cursor: pointer; font-weight: bold; }
</style>

<section class="la-notes">
<p><a href="/courses/gilbert-linear-algebra/notes/">← 返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/">课程资料</a></p>
<h1 id="session-01">第一节：线性方程的几何</h1>
<p>本节对应 MIT 18.06SC 的 <em>The Geometry of Linear Equations</em>。我们的出发点是一个直观想法：把给定向量按不同倍数伸缩，再首尾相接，看看能不能到达目标。本篇把这个想法写成准确的数学语言，并补充习题中用到、初学时容易缺少解释的概念。</p>
<p>阅读目标是理解三件事：<strong>线性方程组、列向量的线性组合、矩阵乘向量，是同一个问题的三种表达。</strong>“线性相关”“可逆”和矩阵乘矩阵在这里作必要的补充；后续课程还会系统展开。</p>
<div class="reading-links">
<p><strong>本节资料：</strong><a href="/files/mit-18-06sc/unit-1-session-1-summary.pdf">讲义摘要 PDF</a> · <a href="/files/mit-18-06sc/unit-1-session-1-problems.pdf">习题 PDF</a> · <a href="/files/mit-18-06sc/unit-1-session-1-solutions.pdf">官方解答 PDF</a></p>
<p><strong>阅读顺序：</strong><a href="#vectors">向量与组合</a> → <a href="#three-views">方程组的两幅图</a> → <a href="#linear-map">线性映射与尺寸</a> → <a href="#dependence">解为什么不唯一</a> → <a href="#exercises">三道习题分析</a> → <a href="#self-check">理解检查</a>。</p>
</div>

<h2 id="vectors">一、向量、点与线性组合</h2>
<h3 id="s01-detail-01">1. 先把“一次”和“二元”分清</h3>
<p>“二元”表示有两个未知数，例如 \(x,y\)；“一次”表示未知数按线性方式出现。例如 \(2x+y=3\) 是二元一次方程。在未知数之间出现 \(x^2\)、\(xy\)、\(\sin x\) 等项，一般就不再是线性方程。二元一次方程在平面中通常表示一条直线；三元一次方程在三维空间中通常表示一个平面。这里“通常”排除了所有未知数系数都为零的退化情形。</p>
<h3 id="s01-detail-02">2. 向量表示位移，点表示位置</h3>
<p>在目前的坐标空间中，一个二维向量可写为 \(\mathbf v=(3,1)^{\mathsf T}\)。上标 \(\mathsf T\) 表示把横排的坐标转成竖排的列；它仍然只有两个坐标。零向量的每个坐标都是 0，两个向量相等则要求对应坐标全部相等。几何上，这个向量表示“向右走 3，向上走 1”的位移。</p>
<p>点 \((3,1)\) 则表示一个位置。从原点出发画这个位移，终点恰好就是该点，因此我们经常用点的坐标表示从原点指向它的向量。但箭头可以平移：只要方向和长度不变，仍表示同一个向量。例如从 \((2,2)\) 指向 \((3,1)\) 的箭头，表示的向量是 \((1,-1)\)，并不是 \((3,1)\)。</p>
<h3 id="s01-detail-03">3. 数乘与加法分别做什么？</h3>
<p>数乘 \(a\mathbf u\) 把向量长度变成原来的 \(|a|\) 倍；当 \(a&lt;0\) 时还要反向，当 \(a=0\) 时得到零向量。向量相加按坐标进行：</p>
<div class="equation">\[
a\begin{pmatrix}u_1\\u_2\end{pmatrix}+b\begin{pmatrix}v_1\\v_2\end{pmatrix}
=\begin{pmatrix}au_1+bv_1\\au_2+bv_2\end{pmatrix}.
\]</div>
<p>几何上，把第二支箭头的起点平移到第一支箭头的终点，最后从最初起点指向最终终点，就是和向量。这样的“数乘再相加”叫作<strong>线性组合</strong>，\(a,b\) 叫作组合的系数。</p>
<p><strong>平移箭头是作图的方法，线性组合本身由数乘和加法组成。</strong>这里也不把“实际走过的路线”当作答案；我们关心的是最终位移。交换相加顺序会改变途中经过的位置，但不会改变终点。</p>

<h2 id="three-views">二、一个方程组，为什么能画出两种不同的图？</h2>
<p>贯穿本节的例子是：</p>
<div class="equation">\[
\begin{cases}x+y=3,\\x-y=1.\end{cases}
\qquad
A=\begin{pmatrix}1&amp;1\\1&amp;-1\end{pmatrix},\quad
\mathbf x=\begin{pmatrix}x\\y\end{pmatrix},\quad
\mathbf b=\begin{pmatrix}3\\1\end{pmatrix}.
\]</div>
<h3 id="s01-detail-04">1. 行的视角：找两条直线的共同点</h3>
<p>第一个方程规定一条直线，第二个方程规定另一条直线。求共同满足两个条件的 \((x,y)\)，就是求两条直线的交点。两式相加得 \(2x=4\)，所以 \(x=2\)，再代回得 \(y=1\)。交点为 \((2,1)\)。</p>
<figure>
<img src="/images/courses/gilbert-linear-algebra/session-01/tikz_row_picture.svg" alt="在未知数平面中，直线 x+y=3 与 x-y=1 相交于点 (2,1)。" width="560" loading="lazy">
<figcaption>图 1：行的视角。横纵轴记录未知数 \(x,y\)；图中的交点 \((2,1)\) 就是方程组的解。</figcaption>
</figure>
<h3 id="s01-detail-05">2. 列的视角：找组合给定向量的倍数</h3>
<p>把两个方程的对应坐标放在一起，可以等价地改写为：</p>
<div class="equation">\[
x\underbrace{\begin{pmatrix}1\\1\end{pmatrix}}_{\mathbf a_1}
+y\underbrace{\begin{pmatrix}1\\-1\end{pmatrix}}_{\mathbf a_2}
=\underbrace{\begin{pmatrix}3\\1\end{pmatrix}}_{\mathbf b}.
\]</div>
<p>问题变成：第一支给定向量取多少倍，第二支取多少倍，才能得到目标 \(\mathbf b\)？取 \(x=2,y=1\)，便有：</p>
<div class="equation">\[
2\mathbf a_1+\mathbf a_2
=\begin{pmatrix}2\\2\end{pmatrix}+\begin{pmatrix}1\\-1\end{pmatrix}
=\begin{pmatrix}3\\1\end{pmatrix}.
\]</div>
<figure>
<img src="/images/courses/gilbert-linear-algebra/session-01/tikz_column_picture.svg" alt="在输出平面中，从原点沿两倍 a1 到达 (2,2)，再沿 a2 到达目标 b=(3,1)。" width="560" loading="lazy">
<figcaption>图 2：列的视角。先走 \(2\mathbf a_1\)，再走 \(\mathbf a_2\)；虚线的原点到目标箭头表示总位移。坐标轴 \(b_1,b_2\) 记录输出向量的坐标。短箭头还标出了原始的 \(\mathbf a_1,\mathbf a_2\)。</figcaption>
</figure>
<p><strong>这两幅图中的点承担不同的角色。</strong>第一幅图是在输入平面找 \((x,y)=(2,1)\)；第二幅图是在输出平面凑出 \(\mathbf b=(3,1)\)。解记录“怎样组合”，目标向量记录“要组合成什么”。本例两边恰好都是二维，容易混淆；以后输入与输出还可能处于不同维数的空间。</p>
<h3 id="s01-detail-06">3. 矩阵乘向量，是把列组合写得更简洁</h3>
<p>矩阵 \(A\) 的两列正是 \(\mathbf a_1,\mathbf a_2\)。矩阵乘向量的规则是：</p>
<div class="equation">\[
A\mathbf x=
\begin{pmatrix}1&amp;1\\1&amp;-1\end{pmatrix}\begin{pmatrix}x\\y\end{pmatrix}
=x\mathbf a_1+y\mathbf a_2
=\begin{pmatrix}x+y\\x-y\end{pmatrix}.
\]</div>
<p>所以 \(A\mathbf x=\mathbf b\) 既是原来的方程组，也是原来的列向量组合。逐行“对应相乘再相加”与按列线性组合，来自同一个公式，并非两套不相干的规定。</p>

<h2 id="linear-map">三、矩阵与线性映射：输入和输出有多少个坐标？</h2>
<p>固定矩阵 \(A\)，定义 \(T(\mathbf x)=A\mathbf x\)，就得到一个从输入到输出的规则。这个规则满足：</p>
<div class="equation">\[
T(\alpha\mathbf u+\beta\mathbf v)
=\alpha T(\mathbf u)+\beta T(\mathbf v).
\]</div>
<p>这里 \(\alpha,\beta\) 是任意实数，\(\mathbf u,\mathbf v\) 是任意输入向量。满足这个性质的映射叫作<strong>线性映射</strong>。矩阵运算保持数乘与加法，因此“先组合再映射”与“先映射再组合”结果相同。特别地，\(T(\mathbf0)=\mathbf0\)。一般的平移 \(\mathbf x\mapsto\mathbf x+\mathbf c\) 在 \(\mathbf c\ne\mathbf0\) 时不满足这一点，因此不是线性映射。</p>
<p>一个 \(m\times n\) 矩阵有 \(m\) 行、\(n\) 列。它的每列有 \(m\) 个坐标，需要 \(n\) 个系数来组合：</p>
<div class="equation">\[
A\mathbf x=x_1\mathbf a_1+\cdots+x_n\mathbf a_n,
\qquad A:\mathbb R^n\longrightarrow\mathbb R^m.
\]</div>
<p>\(\mathbb R^n\) 表示所有实数 \(n\) 元组组成的空间。因此，<strong>列数决定输入有几个坐标；行数决定输出有几个坐标。</strong>例如 \(3\times2\) 矩阵用两个系数组合两个三维列向量，所以它把二维输入变成三维输出。它不保证能到达每一个三维目标。</p>
<p>若 \(\mathbf e_1,\ldots,\mathbf e_n\) 是标准基向量，即第 \(j\) 个向量的第 \(j\) 个坐标为 1、其余为 0，那么 \(A\mathbf e_j=\mathbf a_j\)。这解释了列向量的作用：<strong>第 \(j\) 列记录第 \(j\) 个标准方向经过映射后变成什么。</strong></p>

<h2 id="dependence">四、为什么有时无解，有时又有无穷多个解？</h2>
<p>考虑两个共线向量：</p>
<div class="equation">\[
\mathbf u=\begin{pmatrix}1\\2\end{pmatrix},\qquad
\mathbf v=\begin{pmatrix}2\\4\end{pmatrix}=2\mathbf u.
\]</div>
<p>它们不同，但第二个向量没有提供新的方向。任意组合都满足：</p>
<div class="equation">\[
x\mathbf u+y\mathbf v=(x+2y)\mathbf u
=\begin{pmatrix}x+2y\\2x+4y\end{pmatrix}.
\]</div>
<p>所有能得到的输出组成一条过原点的直线 \(b_2=2b_1\)。由给定向量的所有线性组合组成的集合，叫作它们<strong>张成的空间</strong>；由矩阵各列张成的空间叫作<strong>列空间</strong>。\(A\mathbf x=\mathbf b\) 有解，当且仅当目标 \(\mathbf b\) 属于这个列空间。</p>
<figure>
<img src="/images/courses/gilbert-linear-algebra/session-01/tikz_dependent_line.svg" alt="共线向量 u=(1,2) 与 v=(2,4) 只能生成直线 b2=2b1 上的输出，(3,6) 在直线上，(3,7) 不在。" width="560" loading="lazy">
<figcaption>图 3：两个向量沿同一方向。两支箭头均从原点出发，沿同一直线，\(\mathbf v\) 的长度是 \(\mathbf u\) 的两倍；整条蓝线是所有可达输出。圆点 \((3,6)\) 可达，叉号 \((3,7)\) 不可达。为便于标注，横纵坐标采用不同的显示比例。</figcaption>
</figure>
<h3 id="s01-detail-07">1. 目标在直线上：不止一种组合</h3>
<p>要得到 \((3,6)\)，只需 \(x+2y=3\)。例如：</p>
<div class="table-wrap"><table>
<thead><tr><th>输入 \((x,y)\)</th><th>组合</th><th>输出</th></tr></thead>
<tbody><tr><td>\((3,0)\)</td><td>\(3\mathbf u\)</td><td>\((3,6)\)</td></tr>
<tr><td>\((1,1)\)</td><td>\(\mathbf u+\mathbf v\)</td><td>\((3,6)\)</td></tr>
<tr><td>\((-1,2)\)</td><td>\(-\mathbf u+2\mathbf v\)</td><td>\((3,6)\)</td></tr></tbody>
</table></div>
<p>全部解是 \((x,y)=(3-2t,t)\)，其中 \(t\) 可以取任意实数，所以有无穷多个解。几何上，增加一份 \(\mathbf v\)，同时减少两份 \(\mathbf u\)，总位移不变。</p>
<p>在行的视角中，方程组是 \(x+2y=3\)、\(2x+4y=6\)。两条直线重合，公共点有无穷多个。这条<strong>输入平面里的解直线</strong>，与图 3 中<strong>输出平面里的可达直线</strong>不同。</p>
<h3 id="s01-detail-08">2. 目标在直线外：无法到达</h3>
<p>目标改成 \((3,7)\) 时，要同时满足 \(x+2y=3\)、\(2x+4y=7\)。第一式乘 2 得右端应为 6，与第二式矛盾，所以无解。在输入平面里，这是两条不同的平行直线；在输出平面里，这是目标不属于列空间。</p>
<h3 id="s01-detail-09">3. “存在解”和“解唯一”是两个问题</h3>
<p>向量组<strong>线性无关</strong>，指的是组合成零向量只能使用全部为零的系数；若存在系数不全为零的零组合，就叫作<strong>线性相关</strong>。零组合说明，可以沿某种非零的系数变化而不改变输出。</p>
<p>因此，列向量线性无关时，任意一个可达目标的表示都是唯一的；列向量线性相关时，只要有解，就可以在一个解上加上零组合，得到其他解。对实数线性方程组而言，解的情况只有<strong>无解、唯一解、无穷多个解</strong>，不会恰好只有两个解。</p>
<p>唯一性的理由很直接：若 \(A\mathbf x=A\mathbf y=\mathbf b\)，相减得 \(A(\mathbf x-\mathbf y)=\mathbf0\)。列向量线性无关就迫使 \(\mathbf x-\mathbf y=\mathbf0\)，即两个解相同。反之，若 \(A\mathbf z=\mathbf0\) 且 \(\mathbf z\ne\mathbf0\)，已有一个解 \(\mathbf x_0\) 时，\(\mathbf x_0+t\mathbf z\) 对每个实数 \(t\) 都是解。</p>

<h2 id="higher-dimensions">五、高维空间画不出来，怎样继续理解？</h2>
<p>向量组合的规则与坐标个数无关。例如四维空间中的两个向量</p>
<div class="equation">\[
\mathbf u=(1,0,1,0)^{\mathsf T},\qquad
\mathbf v=(0,1,0,1)^{\mathsf T}
\]</div>
<p>有任意组合 \(a\mathbf u+b\mathbf v=(a,b,a,b)^{\mathsf T}\)。因此，第一与第三个坐标必须相同，第二与第四个坐标必须相同。目标 \((2,3,2,3)\) 可以到达，\((2,3,2,4)\) 不能到达。</p>
<p>这两个向量张成的是<strong>四维空间中的二维平面</strong>：“四维”描述环境允许有几个坐标，“二维”描述这个集合需要几个独立参数。我们没有画出四维图，仍能准确判断可达性和自由度。</p>
<p>行的视角也能推广：\(n\) 个未知数的一个线性方程，在系数不全为零时定义一个 \((n-1)\) 维的<strong>超平面</strong>。多个方程要求这些集合有共同点。但每增加一个方程，并不一定增加一个独立限制；它可能与原有方程重复，也可能造成矛盾。不能只靠“有几个方程”来判断剩余自由度。</p>

<h2 id="exercises">六、习题 1.1：零组合、线性相关与共面</h2>
<h3 id="s01-detail-10">题意与术语</h3>
<p>给定</p>
<div class="equation">\[
\mathbf w_1=\begin{pmatrix}1\\2\\3\end{pmatrix},\quad
\mathbf w_2=\begin{pmatrix}4\\5\\6\end{pmatrix},\quad
\mathbf w_3=\begin{pmatrix}7\\8\\9\end{pmatrix}.
\]</div>
<p>寻找 \(x_1\mathbf w_1+x_2\mathbf w_2+x_3\mathbf w_3=\mathbf0\) 的组合，并判断向量组是 independent（线性无关）还是 dependent（线性相关），以及三个向量位于什么几何对象中。</p>
<p>原题没有在求组合的句子中明确写“系数不全为零”，但结合后面的独立性判断，应该寻找这样的<strong>非平凡零组合</strong>。全取零当然总是成立，却不能用于证明线性相关。</p>
<h3 id="s01-detail-11">方法一：先观察坐标的规律</h3>
<p>相邻向量的差相同：</p>
<div class="equation">\[
\mathbf w_2-\mathbf w_1=\mathbf w_3-\mathbf w_2
=\begin{pmatrix}3\\3\\3\end{pmatrix}.
\]</div>
<p>移项便得：</p>
<div class="equation">\[
\boxed{\mathbf w_1-2\mathbf w_2+\mathbf w_3=\mathbf0}.
\]</div>
<p>所以一组答案是 \((x_1,x_2,x_3)=(1,-2,1)\)。系数不全为零，故三个向量<strong>线性相关</strong>。</p>
<h3 id="s01-detail-12">方法二：把向量等式展开成方程组</h3>
<div class="equation">\[
\begin{cases}
x_1+4x_2+7x_3=0,\\
2x_1+5x_2+8x_3=0,\\
3x_1+6x_2+9x_3=0.
\end{cases}
\]</div>
<p>第二式减第一式的 2 倍，得 \(-3x_2-6x_3=0\)；第三式减第一式的 3 倍，得 \(-6x_2-12x_3=0\)。后一个条件只是前一个的 2 倍，没有提供新的限制。</p>
<p>于是 \(x_2=-2x_3\)。代回第一式得 \(x_1=x_3\)。令 \(x_3=t\)，得到<strong>全部解</strong>：</p>
<div class="equation">\[
\boxed{\begin{pmatrix}x_1\\x_2\\x_3\end{pmatrix}
=t\begin{pmatrix}1\\-2\\1\end{pmatrix},\qquad t\in\mathbb R.}
\]</div>
<p>\(t=0\) 给出平凡解；任意 \(t\ne0\) 都给出非平凡零组合。官方解答选的是 \(t=1\)，并非只有这一组系数。</p>
<h3 id="s01-detail-13">为什么答案填“平面”，但三个终点又共线？</h3>
<p>由 \(\mathbf w_3=2\mathbf w_2-\mathbf w_1\)，第三个向量已经能由前两个表示。与此同时，\(\mathbf w_1\) 与 \(\mathbf w_2\) 不是倍数关系：若 \(\mathbf w_2=c\mathbf w_1\)，第一坐标要求 \(c=4\)，第二坐标却要求 \(c=5/2\)，矛盾。所以前两个向量张成一个过原点的平面，第三个也位于该平面内。</p>
<p>在输出空间使用坐标 \((x,y,z)\)，三个向量都满足 \(x-2y+z=0\)。这个方程给出的过原点平面，正是它们张成的平面。</p>
<figure>
<img src="/images/courses/gilbert-linear-algebra/session-01/tikz_common_plane.svg" alt="过原点的平面内有三个向量 w1、w2、w3，三个终点位于一条不经过原点的虚线上。" width="560" loading="lazy">
<figcaption>图 4：三维平面内关系的斜投影示意，不表示实际角度和长度。蓝色箭头从同一原点出发；三个终点在虚线上，但原点不在该虚线上。因此，终点共线与向量组张成二维平面可以同时成立。</figcaption>
</figure>
<p>三个终点位于直线 \(\mathbf w_1+s(3,3,3)^{\mathsf T}\) 上：取 \(s=0,1,2\) 就得到三个终点。这条直线不经过原点；例如若第一个坐标为零，则 \(s=-1/3\)，第二个坐标却为 1。<strong>“向量共线”要求它们都是同一向量的倍数；仅仅终点共线并不够。</strong></p>
<p>还要区分两个空间：全部系数解 \(t(1,-2,1)\) 是<strong>输入空间中的一条直线</strong>；列向量张成的 \(x-2y+z=0\) 是<strong>输出空间中的一个平面</strong>。这里输入输出都各有三个坐标，但它们扮演不同角色。</p>
<h3 id="s01-detail-14">题目为什么说矩阵不可逆？</h3>
<p>把这三个向量放成列：</p>
<div class="equation">\[
W=\begin{pmatrix}1&amp;4&amp;7\\2&amp;5&amp;8\\3&amp;6&amp;9\end{pmatrix}.
\]</div>
<p>“可逆”指存在一个矩阵 \(W^{-1}\)，能够把运算撤销，满足 \(W^{-1}W=WW^{-1}=I\)；\(I\) 是对每个向量都保持不变的单位矩阵。但这里 \(W\mathbf0=\mathbf0\)，且 \(W(1,-2,1)^{\mathsf T}=\mathbf0\)。两个不同输入产生相同输出，便无法从输出唯一恢复输入。</p>
<p>若逆矩阵存在，将后一个等式左乘 \(W^{-1}\)，就会得到 \((1,-2,1)^{\mathsf T}=\mathbf0\)，矛盾。所以 \(W\) 不可逆。几何上，它也只能输出上述平面内的向量，无法到达整个三维空间。这里不需要先学会行列式，就能说明不可逆的原因。</p>

<h2 id="exercise-12">七、习题 1.2：矩阵乘向量的两种算法</h2>
<p>计算</p>
<div class="equation">\[
\begin{pmatrix}1&amp;2&amp;0\\2&amp;0&amp;3\\4&amp;1&amp;1\end{pmatrix}
\begin{pmatrix}3\\-2\\1\end{pmatrix}.
\]</div>
<h3 id="s01-detail-15">按列：把输入看成三个组合系数</h3>
<div class="equation">\[
3\begin{pmatrix}1\\2\\4\end{pmatrix}
-2\begin{pmatrix}2\\0\\1\end{pmatrix}
+\begin{pmatrix}0\\3\\1\end{pmatrix}
=\begin{pmatrix}3\\6\\12\end{pmatrix}
+\begin{pmatrix}-4\\0\\-2\end{pmatrix}
+\begin{pmatrix}0\\3\\1\end{pmatrix}
=\boxed{\begin{pmatrix}-1\\9\\11\end{pmatrix}}.
\]</div>
<h3 id="s01-detail-16">按行：分别计算输出的每一个坐标</h3>
<div class="equation">\[
\begin{aligned}
b_1&amp;=1\cdot3+2\cdot(-2)+0\cdot1=-1,\\
b_2&amp;=2\cdot3+0\cdot(-2)+3\cdot1=9,\\
b_3&amp;=4\cdot3+1\cdot(-2)+1\cdot1=11.
\end{aligned}
\]</div>
<p>两种算法结果相同。按列看清“哪些向量按什么系数组合”；按行便于逐项计算。输入中的 \(-2\) 表示将第二列反向并放大两倍。</p>
<p>本题已经给出了输入 \((3,-2,1)\)，要求算输出；而求解方程组时，已知的是目标输出，要求找输入。<strong>“计算 \(A\mathbf x\)”与“求解 \(A\mathbf x=\mathbf b\)”的已知量、未知量不同。</strong></p>

<h2 id="exercise-13">八、习题 1.3：矩阵乘矩阵为什么是这个尺寸？</h2>
<p>题目判断：“\(3\times2\) 矩阵 \(A\) 乘 \(2\times3\) 矩阵 \(B\)，得到 \(3\times3\) 的 \(AB\)。”答案是<strong>正确</strong>。为了理解原因，先补上矩阵乘矩阵的含义。</p>
<h3 id="s01-detail-17">1. 把第二个矩阵的每一列交给第一个矩阵</h3>
<p>设 \(B\) 的三列是 \(\mathbf b_1,\mathbf b_2,\mathbf b_3\)，每列都有两个坐标。那么：</p>
<div class="equation">\[
B=[\,\mathbf b_1\ \mathbf b_2\ \mathbf b_3\,],
\qquad
AB=[\,A\mathbf b_1\ A\mathbf b_2\ A\mathbf b_3\,].
\]</div>
<p>\(A\) 有两列，所以可以接收每个二坐标的 \(\mathbf b_j\)；\(A\) 有三行，所以每个结果 \(A\mathbf b_j\) 有三个坐标。把三个结果按列放好，就得到三行三列的矩阵。</p>
<h3 id="s01-detail-18">2. 一般尺寸规则与逐项公式</h3>
<div class="equation">\[
\boxed{A_{m\times n}B_{n\times p}=(AB)_{m\times p}.}
\]</div>
<p>这行公式中的下标仅用于标记矩阵尺寸。两边中间的 \(n\) 必须匹配，表示 \(B\) 的每列有 \(n\) 个数，恰好够组合 \(A\) 的 \(n\) 列；结果有 \(m\) 行、\(p\) 列。若记 \(A\) 第 \(i\) 行第 \(k\) 列的数为 \(a_{ik}\)，\(B\) 第 \(k\) 行第 \(j\) 列的数为 \(b_{kj}\)，那么：</p>
<div class="equation">\[
(AB)_{ij}=\sum_{k=1}^{n}a_{ik}b_{kj}.
\]</div>
<p>符号 \(\sum\) 表示求和，即将 \(a_{i1}b_{1j},a_{i2}b_{2j},\ldots,a_{in}b_{nj}\) 相加。这就是“取 \(A\) 的第 \(i\) 行与 \(B\) 的第 \(j\) 列，对应相乘再相加”。</p>
<p><strong>尺寸核对：</strong>官方解答 PDF 末行给出的结果是 \(m\times p\)，与上式一致。阅读斜体字母时留意 \(m\) 与 \(n\) 的区别；结果保留的是第一个矩阵的行数和第二个矩阵的列数。</p>
<h3 id="s01-detail-19">3. 从映射理解：先做 B，再做 A</h3>
<p>矩阵 \(B\) 将 \(\mathbb R^p\) 中的输入变为 \(\mathbb R^n\) 中的中间结果，\(A\) 再把它变为 \(\mathbb R^m\) 中的输出：</p>
<div class="equation">\[
\mathbb R^p\xrightarrow{\ B\ }\mathbb R^n\xrightarrow{\ A\ }\mathbb R^m,
\qquad (AB)\mathbf x=A(B\mathbf x).
\]</div>
<p>为什么这个等式成立？将 \(\mathbf x\) 的坐标记为 \(x_j\)，则 \(B\mathbf x=\sum_jx_j\mathbf b_j\)。利用 \(A\) 的线性性质：</p>
<div class="equation">\[
A(B\mathbf x)=A\!\left(\sum_jx_j\mathbf b_j\right)
=\sum_jx_jA\mathbf b_j=(AB)\mathbf x.
\]</div>
<p>这说明前面的乘法规则恰好把两次线性运算合成一次。本题 \(AB\) 的过程是“三个坐标输入 → 两个中间坐标 → 三个输出坐标”，即 \(\mathbb R^3\to\mathbb R^2\to\mathbb R^3\)。注意乘法顺序：\(AB\) 表示先作用 \(B\) 再作用 \(A\)，通常不能交换顺序。本题 \(BA\) 也能计算，但尺寸是 \(2\times2\)。</p>

<h2 id="self-check">九、本节需要真正理解的几件事</h2>
<p>会跟着计算，还需要能解释每一步在问什么。可以先不展开答案，尝试回答：</p>
<ol>
<li>同一个方程组的行图里，交点是 \((2,1)\)，为什么列图里目标却是 \((3,1)\)？</li>
<li>两个不互为倍数的三维向量，是否足以到达每个三维目标？</li>
<li>为什么一组系数不全为零的零组合，意味着已有目标的组合方式可以改变？</li>
<li>\(\mathbf w_1,\mathbf w_2,\mathbf w_3\) 的终点共线，为什么它们张成的空间仍是平面？</li>
<li>\(A\) 是 \(2\times3\)，\(B\) 是 \(3\times4\)，那么 \(AB\) 能否计算，尺寸是什么？\(BA\) 呢？</li>
</ol>
<details><summary>展开核对思路</summary>
<ol>
<li>\((2,1)\) 是输入的两个组合系数；\((3,1)\) 是组合后的输出。两幅图的坐标角色不同。</li>
<li>不能。它们张成过原点的二维平面；可达目标只能在这个平面中。在平面内，每个目标的表示唯一。</li>
<li>若 \(A\mathbf z=\mathbf0\) 且 \(\mathbf z\ne\mathbf0\)，则 \(A(\mathbf x_0+t\mathbf z)=A\mathbf x_0\)，所以可以改变输入而不改变输出。</li>
<li>终点所在的直线不经过原点；前两个向量也不是倍数关系。三个向量只需要两个独立方向来生成。</li>
<li>\(AB\) 可以计算，尺寸是 \(2\times4\)；\(BA\) 的中间尺寸 4 与 2 不匹配，不能计算。</li>
</ol>
</details>
<p>下一节学习消元时，可以带着一个具体问题：<strong>怎样通过计算，识别哪些方程提供了新限制，哪些只是重复，哪些又相互矛盾？</strong></p>

<h2 id="sources">资料与说明</h2>
<ul>
<li>MIT OpenCourseWare，18.06SC <em>Linear Algebra</em>，Fall 2011，Gilbert Strang：<a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/">课程主页</a>及本站上方所列第一讲摘要、习题与解答。</li>
<li>题目编号沿用课程 PDF 的 Problem 1.1—1.3；题 1.1 的教材节次编号按原题保留理解，不将其与本站所收教材版本的节次强行对应。</li>
<li>本篇是结合课程与学习讨论独立整理的中文笔记；推导与图示为本站补充，不把补充概念全部归为第一讲视频已经讲授的内容。课程原资料按其 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 许可使用；本篇笔记与原创图示亦以相同许可分享。</li>
</ul>
<p><a href="#session-01">回到本节开头</a> · <a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a> · <a href="/courses/gilbert-linear-algebra/">返回课程资料</a></p>
</section>
