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
<p>本节的核心：<strong>解线性方程组，就是寻找各列向量的组合系数，使结果等于目标向量。</strong></p>
<div class="reading-links">
<p><a href="/files/mit-18-06sc/unit-1-session-1-summary.pdf">讲义摘要</a> · <a href="/files/mit-18-06sc/unit-1-session-1-problems.pdf">习题</a> · <a href="/files/mit-18-06sc/unit-1-session-1-solutions.pdf">参考解答</a></p>
<p><a href="#vectors">向量</a> · <a href="#three-views">方程组与矩阵</a> · <a href="#linear-map">线性映射</a> · <a href="#dependence">解的情况</a> · <a href="#higher-dimensions">高维</a> · <a href="#exercises">习题 1.1</a> · <a href="#exercise-12">1.2</a> · <a href="#exercise-13">1.3</a></p>
</div>

<h2 id="vectors">一、向量与线性组合</h2>
<p>向量表示位移。例如 \((3,1)\) 表示向右走 3、向上走 1；从原点画出它，终点就是点 \((3,1)\)。箭头可以平移，方向和长度不变就仍是同一个向量。</p>
<p>数乘负责伸缩，负数倍还会反向；加法可以用首尾相接表示。\(a\mathbf u+b\mathbf v\) 叫作<strong>线性组合</strong>，\(a,b\) 是系数，按坐标计算：</p>
<div class="equation">\[
a\begin{pmatrix}u_1\\u_2\end{pmatrix}
+b\begin{pmatrix}v_1\\v_2\end{pmatrix}
=\begin{pmatrix}au_1+bv_1\\au_2+bv_2\end{pmatrix}.
\]</div>
<p>首尾相接只是帮助理解运算；我们关心的是总位移。交换相加顺序，终点不变。</p>

<h2 id="three-views">二、方程组、列向量与矩阵</h2>
<p>考虑二元一次方程组：</p>
<div class="equation">\[
\begin{cases}x+y=3,\\x-y=1.\end{cases}
\]</div>
<p><strong>按行看：</strong>每个方程是一条直线，解是两条直线的交点。两式相加得到 \(x=2\)，代回得 \(y=1\)。</p>
<figure>
<img src="/images/courses/gilbert-linear-algebra/session-01/tikz_row_picture.svg" alt="两条方程直线相交于输入点 (2,1)。" width="560" loading="lazy">
<figcaption>图 1：未知数平面中的交点是 \((2,1)\)。</figcaption>
</figure>
<p><strong>按列看：</strong>把对应坐标放在一起，同一个方程组变成：</p>
<div class="equation">\[
x\underbrace{\begin{pmatrix}1\\1\end{pmatrix}}_{\mathbf a_1}
+y\underbrace{\begin{pmatrix}1\\-1\end{pmatrix}}_{\mathbf a_2}
=\underbrace{\begin{pmatrix}3\\1\end{pmatrix}}_{\mathbf b}.
\]</div>
<p>也就是取 \(2\mathbf a_1+\mathbf a_2=\mathbf b\)。</p>
<figure>
<img src="/images/courses/gilbert-linear-algebra/session-01/tikz_column_picture.svg" alt="从原点沿两倍 a1 到 (2,2)，再沿 a2 到目标 (3,1)。" width="560" loading="lazy">
<figcaption>图 2：先走 \(2\mathbf a_1\)，再走 \(\mathbf a_2\)，到达 \(\mathbf b=(3,1)\)。<strong>解 \((2,1)\) 记录组合系数，目标 \((3,1)\) 记录输出。</strong></figcaption>
</figure>
<p>将两个列向量排成矩阵，就能简写为 \(A\mathbf x=\mathbf b\)：</p>
<div class="equation">\[
\underbrace{\begin{pmatrix}1&amp;1\\1&amp;-1\end{pmatrix}}_A
\underbrace{\begin{pmatrix}x\\y\end{pmatrix}}_{\mathbf x}
=x\mathbf a_1+y\mathbf a_2
=\begin{pmatrix}x+y\\x-y\end{pmatrix}
=\begin{pmatrix}3\\1\end{pmatrix}.
\]</div>
<p>因此，矩阵乘向量既可按列作线性组合，也可按行“对应相乘再相加”。</p>

<h2 id="linear-map">三、线性映射与矩阵尺寸</h2>
<p>固定 \(A\)，规则 \(T(\mathbf x)=A\mathbf x\) 把输入变成输出，并保持数乘与加法：</p>
<div class="equation">\[
T(\alpha\mathbf u+\beta\mathbf v)
=\alpha T(\mathbf u)+\beta T(\mathbf v).
\]</div>
<p>这叫作<strong>线性映射</strong>。特别地，\(T(\mathbf0)=\mathbf0\)，所以加上一个非零固定位移的平移不是线性映射。</p>
<p>\(m\times n\) 矩阵有 \(m\) 行、\(n\) 列：用 \(n\) 个系数组合 \(n\) 个 \(m\) 维列向量，得到 \(m\) 维输出：</p>
<div class="equation">\[
A\mathbf x=x_1\mathbf a_1+\cdots+x_n\mathbf a_n,
\qquad A:\mathbb R^n\longrightarrow\mathbb R^m.
\]</div>
<p>\(\mathbb R^n\) 表示实数 \(n\) 元组组成的空间。<strong>列数决定输入坐标数，行数决定输出坐标数。</strong>计算 \(A\mathbf x\) 是已知输入求输出；解 \(A\mathbf x=\mathbf b\) 是已知目标求输入。</p>

<h2 id="dependence">四、什么时候有解，什么时候唯一？</h2>
<p>所有列向量的线性组合组成<strong>列空间</strong>，也就是全部可达输出。目标在列空间中，方程组才有解。</p>
<p>例如 \(\mathbf u=(1,2)^{\mathsf T}\)、\(\mathbf v=(2,4)^{\mathsf T}=2\mathbf u\)；上标 \(\mathsf T\) 表示将横排坐标转成列。任意组合都是：</p>
<div class="equation">\[
x\mathbf u+y\mathbf v=(x+2y)\mathbf u
=\begin{pmatrix}x+2y\\2x+4y\end{pmatrix}.
\]</div>
<figure>
<img src="/images/courses/gilbert-linear-algebra/session-01/tikz_dependent_line.svg" alt="所有输出都位于 b2=2b1 上，(3,6) 可达，(3,7) 不可达。" width="560" loading="lazy">
<figcaption>图 3：可达输出是一条直线。横纵轴显示比例不同。</figcaption>
</figure>
<ul>
<li>目标为 \((3,6)\)：只需 \(x+2y=3\)，全部解为 \((x,y)=(3-2t,t)\)，\(t\) 为任意实数，所以有无穷多种组合。</li>
<li>目标为 \((3,7)\)：不满足第二坐标是第一坐标的两倍，故无解。</li>
</ul>
<p>如果组合成零向量只能把系数全部取零，就叫<strong>线性无关</strong>；如果存在系数不全为零的零组合，就叫<strong>线性相关</strong>。线性相关时，可以改变系数而保持输出不变，例如本例 \(-2\mathbf u+\mathbf v=\mathbf0\)。</p>
<p>所以：<strong>列向量线性无关时，每个可达目标的表示唯一；线性相关时，只要有解，就有无穷多个解。</strong>若 \(A\mathbf z=\mathbf0\) 且 \(\mathbf z\ne\mathbf0\)，一个解 \(\mathbf x_0\) 加上任意 \(t\mathbf z\) 仍是解。</p>

<h2 id="higher-dimensions">五、高维怎样理解？</h2>
<p>高维也按坐标运算，不必先能画出来。例如四维中的两个向量有组合：</p>
<div class="equation">\[
a(1,0,1,0)+b(0,1,0,1)=(a,b,a,b).
\]</div>
<p>因此 \((2,3,2,3)\) 可达，\((2,3,2,4)\) 不可达。这个集合位于四维空间中，却只需要两个独立参数，所以是其中的一个二维平面。环境维数与所张成空间的维数不同。</p>

<h2 id="exercises">六、习题 1.1：零组合与共面</h2>
<p>给定 \(\mathbf w_1=(1,2,3)^{\mathsf T}\)、\(\mathbf w_2=(4,5,6)^{\mathsf T}\)、\(\mathbf w_3=(7,8,9)^{\mathsf T}\)，寻找系数不全为零的零组合。</p>
<p>观察 \(\mathbf w_2-\mathbf w_1=\mathbf w_3-\mathbf w_2\)，立即得到：</p>
<div class="equation">\[
\boxed{\mathbf w_1-2\mathbf w_2+\mathbf w_3=\mathbf0}.
\]</div>
<p>系数 \((1,-2,1)\) 不全为零，所以三个向量<strong>线性相关</strong>。若按方程组求解：</p>
<div class="equation">\[
\begin{cases}
x_1+4x_2+7x_3=0,\\
2x_1+5x_2+8x_3=0,\\
3x_1+6x_2+9x_3=0.
\end{cases}
\]</div>
<p>第二式减第一式的 2 倍，得 \(-3x_2-6x_3=0\)；第三式减第一式的 3 倍，所得条件与它重复。因此 \(x_2=-2x_3\)，代回得 \(x_1=x_3\)，全部解为：</p>
<div class="equation">\[
(x_1,x_2,x_3)=t(1,-2,1),\qquad t\in\mathbb R.
\]</div>
<p><strong>为什么填“平面”？</strong>前两个向量不是倍数关系，张成一个过原点的平面；第三个满足 \(\mathbf w_3=2\mathbf w_2-\mathbf w_1\)，也在其中。平面方程为 \(x-2y+z=0\)。</p>
<figure>
<img src="/images/courses/gilbert-linear-algebra/session-01/tikz_common_plane.svg" alt="三个向量位于过原点的平面内，终点在一条不经过原点的直线上。" width="560" loading="lazy">
<figcaption>图 4：平面内关系的斜投影，不表示实际角度和长度。三个终点共线，但那条直线不经过原点，所以三个向量并不共线。</figcaption>
</figure>
<p><strong>为什么矩阵 \(W=[\mathbf w_1\ \mathbf w_2\ \mathbf w_3]\) 不可逆？</strong>它把 \(\mathbf0\) 和 \((1,-2,1)^{\mathsf T}\) 都映成 \(\mathbf0\)，无法从输出唯一恢复输入。若逆矩阵存在，对 \(W(1,-2,1)^{\mathsf T}=\mathbf0\) 左乘 \(W^{-1}\)，便会得到非零向量等于零的矛盾。</p>

<h2 id="exercise-12">七、习题 1.2：矩阵乘向量</h2>
<div class="equation">\[
\begin{pmatrix}1&amp;2&amp;0\\2&amp;0&amp;3\\4&amp;1&amp;1\end{pmatrix}
\begin{pmatrix}3\\-2\\1\end{pmatrix}
=3\begin{pmatrix}1\\2\\4\end{pmatrix}
-2\begin{pmatrix}2\\0\\1\end{pmatrix}
+\begin{pmatrix}0\\3\\1\end{pmatrix}
=\boxed{\begin{pmatrix}-1\\9\\11\end{pmatrix}}.
\]</div>
<p>按列看是三个向量的组合；按行核对三个坐标：</p>
<div class="equation">\[
1\cdot3+2(-2)+0=-1,\quad
2\cdot3+0+3=9,\quad
4\cdot3-2+1=11.
\]</div>

<h2 id="exercise-13">八、习题 1.3：矩阵乘矩阵</h2>
<p>题目说 \(3\times2\) 的 \(A\) 乘 \(2\times3\) 的 \(B\)，得到 \(3\times3\) 的 \(AB\)，判断<strong>正确</strong>。</p>
<p>把 \(B\) 的每一列交给 \(A\) 运算：</p>
<div class="equation">\[
B=[\mathbf b_1\ \mathbf b_2\ \mathbf b_3]
\quad\Longrightarrow\quad
AB=[A\mathbf b_1\ A\mathbf b_2\ A\mathbf b_3].
\]</div>
<p>\(B\) 的每列有两个坐标，符合 \(A\) 的输入要求；每个输出有三个坐标。三个输出按列排好，就是 \(3\times3\) 矩阵。一般规则为：</p>
<div class="equation">\[
\boxed{A_{m\times n}B_{n\times p}=(AB)_{m\times p}}.
\]</div>
<p>下标在此标记尺寸：中间的 \(n\) 必须匹配，结果保留 \(A\) 的行数、\(B\) 的列数。从映射看，\(AB\) 表示<strong>先做 \(B\)，再做 \(A\)</strong>：</p>
<div class="equation">\[
\mathbb R^p\xrightarrow{B}\mathbb R^n\xrightarrow{A}\mathbb R^m,
\qquad (AB)\mathbf x=A(B\mathbf x).
\]</div>
<p>本题 \(BA\) 也能计算，但尺寸是 \(2\times2\)。矩阵乘法通常不能交换顺序。</p>

<h2 id="self-check">理解检查</h2>
<p>复习时试着解释：解 \((2,1)\) 与目标 \((3,1)\) 有何区别？为什么“有解”不意味着“唯一解”？两个三维向量为何未必能生成任意三维向量？</p>
<h2 id="sources">资料</h2>
<p>依据 MIT OCW 18.06SC 第一讲与习题 1.1—1.3 整理。线性相关、可逆与矩阵乘法是为习题补充的解释。<a href="https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/">课程来源</a>；笔记与原创图示按 <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a> 分享。</p>
<p><a href="/courses/gilbert-linear-algebra/notes/">返回笔记目录</a></p>
</section>
