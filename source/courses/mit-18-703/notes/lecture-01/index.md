---
title: 第 1 讲：群与对称｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-01/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-02/">下一讲 →</a></p>
<div class="note-lead"><p>这一讲先不背群的四条公理。我们先追踪正三角形的顶点，看清“一个对称动作”和“两个动作接着做”是什么意思，再从这个例子抽出群的定义。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-01.pdf">MIT 第 1 讲 PDF</a></nav>

<h2 id="section-1">一、对象究竟是什么？</h2>
<p>把正三角形的顶点标为 \(A,B,C\)。我们研究的不是这三个点，也不是三角形的面积，而是<strong>把三角形放回原位置的动作</strong>。先看两个动作：\(R\) 是逆时针旋转 \(120^\circ\)；\(F\) 是沿过 \(A\) 的中线反射。让每个动作作用于三个顶点，结果如下。</p>
<figure class="note-figure"><div class="symmetry-figures"><img src="/images/notes/assets/mit-18-703/triangle-rotation.svg" alt="旋转 R 依次把 A 送到 C、C 送到 B、B 送到 A"><img src="/images/notes/assets/mit-18-703/triangle-reflection.svg" alt="反射 F 固定 A 并交换 B 与 C"></div><figcaption>左图：\(R:A\mapsto C\mapsto B\mapsto A\)。右图：\(F(A)=A\)，\(F(B)=C\)，\(F(C)=B\)。虚线是反射轴。</figcaption></figure>
<p>还有一个什么都不做的动作，记作 \(e\)。连续旋转两次是 \(R^2\)，旋转三次回到原位，所以 \(R^3=e\)。反射做两次也回到原位，所以 \(F^2=e\)。这些记号描述的是<strong>动作的复合</strong>，不是数字的乘方。</p>

<h2 id="section-2">二、先算一次“乘法”</h2>
<p>约定按函数复合的顺序书写：\(RF\) 中右边的 \(F\) <strong>先做</strong>，左边的 \(R\) <strong>后做</strong>。例如从 \(B\) 出发，先反射得到 \(C\)，再旋转得到 \(B\)。因此 \(RF(B)=B\)。三个顶点都算一遍：</p>
<table><thead><tr><th>出发点</th><th>先做 \(F\)</th><th>再做 \(R\)，即 \(RF\)</th></tr></thead><tbody><tr><td>\(A\)</td><td>\(A\)</td><td>\(C\)</td></tr><tr><td>\(B\)</td><td>\(C\)</td><td>\(B\)</td></tr><tr><td>\(C\)</td><td>\(B\)</td><td>\(A\)</td></tr></tbody></table>
<p>所以 \(RF\) 固定 \(B\)。反过来，\(FR\) 是先旋转再反射；只看 \(A\) 就够了：\(R(A)=C\)，\(F(C)=B\)，因此 \(FR(A)=B\)，而 \(RF(A)=C\)。两个动作的先后顺序改变了结果，故 \(RF\ne FR\)。</p>
<p>三角形的对称共有六个：\(e,R,R^2\) 是三个旋转，\(F,RF,R^2F\) 是三个反射。为什么没有第七个？一个对称先决定 \(A\) 被送到三个顶点中的哪一个，再决定顺时针排列是否被翻转，共有 \(3\times2=6\) 种可能；上面的六个动作已经全部实现。这个六阶群记作 \(D_3\)。</p>

<h2 id="section-3">三、现在才抽出群的定义</h2>
<p>一个<strong>群</strong>是集合 \(G\) 连同一个运算；在三角形例子里，集合是六个动作，运算是“依次做”。它需要满足四件事：</p>
<ol><li><strong>封闭：</strong>两个允许的动作接着做，仍是允许的动作。例如 \(RF\) 仍是三角形的对称。</li><li><strong>结合律：</strong>\((gh)k=g(hk)\)。三个动作的执行顺序都是先 \(k\)、再 \(h\)、最后 \(g\)；只是写括号的方式不同。</li><li><strong>单位元：</strong>有一个动作 \(e\) 什么都不改，满足 \(eg=ge=g\)。</li><li><strong>逆元：</strong>每个动作都能撤销。\(R\) 的逆是 \(R^2\)，而 \(F\) 的逆就是 \(F\) 本身。</li></ol>
<p>结合律和交换律要分清：结合律只改变括号，交换律改变动作的先后。这个例子满足结合律，但刚才已算出 \(RF\ne FR\)，因此它<strong>不交换</strong>。乘法交换的群才称为阿贝尔群。</p>

<h2 id="section-4">四、换一个对象，还要问同样四件事</h2>
<p>\((\mathbb Z,+)\) 是群：整数相加仍是整数，加法满足结合律，\(0\) 是单位元，\(a\) 的逆元是 \(-a\)。\((\mathbb N,+)\) 则不是群：即使约定 \(0\in\mathbb N\)，正整数 \(1\) 在 \(\mathbb N\) 中也没有加法逆元。判断时必须同时写出<strong>集合与运算</strong>；同一批对象换一种运算，答案可能改变。</p>
<p>矩阵也是这样：所有同型矩阵在加法下成群；在乘法下，只有可逆方阵构成群 \(GL_n\)。对不可逆矩阵，乘法的“撤销动作”不存在。</p>
<p>最后，群公理还有一个立即可用的结论：消去律。若 \(ax=ay\)，左边同时乘 \(a^{-1}\)，便得到 \(x=y\)。以后讨论陪集和同态时，我们会反复用这个操作。</p>

<h2 id="section-5">理解检查</h2>
<ol><li>按“先做右边”的规则，逐点算出 \(FR\) 对 \(A,B,C\) 的作用，找出它固定哪个顶点。</li><li>整数在乘法下为何不成群？只删去 \(0\) 是否足够？</li><li>先做 \(b\)，再做 \(a\)；若要把这两个动作撤销，为什么必须先做 \(a^{-1}\)，再做 \(b^{-1}\)？由此证明 \((ab)^{-1}=b^{-1}a^{-1}\)。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 1, <em>Groups</em> 整理。图与顶点映射按本页的逆时针约定绘制；不同教材可能采用顺时针旋转或另一种 \(D_n\) 编号方式。</p>
</section>
