---
title: 第 7 讲：同构与 Cayley 定理｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-07/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-06/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-08/">下一讲 →</a></p>
<div class="note-lead"><p>三角形的六个对称与三个顶点的六种置换“是同一个群”。同构把这句话变成可检验的数学陈述，Cayley 定理进一步说明每个群都能作为置换群的一部分出现。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-07.pdf">MIT 第 7 讲 PDF</a></nav>

<h2 id="section-23">一、同构保留的是运算</h2>
<p>第 1 讲的 \(D_3\) 把一个三角形送回原位；第 5 讲的 \(S_3\) 重新排列三个标签。把每个三角形动作送到它对 \(A,B,C\) 造成的排列，就能一一对应这两个群。例如反射 \(F\) 对应交换 \(B,C\) 的对换；先做 \(F\) 再做 \(R\)，对应的也是先施行相应对换再施行相应循环。这个对应不只配对元素，还保留复合顺序。</p>
<p>这样的一一对应叫<strong>群同构</strong>：它是双射 \(\phi:G\to H\)，满足 \(\phi(ab)=\phi(a)\phi(b)\)。等式说“先运算再翻译”与“先翻译再运算”一致。故单位元、逆元、元素阶和交换性都会被保留。仅有同样多的元素不够：\(\mathbb Z_4\) 有四阶元素，而 \(\mathbb Z_2\times\mathbb Z_2\) 没有，二者不同构。</p>
<p>不同外观也可能给同一个群：\((\mathbb R,+)\) 与 \((\mathbb R_{>0},\times)\) 由 \(x\mapsto e^x\) 对应，因为 \(e^{x+y}=e^xe^y\)，且指数映射是双射。</p>
<p>有限例子：\([k]_6\mapsto([k]_2,[k]_3)\) 把 \(\mathbb Z/6\mathbb Z\) 送到 \(\mathbb Z/2\mathbb Z\times\mathbb Z/3\mathbb Z\)。若两坐标都是 0，则 \(2\mid k\) 且 \(3\mid k\)，故 \(6\mid k\)；映射单射。两边各有 6 个元素，便也是满射。它还保加法，所以是同构；不能只因两边都有 6 个元素就宣布同构。</p>

<h2 id="section-24">二、循环群由什么分类？</h2>
<p>若 \(G=\langle a\rangle,H=\langle b\rangle\) 且二者同阶，定义 \(a^k\mapsto b^k\)。有限阶为 \(n\) 时，\(a^r=a^s\iff n\mid r-s\)，同样推出 \(b^r=b^s\)，故映射不依赖指数写法；无限阶时指数表示本来唯一。它又保乘法且双射，所以任意两个同阶循环群同构。有限循环群统一写作 \(\mathbb Z/n\mathbb Z\)，无限循环群同构于 \(\mathbb Z\)。</p>

<h2 id="section-25">三、Cayley 定理的嵌入怎样构造？</h2>
<p>现在反过来问：一个完全抽象的群，能否像三角形一样看成某个集合上的置换？让 \(G\) 自己充当那个集合。每个 \(a\in G\) 定义一个动作 \(L_a(x)=ax\)，即对所有元素左乘 \(a\)。它可逆，逆动作是左乘 \(a^{-1}\)，所以是一个置换。</p>
<p>复合两个左平移：\(L_aL_b(x)=a(bx)=(ab)x=L_{ab}(x)\)，故 \(a\mapsto L_a\) 保运算。若 \(L_a=L_b\)，把二者都作用于单位元 \(e\)，得 \(a=b\)；所以不同群元素给不同置换。这就是 <strong>Cayley 定理</strong>：任何群都同构于某个置换群的子群。它只保证存在这样的表示，不保证所用集合最小。</p>
<p>同构 \(G\to G\) 称自同构。固定 \(a\) 的共轭映射 \(x\mapsto axa^{-1}\) 是自同构，因为它有逆映射 \(x\mapsto a^{-1}xa\)，且 \(a(xy)a^{-1}=(axa^{-1})(aya^{-1})\)。第 6 讲的“换标签”因此也是一种保结构的变换。</p>
<h2 id="section-26">理解检查</h2>
<ol><li>对 \(G=\mathbb Z/3\mathbb Z\)，写出三个左平移在集合 \(\{[0],[1],[2]\}\) 上的置换。</li><li>为什么 \(\mathbb Z_6\) 和 \(\mathbb Z_2\times\mathbb Z_3\) 同构？给出显式映射。</li><li>若 \(G\) 阿贝尔，所有共轭自同构是什么映射？</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 7, <em>Isomorphisms</em> 整理。</p>
</section>
