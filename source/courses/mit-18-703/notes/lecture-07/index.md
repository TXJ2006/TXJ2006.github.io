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
<p>群同构是双射 \(\phi:G\to H\)，且 \(\phi(ab)=\phi(a)\phi(b)\)。双射保证两边元素一一对应，等式保证先在 \(G\) 中相乘再翻译，与先翻译再在 \(H\) 中相乘结果一致。于是单位元、逆元、元素阶和交换性都会被保留。仅有相同阶数不够：\(\mathbb Z_4\) 有元素阶为 4，\(\mathbb Z_2\times\mathbb Z_2\) 没有，所以它们不同构。</p>
<p>把 \(D_3\) 的每个对称送到它在三个顶点上诱导的置换，得到 \(D_3\cong S_3\)：不同对称作用不同，而每种顶点置换都由一个对称实现。另一类例子是 \((\mathbb R,+)\cong(\mathbb R_{>0},\times)\)，映射 \(x\mapsto e^x\) 满足 \(e^{x+y}=e^xe^y\) 且双射。同一群的两个外观可以完全不同。</p>
<p>另一个有限例子是 \(\mathbb Z/6\mathbb Z\to\mathbb Z/2\mathbb Z\times\mathbb Z/3\mathbb Z\)、\([k]_6\mapsto([k]_2,[k]_3)\)。它保加法；若两边坐标均为零，则 \(2\mid k\) 且 \(3\mid k\)，故 \(6\mid k\)，所以核仅有 \([0]_6\)。两边各有六个元素，单射于是也是双射。这种显式映射比单看阶数可靠。</p>

<h2 id="section-24">二、循环群由什么分类？</h2>
<p>若 \(G=\langle a\rangle,H=\langle b\rangle\) 且二者同阶，定义 \(a^k\mapsto b^k\)。有限阶为 \(n\) 时，\(a^r=a^s\iff n\mid r-s\)，同样推出 \(b^r=b^s\)，故映射不依赖指数写法；无限阶时指数表示本来唯一。它又保乘法且双射，所以任意两个同阶循环群同构。有限循环群统一写作 \(\mathbb Z/n\mathbb Z\)，无限循环群同构于 \(\mathbb Z\)。</p>

<h2 id="section-25">三、Cayley 定理的嵌入怎样构造？</h2>
<p>对每个 \(a\in G\) 定义左平移 \(L_a:G\to G\)，\(L_a(x)=ax\)。它是置换，逆为 \(L_{a^{-1}}\)；\(L_aL_b(x)=a(bx)=(ab)x=L_{ab}(x)\)，所以 \(a\mapsto L_a\) 保运算。如果 \(L_a=L_b\)，将两者作用于 \(e\) 得 \(a=b\)，故它是单射。于是 \(G\) 同构于置换群 \(\operatorname{Sym}(G)\) 的一个子群；有限 \(|G|=n\) 时可视为 \(S_n\) 的子群。这个构造给的是“存在一个忠实的置换表示”，通常并非最小表示。</p>
<p>同构 \(G\to G\) 称自同构。固定 \(a\) 的共轭映射 \(x\mapsto axa^{-1}\) 是自同构，因为它有逆映射 \(x\mapsto a^{-1}xa\)，且 \(a(xy)a^{-1}=(axa^{-1})(aya^{-1})\)。第 6 讲的“换标签”因此也是一种保结构的变换。</p>
<h2 id="section-26">理解检查</h2>
<ol><li>对 \(G=\mathbb Z/3\mathbb Z\)，写出三个左平移在集合 \(\{[0],[1],[2]\}\) 上的置换。</li><li>为什么 \(\mathbb Z_6\) 和 \(\mathbb Z_2\times\mathbb Z_3\) 同构？给出显式映射。</li><li>若 \(G\) 阿贝尔，所有共轭自同构是什么映射？</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 7, <em>Isomorphisms</em> 整理。</p>
</section>
