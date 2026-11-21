---
title: 第 8 讲：同态、核与正规子群｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-08/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-07/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-09/">下一讲 →</a></p>
<div class="note-lead"><p>同构要求“无损翻译”；同态只要求保留乘法，可以把不同元素合并。核精确记录哪些元素被合并到单位元，而正规性保证这种合并尊重整个群的乘法。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-08.pdf">MIT 第 8 讲 PDF</a></nav>

<h2 id="section-27">一、同态保留什么，又可能丢掉什么？</h2>
<p>先取最简单的“保留一部分信息”的映射：把整数送到奇偶类，\(\phi:\mathbb Z\to\mathbb Z/2\mathbb Z\)、\(m\mapsto[m]\)。例如 \(3+5=8\) 被送到 \([1]+[1]=[0]=[8]\)。先加再取奇偶，与先取奇偶再加一样；但 \(0,2,4\) 都被合到 \([0]\)，所以映射不是单射。</p>
<p>这种保运算的映射称<strong>群同态</strong>：\(\phi:G\to K\) 满足 \(\phi(ab)=\phi(a)\phi(b)\)。它自动把单位元送到单位元：\(\phi(e_G)=\phi(e_G)^2\)，在目标群消去后得 \(\phi(e_G)=e_K\)；也把逆元送到逆元。</p>
<p>哪些元素的区别被“忘掉”了？被送到目标单位元的元素组成<strong>核</strong> \(\ker\phi=\{g:\phi(g)=e_K\}\)。奇偶映射的核是 \(2\mathbb Z\)，不是只有整数 0。核与像 \(\operatorname{im}\phi=\{\phi(g):g\in G\}\) 都是子群，因为乘积和逆元仍留在对应集合里。</p>
<p>更一般地，\(\phi(a)=\phi(b)\iff\phi(a^{-1}b)=e_K\iff a^{-1}b\in\ker\phi\)。所以被合成同一个值的元素恰组成核的一个左陪集。奇偶映射的两条“纤维”就是偶数集和奇数集；核越大，合并的信息越多。特别地，核只有 \(e_G\) 时，同态才单射。</p>

<h2 id="section-28">二、为什么核必须正规？</h2>
<p>在奇偶例子里，偶数集不仅是子群，而且无论先加一个整数还是后加，所得集合一样；这与整数加法交换有关。非交换群里，核也必须具备相应的稳定性：若 \(h\in\ker\phi\)，则 \(\phi(ghg^{-1})=\phi(g)e_K\phi(g)^{-1}=e_K\)，所以 \(ghg^{-1}\) 还在核中。</p>
<p>满足 \(gNg^{-1}=N\)（对每个 \(g\in G\)）的子群称<strong>正规子群</strong>，记 \(N\trianglelefteq G\)。等价地，每个 \(g\) 的左、右陪集满足 \(gN=Ng\)。这不是说 \(g\) 要与 \(N\) 中每个元素逐个交换，而是说两边得到的<strong>集合</strong>相同。</p>
<p>阿贝尔群的所有子群都正规。在 \(S_3\) 中，\(A_3=\{e,(1\ 2\ 3),(1\ 3\ 2)\}\) 正规；但 \(H=\{e,(1\ 2)\}\) 不正规，因为 \((2\ 3)(1\ 2)(2\ 3)^{-1}=(1\ 3)\notin H\)。另一个有用的判断是：正规子群是若干完整共轭类的并。第 9 讲将说明它正是“陪集可以相乘”的条件。</p>
<h2 id="section-29">理解检查</h2>
<ol><li>对 \(\phi:\mathbb Z\to\mathbb Z/6\mathbb Z\) 列出核，以及 \([2]\) 的原像。</li><li>给出一个非单射、但核正规且非平凡的矩阵群同态例子。</li><li>直接验证 \(gN=Ng\iff gNg^{-1}=N\)。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 8, <em>Homomorphisms and Kernels</em> 整理。</p>
</section>
