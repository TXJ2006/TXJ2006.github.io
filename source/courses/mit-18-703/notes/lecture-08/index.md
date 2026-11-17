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
<p>映射 \(\phi:G\to K\) 是群同态，当 \(\phi(ab)=\phi(a)\phi(b)\)。由 \(\phi(e_G)=\phi(e_G)^2\) 消去可得 \(\phi(e_G)=e_K\)，进而 \(\phi(a^{-1})=\phi(a)^{-1}\)。例如 \(\phi:\mathbb Z\to\mathbb Z/2\mathbb Z\)、\(m\mapsto[m]\)，把所有偶数合为 \([0]\)，所有奇数合为 \([1]\)。它保加法，却不是单射。</p>
<p>核 \(\ker\phi=\{g\in G:\phi(g)=e_K\}\) 是子群：两个核元素的积仍被送到 \(e_K\)，逆元也如此。像 \(\operatorname{im}\phi=\{\phi(g):g\in G\}\) 同样对乘法与取逆封闭。更重要的是 \(\phi(a)=\phi(b)\iff\phi(a^{-1}b)=e_K\iff a^{-1}b\in\ker\phi\)。因此同态的每一条“纤维”恰是核的一个左陪集；同态单射当且仅当核只有 \(e_G\)。在奇偶映射里，核是 \(2\mathbb Z\)，两条纤维就是偶数和奇数陪集。</p>

<h2 id="section-28">二、为什么核必须正规？</h2>
<p>若 \(h\in\ker\phi\)，则 \(\phi(ghg^{-1})=\phi(g)e_K\phi(g)^{-1}=e_K\)，所以 \(ghg^{-1}\) 仍在核中。一个子群 \(N\leq G\) 若对所有 \(g\) 满足 \(gNg^{-1}=N\)，称<strong>正规子群</strong>，记 \(N\trianglelefteq G\)。等号可由单向包含同时对 \(g^{-1}\) 使用一次得到。正规性也等价于每个 \(g\) 都有 \(gN=Ng\)；它说的是左、右陪集逐一相同，不是说 \(g\) 与 \(N\) 的所有元素逐个交换。</p>
<p>阿贝尔群的所有子群都正规。在 \(S_3\) 中，\(A_3=\{e,(1\ 2\ 3),(1\ 3\ 2)\}\) 正规；但 \(H=\{e,(1\ 2)\}\) 不正规，因为 \((2\ 3)(1\ 2)(2\ 3)^{-1}=(1\ 3)\notin H\)。另一个有用的判断是：正规子群是若干完整共轭类的并。第 9 讲将说明它正是“陪集可以相乘”的条件。</p>
<h2 id="section-29">理解检查</h2>
<ol><li>对 \(\phi:\mathbb Z\to\mathbb Z/6\mathbb Z\) 列出核，以及 \([2]\) 的原像。</li><li>给出一个非单射、但核正规且非平凡的矩阵群同态例子。</li><li>直接验证 \(gN=Ng\iff gNg^{-1}=N\)。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 8, <em>Homomorphisms and Kernels</em> 整理。</p>
</section>
