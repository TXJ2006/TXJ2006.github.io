---
title: 第 16 讲：环同态与理想｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-16/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-15/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-17/">下一讲 →</a></p>
<div class="note-lead"><p>群同态的核给出正规子群；环同态的核则必须对乘法吸收，这种更强的子对象就是理想。理想正是构造商环的条件。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-16.pdf">MIT 第 16 讲 PDF</a></nav>

<h2 id="section-58">一、环同态保留全部结构</h2>
<p>环同态 \(\phi:R\to S\) 满足 \(\phi(a+b)=\phi(a)+\phi(b)\)、\(\phi(ab)=\phi(a)\phi(b)\)，并且 \(\phi(1)=1\)。最后一个条件是本课程的约定。它自动是加法群同态，因此保留 \(0\) 和负号；再加上乘法条件，就能把环中的所有代数等式搬到目标环。</p>
<p>例子包括复共轭 \(z\mapsto\bar z\)，以及多项式平移 \(f(x)\mapsto f(x+1)\)。后者的逆是 \(f(x)\mapsto f(x-1)\)，所以是自同构。</p>

<h2 id="section-59">二、理想为什么比子环更适合做核？</h2>
<p>理想 \(I\triangleleft R\) 是加法子群，并且对任意 \(a\in I,r\in R\) 都有 \(ra,ar\in I\)。若 \(I=\ker\phi\)，则 \(\phi(ra)=\phi(r)\phi(a)=0\)，所以核自动满足吸收性。与群的正规性相比，理想不是只要求共轭不变，而是要求被整个环左右乘后仍留在其中。</p>
<p>由理想生成的商环定义为 \(R/I\)。加法是陪集加法，乘法规定为 \((r+I)(s+I)=rs+I\)。若换代表元 \(r+i,s+j\)，差值为 \(is+rj+ij\)，每项都在 \(I\) 中，因此乘法良定义。自然映射 \(u:R\to R/I\) 的核正是 \(I\)。</p>

<h2 id="section-60">三、商的泛性质与第一同构定理</h2>
<p>若 \(\phi:R\to S\) 的核包含 \(I\)，则 \(\phi\) 在同一陪集上取相同值，所以唯一存在 \(\bar\phi:R/I\to S\)，满足 \(\phi=\bar\phi\circ u\)。这与群论商群的泛性质完全平行。</p>
<p>若 \(\phi\) 满射，第一同构定理给出 \(S\cong R/\ker\phi\)。例如 \(\mathbb Z/(n)\cong\mathbb Z_n\)。在交换环中，主理想 \((a)=\{ra:r\in R\}\) 是最基本的理想；若理想含有单位，则它必等于整个环，因为 \(1=u^{-1}u\in I\)。</p>
<h2 id="section-61">理解检查</h2>
<ol><li>验证 \((x)\) 是 \(R[x]\) 的理想，并说明它与“常数项为零”的多项式集合相同。</li><li>为什么任意环同态的核一定是理想，而任意子环未必是某个同态的核？</li><li>把映射 \(\mathbb Z\to\mathbb Z_n\)、\(m\mapsto[m]\) 写成第一同构定理。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 16, <em>Ring Homomorphisms and Ideals</em> 整理。</p>
</section>
