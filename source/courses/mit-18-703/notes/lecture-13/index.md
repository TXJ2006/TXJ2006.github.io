---
title: 第 13 讲：Sylow 定理与群作用｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-13/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-12/">上一讲</a></p>
<div class="note-lead"><p>Lagrange 定理只限制“已有子群”的阶。Sylow 定理反过来保证最大素数幂阶的子群存在，并通过它们的个数寻找正规子群。群作用把这种计数变成一个同态问题。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-13.pdf">MIT 第 13 讲 PDF</a></nav>

<h2 id="section-46">一、Sylow 定理给哪三条信息？</h2>
<p>若 \(|G|=p^km\)，\(p\) 不整除 \(m\)，阶为 \(p^k\) 的子群称 Sylow \(p\)-子群。定理断言：<strong>存在</strong>这样的子群；任意两个彼此共轭；它们的个数 \(n_p\) 满足 \(n_p\equiv1\pmod p\) 且 \(n_p\mid m\)。特别地，\(n_p=1\) 当且仅当唯一的 Sylow \(p\)-子群正规，因为共轭后仍是 Sylow 子群。原讲义在此陈述并使用 Sylow 定理，没有给完整证明；不能把存在性误说成由 Lagrange 定理直接推出。</p>
<p>例如 \(|G|=15=3\cdot5\)：\(n_5\mid3\) 且 \(n_5\equiv1\pmod5\)，所以 \(n_5=1\)，五阶子群正规。一般若 \(|G|=pq\) 且 \(p\lt q\) 都是素数，\(n_q\mid p\)、\(n_q\equiv1\pmod q\)，也只能 \(n_q=1\)。这里得到的是一个非平凡正规子群；并没有自动证明整个群是循环群。</p>
<p>定理的计数条件也可从作用理解。固定一个 Sylow \(p\)-子群 \(P\)，它以共轭作用在所有 Sylow \(p\)-子群的集合上。若 \(P\) 固定 \(Q\)，则 \(P\) 规范化 \(Q\)，乘积 \(PQ\) 是 \(p\)-子群；由于二者已经是最大 \(p\)-幂阶，只能 \(P=Q\)。因此唯一的固定点是 \(P\)，其余 \(P\)-轨道大小都被 \(p\) 整除，得到 \(n_p\equiv1\pmod p\)。同时共轭性给 \(n_p=[G:N_G(P)]\)；规范化子含 \(P\)，故 \(n_p\mid m\)。这解释了条件的来源，存在性与共轭性的完整证明仍需 Sylow 定理本身。</p>

<h2 id="section-47">二、素数幂群为何有非平凡中心？</h2>
<p>让有限 \(p\)-群 \(G\) 以共轭作用于自身。每个共轭类的大小为 \([G:C_G(x)]\)，因而是 \(p\) 的幂；若类大小不是 1，就被 \(p\) 整除。大小为 1 的类恰对应中心元素。把所有类大小相加得到类方程，模 \(p\) 后有 \(|G|\equiv|Z(G)|\pmod p\)。由于 \(p\mid|G|\)，故 \(p\mid|Z(G)|\)；中心含单位元，所以至少有 \(p\) 个元素。若 \(|G|>p\) 且 \(Z(G)\ne G\)，中心就是非平凡真正规子群；若 \(Z(G)=G\)，群交换，任取非单位元 \(g\)，其阶是 \(p\) 的幂，某个幂次便生成阶为 \(p\) 的真子群，它自动正规。因此简单的有限 \(p\)-群只能是素数阶循环群。</p>

<h2 id="section-48">三、计数和作用怎样排除简单群？</h2>
<p>“简单”指非平凡且没有非平凡真正规子群。若 \(|G|=30\) 且假设简单，则 \(n_5\mid6,n_5\equiv1\pmod5\) 迫使 \(n_5=6\)，六个不同的五阶子群除单位元外互不相交，给出 \(6\cdot4=24\) 个五阶元素。再有 \(n_3\mid10,n_3\equiv1\pmod3\)，简单性排除 1，故 \(n_3=10\)，带来 \(10\cdot2=20\) 个三阶元素。总数至少 44，超过 30，矛盾。</p>
<p>对于 \(|G|=48\)，Sylow 2-子群的阶是 \(16\)（原讲义有一处误写为 8），且 \(n_2\mid3,n_2\equiv1\pmod2\)，所以是 1 或 3。若是 1，立即正规；若是 3，\(G\) 以共轭作用在这三个子群上，给出同态 \(G\to S_3\)。它不可能单射，因为 \(48>6\)；又因三个 Sylow 子群互为共轭，作用非平凡，核不可能是整个 \(G\)。核于是是非平凡真正规子群。这个例子展示了群作用的用途：作用 \(G\curvearrowright X\) 等价于同态 \(G\to\operatorname{Sym}(X)\)，核记录哪些元素在 \(X\) 上完全不起作用。</p>
<h2 id="section-49">理解检查</h2>
<ol><li>群的阶为 21 时，Sylow 7-子群有几个？能否据此断定群交换？</li><li>把上面 30 阶群计数中的“两个不同素数阶子群只交于单位元”证明出来。</li><li>说明为何 48 阶例子中的作用既非单射，也非平凡作用。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 13, <em>Sylow Theorems and Applications</em> 整理；定理本身依原讲义陈述，应用中的阶数已重新核对。</p>
</section>
