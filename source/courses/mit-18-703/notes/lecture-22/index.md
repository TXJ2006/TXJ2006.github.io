---
title: 第 22 讲：快速素性检验｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-22/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-21/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-23/">下一讲 →</a></p>
<div class="note-lead"><p>本讲把环论用于计算数论：Fermat 同余是素性的必要条件，却会被 Carmichael 数欺骗；AKS 用多项式同余把这个条件升级为确定性多项式时间算法。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-22.pdf">MIT 第 22 讲 PDF</a></nav>

<h2 id="section-82">一、从 Fermat 小定理到多项式同余</h2>
<p>若 \(n\) 是素数且 \(\gcd(a,n)=1\)，则 \(a^{n-1}\equiv1\pmod n\)，等价地对所有整数 \(a\) 有 \(a^n\equiv a\pmod n\)。但合数 \(561=3\cdot11\cdot17\) 也满足后一个同余，因此单个底数的 Fermat 检验只能证明“不是素数”，不能证明“是素数”。</p>
<p>若 \(n\) 为素数，特征为 \(n\) 的环中 Frobenius 映射 \(f\mapsto f^n\) 是环同态，所以
\[(x+a)^n=x^n+a\quad\text{in }\mathbb F_n[x].\]
若 \(n\) 合成，二项式展开中某个系数不会被 \(n\) 整除，于是等式失败。问题在于直接计算有 \(n\) 个系数，运行时间随数字值而非位数多项式增长。</p>

<h2 id="section-83">二、AKS 为什么换到商环？</h2>
<p>令 \(d=\lceil\log_2n\rceil\)，选一个较小的 \(r\)，在
\[R=\mathbb Z[x]/(n,x^r-1)\]
中计算。关系 \(x^r=1\) 把所有次数降到小于 \(r\)，所以多项式运算的规模由 \(r\) 和 \(d\) 控制。算法先排除纯幂，再寻找使 \(n\) 在 \((\mathbb Z/r\mathbb Z)^\times\) 中具有足够大阶的 \(r\)，检查 \(\gcd(a,n)\)，最后对有限多个小底数 \(a\) 验证 \((x+a)^n=x^n+a\) 在商环中的等式。</p>
<p>“足够大阶”保证有足够多彼此独立的幂；如果合数通过所有检验，就会在某个有限域的乘法群中同时产生过多元素。另一方面，多项式次数给出上界，二者矛盾。讲义给出的证明由此得到 AKS 的确定性多项式时间复杂度。</p>

<h2 id="section-84">三、为什么还要概率算法？</h2>
<p>AKS 具有理论意义，但实际计算常用重复的随机检验。对奇数 \(n\)，先用 Fermat 检验，再反复平方分解 \(n-1=2^sm\)。若随机底数产生了非平凡的平方根 \(1\)（即既不是 \(1\) 也不是 \(-1\)），就得到合数见证者。对合数，至少一半的互素底数是见证者，因此重复 100 次而未发现见证者时，误判概率可压到约 \(2^{-100}\)。</p>
<p>概率检验给出“很可能是素数”，AKS 给出确定性证明；二者分别代表实际速度与理论确定性的两种取舍。</p>
<h2 id="section-85">理解检查</h2>
<ol><li>为什么 Carmichael 数说明“通过一次 Fermat 检验”不是素数证明？</li><li>解释商环中关系 \(x^r=1\) 如何减少多项式计算量。</li><li>平方检验中出现非平凡平方根 \(1\) 时，为什么能推出 \(n\) 合成？</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 22, <em>A Quick Primality Test</em> 整理。</p>
</section>
