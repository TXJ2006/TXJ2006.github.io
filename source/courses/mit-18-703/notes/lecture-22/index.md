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
<p>只检验一个数值 \(a\) 容易被合数蒙混；换成多项式，要求每个系数同时满足同余，就强得多。若 \(n\) 为素数，模 \(n\) 时二项式中间系数都为零，因此</p>
<div class="equation">\[(x+a)^n=x^n+a\quad\text{in }\mathbb F_n[x].\]</div>
<p>例如合数 \(n=4,a=1\) 时，\((x+1)^4\equiv x^4+2x^2+1\pmod4\)，不等于 \(x^4+1\)。对与合数 \(n\) 互素的 \(a\)，总能在某个中间系数中发现失败；但直接展开约有 \(n\) 个系数，而输入长度只有 \(\log_2n\) 位，所以这样做仍太慢。</p>

<h2 id="section-83">二、AKS 为什么换到商环？</h2>
<p>AKS 的办法是只保留次数小于 \(r\) 的信息。令 \(d=\lceil\log_2n\rceil\)，改在商环中计算：</p>
<div class="equation">\[R=\mathbb Z[x]/(n,x^r-1).\]</div>
<p>这里有 \(x^r=1\)，故高次幂可折回低次，计算只需追踪 \(r\) 个系数。算法先排除纯幂，选取使 \(n\) 模 \(r\) 有足够大乘法阶的 \(r\)，检查小整数与 \(n\) 的最大公因子，最后对一定范围的 \(a\) 验证多项式同余。每步的规模才能按输入位数 \(d\) 估算。</p>
<p>“足够大阶”保证有足够多彼此独立的幂；如果合数通过所有检验，就会在某个有限域的乘法群中同时产生过多元素。另一方面，多项式次数给出上界，二者矛盾。讲义给出的证明由此得到 AKS 的确定性多项式时间复杂度。</p>

<h2 id="section-84">三、为什么还要概率算法？</h2>
<p>AKS 解决“有保证的多项式时间”问题，实际大整数检验常用更快的随机方法。对奇数 \(n\) 写 \(n-1=2^sm\)（\(m\) 为奇数），选底数 \(a\) 看 \(a^m,a^{2m},a^{4m},\ldots\) 的平方链。若在链中出现一个既不为 \(1\) 也不为 \(-1\) 的 \(1\) 的平方根，\(n\) 必是合数，因为素数模下 \((x-1)(x+1)=0\) 只能有 \(x=\pm1\)。</p>
<p>合数的互素底数中，至少一半可提供这种或更早的合数见证；独立随机选 100 次仍未发现见证时，误判概率有 \(2^{-100}\) 量级的上界。一次检验找到见证是确定的“合数”证明；没有找到只表示“高度可能是素数”。</p>
<p>概率检验给出“很可能是素数”，AKS 给出确定性证明；二者分别代表实际速度与理论确定性的两种取舍。</p>
<h2 id="section-85">理解检查</h2>
<ol><li>为什么 Carmichael 数说明“通过一次 Fermat 检验”不是素数证明？</li><li>解释商环中关系 \(x^r=1\) 如何减少多项式计算量。</li><li>平方检验中出现非平凡平方根 \(1\) 时，为什么能推出 \(n\) 合成？</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 22, <em>A Quick Primality Test</em> 整理。</p>
</section>
