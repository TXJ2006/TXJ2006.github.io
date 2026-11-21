---
title: 第 20 讲：欧几里得整环｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-20/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-19/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-21/">下一讲 →</a></p>
<div class="note-lead"><p>欧几里得整环把整数的带余除法抽象出来。它给出计算最大公因子的算法，并且按强度形成链：Euclidean domain \(\Rightarrow\) PID \(\Rightarrow\) UFD。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-20.pdf">MIT 第 20 讲 PDF</a></nav>

<h2 id="section-74">一、带余除法的抽象形式</h2>
<p>整数里先算一个熟悉的除法：\(252=2\cdot105+42\)，再算 \(105=2\cdot42+21\)，最后 \(42=2\cdot21\)。每次余数变小，终会停在 21；它就是 \(252,105\) 的最大公因子。欧几里得整环抽象的正是“可带余除，余数有办法严格变小”。</p>
<p>严格地说，整环 \(R\) 称欧几里得整环，若存在给非零元素赋“大小” \(d:R\setminus\{0\}\to\mathbb N\cup\{0\}\)，满足 \(d(a)\le d(ab)\)，且任意 \(a\ne0,b\) 都可写成</p>
<div class="equation">\[b=aq+r,\qquad r=0\text{ 或 }d(r)\lt d(a).\]</div>
<p>对整数取 \(d(n)=|n|\)，对域 \(K\) 上的多项式取 \(d(f)=\deg f\)。这里的“大小”不是普通距离，只是一个能保证余数严格下降的非负整数值。</p>

<h2 id="section-75">二、为什么欧几里得整环是 PID？</h2>
<p>要证明所有理想都有一个生成元，取非零理想 \(I\) 中 \(d\)-值最小的非零元素 \(a\)。对任意 \(b\in I\) 作除法：\(b=aq+r\)。余数 \(r=b-aq\) 仍在 \(I\)；若 \(r\ne0\)，它的 \(d\)-值比 \(a\) 小，和选择 \(a\) 矛盾。因此每个 \(b\) 都是 \(a\) 的倍数，\(I=(a)\)。零理想本来就是 \((0)\)，所以欧几里得整环是 PID，再由上一讲知道它是 UFD。</p>

<h2 id="section-76">三、高斯整数中的几何除法</h2>
<p>高斯整数 \(\mathbb Z[i]\) 可看成平面整数格点，取范数 \(N(a+bi)=a^2+b^2\)。范数满足 \(N(zw)=N(z)N(w)\)。给定 \(\alpha,\beta\ne0\)，先看复数 \(\beta/\alpha\)，取距离它最近的格点 \(q\in\mathbb Z[i]\)，令 \(r=\beta-q\alpha\)。最近格点保证 \(N(r)\lt N(\alpha)\)，所以得到带余除法。几何上的“找最近格点”就是代数中的欧几里得算法。</p>
<p>例如除以 \(1+i\) 时，商由 \(\beta/(1+i)\) 四舍五入到最近的高斯整数；反复取余就能求最大公因子。于是 \(\mathbb Z[i]\) 是 Euclidean domain，也是 PID 和 UFD。</p>
<h2 id="section-77">理解检查</h2>
<ol><li>说明 \(K[x]\) 中多项式除法为什么满足余数次数小于除数次数。</li><li>用欧几里得算法求 \(\gcd(252,105)\)。</li><li>为什么“范数乘法性”是证明 \(\mathbb Z[i]\) 为欧几里得整环的关键，而不只是一个方便的记号？</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 20, <em>Euclidean Domains</em> 整理。</p>
</section>
