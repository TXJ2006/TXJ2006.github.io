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
<p>整环 \(R\) 称欧几里得整环，如果存在函数 \(d:R\setminus\{0\}\to\mathbb N\cup\{0\}\)，使得 \(d(a)\le d(ab)\)，并且任意 \(a\ne0,b\) 都可写成
\[ b=aq+r,\qquad r=0\text{ 或 }d(r)\lt d(a). \]
\(\mathbb Z\) 取 \(d(n)=|n|\)；域上的多项式环取 \(d(f)=\deg f\)。要求 \(R\) 是整环，是为了保证非零多项式乘积的次数相加，也为了让消去和最小性论证成立。</p>

<h2 id="section-75">二、为什么欧几里得整环是 PID？</h2>
<p>设 \(I\) 是非零理想，取 \(a\in I\setminus\{0\}\) 使 \(d(a)\) 最小。对任意 \(b\in I\) 作除法：\(b=aq+r\)。因为 \(r=b-aq\in I\)，余数不可能非零，否则违反 \(d(a)\) 的最小性。因此 \(r=0\)，所有 \(b\) 都被 \(a\) 整除，故 \(I=(a)\)。零理想显然也是主理想，于是欧几里得整环是 PID，再由上一讲得到 UFD。</p>

<h2 id="section-76">三、高斯整数中的几何除法</h2>
<p>高斯整数 \(\mathbb Z[i]\) 可看成平面整数格点，取范数 \(N(a+bi)=a^2+b^2\)。范数满足 \(N(zw)=N(z)N(w)\)。给定 \(\alpha,\beta\ne0\)，先看复数 \(\beta/\alpha\)，取距离它最近的格点 \(q\in\mathbb Z[i]\)，令 \(r=\beta-q\alpha\)。最近格点保证 \(N(r)\lt N(\alpha)\)，所以得到带余除法。几何上的“找最近格点”就是代数中的欧几里得算法。</p>
<p>例如除以 \(1+i\) 时，商由 \(\beta/(1+i)\) 四舍五入到最近的高斯整数；反复取余就能求最大公因子。于是 \(\mathbb Z[i]\) 是 Euclidean domain，也是 PID 和 UFD。</p>
<h2 id="section-77">理解检查</h2>
<ol><li>说明 \(K[x]\) 中多项式除法为什么满足余数次数小于除数次数。</li><li>用欧几里得算法求 \(\gcd(252,105)\)。</li><li>为什么“范数乘法性”是证明 \(\mathbb Z[i]\) 为欧几里得整环的关键，而不只是一个方便的记号？</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 20, <em>Euclidean Domains</em> 整理。</p>
</section>
