---
title: 第 11 讲：交错群与置换的符号｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-11/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-10/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-12/">下一讲 →</a></p>
<div class="note-lead"><p>第 5 讲知道每个置换都能写成对换的积，却不知道这种写法是否唯一。交错群的关键不是“偶数个对换”这几个字，而是先证明对换个数的奇偶性与写法无关。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-11.pdf">MIT 第 11 讲 PDF</a></nav>

<h2 id="section-38">一、为什么奇偶性有定义？</h2>
<p>第 5 讲把 \((1\ 2\ 3)\) 写成两个对换。但如果另找一种写法，能否恰好用奇数个对换？不能直接凭一个写法回答。先给置换一个<strong>不依赖分解的</strong>数字：把 \(\sigma\) 写成 \(\sigma(1),\ldots,\sigma(n)\)，数其中有多少对位置 \(i\lt j\) 满足 \(\sigma(i)>\sigma(j)\)。这个数叫逆序数。</p>
<p>例如 \((1\ 2\ 3)\) 的一行写法是 \((2,3,1)\)，逆序对是 \((2,1),(3,1)\)，共 2 对；因此它是偶置换。交换相邻两个位置时，只有这两个数之间的先后关系改变，逆序数的奇偶性翻转。任意对换可写成奇数次相邻交换，所以每做一个对换，奇偶性翻转一次。</p>
<p>定义符号 \(\varepsilon(\sigma)=(-1)^{\operatorname{inv}(\sigma)}\)。若 \(\sigma\) 被写成 \(r\) 个对换，它的符号就是 \((-1)^r\)。左边只由 \(\sigma\) 决定，故任何两种分解的对换个数必同奇偶。复合置换又满足 \(\varepsilon(\sigma\tau)=\varepsilon(\sigma)\varepsilon(\tau)\)。原讲义用 Vandermonde 乘积 \(\prod_{i\lt j}(x_i-x_j)\) 给出等价证明：每个对换使乘积变号。</p>
<p>一个 \(k\)-循环可写成 \(k-1\) 个对换，所以符号是 \((-1)^{k-1}\)。对不交循环，直接把各循环的符号相乘即可。例如四循环的符号是 \(-1\)，不必真的把每个对换写出来。</p>

<h2 id="section-39">二、交错群为什么是正规子群？</h2>
<p>符号映射把每个置换送到 \(1\) 或 \(-1\)，且复合时符号相乘，所以 \(\varepsilon:S_n\to\{\pm1\}\) 是同态。它的核就是全部偶置换，记 \(A_n\)。核必正规，故 \(A_n\trianglelefteq S_n\)。</p>
<p>当 \(n\ge2\) 时有对换，符号映射能取到 \(-1\)，因此满射。第一同构定理给 \(S_n/A_n\cong\mathbb Z/2\mathbb Z\)；偶、奇两类一样多，\(|A_n|=n!/2\)。这说明商群 \(S_3/A_3\) 为什么只记录一个比特的奇偶信息。</p>

<h2 id="section-40">三、正规不等于简单</h2>
<p>正规性不意味着群内部再无正规子群。\(A_4\) 中有 \(V_4=\{e,(1\ 2)(3\ 4),(1\ 3)(2\ 4),(1\ 4)(2\ 3)\}\)：三个非单位元素互相相乘会得到第三个，且共轭会在这三个元素之间重排，因此 \(V_4\trianglelefteq A_4\)。它是真子群，所以 \(A_4\) 不简单。</p>
<p>相比之下，\(A_n\) 在 \(n\ge5\) 时简单。这是讲义给出的定理；完整证明需要分析 3-循环怎样生成 \(A_n\) 及其共轭行为，不能只凭 \(|A_n|=n!/2\) 推出。</p>
<h2 id="section-41">理解检查</h2>
<ol><li>求 \((1\ 2\ 3)(4\ 5)\) 的符号。</li><li>证明两个奇置换之积是偶置换，并从同态而非分解长度解释。</li><li>在 \(A_4\) 中核对 \(V_4\) 的四个元素相乘仍在 \(V_4\)。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 11, <em>The Alternating Groups</em> 整理；原讲义用 Vandermonde 乘积证明符号良定义。</p>
</section>
