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
<p>设 \(\Delta(x_1,\ldots,x_n)=\prod_{i\lt j}(x_i-x_j)\)。置换 \(\sigma\) 重排下标后，所有无序指标对仍各出现一次，只可能把某些因子的顺序反过来，因此 \(\sigma\cdot\Delta=\varepsilon(\sigma)\Delta\)，其中 \(\varepsilon(\sigma)\in\{1,-1\}\)。交换相邻的两个变量时，它们之间的因子变号，其余涉及这两个变量的因子只是互换；一般对换 \((a\ b)\) 可写成 \(2(b-a)-1\) 次相邻交换，仍使 \(\Delta\) 变号。复合置换的作用给 \(\varepsilon(\sigma\tau)=\varepsilon(\sigma)\varepsilon(\tau)\)。于是若 \(\sigma\) 写成 \(r\) 个对换，\(\varepsilon(\sigma)=(-1)^r\)。左侧由置换本身决定，故不同分解的 \(r\) 必同奇偶。</p>
<p>例如 \((1\ 2\ 3)=(1\ 3)(1\ 2)\) 是偶置换；\((1\ 2\ 3\ 4)\) 是三个对换之积，故为奇置换。对不交循环分解，还可直接算 \(\varepsilon(\sigma)=\prod_i(-1)^{k_i-1}\)。这比逐次寻找对换分解更快。</p>

<h2 id="section-39">二、交错群为什么是正规子群？</h2>
<p>符号映射 \(\varepsilon:S_n\to\{\pm1\}\) 是满同态（\(n\geq2\)），其核是所有偶置换，记 \(A_n\)。核必正规，所以 \(A_n\trianglelefteq S_n\)，且由第一同构定理 \(S_n/A_n\cong\mathbb Z/2\mathbb Z\)。每个奇置换与一个固定对换相乘，便一一对应到偶置换，因此 \(|A_n|=n!/2\)。这也说明第 9 讲的 \(S_3/A_3\) 为什么自然记录“奇偶”。</p>

<h2 id="section-40">三、正规不等于简单</h2>
<p>\(A_4\) 有正规子群 \(V_4=\{e,(1\ 2)(3\ 4),(1\ 3)(2\ 4),(1\ 4)(2\ 3)\}\)，所以并非简单群。原因是 \(V_4\) 中三个非单位元素在 \(S_4\) 下共轭，构成完整的循环类型，且它们相乘仍在集合内。与此不同，\(A_n\) 在 \(n\geq5\) 时简单；这是讲义给出的定理，完整证明需要进一步分析 3-循环与共轭类，本讲不把它当作由“阶数为 \(n!/2\)”就能推出的结论。</p>
<h2 id="section-41">理解检查</h2>
<ol><li>求 \((1\ 2\ 3)(4\ 5)\) 的符号。</li><li>证明两个奇置换之积是偶置换，并从同态而非分解长度解释。</li><li>在 \(A_4\) 中核对 \(V_4\) 的四个元素相乘仍在 \(V_4\)。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 11, <em>The Alternating Groups</em> 整理；原讲义用 Vandermonde 乘积证明符号良定义。</p>
</section>
