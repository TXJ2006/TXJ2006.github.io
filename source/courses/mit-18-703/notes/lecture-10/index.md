---
title: 第 10 讲：同构定理与交换化｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-10/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-09/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-11/">下一讲 →</a></p>
<div class="note-lead"><p>商群把核压成单位元。同构定理说明：任何同态都可以先做这种压缩，再无损地到达它的像；第二、第三同构定理是这条原则在子群和连续取商中的具体形式。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-10.pdf">MIT 第 10 讲 PDF</a></nav>

<h2 id="section-34">一、第一同构定理：所有损失都在核里</h2>
<p>对同态 \(f:G\to K\)，映射 \(\bar f:G/\ker f\to\operatorname{im}f\)、\(g\ker f\mapsto f(g)\) 良定义且满射。若 \(\bar f(g\ker f)=e\)，则 \(g\in\ker f\)，故 \(g\ker f\) 是商群单位元，\(\bar f\) 也单射。因此</p>
<div class="equation">\[G/\ker f\cong\operatorname{im}f.\]</div>
<p>例如奇偶同态给 \(\mathbb Z/2\mathbb Z\cong\{\pm1\}\)，这里映射可以写 \([n]\mapsto(-1)^n\)。若 \(f\) 本来满射，右边才可直接写成目标群 \(K\)；不满射时不能漏掉“像”。</p>

<h2 id="section-35">二、第二、第三同构定理读成什么？</h2>
<p>设 \(H\leq G,N\trianglelefteq G\)。\(HN=\{hn:h\in H,n\in N\}\) 是子群，因为正规性允许把乘积里的 \(n h\) 改写成 \(h n'\)。映射 \(H\to HN/N\)、\(h\mapsto hN\) 满射，其核为 \(H\cap N\)，故</p>
<div class="equation">\[H/(H\cap N)\cong HN/N.\]</div>
<p>这说的是：把 \(H\) 放进商群时，恰好遗忘它与 \(N\) 的交集。再设 \(K\trianglelefteq G,H\trianglelefteq G\) 且 \(K\subseteq H\)。映射 \(G/K\to G/H\)、\(gK\mapsto gH\) 满射，核为 \(H/K\)，所以</p>
<div class="equation">\[(G/K)/(H/K)\cong G/H.\]</div>
<p>这说的是：先压掉 \(K\)，再压掉余下的 \(H/K\)，等于一步压掉 \(H\)。两式的证明都只需确定一个自然同态的像与核，然后应用第一定理。</p>
<p>用整数试一次：取 \(G=\mathbb Z,H=2\mathbb Z,N=3\mathbb Z\)。此时 \(H+N=\mathbb Z\)，而 \(H\cap N=6\mathbb Z\)，第二定理成为 \(2\mathbb Z/6\mathbb Z\cong\mathbb Z/3\mathbb Z\)：左侧的三个类 \(0,2,4\) 分别对应右侧 \(0,2,1\)。取 \(K=6\mathbb Z\subseteq H=2\mathbb Z\)，第三定理则说 \((\mathbb Z/6\mathbb Z)/(2\mathbb Z/6\mathbb Z)\cong\mathbb Z/2\mathbb Z\)。这样能看出各层商掉的是哪一批差异。</p>

<h2 id="section-36">三、中心、交换子与直积的位置</h2>
<p>中心 \(Z(G)=\{z:zg=gz\text{ 对所有 }g\in G\}\) 保留在任何自同构下；这种比正规性更强的性质称<strong>特征子群</strong>。把所有交换子 \([x,y]=x^{-1}y^{-1}xy\) 生成的子群记 \([G,G]\)。在 \(G/[G,G]\) 中一切交换子变为单位元，所以商群交换；任何到阿贝尔群的同态都杀掉这些交换子，因而必经过这个商，称<strong>交换化</strong>。对于 \(S_3\)，交换化是 \(\mathbb Z/2\mathbb Z\)。</p>
<p>直积 \(G\times H\) 的乘法逐坐标进行：\((g_1,h_1)(g_2,h_2)=(g_1g_2,h_1h_2)\)。给定两同态 \(f:K\to G\)、\(q:K\to H\)，唯一同时实现它们的映射是 \(k\mapsto(f(k),q(k))\)。这是讲义所说直积的泛性质，和第 9 讲的商泛性质一样，描述的是映射如何被唯一决定。</p>
<h2 id="section-37">理解检查</h2>
<ol><li>将 \(f:\mathbb Z\to\mathbb Z/12\mathbb Z\)、\(n\mapsto[4n]\) 代入第一同构定理，分别求核与像。</li><li>取 \(G=\mathbb Z,H=2\mathbb Z,N=3\mathbb Z\)，把第二同构定理两侧具体写出。</li><li>解释为什么到阿贝尔群的任何同态都把 \([x,y]\) 送到单位元。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 10, <em>The Isomorphism Theorems</em> 整理。</p>
</section>
