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
<p>奇偶映射 \(f:\mathbb Z\to\{\pm1\}\)、\(n\mapsto(-1)^n\) 把所有偶数送到 \(1\)，所有奇数送到 \(-1\)。它先把整数按核 \(2\mathbb Z\) 分成两块；这两块与 \(\{\pm1\}\) 的两个元素再一一对应。此时“损失的信息”恰好就是核。</p>
<p>一般同态 \(f:G\to K\) 也一样。第 9 讲已知，同一核陪集上的元素有相同像，因此 \(g\ker f\mapsto f(g)\) 定义了从 \(G/\ker f\) 到 \(\operatorname{im}f\) 的映射。它满射，因为目标就是像；它单射，因为只有单位陪集会被送到单位元。于是得到<strong>第一同构定理</strong>：</p>
<div class="equation">\[G/\ker f\cong\operatorname{im}f.\]</div>
<p>在刚才例子中，它就是 \(\mathbb Z/2\mathbb Z\cong\{\pm1\}\)、\([n]\mapsto(-1)^n\)。注意右边写的是 \(\operatorname{im}f\)；只有 \(f\) 满射时，才能把它直接写成整个目标群 \(K\)。</p>

<h2 id="section-35">二、第二、第三同构定理读成什么？</h2>
<p>第二定理回答：把子群 \(H\) 放到商群 \(G/N\) 里，\(H\) 内哪部分会被压成单位元？恰是 \(H\cap N\)。设 \(H\leq G,N\trianglelefteq G\)，令 \(HN=\{hn:h\in H,n\in N\}\)；正规性保证它是子群。映射 \(H\to HN/N\)、\(h\mapsto hN\) 满射，核为 \(H\cap N\)，于是由第一定理得到</p>
<div class="equation">\[H/(H\cap N)\cong HN/N.\]</div>
<p>第三定理回答：能否分两步压缩？若 \(K\trianglelefteq G,H\trianglelefteq G\) 且 \(K\subseteq H\)，先构造 \(G/K\)，再把其中的 \(H/K\) 压成单位元。映射 \(G/K\to G/H\)、\(gK\mapsto gH\) 满射，核正是 \(H/K\)，所以</p>
<div class="equation">\[(G/K)/(H/K)\cong G/H.\]</div>
<p>两式都不是凭形式猜出来的：先写自然同态，算它的核与像，再用第一定理。拿整数核对第二式：\(H=2\mathbb Z,N=3\mathbb Z\) 时，\(H+N=\mathbb Z\)、\(H\cap N=6\mathbb Z\)，所以 \(2\mathbb Z/6\mathbb Z\cong\mathbb Z/3\mathbb Z\)。左边的 \(0,2,4\) 三个类分别送到右边的 \(0,2,1\)。</p>
<p>第三式可取 \(K=6\mathbb Z\subseteq H=2\mathbb Z\)：先取 \(\mathbb Z/6\mathbb Z\)，再把其中的偶数类压成单位元，最后只剩奇偶两类，恰是 \(\mathbb Z/2\mathbb Z\)。</p>

<h2 id="section-36">三、中心、交换子与直积的位置</h2>
<p>还有一种“只保留交换部分信息”的商。若目标群交换，则任何同态都会把 \(xy\) 与 \(yx\) 送到同一结果，故把交换子 \([x,y]=x^{-1}y^{-1}xy\) 送到单位元。由所有交换子生成的子群记 \([G,G]\)；商 \(G/[G,G]\) 因而交换，称为<strong>交换化</strong>。\(S_3\) 的交换化是 \(\mathbb Z/2\mathbb Z\)，只留下置换的奇偶信息。</p>
<p>另一类稳定子群是中心 \(Z(G)=\{z:zg=gz\text{ 对所有 }g\in G\}\)。任意自同构都把中心送到中心；在所有自同构下都不变的子群叫<strong>特征子群</strong>。它比仅在内自同构下不变的正规子群要求更强。</p>
<p>直积 \(G\times H\) 的乘法逐坐标进行：\((g_1,h_1)(g_2,h_2)=(g_1g_2,h_1h_2)\)。给定两同态 \(f:K\to G\)、\(q:K\to H\)，唯一同时实现它们的映射是 \(k\mapsto(f(k),q(k))\)。这是讲义所说直积的泛性质，和第 9 讲的商泛性质一样，描述的是映射如何被唯一决定。</p>
<h2 id="section-37">理解检查</h2>
<ol><li>将 \(f:\mathbb Z\to\mathbb Z/12\mathbb Z\)、\(n\mapsto[4n]\) 代入第一同构定理，分别求核与像。</li><li>取 \(G=\mathbb Z,H=2\mathbb Z,N=3\mathbb Z\)，把第二同构定理两侧具体写出。</li><li>解释为什么到阿贝尔群的任何同态都把 \([x,y]\) 送到单位元。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 10, <em>The Isomorphism Theorems</em> 整理。</p>
</section>
