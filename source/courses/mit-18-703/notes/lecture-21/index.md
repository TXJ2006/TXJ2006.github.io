---
title: 第 21 讲：多项式环｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-21/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-20/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-22/">下一讲 →</a></p>
<div class="note-lead"><p>多项式环把上一讲的欧几里得思想推广到代数。核心链条是：评价映射识别根，Gauss 引理把 \(R[x]\) 与分式域中的分解联系起来，Eisenstein 判别法提供不可约性的快速证据。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-21.pdf">MIT 第 21 讲 PDF</a></nav>

<h2 id="section-78">一、单位、评价与根</h2>
<p>若 \(R\) 是整环，则 \(R[x]\) 中的单位恰好是 \(R\) 中的单位。因为 \(\deg(fg)=\deg f+\deg g\)，若 \(fg=1\)，两者次数都只能为 0。给定环同态 \(\phi:R\to S\) 和 \(s\in S\)，存在唯一同态 \(R[x]\to S\)，它在系数上等于 \(\phi\)，并把 \(x\) 送到 \(s\)。这就是多项式环的泛性质。</p>
<p>令 \(\operatorname{ev}_\alpha:R[x]\to R\) 把 \(x\) 送到 \(\alpha\)，则 \(f(\alpha)=0\) 正好表示 \(f\) 在评价映射的核中。若 \(K\) 是域，\(\ker\operatorname{ev}_\alpha=(x-\alpha)\)，所以
\[f(\alpha)=0\iff x-\alpha\mid f(x).\]
二次或三次多项式在域上不可约，当且仅当没有根。</p>

<h2 id="section-79">二、内容与 Gauss 引理</h2>
<p>在 UFD \(R\) 中，多项式系数的最大公因子称内容 \(c(f)\)。把内容提出后，剩下的多项式称本原多项式。Gauss 引理的关键结论是
\[c(gh)\sim c(g)c(h),\]
其中 \(\sim\) 表示相差一个单位；特别地，本原多项式的乘积仍本原。证明的核心是：若某素元 \(p\) 整除乘积的所有系数，取两个因子中第一个不被 \(p\) 整除的系数，乘积中对应系数会出现唯一一个不被 \(p\) 整除的项，矛盾。</p>
<p>设 \(F=\operatorname{Frac}(R)\)。Gauss 引理说明：本原 \(f\in R[x]\) 若在 \(F[x]\) 中可分解，就能把分母清掉并重新调整单位，得到 \(R[x]\) 中同次数分解。因此 \(R\) 为 UFD 时，\(R[x]\) 也是 UFD；特别地，\(\mathbb Z[x]\) 是 UFD。</p>

<h2 id="section-80">三、Eisenstein 判别法为什么有效？</h2>
<p>设 \(f(x)=a_nx^n+\cdots+a_0\in\mathbb Z[x]\)。若存在素数 \(p\) 使 \(p\mid a_i\)（\(i\lt n\)）、\(p\nmid a_n\)、且 \(p^2\nmid a_0\)，则 \(f\) 在 \(\mathbb Q[x]\) 中不可约。模 \(p\) 后，\(\bar f=\bar a_nx^n\)。若 \(f=gh\) 是正次数分解，则 \(\bar g,\bar h\) 只能分别是首项乘 \(x^d,x^e\)，所以 \(g,h\) 的常数项都被 \(p\) 整除，迫使 \(p^2\mid a_0\)，矛盾。</p>
<p>例如下面的多项式用 \(p=3\) 满足判别条件：</p>
<div class="equation">\[\begin{aligned}f(x)={}&2x^7-15x^6+60x^5-18x^4\\&{}-9x^3+45x^2-3x+6.\end{aligned}\]</div>
<p>圆分多项式 \(1+x+\cdots+x^{p-1}\) 则通过代换 \(x=y+1\) 转化为 Eisenstein 形式。</p>
<h2 id="section-81">理解检查</h2>
<ol><li>为什么“有根”只足以判定二次、三次多项式可约，不能直接用于四次多项式？</li><li>把 \(6x^3-9x+12\) 写成内容乘本原多项式。</li><li>用 Eisenstein 判别法判断 \(x^5+10x+5\) 在 \(\mathbb Q[x]\) 中是否不可约。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 21, <em>Polynomial Rings</em> 整理。</p>
</section>
