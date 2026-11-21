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
<p>在 \(R[x]\) 中，多项式 \(f(x)=x^2-2\) 还只是一个形式表达式；选定一个数 \(\alpha\) 并代入，才得到数值 \(f(\alpha)=\alpha^2-2\)。把“代入 \(\alpha\)”视为映射 \(\operatorname{ev}_\alpha:R[x]\to R\)，问题“\(\alpha\) 是根吗？”就变成“\(f\) 会被这个映射送到 0 吗？”</p>
<p>若系数在域 \(K\) 中，多项式带余除法给 \(f(x)=(x-\alpha)q(x)+r\)，其中余数 \(r\) 是常数。代入 \(\alpha\) 得 \(r=f(\alpha)\)，于是
\[f(\alpha)=0\iff x-\alpha\mid f(x).\]
二次或三次多项式若能分成两个正次数因子，其中必有一次因子；故它在域上不可约，当且仅当没有根。四次多项式可能拆成两个二次因子，不能只检查根。</p>
<p>多项式环还有一个统一性质：给定环同态 \(\phi:R\to S\) 和元素 \(s\in S\)，恰有一个环同态 \(R[x]\to S\) 在系数上等于 \(\phi\)、把 \(x\) 送到 \(s\)；多项式的每一项只能按这两项要求求值。这是它的泛性质。若 \(R\) 是整环，\(\deg(fg)=\deg f+\deg g\)，所以 \(R[x]\) 中可逆的多项式只能是 \(R\) 中的常数单位。</p>

<h2 id="section-79">二、内容与 Gauss 引理</h2>
<p>多项式若系数在整数中，却允许用有理系数分解，会不会出现原来没有的因子？例如 \(6x^2+9x+3=3(2x^2+3x+1)\)：系数的公因子 3 应先单独提出。对 UFD \(R\)，把系数的最大公因子称为内容 \(c(f)\)；内容为单位的多项式称<strong>本原</strong>。Gauss 引理的关键结论是</p>
<div class="equation">\[c(gh)\sim c(g)c(h).\]</div>
<p>这里 \(\sim\) 表示相差一个单位；特别地，本原乘本原仍本原。若不然，某个素元 \(p\) 会整除乘积的全部系数。取两个因子中各自<strong>第一个</strong>不被 \(p\) 整除的系数，它们乘积对应的系数里恰有这一项不被 \(p\) 整除，矛盾。这是证明中真正起作用的一步。</p>
<p>设 \(F=\operatorname{Frac}(R)\)。Gauss 引理说明：本原 \(f\in R[x]\) 若在 \(F[x]\) 中可分解，就能把分母清掉并重新调整单位，得到 \(R[x]\) 中同次数分解。因此 \(R\) 为 UFD 时，\(R[x]\) 也是 UFD；特别地，\(\mathbb Z[x]\) 是 UFD。</p>

<h2 id="section-80">三、Eisenstein 判别法为什么有效？</h2>
<p>实际判定不可约时，可以先看简单例子 \(x^5+10x+5\)。除首项外的系数都被 5 整除，常数项 5 却不被 \(5^2\) 整除。<strong>Eisenstein 判别法</strong>说：若 \(f(x)=a_nx^n+\cdots+a_0\in\mathbb Z[x]\) 存在素数 \(p\) 满足 \(p\mid a_i\)（\(i\lt n\)）、\(p\nmid a_n\)、\(p^2\nmid a_0\)，则 \(f\) 在 \(\mathbb Q[x]\) 中不可约。上例取 \(p=5\)。</p>
<p>理由是模 \(p\) 后只剩 \(\bar f=\bar a_nx^n\)。若 \(f=gh\) 有两个正次数整数因子，它们模 \(p\) 后只能各是常数乘一个 \(x\) 的幂，因此两者常数项都被 \(p\) 整除；它们的乘积 \(a_0\) 就会被 \(p^2\) 整除，矛盾。从 \(\mathbb Q[x]\) 中的分解转回整数分解，用的是上一段 Gauss 引理。</p>
<p>例如下面的多项式用 \(p=3\) 满足判别条件：</p>
<div class="equation">\[\begin{aligned}f(x)={}&2x^7-15x^6+60x^5-18x^4\\&{}-9x^3+45x^2-3x+6.\end{aligned}\]</div>
<p>圆分多项式 \(1+x+\cdots+x^{p-1}\) 则通过代换 \(x=y+1\) 转化为 Eisenstein 形式。</p>
<h2 id="section-81">理解检查</h2>
<ol><li>为什么“有根”只足以判定二次、三次多项式可约，不能直接用于四次多项式？</li><li>把 \(6x^3-9x+12\) 写成内容乘本原多项式。</li><li>用 Eisenstein 判别法判断 \(x^4+6x+3\) 在 \(\mathbb Q[x]\) 中是否不可约。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 21, <em>Polynomial Rings</em> 整理。</p>
</section>
