---
title: 第 19 讲：特殊整环｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-19/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-18/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-20/">下一讲 →</a></p>
<div class="note-lead"><p>整数的素因数分解包含两个不同问题：分解是否存在，以及分解是否唯一。本讲用整环中的整除、不可约元、素元和理想链把这两个问题拆开。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-19.pdf">MIT 第 19 讲 PDF</a></nav>

<h2 id="section-70">一、素元、不可约元与伴随</h2>
<p>整数的 \(120=2^3\cdot3\cdot5\) 也能写成 \((-2)^3\cdot3\cdot(-5)\)。这不算两种不同的素因数分解，因为 \(2\) 与 \(-2\) 只差一个可逆因子 \(-1\)。在一般整环中，\(a\mid b\) 表示 \(b=qa\)；若非零 \(a,b\) 互相整除，就称<strong>伴随</strong>，等价于 \(a=ub\)，其中 \(u\) 是单位，也等价于 \((a)=(b)\)。因此“唯一”最多只能要求到伴随和重排为止。</p>
<p>还有两个看似相近的概念。非零非单位 \(p\) 若 \(p\mid ab\Rightarrow p\mid a\) 或 \(p\mid b\)，称<strong>素元</strong>；若 \(p=ab\) 时 \(a,b\) 至少有一个是单位，称<strong>不可约元</strong>。素元一定不可约；反方向在一般整环中不成立。前者控制分解的<strong>唯一性</strong>，后者是反复拆分<strong>终止时</strong>得到的对象。</p>

<h2 id="section-71">二、ACC 如何保证分解存在？</h2>
<p>从非零非单位元素 \(a_1\) 开始，若可非平凡地写成 \(a_1=a_2b_2\)，就继续拆 \(a_2\)。每拆一次，\((a_1)\subseteq(a_2)\)；若两者相等，在整环里消去非零因子可推出 \(b_2\) 是单位，和“非平凡拆分”矛盾。因此无限拆分会给严格递增的主理想链：</p>
<div class="equation">\[(a_1)\subsetneq(a_2)\subsetneq(a_3)\subsetneq\cdots,\qquad a_i=a_{i+1}b_{i+1}.\]</div>
<p>若这样的链不可能无限继续，即满足升链条件（ACC），分解就必须终止，得到不可约元的乘积。PID 指每个理想都由一个元素生成；在 PID 中，递增理想链的并仍是理想，生成这个并的元素必出现在链的某一阶段，所以链随后稳定。故 PID 满足 ACC。</p>

<h2 id="section-72">三、UFD、最大公因子与 PID</h2>
<p>整环称唯一分解整环（UFD），如果每个非零非单位元素都能分解为素元乘积，且分解只差顺序和伴随。若 \(R\) 是 UFD，把两个元素的素因子分解中每个素元的指数取最小值，就得到最大公因子。</p>
<p>PID 还保证 Bézout 等式。由 \((a,b)=(d)\) 可知 \(d=ra+sb\)；\(d\) 同时整除 \(a,b\)，而每个公因子又整除 \(d\)，所以 \(d\) 是最大公因子。具体地，\(1=-13\cdot15+7\cdot28\)；若 \(15\mid28c\)，两边乘 \(c\)，右侧两项都被 15 整除，故 \(15\mid c\)。</p>
<p>一般地，若不可约元 \(p\mid bc\)，取 \(d=\gcd(p,b)\)。不可约性使 \(d\) 只能是单位或与 \(p\) 伴随：后者给 \(p\mid b\)；前者有 \(1=rp+sb\)，乘 \(c\) 并利用 \(p\mid bc\)，得到 \(p\mid c\)。所以 PID 中不可约元也是素元；结合 ACC，PID 是 UFD。</p>
<h2 id="section-73">理解检查</h2>
<ol><li>在 \(\mathbb Z\) 中区分 \(2\) 与 \(-2\) 的关系，并说明为什么分解中的符号不影响唯一性。</li><li>解释严格递增的主理想链为何对应不断非平凡分解。</li><li>若 \(\gcd(a,b)=1\) 且 \(a\mid bc\)，用 Bézout 等式证明 \(a\mid c\)。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 19, <em>Special Domains</em> 整理。</p>
</section>
