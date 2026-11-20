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
<p>在整环中，\(a\mid b\) 表示存在 \(q\) 使 \(b=qa\)。两个非零元素互为伴随，当且仅当它们相差一个单位，即 \(a=ub\)。这也等价于主理想相同：\((a)=(b)\)。所以因式分解不可能比“乘以单位”更唯一。</p>
<p>非单位 \(p\) 若 \(p\mid ab\) 必推出 \(p\mid a\) 或 \(p\mid b\)，称素元；非单位且 \(a=bc\) 时必有 \(b\) 或 \(c\) 为单位，称不可约元。素元一定不可约，但一般整环中不可约元未必是素元；这正是唯一分解可能失败的原因。</p>

<h2 id="section-71">二、ACC 如何保证分解存在？</h2>
<p>把一个非零非单位元素不断分解。若过程无限进行，就得到严格递增的主理想链
\[ (a_1)\subsetneq(a_2)\subsetneq(a_3)\subsetneq\cdots, \qquad a_i=a_{i+1}b_{i+1}. \]
因此，只要主理想满足升链条件（ACC），分解过程必终止，得到不可约元的乘积。PID 中每个理想都是主理想，任意递增理想链的并仍是理想且必由某个阶段的元素生成，所以 PID 自动满足 ACC。</p>

<h2 id="section-72">三、UFD、最大公因子与 PID</h2>
<p>整环称唯一分解整环（UFD），如果每个非零非单位元素都能分解为素元乘积，且分解只差顺序和伴随。若 \(R\) 是 UFD，把两个元素的素因子分解中每个素元的指数取最小值，就得到最大公因子。</p>
<p>在 PID 中，\((a,b)=(d)\)，所以 \(d=ra+sb\)；这就是 Bézout 形式。若不可约元 \(p\) 整除 \(bc\)，令 \(d=\gcd(p,b)\)。不可约性迫使 \(d\) 是单位或与 \(p\) 伴随：前者由 Bézout 等式推出 \(p\mid c\)，后者直接给出 \(p\mid b\)。故 PID 中不可约元都是素元，结合 ACC 得到 PID 是 UFD。</p>
<h2 id="section-73">理解检查</h2>
<ol><li>在 \(\mathbb Z\) 中区分 \(2\) 与 \(-2\) 的关系，并说明为什么分解中的符号不影响唯一性。</li><li>解释严格递增的主理想链为何对应不断非平凡分解。</li><li>用 Bézout 等式证明 \(\gcd(15,28)=1\) 时，15 不整除 28 的任意素因子。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 19, <em>Special Domains</em> 整理。</p>
</section>
