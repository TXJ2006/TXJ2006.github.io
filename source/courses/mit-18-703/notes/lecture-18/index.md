---
title: 第 18 讲：素理想与极大理想｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-18/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-17/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-19/">下一讲 →</a></p>
<div class="note-lead"><p>理想的大小关系由商环的性质来刻画：商环是整环对应素理想，是域对应极大理想。于是抽象的理想包含关系变成可计算的代数性质。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-18.pdf">MIT 第 18 讲 PDF</a></nav>

<h2 id="section-66">一、素理想等价于商环无零因子</h2>
<p>先对比两个商环：\(\mathbb Z/5\mathbb Z\) 中非零类相乘不会得到零；\(\mathbb Z/6\mathbb Z\) 中却有 \([2][3]=[0]\)。两者都是从整数对理想取商而来，区别恰好由被压成零的理想决定：\((5)\) 保留了“非零乘非零仍非零”，\((6)\) 没有。</p>
<p>真理想 \(I\subsetneq R\) 若满足 \(ab\in I\Rightarrow a\in I\) 或 \(b\in I\)，称<strong>素理想</strong>。在商环中，\((a+I)(b+I)=I\) 正好等价于 \(ab\in I\)，所以 \(I\) 素当且仅当 \(R/I\) 没有零因子。在交换环的语境下，就是商环为整环。看商环往往比直接检查所有 \(a,b\) 更快。</p>

<h2 id="section-67">二、极大理想等价于商环是域</h2>
<p>“没有零因子”还弱于“每个非零元素都可逆”。\(\mathbb Z\) 没有零因子，但 2 没有整数乘法逆元。对应到理想：素理想使商环成为整环；若还希望商环成为域，就需要更强的<strong>极大理想</strong>。</p>
<p>真理想 \(M\) 极大，指不存在严格夹在 \(M\) 与 \(R\) 之间的理想。包含 \(M\) 的原环理想与商环 \(R/M\) 的理想一一对应；因此 \(M\) 极大时，\(R/M\) 只有零理想和整个环。</p>
<p>对交换环，任取非零 \(a\in R/M\)，其生成的理想 \((a)\) 非零，便只能是整个环；于是 \(1=ra\)，\(a\) 可逆。反向也成立，故</p>
<div class="equation">\[M\text{ 极大}\iff R/M\text{ 是域}.\]</div>
域没有零因子，所以每个极大理想都素。反向一般不成立：\((x^2+1)\subset\mathbb Z[x]\) 的商环是 \(\mathbb Z[i]\)，没有零因子却不是域；同一多项式放在 \(\mathbb R[x]\) 中，商环变成 \(\mathbb C\)，理想就极大了。</p>

<h2 id="section-68">三、特征从哪里来？</h2>
<p>任意带单位环都有唯一环同态 \(\mathbb Z\to R\)，由 \(1\mapsto1\) 决定。若最小正整数 \(n\) 满足 \(n\cdot1=0\)，称环的特征为 \(n\)；若不存在这样的整数，特征为 0。若 \(R\) 是整环且特征有限，\(n\) 必为素数：若 \(n=ab\) 可分解，则 \((a1)(b1)=n1=0\)，整环迫使某个因子为零，违背最小性。</p>
<p>\(\mathbb Q,\mathbb R,\mathbb C\) 的特征为 0，\(\mathbb Z_p\) 的特征为 \(p\)。函数环的“在一点取值”映射给出重要例子：在 \(R^X\) 中，某点处为零的函数构成理想，商环同构于 \(R\)，因此该理想是否素或极大完全取决于 \(R\) 是否整环或域。</p>
<h2 id="section-69">理解检查</h2>
<ol><li>比较 \((x^2+1)\) 在 \(\mathbb R[x]\) 与 \(\mathbb Z[x]\) 中的地位：前者的商环同构于 \(\mathbb C\)，故理想极大；后者的商环同构于 \(\mathbb Z[i]\)，故理想素但不极大。</li><li>为什么“极大理想必素”只需要域是整环这一事实？</li><li>求 \(\mathbb Z/12\mathbb Z\) 的特征，并判断 \((4)\) 是否为素理想。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 18, <em>Prime and Maximal Ideals</em> 整理。</p>
</section>
