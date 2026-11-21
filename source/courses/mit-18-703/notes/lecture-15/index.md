---
title: 第 15 讲：环的基本性质｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-15/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-14/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-16/">下一讲 →</a></p>
<div class="note-lead"><p>本讲把环公理真正用起来：先推出零和负号的乘法规则，再区分交换环、除环、域、整环与零因子。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-15.pdf">MIT 第 15 讲 PDF</a></nav>

<h2 id="section-54">一、零和负号不是额外公理</h2>
<p>由分配律，\(a0=a(0+0)=a0+a0\)。在加法群中消去一个 \(a0\)，得到 \(a0=0\)；同理 \(0a=0\)。于是 \(a(-b)+ab=a(-b+b)=0\)，所以 \(a(-b)=-(ab)\)，并且 \((-a)b=-(ab)\)。这些恒等式使得环中的符号运算可以像整数一样展开。</p>
<p>讲义还指出：如果一个结构满足除加法交换律以外的全部环公理，那么加法交换律其实也会被迫成立。比较 \((a+b)(1+1)\) 的两种展开方式，再用加法消去律，就得到 \(a+b=b+a\)。因此非交换性只能出现在乘法上。</p>

<h2 id="section-55">二、交换环、布尔环与除环</h2>
<p>乘法也交换的环称<strong>交换环</strong>。例如模 2 的环只有 \([0],[1]\)，两者都满足 \(a^2=a\)。若一个环里每个元素都像这样平方后不变，称为<strong>布尔环</strong>。它的乘法一定交换，但这不是定义里预先要求的。</p>
<p>证明分两步：把 \((a+b)^2\) 展开成 \(a^2+ab+ba+b^2\)，又由布尔条件等于 \(a+b=a^2+b^2\)，故 \(ab=-ba\)。令 \(b=1\)，得 \(a=-a\)，于是负号不会改变元素；代回 \(ab=-ba\) 得 \(ab=ba\)。</p>
<p>若 \(R\setminus\{0\}\) 在乘法下成群，\(R\) 称除环；若乘法还交换，则称域。\(\mathbb Q,\mathbb R,\mathbb C\) 是域，四元数是非交换除环，\(\mathbb Z\) 是整环但不是域。</p>

<h2 id="section-56">三、零因子与整环</h2>
<p>第 14 讲在 \(\mathbb Z/6\mathbb Z\) 看到 \([2][3]=0\)。这样的非零元素 \([2],[3]\) 叫<strong>零因子</strong>：一般地，非零 \(a\) 若能与某个非零 \(b\) 相乘得到 0（非交换时允许 \(ab=0\) 或 \(ba=0\)），就是零因子。</p>
<p>可逆元素不可能是零因子。若 \(ba=0\) 且 \(a^{-1}\) 存在，右乘 \(a^{-1}\) 得 \(b=0\)，与假设矛盾。没有零因子的环称 domain；乘法还交换时称 integral domain（整环）。所以域必是整环，但 \(\mathbb Z\) 是整环却不是域：2 没有整数乘法逆元。</p>
<p>函数环和矩阵环说明零因子很常见：在函数环中，支集分离的两个非零函数乘积可以为零；在矩阵环中，\(\begin{pmatrix}0&1\\0&0\end{pmatrix}^2=0\)。整环的价值在于可以消去非零因子，并且多项式次数满足 \(\deg(fg)=\deg f+\deg g\)，这会成为后面因式分解理论的基础。</p>
<h2 id="section-57">理解检查</h2>
<ol><li>用环公理证明 \((-a)(-b)=ab\)。</li><li>为什么域一定是整环，而整环不一定是域？</li><li>证明布尔环的乘法交换，指出令 \(b=1\) 的作用。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 15, <em>Basic Properties of Rings</em> 整理。</p>
</section>
