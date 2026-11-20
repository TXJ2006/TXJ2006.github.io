---
title: 第 17 讲：分式域｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-17/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-16/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-18/">下一讲 →</a></p>
<div class="note-lead"><p>整数没有除法，但可以嵌入有理数。对任意整环，分式域用同样方法添加所有非零元素的逆，并且是“最小的域化”构造。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-17.pdf">MIT 第 17 讲 PDF</a></nav>

<h2 id="section-62">一、分式不是一对代表元</h2>
<p>设 \(R\) 是整环，取 \(R\times(R\setminus\{0\})\) 中的有序对 \((a,b)\)，把它理解为 \(a/b\)。不同有序对可能代表同一个分式，因此定义 \((a,b)\sim(c,d)\) 当且仅当 \(ad=bc\)。整环的消去律保证传递性成立；没有消去律时这个关系会失去所需性质。</p>
<p>分式域由这些等价类组成，记作 \(\operatorname{Frac}(R)\)。运算为
\[ [a,b]+[c,d]=[ad+bc,bd],\qquad [a,b][c,d]=[ac,bd]. \]
换代表元后的等式由交叉相乘验证，因此运算良定义。\([0,1]\) 是零元，\([1,1]\) 是单位元，非零元素 \([a,b]\) 的逆是 \([b,a]\)。</p>

<h2 id="section-63">二、为什么必须先是整环？</h2>
<p>若 \([a,b]=[0,1]\)，则 \(a=0\)，所以 \([a,b]\) 非零正好对应分子非零。要证明 \([a,b][b,a]=[1,1]\)，需要 \(a,b\ne0\) 且分母乘积仍非零；这正是整环条件在构造中的作用。对于有零因子的环，不能直接把所有非零元素都当作分母而得到这样的域。</p>
<p>自然嵌入 \(R\to\operatorname{Frac}(R)\)、\(a\mapsto[a,1]\) 是单射：若 \([a,1]=[c,1]\)，则 \(a=c\)。因此可以把 \(R\) 视为分式域的子环。</p>

<h2 id="section-64">三、泛性质：它是所有域嵌入的共同起点</h2>
<p>若 \(\psi:R\to K\) 是到域的单射，则唯一存在域同态 \(\widetilde\psi:\operatorname{Frac}(R)\to K\)，满足 \(\widetilde\psi([a,b])=\psi(a)\psi(b)^{-1}\)。代表元改变时，\(ad=bc\) 在 \(K\) 中仍成立，所以定义良好；唯一性来自每个分式都等于 \([a,1][b,1]^{-1}\)。</p>
<p>\(R=\mathbb Z\) 时得到 \(\mathbb Q\)；\(R=\mathbb Z[i]\) 时得到系数在 \(\mathbb Q\) 中的高斯数；\(R=K[x]\) 时得到有理函数域 \(K(x)\)。</p>
<h2 id="section-65">理解检查</h2>
<ol><li>证明 \([a,b]=[ka,kb]\)（\(k\ne0\)）并说明这不是代表元唯一。</li><li>验证分式加法的结合律可以从分子分母的普通运算继承。</li><li>为什么 \(\operatorname{Frac}(R)\) 的泛性质要求目标映射是单射？</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 17, <em>Field of Fractions</em> 整理。</p>
</section>
