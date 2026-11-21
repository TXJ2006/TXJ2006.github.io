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
<p>在 \(\mathbb Q\) 中，\(1/2\) 与 \(2/4\) 是同一个数，但写成了不同的“分子、分母”对。一般整环 \(R\) 也照此构造：先取所有 \((a,b)\)，其中 \(a\in R\)、\(b\in R\setminus\{0\}\)；再规定 \((a,b)\sim(c,d)\) 当且仅当 \(ad=bc\)。等价类 \([a,b]\) 才是分式本身，有序对只是一个写法。</p>
<p>为什么要求整环？例如从 \(ad=bc\)、\(cf=de\) 推出 \(af=be\) 时，会得到 \(afd=bed\)，必须消去非零的 \(d\)。有零因子的环不能这样做，等价关系的传递性就可能失败。</p>
<p>加法和乘法沿用分数的规则。例如 \([1,2]+[1,3]=[5,6]\)。一般式子是
\[ [a,b]+[c,d]=[ad+bc,bd],\qquad [a,b][c,d]=[ac,bd]. \]
换代表元后用交叉相乘核对，可知答案仍是同一个类。\([0,1]\) 是零元，\([1,1]\) 是乘法单位元；若 \(a\ne0\)，\([a,b]\) 的逆是 \([b,a]\)。这些类构成的域记作 \(\operatorname{Frac}(R)\)。</p>

<h2 id="section-63">二、为什么必须先是整环？</h2>
<p>分式 \([a,b]\) 为零当且仅当 \(a=0\)。要用 \([b,a]\) 做它的逆，\(a\) 必须非零；而两个非零分母相乘也必须非零，否则乘法会离开允许的有序对集合。整环恰好保证这些步骤。有零因子的环不能照搬“所有非零元素都可当分母”的做法。</p>
<p>自然嵌入 \(R\to\operatorname{Frac}(R)\)、\(a\mapsto[a,1]\) 是单射：若 \([a,1]=[c,1]\)，则 \(a=c\)。因此可以把 \(R\) 视为分式域的子环。</p>

<h2 id="section-64">三、泛性质：它是所有域嵌入的共同起点</h2>
<p>若 \(\psi:R\to K\) 是到域的单射，则唯一存在域同态 \(\widetilde\psi:\operatorname{Frac}(R)\to K\)，满足 \(\widetilde\psi([a,b])=\psi(a)\psi(b)^{-1}\)。代表元改变时，\(ad=bc\) 在 \(K\) 中仍成立，所以定义良好；唯一性来自每个分式都等于 \([a,1][b,1]^{-1}\)。</p>
<p>\(R=\mathbb Z\) 时得到 \(\mathbb Q\)；\(R=\mathbb Z[i]\) 时得到系数在 \(\mathbb Q\) 中的高斯数；\(R=K[x]\) 时得到有理函数域 \(K(x)\)。</p>
<h2 id="section-65">理解检查</h2>
<ol><li>证明 \([a,b]=[ka,kb]\)（\(k\ne0\)）并说明这不是代表元唯一。</li><li>验证分式加法的结合律可以从分子分母的普通运算继承。</li><li>为什么 \(\operatorname{Frac}(R)\) 的泛性质要求目标映射是单射？</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 17, <em>Field of Fractions</em> 整理。</p>
</section>
