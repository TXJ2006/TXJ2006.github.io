---
title: 第 3 讲：陪集与 Lagrange 定理｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-03/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-02/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-04/">下一讲 →</a></p>
<div class="note-lead"><p>子群 \(H\) 只占 \(G\) 的一部分。把 \(H\) 用每个 \(g\) 左乘，会得到大小相同、互不重叠的块；有限群的阶数限制便从这里产生。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-03.pdf">MIT 第 3 讲 PDF</a></nav>

<h2 id="section-8">一、陪集为什么恰好分割群？</h2>
<p>给定 \(H\leq G\)，定义 \(gH=\{gh:h\in H\}\)，称为 \(H\) 的<strong>左陪集</strong>。规定 \(a\sim b\) 当且仅当 \(a^{-1}b\in H\)。反身性来自 \(a^{-1}a=e\)；对称性来自取逆；若 \(a^{-1}b,b^{-1}c\in H\)，则 \(a^{-1}c=(a^{-1}b)(b^{-1}c)\in H\)，所以传递。于是这是等价关系，\(g\) 的等价类正是 \(gH\)。等价类或相同或不交，故左陪集分割 \(G\)。</p>
<p>代表元不是唯一的：\(aH=bH\) 当且仅当 \(a^{-1}b\in H\)。在 \((\mathbb Z,+)\) 中，\(H=3\mathbb Z\) 给出 \(0+H,1+H,2+H\) 三块；\(1+H=4+H\) 是同一块。这里的“加 1”是把整个子群平移，不能把陪集误当成单个元素。</p>
<figure class="note-figure"><div class="coset-partition" role="img" aria-label="整数分成模三余数为零、一、二的三个左陪集"><div><strong>\(0+3\mathbb Z\)</strong><span>\(\ldots,-6,-3,0,3,6,\ldots\)</span></div><div><strong>\(1+3\mathbb Z\)</strong><span>\(\ldots,-5,-2,1,4,7,\ldots\)</span></div><div><strong>\(2+3\mathbb Z\)</strong><span>\(\ldots,-4,-1,2,5,8,\ldots\)</span></div></div><figcaption>三行互不相交，并且合起来是整个整数群。</figcaption></figure>

<h2 id="section-9">二、Lagrange 定理的计数从哪里来？</h2>
<p>固定 \(g\) 后，\(H\to gH\)、\(h\mapsto gh\) 是双射，逆映射是 \(gh\mapsto g^{-1}(gh)=h\)。因此每个陪集与 \(H\) 等势。有限时若陪集数是 \([G:H]\)，分割立刻给出 \(|G|=[G:H]|H|\)。这就是 <strong>Lagrange 定理</strong>；所以子群的阶必整除群的阶，元素 \(g\) 的阶 \(|\langle g\rangle|\) 也必整除 \(|G|\)。逆命题一般不成立：一个整除 \(|G|\) 的数未必对应某个子群。</p>
<p>左陪集 \(gH\) 与右陪集 \(Hg\) 在非交换群中可以不同。本讲只需它们作为集合来计数；到第 9 讲才会问这些块本身何时可以相乘。</p>
<p>具体取 \(D_3\) 的反射子群 \(H=\{e,F\}\)。则 \(RH=\{R,RF\}\)，而 \(HR=\{R,FR\}\)；第 1 讲已经算出 \(RF\ne FR\)，所以这两个陪集不同。由此可见“左”字不是装饰，后面讨论正规子群时必须认真处理。</p>
<h2 id="section-10">理解检查</h2>
<ol><li>写出 \(D_3\) 对 \(H=\{e,R,R^2\}\) 的全部左陪集，并核对阶数公式。</li><li>证明 \(aH=bH\iff a^{-1}b\in H\)。</li><li>若有限群的阶是素数，取任意非单位元素 \(g\)，\(\langle g\rangle\) 有多大？</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 3, <em>Cosets</em> 整理。</p>
</section>
