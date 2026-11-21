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
<p>先在整数加法群里看 \(H=3\mathbb Z=\{\ldots,-3,0,3,\ldots\}\)。把 \(H\) 的每个数都加 1，就得到 \(1+H=\{\ldots,-2,1,4,\ldots\}\)；都加 2 得 \(2+H\)。这三块恰好按除以 3 的余数分完所有整数。\(1+H=4+H\)，因为加 1 与加 4 只差一个 \(H\) 中的元素；一个陪集有很多代表元。</p>
<p>一般地，若 \(H\leq G\)，用 \(g\) 左乘整个 \(H\) 得 \(gH=\{gh:h\in H\}\)，称为<strong>左陪集</strong>。它是一个集合，不是元素 \(g\)。什么时候 \(a,b\) 在同一块？恰当条件是 \(a^{-1}b\in H\)：这相当于说，从 \(a\) 走到 \(b\) 只需再做一个 \(H\) 内的动作。</p>
<p>这个条件是等价关系：\(a^{-1}a=e\in H\)；若 \(a^{-1}b\in H\)，取逆得 \(b^{-1}a\in H\)；若 \(a^{-1}b,b^{-1}c\in H\)，相乘得 \(a^{-1}c\in H\)。\(g\) 的等价类正是 \(gH\)，所以不同陪集或相同或不交，并且合起来是整个 \(G\)。这才是“陪集分割群”的理由。</p>
<figure class="note-figure"><div class="coset-partition" role="img" aria-label="整数分成模三余数为零、一、二的三个左陪集"><div><strong>\(0+3\mathbb Z\)</strong><span>\(\ldots,-6,-3,0,3,6,\ldots\)</span></div><div><strong>\(1+3\mathbb Z\)</strong><span>\(\ldots,-5,-2,1,4,7,\ldots\)</span></div><div><strong>\(2+3\mathbb Z\)</strong><span>\(\ldots,-4,-1,2,5,8,\ldots\)</span></div></div><figcaption>三行互不相交，并且合起来是整个整数群。</figcaption></figure>

<h2 id="section-9">二、Lagrange 定理的计数从哪里来？</h2>
<p>上图每一行都和 \(H\) 一样大。一般情形也如此：固定 \(g\)，映射 \(h\mapsto gh\) 把 \(H\) 一一对应到 \(gH\)；乘 \(g^{-1}\) 就能返回。若有限群有 \([G:H]\) 个不同陪集，每块都有 \(|H|\) 个元素，故</p>
<div class="equation">\[|G|=[G:H]\,|H|.\]</div>
<p>这就是 <strong>Lagrange 定理</strong>。例如 \(D_3\) 有 6 个对称，旋转子群有 3 个元素，因而只有 2 个陪集。它也推出：任何子群的阶都整除 \(|G|\)；特别地，元素 \(g\) 生成的子群 \(\langle g\rangle\) 的大小也整除 \(|G|\)。但整除只是必要条件，不保证每个因子都对应子群。</p>
<p>左陪集 \(gH\) 与右陪集 \(Hg\) 在非交换群中可以不同。本讲只需它们作为集合来计数；到第 9 讲才会问这些块本身何时可以相乘。</p>
<p>具体取 \(D_3\) 的反射子群 \(H=\{e,F\}\)。则 \(RH=\{R,RF\}\)，而 \(HR=\{R,FR\}\)；第 1 讲已经算出 \(RF\ne FR\)，所以这两个陪集不同。由此可见“左”字不是装饰，后面讨论正规子群时必须认真处理。</p>
<h2 id="section-10">理解检查</h2>
<ol><li>写出 \(D_3\) 对 \(H=\{e,R,R^2\}\) 的全部左陪集，并核对阶数公式。</li><li>证明 \(aH=bH\iff a^{-1}b\in H\)。</li><li>若有限群的阶是素数，取任意非单位元素 \(g\)，\(\langle g\rangle\) 有多大？</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 3, <em>Cosets</em> 整理。</p>
</section>
