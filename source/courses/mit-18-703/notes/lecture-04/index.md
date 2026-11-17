---
title: 第 4 讲：循环群与模运算｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-04/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-03/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-05/">下一讲 →</a></p>
<div class="note-lead"><p>一个元素能生成多大的群？答案把 Lagrange 定理、模 \(n\) 加法、可逆剩余类和 Euler 定理串在一起。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-04.pdf">MIT 第 4 讲 PDF</a></nav>

<h2 id="section-11">一、从一个元素到循环群</h2>
<p>包含集合 \(S\) 的最小子群记 \(\langle S\rangle\)：它是所有包含 \(S\) 的子群之交。特别地，\(\langle a\rangle=\{a^k:k\in\mathbb Z\}\)；若它等于整个 \(G\)，则 \(G\) 是循环群。元素的阶 \(\operatorname{ord}(a)=|\langle a\rangle|\)：有限时等于使 \(a^r=e\) 的最小正整数 \(r\)。由除法算法，\(a^k=e\iff r\mid k\)；因此一个有限阶为 \(r\) 的元素只产生 \(e,a,\ldots,a^{r-1}\) 这 \(r\) 个不同元素。</p>
<p>Lagrange 定理给 \(\operatorname{ord}(a)\mid |G|\)，故有限群中 \(a^{|G|}=e\)。若 \(|G|=p\) 是素数，任一 \(a\ne e\) 的阶大于 1 又整除 \(p\)，只能是 \(p\)，所以整个群由 \(a\) 生成。循环群必交换，因为 \(a^ra^s=a^{r+s}=a^sa^r\)；反过来交换群未必循环，例如 \(\mathbb Z_2\times\mathbb Z_2\)。</p>
<p>幂的阶也可精确算。若 \(a\) 的阶为 \(r\)，则 \((a^k)^t=e\iff r\mid kt\)，使它成立的最小正 \(t\) 是 \(r/\gcd(r,k)\)。例如十二阶元素的八次幂只有三阶。它把“绕圈每次跨 \(k\) 格，几步回到起点”的直觉变成了整除计算。</p>

<h2 id="section-12">二、\(\mathbb Z/n\mathbb Z\) 的元素到底是什么？</h2>
<p>元素是整数的同余类 \([a]=a+n\mathbb Z\)，而不是某个固定代表整数。定义 \([a]+[b]=[a+b]\)。若换代表 \(a'=a+kn,b'=b+ln\)，则 \(a'+b'-(a+b)=(k+l)n\)，结果同余，故运算<strong>良定义</strong>。单位元为 \([0]\)，逆元为 \([-a]\)，\([1]\) 生成全部 \(n\) 个元素。乘法 \([a][b]=[ab]\) 同样良定义，却不会让全部剩余类成群，因为 \([0]\) 无乘法逆元。</p>
<p>可逆的乘法剩余类恰是 \(\gcd(a,n)=1\) 的 \([a]\)，构成单位群 \(U_n\)。理由是 Bézout 等式：存在整数 \(u,v\) 使 \(au+nv=1\) 当且仅当 \(a,n\) 互素，此时 \([u]\) 就是 \([a]^{-1}\)。例如 \(U_8=\{[1],[3],[5],[7]\}\) 有四个元素，但后三者平方都为 \([1]\)，所以它不是四阶循环群。</p>

<h2 id="section-13">三、Euler 与 Fermat 定理只是群阶结论吗？</h2>
<p>记 \(\varphi(n)=|U_n|\)。对 \(\gcd(a,n)=1\)，把有限群结论用于 \([a]\in U_n\)，即得 \(a^{\varphi(n)}\equiv1\pmod n\)。素数 \(p\) 时 \(\varphi(p)=p-1\)，得到 Fermat 小定理的非零情形；若 \(p\mid a\)，另有 \(a^p\equiv a\equiv0\pmod p\)，合起来得到对任意整数 \(a\) 的 \(a^p\equiv a\pmod p\)。</p>
<p>数出 \(1\) 至 \(p^k\) 中不被 \(p\) 整除的数，可得 \(\varphi(p^k)=p^k-p^{k-1}\)。互素的 \(m,n\) 还满足 \(\varphi(mn)=\varphi(m)\varphi(n)\)；这可由中国剩余定理给出的单位群同构证明，但原讲义此处把证明留到后面，不能把“乘法性”误认为对所有 \(m,n\) 都成立。</p>
<h2 id="section-14">理解检查</h2>
<ol><li>在 \(\mathbb Z/6\mathbb Z\) 中分别列出加法群的生成元和乘法单位群。</li><li>若 \(a\) 阶为 12，求 \(a^8\) 的阶；先用 \((a^8)^t=e\) 求最小 \(t\)。</li><li>用 Bézout 等式求 \([7]\) 在 \(U_{20}\) 中的逆元。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 4, <em>Cyclic Groups</em> 整理；同余类的运算先核对良定义，再讨论群结构。</p>
</section>
