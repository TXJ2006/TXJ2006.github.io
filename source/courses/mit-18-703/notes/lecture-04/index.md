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
<p>在三角形的旋转子群里，只要拿到 \(R\)，不断做 \(R\) 就会依次得到 \(R,R^2,e\)，又回到起点。对一般 \(a\in G\)，把所有正次、负次和零次幂收在一起，得到 \(\langle a\rangle=\{a^k:k\in\mathbb Z\}\)，称为 \(a\) <strong>生成的子群</strong>。若它就是整个 \(G\)，则 \(G\) 是循环群。</p>
<p>元素的<strong>阶</strong>是 \(\operatorname{ord}(a)=|\langle a\rangle|\)。有限时，它也是最早回到单位元的步数：最小的正整数 \(r\) 满足 \(a^r=e\)。把任意整数 \(k\) 除以 \(r\) 写成 \(k=qr+s\)（\(0\le s\lt r\)），便有 \(a^k=a^s\)。由 \(r\) 的最小性，\(e,a,\ldots,a^{r-1}\) 两两不同；因此 \(a^k=e\) 当且仅当 \(r\mid k\)。</p>
<p>上一讲的 Lagrange 定理给 \(\operatorname{ord}(a)\mid |G|\)。若 \(|G|=p\) 为素数，任何 \(a\ne e\) 的阶只能是 \(p\)，所以 \(a\) 生成全群。循环群必交换，因为 \(a^ra^s=a^{r+s}=a^sa^r\)；但交换群未必循环，例如 \(\mathbb Z_2\times\mathbb Z_2\) 没有四阶元素。</p>
<p>若 \(a\) 的阶为 12，每次做 \(a^8\) 相当于在 12 格圆环上跨 8 格：\(0\to8\to4\to0\)，所以 \(a^8\) 的阶是 3。一般地，\((a^k)^t=e\iff r\mid kt\)，最小正步数为 \(r/\gcd(r,k)\)。先把这个例子算通，再记公式就不会把 \(r\) 和 \(k\) 的作用弄反。</p>

<h2 id="section-12">二、\(\mathbb Z/n\mathbb Z\) 的元素到底是什么？</h2>
<p>把整数按“相差 \(n\) 的倍数”分组，就得到 \(\mathbb Z/n\mathbb Z\)。符号 \([a]\) 表示整个集合 \(a+n\mathbb Z\)，不是只表示整数 \(a\)；例如模 6 时 \([2]=[8]\)。我们想规定 \([a]+[b]=[a+b]\)，但同一类可换代表，必须先检查答案不会变。</p>
<p>若改写为 \(a'=a+kn,b'=b+ln\)，则 \((a'+b')-(a+b)=(k+l)n\)，所以两种答案属于同一类。这叫运算<strong>良定义</strong>。加法单位元是 \([0]\)，\([a]\) 的逆是 \([-a]\)，\([1]\) 生成全部 \(n\) 个类。</p>
<p>类之间也能乘：\([a][b]=[ab]\)。但全体类在乘法下不是群，因为 \([0]\) 没有乘法逆元。只保留有逆元的类，得到单位群 \(U_n\)。\([a]\) 可逆恰好当 \(\gcd(a,n)=1\)：Bézout 等式 \(au+nv=1\) 给出 \([a][u]=[1]\)。例如 \(U_8=\{[1],[3],[5],[7]\}\)，其中后三者平方都是 \([1]\)，所以它不是四阶循环群。</p>

<h2 id="section-13">三、Euler 与 Fermat 定理只是群阶结论吗？</h2>
<p>记 \(\varphi(n)=|U_n|\)。对 \(\gcd(a,n)=1\)，把有限群结论用于 \([a]\in U_n\)，即得 \(a^{\varphi(n)}\equiv1\pmod n\)。素数 \(p\) 时 \(\varphi(p)=p-1\)，得到 Fermat 小定理的非零情形；若 \(p\mid a\)，另有 \(a^p\equiv a\equiv0\pmod p\)，合起来得到对任意整数 \(a\) 的 \(a^p\equiv a\pmod p\)。</p>
<p>数出 \(1\) 至 \(p^k\) 中不被 \(p\) 整除的数，可得 \(\varphi(p^k)=p^k-p^{k-1}\)。互素的 \(m,n\) 还满足 \(\varphi(mn)=\varphi(m)\varphi(n)\)；这可由中国剩余定理给出的单位群同构证明，但原讲义此处把证明留到后面，不能把“乘法性”误认为对所有 \(m,n\) 都成立。</p>
<h2 id="section-14">理解检查</h2>
<ol><li>在 \(\mathbb Z/6\mathbb Z\) 中分别列出加法群的生成元和乘法单位群。</li><li>若 \(a\) 阶为 12，求 \(a^8\) 的阶；先用 \((a^8)^t=e\) 求最小 \(t\)。</li><li>用 Bézout 等式求 \([7]\) 在 \(U_{20}\) 中的逆元。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 4, <em>Cyclic Groups</em> 整理；同余类的运算先核对良定义，再讨论群结构。</p>
</section>
