---
title: 第 9 讲：商群与泛性质｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-09/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-08/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-10/">下一讲 →</a></p>
<div class="note-lead"><p>第 3 讲已经把群分成陪集，第 8 讲说明核一定正规。现在的问题是：能否把每个陪集当作一个新元素，并让这些新元素仍组成群？</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-09.pdf">MIT 第 9 讲 PDF</a></nav>

<h2 id="section-30">一、乘积为什么不能直接写下就算完成？</h2>
<p>给定 \(N\trianglelefteq G\)，希望定义 \((aN)(bN)=abN\)。难处在于同一陪集有许多代表元。若 \(a'=an,b'=bm\)（\(n,m\in N\)），则 \(a'b'=anbm=ab(b^{-1}nb)m\)。正规性保证 \(b^{-1}nb\in N\)，所以 \(a'b'\in abN\)，即 \(a'b'N=abN\)。只有完成这一步，乘法才是<strong>良定义</strong>的。若 \(N\) 不正规，从不同代表元出发可能得到不同陪集。</p>
<p>于是陪集集合 \(G/N\) 成群：单位元是 \(N=eN\)，\((aN)^{-1}=a^{-1}N\)，结合律继承自 \(G\)。自然映射 \(\pi:G\to G/N\)、\(g\mapsto gN\) 是满同态，且 \(\ker\pi=N\)。这证明第 8 讲“核必正规”的逆向说法：<strong>每个正规子群都能作为某个同态的核</strong>。例如 \(\mathbb Z/3\mathbb Z\) 的三个元素，就是 \(3\mathbb Z\) 在 \(\mathbb Z\) 中的三个加法陪集。</p>
<p>正规性也确实必要。假如陪集乘法良定义，\((gN)(g^{-1}N)=N\)。把第一个陪集的代表元换为 \(gn\)（\(n\in N\)），积仍须是 \(N\)，所以 \(gng^{-1}\in N\)。对任意 \(g,n\) 都成立，正是 \(N\trianglelefteq G\)。因此“正规”不是人为多加的条件，而是商运算能存在的要求。</p>

<h2 id="section-31">二、商群究竟“忘掉”了什么？</h2>
<p>\(aN=bN\iff a^{-1}b\in N\)：商映射忘掉了两元素之间属于 \(N\) 的差别。若 \(G\) 有限，\(|G/N|=[G:N]=|G|/|N|\)。例如 \(S_3/A_3\) 有两个元素，记录置换的奇偶性；在商群中，两个不同对换被看成同一个元素。商群绝不是从 \(G\) 中挑出若干代表元组成的子群，它的元素本身是集合。</p>

<h2 id="section-32">三、泛性质如何避免重复做代表元计算？</h2>
<p>若同态 \(f:G\to K\) 满足 \(N\subseteq\ker f\)，则 \(f\) 对同一陪集取相同值，故唯一地降为 \(\bar f:G/N\to K\)，由 \(\bar f(gN)=f(g)\) 定义，且 \(f=\bar f\circ\pi\)。良定义来自 \(aN=bN\Rightarrow a^{-1}b\in N\subseteq\ker f\Rightarrow f(a)=f(b)\)。这叫商的<strong>泛性质</strong>；它说：任何已经把 \(N\) 压成单位元的同态，都必须经过 \(G/N\)。讲义用范畴语言表达这一点，真正计算时先记住这个唯一分解即可。</p>
<h2 id="section-33">理解检查</h2>
<ol><li>在 \(S_3\) 中令 \(H=\{e,(1\ 2)\}\)。找一个 \(g\) 使 \(gH\ne Hg\)，解释为什么不能按上式构造 \(S_3/H\)。</li><li>验证 \(\mathbb Z/6\mathbb Z\to\mathbb Z/3\mathbb Z\)、\([a]_6\mapsto[a]_3\) 的核，并写出它经过哪个商群。</li><li>证明 \(\bar f\) 是同态，以及为什么它只能有这一个定义。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 9, <em>Quotient Groups</em> 整理。</p>
</section>
