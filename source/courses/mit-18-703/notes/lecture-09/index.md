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
<p>第 3 讲把整数分成 \(3\mathbb Z,1+3\mathbb Z,2+3\mathbb Z\) 三块。若把每一块当作一个新元素，自然想定义 \((a+3\mathbb Z)+(b+3\mathbb Z)=(a+b)+3\mathbb Z\)。例如用代表元 1 和 2 得到 0 类；换代表元 4 和 5，得到 9，仍是 0 类。换代表元后答案不变，才能真正把“块”拿来运算。</p>
<p>在一般群中，给定正规子群 \(N\)，希望定义 \((aN)(bN)=abN\)。需要证明代表元的选择不影响结果。若 \(a'=an,b'=bm\)，其中 \(n,m\in N\)，则</p>
<div class="equation">\[a'b'=anbm=ab(b^{-1}nb)m.\]</div>
<p>正规性保证 \(b^{-1}nb\in N\)，因此最后两个因子仍在 \(N\) 中，\(a'b'N=abN\)。这叫乘法<strong>良定义</strong>；若 \(N\) 不正规，换代表元就可能换出不同的答案。</p>
<p>这样得到的陪集集合 \(G/N\) 是<strong>商群</strong>，单位元是 \(N=eN\)，\((aN)^{-1}=a^{-1}N\)。自然映射 \(\pi:G\to G/N\)、\(g\mapsto gN\) 是满同态，核正好是 \(N\)。所以第 8 讲“核一定正规”的逆向也成立：每个正规子群都能作为某个同态的核。</p>
<p>正规性确实不能删：若陪集乘法对所有代表元都良定义，把 \((gN)(g^{-1}N)=N\) 中第一个 \(g\) 换成同陪集的 \(gn\)，乘积仍须是 \(N\)，得到 \(gng^{-1}\in N\)。任意 \(g,n\) 都如此，正是正规性。</p>

<h2 id="section-31">二、商群究竟“忘掉”了什么？</h2>
<p>\(aN=bN\iff a^{-1}b\in N\)：商映射忘掉了两元素之间属于 \(N\) 的差别。若 \(G\) 有限，\(|G/N|=[G:N]=|G|/|N|\)。例如 \(S_3/A_3\) 有两个元素，记录置换的奇偶性；在商群中，两个不同对换被看成同一个元素。商群绝不是从 \(G\) 中挑出若干代表元组成的子群，它的元素本身是集合。</p>

<h2 id="section-32">三、泛性质如何避免重复做代表元计算？</h2>
<p>仍看模 3 分组。若某个同态 \(f:\mathbb Z\to K\) 把所有 3 的倍数送到单位元，那么整数 1 与 4 必有同样的像，因为它们只差 3；于是 \(f\) 实际上只取决于“模 3 的类”。它可以先经过 \(\mathbb Z/3\mathbb Z\)，再送到 \(K\)。</p>
<p>一般地，若 \(N\subseteq\ker f\)，同一陪集内的 \(a,b\) 满足 \(a^{-1}b\in N\)，所以 \(f(a)=f(b)\)。于是唯一能定义的映射是 \(\bar f:G/N\to K\)、\(\bar f(gN)=f(g)\)，且 \(f=\bar f\circ\pi\)。这个“每个杀掉 \(N\) 的同态，都唯一地经过 \(G/N\)”的结论叫商的<strong>泛性质</strong>。</p>
<h2 id="section-33">理解检查</h2>
<ol><li>在 \(S_3\) 中令 \(H=\{e,(1\ 2)\}\)。找一个 \(g\) 使 \(gH\ne Hg\)，解释为什么不能按上式构造 \(S_3/H\)。</li><li>验证 \(\mathbb Z/6\mathbb Z\to\mathbb Z/3\mathbb Z\)、\([a]_6\mapsto[a]_3\) 的核，并写出它经过哪个商群。</li><li>证明 \(\bar f\) 是同态，以及为什么它只能有这一个定义。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 9, <em>Quotient Groups</em> 整理。</p>
</section>
