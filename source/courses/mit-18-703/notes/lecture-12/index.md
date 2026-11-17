---
title: 第 12 讲：群的呈示与小阶群｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-12/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-11/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-13/">下一讲 →</a></p>
<div class="note-lead"><p>只列元素不利于描述大群。呈示用“生成元能做什么”和“哪些复合等于单位元”压缩乘法信息；小阶群分类则检验这些关系究竟能强制出多少结构。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-12.pdf">MIT 第 12 讲 PDF</a></nav>

<h2 id="section-42">一、从自由群到关系</h2>
<p>给定字母集 \(S\)，用字母及其形式逆元写有限长词，约去相邻的 \(aa^{-1}\) 或 \(a^{-1}a\)。约化词在“连接后再约化”下构成自由群 \(F(S)\)。它的关键性质是：任意赋值 \(S\to G\) 都唯一延伸为同态 \(F(S)\to G\)，因为一个词的像只能按同样顺序把各字母的像相乘。若 \(S\) 生成 \(G\)，延伸同态满射；它的核收集所有“在 \(G\) 中等于单位元”的词。于是 \(G\cong F(S)/\ker\phi\)。</p>
<p>写 \(G=\langle S\mid R\rangle\) 时，关系集合 \(R\) 要生成的是<strong>正规闭包</strong>：包含 \(R\) 的最小正规子群，而不只是普通生成子群。这个区别保证取商有定义。循环群 \(C_n=\langle a\mid a^n=e\rangle\)；正方形对称群 \(D_4=\langle r,f\mid r^4=f^2=e,\ frf^{-1}=r^{-1}\rangle\)。关系 \(fr=r^{-1}f\) 让每个词归约为 \(r^if^j\)（\(0\leq i\lt4,\ 0\leq j\lt2\)），所以至多 8 个元素；实际正方形的 8 个不同对称满足这些关系，故恰为 8 个。</p>

<h2 id="section-43">二、阿贝尔群为什么能分类？</h2>
<p>若再强制生成元彼此交换，自由群变成自由阿贝尔群；有限个生成元时它是 \(\mathbb Z^d\)。有限生成阿贝尔群都是某个 \(\mathbb Z^d\) 的商。结构定理进一步给唯一的分解类型 \(\mathbb Z^r\times\mathbb Z_{m_1}\times\cdots\times\mathbb Z_{m_t}\)，其中 \(m_1\mid m_2\mid\cdots\mid m_t\) 且 \(m_i>1\)。也可按素数幂分解有限部分；两种写法必须在相应排序规则下理解“唯一”。</p>
<p>例如阶为 60 的阿贝尔群只有 \(\mathbb Z_{60}\) 与 \(\mathbb Z_2\times\mathbb Z_{30}\) 两型：\(60=2^2\cdot3\cdot5\)，只有 2-部分可选 \(\mathbb Z_4\) 或 \(\mathbb Z_2\times\mathbb Z_2\)。这里不能仅凭阶数把两者认成同一群，前者有阶为 4 的元素，后者没有。</p>

<h2 id="section-44">三、小阶非交换群怎样被迫出现？</h2>
<p>阶为素数的群必循环。阶为 4 的群只有 \(C_4\) 和 \(C_2\times C_2\)；阶为 6 的群是 \(C_6\) 或 \(S_3\)。关键是指数 2 的子群自动正规：左右陪集都只有 \(H\) 与它的补集。阶为 8 的非交换群必有四阶元素 \(r\)：否则每个非单位元素都满足 \(x^2=e\)，由 \((xy)^2=e\) 可推出 \(xy=yx\)。于是 \(\langle r\rangle\) 指数为 2。外面的元素 \(f\) 通过共轭给 \(\langle r\rangle\cong C_4\) 的自同构；非交换情形必为 \(frf^{-1}=r^{-1}\)。又有 \(f^2\in\langle r\rangle\)，且 \(ff^2f^{-1}=f^2\)，所以 \(f^2\) 必是反演自同构的不动点，只能为 \(e\) 或 \(r^2\)。前者给 \(D_4\)，后者给四元数群 \(Q_8\)。两者不同构：\(D_4\) 的非单位二阶元素有五个，\(Q_8\) 只有 \(-1\) 一个。原讲义主要给分类思路，结构定理与一般呈示的判定问题并未在这一讲证明。</p>
<h2 id="section-45">理解检查</h2>
<ol><li>仅用三个关系把 \(D_4\) 的任意词归约为 \(r^if^j\)，说明为什么还需一个真实模型证明“至多 8”是“恰好 8”。</li><li>列出阶为 8 的全部阿贝尔群，并用元素阶区分。</li><li>为什么在取商 \(F(S)/N\) 时必须要求 \(N\) 正规？</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 12, <em>Presentations and Groups of Small Order</em> 整理。</p>
</section>
