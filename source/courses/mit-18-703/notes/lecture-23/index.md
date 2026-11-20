---
title: 第 23 讲：群作用与自同构｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-23/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-22/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-24/">下一讲 →</a></p>
<div class="note-lead"><p>群作用把抽象群变成对集合的具体置换。轨道—稳定子公式、共轭类方程和 Sylow 定理都来自同一个事实：群作用等价于到置换群的同态。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-23.pdf">MIT 第 23 讲 PDF</a></nav>

<h2 id="section-86">一、作用、轨道与稳定子</h2>
<p>作用是映射 \(G\times S\to S\)，满足 \(e\cdot s=s\)、\((gh)\cdot s=g\cdot(h\cdot s)\)。它等价于表示 \(\rho:G\to\operatorname{Sym}(S)\)。点 \(s\) 的轨道是 \(G\cdot s=\{g\cdot s:g\in G\}\)，稳定子是 \(G_s=\{g:g\cdot s=s\}\)。轨道是等价类；若作用传递，则整个集合只有一个轨道。</p>
<p>当作用传递时，映射 \(gG_s\mapsto g\cdot s\) 给出 \(G/G_s\cong S\)，有限情形得到
\[|G|=|G_s|\,|S|.\]
这不是一个孤立的计数公式：它说明一个点的稳定子完全记录了该点在群作用中的重复对称。</p>

<h2 id="section-87">二、共轭作用与类方程</h2>
<p>令 \(G\) 以共轭作用于自身：\(g\cdot x=gxg^{-1}\)。轨道就是共轭类，稳定子是中心化子 \(C_G(x)\)，所以共轭类大小为 \([G:C_G(x)]\)。把所有轨道相加得到类方程
\[|G|=|Z(G)|+\sum_{[x]:\,|[x]|>1}|[x]|.\]
若 \(G\) 是有限 \(p\)-群，每个非中心共轭类大小被 \(p\) 整除，因此 \(|Z(G)|\equiv|G|\equiv0\pmod p\)。中心非平凡；这给出上一部分 Sylow 理论的重要应用。</p>

<h2 id="section-88">三、Sylow 存在性与自同构</h2>
<p>为证明 Sylow \(p\)-子群存在，令 \(|G|=p^km\)，考虑所有大小为 \(p^k\) 的子集，群通过左平移作用。子集总数不被 \(p\) 整除，所以有一个轨道大小不被 \(p\) 整除。把该轨道中的子集平移到包含单位元的位置，其稳定子既包含在这个子集中，又因轨道—稳定子公式具有足够大的阶，二者只能相等；这个稳定子就是 Sylow \(p\)-子群。</p>
<p>群对自身的共轭给出内自同构 \(\sigma_a(x)=axa^{-1}\)。映射 \(G\to\operatorname{Aut}(G)\) 的核是中心，因此
\[\operatorname{Inn}(G)\cong G/Z(G).\]
\(\operatorname{Inn}(G)\) 在全自同构群中正规，共轭关系由 \(\phi^{-1}\sigma_a\phi=\sigma_{\phi(a)}\) 直接得到。</p>
<p>对 \(S_n\)，当 \(n\ne6\) 时所有自同构都是内自同构；\(S_6\) 有一个著名的外自同构。其特殊性来自：对换与三个不交对换的共轭类大小在 \(S_6\) 中相同，可以被外自同构交换。</p>
<h2 id="section-89">理解检查</h2>
<ol><li>证明 \(G\) 作用在左陪集集合上时，单位陪集的稳定子就是原子群。</li><li>从类方程重新证明有限 \(p\)-群的中心非平凡。</li><li>为什么 \(G/Z(G)\) 而不是 \(G\) 本身是内自同构群？</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 23, <em>Group Actions and Automorphisms</em> 整理。</p>
</section>
