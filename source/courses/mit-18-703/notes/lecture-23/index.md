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
<p>三角形的六个对称作用在顶点集合 \(S=\{A,B,C\}\) 上。固定 \(A\)：它可以被送到三个顶点，所以 \(A\) 的<strong>轨道</strong>是整个 \(S\)；保持 \(A\) 不动的只有 \(e\) 与过 \(A\) 的反射，所以 \(A\) 的<strong>稳定子</strong>有 2 个元素。这里 \(6=3\cdot2\)，不是偶然的乘法巧合。</p>
<p>一般地，群作用是映射 \(G\times S\to S\)，满足 \(e\cdot s=s\)、\((gh)\cdot s=g\cdot(h\cdot s)\)。它等价于同态 \(\rho:G\to\operatorname{Sym}(S)\)。点 \(s\) 的轨道为 \(G\cdot s=\{g\cdot s:g\in G\}\)，稳定子为 \(G_s=\{g:g\cdot s=s\}\)。轨道回答“能到哪里”，稳定子回答“哪些动作没改变它”。</p>
<p>为什么它们的大小相乘等于群的阶？\(g\) 与 \(h\) 把 \(s\) 送到同一点，恰当且仅当 \(h^{-1}g\in G_s\)，即 \(g,h\) 属于同一个左陪集。因此 \(gG_s\mapsto g\cdot s\) 是左陪集<strong>集合</strong>与轨道的双射，有限时 \(|G\cdot s|=[G:G_s]\)。若作用传递，轨道就是全部 \(S\)，于是</p>
<div class="equation">\[|G|=|G_s|\,|S|.\]</div>
<p>稳定子一般不正规，\(G/G_s\) 在这里仅表示左陪集集合，不应误认为商群。上面的三角形例子给出 \(6=2\cdot3\)。</p>

<h2 id="section-87">二、共轭作用与类方程</h2>
<p>令 \(G\) 以共轭作用于自身：\(g\cdot x=gxg^{-1}\)。轨道就是共轭类，稳定子是中心化子 \(C_G(x)\)，所以共轭类大小为 \([G:C_G(x)]\)。大小为 1 的共轭类恰对应中心元素；把其余类的大小相加，得到<strong>类方程</strong>：</p>
<div class="equation">\[|G|=|Z(G)|+\sum_{[x]:\,|[x]|>1}|[x]|.\]</div>
<p>若 \(G\) 是有限 \(p\)-群，每个非中心共轭类大小被 \(p\) 整除，因此 \(|Z(G)|\equiv|G|\equiv0\pmod p\)，中心非平凡。第 13 讲已使用这一计数；这里终于看清它来自轨道—稳定子公式。</p>

<h2 id="section-88">三、Sylow 存在性与自同构</h2>
<p>同一套“作用—轨道—稳定子”也能证明 Sylow 子群存在。令 \(|G|=p^km\) 且 \(p\nmid m\)，考虑 \(G\) 的所有 \(p^k\) 元子集，让 \(G\) 以左乘作用。一个组合数计算表明，这些子集的总数不被 \(p\) 整除；故至少有一条轨道的大小不被 \(p\) 整除。</p>
<p>在这条轨道中取子集 \(X\)，先左移它使 \(e\in X\)，记稳定子为 \(H\)。若 \(h\in H\)，则 \(h=he\in hX=X\)，所以 \(H\subseteq X\)，\(|H|\le p^k\)。另一方面，轨道大小 \([G:H]\) 不被 \(p\) 整除，故 \(|H|\) 含有 \(|G|\) 中的整个因子 \(p^k\)；只能 \(|H|=p^k\)。因此 \(H=X\) 是一个 Sylow \(p\)-子群。</p>
<p>还有一类作用：让群元素 \(a\) 通过共轭改变整个群，\(\sigma_a(x)=axa^{-1}\)。它是从 \(G\) 到自身的同构，称<strong>内自同构</strong>。不同的 \(a\) 可能给同一个自同构；\(\sigma_a\) 是恒等映射，恰当且仅当 \(a\) 与每个 \(x\) 交换，即 \(a\in Z(G)\)。所以同态 \(G\to\operatorname{Aut}(G)\) 的核是中心，第一同构定理给</p>
<div class="equation">\[\operatorname{Inn}(G)\cong G/Z(G).\]</div>
<p>\(\operatorname{Inn}(G)\) 在全自同构群中正规：任一自同构 \(\phi\) 满足 \(\phi\sigma_a\phi^{-1}=\sigma_{\phi(a)}\)，把内自同构共轭后仍留在内自同构里。</p>
<p>对 \(S_n\)，当 \(n\ne6\) 时所有自同构都是内自同构；\(S_6\) 有一个著名的外自同构。其特殊性来自：对换与三个不交对换的共轭类大小在 \(S_6\) 中相同，可以被外自同构交换。</p>
<h2 id="section-89">理解检查</h2>
<ol><li>证明 \(G\) 作用在左陪集集合上时，单位陪集的稳定子就是原子群。</li><li>从类方程重新证明有限 \(p\)-群的中心非平凡。</li><li>为什么 \(G/Z(G)\) 而不是 \(G\) 本身是内自同构群？</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 23, <em>Group Actions and Automorphisms</em> 整理。</p>
</section>
