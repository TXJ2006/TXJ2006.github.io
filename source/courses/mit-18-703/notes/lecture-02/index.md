---
title: 第 2 讲：子群｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-02/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-01/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-03/">下一讲 →</a></p>
<div class="note-lead"><p>上一讲研究整组动作。本讲研究其中哪些动作能单独构成一组完整的“可复合、可撤销”动作，这就是子群。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-02.pdf">MIT 第 2 讲 PDF</a></nav>

<h2 id="section-4">一、子群要继承什么？</h2>
<p>先回到三角形。六个对称中只挑 \(H=\{e,R,R^2\}\)：任意两个旋转接着做仍是旋转，\(R\) 可由 \(R^2\) 撤销，\(e\) 也在其中。因此这三个动作单独就构成群。这样的“群里的一小群”，叫<strong>子群</strong>。</p>
<p>严格地说，\(H\subseteq G\) 是子群，表示 \(H\) 沿用 \(G\) 的<strong>同一个运算</strong>，自己仍是群。实际检查时只需三步：\(H\) 非空；\(a,b\in H\Rightarrow ab\in H\)；\(a\in H\Rightarrow a^{-1}\in H\)。结合律从 \(G\) 继承，任选 \(a\in H\) 就有 \(e=aa^{-1}\in H\)，所以不必另找一个单位元。</p>
<p>在整数加法群中，偶数 \(2\mathbb Z\) 也满足这三步；奇数不满足加法封闭（奇数加奇数变偶数），非负整数不满足取逆封闭（\(1\) 的逆是 \(-1\)）。判别法也可压成一句：\(H\ne\varnothing\)，且任意 \(a,b\in H\) 有 \(ab^{-1}\in H\)。先令 \(a=b\) 得到单位元，再令 \(a=e\) 得到逆元。</p>

<h2 id="section-5">二、把判别法用于非显然的集合</h2>
<p>有时子集不是直接列出来的，而由一个条件筛选。固定 \(g\in G\)，只取与 \(g\) 交换的元素，就得到<strong>中心化子</strong> \(C_G(g)=\{h\in G:hg=gh\}\)。在 \(D_3\) 中，\(C_{D_3}(R)=\{e,R,R^2\}\)：旋转彼此交换，反射却把 \(R\) 共轭成 \(R^{-1}\ne R\)。</p>
<p>为什么中心化子总是子群？若 \(h,k\) 都与 \(g\) 交换，先把 \(g\) 移过 \(k\)，再移过 \(h\)，就有 \((hk)g=g(hk)\)。由 \(hg=gh\) 左右分别乘 \(h^{-1}\)，得到 \(gh^{-1}=h^{-1}g\)。所以乘积和逆元仍满足筛选条件。</p>
<p>有限集合还有一个方便之处：若 \(H\) 是<strong>非空有限</strong>的乘法封闭子集，则任取 \(a\in H\)，幂 \(a,a^2,\ldots\) 必重复。设 \(a^r=a^s\) 且 \(r\lt s\)，在 \(G\) 中消去 \(a^r\) 得 \(a^{s-r}=e\)；因此 \(a^{-1}=a^{s-r-1}\in H\)。无限情形不行：非负整数对加法封闭，却缺少 \(-1\)。</p>

<h2 id="section-6">三、三角形的子群为什么只有这些？</h2>
<p>从 \(D_3\) 中任取一个动作，子群条件会强制补齐其他动作：若含 \(R\)，就得含 \(R^2\) 和 \(e\)，得到旋转子群；若只含一个反射 \(F_i\)，则须含 \(e\)，得到 \(\{e,F_i\}\)。若同时含两个不同反射，它们的积是非平凡旋转，于是所有旋转和反射都被迫出现，得到整个 \(D_3\)。</p>
<p>所以全部子群是 \(\{e\}\)、\(\{e,R,R^2\}\)、三个 \(\{e,F_i\}\)，以及 \(D_3\) 本身。找子群的实用方法不是猜名单，而是取一个元素后不断补齐它的积和逆元。</p>
<h2 id="section-7">理解检查</h2>
<ol><li>用 \(ab^{-1}\) 判别法直接证明 \(n\mathbb Z\leq(\mathbb Z,+)\)。</li><li>在 \(D_3\) 中，旋转 \(R\) 的中心化子有哪些元素？</li><li>指出“非空且闭合于乘法”在无限集合中失败的具体环节。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 2, <em>Subgroups</em> 整理。</p>
</section>
