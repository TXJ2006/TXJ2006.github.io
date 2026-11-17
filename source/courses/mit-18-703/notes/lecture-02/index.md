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
<p>子集 \(H\subseteq G\) 是子群，意指用 \(G\) 的<strong>同一个运算</strong>限制到 \(H\)，\(H\) 自己仍是群。非空子集的判别法是：任意 \(a,b\in H\) 有 \(ab\in H\)，且 \(a^{-1}\in H\)。结合律已经由 \(G\) 保证；任选 \(a\in H\)，便有 \(e=aa^{-1}\in H\)。因此单位元不能另造一个，逆元也正是它在 \(G\) 中的逆元。等价的紧凑判别法是非空且 \(ab^{-1}\in H\)。</p>
<p>在 \((\mathbb Z,+)\) 中，\(2\mathbb Z\) 是子群；奇数集合不是，因为两个奇数之和为偶数；非负整数也不是，因为 \(1\) 的逆元 \(-1\) 不在其中。正 \(n\) 边形的旋转构成 \(D_n\) 的子群。这里“闭合”必须同时检查运算和逆运算，不能仅因一批元素看起来有规律就称它为子群。</p>

<h2 id="section-5">二、把判别法用于非显然的集合</h2>
<p>固定 \(g\in G\)，它的<strong>中心化子</strong>是 \(C_G(g)=\{h\in G: hg=gh\}\)。若 \(h,k\) 都与 \(g\) 交换，则 \((hk)g=h(kg)=h(gk)=(hg)k=g(hk)\)；从 \(hg=gh\) 两侧乘 \(h^{-1}\) 得 \(h^{-1}g=gh^{-1}\)。所以 \(C_G(g)\) 是子群。它衡量的是“哪些元素和这个指定元素交换”，不要求整个 \(G\) 交换。</p>
<p>如果 \(H\) 是 \(G\) 的<strong>非空有限子集</strong>，只检查乘法闭合也够了。对 \(a\in H\)，序列 \(a,a^2,a^3,\ldots\) 必重复；若 \(a^r=a^s\) 且 \(r\lt s\)，消去得到 \(a^{s-r}=e\)，于是 \(a^{-1}=a^{s-r-1}\in H\)（指数为零时就是 \(e\)）。无限情形不能这样省略逆元：\(\{0,1,2,\ldots\}\subset\mathbb Z\) 对加法闭合，却不是子群。</p>

<h2 id="section-6">三、三角形的子群为什么只有这些？</h2>
<p>在 \(D_3\) 中有平凡子群 \(\{e\}\)、旋转子群 \(\{e,R,R^2\}\)、三个二元反射子群 \(\{e,F_i\}\)，以及整个 \(D_3\)。若子群含一个非平凡旋转，就被迫含全部旋转；再含一个反射，复合便产生其余反射。若它含两个不同反射，二者之积是非平凡旋转，也就得到全群。这个“取元素后不断强制补齐积和逆”的思路，是找子群的实用方法。</p>
<p>中心化子的例子在这里也可算到底：\(C_{D_3}(R)=\{e,R,R^2\}\)。旋转之间都交换，但任一反射 \(F\) 满足 \(FRF^{-1}=R^{-1}\ne R\)，所以反射不在 \(C_{D_3}(R)\)。</p>
<h2 id="section-7">理解检查</h2>
<ol><li>用 \(ab^{-1}\) 判别法直接证明 \(n\mathbb Z\leq(\mathbb Z,+)\)。</li><li>在 \(D_3\) 中，旋转 \(R\) 的中心化子有哪些元素？</li><li>指出“非空且闭合于乘法”在无限集合中失败的具体环节。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 2, <em>Subgroups</em> 整理。</p>
</section>
