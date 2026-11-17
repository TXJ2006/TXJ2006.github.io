---
title: 第 1 讲：群与对称｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-01/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-02/">下一讲 →</a></p>
<div class="note-lead"><p>群不是一类特殊的数字，而是“一组可复合、可撤销的动作”。正三角形的对称把定义里的每个条件都变成了可检查的操作。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-01.pdf">MIT 第 1 讲 PDF</a></nav>

<h2 id="section-1">一、先确定对象和运算</h2>
<p>把正三角形的顶点标为 \(A,B,C\)。逆时针旋转 \(120^\circ\) 记为 \(R\)，过 \(A\) 的轴反射记为 \(F\)。六个对称是 \(e,R,R^2,F,RF,R^2F\)。这里采用函数复合的顺序：\(RF\) 表示<strong>先做 \(F\)，后做 \(R\)</strong>。例如 \(R(A)=C,R(B)=A,R(C)=B\) 时，\(RF\) 和 \(FR\) 固定的顶点不同，因此 \(RF\ne FR\)。运算对象是“动作”，乘积是“接着做动作”，不是数值乘法。</p>
<figure class="note-figure"><svg viewBox="0 0 360 230" role="img" aria-labelledby="triangle-title triangle-desc"><title id="triangle-title">正三角形的旋转与反射</title><desc id="triangle-desc">三角形顶点 A 在上，B 在右下，C 在左下。虚线是固定 A、交换 B 和 C 的反射轴。</desc><polygon points="180,30 305,190 55,190" fill="none" stroke="#557f79" stroke-width="3"/><path d="M180 18V205" stroke="#bb6a4b" stroke-dasharray="6 5"/><text x="172" y="24">A</text><text x="312" y="195">B</text><text x="35" y="195">C</text><text x="190" y="95">F</text></svg><figcaption>虚线为 \(F\) 的反射轴，旋转 \(R\) 依次令 \(A\mapsto C\mapsto B\mapsto A\)。原讲义记这个六阶群为 \(D_3\)；本页的 \(D_n\) 指正 \(n\) 边形的 \(2n\) 个对称。</figcaption></figure>
<p>群 \((G,\cdot)\) 的定义要求：任意 \(g,h\in G\) 的积仍在 \(G\)；\((gh)k=g(hk)\)；存在 \(e\) 满足 \(eg=ge=g\)；每个 \(g\) 都有 \(g^{-1}\) 满足 \(gg^{-1}=g^{-1}g=e\)。结合律来自函数复合：两种加括号方式作用到任意顶点，结果都是“先第三个动作，再第二个，再第一个”。它<strong>不</strong>意味着 \(gh=hg\)；后者称交换律，成立的群才叫阿贝尔群。</p>
<p>这些六个对称的复合仍把三角形送回原处，所以满足封闭性；每个动作都能倒做，故有逆元。旋转的逆为 \(R^2\)，反射做两次恢复原样。这样，具体例子确实满足定义，而不只是“看起来有六个元素”。</p>

<h2 id="section-2">二、怎样判断一个例子是不是群？</h2>
<p>先写清运算。\((\mathbb Z,+)\) 的单位元是 \(0\)，\(a\) 的逆元是 \(-a\)；\((\mathbb Q\setminus\{0\},\times)\) 的单位元是 \(1\)，逆元是 \(1/a\)。\(\mathbb N\) 在加法下即使包含 \(0\)，正整数也没有加法逆元。所有 \(m\times n\) 矩阵在<strong>加法</strong>下成群；只有可逆方阵在<strong>乘法</strong>下成群，记 \(GL_n\)。说“矩阵构成群”而不说运算，会混淆这两个完全不同的结论。</p>
<p>定义还给出两个基本事实。若 \(e,e'\) 都是单位元，则 \(e=ee'=e'\)；若 \(b,c\) 都是 \(a\) 的逆元，则 \(b=be=b(ac)=(ba)c=ec=c\)。因此单位元和逆元是唯一的。由逆元还得消去律：\(ax=ay\) 左乘 \(a^{-1}\)，即得 \(x=y\)。这些小结论之后会用于陪集和同态的证明。</p>
<h2 id="section-3">理解检查</h2>
<ol><li>分别算 \(RF(A)\) 与 \(FR(A)\)，确认二者不能随意交换。</li><li>整数在乘法下为何不成群？把零删去是否已经足够？</li><li>证明 \((ab)^{-1}=b^{-1}a^{-1}\)，解释逆序出现的原因。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 1, <em>Groups</em> 整理。原讲义关于 \(D_n\) 的阶数与一个反射乘积有排版笔误；本页按实际对称操作核对。</p>
</section>
