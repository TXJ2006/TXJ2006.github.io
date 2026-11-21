---
title: 第 6 讲：对称群中的共轭｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-06/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-05/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-07/">下一讲 →</a></p>
<div class="note-lead"><p>共轭 \(g\sigma g^{-1}\) 看似多绕了一圈，实际是在换标签后描述同一循环结构。因此它既测出群的非交换性，也把置换按“形状”分类。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-06.pdf">MIT 第 6 讲 PDF</a></nav>

<h2 id="section-19">一、共轭为什么是重命名？</h2>
<p>设 \(\sigma=(1\ 2\ 3)\)，再让 \(g=(1\ 4)\) 把标签 1 与 4 互换。若同一个循环按新标签描述，就应写作 \((4\ 2\ 3)\)：沿循环走动的方式没变，只是 1 现在叫 4。这正是 \(g\sigma g^{-1}\) 的效果。</p>
<p>为什么要在两侧各放一个 \(g\)？计算新标签 \(g(a)\) 的去处时，先用 \(g^{-1}\) 找回旧标签 \(a\)，让 \(\sigma\) 把它送到下一个旧标签，再用 \(g\) 翻译回来。因此</p>
<div class="equation">\[g(a_1\ a_2\cdots a_k)g^{-1}=(g(a_1)\ g(a_2)\cdots g(a_k)).\]</div>
<p>这个运算叫<strong>共轭</strong>。在 \(S_n\) 中，它改变循环里写的数字，却不改变循环长度；对不交循环的乘积，只需逐个换标签。第 1 讲反射把旋转 \(R\) 变成 \(R^{-1}\)，也是一个共轭例子。</p>

<h2 id="section-20">二、共轭类为什么由循环类型决定？</h2>
<p>把能互相换标签得到的置换放成一组，这组叫<strong>共轭类</strong>。例如 \(S_3\) 正好分成三组：\(\{e\}\)、三个对换、两个 3-循环，数量 \(1+3+2=6\)。前一讲的循环分解让我们看出分组规则：在 \(S_n\) 中，两置换共轭，当且仅当不交循环的长度清单相同。反向的理由是可以逐个循环、逐个位置把一套标签对应到另一套。</p>
<p>一般群中，共轭类有多大？固定 \(a\)，让 \(g\) 变化，得到 \(gag^{-1}\)。使结果仍为 \(a\) 的元素满足 \(ga=ag\)，恰组成第 2 讲的中心化子 \(C_G(a)\)。两个 \(g,h\) 产生同一结果，当且仅当 \(h^{-1}g\in C_G(a)\)，即它们在同一个左陪集。于是有限群中，共轭类大小为 \([G:C_G(a)]\)：这把中心化子和陪集的计数真正接起来。</p>

<h2 id="section-21">三、共轭与交换性的关系</h2>
<p>\(gag^{-1}=a\iff ga=ag\)。因此群交换当且仅当每个共轭类只有一个元素。不能把“某个元素的共轭类是单点”误认为整个群交换；那只说明这个元素位于中心 \(Z(G)\)。</p>
<h2 id="section-22">理解检查</h2>
<ol><li>在 \(S_4\) 中求 \((1\ 4)(1\ 2\ 3)(1\ 4)^{-1}\)。</li><li>列出 \(S_4\) 的所有循环类型，并据此数出共轭类的个数。</li><li>\(S_3\) 中一个对换的中心化子有几个元素？与其共轭类大小核对。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 6, <em>Conjugation in \(S_n\)</em> 整理。</p>
</section>
