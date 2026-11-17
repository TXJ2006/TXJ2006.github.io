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
<p>在任意群中，\(g\sigma g^{-1}\) 称为 \(\sigma\) 被 \(g\) 共轭。在 \(S_n\) 中，先用 \(g^{-1}\) 找到旧标签，施行 \(\sigma\)，再用 \(g\) 换回新标签，所以</p>
<div class="equation">\[g(a_1\ a_2\cdots a_k)g^{-1}=(g(a_1)\ g(a_2)\cdots g(a_k)).\]</div>
<p>例如 \(\sigma=(1\ 2\ 3)\)、\(g=(1\ 4)\)，则 \(g\sigma g^{-1}=(4\ 2\ 3)\)。循环里哪些<strong>数字</strong>出现了会变，长度 3 不变。一般的乘积对每个不交循环逐一重命名。</p>

<h2 id="section-20">二、共轭类为什么由循环类型决定？</h2>
<p>“\(a\) 与 \(b\) 共轭”是一种等价关系：单位元给反身性，\(g^{-1}\) 给对称性，连续两次共轭等于用两个共轭元之积共轭，给传递性。其等价类称<strong>共轭类</strong>。在 \(S_n\) 中，重命名不改变各循环长度；反过来，两置换若循环长度多重集相同，就按对应循环的位置逐个指定一个标签双射 \(g\)，得到 \(g\sigma g^{-1}=\tau\)。所以 <strong>\(S_n\) 的共轭类恰由循环类型刻画</strong>。</p>
<p>在 \(S_3\) 中，三个类分别是 \(\{e\}\)、三个对换、两个 3-循环。它们的大小是 1、3、2，总数为 6。一般群中，固定 \(a\) 后使 \(gag^{-1}=a\) 的 \(g\) 恰是中心化子 \(C_G(a)\)；\(gag^{-1}=hah^{-1}\iff h^{-1}g\in C_G(a)\)，也就是 \(g,h\) 在同一个左陪集。故不同的左陪集与不同的共轭结果一一对应，有限时共轭类大小为 \([G:C_G(a)]\)。这把第 2 讲的中心化子和第 3 讲的陪集连起来。</p>

<h2 id="section-21">三、共轭与交换性的关系</h2>
<p>\(gag^{-1}=a\iff ga=ag\)。因此群交换当且仅当每个共轭类只有一个元素。不能把“某个元素的共轭类是单点”误认为整个群交换；那只说明这个元素位于中心 \(Z(G)\)。</p>
<h2 id="section-22">理解检查</h2>
<ol><li>在 \(S_4\) 中求 \((1\ 4)(1\ 2\ 3)(1\ 4)^{-1}\)。</li><li>列出 \(S_4\) 的所有循环类型，并据此数出共轭类的个数。</li><li>\(S_3\) 中一个对换的中心化子有几个元素？与其共轭类大小核对。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 6, <em>Conjugation in \(S_n\)</em> 整理。</p>
</section>
