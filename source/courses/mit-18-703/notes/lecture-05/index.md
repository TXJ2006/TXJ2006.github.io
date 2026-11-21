---
title: 第 5 讲：置换与循环分解｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-05/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-04/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-06/">下一讲 →</a></p>
<div class="note-lead"><p>三角形对称是顶点的重新排列。把“重新排列”抽出来，就得到置换群；循环分解则把一个复杂置换拆成互不干扰的轨道。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-05.pdf">MIT 第 5 讲 PDF</a></nav>

<h2 id="section-15">一、置换是双射，不只是一个排列字符串</h2>
<p>第 1 讲的三角形对称会重新安排三个顶点。若暂时不管三角形，只研究有限集合 \(S\) 的<strong>重新安排</strong>，就是置换：一个双射 \(\sigma:S\to S\)。它必须是双射，因为每个旧位置要有唯一去处，每个新位置也要恰好被占一次。</p>
<p>令 \(S=\{1,2,3,4,5\}\)。如果 \(1\mapsto3,2\mapsto4,3\mapsto1,4\mapsto5,5\mapsto2\)，可用两行记法压缩为 \(\left(\begin{smallmatrix}1&2&3&4&5\\3&4&1&5&2\end{smallmatrix}\right)\)。沿着箭头追踪：\(1\to3\to1\)，以及 \(2\to4\to5\to2\)。因此更紧凑的循环记法是 \((1\ 3)(2\ 4\ 5)\)。两个循环没有共同数字，各管各的点，所以可以交换顺序。</p>
<p>所有 \(n\) 个点的置换组成 \(S_n\)：复合两个双射仍是双射，恒等映射是单位元，反向安排是逆元。共有 \(n!\) 个，因为第一个点有 \(n\) 个去处、第二个剩 \(n-1\) 个，依此类推。复合 \(\sigma\tau\) 仍按<strong>右边先做</strong>的约定读。</p>
<p>不能随意把有公共数字的循环换顺序。例如 \((1\ 2)(2\ 3)\) 先做 \((2\ 3)\)：\(1\to1\to2\)、\(2\to3\to3\)、\(3\to2\to1\)，结果是 \((1\ 2\ 3)\)。反过来 \((2\ 3)(1\ 2)=(1\ 3\ 2)\)。先逐点追踪，比分辨符号顺序可靠。</p>

<h2 id="section-16">二、为什么循环分解存在且唯一？</h2>
<p>上例分成两个循环不是碰巧。对任意起点 \(a\)，不断看 \(a,\sigma(a),\sigma^2(a),\ldots\)。由于 \(S\) 有限，某个点必重复；而 \(\sigma\) 是单射，第一次重复只能回到 \(a\)，否则会有两个不同前驱映到同一点。这样得到一个循环，再从未访问的点重新开始，最终覆盖 \(S\)。</p>
<p>循环中的点及它们的先后顺序由 \(\sigma\) 决定，所以分解除“从循环中哪个点开始写”、不相交循环的排列顺序以及是否写出固定点外，是唯一的。</p>
<p>若不交循环的长度为 \(k_1,\ldots,k_r\)，则 \(\sigma^t=e\) 当且仅当每个 \(k_i\mid t\)，因此 \(\operatorname{ord}(\sigma)=\operatorname{lcm}(k_1,\ldots,k_r)\)。上例的阶是 \(\operatorname{lcm}(2,3)=6\)，不是 \(2+3=5\)。</p>

<h2 id="section-17">三、为什么对换足以生成整个 \(S_n\)？</h2>
<p>只交换两个点的置换叫<strong>对换</strong>，记作 \((a\ b)\)。例如 \((1\ 2\ 3\ 4)=(1\ 4)(1\ 3)(1\ 2)\)：从右向左看，\(1\) 先到 \(2\)，而 \(2\) 先被最右因子送到 \(1\)，下一步到 \(3\)。逐点检查可得到整个四循环。</p>
<p>一般的 \(k\)-循环也能写成 \((a_1\ a_k)(a_1\ a_{k-1})\cdots(a_1\ a_2)\)。任意置换先拆成不交循环，再把每个循环拆成对换，故对换生成 \(S_n\)。拆法并不唯一；但对换个数的<strong>奇偶性</strong>不变，这件事需要第 11 讲单独证明，不能从一个拆法直接宣布。</p>
<h2 id="section-18">理解检查</h2>
<ol><li>求 \((1\ 2\ 3)(2\ 4)\) 的两行表示，注意右侧先作用。</li><li>置换 \((1\ 4\ 2)(3\ 5)(6\ 7\ 8\ 9)\) 的阶是多少？</li><li>把 \((1\ 2\ 3\ 4)\) 写成三个对换并逐点核查。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 5, <em>Permutation Groups</em> 整理。</p>
</section>
