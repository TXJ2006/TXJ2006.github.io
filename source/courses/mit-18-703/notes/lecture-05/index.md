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
<p>集合 \(S\) 的置换是双射 \(\sigma:S\to S\)。置换在函数复合下成群：恒等函数是单位元，反函数是逆元，复合满足结合律。当 \(S=\{1,\ldots,n\}\) 时记作 \(S_n\)，共有 \(n!\) 个元素，因为 \(1\) 的像有 \(n\) 种选择，随后依次有 \(n-1,\ldots,1\) 种。写 \(\sigma\tau\) 时始终<strong>先施行 \(\tau\)</strong>，因此一般 \(\sigma\tau\ne\tau\sigma\)。</p>
<p>两行记法记录每个输入的像：\(\left(\begin{smallmatrix}1&2&3&4&5\\3&4&1&5&2\end{smallmatrix}\right)\) 表示 \(1\mapsto3,2\mapsto4,\ldots\)。循环记法则追踪连续的像：\(1\mapsto3\mapsto1\) 是 \((1\ 3)\)，\(2\mapsto4\mapsto5\mapsto2\) 是 \((2\ 4\ 5)\)。故此置换为 \((1\ 3)(2\ 4\ 5)\)。不相交循环可交换，因为各自只改变自己的元素；有公共元素的循环通常不能交换。</p>
<p>例如右侧先作用时，\((1\ 2)(2\ 3)=(1\ 2\ 3)\)，但 \((2\ 3)(1\ 2)=(1\ 3\ 2)\)。计算第一个积时，\(1\mapsto1\mapsto2\)、\(2\mapsto3\mapsto3\)、\(3\mapsto2\mapsto1\)。只看这三个像，就能发现换乘法顺序改变了结果。</p>

<h2 id="section-16">二、为什么循环分解存在且唯一？</h2>
<p>从任意 \(a\) 出发不断取 \(\sigma(a),\sigma^2(a),\ldots\)。集合有限，必有重复；因 \(\sigma\) 单射，第一次回到的只能是起点，否则两个不同前驱会有同一个像。这得到一个循环。再从未出现的元素出发，直到所有元素被覆盖。每个元素的轨道已由 \(\sigma\) 决定，所以分解除循环起点、循环书写顺序和省略的 1-循环外唯一。</p>
<p>若不交循环的长度为 \(k_1,\ldots,k_r\)，则 \(\sigma^t=e\) 当且仅当每个 \(k_i\mid t\)，因此 \(\operatorname{ord}(\sigma)=\operatorname{lcm}(k_1,\ldots,k_r)\)。上例的阶是 \(\operatorname{lcm}(2,3)=6\)，不是 \(2+3=5\)。</p>

<h2 id="section-17">三、为什么对换足以生成整个 \(S_n\)？</h2>
<p>对换 \((a\ b)\) 交换两个元素。采用右侧先作用的约定，\((a_1\ a_2\cdots a_k)=(a_1\ a_k)(a_1\ a_{k-1})\cdots(a_1\ a_2)\)。检查右侧对 \(a_1\) 的作用：最右因子先把它送到 \(a_2\)；对 \(a_j\) 则经 \((a_1\ a_j)\) 到 \(a_1\)，再经 \((a_1\ a_{j+1})\) 到 \(a_{j+1}\)。每个循环可如此分解，所以每个置换都是对换的积。对换的<strong>个数</strong>不唯一；其奇偶性到第 11 讲才证明不变。</p>
<h2 id="section-18">理解检查</h2>
<ol><li>求 \((1\ 2\ 3)(2\ 4)\) 的两行表示，注意右侧先作用。</li><li>置换 \((1\ 4\ 2)(3\ 5)(6\ 7\ 8\ 9)\) 的阶是多少？</li><li>把 \((1\ 2\ 3\ 4)\) 写成三个对换并逐点核查。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 5, <em>Permutation Groups</em> 整理。</p>
</section>
