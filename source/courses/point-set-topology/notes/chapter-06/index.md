---
title: 第 6 篇：紧性与有限子覆盖｜阅读笔记
layout: page
permalink: /courses/point-set-topology/notes/chapter-06/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/point-set-topology/">← 课程目录</a> · <a href="/courses/point-set-topology/notes/chapter-05/">上一篇</a> · <a href="/courses/point-set-topology/notes/chapter-07/">下一篇 →</a></p>
<div class="note-lead"><p>紧性把“可能需要无限多个局部信息”压缩为“有限多个已经够用”。这个定义不需要距离；欧氏空间里熟悉的“闭且有界”只是 Heine–Borel 定理给出的特殊判别。</p></div>
<nav class="reading-links" aria-label="参考讲义"><a href="/files/math-resources/archive-2026-11-08-59.pdf">林剑峰讲义：第 4 章</a><a href="/files/math-resources/archive-2026-11-08-34.pdf">Körner 讲义：9–10、12</a></nav>

<h2 id="section-33">一、定义从开覆盖开始</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/06-open-cover.svg" alt="黑白球面上三块相互重叠的开片构成有限子覆盖" loading="lazy"><figcaption>示意图中的三块开片已经盖住整个球面；紧性的要求是对<strong>任意</strong>开覆盖都能选出某个有限子覆盖。</figcaption></figure>
<p>一族开集 \(\{U_i\}_{i\in I}\) 若满足 \(X=\bigcup_{i\in I}U_i\)，称为 \(X\) 的<strong>开覆盖</strong>。空间 \(X\) <strong>紧</strong>，若每个开覆盖都有有限子覆盖，即能选出 \(i_1,\ldots,i_m\) 仍有 \(X=U_{i_1}\cup\cdots\cup U_{i_m}\)。关键是“每个”覆盖，而不是某个覆盖。</p>
<p>\((0,1)\) 不紧：开集 \(U_n=(1/n,1)\)（\(n\ge2\)）覆盖它，但有限个这样的集合只覆盖 \((1/N,1)\)，留下靠近 0 的点。\([0,1]\) 紧则需要证明，不是由“端点都在”直接推出；在 \(\mathbb R\) 中可用 Heine–Borel 定理。</p>

<h2 id="section-34">二、闭集的有限交性质</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/06-finite-intersection.svg" alt="三个逐层嵌套的闭区域保留一个共同点 x" loading="lazy"><figcaption>这里闭区域逐层缩小仍留有公共点；紧性把“每个有限交非空”提升为“全部交非空”。</figcaption></figure>
<p>对闭集族 \(\{F_i\}\)，称它有<strong>有限交性质</strong>，若任意有限多个成员的交非空。取补集并用德摩根律，就得到等价命题：\(X\) 紧，当且仅当每族具有有限交性质的闭集，其全部成员的交仍非空。</p>
<p>这个版本适合构造“同时满足所有条件”的点：若每次只施加有限多个闭条件时都有解，紧性保证所有闭条件一起也有解。注意每个 \(F_i\) 必须在同一个紧空间里闭。</p>

<h2 id="section-35">三、紧性在映射和子空间下如何传递？</h2>
<p>连续像保紧。若 \(f:X\to Y\) 连续且 \(X\) 紧，取 \(f(X)\) 的开覆盖，把开集拉回 \(X\) 得开覆盖；选有限子覆盖再映回去即可。紧空间的闭子集也紧：把闭子集 \(F\) 的一个相对开覆盖与 \(X\setminus F\) 合并成 \(X\) 的开覆盖。</p>
<p>若 \(Y\) Hausdorff，则 \(Y\) 中的紧子集闭。对 \(y\notin K\)，分别把 \(y\) 与每个 \(x\in K\) 用不交开邻域分开；覆盖 \(K\) 的那侧只需有限多个，从而在 \(y\) 侧取有限交，得到不碰 \(K\) 的开邻域。这一步说明 Hausdorff 假设在哪里用到。</p>
<p>于是连续双射 \(f:X\to Y\) 若 \(X\) 紧、\(Y\) Hausdorff，便是同胚：\(X\) 的任意闭集紧，其像紧且在 \(Y\) 中闭，所以 \(f\) 是闭映射，逆映射连续。第 3 篇“端点粘合成圆”的严格证明正用这条定理。</p>

<h2 id="section-36">四、有限积与任意积</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/06-compact-product.svg" alt="透视长方体的有限个坐标切片对应积空间中的覆盖" loading="lazy"><figcaption>有限积证明先固定一个 \(X\) 坐标，用 \(Y\) 的紧性覆盖整根“管”，再用 \(X\) 的紧性选有限根管。</figcaption></figure>
<p>两个紧空间 \(X,Y\) 的积是紧的。证明时先固定 \(x\in X\)：覆盖 \(\{x\}\times Y\) 的有限多个积邻域可压缩成 \(x\) 的一个开邻域 \(U_x\)，使 \(U_x\times Y\) 被原覆盖有限个成员覆盖。再用 \(X\) 的紧性，从 \(\{U_x\}\) 选有限个，就得到整个积的有限子覆盖。</p>
<p><strong>Tychonoff 定理</strong>将结论推广到任意多个紧空间的<strong>积拓扑</strong>。可用超滤子理解证明：紧性等价于每个超滤子至少有一个极限；积上的超滤子经每个坐标投影，分别在紧的 \(X_i\) 中取极限 \(x_i\)。积拓扑的基本开集只约束有限个坐标，因而各坐标的有限个“最终条件”可以同时满足，原超滤子收敛到 \((x_i)\)。无限积证明通常使用选择公理；方框拓扑一般没有这条结论。</p>

<h2 id="section-37">五、度量空间中如何识别紧性？</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/06-total-bounded.svg" alt="环境平面中的线段 X 由有限个同半径的小球覆盖" loading="lazy"><figcaption>图中黑线段是被覆盖的 \(X\)，透视底面只是环境；全有界要求对<strong>每个</strong>半径 \(\varepsilon>0\) 都能重新找有限个球覆盖 \(X\)。</figcaption></figure>
<p>在度量空间中，紧性等价于<strong>序列紧</strong>（每个序列有收敛子列），也等价于“完备且全有界”。全有界指对任意 \(\varepsilon>0\)，有限多个半径 \(\varepsilon\) 的球就能覆盖空间。它比有界强：无限离散空间的离散度量有界，却不能用有限个半径小于 1 的球覆盖。</p>
<p>在 \(\mathbb R^n\) 的通常度量下，闭且有界等价于紧，这才是 Heine–Borel 定理。将它当作一般拓扑空间的定义会出错，因为一般空间根本没有距离或“有界”的概念。</p>

<h2 id="section-38">六、理解检查</h2>
<ol><li>写出覆盖 \((0,1)\) 却没有有限子覆盖的另一族开集。</li><li>把“开覆盖有有限子覆盖”逐步改写为“闭集族有有限交性质则总交非空”。</li><li>证明连续映射 \([0,1]\to\mathbb R\) 有界并取到最大值，指出哪里用紧性。</li><li>为什么无限离散空间虽有界却不紧？</li></ol>
<p class="note-footer">依据林剑峰 / 陈晓《拓扑学讲义》第 4 章及 Körner <em>Metric and Topological Spaces</em> 第 9–10、12 节整理。</p>
</section>
