---
title: 第 8 篇：完备、全有界与 Baire 定理｜阅读笔记
layout: page
permalink: /courses/point-set-topology/notes/chapter-08/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/point-set-topology/">← 课程目录</a> · <a href="/courses/point-set-topology/notes/chapter-07/">上一篇</a> · <a href="/courses/point-set-topology/notes/chapter-09/">下一篇 →</a></p>
<div class="note-lead"><p>拓扑只能描述哪些点靠近；要讨论 Cauchy 序列，还需要具体度量。本篇区分“完备”与“紧”，说明如何补齐缺失的极限，并用 Baire 定理展示完备性的力量。</p></div>
<nav class="reading-links" aria-label="参考讲义"><a href="/files/math-resources/archive-2026-11-08-59.pdf">林剑峰讲义：3.3、4.2</a><a href="/files/math-resources/archive-2026-11-08-34.pdf">Körner 讲义：12</a></nav>

<h2 id="section-45">一、Cauchy 和收敛的差别</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/08-cauchy-hole.svg" alt="点列彼此越来越近，却逼近空间中一个缺失的空心点" loading="lazy"><figcaption>Cauchy 只检查项与项的距离；空心点不在原空间时，序列可能没有空间内极限。</figcaption></figure>
<p>在度量空间 \((X,d)\) 中，序列 \((x_n)\) 是 Cauchy 序列，若对每个 \(\varepsilon>0\)，存在 \(N\) 使 \(m,n\ge N\) 时 \(d(x_m,x_n)\lt\varepsilon\)。它只比较序列项彼此是否越来越近，不预设极限存在于 \(X\)。若每个 Cauchy 序列都在 \(X\) 中收敛，称 \((X,d)\) 完备。</p>
<p>在 \(\mathbb Q\) 的通常度量中，有理数可逐位逼近 \(\sqrt2\)，形成 Cauchy 序列，但在 \(\mathbb Q\) 内没有极限，所以不完备。\(\mathbb R\) 的通常度量完备；其闭子集也完备。若 \(X\) 完备且子空间 \(A\subseteq X\) 完备，则 \(A\) 必闭：任意 \(x\in\overline A\) 可在 \(A\) 中选序列逼近 \(x\)，该序列 Cauchy，在 \(A\) 中的极限必须为 \(x\)。</p>

<h2 id="section-46">二、完备不是纯拓扑性质</h2>
<p>令 \(h:\mathbb R\to(0,1)\)、\(h(x)=e^x/(1+e^x)\)，并把 \((0,1)\) 的欧氏度量拉回 \(\mathbb R\)：\(d_h(x,y)=|h(x)-h(y)|\)。它与通常度量给出相同的开集，因为 \(h\) 是同胚；但 \((\mathbb R,d_h)\) 不完备，因为 \(n\to\infty\) 时 \(h(n)\to1\)，\((n)\) 在 \(d_h\) 下是 Cauchy，却没有 \(d_h\)-极限。</p>
<p>所以“这一个度量完备”不是拓扑性质；“存在某个与此拓扑相容的完备度量”才称<strong>可完全度量化</strong>，它是拓扑性质。这个区分很重要，不能从两个空间同胚直接推断它们选定的度量都完备。</p>

<h2 id="section-47">三、全有界 + 完备 = 度量紧</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/08-total-complete.svg" alt="逐层缩小的邻域中抽出 Cauchy 子列并找到极限" loading="lazy"><figcaption>全有界让我们反复抽出落在更小球中的子列；这些子列形成 Cauchy 子列，完备性才给出极限。</figcaption></figure>
<p>集合 \(X\) 全有界，若对每个 \(\varepsilon>0\)，存在有限个半径 \(\varepsilon\) 的开球覆盖 \(X\)。它比“有界”强：无限集合上的离散度量 \(d(x,y)=1\)（不同点）有界，但当 \(\varepsilon<1\) 时，每个球只含一个点，不可能有限覆盖。</p>
<p>度量空间紧当且仅当完备且全有界。证明“\(\Leftarrow\)”的主线是：对 \(\varepsilon=1,1/2,1/3,\ldots\) 依次用有限球覆盖，从任意序列中连续抽取子列，使它们在越来越小的球内，得到 Cauchy 子列；完备性给极限，于是序列紧。度量空间中序列紧等价于开覆盖紧。反向则分别用紧性排除没有极限的 Cauchy 序列，并为每个 \(\varepsilon\)-球覆盖取有限子覆盖。</p>

<h2 id="section-48">四、完备化把缺失极限补回来</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/08-completion.svg" alt="下层数轴的缺口在上层完备化数轴中被填入" loading="lazy"><figcaption>把彼此距离趋于零的 Cauchy 序列视作同一点，便在完备化里补上原空间缺失的极限。</figcaption></figure>
<p>从 \(X\) 中全部 Cauchy 序列出发，规定 \((x_n)\sim(y_n)\) 当且仅当 \(d(x_n,y_n)\to0\)。在等价类上定义 \(\widehat d([(x_n)],[(y_n)])=\lim_n d(x_n,y_n)\)；三角不等式保证极限存在，等价关系保证定义不依赖代表元。把 \(x\in X\) 送到常值序列 \((x,x,\ldots)\)，得到等距嵌入，且其像在新空间稠密。</p>
<p>得到的 \(\widehat X\) 完备，称 \(X\) 的完备化；它在保持 \(X\) 不变的等距同构意义下唯一。\(\mathbb Q\) 的通常度量完备化是 \(\mathbb R\)。这个构造与第 3 篇的商类似：先构造更大的集合，再把“距离趋于零”的序列认作同一个点。</p>

<h2 id="section-49">五、Baire 定理为何令人意外？</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/08-baire-balls.svg" alt="越来越小的嵌套闭球具有共同的极限点" loading="lazy"><figcaption>每一步把闭球缩进下一个稠密开集；直径趋零与完备性共同保证最终有一个公共点。</figcaption></figure>
<p>若 \(U_n\) 都开且稠密，<strong>Baire category theorem</strong> 断言在非空完备度量空间中 \(\bigcap_{n\ge1}U_n\) 仍稠密。每个 \(U_n\) 单独“到处都有”，结论说可同时满足可数多个这样的条件。</p>
<p>证明思路：任取非空开集 \(V\)，在 \(V\cap U_1\) 中选一个小闭球；再在它的内部与 \(U_2\) 的交中选更小闭球，依次使闭球嵌套且直径趋于零。球心成为 Cauchy 序列，完备性给出一个点落在所有闭球中，因而落在 \(V\cap\bigcap_nU_n\)。这说明交集与每个非空开集都相交，即稠密。</p>
<p>对比 \(\mathbb Q\)：它是自身中每个单点闭集的可数并，单点都无内部。因此 \(\mathbb Q\) 不是 Baire 空间，也不可能在它的通常拓扑上找到任何相容的完备度量。</p>

<h2 id="section-50">六、理解检查</h2>
<ol><li>证明收敛序列必为 Cauchy，并找一个 Cauchy 但在原空间不收敛的序列。</li><li>判断 \((0,1)\) 的欧氏度量是否全有界、是否完备、是否紧。</li><li>解释 \(d_h\) 与普通距离开集相同但完备性不同的关键步骤。</li><li>在 Baire 证明中，指出哪里使用了稠密性，哪里使用了完备性。</li></ol>
<p class="note-footer">依据林剑峰 / 陈晓《拓扑学讲义》第 3.3、4.2 节及 Körner <em>Metric and Topological Spaces</em> 第 12 节整理。</p>
</section>
