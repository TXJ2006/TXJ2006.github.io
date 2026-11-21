---
title: 第 3 篇：积、拓扑和与商空间｜阅读笔记
layout: page
permalink: /courses/point-set-topology/notes/chapter-03/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/point-set-topology/">← 课程目录</a> · <a href="/courses/point-set-topology/notes/chapter-02/">上一篇</a> · <a href="/courses/point-set-topology/notes/chapter-04/">下一篇 →</a></p>
<div class="note-lead"><p>已经知道怎样给子集继承拓扑。现在反过来：把多个空间并排放置、分开放置，或把若干点粘成一个点，分别应给结果什么拓扑？答案都由连续映射的检验方式决定。</p></div>
<nav class="reading-links" aria-label="参考讲义"><a href="/files/math-resources/archive-2026-11-08-59.pdf">林剑峰讲义：第 3、6 章</a><a href="/files/math-resources/archive-2026-11-08-45.pdf">Yi Li 讲义：1.3</a></nav>

<h2 id="section-15">一、两个空间的积：小矩形为什么是基？</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/03-product-box.svg" alt="透视长方体表示两个方向的开集相乘得到积空间的基本开集" loading="lazy"><figcaption>积空间的基本邻域同时限制两个坐标：第一个落在 \(U\)，第二个落在 \(V\)。</figcaption></figure>
<p>若 \(X,Y\) 是拓扑空间，集合 \(X\times Y\) 的点是有序对 \((x,y)\)。规定所有 \(U\times V\)（\(U\subseteq X,V\subseteq Y\) 分别开）为基元素，生成的拓扑称<strong>积拓扑</strong>。两个小矩形的交为 \((U_1\cap U_2)\times(V_1\cap V_2)\)，仍是小矩形，因此确实构成基。\(\mathbb R^2\) 的通常拓扑就可由开矩形生成；开圆盘形状不同，但每个盘内的点都能放进一个更小的开矩形。</p>
<p>投影 \(\pi_X:X\times Y\to X\) 连续，因为 \(\pi_X^{-1}(U)=U\times Y\) 开。更重要的是，\(f:Z\to X\times Y\) 连续，当且仅当两个坐标函数 \(\pi_Xf\)、\(\pi_Yf\) 都连续：对基元素 \(U\times V\) 的原像恰是 \((\pi_Xf)^{-1}(U)\cap(\pi_Yf)^{-1}(V)\)。这是积拓扑被选中的原因。</p>

<h2 id="section-16">二、无限积为什么只限制有限个坐标？</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/03-finite-coordinates.svg" alt="无限叠层坐标平面中只有前三层标出受限制的邻域" loading="lazy"><figcaption>积拓扑的一个基本邻域只在有限层施加条件；未标出的坐标可以任意变化。</figcaption></figure>
<p>对一族 \(\{X_i\}_{i\in I}\)，积集合的点是坐标列 \((x_i)_{i\in I}\)。积拓扑的基本开集形如 \(\prod_iU_i\)，其中每个 \(U_i\subseteq X_i\) 开，且只有<strong>有限多个</strong> \(U_i\ne X_i\)。因此 \((x^{(n)}_i)\to(x_i)\) 当且仅当每个固定坐标 \(i\) 都有 \(x^{(n)}_i\to x_i\)。证明一个方向用投影连续，另一个方向用基本开集仅检验有限个坐标。</p>
<p>若允许每个坐标都受限制，就得到更细的方框拓扑。例子：在 \(\mathbb R^{\mathbb N}\) 中令 \(e_n\) 的第 \(n\) 个坐标为 1，其余为 0。对每个固定坐标，\(e_n\) 最终为 0，故 \(e_n\to0\) 在积拓扑中成立；但方框开集 \(\prod_n(-1/2,1/2)\) 不含任何 \(e_n\)。这解释了为什么无限积不能随意换成方框拓扑。</p>

<h2 id="section-17">三、拓扑和：把空间分开放置</h2>
<p>拓扑和 \(\coprod_iX_i\) 先给每个空间加标签，防止原来同名的点被混淆。集合 \(U\subseteq\coprod_iX_i\) 开，当且仅当它在每个分量 \(X_i\) 中截出的部分都开。于是映射 \(f:\coprod_iX_i\to Y\) 连续，当且仅当每个限制 \(f|_{X_i}\) 连续，因为开集原像在每个分量上可分别检验。</p>
<p>两个分开的实数轴是 \(\mathbb R\coprod\mathbb R\)，并非一条更长的实数轴：两个分量各自是开集，也各自是闭集。这个构造与积的方向相反：积检验进入积空间的映射，拓扑和检验从拓扑和出发的映射。</p>

<h2 id="section-18">四、商空间：把点粘起来以后怎样定义开集？</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/03-quotient-cylinder.svg" alt="透视方形两条边粘合为圆柱，背面边界用虚线表示" loading="lazy"><figcaption>把方形的两条竖边按相同高度识别，得到圆柱；粘合后的开集由原像是否开来判定。</figcaption></figure>
<p>给 \(X\) 上的等价关系 \(\sim\)，商集 \(X/\!\sim\) 的元素是等价类。记 \(q:X\to X/\!\sim\)、\(x\mapsto[x]\)。规定 \(V\subseteq X/\!\sim\) 开，当且仅当 \(q^{-1}(V)\) 在 \(X\) 中开。这是<strong>商拓扑</strong>。由于原像保任意并及有限交，这些 \(V\) 的确构成拓扑。</p>
<p>注意不是“\(X\) 中开集的像都开”。若 \(q\) 把几个点压在一起，只有在原像是整个等价类的并且开放时，商中的集合才开。称满足 \(q^{-1}(q(U))=U\) 的集合 \(U\) 为<strong>饱和集</strong>；商开集正是开且饱和的原像。这个区别是计算商空间拓扑的关键。</p>
<p>例如把 \([0,1]\) 的端点 0 和 1 认作一点。映射 \([t]\mapsto e^{2\pi it}\) 给商空间到圆周的连续双射。原空间紧、圆周 Hausdorff，故这映射是同胚；单凭“图像像圆”并不足以得出同胚。</p>

<h2 id="section-19">五、商的泛性质与粘合例子</h2>
<p>若 \(f:X\to Z\) 在每个等价类上取同一个值，就存在唯一集合映射 \(\bar f:X/\!\sim\to Z\)，使 \(f=\bar f\circ q\)。商拓扑还保证：\(\bar f\) 连续当且仅当 \(f\) 连续，因为 \(q^{-1}(\bar f^{-1}(W))=f^{-1}(W)\)。这是判断粘合后映射是否连续的标准工具。</p>
<p>把方形的左右边同向对应点粘合，得到圆柱；再把上下边同向粘合，得到环面。若左右边反向对应，则先得到 Möbius 带。商空间可能失去好性质：把两条实数轴的非零对应点粘合，留下两个原点，这两个原点不能被不交开邻域分开。商拓扑需要按定义检验，不能只凭画图推断。</p>

<h2 id="section-20">六、练习与自检</h2>
<ol><li>证明 \(f:Z\to X\times Y\) 连续当且仅当它的两个坐标函数连续。</li><li>在端点粘合的 \([0,1]/(0\sim1)\) 中，写出粘合点的一个开邻域在原区间中的原像。</li><li>举例说明商映射不一定是开映射。</li><li>解释两条实数轴的非零点粘合后，为什么两个原点不能被不交开集分开。</li></ol>
<p class="note-footer">依据林剑峰 / 陈晓《拓扑学讲义》第 3、6 章及 Yi Li <em>Topology I</em> 第 1.3 节整理。</p>
</section>
