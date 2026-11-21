---
title: 第 10 篇：局部紧性、紧化与局部有限覆盖｜阅读笔记
layout: page
permalink: /courses/point-set-topology/notes/chapter-10/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/point-set-topology/">← 课程目录</a> · <a href="/courses/point-set-topology/notes/chapter-09/">上一篇</a></p>
<div class="note-lead"><p>紧性要求整个空间能从无限开覆盖中取有限部分；局部紧性只要求每个点附近有一小块可控区域。它使一点紧化成为可能；局部有限覆盖则把“局部可控”扩展成全空间的连续拼接方法。</p></div>
<nav class="reading-links" aria-label="参考讲义"><a href="/files/math-resources/archive-2026-11-08-59.pdf">林剑峰讲义：4.3、8</a><a href="/files/math-resources/archive-2026-11-08-43.pdf">Munkres 教材</a></nav>

<h2 id="section-56">一、局部紧与全局紧有什么不同？</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/10-local-compact.svg" alt="无界透视平面中的点 x 周围有一块闭包紧的邻域" loading="lazy"><figcaption>整个平面可以非紧，但每个点仍可拥有闭包紧的一小块邻域。</figcaption></figure>
<p>在 Hausdorff 空间中，称 \(X\) <strong>局部紧</strong>，若每个点 \(x\) 有开邻域 \(U\)，其闭包 \(\overline U\) 紧。\(\mathbb R^n\) 局部紧：在点周围取一个有界开球，其闭包闭且有界，由 Heine–Borel 定理紧。但 \(\mathbb R^n\) 本身不紧，例如递增开球覆盖没有有限子覆盖。</p>
<p>事实上，赋范空间局部紧当且仅当它有限维。无限维时，Riesz 引理可逐次选出单位球中彼此保持正距离的点，使单位闭球不可能紧；若某个零点邻域闭包紧，缩放后单位闭球也会紧，矛盾。这是把有限维几何论证直接搬到函数空间时会失败的地方。局部紧性还是第 9 篇紧开拓扑的评价映射与指数律所需的关键条件。</p>

<h2 id="section-57">二、一点紧化的开集怎样定义？</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/10-one-point.svg" alt="透视平面通过立体投影对应球面，远处方向汇到北极点无穷" loading="lazy"><figcaption>对 \(\mathbb R^2\)，离开每个紧区域的方向在球面北极合成一个新点 \(\infty\)。</figcaption></figure>
<p>设 \(X\) 非紧、局部紧且 Hausdorff。添加一个新点 \(\infty\)，得到 \(X^*=X\cup\{\infty\}\)。原空间中的开集仍开；包含 \(\infty\) 的集合规定为开，当且仅当它的补集是 \(X\) 中的紧闭集。换句话说，\(\infty\) 的基本邻域是 \(X^*\setminus K\)，其中 \(K\subseteq X\) 紧。</p>
<p>这样定义的直觉是：离开任意固定紧区域，就越来越靠近新点 \(\infty\)。\(X^*\) 紧：任意覆盖中取一个含 \(\infty\) 的成员，它已经盖住某个紧集之外的全部空间；剩余紧集可再由有限个成员盖住。\(X^*\) Hausdorff 则用局部紧性：给 \(x\in X\) 找紧闭包的开邻域 \(U\)，它与 \(X^*\setminus\overline U\) 分开 \(x\) 和 \(\infty\)。</p>
<p>\(\mathbb R\) 的一点紧化同胚于圆 \(S^1\)，\(\mathbb R^n\) 的一点紧化同胚于 \(S^n\)；这些结论可用立体投影具体构造。请勿把“一点紧化”误认为给原空间随意补一个度量极限点：它一次性代表离开所有紧集的方向。</p>

<h2 id="section-58">三、局部有限覆盖控制无限拼接</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/10-locally-finite.svg" alt="开片沿平面排列，点 x 附近的小邻域只碰到其中有限片" loading="lazy"><figcaption>固定 \(x\) 后，只有与它某个小邻域相交的有限片参与局部计算；远处的开片不影响这里的连续性。</figcaption></figure>
<p>开集族 \(\{U_i\}\) <strong>局部有限</strong>，若每个点都有一个邻域，只与有限多个 \(U_i\) 相交。它允许我们在无穷多个局部对象之间做求和，而每个点附近实际上只涉及有限项。称空间<strong>仿紧</strong>，若每个开覆盖都有局部有限的开细化；“细化”指新覆盖的每个成员都包含于原覆盖的某个成员。</p>
<p>每个度量空间都仿紧，这是比第二可数性更广的结论。例如不可数离散空间是度量空间，却不第二可数；它仍有局部有限细化。证明一般的度量空间仿紧需要系统构造层级覆盖，本篇使用这个标准定理而不把其证明简化成“选小球”一句话。</p>

<h2 id="section-59">四、单位分解连接拓扑与几何</h2>
<figure class="note-figure point-topology-figure"><img src="/images/notes/assets/point-set-topology/10-partition-unity.svg" alt="两条互补的函数高度线在重叠区域逐点相加为一" loading="lazy"><figcaption>重叠区内一条函数逐渐降到 0，另一条升到 1；每一点的权重相加仍是 1。</figcaption></figure>
<p>在仿紧 Hausdorff 空间中，对适当开覆盖可构造从属的<strong>单位分解</strong> \(\{\varphi_i:X\to[0,1]\}\)：各函数非零的区域落在相应的覆盖成员里；函数族局部有限；并且每点都有 \(\sum_i\varphi_i(x)=1\)。局部有限使总和在每点附近是有限和，因此连续。</p>
<p>实际用途是把局部构造拼成全局对象。若每个 \(U_i\) 上有局部函数 \(f_i\)，可以考虑 \(\sum_i\varphi_i f_i\)；在微分几何中，同样方法把局部坐标上的度量、向量场或微分形式拼起来。点集拓扑到这里提供了后续几何与分析所需的空间控制手段。</p>

<h2 id="section-60">五、这套笔记的主线</h2>
<p>第 1–2 篇建立开集、基、子空间、闭包与连续性；第 3 篇解释积、拓扑和、商怎样构造新空间；第 4–5 篇回答序列何时够用、点与闭集如何分开；第 6–8 篇处理紧、连通、完备三种不同的控制条件；第 9–10 篇说明何时可把拓扑重新表示为距离或函数值，以及如何在局部与全局之间过渡。</p>
<p>三条边界要一直记住：紧性是拓扑性质，完备性依赖选定度量；连通不必道路连通；连续双射不必同胚。后续同伦论、流形和代数拓扑会不断使用这些区分。</p>

<h2 id="section-61">六、理解检查</h2>
<ol><li>验证 \(\mathbb R\cup\{\infty\}\) 中，\(\infty\) 的邻域为何不能只写成“\(\mathbb R\) 上足够大的开区间”。</li><li>用紧性定义证明一点紧化的紧性，并指出 Hausdorff 证明中哪里用了局部紧性。</li><li>在不可数离散空间中，给出一个局部有限开覆盖，并说明它不是第二可数。</li><li>解释局部有限性怎样保证 \(\sum_i\varphi_i\) 在每点附近仍是有限个连续函数的和。</li></ol>
<p class="note-footer">依据林剑峰 / 陈晓《拓扑学讲义》第 4.3、8 章，并结合一般拓扑中的一点紧化、仿紧性和单位分解标准结论整理。</p>
</section>
