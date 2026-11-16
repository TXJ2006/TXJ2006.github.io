---
title: 第三单元 7：换基与图像压缩｜阅读笔记
layout: page
permalink: /courses/gilbert-linear-algebra/notes/unit-3-session-07/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--unit3">
<p class="note-nav"><a href="/courses/gilbert-linear-algebra/notes/">← 笔记目录</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-3-session-06/">上一讲</a> · <a href="/courses/gilbert-linear-algebra/notes/unit-3-session-08/">下一讲 →</a></p>
<div class="note-lead"><p>上一讲说明：选基以后才有矩阵。本讲追问换基究竟改了什么、没有改什么；随后看实际用途。像素值在标准基里看似杂乱，在合适的频率或小波基里可能只需少数几个较大的系数。</p></div>
<nav class="reading-links" aria-label="本讲资料"><a href="/files/mit-18-06sc/unit-3-session-7-summary.pdf">讲义摘要 PDF</a><a href="/files/mit-18-06sc/unit-3-session-7-problems.pdf">习题 PDF</a><a href="/files/mit-18-06sc/unit-3-session-7-solutions.pdf">参考解答 PDF</a></nav>

<h2 id="coordinates">一、同一个向量，两套坐标</h2>
<p>把新基 \(w_1,\ldots,w_n\) 按<strong>旧坐标</strong>排成可逆矩阵 \(W=[w_1\ \cdots\ w_n]\)。若向量在新基中的坐标为 \(c=(c_1,\ldots,c_n)^T\)，其旧坐标 \(x\) 满足 \(x=Wc\)，所以 \(c=W^{-1}x\)。顺序要从这个等式推，不能凭“换基应乘哪个方向”来记。</p>
<p>例如新基取 \(w_1=(1,1)^T\)、\(w_2=(1,-1)^T\)，向量旧坐标 \(x=(3,1)^T\)。解 \(x=c_1w_1+c_2w_2\) 得 \(c=(2,1)^T\)。原向量没有变，只是“沿新基各走多少”的两个数变了。若新基标准正交，\(W^{-1}=W^T\)，每个新坐标就是 \(w_j^Tx\)；一般基不能直接用转置。</p>

<h2 id="similarity">二、矩阵换基公式是怎样来的？</h2>
<p>同一个线性映射在旧基下由 \(A\) 表示，在新基下由 \(B\) 表示。先把新坐标 \(c\) 送到旧坐标 \(Wc\)，作用 \(A\)，再转回新坐标：\(Bc=W^{-1}AWc\)。因任意 \(c\) 都成立，</p>
<div class="equation">\[B=W^{-1}AW.\]</div>
<p>这就是第 4 讲的相似关系。若新基恰好由特征向量组成，\(B\) 为对角阵。若输入与输出是<strong>两个不同空间</strong>且两边都换基，则应写 \(B=W_{\mathrm{out}}^{-1}AW_{\mathrm{in}}\)，不再是相似式。两个基矩阵各自负责一边的坐标，不能混成同一个 \(W\)。</p>

<h2 id="compression">三、换基为什么可能帮助压缩？</h2>
<p>把一幅灰度图像看作由像素亮度组成的长向量。换基 \(c=Q^Tx\) 本身是可逆的，不会减少维数或信息量；只有在新坐标中许多系数很小，随后<strong>量化、舍弃、编码</strong>这些系数，才可能省存储空间。若只保留索引集合 \(I\) 上的坐标，重建为 \(\widehat x=\sum_{i\in I}c_iq_i\)。因 \(Q\) 标准正交，平方误差精确等于 \(\|x-\widehat x\|^2=\sum_{i\notin I}|c_i|^2\)；这告诉我们优先保留绝对值较大的系数。</p>
<p>取四个像素的简单信号 \(x=(8,8,4,4)^T\)。在像素基里四个坐标都非零；取 Haar 方向</p>
<div class="equation">\[\begin{aligned}
q_0&=\tfrac12(1,1,1,1)^T,\\
q_1&=\tfrac12(1,1,-1,-1)^T,\\
q_2&=\tfrac1{\sqrt2}(1,-1,0,0)^T,\\
q_3&=\tfrac1{\sqrt2}(0,0,1,-1)^T.
\end{aligned}\]</div>
<p>四个方向长度都是 1，且两两正交。计算 \(c_j=q_j^Tx\) 得 \(c=(12,4,0,0)^T\)，因此 \(x=12q_0+4q_1\)。这个例子展示了“整体亮度 + 左右半幅差异”两种结构；这里只是稀疏表示的示意，是否真的省字节还取决于记录基、系数、位置及编码的成本。</p>
<figure class="note-figure"><svg viewBox="0 0 560 235" role="img" aria-labelledby="haar-title haar-desc"><title id="haar-title">四像素信号的 Haar 分解</title><desc id="haar-desc">输入亮度为 8、8、4、4；它分解为一个四像素均为 6 的平均分量，以及左半为 2、右半为负 2 的差异分量。</desc><g text-anchor="middle"><text x="92" y="30">输入 x</text><text x="282" y="30">12q₀：平均</text><text x="466" y="30">4q₁：左右差</text></g><g stroke="#87958f" fill="none"><path d="M25 165H160M210 165H345M395 165H535"/></g><g fill="#557f79"><rect x="33" y="69" width="27" height="96"/><rect x="65" y="69" width="27" height="96"/><rect x="97" y="117" width="27" height="48"/><rect x="129" y="117" width="27" height="48"/><rect x="218" y="93" width="27" height="72"/><rect x="250" y="93" width="27" height="72"/><rect x="282" y="93" width="27" height="72"/><rect x="314" y="93" width="27" height="72"/><rect x="403" y="141" width="27" height="24"/><rect x="435" y="141" width="27" height="24"/></g><g fill="#bb6a4b"><rect x="467" y="165" width="27" height="24"/><rect x="499" y="165" width="27" height="24"/></g><text x="178" y="128">=</text><text x="362" y="128">+</text><text x="59" y="217">8, 8, 4, 4</text><text x="240" y="217">6, 6, 6, 6</text><text x="446" y="217">2, 2, -2, -2</text></svg><figcaption>两项之和逐像素恢复原信号；基中另外两个局部差异方向的系数为零。橙色柱在基线下表示负值。</figcaption></figure>
<p>讲义还讨论 Fourier 与余弦基。常见 JPEG 的 8×8 块主要使用二维离散余弦变换（DCT），随后量化并编码；它与前面的复数 DFT 有亲缘关系，但不能简单说成只取 Fourier 基向量的实部。选择基还要兼顾两件事：变换及逆变换算得快，且常见图像在这组基下容易用少数重要系数近似。</p>
<h2 id="check">理解检查</h2>
<ol><li>先逐对检查本页四个 Haar 方向正交，并求信号 \((8,0,0,0)^T\) 的四个系数；习题 31.1 则要求核对讲义中的八维 Haar 基并归一化。</li><li>若只保留 \(q_0\) 的系数，重建 \(x\) 为何？直接核对误差平方等于被舍弃系数平方和。</li><li>习题 31.2：为全体 \(2\times2\) 实矩阵给出两组基，比较哪组更方便描述对角矩阵与对称矩阵。</li></ol>
<p class="note-footer">依据 MIT OCW 18.06SC Unit III, Session 7 <em>Change of Basis; Image Compression</em> 整理。压缩部分区分可逆换基与有损编码，并注明 JPEG 所用的 DCT。<a href="/courses/gilbert-linear-algebra/notes/">返回目录</a>。</p>
</section>
