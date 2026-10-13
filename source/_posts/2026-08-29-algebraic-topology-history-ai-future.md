---
title: "从 Analysis Situs 到可学习不变量：代数拓扑的数学基础与 AI 时代"
date: 2026-08-29 22:00:00
series: 代数拓扑入门讲义
series_order: 1
subtitle: 从拓扑不变量、同调与基本群到持久同调的概览
description: 从拓扑不变量、同调与基本群到持久同调的概览
categories:
  - 理论数学
tags:
  - 代数拓扑
  - 同调
  - 持续同调
  - 拓扑数据分析
  - 数学史
  - 机器学习
mathjax: true
toc: true
comments: true
---

闭合曲线所围成的区域与曲线的长度、曲率属于不同的研究对象。前者要求先规定允许的变形，再研究在这些变形下保持不变的量。Poincare 的 *Analysis Situs* 是这一思想的早期系统表述；基本群、同调与上同调则把它发展成了可计算的理论。

本讲义先区分参数曲线、嵌入及其像空间，再以链复形和边界算子计算圆周的同调。随后说明 Euler 示性数、Betti 数、基本群与上同调之间的关系，最后介绍过滤、持久同调及其在拓扑数据分析中的作用。经典部分主要参照 Hatcher；关于映射、链复形和泛性质的组织方式，参考 Grodal 等人的公开讲义。

<!--more-->

## 1. Poincare 的问题：如何描述可变形的空间

长度、面积和曲率会随几何形状改变，而圆周上的闭合路径能否在圆周内连续收缩为一点则是拓扑性质。因此必须先固定研究对象：在平面中移动一条嵌入的线段并不会改变其同胚类型；识别两个端点，或允许不同参数点相交，则会得到另一个空间。下面的例子说明这一点。

### 1.1 自交意味着像空间发生变化

这里容易混淆三个不同的对象：线段作为抽象空间、线段在平面中的嵌入，以及一条参数曲线的像。若

$$
\gamma:[0,1]\to\mathbb{R}^2
$$

始终是单射，那么它的像 $X=\gamma([0,1])$ 与区间 $[0,1]$ 同胚，因而

$$
H_1(X)=0.
$$

这里的结论来自同调的不变性，而不是来自图形的直观外观。因为 $[0,1]$ 紧、$\mathbb{R}^2$ Hausdorff，连续单射 $\gamma:[0,1]\to X$ 是到其像的同胚；区间又是可缩空间，所以

$$
H_1(X)\cong H_1([0,1])=0.
$$

这正是“嵌入的线段没有一维洞”的严格理由。关于同伦不变性和区间的同调计算，可参见 Hatcher 的 [*Algebraic Topology*，Chapter 2](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf)；本文只使用其中的标准结论，并把中间的同胚步骤写出。

只要不同参数点不被识别，线段作为拓扑空间的类型就没有改变。曲率是微分几何量，不存在“曲率足够大便产生拓扑洞”的阈值；改变拓扑对象的是端点识别或自交。识别端点得到圆周；若存在 $s\neq t$ 使 $\gamma(s)=\gamma(t)$，则 $\gamma$ 不再是嵌入，其像空间可能成为含有环的平面图。

最典型的例子是“8”字形。把交点看成一个顶点、把两个圆瓣看成两条边，得到一个图 $X$，其中

$$
V=1,\qquad E=2,
$$

所以

$$
b_1(X)=E-V+1=2.
$$

![图：把八字形视为一个顶点和两条环边](/images/notes/assets/algebraic-topology/figure-eight-graph.svg)

*图 1. 八字形的抽象图模型。几何上的两个圆瓣对应两条环边 $e_1,e_2$，交点对应同一个顶点 $v$。*

这个计算可以直接从链复形核对。取整数系数，有 $C_1(X)=\mathbb{Z}e_1\oplus\mathbb{Z}e_2$、$C_0(X)=\mathbb{Z}v$，而两条边都是从 $v$ 回到 $v$，所以 $\partial_1(e_1)=\partial_1(e_2)=0$。图中没有二维胞腔，故 $C_2(X)=0$，从而

$$
H_1(X)=\ker\partial_1/\operatorname{im}\partial_2
      =C_1(X)\cong\mathbb{Z}^2.
$$

因此 $b_1(X)=\operatorname{rank}H_1(X)=2$。对任意有限连通图，选一棵极大树后，未被选入的边各贡献一个独立环；极大树有 $V-1$ 条边，于是得到一般公式 $b_1=E-(V-1)=E-V+1$。这也是 Hatcher [Chapter 1 的图与 van Kampen 例子](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf)中“图的基本群是自由群”的同调版本。

这说明它有两个独立的一维环。此时参数化曲线的定义域仍可以是一个圆周，甚至可以用一个区间从头走到尾；但曲线的**像**已经是两个圆在交点处的粘合，不能再把它当作原来的嵌入线段。

还要区分“曲线自身的洞”和“曲线在平面中围出的区域”。对具有 $n\geq1$ 个横截双重点、没有三重交点的连通闭曲线，把每个交点当作图的顶点，则

$$
V=n,\qquad E=2n.
$$

由平面 Euler 公式

$$
V-E+F=2
$$

得到总区域数

$$
F=n+2.
$$

其中一个区域是无界区域，所以有 $n+1$ 个有界区域；另一方面，连通图的第一 Betti 数为

$$
b_1=E-V+1=n+1.
$$

在这个平面图情形下，“有多少个独立的环”和“补空间有多少个有界区域”恰好相等。Alexander 对偶性把这种现象写成

$$
\widetilde H_0(S^2\setminus X;\mathbb{F})
\cong
\widetilde H^1(X;\mathbb{F}),
$$

其中 $\mathbb{F}$ 是系数域。右侧描述曲线自身的一维上同调，左侧描述从球面中删去曲线后剩下的连通分支。因此，弯曲、交错并形成洞的现象确实可以出现；所研究的对象是自交曲线的像及其补空间，而不是始终保持嵌入的线段。

上式是 Alexander 对偶性的 $S^2$ 特例；一般形式需要紧子集和约化同调/上同调的适当条件。标准出处见 Hatcher [*Algebraic Topology*，Chapter 3 的 Alexander duality 部分](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf)。这里把它用于平面图时，$\widetilde H_0$ 的秩就是补空间连通分支数减一，所以与 $b_1$ 的数值吻合。

这类对象早已有系统理论。Jordan 曲线定理研究无自交闭曲线如何把平面分成内外两部分；Whitney–Graustein 定理研究允许自交的正则闭曲线，并证明它们在正则同伦下由旋转数分类；计算拓扑中还会把自交曲线编码成像图、面与 Gauss 编码。若曲线实际位于三维空间，平面图上的交叉还可能只是投影中的交叉，这又进入结理论和 Reidemeister 变换。

因此需要先决定什么叫“同一种形状”。存在连续双射且逆映射也连续时，两个空间**同胚**。另一种更宽的关系是**同伦等价**：存在 $f:X\to Y$ 与 $g:Y\to X$，使 $g\circ f$ 和 $f\circ g$ 分别同伦于各自的恒等映射。圆盘与一点同伦等价，却不同胚；圆周不与一点同伦等价，后面计算的 $H_1(S^1)$ 会给出理由。这里比较的是空间本身，不能把“某条画在平面上的曲线看起来变圆了”当作同伦等价的定义。

Poincare 在 1895 年发表 *Analysis Situs*。它留下的计算问题是：能否从空间的一个剖分出发，构造不依赖于剖分细节的不变量？

最早的指纹之一是 Euler 示性数。对一个有限三角剖分，记顶点、边和三角形的个数分别为 $V,E,F$，定义

$$
\chi=V-E+F.
$$

对有限 CW 复形，Euler 示性数还可以按胞腔数写成 $\chi(X)=\sum_k(-1)^k a_k$，并且等于同调秩的交替和 $\sum_k(-1)^k\operatorname{rank}H_k(X)$。这不是额外的记忆公式：链群的秩按短正合列相减时，循环群和边界群的项逐维抵消，最后只留下同调群的秩。MIT OpenCourseWare 的 [Lecture 18: Euler Characteristic and Homology Approximation](https://ocw.mit.edu/courses/18-905-algebraic-topology-i-fall-2016/resources/mit18_905f16_lec18/) 在第 45 页给出了这一命题和证明；本仓库保存了该页的出处截图和获得许可的 PDF。

![MIT OpenCourseWare Lecture 18 第 45 页：Euler 示性数与同调秩公式](/images/notes/assets/algebraic-topology/mit-ocw-euler-characteristic-excerpt.png)

> **原文摘录。** “This integer $\chi(X)$ is called the Euler characteristic of $X$.” 随后 Lecture 18 的 Theorem 18.3 给出 $\chi(X)=\sum_k(-1)^k\operatorname{rank}H_k(X)$。
>
> 来源：Sanath Devalapurkar，*MIT 18.905 Algebraic Topology I*，Lecture 18，p. 45，[课程页面](https://ocw.mit.edu/courses/18-905-algebraic-topology-i-fall-2016/resources/mit18_905f16_lec18/)，[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。

本讲义仓库还保存了这份 [Lecture 18 PDF](/files/algebraic-topology/mit-ocw/MIT18_905F16_lec18.pdf)；同一课程的其他已授权讲义和原始链接见[素材索引](/files/algebraic-topology/README.html)。

一个球面的三角剖分可以有不同数量的顶点和边，但 $\chi$ 总是 $2$；环面的 Euler 示性数则是 $0$。这已经说明，计数并不只是计数：交替相减后，局部剖分的细节会被抵消，剩下的是整体形状的一部分。

但 Euler 示性数只有一个整数，无法说明差异发生在哪个维度。例如 $\mathbb{R}P^2$ 与 $S^1\vee S^2$ 都有 $\chi=1$；它们的第一整数同调分别是 $\mathbb Z/2\mathbb Z$ 和 $\mathbb Z$。前者有一个二阶元素，后者则有无限阶元素，单看 $\chi$ 根本读不出来。同调群正是为了保留这些信息。

## 2. 同调：把“洞”变成可以计算的群

把空间剖分成点、边、三角形、四面体等单纯形。所有 $k$ 维单纯形的整数线性组合组成链群，记为 $C_k$。例如，一个三角形 $[v_0v_1v_2]$ 的边界是三条有向边的交替和：

$$
\partial_2[v_0v_1v_2]
=[v_1v_2]-[v_0v_2]+[v_0v_1].
$$

边界算子 $\partial_k:C_k\to C_{k-1}$ 把一个 $k$ 维对象送到它的边界。关键事实是

$$
\partial_{k-1}\partial_k=0.
$$

直观地说，边界本身没有边界：一条线段的边界是两个端点，而端点再没有边界；一个三角形的三条边再取边界，每个顶点会出现两次，方向相反，最后全部抵消。这个事实不是记号游戏，它使下面的商群有意义。

定义 $k$-循环群和 $k$-边界群：

$$
Z_k:=\ker\partial_k,
\qquad
B_k:=\operatorname{im}\partial_{k+1}.
$$

因为 $\partial_k\partial_{k+1}=0$，每个边界都是循环，因此 $B_k\subseteq Z_k$。第 $k$ 个同调群定义为

$$
H_k:=Z_k/B_k.
$$

这个商的含义是：先收集所有“没有边界的闭合对象”，再把那些本身就是某个更高维对象边界的闭合对象视为零。一个圆周上的闭合路径不是某个二维区域的边界，所以它代表一个非平凡的 $H_1$ 类；三角形的边界虽然也是闭合路径，却在 $H_1$ 中被当作零。

如果把整数系数换成有限域 $\mathbb{F}_2$，方向和负号会消失，计算会更简单；如果保留整数系数，则可以看到扭结等更细的现象。系数的选择不是技术细节，它决定我们允许观察哪些结构。

## 3. 从 Betti 数到同伦：代数拓扑逐渐长出自己的语言

同调群通常可能很复杂，但在有限生成的情形，可以用自由部分的秩来定义 Betti 数：

$$
b_k=\operatorname{rank}H_k.
$$

$b_0$ 记录连通分支的数量，$b_1$ 可以记录独立的一维洞，$b_2$ 可以记录空腔。对有限复形，Euler 示性数可以写成 Betti 数的交替和：

$$
\chi=\sum_{k\geq0}(-1)^kb_k.
$$

这条公式解释了上一节的 $V-E+F$ 为什么与同调有关：链群的秩交替相加时，相邻维度的边界部分互相抵消，剩下同调群自由部分的秩。要证明它不依赖剖分，还需要同调的拓扑不变性；仅靠这个计数恒等式并不足以推出这一点。附录 E 写出了秩的抵消过程。

但同调并没有回答所有问题。两个空间可能有相同的同调群，却有不同的基本群；一个空间的局部环路如何相乘，不能仅靠“有几个洞”描述。于是人们研究基本群 $\pi_1$、高阶同伦群和上同调。与同调相比，同伦群通常更难计算，却保留了更多关于连续映射的信息。

只给每个空间列一张同调群表仍不够。一个连续映射 $f:X\to Y$ 还会诱导同调群之间的映射

$$
f_*:H_k(X)\to H_k(Y).
$$

如果 $f$ 和 $g$ 同伦，那么它们诱导的同调映射相同。对可复合的映射 $f,g$，还有

$$
H_k(g\circ f)=H_k(g)\circ H_k(f),
\qquad
H_k(\operatorname{id}_X)=\operatorname{id}_{H_k(X)}.
$$

复合与恒等映射这两条规则是“同调是函子”的具体含义。Grodal 的《Categories and Topology》从对象与映射及其复合出发组织理论；这里我们只借用这层语言，不预设读者已经学过讲义后面的单纯集或高阶范畴。

Eilenberg 与 Mac Lane 在 1945 年关于自然变换的工作，把这种“映射之间的映射”提升为范畴论语言。范畴论有时会让初学者觉得过于抽象，但它解决了一个实在的问题：当我们有很多种不变量、很多种构造时，怎样区分自然出现的关系和人为拼接的公式？

例如，同一空间可以用不同的剖分计算同调；真正有用的结论不只是两次计算得到同构的群，还要说明这些同构与连续映射相容。这正是自然性要回答的问题。这里的次序是先理解一个空间的计算，再理解映射如何作用于计算结果。

## 4. 现代观点：过滤、持久同调与结构定理

经典代数拓扑通常从一个空间出发；持久同调则从一串彼此包含的空间出发。设

$$
X_0\subseteq X_1\subseteq\cdots\subseteq X_m
$$

是有限过滤。每个包含映射都会诱导同调映射，于是对固定的 $k$ 得到一个线性对象

$$
H_k(X_0)\longrightarrow H_k(X_1)
\longrightarrow\cdots\longrightarrow H_k(X_m).
$$

若系数取在域 $\mathbb{F}$ 上，每个 $H_k(X_i)$ 都是有限维向量空间。对 $i\leq j$，记复合映射的像的维数为

$$
\rho_k(i,j)
:=\dim\operatorname{im}
\left(H_k(X_i)\to H_k(X_j)\right).
$$

这个秩函数记录一个同调类在过滤中能存活多久。在线性代数意义下，有限型持久模可以分解成区间模的直和：每一个区间表示一个类从某一步出生，在另一步死亡，或一直存活到过滤末端。把所有区间画在同一条数轴上，就是 barcode。

对单纯复形而言，这个分解可以通过边界矩阵的列消元得到。按过滤顺序排列所有单纯形，把每个 $k$-单纯形的边界写成一列；在 $\mathbb{F}_2$ 上，列加法不改变生成的边界空间。消元时出现的主元把一个较低维的单纯形和一个较高维的单纯形配对，配对位置正好记录同调类的出生与死亡。Edelsbrunner 与 Harer 的 *Computational Topology* 系统整理了这一计算观点，Ghrist 的 barcode 文章则给出了简洁的几何解释。

持久同调的数学价值不在于把任意点集都变成漂亮图形，而在于它把“尺度”作为结构的一部分。若过滤来自一个函数 $f:X\to\mathbb{R}$ 的次水平集

$$
X_a=f^{-1}((-\infty,a]),
$$

那么 $a$ 变化时的同调记录了函数的全局形状。在函数满足通常的有限型（tame）条件、过滤有限且系数取在同一个域时，稳定性定理给出

$$
d_B\bigl(D(f),D(g)\bigr)
\leq\lVert f-g\rVert_\infty,
$$

其中左侧是 persistence diagram 的 bottleneck 距离，右侧是函数的上确界距离。这个不等式说明：函数只发生小扰动时，拓扑摘要不会任意跳变；它是持久同调能够成为数学工具，而不只是可视化技巧的原因。

## 5. 上同调、乘法结构与对偶性

同调用链来研究“边界为零”的对象。上同调则把链看成输入，把数值函数放在链上。给定交换系数环 $R$，定义 $k$-上链群

$$
C^k(X;R):=\operatorname{Hom}(C_k(X),R).
$$

若 $\varphi\in C^k(X;R)$，定义余边界算子

$$
\delta^k\varphi:=\varphi\circ\partial_{k+1}.
$$

由于 $\partial_k\partial_{k+1}=0$，有

$$
\delta^{k+1}\delta^k\varphi
=\varphi\circ\partial_{k+1}\circ\partial_{k+2}=0.
$$

因此同样可以定义

$$
Z^k=\ker\delta^k,
\qquad
B^k=\operatorname{im}\delta^{k-1},
\qquad
H^k(X;R)=Z^k/B^k.
$$

上同调比同调多出一个重要结构：不同次数的上链可以相乘。对单纯上链，若 $\alpha$ 是 $p$-上链、$\beta$ 是 $q$-上链，可以定义 cup product

$$
(\alpha\smile\beta)[v_0\ldots v_{p+q}]
=\alpha[v_0\ldots v_p]
\,\beta[v_p\ldots v_{p+q}].
$$

它在上同调类上给出

$$
H^p(X;R)\times H^q(X;R)
\longrightarrow H^{p+q}(X;R),
$$

使得 $H^*(X;R)$ 不只是各个群的列表，而是一个分次环。这个乘法能区分一些同调群相同、但整体拓扑结构不同的空间。

对闭的、可定向的 $n$ 维流形 $M$，Poincare 对偶性给出下列同构。关于该定理的标准证明路线，可参见 Hatcher [Chapter 3](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf) 和 MIT OpenCourseWare [Lecture 37: Poincare duality](https://ocw.mit.edu/courses/18-905-algebraic-topology-i-fall-2016/resources/mit18_905f16_lec37/)。

$$
H^k(M;R)\cong H_{n-k}(M;R),
$$

在合适的系数条件下成立。它把“$k$ 维的上同调信息”和“$n-k$ 维的几何洞”联系起来。对光滑流形，de Rham 定理又把微分形式构成的上同调与奇异上同调联系起来：

$$
H_{\mathrm{dR}}^k(M)
\cong H^k(M;\mathbb{R}).
$$

于是同一个拓扑不变量可以用三种语言描述：单纯形上的代数、连续映射下的同调类，以及满足 $d\omega=0$ 的微分形式。代数拓扑的力量，正来自这些看似不同的语言最终能够相互翻译。

## 6. 回到 AI：拓扑提供的是数学约束

在这里，AI 只作为一个自然的后续问题出现。神经网络的表示空间、图结构和高阶关系都可以被看成某种复形或过滤，但真正有意义的连接必须先回答两个数学问题：哪些变换应被视为等价，以及哪些结构在这些变换下保持不变。

如果一组变换构成群 $G$，一个表示 $F:X\to V$ 的不变性可以写成

$$
F(g\cdot x)=F(x),
\qquad g\in G;
$$

等变性则要求存在表示 $\rho:G\to\operatorname{GL}(V)$，使

$$
F(g\cdot x)=\rho(g)F(x).
$$

这与代数拓扑中的函子性有相似的精神：先规定允许的映射，再要求构造与这些映射兼容。单纯复形和胞腔复形还可以承载高阶关系，持久同调则提供观察表示空间全局变化的一种方式。

但拓扑不自动产生语义。一个同调类是否对应任务中的概念，必须由额外的数学假设和实验验证；拓扑不变量的稳定性，也不能替代对数据生成机制的理解。更稳妥的说法是：代数拓扑为 AI 提供了一组关于等价性、全局结构和稳定性的语言，至于这些语言能否改善具体模型，仍然是需要证明的问题。

## 参考文献

1. H. Poincare, “Analysis Situs,” *Journal de l'École Polytechnique*, 1895. [Gallica](https://gallica.bnf.fr/ark:/12148/bpt6k4336762/f285.item).
2. E. Betti, “Sopra gli spazi di un numero qualunque di dimensioni,” *Annali di Matematica Pura ed Applicata*, 1871.
3. S. Eilenberg and S. Mac Lane, “General Theory of Natural Equivalences,” *Transactions of the American Mathematical Society*, 1945. [JSTOR](https://www.jstor.org/stable/1990284).
4. A. Hatcher, *Algebraic Topology*, Cambridge University Press, 2002, Chapters 0--3. [Author's site](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf).
5. J. Grodal et al., *Categories and Topology*, University of Copenhagen lecture notes, Sections 1.1--1.2, 1.8 and 3.2. [Course notes](https://web.math.ku.dk/~jg/papers/cattop.pdf). 这份讲义更进阶，本文只借用其中与链复形、映射和泛性质有关的视角。
6. H. Edelsbrunner and J. Harer, *Computational Topology: An Introduction*, American Mathematical Society, 2010. [AMS](https://bookstore.ams.org/mbk-69/).
7. R. Ghrist, “Barcodes: The Persistent Topology of Data,” *Bulletin of the American Mathematical Society*, 2008. [Project Euclid](https://projecteuclid.org/journals/bulletin-of-the-american-mathematical-society/volume-45/issue-1/Barcodes-The-persistent-topology-of-data/10.1090/S0273-0979-07-01191-3.full).
8. G. Carlsson, “Topology and Data,” *Bulletin of the American Mathematical Society*, 2009. [Project Euclid](https://projecteuclid.org/journals/bulletin-of-the-american-mathematical-society/volume-46/issue-2/Topology-and-data/10.1090/S0273-0979-09-01249-X.full).
9. M. M. Bronstein, J. Bruna, Y. LeCun, A. Szlam, and P. Vandergheynst, “Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges,” *IEEE Signal Processing Magazine*, 2021. [arXiv](https://arxiv.org/abs/2104.13478).
10. G. de Rham, “Sur l'analysis situs des variétés à $n$ dimensions,” *Journal de Mathématiques Pures et Appliquées*, 1931.
11. E. H. Spanier, *Algebraic Topology*, McGraw–Hill, 1966.
12. M. Hajij et al., “Topological Deep Learning: Going Beyond Graph Data,” 2023. [arXiv](https://arxiv.org/abs/2206.00606).
13. H. Whitney, “On regular closed curves on the plane,” *Compositio Mathematica*, 4 (1937), 276–284.
14. H. Geiges, “A contact geometric proof of the Whitney–Graustein Theorem,” 2007. [arXiv](https://arxiv.org/abs/0801.0046).
15. J. Erickson, “Generic Planar Curves,” *Computational Topology* lecture notes, 2023. [Notes](https://jeffe.cs.illinois.edu/teaching/comptop/2023/notes/06-generic-curves.html).
16. D. S. Dummit and R. M. Foote, *Abstract Algebra*, 3rd ed., Wiley, 2004.
17. S. Devalapurkar, *MIT 18.905 Algebraic Topology I*, Fall 2016 lecture notes. [MIT OpenCourseWare](https://ocw.mit.edu/courses/18-905-algebraic-topology-i-fall-2016/), [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

# 附录

如下为正文附录补充。

## A. 为什么边界的边界为零

先对一个有向 $k$-单纯形 $[v_0v_1\ldots v_k]$ 写出边界：

$$
\partial_k[v_0v_1\ldots v_k]
=\sum_{i=0}^{k}(-1)^i
[v_0\ldots\widehat{v_i}\ldots v_k].
$$

帽子表示删掉对应的顶点。再次取边界：

$$
\partial_{k-1}\partial_k[v_0\ldots v_k]
=\sum_{i=0}^{k}(-1)^i
\partial_{k-1}[v_0\ldots\widehat{v_i}\ldots v_k].
$$

固定两个不同的指标 $i<j$。删掉 $v_i$ 后再删掉原来的 $v_j$，在第二次边界中，$v_j$ 在剩余顶点中的位置是 $j-1$，因此这项的符号是

$$
(-1)^i(-1)^{j-1}=(-1)^{i+j-1}.
$$

反过来，先删掉 $v_j$，再删掉 $v_i$。因为 $i<j$，删掉 $v_j$ 不会改变 $v_i$ 的位置，所以符号是

$$
(-1)^j(-1)^i=(-1)^{i+j}.
$$

两项对应同一个 $(k-2)$-单纯形，但符号相反：

$$
(-1)^{i+j-1}+(-1)^{i+j}=0.
$$

在完整的双重求和中，每个删去两个顶点后的单纯形恰好以这两种顺序出现一次。因此所有项两两抵消，得到

$$
\partial_{k-1}\partial_k[v_0\ldots v_k]=0.
$$

边界算子是线性的，所以对任意 $c\in C_k$ 都有

$$
\partial_{k-1}\partial_k c=0.
$$

这就证明了 $\operatorname{im}\partial_{k+1}\subseteq\ker\partial_k$，也就证明了同调商群 $H_k=\ker\partial_k/\operatorname{im}\partial_{k+1}$ 确实定义良好。

## B. 三角剖分圆周的同调群

取三个顶点 $v_0,v_1,v_2$，三条定向边

$$
e_0=[v_0v_1],
\qquad
e_1=[v_1v_2],
\qquad
e_2=[v_2v_0].
$$

没有二维单纯形，因此

$$
C_2=0,
\qquad
B_1=\operatorname{im}\partial_2=0.
$$

### B.1 计算 $H_1$

任意 $1$-链写成

$$
c=ae_0+be_1+ce_2,
\qquad a,b,c\in\mathbb{Z}.
$$

逐条计算边界：

$$
\partial_1e_0=v_1-v_0,
\qquad
\partial_1e_1=v_2-v_1,
\qquad
\partial_1e_2=v_0-v_2.
$$

因此

$$
\begin{aligned}
\partial_1c
&=a(v_1-v_0)+b(v_2-v_1)+c(v_0-v_2)\\\\
&=(-a+c)v_0+(a-b)v_1+(b-c)v_2.
\end{aligned}
$$

$c$ 是循环当且仅当三个系数同时为零：

$$
-a+c=0,
\qquad a-b=0,
\qquad b-c=0.
$$

第一式给出 $c=a$，第二式给出 $b=a$，第三式自动成立。因此

$$
Z_1=\lbrace a(e_0+e_1+e_2):a\in\mathbb{Z}\rbrace
\cong\mathbb{Z}.
$$

又因为 $B_1=0$，所以

$$
H_1=Z_1/B_1\cong\mathbb{Z}.
$$

生成元 $e_0+e_1+e_2$ 正是绕圆周一圈的闭合路径。

### B.2 计算 $H_0$

$0$-链群是

$$
C_0=\lbrace x_0v_0+x_1v_1+x_2v_2:x_0,x_1,x_2\in\mathbb{Z}\rbrace.
$$

因为 $\partial_0=0$，所以 $Z_0=C_0$。边界群由三条边生成：

$$
B_0=\operatorname{span}_{\mathbb{Z}}
\lbrace v_1-v_0,\ v_2-v_1,\ v_0-v_2\rbrace.
$$

第三个生成元等于前两个的负和，因此只需要前两个。对任意 $0$-链，模掉这些差异后，三个顶点的系数只剩下总和：

$$
x_0v_0+x_1v_1+x_2v_2
\equiv (x_0+x_1+x_2)v_0
\pmod{B_0}.
$$

定义

$$
\Phi:C_0\to\mathbb{Z},
\qquad
\Phi(x_0v_0+x_1v_1+x_2v_2)=x_0+x_1+x_2.
$$

它在 $B_0$ 上为零，因此诱导满射

$$
\overline\Phi:H_0=C_0/B_0\to\mathbb{Z}.
$$

若一个 $0$-链的系数总和为零，可以写成

$$
x_0v_0+x_1v_1+x_2v_2
=x_1(v_1-v_0)+x_2(v_2-v_0),
$$

而 $v_2-v_0=(v_2-v_1)+(v_1-v_0)\in B_0$。所以 $\ker\overline\Phi=0$，得到

$$
H_0\cong\mathbb{Z}.
$$

圆周只有一个连通分支，计算结果正好反映了这一点。

## C. 链映射为什么会诱导同调映射

设 $f:X\to Y$ 是保持面关系的单纯映射。它把每个 $k$-单纯形送到 $Y$ 中的一个 $k$-链，因此通过线性延拓得到

$$
f_k:C_k(X)\to C_k(Y).
$$

保持面关系意味着先取边界再映射，与先映射再取边界相同：

$$
\partial_k^Y f_k=f_{k-1}\partial_k^X.
$$

下面逐步检查它把循环送到循环。取 $z\in Z_k(X)$，则 $\partial_k^Xz=0$。于是

$$
\partial_k^Y f_k(z)
=f_{k-1}\partial_k^X(z)
=f_{k-1}(0)=0.
$$

因此 $f_k(z)\in Z_k(Y)$。

再检查它把边界送到边界。取 $b\in B_k(X)$，则存在 $c\in C_{k+1}(X)$ 使得 $b=\partial_{k+1}^Xc$。于是

$$
\begin{aligned}
f_k(b)
&=f_k\partial_{k+1}^X(c)\\\\
&=\partial_{k+1}^Yf_{k+1}(c)\\\\
&\in B_k(Y).
\end{aligned}
$$

因此公式

$$
f_{\ast,k}:H_k(X)\to H_k(Y),
\qquad
f_{\ast,k}([z])=[f_k(z)]
$$

是有意义的。还需要检查代表元的选择不会影响结果。若 $[z]=[z']$，则 $z-z'\in B_k(X)$；刚才已经证明 $f_k(z-z')\in B_k(Y)$，所以

$$
[f_k(z)]=[f_k(z')].
$$

这就证明了单纯映射确实诱导同调映射。恒等映射诱导恒等同调映射，复合映射满足

$$
(g\circ f)_\ast=g_\ast\circ f_\ast.
$$

这两个性质正是函子性的来源。

## D. 一个最小持久同调例子的完整计算

仍取圆周的三个顶点，但按如下过滤逐步加边：

$$
X_0=\lbrace v_0,v_1,v_2\rbrace,
$$

$$
X_1=X_0\cup\lbrace e_0\rbrace,
\qquad
X_2=X_1\cup\lbrace e_1\rbrace,
$$

$$
X_3=X_2\cup\lbrace e_2\rbrace.
$$

为避免方向造成干扰，以下先使用系数域 $\mathbb{F}_2$，此时 $-1=1$。

### D.1 $H_0$ 的变化

在 $X_0$ 中没有边，所以三个顶点各自是一个连通分支：

$$
H_0(X_0)\cong\mathbb{F}_2^3.
$$

加入 $e_0$ 后，$v_0$ 和 $v_1$ 连通。边界

$$
\partial e_0=v_0+v_1
$$

说明在商空间中 $[v_0]=[v_1]$，但 $[v_2]$ 仍独立，所以

$$
H_0(X_1)\cong\mathbb{F}_2^2.
$$

加入 $e_1$ 后，$v_1$ 和 $v_2$ 连通，于是三个顶点代表同一个连通分支：

$$
H_0(X_2)\cong\mathbb{F}_2.
$$

最后加入 $e_2$ 不会再减少连通分支，因此

$$
H_0(X_3)\cong\mathbb{F}_2.
$$

所以三个最初的连通分支中，一个在加入 $e_0$ 时与另一个合并，另一个在加入 $e_1$ 时合并。若把出现时刻记为 $0$，则对应的有限区间可以写成

$$
[0,1),\qquad [0,2),\qquad [0,\infty).
$$

这里的区间端点取决于我们如何编号过滤步骤；重要的是“出生”表示一个新分支出现，“死亡”表示它与更早的分支合并。

### D.2 $H_1$ 的变化

在 $X_0,X_1,X_2$ 中都没有闭合路径。逐项检查可知：

$$
Z_1(X_0)=0,
\qquad
Z_1(X_1)=0,
\qquad
Z_1(X_2)=0.
$$

在 $X_3$ 中，三条边组成

$$
z=e_0+e_1+e_2.
$$

其边界为

$$
\begin{aligned}
\partial z
&=(v_0+v_1)+(v_1+v_2)+(v_2+v_0)\\\\
&=2v_0+2v_1+2v_2\\\\
&=0
\quad\text{in }\mathbb{F}_2.
\end{aligned}
$$

因为没有二维单纯形，$B_1(X_3)=0$，所以 $[z]$ 是一个非零的一维同调类，并且之后没有更高维单纯形把它填平。它给出一个无限持久的区间 $[3,\infty)$。

这个小例子包含 barcode 的全部基本动作：连通分支可以合并，环可以出生，也可以在加入二维单纯形后死亡。一般的持久同调会把同样的过程放到更大的过滤中，并用线性代数追踪每个类的出生和死亡。

### D.3 把区间分解写成边界矩阵消元

把三个顶点和三条边按过滤顺序排列。在 $\mathbb{F}_2$ 上，边界矩阵的三列分别是三条边的端点：

$$
D_1=
\\begin{array}{c|ccc}
 & e_0 & e_1 & e_2\\\\
v_0 & 1 & 0 & 1\\\\
v_1 & 1 & 1 & 0\\\\
v_2 & 0 & 1 & 1
\\end{array}.
$$

消元只允许把一列加到另一列，这不会改变列空间。先保留第一列：

$$
R_0=e_0=(1,1,0)^T.
$$

第二列与第一列的主元位置不同，因此保留为

$$
R_1=e_1=(0,1,1)^T.
$$

第三列逐步相加：

$$
\\begin{aligned}
R_2
&=e_2+R_0\\\\
&=(1,0,1)^T+(1,1,0)^T\\\\
&=(0,1,1)^T\\\\
&=R_1.
\\end{aligned}
$$

再减去（在 $\mathbb{F}_2$ 中就是再相加）$R_1$，得到

$$
R_2+R_1=0.
$$

因此第三列最终成为零列。它表示 $e_2$ 加入时产生了一个新的 $1$-循环，也就是 $H_1$ 中出生于时刻 $3$ 的无限区间。前两列各有主元，说明它们分别把两个新的顶点关系消掉，正对应 $H_0$ 中两个有限区间的死亡。这个逐列过程就是持久同调软件所执行的基本分解步骤；“区间”并不是额外画出来的标签，而是主元列与零列在过滤顺序中的配对结果。

## E. Euler–Poincare 公式的完整分解

设

$$
0\longrightarrow C_n
\xrightarrow{\partial_n}C_{n-1}
\longrightarrow\cdots
\xrightarrow{\partial_1}C_0
\longrightarrow0
$$

是域 $\mathbb{F}$ 上的有限维链复形。仍记

$$
Z_k=\ker\partial_k,
\qquad
B_k=\operatorname{im}\partial_{k+1},
\qquad
H_k=Z_k/B_k.
$$

先把每个链群 $C_k$ 分解。线性映射 $\partial_k:C_k\to C_{k-1}$ 的核是 $Z_k$，像是 $B_{k-1}$。秩–零度定理逐项给出

$$
\dim C_k
=\dim\ker\partial_k
+\dim\operatorname{im}\partial_k
=\dim Z_k+\dim B_{k-1}.
$$

再分解循环空间。因为 $B_k\subseteq Z_k$，有短正合列

$$
0\longrightarrow B_k
\longrightarrow Z_k
\longrightarrow H_k
\longrightarrow0.
$$

有限维向量空间的短正合列中，中间项的维数等于两端维数之和，因此

$$
\dim Z_k=\dim B_k+\dim H_k.
$$

代回第一步：

$$
\dim C_k
=\dim H_k
+\dim B_k
+\dim B_{k-1}.
$$

对所有 $k$ 取交替和：

$$
\begin{aligned}
\sum_k(-1)^k\dim C_k
&=\sum_k(-1)^k\dim H_k\\\\
&\quad+\sum_k(-1)^k\dim B_k\\\\
&\quad+\sum_k(-1)^k\dim B_{k-1}.
\end{aligned}
$$

把最后一项换指标 $j=k-1$：

$$
\begin{aligned}
\sum_k(-1)^k\dim B_{k-1}
&=\sum_j(-1)^{j+1}\dim B_j\\\\
&=-\sum_j(-1)^j\dim B_j.
\end{aligned}
$$

它与前一个边界空间的交替和恰好抵消，于是只剩下

$$
\sum_k(-1)^k\dim C_k
=\sum_k(-1)^k\dim H_k.
$$

对有限单纯复形，$\dim C_k$ 就是 $k$ 维单纯形的个数，$\dim H_k=b_k$。因此

$$
\chi
=\sum_k(-1)^k\dim C_k
=\sum_k(-1)^kb_k.
$$

这也精确解释了“局部剖分信息为什么会消失”：每个边界空间 $B_k$ 在相邻两项中各出现一次，符号相反，所以在交替和中被完全消去。

## F. cup product 为什么能定义在上同调类上

设 $\alpha\in C^p(X;R)$、$\beta\in C^q(X;R)$。关键是余边界对 cup product 满足分次 Leibniz 公式

$$
\delta(\alpha\smile\beta)
=(\delta\alpha)\smile\beta
+(-1)^p\alpha\smile(\delta\beta).
$$

下面把符号逐项展开。取一个 $(p+q+1)$-单纯形

$$
\sigma=[v_0\ldots v_{p+q+1}].
$$

从左侧开始：

$$
\delta(\alpha\smile\beta)(\sigma)
=\sum_{i=0}^{p+q+1}(-1)^i
(\alpha\smile\beta)
[v_0\ldots\widehat{v_i}\ldots v_{p+q+1}].
$$

当 $0\leq i\leq p$ 时，被删去的顶点位于前半段，所以对应项为

$$
(-1)^i
\alpha[v_0\ldots\widehat{v_i}\ldots v_{p+1}]
\beta[v_{p+1}\ldots v_{p+q+1}].
$$

这些正是 $((\delta\alpha)\smile\beta)(\sigma)$ 中除去 $i=p+1$ 以外的各项。缺少的那一项是

$$
(-1)^{p+1}
\alpha[v_0\ldots v_p]
\beta[v_{p+1}\ldots v_{p+q+1}].
$$

当 $p+1\leq i\leq p+q+1$ 时，被删去的顶点位于后半段，原和式中的对应项为

$$
(-1)^i
\alpha[v_0\ldots v_p]
\beta[v_p\ldots\widehat{v_i}\ldots v_{p+q+1}].
$$

令后半段中的局部指标为 $j=i-p$。因为

$$
(-1)^p(-1)^j=(-1)^p(-1)^{i-p}=(-1)^i,
$$

这些正是 $(-1)^p(\alpha\smile\delta\beta)(\sigma)$ 中除去 $j=0$ 以外的各项。缺少的 $j=0$ 项是

$$
(-1)^p
\alpha[v_0\ldots v_p]
\beta[v_{p+1}\ldots v_{p+q+1}].
$$

两个缺项的系数相加为

$$
(-1)^{p+1}+(-1)^p=0.
$$

所以它们互相抵消，Leibniz 公式成立。

现在若 $\delta\alpha=0$ 且 $\delta\beta=0$，则

$$
\delta(\alpha\smile\beta)=0,
$$

所以两个 cocycle 的 cup product 仍是 cocycle。还要检查改变代表元不会改变上同调类。若把 $\alpha$ 换成 $\alpha+\delta\mu$，其中 $\mu\in C^{p-1}(X;R)$，则

$$
\begin{aligned}
(\alpha+\delta\mu)\smile\beta-\alpha\smile\beta
&=(\delta\mu)\smile\beta\\\\
&=\delta(\mu\smile\beta)
-(-1)^{p-1}\mu\smile\delta\beta\\\\
&=\delta(\mu\smile\beta).
\end{aligned}
$$

差是一个 coboundary。类似地，若把 $\beta$ 换成 $\beta+\delta\nu$，因为 $\delta\alpha=0$，

$$
\begin{aligned}
\alpha\smile(\beta+\delta\nu)-\alpha\smile\beta
&=\alpha\smile\delta\nu\\\\
&=(-1)^p\delta(\alpha\smile\nu).
\end{aligned}
$$

差仍是一个 coboundary。因此 $[\alpha\smile\beta]$ 只依赖于 $[\alpha]$ 和 $[\beta]$，cup product 确实诱导出

$$
H^p(X;R)\times H^q(X;R)
\longrightarrow H^{p+q}(X;R).
$$

## G. 群、环与域：代数拓扑中的代数背景

这一节先把记号说清楚。字母 $G$ 表示一个群，字母 $R$ 表示一个环，字母 $F$ 表示一个域，字母 $M$ 表示一个模；$K$ 表示一个有限单纯复形，$k$ 表示非负整数维数。符号 $\mathbb{Z}$、$\mathbb{Q}$、$\mathbb{R}$、$\mathbb{C}$ 分别表示整数、有理数、实数和复数。

### G.1 群：只有一个运算的对称结构

一个群是一个集合 $G$ 和一个二元运算 $\cdot$。这里 $a,b,c$ 表示 $G$ 中任意三个元素。群公理要求：

1. 封闭性：$a\cdot b$ 仍属于 $G$；
2. 结合律：$(a\cdot b)\cdot c=a\cdot(b\cdot c)$；
3. 单位元：存在元素 $e\in G$，使 $e\cdot a=a\cdot e=a$；
4. 逆元：每个 $a\in G$ 都有元素 $a^{-1}\in G$，使 $a\cdot a^{-1}=a^{-1}\cdot a=e$。

其中 $e$ 是单位元，$a^{-1}$ 是 $a$ 的逆元。如果还满足 $a\cdot b=b\cdot a$，就称为阿贝尔群。整数在加法下是阿贝尔群；正整数 $n$ 个元素的置换在复合下构成对称群 $S_n$，但当 $n\geq3$ 时通常不是阿贝尔群。

代数拓扑中的链群就是阿贝尔群。若 $K$ 是有限单纯复形，$K_k$ 表示所有 $k$-单纯形组成的集合，$\sigma$ 表示其中一个 $k$-单纯形，则

$$
C_k(K;\mathbb{Z})
=\bigoplus_{\sigma\in K_k}\mathbb{Z}\,\sigma.
$$

符号 $\bigoplus$ 表示“有限形式和”：每个 $k$-链只含有限多个非零系数；$\mathbb{Z}\,\sigma$ 表示 $\sigma$ 的整数倍组成的自由阿贝尔群。于是 $C_k(K;\mathbb{Z})$ 的元素可以写成

$$
c=\sum_{\sigma\in K_k}m_\sigma\,\sigma,
\qquad m_\sigma\in\mathbb{Z}.
$$

这里 $c$ 是一条 $k$-链，$m_\sigma$ 是单纯形 $\sigma$ 的整数系数。边界算子 $\partial_k$ 是群同态：

$$
\partial_k:C_k(K;\mathbb{Z})
\longrightarrow C_{k-1}(K;\mathbb{Z}).
$$

因此 $Z_k=\ker\partial_k$ 和 $B_k=\operatorname{im}\partial_{k+1}$ 都是阿贝尔子群，同调群

$$
H_k(K;\mathbb{Z})=Z_k/B_k
$$

就是它们的商群。符号 $\ker$ 表示核，$\operatorname{im}$ 表示像，斜杠表示商群。

### G.2 环：允许系数相乘的结构

一个环 $R$ 首先是一个阿贝尔群；它还有一个乘法，满足结合律和分配律。这里 $r,s,t$ 表示 $R$ 中任意元素，$0$ 表示加法单位元，$1$ 表示乘法单位元。分配律写成

$$
r(s+t)=rs+rt,
\qquad
(r+s)t=rt+st.
$$

整数环 $\mathbb{Z}$、剩余类环 $\mathbb{Z}/n\mathbb{Z}$ 和实系数多项式环 $\mathbb{R}[x]$ 都是环。符号 $\mathbb{Z}/n\mathbb{Z}$ 表示把相差 $n$ 的整数视为同一个剩余类；$\mathbb{R}[x]$ 表示变量 $x$ 的实系数多项式集合。

环 $R$ 的标量可以作用在另一个阿贝尔群 $M$ 上，这样的对象叫作 $R$-模。若 $m_1,m_2\in M$、$r_1,r_2\in R$，模的运算满足

$$
r(m_1+m_2)=rm_1+rm_2,
\qquad
(r_1+r_2)m=r_1m+r_2m,
\qquad
(r_1r_2)m=r_1(r_2m).
$$

将链的整数系数换成环 $R$ 中的系数，就得到

$$
C_k(K;R)
=\bigoplus_{\sigma\in K_k}R\,\sigma.
$$

此时边界算子对 $R$-系数线性，因而

$$
H_k(K;R)
=\ker\partial_k/\operatorname{im}\partial_{k+1}
$$

是一个 $R$-模。也就是说，系数环决定了链可以用哪些数相加，以及边界关系中的整数如何被解释。

群与环还可以合成群环。给定群 $G$ 和环 $R$，群环 $R[G]$ 的元素是有限形式和

$$
\sum_{g\in G}a_g\,g,
\qquad a_g\in R.
$$

这里的 $g$ 是群元素，$a_g$ 是它对应的环系数；“有限”表示只有有限多个 $a_g$ 不为零。若 $h$ 也是群元素，$b_h$ 表示 $h$ 对应的环系数，乘法由群中的乘法 $gh$ 和环中的乘法 $a_g b_h$ 共同决定：

$$
\left(\sum_{g\in G}a_g g\right)
\left(\sum_{h\in G}b_h h\right)
=\sum_{g,h\in G}a_g b_h(gh).
$$

群环把“对称性”和“线性组合”放进同一个代数对象，在基本群的表示、覆盖空间和同调代数中都会出现。

### G.3 域：每个非零元素都可以除

域 $F$ 是一个交换的单位环，并且每个非零元素都有乘法逆元。符号 $F\setminus\{0\}$ 表示从 $F$ 中删去零元素；要求

$$
(F\setminus\{0\},\cdot)
$$

在乘法下构成阿贝尔群。$\mathbb{Q}$、$\mathbb{R}$ 和 $\mathbb{C}$ 都是域；$\mathbb{Z}$ 不是域，因为 $2$ 没有整数逆元。

对正整数 $n$，记 $[a]$ 为整数 $a$ 在 $\mathbb{Z}/n\mathbb{Z}$ 中的剩余类。下面证明

$$
\mathbb{Z}/n\mathbb{Z}\text{ 是域}
\quad\Longleftrightarrow\quad
n\text{ 是素数}.
$$

若 $n$ 是合数，可以写成 $n=ab$，其中 $a$ 和 $b$ 都严格位于 $1$ 与 $n$ 之间。于是

$$
[a]\neq[0],
\qquad
[b]\neq[0],
\qquad
[a][b]=[ab]=[0].
$$

存在两个非零元素相乘得到零，说明有零因子，所以 $\mathbb{Z}/n\mathbb{Z}$ 不是域。

反过来，令 $n=p$ 为素数。取任意非零剩余类 $[a]\neq[0]$，则 $p$ 不整除 $a$，所以最大公因数 $\gcd(a,p)=1$。根据 Bezout 恒等式，存在整数 $u,v$ 使

$$
ua+vp=1.
$$

对这个等式模 $p$ 取剩余类，得到

$$
[u][a]+[v][p]=[1].
$$

由于 $[p]=[0]$，上式化为

$$
[u][a]=[1].
$$

因此 $[u]$ 是 $[a]$ 的乘法逆元。每个非零剩余类都有逆元，故 $\mathbb{Z}/p\mathbb{Z}$ 是域，通常记为 $\mathbb{F}_p$。这里 $\gcd$ 表示最大公因数，$\mathbb{F}_p$ 表示含有 $p$ 个元素的有限域。

域上的模就是向量空间。向量空间可以使用维数、秩、矩阵主元等线性代数工具；这正是持久同调常在有限域上计算的原因。一般环上的模未必具有这些性质，所以“取域系数”不只是记号变化，而是改变了可用的计算工具。

### G.4 系数为什么会改变同调

设 $X$ 是一个拓扑空间。对同一个 $X$，可以分别计算

$$
H_k(X;\mathbb{Z}),
\qquad
H_k(X;\mathbb{F}_2),
\qquad
H_k(X;\mathbb{R}).
$$

这里 $\mathbb{F}_2$ 是含有两个元素的有限域；$H_k(X;R)$ 表示用系数环 $R$ 构造的第 $k$ 个同调模。

令 $S^1$ 表示平面中的单位圆，即满足 $x^2+y^2=1$ 的点集，其中 $x,y$ 是实坐标。整数系数给出

$$
H_0(S^1;\mathbb{Z})\cong\mathbb{Z},
\qquad
H_1(S^1;\mathbb{Z})\cong\mathbb{Z}.
$$

符号 $\cong$ 表示同构，也就是存在保持代数运算的双射。换成 $\mathbb{F}_p$ 系数后，圆周的同调变为

$$
H_0(S^1;\mathbb{F}_p)\cong\mathbb{F}_p,
\qquad
H_1(S^1;\mathbb{F}_p)\cong\mathbb{F}_p.
$$

圆周仍有一个连通分支和一个一维环，但整数“绕行多少次”的信息被压缩成模 $p$ 的信息。整数系数可以保留扭结，例如实射影平面 $\mathbb{R}P^2$ 的第一同调满足

$$
H_1(\mathbb{R}P^2;\mathbb{Z})
\cong\mathbb{Z}/2\mathbb{Z}.
$$

这里 $\mathbb{R}P^2$ 是三维实向量空间 $\mathbb{R}^3$ 中所有一维子空间组成的空间；等价地，它把球面上互为相反点的两个点视为同一点。这个 $\mathbb{Z}/2\mathbb{Z}$ 正是整数系数下可见的 $2$-扭结；改用 $\mathbb{F}_2$ 时，它表现为一个一维向量空间 $\mathbb{F}_2$。

因此，系数环不是计算时随手选择的参数。它决定方向、倍数和扭结哪些会被保留下来，也决定我们能否使用向量空间的线性代数。代数拓扑中从 $\mathbb{Z}$ 切换到有限域，实质上是在改变观察空间的代数镜头。
