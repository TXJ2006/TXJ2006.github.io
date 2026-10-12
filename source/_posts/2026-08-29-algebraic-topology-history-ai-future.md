---
title: "从 Analysis Situs 到可学习不变量：代数拓扑的数学基础与 AI 时代"
date: 2026-08-29 22:00:00
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

代数拓扑最初并不是为了给数据寻找特征，也不是为了服务于机器学习。它诞生于一个更古老的问题：当一个空间被连续地拉伸、弯曲，却没有被撕裂或粘合时，究竟什么东西保持不变？Poincare 在十九世纪末提出这个问题时，现代计算机尚不存在；一个多世纪之后，人工智能又让这个问题获得了新的应用场景，但问题本身仍然是数学的。

本文从 Poincare 的 *Analysis Situs* 讲起，沿着同调、同伦、范畴和持续同调的路线，说明代数拓扑怎样把连续的形状转成可计算的代数对象。最后再回到 AI，讨论拓扑结构能提供什么数学约束，以及它不能替代什么。

<!--more-->

## 1. Poincare 的问题：如何描述一个会变形的空间

如果只问一个图形有多长、多大、弯曲得多厉害，微积分和微分几何已经有一套成熟语言。但有些性质与长度和角度无关：一个圆环中间有一个洞，正方形的边界围出一个洞；把它们拉伸，洞仍然在那里。另一方面，如果“弯曲”始终指保持嵌入的连续变形，一条线段无论弯曲多大，也不会变成一个圆。

### 1.1 自交并不是反例，而是换了研究对象

这里容易混淆三个不同的对象：线段作为抽象空间、线段在平面中的嵌入，以及一条参数曲线的像。若

$$
\gamma:[0,1]\to\mathbb{R}^2
$$

始终是单射，那么它的像 $X=\gamma([0,1])$ 与区间 $[0,1]$ 同胚，因而

$$
H&#95;1(X)=0.
$$

这不是说线段不能画出复杂图案，而是说只要不允许不同参数点落到同一个位置，线段的拓扑类型就没有改变。曲率是微分几何量，并不存在一个“曲率足够大”就自动产生拓扑洞的阈值；真正改变对象的是端点识别或自交。端点一旦粘合，得到的是圆周；若存在 $s\neq t$ 使 $\gamma(s)=\gamma(t)$，$\gamma$ 就不再是嵌入，像空间可能变成带环的平面图。

最典型的例子是“8”字形。把交点看成一个顶点、把两个圆瓣看成两条边，得到一个图 $X$，其中

$$
V=1,\qquad E=2,
$$

所以

$$
\beta&#95;1(X)=E-V+1=2.
$$

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
\beta&#95;1=E-V+1=n+1.
$$

在这个平面图情形下，“有多少个独立的环”和“补空间有多少个有界区域”恰好相等。Alexander 对偶性把这种现象写成

$$
\widetilde H&#95;0(S^2\setminus X;\mathbb{F})
\cong
\widetilde H^1(X;\mathbb{F}),
$$

其中 $\mathbb{F}$ 是系数域。右侧描述曲线自身的一维上同调，左侧描述从球面中删去曲线后剩下的连通分支。因此，你说的“弯曲后交错并形成空洞”，在数学上确实存在，只是它研究的是自交曲线的像及其补空间，而不是始终保持嵌入的线段。

这类对象早已有系统理论。Jordan 曲线定理研究无自交闭曲线如何把平面分成内外两部分；Whitney–Graustein 定理研究允许自交的正则闭曲线，并证明它们在正则同伦下由旋转数分类；计算拓扑中还会把自交曲线编码成 image graph、face 和 Gauss code。若曲线实际位于三维空间，平面图上的交叉还可能只是投影中的交叉，这又进入结理论和 Reidemeister moves。

因此需要先决定什么叫“同一种形状”。代数拓扑通常不要求两个空间保持距离，只要求它们之间存在连续的双射及连续的逆映射，这叫**同胚**。在很多问题中，还会进一步放宽到**同伦等价**：允许一个连续映射逐渐变形成另一个映射。一个圆盘可以收缩到一个点，所以它与一点同伦等价；圆周却不能这样收缩，因为圆周中间的洞阻止了这个过程。

Poincare 在 1895 年发表 *Analysis Situs*，提出了一个影响深远的方向：不要直接追踪空间的每一个点，而是为整个空间构造一些在连续变形下保持不变的量。这个想法把问题从“画出空间”转成了“提取空间的代数指纹”。

最早的指纹之一是 Euler 示性数。对一个有限三角剖分，记顶点、边和三角形的个数分别为 $V,E,F$，定义

$$
\chi=V-E+F.
$$

一个球面的三角剖分可以有不同数量的顶点和边，但 $\chi$ 总是 $2$；环面的 Euler 示性数则是 $0$。这已经说明，计数并不只是计数：交替相减后，局部剖分的细节会被抵消，剩下的是整体形状的一部分。

但 Euler 示性数仍然过于粗糙。球面和某些其他空间可能有相同的 $\chi$，却并不具有相同的拓扑结构。我们需要的不只是一个数，而是一组能够记录不同维度洞的代数对象。这就引出了同调。

## 2. 同调：把“洞”变成可以计算的群

把空间剖分成点、边、三角形、四面体等单纯形。所有 $k$ 维单纯形的整数线性组合组成链群，记为 $C&#95;k$。例如，一个三角形 $[v&#95;0v&#95;1v&#95;2]$ 的边界是三条有向边的交替和：

$$
\partial&#95;2[v&#95;0v&#95;1v&#95;2]
=[v&#95;1v&#95;2]-[v&#95;0v&#95;2]+[v&#95;0v&#95;1].
$$

边界算子 $\partial&#95;k:C&#95;k\to C&#95;{k-1}$ 把一个 $k$ 维对象送到它的边界。关键事实是

$$
\partial&#95;{k-1}\partial&#95;k=0.
$$

直观地说，边界本身没有边界：一条线段的边界是两个端点，而端点再没有边界；一个三角形的三条边再取边界，每个顶点会出现两次，方向相反，最后全部抵消。这个事实不是记号游戏，它使下面的商群有意义。

定义 $k$-循环群和 $k$-边界群：

$$
Z&#95;k:=\ker\partial&#95;k,
\qquad
B&#95;k:=\operatorname{im}\partial&#95;{k+1}.
$$

因为 $\partial&#95;k\partial&#95;{k+1}=0$，每个边界都是循环，因此 $B&#95;k\subseteq Z&#95;k$。第 $k$ 个同调群定义为

$$
H&#95;k:=Z&#95;k/B&#95;k.
$$

这个商的含义是：先收集所有“没有边界的闭合对象”，再把那些本身就是某个更高维对象边界的闭合对象视为零。一个圆周上的闭合路径不是某个二维区域的边界，所以它代表一个非平凡的 $H&#95;1$ 类；三角形的边界虽然也是闭合路径，却在 $H&#95;1$ 中被当作零。

如果把整数系数换成有限域 $\mathbb{F}&#95;2$，方向和负号会消失，计算会更简单；如果保留整数系数，则可以看到扭结等更细的现象。系数的选择不是技术细节，它决定我们允许观察哪些结构。

## 3. 从 Betti 数到同伦：代数拓扑逐渐长出自己的语言

同调群通常可能很复杂，但在有限生成的情形，可以用自由部分的秩来定义 Betti 数：

$$
b&#95;k=\operatorname{rank}H&#95;k.
$$

$b&#95;0$ 记录连通分支的数量，$b&#95;1$ 可以记录独立的一维洞，$b&#95;2$ 可以记录空腔。对有限复形，Euler 示性数可以写成 Betti 数的交替和：

$$
\chi=\sum&#95;{k\geq0}(-1)^kb&#95;k.
$$

这条公式解释了上一节的 $V-E+F$ 为什么能抵抗三角剖分的改变：链群的维数交替和，等于同调群维数的交替和；边界映射带来的部分在相邻维度中正好抵消。

但同调并没有回答所有问题。两个空间可能有相同的同调群，却有不同的基本群；一个空间的局部环路如何相乘，不能仅靠“有几个洞”描述。于是人们研究基本群 $\pi&#95;1$、高阶同伦群和上同调。与同调相比，同伦群通常更难计算，却保留了更多关于连续映射的信息。

二十世纪上半叶，代数拓扑的语言从“为每个空间指定一个群”逐渐转向“空间之间的映射也必须被记录”。一个连续映射 $f:X\to Y$ 会诱导同调群之间的映射

$$
f&#95;*:H&#95;k(X)\to H&#95;k(Y).
$$

如果 $f$ 和 $g$ 同伦，那么它们诱导的同调映射相同。这使 $H&#95;k$ 不只是一个数字计算器，而是一个从拓扑空间到代数对象的函子：空间的结构通过映射被传递到代数中。

Eilenberg 与 Mac Lane 在 1945 年关于自然变换的工作，把这种“映射之间的映射”提升为范畴论语言。范畴论有时会让初学者觉得过于抽象，但它解决了一个实在的问题：当我们有很多种不变量、很多种构造时，怎样区分自然出现的关系和人为拼接的公式？

从这个角度看，代数拓扑的历史并不是从几何跳到代数后就结束了，而是经历了三次扩展：先记录空间，再记录空间之间的映射，最后记录不同构造之间的兼容关系。

## 4. 现代观点：过滤、持久同调与结构定理

经典代数拓扑通常从一个空间出发；持久同调则从一串彼此包含的空间出发。设

$$
X&#95;0\subseteq X&#95;1\subseteq\cdots\subseteq X&#95;m
$$

是有限过滤。每个包含映射都会诱导同调映射，于是对固定的 $k$ 得到一个线性对象

$$
H&#95;k(X&#95;0)\longrightarrow H&#95;k(X&#95;1)
\longrightarrow\cdots\longrightarrow H&#95;k(X&#95;m).
$$

若系数取在域 $\mathbb{F}$ 上，每个 $H&#95;k(X&#95;i)$ 都是有限维向量空间。对 $i\leq j$，记复合映射的像的维数为

$$
\rho&#95;k(i,j)
:=\dim\operatorname{im}
\left(H&#95;k(X&#95;i)\to H&#95;k(X&#95;j)\right).
$$

这个秩函数记录一个同调类在过滤中能存活多久。在线性代数意义下，有限型持久模可以分解成区间模的直和：每一个区间表示一个类从某一步出生，在另一步死亡，或一直存活到过滤末端。把所有区间画在同一条数轴上，就是 barcode。

对单纯复形而言，这个分解可以通过边界矩阵的列消元得到。按过滤顺序排列所有单纯形，把每个 $k$-单纯形的边界写成一列；在 $\mathbb{F}&#95;2$ 上，列加法不改变生成的边界空间。消元时出现的主元把一个较低维的单纯形和一个较高维的单纯形配对，配对位置正好记录同调类的出生与死亡。Edelsbrunner 与 Harer 的 *Computational Topology* 系统整理了这一计算观点，Ghrist 的 barcode 文章则给出了简洁的几何解释。

持久同调的数学价值不在于把任意点集都变成漂亮图形，而在于它把“尺度”作为结构的一部分。若过滤来自一个函数 $f:X\to\mathbb{R}$ 的次水平集

$$
X&#95;a=f^{-1}((-\infty,a]),
$$

那么 $a$ 变化时的同调记录了函数的全局形状。在函数满足通常的 tame 条件、过滤有限且系数取在同一个域时，稳定性定理给出

$$
d&#95;B\bigl(D(f),D(g)\bigr)
\leq\lVert f-g\rVert&#95;\infty,
$$

其中左侧是 persistence diagram 的 bottleneck 距离，右侧是函数的上确界距离。这个不等式说明：函数只发生小扰动时，拓扑摘要不会任意跳变；它是持久同调能够成为数学工具，而不只是可视化技巧的原因。

## 5. 上同调、乘法结构与对偶性

同调用链来研究“边界为零”的对象。上同调则把链看成输入，把数值函数放在链上。给定交换系数环 $R$，定义 $k$-上链群

$$
C^k(X;R):=\operatorname{Hom}(C&#95;k(X),R).
$$

若 $\varphi\in C^k(X;R)$，定义余边界算子

$$
\delta^k\varphi:=\varphi\circ\partial&#95;{k+1}.
$$

由于 $\partial&#95;k\partial&#95;{k+1}=0$，有

$$
\delta^{k+1}\delta^k\varphi
=\varphi\circ\partial&#95;{k+1}\circ\partial&#95;{k+2}=0.
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
(\alpha\smile\beta)[v&#95;0\ldots v&#95;{p+q}]
=\alpha[v&#95;0\ldots v&#95;p]
\,\beta[v&#95;p\ldots v&#95;{p+q}].
$$

它在上同调类上给出

$$
H^p(X;R)\times H^q(X;R)
\longrightarrow H^{p+q}(X;R),
$$

使得 $H^*(X;R)$ 不只是各个群的列表，而是一个分次环。这个乘法能区分一些同调群相同、但整体拓扑结构不同的空间。

对闭的、可定向的 $n$ 维流形 $M$，Poincare 对偶性给出

$$
H^k(M;R)\cong H&#95;{n-k}(M;R),
$$

在合适的系数条件下成立。它把“$k$ 维的上同调信息”和“$n-k$ 维的几何洞”联系起来。对光滑流形，de Rham 定理又把微分形式构成的上同调与奇异上同调联系起来：

$$
H&#95;{\mathrm{dR}}^k(M)
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
4. A. Hatcher, *Algebraic Topology*, Cambridge University Press, 2002. [Author's site](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf).
5. H. Edelsbrunner and J. Harer, *Computational Topology: An Introduction*, American Mathematical Society, 2010. [AMS](https://bookstore.ams.org/mbk-69/).
6. R. Ghrist, “Barcodes: The Persistent Topology of Data,” *Bulletin of the American Mathematical Society*, 2008. [Project Euclid](https://projecteuclid.org/journals/bulletin-of-the-american-mathematical-society/volume-45/issue-1/Barcodes-The-persistent-topology-of-data/10.1090/S0273-0979-07-01191-3.full).
7. G. Carlsson, “Topology and Data,” *Bulletin of the American Mathematical Society*, 2009. [Project Euclid](https://projecteuclid.org/journals/bulletin-of-the-american-mathematical-society/volume-46/issue-2/Topology-and-data/10.1090/S0273-0979-09-01249-X.full).
8. M. M. Bronstein, J. Bruna, Y. LeCun, A. Szlam, and P. Vandergheynst, “Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges,” *IEEE Signal Processing Magazine*, 2021. [arXiv](https://arxiv.org/abs/2104.13478).
9. G. de Rham, “Sur l'analysis situs des variétés à $n$ dimensions,” *Journal de Mathématiques Pures et Appliquées*, 1931.
10. E. H. Spanier, *Algebraic Topology*, McGraw–Hill, 1966.
11. M. Hajij et al., “Topological Deep Learning: Going Beyond Graph Data,” 2023. [arXiv](https://arxiv.org/abs/2206.00606).
12. H. Whitney, “On regular closed curves on the plane,” *Compositio Mathematica*, 4 (1937), 276–284.
13. H. Geiges, “A contact geometric proof of the Whitney–Graustein Theorem,” 2007. [arXiv](https://arxiv.org/abs/0801.0046).
14. J. Erickson, “Generic Planar Curves,” *Computational Topology* lecture notes, 2023. [Notes](https://jeffe.cs.illinois.edu/teaching/comptop/2023/notes/06-generic-curves.html).
15. D. S. Dummit and R. M. Foote, *Abstract Algebra*, 3rd ed., Wiley, 2004.

---

# 附录

如下为正文附录补充。

## A. 为什么边界的边界为零

先对一个有向 $k$-单纯形 $[v&#95;0v&#95;1\ldots v&#95;k]$ 写出边界：

$$
\partial&#95;k[v&#95;0v&#95;1\ldots v&#95;k]
=\sum&#95;{i=0}^{k}(-1)^i
[v&#95;0\ldots\widehat{v&#95;i}\ldots v&#95;k].
$$

帽子表示删掉对应的顶点。再次取边界：

$$
\partial&#95;{k-1}\partial&#95;k[v&#95;0\ldots v&#95;k]
=\sum&#95;{i=0}^{k}(-1)^i
\partial&#95;{k-1}[v&#95;0\ldots\widehat{v&#95;i}\ldots v&#95;k].
$$

固定两个不同的指标 $i<j$。删掉 $v&#95;i$ 后再删掉原来的 $v&#95;j$，在第二次边界中，$v&#95;j$ 在剩余顶点中的位置是 $j-1$，因此这项的符号是

$$
(-1)^i(-1)^{j-1}=(-1)^{i+j-1}.
$$

反过来，先删掉 $v&#95;j$，再删掉 $v&#95;i$。因为 $i<j$，删掉 $v&#95;j$ 不会改变 $v&#95;i$ 的位置，所以符号是

$$
(-1)^j(-1)^i=(-1)^{i+j}.
$$

两项对应同一个 $(k-2)$-单纯形，但符号相反：

$$
(-1)^{i+j-1}+(-1)^{i+j}=0.
$$

在完整的双重求和中，每个删去两个顶点后的单纯形恰好以这两种顺序出现一次。因此所有项两两抵消，得到

$$
\partial&#95;{k-1}\partial&#95;k[v&#95;0\ldots v&#95;k]=0.
$$

边界算子是线性的，所以对任意 $c\in C&#95;k$ 都有

$$
\partial&#95;{k-1}\partial&#95;k c=0.
$$

这就证明了 $\operatorname{im}\partial&#95;{k+1}\subseteq\ker\partial&#95;k$，也就证明了同调商群 $H&#95;k=\ker\partial&#95;k/\operatorname{im}\partial&#95;{k+1}$ 确实定义良好。

## B. 三角剖分圆周的同调群

取三个顶点 $v&#95;0,v&#95;1,v&#95;2$，三条定向边

$$
e&#95;0=[v&#95;0v&#95;1],
\qquad
e&#95;1=[v&#95;1v&#95;2],
\qquad
e&#95;2=[v&#95;2v&#95;0].
$$

没有二维单纯形，因此

$$
C&#95;2=0,
\qquad
B&#95;1=\operatorname{im}\partial&#95;2=0.
$$

### B.1 计算 $H&#95;1$

任意 $1$-链写成

$$
c=ae&#95;0+be&#95;1+ce&#95;2,
\qquad a,b,c\in\mathbb{Z}.
$$

逐条计算边界：

$$
\partial&#95;1e&#95;0=v&#95;1-v&#95;0,
\qquad
\partial&#95;1e&#95;1=v&#95;2-v&#95;1,
\qquad
\partial&#95;1e&#95;2=v&#95;0-v&#95;2.
$$

因此

$$
\begin{aligned}
\partial&#95;1c
&=a(v&#95;1-v&#95;0)+b(v&#95;2-v&#95;1)+c(v&#95;0-v&#95;2)\\\\
&=(-a+c)v&#95;0+(a-b)v&#95;1+(b-c)v&#95;2.
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
Z&#95;1=\lbrace a(e&#95;0+e&#95;1+e&#95;2):a\in\mathbb{Z}\rbrace
\cong\mathbb{Z}.
$$

又因为 $B&#95;1=0$，所以

$$
H&#95;1=Z&#95;1/B&#95;1\cong\mathbb{Z}.
$$

生成元 $e&#95;0+e&#95;1+e&#95;2$ 正是绕圆周一圈的闭合路径。

### B.2 计算 $H&#95;0$

$0$-链群是

$$
C&#95;0=\lbrace x&#95;0v&#95;0+x&#95;1v&#95;1+x&#95;2v&#95;2:x&#95;0,x&#95;1,x&#95;2\in\mathbb{Z}\rbrace.
$$

因为 $\partial&#95;0=0$，所以 $Z&#95;0=C&#95;0$。边界群由三条边生成：

$$
B&#95;0=\operatorname{span}_{\mathbb{Z}}
\lbrace v&#95;1-v&#95;0,\ v&#95;2-v&#95;1,\ v&#95;0-v&#95;2\rbrace.
$$

第三个生成元等于前两个的负和，因此只需要前两个。对任意 $0$-链，模掉这些差异后，三个顶点的系数只剩下总和：

$$
x&#95;0v&#95;0+x&#95;1v&#95;1+x&#95;2v&#95;2
\equiv (x&#95;0+x&#95;1+x&#95;2)v&#95;0
\pmod{B&#95;0}.
$$

定义

$$
\Phi:C&#95;0\to\mathbb{Z},
\qquad
\Phi(x&#95;0v&#95;0+x&#95;1v&#95;1+x&#95;2v&#95;2)=x&#95;0+x&#95;1+x&#95;2.
$$

它在 $B&#95;0$ 上为零，因此诱导满射

$$
\overline\Phi:H&#95;0=C&#95;0/B&#95;0\to\mathbb{Z}.
$$

若一个 $0$-链的系数总和为零，可以写成

$$
x&#95;0v&#95;0+x&#95;1v&#95;1+x&#95;2v&#95;2
=x&#95;1(v&#95;1-v&#95;0)+x&#95;2(v&#95;2-v&#95;0),
$$

而 $v&#95;2-v&#95;0=(v&#95;2-v&#95;1)+(v&#95;1-v&#95;0)\in B&#95;0$。所以 $\ker\overline\Phi=0$，得到

$$
H&#95;0\cong\mathbb{Z}.
$$

圆周只有一个连通分支，计算结果正好反映了这一点。

## C. 链映射为什么会诱导同调映射

设 $f:X\to Y$ 是保持面关系的单纯映射。它把每个 $k$-单纯形送到 $Y$ 中的一个 $k$-链，因此通过线性延拓得到

$$
f&#95;k:C&#95;k(X)\to C&#95;k(Y).
$$

保持面关系意味着先取边界再映射，与先映射再取边界相同：

$$
\partial&#95;k^Y f&#95;k=f&#95;{k-1}\partial&#95;k^X.
$$

下面逐步检查它把循环送到循环。取 $z\in Z&#95;k(X)$，则 $\partial&#95;k^Xz=0$。于是

$$
\partial&#95;k^Y f&#95;k(z)
=f&#95;{k-1}\partial&#95;k^X(z)
=f&#95;{k-1}(0)=0.
$$

因此 $f&#95;k(z)\in Z&#95;k(Y)$。

再检查它把边界送到边界。取 $b\in B&#95;k(X)$，则存在 $c\in C&#95;{k+1}(X)$ 使得 $b=\partial&#95;{k+1}^Xc$。于是

$$
\begin{aligned}
f&#95;k(b)
&=f&#95;k\partial&#95;{k+1}^X(c)\\\\
&=\partial&#95;{k+1}^Yf&#95;{k+1}(c)\\\\
&\in B&#95;k(Y).
\end{aligned}
$$

因此公式

$$
f&#95;{\ast,k}:H&#95;k(X)\to H&#95;k(Y),
\qquad
f&#95;{\ast,k}([z])=[f&#95;k(z)]
$$

是有意义的。还需要检查代表元的选择不会影响结果。若 $[z]=[z']$，则 $z-z'\in B&#95;k(X)$；刚才已经证明 $f&#95;k(z-z')\in B&#95;k(Y)$，所以

$$
[f&#95;k(z)]=[f&#95;k(z')].
$$

这就证明了单纯映射确实诱导同调映射。恒等映射诱导恒等同调映射，复合映射满足

$$
(g\circ f)&#95;\ast=g&#95;\ast\circ f&#95;\ast.
$$

这两个性质正是函子性的来源。

## D. 一个最小持久同调例子的完整计算

仍取圆周的三个顶点，但按如下过滤逐步加边：

$$
X&#95;0=\lbrace v&#95;0,v&#95;1,v&#95;2\rbrace,
$$

$$
X&#95;1=X&#95;0\cup\lbrace e&#95;0\rbrace,
\qquad
X&#95;2=X&#95;1\cup\lbrace e&#95;1\rbrace,
$$

$$
X&#95;3=X&#95;2\cup\lbrace e&#95;2\rbrace.
$$

为避免方向造成干扰，以下先使用系数域 $\mathbb{F}&#95;2$，此时 $-1=1$。

### D.1 $H&#95;0$ 的变化

在 $X&#95;0$ 中没有边，所以三个顶点各自是一个连通分支：

$$
H&#95;0(X&#95;0)\cong\mathbb{F}&#95;2^3.
$$

加入 $e&#95;0$ 后，$v&#95;0$ 和 $v&#95;1$ 连通。边界

$$
\partial e&#95;0=v&#95;0+v&#95;1
$$

说明在商空间中 $[v&#95;0]=[v&#95;1]$，但 $[v&#95;2]$ 仍独立，所以

$$
H&#95;0(X&#95;1)\cong\mathbb{F}&#95;2^2.
$$

加入 $e&#95;1$ 后，$v&#95;1$ 和 $v&#95;2$ 连通，于是三个顶点代表同一个连通分支：

$$
H&#95;0(X&#95;2)\cong\mathbb{F}&#95;2.
$$

最后加入 $e&#95;2$ 不会再减少连通分支，因此

$$
H&#95;0(X&#95;3)\cong\mathbb{F}&#95;2.
$$

所以三个最初的连通分支中，一个在加入 $e&#95;0$ 时与另一个合并，另一个在加入 $e&#95;1$ 时合并。若把出现时刻记为 $0$，则对应的有限区间可以写成

$$
[0,1),\qquad [0,2),\qquad [0,\infty).
$$

这里的区间端点取决于我们如何编号过滤步骤；重要的是“出生”表示一个新分支出现，“死亡”表示它与更早的分支合并。

### D.2 $H&#95;1$ 的变化

在 $X&#95;0,X&#95;1,X&#95;2$ 中都没有闭合路径。逐项检查可知：

$$
Z&#95;1(X&#95;0)=0,
\qquad
Z&#95;1(X&#95;1)=0,
\qquad
Z&#95;1(X&#95;2)=0.
$$

在 $X&#95;3$ 中，三条边组成

$$
z=e&#95;0+e&#95;1+e&#95;2.
$$

其边界为

$$
\begin{aligned}
\partial z
&=(v&#95;0+v&#95;1)+(v&#95;1+v&#95;2)+(v&#95;2+v&#95;0)\\\\
&=2v&#95;0+2v&#95;1+2v&#95;2\\\\
&=0
\quad\text{in }\mathbb{F}&#95;2.
\end{aligned}
$$

因为没有二维单纯形，$B&#95;1(X&#95;3)=0$，所以 $[z]$ 是一个非零的一维同调类，并且之后没有更高维单纯形把它填平。它给出一个无限持久的区间 $[3,\infty)$。

这个小例子包含 barcode 的全部基本动作：连通分支可以合并，环可以出生，也可以在加入二维单纯形后死亡。一般的持久同调会把同样的过程放到更大的过滤中，并用线性代数追踪每个类的出生和死亡。

### D.3 把区间分解写成边界矩阵消元

把三个顶点和三条边按过滤顺序排列。在 $\mathbb{F}&#95;2$ 上，边界矩阵的三列分别是三条边的端点：

$$
D&#95;1=
\\begin{array}{c|ccc}
 & e&#95;0 & e&#95;1 & e&#95;2\\\\
v&#95;0 & 1 & 0 & 1\\\\
v&#95;1 & 1 & 1 & 0\\\\
v&#95;2 & 0 & 1 & 1
\\end{array}.
$$

消元只允许把一列加到另一列，这不会改变列空间。先保留第一列：

$$
R&#95;0=e&#95;0=(1,1,0)^T.
$$

第二列与第一列的主元位置不同，因此保留为

$$
R&#95;1=e&#95;1=(0,1,1)^T.
$$

第三列逐步相加：

$$
\\begin{aligned}
R&#95;2
&=e&#95;2+R&#95;0\\\\
&=(1,0,1)^T+(1,1,0)^T\\\\
&=(0,1,1)^T\\\\
&=R&#95;1.
\\end{aligned}
$$

再减去（在 $\mathbb{F}&#95;2$ 中就是再相加）$R&#95;1$，得到

$$
R&#95;2+R&#95;1=0.
$$

因此第三列最终成为零列。它表示 $e&#95;2$ 加入时产生了一个新的 $1$-循环，也就是 $H&#95;1$ 中出生于时刻 $3$ 的无限区间。前两列各有主元，说明它们分别把两个新的顶点关系消掉，正对应 $H&#95;0$ 中两个有限区间的死亡。这个逐列过程就是持久同调软件所执行的基本分解步骤；“区间”并不是额外画出来的标签，而是主元列与零列在过滤顺序中的配对结果。

## E. Euler–Poincare 公式的完整分解

设

$$
0\longrightarrow C&#95;n
\xrightarrow{\partial&#95;n}C&#95;{n-1}
\longrightarrow\cdots
\xrightarrow{\partial&#95;1}C&#95;0
\longrightarrow0
$$

是域 $\mathbb{F}$ 上的有限维链复形。仍记

$$
Z&#95;k=\ker\partial&#95;k,
\qquad
B&#95;k=\operatorname{im}\partial&#95;{k+1},
\qquad
H&#95;k=Z&#95;k/B&#95;k.
$$

先把每个链群 $C&#95;k$ 分解。线性映射 $\partial&#95;k:C&#95;k\to C&#95;{k-1}$ 的核是 $Z&#95;k$，像是 $B&#95;{k-1}$。秩–零度定理逐项给出

$$
\dim C&#95;k
=\dim\ker\partial&#95;k
+\dim\operatorname{im}\partial&#95;k
=\dim Z&#95;k+\dim B&#95;{k-1}.
$$

再分解循环空间。因为 $B&#95;k\subseteq Z&#95;k$，有短正合列

$$
0\longrightarrow B&#95;k
\longrightarrow Z&#95;k
\longrightarrow H&#95;k
\longrightarrow0.
$$

有限维向量空间的短正合列中，中间项的维数等于两端维数之和，因此

$$
\dim Z&#95;k=\dim B&#95;k+\dim H&#95;k.
$$

代回第一步：

$$
\dim C&#95;k
=\dim H&#95;k
+\dim B&#95;k
+\dim B&#95;{k-1}.
$$

对所有 $k$ 取交替和：

$$
\begin{aligned}
\sum&#95;k(-1)^k\dim C&#95;k
&=\sum&#95;k(-1)^k\dim H&#95;k\\\\
&\quad+\sum&#95;k(-1)^k\dim B&#95;k\\\\
&\quad+\sum&#95;k(-1)^k\dim B&#95;{k-1}.
\end{aligned}
$$

把最后一项换指标 $j=k-1$：

$$
\begin{aligned}
\sum&#95;k(-1)^k\dim B&#95;{k-1}
&=\sum&#95;j(-1)^{j+1}\dim B&#95;j\\\\
&=-\sum&#95;j(-1)^j\dim B&#95;j.
\end{aligned}
$$

它与前一个边界空间的交替和恰好抵消，于是只剩下

$$
\sum&#95;k(-1)^k\dim C&#95;k
=\sum&#95;k(-1)^k\dim H&#95;k.
$$

对有限单纯复形，$\dim C&#95;k$ 就是 $k$ 维单纯形的个数，$\dim H&#95;k=b&#95;k$。因此

$$
\chi
=\sum&#95;k(-1)^k\dim C&#95;k
=\sum&#95;k(-1)^kb&#95;k.
$$

这也精确解释了“局部剖分信息为什么会消失”：每个边界空间 $B&#95;k$ 在相邻两项中各出现一次，符号相反，所以在交替和中被完全消去。

## F. cup product 为什么能定义在上同调类上

设 $\alpha\in C^p(X;R)$、$\beta\in C^q(X;R)$。关键是余边界对 cup product 满足分次 Leibniz 公式

$$
\delta(\alpha\smile\beta)
=(\delta\alpha)\smile\beta
+(-1)^p\alpha\smile(\delta\beta).
$$

下面把符号逐项展开。取一个 $(p+q+1)$-单纯形

$$
\sigma=[v&#95;0\ldots v&#95;{p+q+1}].
$$

从左侧开始：

$$
\delta(\alpha\smile\beta)(\sigma)
=\sum&#95;{i=0}^{p+q+1}(-1)^i
(\alpha\smile\beta)
[v&#95;0\ldots\widehat{v&#95;i}\ldots v&#95;{p+q+1}].
$$

当 $0\leq i\leq p$ 时，被删去的顶点位于前半段，所以对应项为

$$
(-1)^i
\alpha[v&#95;0\ldots\widehat{v&#95;i}\ldots v&#95;{p+1}]
\beta[v&#95;{p+1}\ldots v&#95;{p+q+1}].
$$

这些正是 $((\delta\alpha)\smile\beta)(\sigma)$ 中除去 $i=p+1$ 以外的各项。缺少的那一项是

$$
(-1)^{p+1}
\alpha[v&#95;0\ldots v&#95;p]
\beta[v&#95;{p+1}\ldots v&#95;{p+q+1}].
$$

当 $p+1\leq i\leq p+q+1$ 时，被删去的顶点位于后半段，原和式中的对应项为

$$
(-1)^i
\alpha[v&#95;0\ldots v&#95;p]
\beta[v&#95;p\ldots\widehat{v&#95;i}\ldots v&#95;{p+q+1}].
$$

令后半段中的局部指标为 $j=i-p$。因为

$$
(-1)^p(-1)^j=(-1)^p(-1)^{i-p}=(-1)^i,
$$

这些正是 $(-1)^p(\alpha\smile\delta\beta)(\sigma)$ 中除去 $j=0$ 以外的各项。缺少的 $j=0$ 项是

$$
(-1)^p
\alpha[v&#95;0\ldots v&#95;p]
\beta[v&#95;{p+1}\ldots v&#95;{p+q+1}].
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

其中 $e$ 是单位元，$a^{-1}$ 是 $a$ 的逆元。如果还满足 $a\cdot b=b\cdot a$，就称为 Abel 群。整数在加法下是 Abel 群；正整数 $n$ 个元素的置换在复合下构成对称群 $S&#95;n$，但当 $n\geq3$ 时通常不是 Abel 群。

代数拓扑中的链群就是 Abel 群。若 $K$ 是有限单纯复形，$K&#95;k$ 表示所有 $k$-单纯形组成的集合，$\sigma$ 表示其中一个 $k$-单纯形，则

$$
C&#95;k(K;\mathbb{Z})
=\bigoplus_{\sigma\in K&#95;k}\mathbb{Z}\,\sigma.
$$

符号 $\bigoplus$ 表示“有限形式和”：每个 $k$-链只含有限多个非零系数；$\mathbb{Z}\,\sigma$ 表示 $\sigma$ 的整数倍组成的自由 Abel 群。于是 $C&#95;k(K;\mathbb{Z})$ 的元素可以写成

$$
c=\sum_{\sigma\in K&#95;k}m&#95;\sigma\,\sigma,
\qquad m&#95;\sigma\in\mathbb{Z}.
$$

这里 $c$ 是一条 $k$-链，$m&#95;\sigma$ 是单纯形 $\sigma$ 的整数系数。边界算子 $\partial&#95;k$ 是群同态：

$$
\partial&#95;k:C&#95;k(K;\mathbb{Z})
\longrightarrow C&#95;{k-1}(K;\mathbb{Z}).
$$

因此 $Z&#95;k=\ker\partial&#95;k$ 和 $B&#95;k=\operatorname{im}\partial&#95;{k+1}$ 都是 Abel 子群，同调群

$$
H&#95;k(K;\mathbb{Z})=Z&#95;k/B&#95;k
$$

就是它们的商群。符号 $\ker$ 表示核，$\operatorname{im}$ 表示像，斜杠表示商群。

### G.2 环：允许系数相乘的结构

一个环 $R$ 首先是一个 Abel 群；它还有一个乘法，满足结合律和分配律。这里 $r,s,t$ 表示 $R$ 中任意元素，$0$ 表示加法单位元，$1$ 表示乘法单位元。分配律写成

$$
r(s+t)=rs+rt,
\qquad
(r+s)t=rt+st.
$$

整数环 $\mathbb{Z}$、剩余类环 $\mathbb{Z}/n\mathbb{Z}$ 和实系数多项式环 $\mathbb{R}[x]$ 都是环。符号 $\mathbb{Z}/n\mathbb{Z}$ 表示把相差 $n$ 的整数视为同一个剩余类；$\mathbb{R}[x]$ 表示变量 $x$ 的实系数多项式集合。

环 $R$ 的标量可以作用在另一个 Abel 群 $M$ 上，这样的对象叫作 $R$-模。若 $m&#95;1,m&#95;2\in M$、$r&#95;1,r&#95;2\in R$，模的运算满足

$$
r(m&#95;1+m&#95;2)=rm&#95;1+rm&#95;2,
\qquad
(r&#95;1+r&#95;2)m=r&#95;1m+r&#95;2m,
\qquad
(r&#95;1r&#95;2)m=r&#95;1(r&#95;2m).
$$

将链的整数系数换成环 $R$ 中的系数，就得到

$$
C&#95;k(K;R)
=\bigoplus_{\sigma\in K&#95;k}R\,\sigma.
$$

此时边界算子对 $R$-系数线性，因而

$$
H&#95;k(K;R)
=\ker\partial&#95;k/\operatorname{im}\partial&#95;{k+1}
$$

是一个 $R$-模。也就是说，系数环决定了链可以用哪些数相加，以及边界关系中的整数如何被解释。

群与环还可以合成群环。给定群 $G$ 和环 $R$，群环 $R[G]$ 的元素是有限形式和

$$
\sum_{g\in G}a&#95;g\,g,
\qquad a&#95;g\in R.
$$

这里的 $g$ 是群元素，$a&#95;g$ 是它对应的环系数；“有限”表示只有有限多个 $a&#95;g$ 不为零。若 $h$ 也是群元素，$b&#95;h$ 表示 $h$ 对应的环系数，乘法由群中的乘法 $gh$ 和环中的乘法 $a&#95;g b&#95;h$ 共同决定：

$$
\left(\sum&#95;{g\in G}a&#95;g g\right)
\left(\sum&#95;{h\in G}b&#95;h h\right)
=\sum&#95;{g,h\in G}a&#95;g b&#95;h(gh).
$$

群环把“对称性”和“线性组合”放进同一个代数对象，在基本群的表示、覆盖空间和同调代数中都会出现。

### G.3 域：每个非零元素都可以除

域 $F$ 是一个交换的单位环，并且每个非零元素都有乘法逆元。符号 $F\setminus\{0\}$ 表示从 $F$ 中删去零元素；要求

$$
(F\setminus\{0\},\cdot)
$$

在乘法下构成 Abel 群。$\mathbb{Q}$、$\mathbb{R}$ 和 $\mathbb{C}$ 都是域；$\mathbb{Z}$ 不是域，因为 $2$ 没有整数逆元。

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

因此 $[u]$ 是 $[a]$ 的乘法逆元。每个非零剩余类都有逆元，故 $\mathbb{Z}/p\mathbb{Z}$ 是域，通常记为 $\mathbb{F}&#95;p$。这里 $\gcd$ 表示最大公因数，$\mathbb{F}&#95;p$ 表示含有 $p$ 个元素的有限域。

域上的模就是向量空间。向量空间可以使用维数、秩、矩阵主元等线性代数工具；这正是持久同调常在有限域上计算的原因。一般环上的模未必具有这些性质，所以“取域系数”不只是记号变化，而是改变了可用的计算工具。

### G.4 系数为什么会改变同调

设 $X$ 是一个拓扑空间。对同一个 $X$，可以分别计算

$$
H&#95;k(X;\mathbb{Z}),
\qquad
H&#95;k(X;\mathbb{F}&#95;2),
\qquad
H&#95;k(X;\mathbb{R}).
$$

这里 $\mathbb{F}&#95;2$ 是含有两个元素的有限域；$H&#95;k(X;R)$ 表示用系数环 $R$ 构造的第 $k$ 个同调模。

令 $S^1$ 表示平面中的单位圆，即满足 $x^2+y^2=1$ 的点集，其中 $x,y$ 是实坐标。整数系数给出

$$
H&#95;0(S^1;\mathbb{Z})\cong\mathbb{Z},
\qquad
H&#95;1(S^1;\mathbb{Z})\cong\mathbb{Z}.
$$

符号 $\cong$ 表示同构，也就是存在保持代数运算的双射。换成 $\mathbb{F}&#95;p$ 系数后，圆周的同调变为

$$
H&#95;0(S^1;\mathbb{F}&#95;p)\cong\mathbb{F}&#95;p,
\qquad
H&#95;1(S^1;\mathbb{F}&#95;p)\cong\mathbb{F}&#95;p.
$$

圆周仍有一个连通分支和一个一维环，但整数“绕行多少次”的信息被压缩成模 $p$ 的信息。整数系数可以保留扭结，例如实射影平面 $\mathbb{R}P^2$ 的第一同调满足

$$
H&#95;1(\mathbb{R}P^2;\mathbb{Z})
\cong\mathbb{Z}/2\mathbb{Z}.
$$

这里 $\mathbb{R}P^2$ 是三维实向量空间 $\mathbb{R}^3$ 中所有一维子空间组成的空间；等价地，它把球面上互为相反点的两个点视为同一点。这个 $\mathbb{Z}/2\mathbb{Z}$ 正是整数系数下可见的 $2$-扭结；改用 $\mathbb{F}&#95;2$ 时，它表现为一个一维向量空间 $\mathbb{F}&#95;2$。

因此，系数环不是计算时随手选择的参数。它决定方向、倍数和扭结哪些会被保留下来，也决定我们能否使用向量空间的线性代数。代数拓扑中从 $\mathbb{Z}$ 切换到有限域，实质上是在改变观察空间的代数镜头。
