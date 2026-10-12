---
title: "从环路到自由群：基本群与空间的拼接"
date: 2026-08-30 16:00:00
categories:
  - 理论数学
tags:
  - 代数拓扑
  - 基本群
  - 同伦
  - van Kampen 定理
  - 自由群
mathjax: true
toc: true
comments: true
---

第一篇文章里，同调把“洞”翻译成了可以计算的群。但同调有一个明显的遗憾：它只记录环路相加后的结果，不记录环路经过的先后顺序。两个环路先走哪一个、后走哪一个，在同调群里会被交换掉；在基本群里却可能得到不同的元素。

这篇文章只讨论一个问题：怎样把“环路可以连续变形”组织成一个群。我们会从环路和同伦的定义开始，计算圆周的基本群，解释八字形为什么产生自由群，最后介绍 van Kampen 定理如何把大空间的基本群分解成小空间的基本群。所有新记号都在首次出现时说明，完整的计算放在附录中。

<!--more-->

## 0. 预备语言：我们究竟在什么对象上运算

为了让后面的证明可以独立阅读，先固定几条基本定义。一个**拓扑空间**是一个集合 $X$ 和一族子集 $\mathcal{T}$，其中 $\mathcal{T}$ 中的集合称为开集，并满足：空集 $\varnothing$ 和 $X$ 本身是开集；任意多个开集的并仍是开集；有限多个开集的交仍是开集。写作

$$
(X,\mathcal{T}).
$$

本文中 $\mathbb{R}$ 表示实数集，$\mathbb{Z}$ 表示整数集。一个空间称为**连通**，如果不能写成两个不相交的非空开集的并；称为**离散空间**，如果每一个子集都是开集。把 $\mathbb{Z}$ 看成 $\mathbb{R}$ 的子空间时，它是离散的：对每个整数 $n$，区间 $(n-\tfrac12,n+\tfrac12)$ 与 $\mathbb{Z}$ 的交恰好是单点 $\lbrace n\rbrace$。因此从连通区间到 $\mathbb{Z}$ 的连续函数只能是常值函数。符号 $\bigsqcup$ 表示不交并集。

若 $U\subseteq X$，则 $U$ 的子空间拓扑由所有形如 $U\cap O$ 的集合组成，其中 $O\in\mathcal{T}$。因此“$U$ 是 $X$ 的开子空间”表示 $U$ 本身是 $\mathcal{T}$ 中的开集。

设 $(X,\mathcal{T}_X)$ 和 $(Y,\mathcal{T}_Y)$ 是拓扑空间。映射 $f:X\to Y$ 称为**连续**，如果对 $Y$ 中每个开集 $O\in\mathcal{T}_Y$，其原像

$$
f^{-1}(O)=\lbrace x\in X:f(x)\in O\rbrace
$$

都是 $X$ 中的开集。这里 $f^{-1}(O)$ 表示集合的原像，不要求 $f$ 有逆函数。

一条路径是连续映射 $\alpha:[0,1]\to X$。如果任意两点 $x,y\in X$ 都能由一条路径连接，即存在 $\alpha$ 满足 $\alpha(0)=x$、$\alpha(1)=y$，就称 $X$ **道路连通**。道路连通比通常的连通性更强，但 van Kampen 定理需要的是道路连通。

两个拓扑空间 $X,Y$ **同胚**，是指存在双射 $f:X\to Y$，使 $f$ 和逆映射 $f^{-1}:Y\to X$ 都连续；这表示它们的拓扑结构完全相同。一个满射 $p:E\to B$ 称为**覆盖映射**，如果对每个 $b\in B$，都存在开邻域 $W\ni b$，使得

$$
p^{-1}(W)=\bigsqcup_{j\in J}E_j,
$$

其中各 $E_j$ 两两不交且开，并且每个限制 $p|_{E_j}:E_j\to W$ 都是同胚。这样的 $W$ 称为被均匀覆盖的邻域。直观上，覆盖映射把底空间的每个小片复制成许多互不相交、形状相同的小片。

如果恒等映射可以在整个空间中连续变形为某个常值映射，就称空间**可缩**；如果子空间 $A\subseteq X$ 可以在 $X$ 内连续变形到一点，并且变形过程中 $A$ 中的点仍留在 $A$ 中，就称 $A$ 在 $X$ 中**形变收缩**。可缩空间的基本群只有单位元。

如果 $G$ 和 $H$ 是群，映射 $\varphi:G\to H$ 满足

$$
\varphi(gh)=\varphi(g)\varphi(h)
\qquad(g,h\in G),
$$

就称为群同态。若群同态同时是一一对应，就称为群同构，记为 $G\cong H$。单位元分别记为 $1_G$ 和 $1_H$；在不引起混淆时都写成 $1$。

若 $N\subseteq G$ 是子群，并且对所有 $g\in G$ 都有 $gNg^{-1}=N$，就称 $N$ 是 $G$ 的**正规子群**，记作 $N\trianglelefteq G$。此时可以把 $N$ 中的元素视为单位元，得到商群 $G/N$。更一般地，若一组元素 $S\subseteq G$ 给出关系，就用 $\langle\!\langle S\rangle\!\rangle$ 表示包含 $S$ 的最小正规子群。

更具体地，商群的元素是左陪集

$$
G/N=\lbrace gN:g\in G\rbrace,
$$

乘法定义为 $(gN)(hN)=ghN$。正规性正好保证这个定义与陪集代表元的选择无关。

本文还会用到圆盘和实射影平面。闭圆盘定义为

$$
D^2=\lbrace(x,y)\in\mathbb{R}^2:x^2+y^2\leq1\rbrace,
$$

其边界是单位圆 $\partial D^2=S^1$。实射影平面 $\mathbb{R}P^2$ 可以定义为：把 $D^2$ 的边界上每一对相反点 $z$ 和 $-z$ 识别后得到的商空间。更一般地，给定等价关系 $\sim$，商集 $X/\!\sim$ 由所有等价类组成，商映射 $q:X\to X/\!\sim$ 把点送到它的等价类；商空间拓扑规定 $O\subseteq X/\!\sim$ 开，当且仅当 $q^{-1}(O)$ 在 $X$ 中开。这是使 $q$ 连续的最强（也称最终）拓扑。

## 1. 从闭合路径到同伦类

设 $X$ 是一个拓扑空间，$x_0\in X$ 是选定的基点。基点不是装饰：基本群研究的是从 $x_0$ 出发、最后又回到 $x_0$ 的环路。

一条连续映射

$$
\gamma:[0,1]\longrightarrow X
$$

如果满足

$$
\gamma(0)=\gamma(1)=x_0,
$$

就称为基于 $x_0$ 的环路。参数 $t\in[0,1]$ 表示沿路径行走的位置；它不是空间中的额外坐标。

两条环路 $\gamma$ 和 $\eta$ 如果可以在始终固定端点的条件下连续变形，就称它们基于 $x_0$ 同伦。严格地说，这意味着存在连续映射

$$
H:[0,1]\times[0,1]\longrightarrow X
$$

满足

$$
H(t,0)=\gamma(t),\qquad H(t,1)=\eta(t),
$$

以及

$$
H(0,s)=H(1,s)=x_0
\qquad (0\leq s\leq1).
$$

这里 $s$ 是变形参数，$t$ 仍然是路径参数。最后一条条件保证变形过程中每条中间路径都从 $x_0$ 出发并回到 $x_0$。

把彼此同伦的环路视为同一个元素，得到一个集合，记作

$$
\pi_1(X,x_0).
$$

符号 $\pi_1$ 读作“第一基本群”；下标 $1$ 表示它研究一维的环路。一个环路的同伦类用 $[\gamma]$ 表示，方括号意味着我们不再区分同一个同伦类中的不同参数化。

## 2. 环路如何相乘

两条基于 $x_0$ 的环路可以首尾相接。先走 $\gamma$，再走 $\eta$，定义拼接路径

$$
(\gamma\ast\eta)(t)
=
\begin{cases}
\gamma(2t),&0\leq t\leq\frac12,\\\\
\eta(2t-1),&\frac12\leq t\leq1.
\end{cases}
$$

第一段把 $[0,\frac12]$ 的时间压缩到 $\gamma$ 的整个区间，第二段把 $[\frac12,1]$ 的时间压缩到 $\eta$ 的整个区间。由于 $\gamma(1)=x_0=\eta(0)$，两段在 $t=\frac12$ 处相接。

在同伦类上定义

$$
[\gamma]\cdot[\eta]:=[\gamma\ast\eta].
$$

常值环路

$$
c_{x_0}(t)=x_0
$$

给出单位元。反向行走的环路定义为

$$
\gamma^{-1}(t)=\gamma(1-t),
$$

它给出 $[\gamma]$ 的逆元。拼接在路径层面并非严格结合，因为不同的时间重参数化会产生不同的公式；但这些重参数化之间存在固定端点同伦，所以在同伦类上满足结合律。附录 A 会把这三件事逐一验证。

因此，$\pi_1(X,x_0)$ 是一个群。它一般不是 Abel 群。群运算不交换的原因很具体：先沿一条环路行走再沿另一条环路行走，和交换两条环路的先后顺序，可能无法在不穿过障碍的情况下互相变形。

## 3. 圆周：绕行数就是整数

令

$$
S^1=\lbrace(x,y)\in\mathbb{R}^2:x^2+y^2=1\rbrace,
$$

取基点 $x_0=(1,0)$。从 $x_0$ 出发绕圆周一圈的环路记作 $a$。绕两圈得到 $a^2$，反向绕一圈得到 $a^{-1}$。直观上，任何环路都应该由一个整数记录：正数表示逆时针绕行，负数表示顺时针绕行，零表示可以缩回基点。

这个直觉由覆盖映射严格实现。定义

$$
p:\mathbb{R}\longrightarrow S^1,
\qquad
p(u)=(\cos 2\pi u,\sin 2\pi u).
$$

实数轴每增加 $1$，$p(u)$ 就在圆周上多走一圈。若环路 $\gamma$ 满足 $\gamma(0)=\gamma(1)=x_0$，并且存在唯一提升 $\widetilde\gamma$ 使

$$
\widetilde\gamma(0)=0,\qquad p(\widetilde\gamma(t))=\gamma(t),
$$

那么 $\widetilde\gamma(1)$ 必须是整数。这个整数就是绕行数：

$$
\operatorname{wind}(\gamma):=\widetilde\gamma(1)\in\mathbb{Z}.
$$

路径提升、绕行数的同伦不变性，以及

$$
\operatorname{wind}(\gamma\ast\eta)
=\operatorname{wind}(\gamma)+\operatorname{wind}(\eta)
$$

都在附录 B 中逐步证明。最终结果是

$$
\pi_1(S^1,x_0)\cong\mathbb{Z}.
$$

这里的同构不仅是集合之间的一一对应，还保持群运算：环路的拼接对应整数的加法。

## 4. 八字形与自由群

把两个圆周在一个基点处粘合，得到八字形空间，记为

$$
X=S^1_a\vee S^1_b.
$$

符号 $\vee$ 表示楔和：先取两个空间的不交并，再把各自选定的基点识别成同一个点。记左边圆周的基本环路为 $a$，右边圆周的基本环路为 $b$。

在八字形中，一条环路可以先绕左圈，再绕右圈，于是得到 $ab$；也可以得到 $ba$。这两个环路通常不相等。一个一般的环路可以写成

$$
a\ b^{-1}\ a\ a^{-1}\ b\cdots,
$$

其中只允许删除相邻的

$$
aa^{-1},\quad a^{-1}a,\quad bb^{-1},\quad b^{-1}b.
$$

删除后剩下的字列称为约化字。由两个符号 $a,b$ 及其逆元组成的所有约化字，在连接并继续约化的运算下构成两个生成元的自由群，记为

$$
F(a,b).
$$

八字形的基本群正是

$$
\pi_1(S^1_a\vee S^1_b,x_0)\cong F(a,b).
$$

为什么这里不是 $\mathbb{Z}^2$？因为 $\mathbb{Z}^2$ 要求 $ab=ba$，而八字形中的两条基本环路没有二维区域把交换子

$$
aba^{-1}b^{-1}
$$

填平。只有把基本群阿贝尔化，强制所有元素交换，才得到

$$
F(a,b)_{\mathrm{ab}}\cong\mathbb{Z}^2.
$$

下标 $\mathrm{ab}$ 表示阿贝尔化；它把一个群除以由所有交换子生成的正规子群，使商群成为 Abel 群。

## 5. van Kampen 定理：把拼接变成代数分解

基本群最有用的地方，不只是计算圆周，而是能够处理空间的拼接。设

$$
X=U\cup V,
$$

其中 $U,V$ 是 $X$ 的开子空间，$U$、$V$ 以及交集 $U\cap V$ 都道路连通，并且基点 $x_0$ 位于 $U\cap V$ 中。道路连通的意思是：任意两点之间都存在连续路径。

包含映射

$$
i_U:U\cap V\hookrightarrow U,
\qquad
i_V:U\cap V\hookrightarrow V
$$

诱导基本群同态

$$
(i_U)_*:\pi_1(U\cap V,x_0)\to\pi_1(U,x_0),
$$

$$
(i_V)_*:\pi_1(U\cap V,x_0)\to\pi_1(V,x_0).
$$

van Kampen 定理说：

$$
\pi_1(X,x_0)
\cong
\pi_1(U,x_0)
\ast_{\pi_1(U\cap V,x_0)}
\pi_1(V,x_0).
$$

右侧的符号 $\ast_{\pi_1(U\cap V,x_0)}$ 表示**带 amalgamation 的自由积**。它的意思是：先把 $U$ 和 $V$ 的基本群自由地拼在一起，再把交集中的同一条环路在两边的像识别起来。

如果 $U\cap V$ 可缩，则

$$
\pi_1(U\cap V,x_0)=1,
$$

于是公式简化为自由积

$$
\pi_1(X,x_0)
\cong
\pi_1(U,x_0)\ast\pi_1(V,x_0).
$$

这里 $1$ 表示只有单位元的平凡群。这个特殊情形正是八字形的来源：取 $U$、$V$ 分别围住左右两个圆，交集取一个可缩的小区域，就得到

$$
\pi_1(X,x_0)\cong\mathbb{Z}\ast\mathbb{Z}=F(a,b).
$$

## 6. 一个带关系的例子：实射影平面

自由群记录“可以任意组合”的环路；如果空间中存在二维区域，就会给这些环路增加关系。最简单的例子是实射影平面 $\mathbb{R}P^2$。

把一个圆盘 $D^2$ 的边界上的相反点识别，得到 $\mathbb{R}P^2$。设边界上的基本环路为 $a$。沿边界走一圈时，识别关系使得走第二个半圈与第一个半圈方向相同，因此整个边界在基本群中代表

$$
a^2.
$$

圆盘本身是可缩的，所以它的边界必须在粘合后的空间中变成单位元。于是

$$
\pi_1(\mathbb{R}P^2)
\cong
\langle a\mid a^2=1\rangle
\cong
\mathbb{Z}/2\mathbb{Z}.
$$

尖括号中的

$$
\langle a\mid a^2=1\rangle
$$

表示由生成元 $a$ 生成、并加入关系 $a^2=1$ 的群。这个例子说明 van Kampen 的实际作用：每加入一个二维胞腔，就可能给一维环路加入一个关系。

## 7. 基本群与第一同调群的关系

基本群保留非交换的环路信息，而第一同调群把环路相加并强制交换。对于道路连通的单纯复形（由单纯形粘合成的空间），有自然同构；附录 F 给出链群、极大树和二维关系的完整推导。

$$
H_1(X;\mathbb{Z})
\cong
\pi_1(X,x_0)_{\mathrm{ab}}.
$$

右侧是基本群的阿贝尔化。对八字形，

$$
\pi_1(X,x_0)=F(a,b),
\qquad
H_1(X;\mathbb{Z})\cong\mathbb{Z}^2.
$$

前者知道环路的完整字列，后者只知道绕左圈和右圈各多少次。对实射影平面，

$$
\pi_1(\mathbb{R}P^2)\cong\mathbb{Z}/2\mathbb{Z},
\qquad
H_1(\mathbb{R}P^2;\mathbb{Z})\cong\mathbb{Z}/2\mathbb{Z},
$$

因为这里的基本群本来就是交换群。

从同调到基本群，代数拓扑的观察精度提高了一层：同调看见线性叠加后的洞，基本群看见环路的组合规则；而 van Kampen 定理把空间的几何拼接翻译成群的代数拼接。

## 参考文献

1. A. Hatcher, *Algebraic Topology*, Cambridge University Press, 2002, Chapter 1. [Author's site](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf).
2. R. M. Brown, *Topology and Groupoids*, BookSurge, 2006.
3. E. H. Spanier, *Algebraic Topology*, McGraw--Hill, 1966.
4. J. P. May, *A Concise Course in Algebraic Topology*, University of Chicago Press, 1999.
5. C. R. F. Maunder, *Algebraic Topology*, Cambridge University Press, 1980.

---

# 附录

如下为正文附录补充。

## A. 环路拼接为什么在同伦类上给出群

### A.1 拼接仍然是环路

设 $\gamma(0)=\gamma(1)=x_0$，$\eta(0)=\eta(1)=x_0$。按定义

$$
(\gamma\ast\eta)(0)=\gamma(0)=x_0,
$$

而

$$
(\gamma\ast\eta)(1)=\eta(1)=x_0.
$$

在中点处，第一段的终值和第二段的初值分别为

$$
\gamma(1)=x_0,
\qquad
\eta(0)=x_0.
$$

因此拼接函数连续，并且仍然是基于 $x_0$ 的环路。

### A.2 拼接与同伦相容

若 $\gamma\simeq\gamma'$、$\eta\simeq\eta'$，设 $H$ 和 $K$ 分别是保持基点的同伦。定义

$$
L(t,s)
=
\begin{cases}
H(2t,s),&0\leq t\leq\frac12,\\\\
K(2t-1,s),&\frac12\leq t\leq1.
\end{cases}
$$

在 $t=\frac12$ 处，

$$
H(1,s)=x_0=K(0,s),
$$

所以 $L$ 连续；并且

$$
L(t,0)=(\gamma\ast\eta)(t),
\qquad
L(t,1)=(\gamma'\ast\eta')(t).
$$

因此

$$
[\gamma]=[\gamma'],\quad [\eta]=[\eta']
\Longrightarrow
[\gamma\ast\eta]=[\gamma'\ast\eta'].
$$

这说明在同伦类上定义的乘法与代表元的选择无关。

### A.3 结合律

路径层面的 $(\gamma\ast\eta)\ast\zeta$ 与 $\gamma\ast(\eta\ast\zeta)$ 在 $[0,1]$ 上使用了不同的时间分割。前者在

$$
0,\quad\frac14,\quad\frac12,\quad1
$$

这些位置分别走完 $\gamma$、走完 $\eta$、走完 $\zeta$；后者在

$$
0,\quad\frac12,\quad\frac34,\quad1
$$

这些位置完成同样的三段行走。为了把“只是速度不同”写成公式，先定义共同的三段路径

$$
\Lambda(u)=
\begin{cases}
\gamma(3u),&0\leq u\leq\frac13,\\
\eta(3u-1),&\frac13\leq u\leq\frac23,\\
\zeta(3u-2),&\frac23\leq u\leq1.
\end{cases}
$$

左结合的路径是 $\Lambda\circ r_L$，其中

$$
r_L(t)=
\begin{cases}
\frac{4t}{3},&0\leq t\leq\frac12,\\
\frac{2t+1}{3},&\frac12\leq t\leq1,
\end{cases}
$$

右结合的路径是 $\Lambda\circ r_R$，其中

$$
r_R(t)=
\begin{cases}
\frac{2t}{3},&0\leq t\leq\frac12,\\
\frac{4t-1}{3},&\frac12\leq t\leq1.
\end{cases}
$$

例如在 $0\leq t\leq\frac14$ 时，$\Lambda(r_L(t))=\gamma(4t)$，正是 $(\gamma\ast\eta)\ast\zeta$ 的第一段；其余区间逐段代入也分别得到 $\eta(4t-1)$ 和 $\zeta(2t-1)$。对 $r_R$ 同样代入可得到 $\gamma(2t)$、$\eta(4t-2)$ 和 $\zeta(4t-3)$。

对 $0\leq s\leq1$，令

$$
r_s(t)=(1-s)r_L(t)+s r_R(t),
\qquad
H(t,s)=\Lambda(r_s(t)).
$$

$r_L,r_R$ 都连续、单调不减，并满足 $r_L(0)=r_R(0)=0$、$r_L(1)=r_R(1)=1$；凸组合 $r_s$ 因而也具有这些性质。于是 $H$ 连续，且

$$
H(t,0)=((\gamma\ast\eta)\ast\zeta)(t),
\qquad
H(t,1)=(\gamma\ast(\eta\ast\zeta))(t).
$$

端点始终满足 $H(0,s)=x_0=H(1,s)$，所以这是固定基点同伦。

所以

$$
\bigl(\lbrack\gamma\rbrack\cdot\lbrack\eta\rbrack\bigr)\cdot\lbrack\zeta\rbrack
=\lbrack\gamma\rbrack\cdot\bigl(\lbrack\eta\rbrack\cdot\lbrack\zeta\rbrack\bigr).
$$

### A.4 单位元和逆元

常值环路 $c_{x_0}$ 满足

$$
(\gamma\ast c_{x_0})(t)
=
\begin{cases}
\gamma(2t),&0\leq t\leq\frac12,\\\\
x_0,&\frac12\leq t\leq1.
\end{cases}
$$

它与 $\gamma$ 只差一个“走完 $\gamma$ 后在终点停留”的速度变化。取

$$
H(t,s)=
\begin{cases}
\gamma\left(\dfrac{2t}{1+s}\right),
&0\leq t\leq\dfrac{1+s}{2},\\\\
x_0,
&\dfrac{1+s}{2}\leq t\leq1,
\end{cases}
$$

可见

$$
H(t,0)=(\gamma\ast c_{x_0})(t),
\qquad
H(t,1)=\gamma(t).
$$

因此 $[\gamma][c_{x_0}]=[\gamma]$。右单位元也可以写出同样具体的同伦：令

$$
r_0(t)=
\begin{cases}
0,&0\leq t\leq\frac12,\\
2t-1,&\frac12\leq t\leq1,
\end{cases}
\qquad
r_s(t)=(1-s)r_0(t)+st,
$$

并定义 $H_R(t,s)=\gamma(r_s(t))$。在 $t=\frac12$ 处两段的值都为 $s/2$，所以 $r_s$ 连续；而且 $r_s(0)=0,r_s(1)=1$。于是

$$
H_R(t,0)=(c_{x_0}\ast\gamma)(t),
\qquad
H_R(t,1)=\gamma(t),
$$

并且 $H_R(0,s)=H_R(1,s)=x_0$。故 $[c_{x_0}][\gamma]=[\gamma]$。

对于逆元，$\gamma\ast\gamma^{-1}$ 先沿原路径前进，再沿原路返回。定义

$$
H(t,s)=
\begin{cases}
\gamma\bigl((1-s)2t\bigr),
&0\leq t\leq\frac12,\\\\
\gamma\bigl((1-s)(2-2t)\bigr),
&\frac12\leq t\leq1.
\end{cases}
$$

当 $s=0$ 时这是 $\gamma\ast\gamma^{-1}$；当 $s=1$ 时两段都恒等于 $\gamma(0)=x_0$。端点始终固定，所以

$$
[\gamma][\gamma^{-1}]=[c_{x_0}].
$$

这就逐项验证了基本群的群公理。

## B. 圆周基本群的完整证明

### B.1 路径提升

仍取

$$
p(u)=(\cos 2\pi u,\sin 2\pi u).
$$

设 $\alpha:[0,1]\to S^1$ 连续，且给定 $u_0\in\mathbb{R}$ 满足 $p(u_0)=\alpha(0)$。我们证明存在唯一连续函数 $\widetilde\alpha:[0,1]\to\mathbb{R}$，使

$$
\widetilde\alpha(0)=u_0,
\qquad
p(\widetilde\alpha(t))=\alpha(t).
$$

先取圆周上长度严格小于 $2\pi$ 的开弧 $U$。在 $U$ 上可以连续选择辐角函数 $\theta_U:U\to\mathbb{R}$，满足

$$
p\left(\frac{\theta_U(z)}{2\pi}\right)=z.
$$

令

$$
q_U(z)=\frac{\theta_U(z)}{2\pi}.
$$

那么 $q_U$ 是 $p$ 在这一段上的连续局部逆映射；$q_U(U)+n=\lbrace q_U(z)+n:z\in U\rbrace$ 是它的整数平移，其中 $n\in\mathbb{Z}$。

因此每个这样的开弧都被 $p$ 的若干互不相交的开区间同胚覆盖：对某个分支区间 $I$，限制 $p|_I:I\to U$ 的逆就是 $q_U$ 的一个整数平移。长度小于 $2\pi$ 的开弧覆盖整个圆周，所以 $p$ 确实是覆盖映射。

由于 $\alpha$ 在紧区间 $[0,1]$ 上一致连续，可以取有限分割

$$
0=t_0<t_1<\cdots<t_m=1
$$

使每个 $\alpha([t_{j-1},t_j])$ 都包含在某个开弧 $U_j$ 中。

在第一段 $[t_0,t_1]$ 上，$q_{U_1}\circ\alpha$ 是一个提升。因为

$$
p(q_{U_1}(\alpha(t_0)))=p(u_0),
$$

两边只相差一个整数，所以存在唯一 $n_1\in\mathbb{Z}$，使

$$
q_{U_1}(\alpha(t_0))+n_1=u_0.
$$

定义

$$
\widetilde\alpha(t)
=q_{U_1}(\alpha(t))+n_1,
\qquad t\in[t_0,t_1].
$$

假设已经构造到 $[0,t_{j-1}]$，并且已知 $\widetilde\alpha(t_{j-1})$。在下一段上选唯一整数 $n_j$ 满足

$$
q_{U_j}(\alpha(t_{j-1}))+n_j
=\widetilde\alpha(t_{j-1}),
$$

然后令

$$
\widetilde\alpha(t)
=q_{U_j}(\alpha(t))+n_j,
\qquad t\in[t_{j-1},t_j].
$$

相邻两段在公共端点取值一致，因而拼接后得到连续提升。

唯一性也按同样的分割归纳。两条提升在 $t_0$ 处相等；在第一段中它们必须落在同一个平移区间 $q_{U_1}(U_1)+n_1$，所以相等。若已经在 $[0,t_{j-1}]$ 上相等，则在 $t_{j-1}$ 处相等，下一段使用的整数平移也唯一，因此两条提升在下一段相等。归纳得到整个区间上相等。

### B.2 绕行数

设 $\gamma$ 是基于 $x_0=p(0)$ 的环路，取从 $0$ 出发的唯一提升 $\widetilde\gamma$。因为 $\gamma(1)=x_0=p(0)$，所以 $p(\widetilde\gamma(1))=p(0)$。

先验证这里使用的整数判别。若 $p(u)=p(v)$，则

$$
\cos(2\pi u-2\pi v)=1,
\qquad
\sin(2\pi u-2\pi v)=0.
$$

基本三角函数的周期性告诉我们这等价于 $2\pi(u-v)=2\pi k$（某个 $k\in\mathbb{Z}$），即 $u-v\in\mathbb{Z}$；反向显然成立。因此

$$
p(u)=p(v)\Longleftrightarrow u-v\in\mathbb{Z},
$$

从而 $\widetilde\gamma(1)\in\mathbb{Z}$。定义

$$
\operatorname{wind}(\gamma)=\widetilde\gamma(1).
$$

现在说明同伦提升确实连续。设 $H:[0,1]^2\to S^1$ 是从 $\gamma$ 到 $\eta$ 的固定基点同伦。我们使用紧性度量空间的 Lebesgue 数性质：给定有限开覆盖，存在 $\delta>0$，使直径小于 $\delta$ 的任意子集都落在覆盖中的某个成员内。圆周的有限个短开弧构成 $S^1$ 的开覆盖；由此并结合 $H$ 的一致连续性，取足够细的矩形网格，使每个小矩形 $R_{ij}$ 的像 $H(R_{ij})$ 落在某个短开弧 $U_{ij}$ 中。

在每个 $U_{ij}$ 上取局部逆 $q_{ij}$。从左下角开始逐格定义

$$
\widetilde H|\_{R\_{ij}}=q\_{ij}\circ H+n\_{ij},
\qquad n_{ij}\in\mathbb{Z},
$$

其中整数 $n_{ij}$ 选到使该公式在一个已经构造的公共边（第一格用左下角值 $0$）处相等。若两条局部提升在公共边的一点相等，那么它们在整条公共边上相等：两者之差连续且取值于离散集 $\mathbb{Z}$，故在连通的公共边上为常数。于是相邻小矩形的定义可以无缝拼接。这里用到的粘贴引理是：有限个闭集覆盖一个空间时，只要每块上的连续函数在重叠处相等，拼在一起的函数仍连续。有限次使用该引理得到全局连续的 $\widetilde H$，满足

$$
p(\widetilde H(t,s))=H(t,s),
\qquad
\widetilde H(0,0)=0.
$$

由于 $H(0,s)=x_0$，第一列矩形上的同样唯一性说明 $\widetilde H(0,s)=0$；因此我们得到以左边界恒为 $0$ 的提升。唯一性仍由“同一点处相等后，差值是连通边上的整数值常函数”逐格推出。

由于 $H(1,s)=x_0$，有 $\widetilde H(1,s)\in\mathbb{Z}$。函数 $s\mapsto\widetilde H(1,s)$ 连续，而 $[0,1]$ 连通、$\mathbb{Z}$ 离散，所以它是常值函数。因此

$$
\operatorname{wind}(\gamma)=\operatorname{wind}(\eta).
$$

### B.3 绕行数的加法性质

令

$$
\operatorname{wind}(\gamma)=n,
\qquad
\operatorname{wind}(\eta)=m.
$$

记相应提升为 $\widetilde\gamma$ 和 $\widetilde\eta$。定义

$$
\Lambda(t)=
\begin{cases}
\widetilde\gamma(2t),
&0\leq t\leq\frac12,\\\\
n+\widetilde\eta(2t-1),
&\frac12\leq t\leq1.
\end{cases}
$$

在中点处，两段都取值 $n$，所以 $\Lambda$ 连续；又因为 $p(u+n)=p(u)$，

$$
p(\Lambda(t))=(\gamma\ast\eta)(t).
$$

并且 $\Lambda(0)=0$。由提升的唯一性，$\Lambda$ 就是 $\gamma\ast\eta$ 的提升，于是

$$
\operatorname{wind}(\gamma\ast\eta)
=\Lambda(1)
=n+m.
$$

### B.4 满射与单射

对任意 $n\in\mathbb{Z}$，定义

$$
\gamma_n(t)=p(nt).
$$

它的提升为 $\widetilde\gamma_n(t)=nt$，所以

$$
\operatorname{wind}(\gamma_n)=n.
$$

绕行数映射是满射。

若 $\operatorname{wind}(\gamma)=0$，则

$$
\widetilde\gamma(0)=\widetilde\gamma(1)=0.
$$

定义

$$
K(t,s)=p\bigl((1-s)\widetilde\gamma(t)\bigr).
$$

当 $s=0$ 时 $K(t,0)=\gamma(t)$；当 $s=1$ 时 $K(t,1)=x_0$。由于提升的两个端点都是 $0$，对所有 $s$ 都有

$$
K(0,s)=K(1,s)=x_0.
$$

所以绕行数为零的环路是零同伦环路。

如果 $\gamma$ 和 $\eta$ 的绕行数相同，则

$$
\operatorname{wind}(\gamma\ast\eta^{-1})
=\operatorname{wind}(\gamma)-\operatorname{wind}(\eta)=0.
$$

因此 $\gamma\ast\eta^{-1}$ 零同伦，从而 $[\gamma]=[\eta]$。绕行数既满射又单射，并保持拼接与加法，故

$$
\pi_1(S^1,x_0)\cong(\mathbb{Z},+).
$$

## C. 八字形自由群的逐字列计算

### C.1 约化字

令字母表为

$$
\mathcal{A}=\lbrace a,a^{-1},b,b^{-1}\rbrace.
$$

这里 $a^{-1}$ 和 $b^{-1}$ 是 $a,b$ 的形式逆元；对字母 $x$，记 $x^{-1}$ 为交换正、逆两种符号（例如 $(a^{-1})^{-1}=a$）。

一个有限字列是 $\mathcal{A}$ 中元素的有限序列，例如

$$
w=a\ b^{-1}\ a.
$$

如果字列中没有相邻的 $a a^{-1}$、$a^{-1}a$、$bb^{-1}$ 或 $b^{-1}b$，就称它是约化的。约化规则是反复删除相邻逆元：

$$
aa^{-1}\to 1,\quad a^{-1}a\to1,\quad
bb^{-1}\to1,\quad b^{-1}b\to1.
$$

这里 $1$ 表示空字列，也就是群的单位元。

### C.2 约化结果唯一

每次删除一对相邻逆元都会减少字列长度，因此过程一定在有限步后停止。还需说明最终的约化字不依赖删除顺序。

设一个字列有两种一步删除选择。若两对可删除位置互不重叠，例如

$$
u\ x x^{-1}\ v\ y y^{-1}\ z,
$$

那么先删 $xx^{-1}$ 再删 $yy^{-1}$，或先删 $yy^{-1}$ 再删 $xx^{-1}$，都会得到 $u v z$。若两对位置重叠，则局部字列只能是

$$
x x^{-1}x
\quad\text{或}\quad
x^{-1}x x^{-1},
$$

其中 $x$ 是 $a$ 或 $b$；删左边一对与删右边一对都会留下同一个单字母 $x$（或 $x^{-1}$）。所以任意两个一步结果，都能在至多再一步后变成同一个字列。

现在按原字列长度归纳。长度为 $0$ 或 $1$ 时没有删除选择。对长度 $n$ 的字列 $w$，若两条完整删除过程的第一步相同，则对剩余的较短字列使用归纳假设；若第一步不同，上面的“不相交交换”或“重叠菱形”给出一个共同的中间字列，随后仍对更短字列使用归纳假设。因此所有最大删除过程都得到同一个约化字。记作 $\operatorname{red}(w)$。

### C.3 群运算

两个约化字 $u,v$ 的乘积定义为

$$
u\cdot v=\operatorname{red}(uv),
$$

其中 $uv$ 表示把两个字列直接首尾连接。单位元是空字列 $1$，字列的长度（字母个数）记作 $|w|$。字列

$$
w^{-1}
$$

通过先反转顺序、再把每个字母换成逆元得到。例如

$$
(a\ b^{-1}\ a)^{-1}
=a^{-1}\ b\ a^{-1}.
$$

由于约化形式唯一，连接三个字列时，无论先约化前两个还是先约化后两个，最终结果都等于原始连接字列的唯一约化形式。因此运算结合。直接连接一个字列与其逆序反字列，会逐步完全约去，得到空字列。于是约化字构成群 $F(a,b)$。

八字形中每条环路都可以切分为依次落在左圈和右圈上的若干段；每一段按方向对应 $a,a^{-1},b$ 或 $b^{-1}$。如果一段先沿某圈正向走、紧接着又沿同一圈反向走，就能在该圆盘邻域中缩掉，正对应约化规则。为了证明“没有其他同伦关系”，构造一个普遍覆盖图 $T$：

* $T$ 的顶点是所有约化字，根顶点是空字列 $1$；
* 对每个顶点 $w$ 和每个字母 $x\in\lbrace a,a^{-1},b,b^{-1}\rbrace$，连一条标记为 $x$ 的边，把 $w$ 连到 $\operatorname{red}(wx)$。同一条无向边也会由 $(w,x)$ 和 $(\operatorname{red}(wx),x^{-1})$ 两次描述，但只计作一条边。

若 $w$ 的最后一个字母不是 $x^{-1}$，则这条边把长度从 $|w|$ 增加到 $|w|+1$；若最后一个字母是 $x^{-1}$，则它删去最后一个字母，长度减少 $1$。因此每个非根顶点都有唯一的“父顶点”（删去最后一个字母），从根到 $w$ 也有唯一的路径，依次读取 $w$ 的字母。若存在环，取环上距根最远的顶点；环在该点相邻的两条边都不能增加长度，只能都指向唯一的父顶点，因而其实是同一条无向边，矛盾。故 $T$ 是树。

把 $T$ 的每条标记为 $a^{\pm1}$ 的边映到左圆周，把每条标记为 $b^{\pm1}$ 的边映到右圆周，并把所有顶点映到楔点 $x_0$，得到 $q:T\to S^1_a\vee S^1_b$。在 $T$ 的每个顶点，四个出发方向分别对应 $a,a^{-1},b,b^{-1}$；它们恰好一一对应八字形楔点的四个局部方向。因此 $q$ 在每个顶点邻域和每条边内部都是同胚，故是覆盖映射。树 $T$ 沿唯一父路径可以连续收缩到根；任意紧路径的像只涉及有限个边，所以这种收缩足以说明 $T$ 中的闭路是零同伦。因为覆盖空间 $T$ 是树（每个闭路都可缩），它也常被称为八字形的普遍覆盖。

给定八字形中的环路 $\gamma$，从根顶点 $1$ 唯一提升到 $T$。每穿过一条标记边，就在当前字列右侧添加相应字母并约化，因此提升终点正是 $\operatorname{red}(w_\gamma)$，其中 $w_\gamma$ 是环路读出的字列。覆盖映射的同伦提升性质（证明与 B.2 的有限网格完全相同）说明同伦环路有相同终点。反过来，若两个环路提升到同一终点，则 $\gamma\ast\eta^{-1}$ 的提升是 $T$ 中的闭路，因 $T$ 是树而零同伦，故 $[\gamma]=[\eta]$。每个约化字都由相应的边路径实现，所以得到环路同伦类与约化字的一一对应，进而

$$
\pi_1(S^1_a\vee S^1_b,x_0)\cong F(a,b).
$$

## D. van Kampen 定理：从群胚证明到计算规则

### D.1 路径群胚和粘合对象

只用一个基点时，路径在交集中的端点未必等于基点；因此先把所有点同时作为对象。对空间 $Y$，定义**基本群胚** $\Pi_1(Y)$：对象是 $Y$ 的点，从 $y$ 到 $z$ 的态射是所有连接 $y,z$ 的路径按固定端点同伦后的等价类 $[\alpha]$；态射复合仍由首尾拼接给出，逆态射由反向路径给出。

令 $A=U\cap V$。由包含映射得到群胚 $\Pi_1(A)\to\Pi_1(U)$ 和 $\Pi_1(A)\to\Pi_1(V)$。把 $U$、$V$ 中的态射交替写成字列，并加入三类关系：同一群胚内的复合关系；逆态射与单位态射关系；以及对 $A$ 中的同一条路径，在 $U$ 和 $V$ 中的两个副本相等。所得群胚记为

$$
\mathcal P=\Pi_1(U)\amalg_{\Pi_1(A)}\Pi_1(V).
$$

把路径看成 $X=U\cup V$ 中的路径，得到自然函子

$$
F:\mathcal P\longrightarrow\Pi_1(X).
$$

### D.2 满射：把任意路径切成有限段

取 $X$ 中一条路径 $\alpha:[0,1]\to X$。由于 $U,V$ 开，$\alpha^{-1}(U)$ 与 $\alpha^{-1}(V)$ 是区间 $[0,1]$ 的开集并覆盖它。紧性给出有限个小区间，使每个小区间的像完全落在 $U$ 或 $V$ 中；在两个标签发生改变的地方，可以利用 $U\cap V$ 的开性把分界点移到一个仍落在交集中的小区间内。于是存在分割

$$
0=t_0<t_1<\cdots<t_n=1
$$

使每段 $\alpha_j=\alpha|\_{[t\_{j-1},t\_j]}$ 完全落在 $U$ 或 $V$，且相邻段的端点在需要换集合时属于 $A$。每段都是相应群胚中的态射，因此

$$
[\alpha]=[\alpha_1]\cdots[\alpha_n]
$$

在 $\mathcal P$ 中有代表元。这说明 $F$ 在每个端点之间都是满射。

### D.3 单射：用同伦方形逐格剥离

设 $\omega$ 是 $\mathcal P$ 中的交替字列，并且 $F(\omega)$ 在 $\Pi_1(X)$ 中等于单位态射。取一个固定端点同伦

$$
H:[0,1]^2\to X
$$

把它变到常值路径。由 $H$ 的一致连续性和紧方形的 Lebesgue 数，取足够细的矩形网格，使每个小矩形的像全部落在 $U$ 或全部落在 $V$ 中；同时细分底边，使其依次正好对应 $\omega$ 的各段。

若一个小矩形的像在 $U$ 中，那么其下边和上边是 $U$ 中同端点同伦的路径，可以在 $\Pi_1(U)$ 内互换；若像在 $V$ 中，则在 $\Pi_1(V)$ 内互换。逐行把底边替换成顶边时，相邻小矩形共享的内部边以相反方向出现，因而在群胚字列中抵消；当一条共享边的像落在 $A$ 中时，它还可以用定义中的交集关系在 $U$、$V$ 两侧互换。有限个小矩形全部剥离后，底边字列被化为顶端的常值态射，所以 $\omega$ 在 $\mathcal P$ 中本来就是单位态射。故 $F$ 也是单射。

因此 $F$ 是群胚同构。由于 $U,V,A$ 道路连通，各基本群胚都是连通群胚；选定 $x_0\in A$ 后，选一条 $A$ 中路径 $\tau_y$ 从 $x_0$ 到每个 $y\in A$，即可把任意态射 $[\alpha]:y\to z$ 表成基点处的环路

$$
[\tau_y\ast\alpha\ast\tau_z^{-1}].
$$

这把上述群胚粘合在 $x_0$ 处识别为群的粘合，得到

$$
\pi_1(X,x_0)
\cong
\pi_1(U,x_0)\ast_{\pi_1(A,x_0)}\pi_1(V,x_0).
$$

### D.4 计算规则

设

$$
G=\pi_1(U,x_0),\qquad K=\pi_1(A,x_0),\qquad L=\pi_1(V,x_0),
$$

交集包含映射诱导 $\varphi:K\to G$ 与 $\psi:K\to L$。自由积 $G\ast L$ 的元素是两因子元素交替组成的字列；带 amalgamation 的自由积定义为

$$
G\ast_K L
:=
(G\ast L)\big/\left\langle\!\left\langle
\varphi(k)\psi(k)^{-1}:k\in K
\right\rangle\!\right\rangle.
$$

双尖括号表示由这些元素生成的正规子群，取商正是强制 $\varphi(k)=\psi(k)$。实际计算时按顺序：分别计算 $G,K,L$；写出交集生成环路在两边的像；取自由积；加入这些识别关系；最后化简。

若 $A$ 可缩，则 $K=1$，结果简化为 $\pi_1(U)\ast\pi_1(V)$。八字形因此给出 $\mathbb{Z}\ast\mathbb{Z}=F(a,b)$。对实射影平面，交集中的边界环路在圆盘一侧是 $a^2$，在圆盘内部是单位元，故关系为 $a^2=1$。

## E. 实射影平面的 CW 分解与关系式

把边界圆周上的相反点识别的商映射记为 $q:D^2\to\mathbb{R}P^2$。商空间有一个标准的 CW 分解

$$
\mathbb{R}P^2=e^0\cup e^1\cup e^2.
$$

CW 分解的意思是先放入 0-胞腔（点），再沿端点附着 1-胞腔（开区间），再沿边界附着 2-胞腔（开圆盘），每一步都用连续映射把新胞腔的边界粘到已有部分。这里 $e^0$ 是圆盘中心的像，$e^1$ 是边界商圆周去掉一个点后的开弧再补上端点，$e^2$ 是圆盘内部的像。先看一维骨架 $X^1=e^0\cup e^1$。它同胚于一个圆周，所以

$$
\pi_1(X^1,x_0)=\langle a\rangle\cong\mathbb Z.
$$

其中 $\langle a\rangle$ 表示由 $a$ 及其整数次幂组成的无限循环群。

边界附着映射是 $q|_{\partial D^2}:S^1\to X^1$。把商圆周参数化为

$$
[e^{i\theta}]\longmapsto e^{2i\theta},
$$

其中 $[e^{i\theta}]$ 表示 $e^{i\theta}$ 与 $-e^{i\theta}$ 的等价类；这个公式在 $\theta$ 增加 $\pi$ 时不变，因而定义良好。在该参数化下，附着映射就是

$$
e^{i\theta}\longmapsto e^{2i\theta},
$$

即绕一维骨架两圈，代表基本群中的 $a^2$。

取 $U$ 为一维骨架的开邻域，取 $V$ 为二胞腔连同边界的开邻域，使 $U\cap V$ 形变收缩到附着圆周。$V$ 可缩，$\pi_1(V)=1$；交集的生成环路在 $U$ 中映为 $a^2$，在 $V$ 中映为单位元。D.4 的粘合公式因此给出

$$
\pi_1(\mathbb{R}P^2,x_0)
\cong
\langle a\rangle/\langle\!\langle a^2\rangle\!\rangle
=\langle a\mid a^2=1\rangle
\cong\mathbb Z/2\mathbb Z.
$$

这一步明确说明了“二维胞腔增加关系”：附着圆周的同伦类被填进一个可缩的圆盘，因此必须在基本群中变成单位元。

## F. 为什么 $H_1$ 是基本群的阿贝尔化

本节把范围明确为道路连通单纯复形 $X=|K|$，其中 $K$ 是抽象单纯复形，$|K|$ 表示把它的顶点、边、三角形等按面关系实际粘合得到的几何空间。记 $K^i$ 为 $K$ 的 $i$-骨架；$C_i(K;\mathbb Z)$ 是由每条几何 $i$-单纯形任选一个方向后生成的自由 Abel 群（反向取负号），边界算子 $\partial_i:C_i\to C_{i-1}$ 按交替删去顶点定义。于是

$$
H_1(X;\mathbb Z)=\ker\partial_1/\operatorname{im}\partial_2;
$$

分子是 1-循环群，分母是由三角形边界生成的 1-边界群。

这里 $\ker$ 表示被映到零链的元素集合，$\operatorname{im}$ 表示所有像元素组成的集合；$\mathbb Z^{E_{\mathrm{nt}}}$ 表示以非树边集合 $E_{\mathrm{nt}}$ 为基的自由 Abel 群，即只有有限个坐标非零的整数列。

选取 $K^1$ 的极大树 $T$，并选定基点顶点 $v_0$。树是连通且无环的子图，因此每个顶点 $v$ 在 $T$ 中都有唯一从 $v_0$ 到它的路径，记为 $\tau_v$。对每条不在 $T$ 中的有向边 $e:v\to w$，定义基点环路

$$
\ell_e=\tau_v\ast e\ast\tau_w^{-1}.
$$

记所有不在 $T$ 中的有向边组成的集合为 $E_{\mathrm{nt}}$（下标 $\mathrm{nt}$ 是 non-tree 的缩写）。

把树 $T$ 连续收缩为一点，$K^1/T$ 变成由每条非树边产生的一圈的楔和；因此

$$
\pi_1(K^1,v_0)\cong F(\ell_e\mid e\in E_{\mathrm{nt}}),
$$

其中 $F(S)$ 表示以集合 $S$ 为自由生成元的自由群。

每个 2-单纯形 $\sigma$ 的边界沿三个有向边行走，代入对应的 $\ell_e$ 并约去树路径，得到自由群中的关系字 $r_\sigma$。逐个附着 2-单纯形并应用 D 的粘合证明，得到
每个 2-单纯形 $\sigma$ 的边界沿三个有向边行走，代入对应的 $\ell_e$ 并约去树路径，得到自由群中的关系字 $r_\sigma$。逐个附着 2-单纯形并应用 D 的粘合证明，得到

$$
\pi_1(X,v_0)\cong
F(\ell_e\mid e\in E_{\mathrm{nt}})/\langle\!\langle r_\sigma\rangle\!\rangle.
$$

高于 2 维的单纯形不会给基本群增加新的关系，因为它们的边界维数至少为 2；等价地，基本群只由 2-骨架决定。

现在对上式阿贝尔化。自由群的阿贝尔化把每个 $\ell_e$ 变成一个整数坐标，因此

$$
\pi\_1(X,v\_0)\_{\mathrm{ab}}
\cong
\mathbb Z^{E\_{\mathrm{nt}}}/\langle \operatorname{exp}(r_\sigma)\rangle.
$$

这里 $\operatorname{exp}(r_\sigma)$ 是关系字中各非树边的指数和向量：某条边正向出现贡献 $+1$，反向出现贡献 $-1$，重复出现则累加。

另一方面，对每条非树边 $e:v\to w$，令 $P_e$ 是树中从 $w$ 回到 $v$ 的唯一有向路径，并定义基本 1-循环

$$
c_e=e+P_e.
$$

它的边界为零。若 $z\in\ker\partial_1$，记 $n_e$ 为 $z$ 中非树边 $e$ 的系数，则 $z-\sum_e n_e c_e$ 只含树边且仍是循环。树没有环，任何只由树边组成的有限 1-循环都必须为零（从一个有限支撑子树的叶顶点开始，边界系数迫使该叶边系数为零，再逐叶剥离）。因此

$$
z=\sum_{e\notin T}n_e c_e,
$$

且表示唯一；“取非树边系数”给出同构

$$
\ker\partial_1\cong\mathbb Z^{E_{\mathrm{nt}}}.
$$

对一个 2-单纯形 $\sigma$，其链边界 $\partial_2\sigma$ 的非树边系数恰好就是边界字 $r_\sigma$ 的指数和向量；树边只负责把三个顶点首尾接起来，不改变非树坐标。因此在上述同构下，$\operatorname{im}\partial_2$ 正好对应于由所有 $\operatorname{exp}(r_\sigma)$ 生成的子群。取商得到自然同构

$$
H_1(X;\mathbb Z)
=\ker\partial_1/\operatorname{im}\partial_2
\cong
\pi_1(X,v_0)_{\mathrm{ab}}.
$$

这个证明也解释了八字形和实射影平面的例子：八字形没有 2-单纯形关系，留下两个独立整数坐标；实射影平面的唯一 2-胞腔边界贡献关系 $2a=0$，所以得到 $\mathbb Z/2\mathbb Z$。
