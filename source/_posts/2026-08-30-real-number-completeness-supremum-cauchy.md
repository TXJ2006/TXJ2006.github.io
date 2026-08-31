---
title: "极限为何存在：实数的完备性、上确界与 Cauchy 列"
date: 2026-08-30 22:00:00
categories:
  - 基础数学
  - 分析学
tags:
  - 数学基础
  - 分析学
  - 实数
  - 完备性
  - 极限
  - Cauchy 列
mathjax: true
toc: true
toc_number: false
comments: true
---

## 极限是什么

数列是按正整数下标排列的一串数：

$$
x&#95;1,x&#95;2,x&#95;3,\ldots.
$$

下标 $n$ 表示项的位置，$x&#95;n$ 表示第 $n$ 项。给定实数 $L$，称数列 $(x&#95;n)$ **收敛到** $L$，如果

$$
(\forall\varepsilon>0)(\exists N)
(\forall n\geq N)
\quad
|x&#95;n-L|<\varepsilon.
$$

这里 $N$ 与 $n$ 都取正整数。符号 $\forall$ 表示“对每一个”，$\exists$ 表示“存在”。正数 $\varepsilon$ 表示预先给定的误差，$N$ 是允许依赖于 $\varepsilon$ 的起始位置，$|x&#95;n-L|$ 表示第 $n$ 项与 $L$ 在数轴上的距离。

固定一个 $\varepsilon>0$ 后，不等式

$$
|x&#95;n-L|<\varepsilon
$$

等价于

$$
L-\varepsilon<x&#95;n<L+\varepsilon.
$$

所以极限条件的含义是：无论围绕 $L$ 取多窄的区间 $(L-\varepsilon,L+\varepsilon)$，总能找到一个位置 $N$，使从

$$
x&#95;N,x&#95;{N+1},x&#95;{N+2},\ldots
$$

开始的所有项都留在这个区间内。这一串从第 $N$ 项开始的数称为数列的**尾部**。

数列的项不必等于 $L$，相邻两项到 $L$ 的距离也不必逐次减小。定义要求的是：每一个精度 $\varepsilon$ 一旦给定，数列的整个尾部最终进入相应区间，并且以后不再离开。此时记作

$$
x&#95;n\longrightarrow L
$$

或

$$
\lim&#95;{n\to\infty}x&#95;n=L.
$$

符号 $n\to\infty$ 表示正整数下标 $n$ 不断增大，$\lim$ 是英文 limit 的缩写。

这个定义给出了判断 $L$ 是否为极限的标准。它还没有保证满足条件的 $L$ 一定存在，也没有保证这个 $L$ 属于当前使用的数系。极限的定义与极限的存在是两个相接的问题；实数的完备性处理后一个问题。

方程

$$
x^2=2
$$

在有理数中没有解。这个缺口可以从两个方向看见。

先考虑所有平方小于 $2$ 的非负有理数：

$$
A
:=
\lbrace q\in\mathbb Q:q\geq0,\ q^2<2\rbrace.
$$

集合 $A$ 有上界。例如每个 $q\in A$ 都小于 $2$。但在有理数中，无论怎样选择一个上界，总能继续向左或向右微调；没有一个有理数恰好位于这道分界线上。集合已经把分界的位置确定下来，$\mathbb Q$ 中却没有元素能够占据这个位置。

沿着这道分界逐位逼近，会得到一列彼此越来越接近的有理数。数列的后项之间可以任意接近，它在 $\mathbb Q$ 中仍然没有极限。集合语言把问题写成“上确界是否存在”，数列语言把问题写成“Cauchy 列是否收敛”；二者描述的是同一个缺失的数。

实数的完备性同时填补这两种缺口。把取值范围扩充到 $\mathbb R$ 后，所有平方小于 $2$ 的非负实数组成的集合存在最小上界，这个上确界就是 $\sqrt2$。同一个性质还会保证单调有界数列收敛、闭区间套有公共点、每个有界数列存在收敛子列，以及每个 Cauchy 列都收敛。分析学中的许多“存在”都来自这里。

<!--more-->

## 1. 有序域：实数的代数与次序

先固定四个基本数集：

$$
\mathbb N
=
\lbrace1,2,3,\ldots\rbrace
$$

是正整数集，$\mathbb Z$ 是整数集，

$$
\mathbb Q
=
\left\lbrace
\frac mn:m\in\mathbb Z,\ n\in\mathbb N
\right\rbrace
$$

是有理数集，$\mathbb R$ 是实数集。

有理数和实数都可以进行加法、减法、乘法和除法，其中除数不能为零。抽象地说，它们都是**域**：加法与乘法满足结合律和交换律；存在加法单位元 $0$ 与乘法单位元 $1$；每个元素 $x$ 有加法逆元 $-x$；每个非零元素 $x$ 有乘法逆元 $x^{-1}$；乘法对加法满足分配律。

它们还带有全序 $\leq$。全序满足：

1. $x\leq x$；
2. 若 $x\leq y$ 且 $y\leq x$，则 $x=y$；
3. 若 $x\leq y$ 且 $y\leq z$，则 $x\leq z$；
4. 任意 $x,y$ 都满足 $x\leq y$ 或 $y\leq x$。

次序还要与运算相容。若 $x\leq y$，则对任意 $z$，

$$
x+z\leq y+z;
$$

若 $0\leq x$ 且 $0\leq y$，则

$$
0\leq xy.
$$

满足这些条件的域称为**有序域**。$\mathbb Q$ 和 $\mathbb R$ 都满足上述代数与次序公理；完备性把二者区分开来。

实数 $x$ 的绝对值定义为

$$
|x|
:=
\begin{cases}
x,&x\geq0,\\\\
-x,&x<0.
\end{cases}
$$

$|x-y|$ 表示数轴上 $x$ 与 $y$ 的距离。绝对值满足三角不等式

$$
|x+y|\leq|x|+|y|.
$$

附录 A 会从有序域公理证明这条不等式。

## 2. 上界、最大值与上确界

设 $E\subseteq\mathbb R$。若存在 $M\in\mathbb R$，使得

$$
(\forall x\in E)\quad x\leq M,
$$

则称 $E$ **有上界**，$M$ 称为 $E$ 的一个上界。符号 $\forall$ 表示“对每一个”。

如果 $M$ 还属于 $E$，则 $M$ 是 $E$ 的最大值，记作

$$
M=\max E.
$$

最大值可能不存在。开区间

$$
(0,1)
=
\lbrace x\in\mathbb R:0<x<1\rbrace
$$

有许多上界，$1$ 是其中最小的一个，但 $1\notin(0,1)$，所以这个集合没有最大值。

### 2.1 最小上界

若实数 $\alpha$ 满足：

1. $\alpha$ 是 $E$ 的上界；
2. 每个小于 $\alpha$ 的数都不是 $E$ 的上界；

则称 $\alpha$ 是 $E$ 的**上确界**，记作

$$
\alpha=\sup E.
$$

上确界与最大值的区别在于，$\sup E$ 不必属于 $E$。若最大值存在，则它一定等于上确界。

**命题 2.1**　一个集合至多有一个上确界。

**证明。** 假设 $\alpha$ 与 $\beta$ 都是 $E$ 的上确界。因为 $\alpha$ 是最小上界，而 $\beta$ 是一个上界，所以 $\alpha\leq\beta$。交换二者的地位，同理得到 $\beta\leq\alpha$。全序的反对称性给出 $\alpha=\beta$。证毕。

### 2.2 用 $\varepsilon$ 刻画上确界

希腊字母 $\varepsilon$ 表示任意给定的正实数。它通常用来描述可以任意缩小的误差。

**命题 2.2**　设 $E$ 非空且 $\alpha$ 是 $E$ 的上界。则

$$
\alpha=\sup E
$$

当且仅当

$$
(\forall\varepsilon>0)(\exists x\in E)
\quad
\alpha-\varepsilon<x\leq\alpha.
$$

符号 $\exists$ 表示“至少存在一个”。

**证明。** 先设 $\alpha=\sup E$。对任意 $\varepsilon>0$，数 $\alpha-\varepsilon$ 严格小于最小上界，因此它不是上界。于是存在 $x\in E$ 使 $x>\alpha-\varepsilon$。又因为 $\alpha$ 是上界，所以 $x\leq\alpha$。

反过来，设上式对每个 $\varepsilon>0$ 成立。若存在一个上界 $M<\alpha$，取

$$
\varepsilon:=\alpha-M>0.
$$

按照假设，存在 $x\in E$ 满足 $x>\alpha-\varepsilon=M$，这与 $M$ 是上界矛盾。因此没有比 $\alpha$ 更小的上界，故 $\alpha=\sup E$。证毕。

下界与下确界完全对偶。若

$$
(\forall x\in E)\quad m\leq x,
$$

则 $m$ 是下界；最大的下界称为下确界，记作 $\inf E$。附录 B 会证明

$$
\inf E=-\sup(-E),
$$

其中

$$
-E:=\lbrace-x:x\in E\rbrace.
$$

## 3. 有理数中的缺口

回到集合

$$
A
=
\lbrace q\in\mathbb Q:q\geq0,\ q^2<2\rbrace.
$$

它非空，因为 $1\in A$；它在 $\mathbb Q$ 中有上界，因为每个 $q\in A$ 都满足 $q<2$。

这里“在 $\mathbb Q$ 中”表示集合元素、上界以及可能的上确界都只能取有理数。

**定理 3.1**　集合 $A$ 在有理数中没有上确界。

**证明。** 假设存在 $s\in\mathbb Q$ 使 $s=\sup A$。因为 $1\in A$ 且 $2$ 是上界，所以

$$
1\leq s\leq2.
$$

附录 C.1 将证明不存在有理数 $s$ 满足 $s^2=2$，因此只剩 $s^2<2$ 或 $s^2>2$ 两种情形。

若 $s^2<2$，取有理数

$$
\delta
:=
\frac{2-s^2}{2s+2}>0.
$$

由 $1\leq s$ 可知 $0<\delta<1$。于是

$$
\begin{aligned}
(s+\delta)^2-s^2
&=2s\delta+\delta^2\\\\
&<(2s+1)\delta\\\\
&<2-s^2.
\end{aligned}
$$

所以 $(s+\delta)^2<2$，即 $s+\delta\in A$。但 $s+\delta>s$，与 $s$ 是上界矛盾。

若 $s^2>2$，取

$$
\delta
:=
\frac{s^2-2}{2s}>0.
$$

因为

$$
\begin{aligned}
s-\delta
&=s-\frac{s^2-2}{2s}\\\\
&=\frac{s^2+2}{2s}\\\\
&>0,
\end{aligned}
$$

并且

$$
\begin{aligned}
(s-\delta)^2
&=s^2-2s\delta+\delta^2\\\\
&=2+\delta^2\\\\
&>2.
\end{aligned}
$$

若 $q\in A$ 且 $q\geq s-\delta$，则 $q^2\geq(s-\delta)^2>2$，与 $q\in A$ 矛盾。因此每个 $q\in A$ 都满足 $q<s-\delta$，说明 $s-\delta$ 是一个比 $s$ 更小的上界，也产生矛盾。

两种情形都不成立，所以 $A$ 在 $\mathbb Q$ 中没有上确界。证毕。

![有理数逼近平方根二形成的缺口](/images/notes/assets/mathematical-foundations/real-completeness-cut.svg)

## 4. 实数的完备性公理

实数与有理数之间的根本差别可以写成一条公理。

**上确界公理**　每个非空且有上界的实数子集都有实数上确界。

也就是说，若

$$
\varnothing\neq E\subseteq\mathbb R
$$

且 $E$ 有上界，则存在唯一的 $\alpha\in\mathbb R$ 满足

$$
\alpha=\sup E.
$$

满足上确界公理的有序域称为**完备有序域**。

### 4.1 用上确界得到 $\sqrt2$

在 $\mathbb R$ 中定义

$$
B
:=
\lbrace x\in\mathbb R:x\geq0,\ x^2<2\rbrace.
$$

集合 $B$ 非空，因为 $1\in B$；它有上界，因为每个 $x\in B$ 都满足 $x<2$。上确界公理保证存在

$$
\alpha:=\sup B\in\mathbb R.
$$

因为 $1\in B$ 且 $2$ 是上界，所以 $1\leq\alpha\leq2$。下面证明 $\alpha^2=2$。

若 $\alpha^2<2$，取

$$
\delta
:=
\frac{2-\alpha^2}{2\alpha+2}>0.
$$

由 $1\leq\alpha$ 可知 $0<\delta<1$，从而

$$
\begin{aligned}
(\alpha+\delta)^2-\alpha^2
&=2\alpha\delta+\delta^2\\\\
&<(2\alpha+1)\delta\\\\
&<2-\alpha^2.
\end{aligned}
$$

因此 $(\alpha+\delta)^2<2$，即 $\alpha+\delta\in B$。但 $\alpha+\delta>\alpha$，与 $\alpha$ 是 $B$ 的上界矛盾。

若 $\alpha^2>2$，取

$$
\delta
:=
\frac{\alpha^2-2}{2\alpha}>0.
$$

此时

$$
\begin{aligned}
\alpha-\delta
&=\frac{\alpha^2+2}{2\alpha}\\\\
&>0
\end{aligned}
$$

且

$$
\begin{aligned}
(\alpha-\delta)^2
&=\alpha^2-2\alpha\delta+\delta^2\\\\
&=2+\delta^2\\\\
&>2.
\end{aligned}
$$

若 $x\in B$ 且 $x\geq\alpha-\delta$，则 $x^2\geq(\alpha-\delta)^2>2$，与 $x\in B$ 矛盾。因此 $\alpha-\delta$ 是 $B$ 的上界，而且 $\alpha-\delta<\alpha$，这与 $\alpha$ 是最小上界矛盾。

两种不等情形都不成立，所以

$$
\alpha^2=2.
$$

取非负平方根的记号后，

$$
\sqrt2:=\alpha=\sup B.
$$

完备性由集合 $B$ 的上确界产生了实数 $\sqrt2$。

## 5. Archimedean 性质与有理数稠密性

完备性会限制自然数在实数轴中的位置。

**定理 5.1（Archimedean 性质）**　对每个 $x\in\mathbb R$，都存在 $n\in\mathbb N$ 使

$$
n>x.
$$

**证明。** 假设结论不成立，则 $\mathbb N$ 在 $\mathbb R$ 中有上界。由完备性，存在

$$
\alpha:=\sup\mathbb N.
$$

数 $\alpha-1$ 小于上确界，所以它不是 $\mathbb N$ 的上界。存在 $n\in\mathbb N$ 使

$$
n>\alpha-1.
$$

于是 $n+1>\alpha$。但 $n+1\in\mathbb N$，这与 $\alpha$ 是 $\mathbb N$ 的上界矛盾。故 $\mathbb N$ 没有实数上界。证毕。

一个直接推论是

$$
\lim&#95;{n\to\infty}\frac1n=0.
$$

符号 $n\to\infty$ 表示让正整数 $n$ 任意增大。严格地说，对每个 $\varepsilon>0$，Archimedean 性质保证存在 $N\in\mathbb N$ 使 $N>1/\varepsilon$。当 $n\geq N$ 时，

$$
0<\frac1n\leq\frac1N<\varepsilon.
$$

这正是数列 $1/n$ 收敛到 $0$ 的定义，下一节会正式给出。

**定理 5.2（有理数的稠密性）**　若 $x,y\in\mathbb R$ 且 $x<y$，则存在 $q\in\mathbb Q$ 使

$$
x<q<y.
$$

证明的关键是先用 Archimedean 性质选择 $n$ 使 $n(y-x)>1$，再在 $nx$ 与 $ny$ 之间找到一个整数。完整构造见附录 C.3。

稠密性说明，任意两个不同实数之间都有有理数。有理数可以无限逼近 $\sqrt2$；$\sqrt2\notin\mathbb Q$ 又说明这种逼近不会在 $\mathbb Q$ 内取得极限。

### 5.1 稠密性为何仍允许缺口

给定任意 $\varepsilon>0$，有理数稠密性保证存在 $q\in\mathbb Q$ 使

$$
\sqrt2-\varepsilon<q<\sqrt2.
$$

因此有理数能够以任意精度接近 $\sqrt2$。精度无论多高，所选的 $q$ 仍然小于 $\sqrt2$，而 $\sqrt2$ 本身不属于 $\mathbb Q$。

稠密性回答两个不同的点之间能否再插入一个点。完备性回答一个非空有界集合所确定的边界是否仍在当前数系中。集合

$$
A
=
\lbrace q\in\mathbb Q:q\geq0,\ q^2<2\rbrace
$$

在 $\mathbb Q$ 中没有上确界；集合

$$
B
=
\lbrace x\in\mathbb R:x\geq0,\ x^2<2\rbrace
$$

在 $\mathbb R$ 中有上确界 $\sqrt2$。两个集合由同一个不等式定义，边界能否留在数系内部取决于数系是否完备。

## 6. 数列与极限

一个实数数列是映射

$$
x:\mathbb N\to\mathbb R.
$$

第 $n$ 个值写作 $x&#95;n:=x(n)$，整个数列写作 $(x&#95;n)&#95;{n\geq1}$ 或简写为 $(x&#95;n)$。

### 6.1 极限的定义

称数列 $(x&#95;n)$ 收敛到 $L\in\mathbb R$，如果

$$
(\forall\varepsilon>0)(\exists N\in\mathbb N)
(\forall n\geq N)
\quad
|x&#95;n-L|<\varepsilon.
$$

记作

$$
x&#95;n\longrightarrow L
$$

或

$$
\lim&#95;{n\to\infty}x&#95;n=L.
$$

量词的顺序不能交换。先给定误差 $\varepsilon$，再允许起点 $N$ 依赖于这个误差；一旦 $n\geq N$，所有后续项都必须落在区间 $(L-\varepsilon,L+\varepsilon)$ 中。

**命题 6.1**　数列的极限唯一。

**证明。** 假设 $x&#95;n\to L$ 且 $x&#95;n\to M$。若 $L\neq M$，取

$$
\varepsilon:=\frac{|L-M|}{3}>0.
$$

存在 $N&#95;1,N&#95;2\in\mathbb N$，使得当 $n\geq N&#95;1$ 时 $|x&#95;n-L|<\varepsilon$，当 $n\geq N&#95;2$ 时 $|x&#95;n-M|<\varepsilon$。取 $n\geq\max\lbrace N&#95;1,N&#95;2\rbrace$，由三角不等式，

$$
\begin{aligned}
|L-M|
&\leq|L-x&#95;n|+|x&#95;n-M|\\\\
&<2\varepsilon\\\\
&=\frac23|L-M|,
\end{aligned}
$$

矛盾。因此 $L=M$。证毕。

**命题 6.2**　每个收敛数列都有界。

**证明。** 设 $x&#95;n\to L$。取 $\varepsilon=1$，存在 $N$ 使 $n\geq N$ 时

$$
|x&#95;n-L|<1.
$$

所以 $|x&#95;n|<|L|+1$。前 $N-1$ 项只有有限多个，令

$$
M
:=
\max\lbrace
|x&#95;1|,\ldots,|x&#95;N|,|L|+1
\rbrace.
$$

则对所有 $n\in\mathbb N$ 都有 $|x&#95;n|\leq M$。证毕。

### 6.2 单调有界数列

若

$$
x&#95;n\leq x&#95;{n+1}
$$

对每个 $n$ 成立，称 $(x&#95;n)$ 单调递增。这里允许相邻两项相等。

**定理 6.3（单调收敛定理）**　每个有上界的单调递增实数数列都收敛。

**证明。** 令

$$
E:=\lbrace x&#95;n:n\in\mathbb N\rbrace.
$$

$E$ 非空且有上界，所以完备性给出

$$
L:=\sup E.
$$

任取 $\varepsilon>0$。根据命题 2.2，存在 $N\in\mathbb N$ 使

$$
L-\varepsilon<x&#95;N\leq L.
$$

当 $n\geq N$ 时，单调性给出

$$
L-\varepsilon
{}<x&#95;N
\leq x&#95;n
\leq L.
$$

因此 $|x&#95;n-L|<\varepsilon$，即 $x&#95;n\to L$。证毕。

单调递减且有下界的数列同样收敛，其极限是所有项的下确界。附录 D 给出对偶证明。

## 7. 区间套与收敛子列

闭区间记作

$$
[a,b]
=
\lbrace x\in\mathbb R:a\leq x\leq b\rbrace.
$$

设

$$
I&#95;n=[a&#95;n,b&#95;n].
$$

若 $I&#95;{n+1}\subseteq I&#95;n$，称这些区间形成一个闭区间套。

**定理 7.1（闭区间套定理）**　若 $(I&#95;n)$ 是非空闭区间套，且

$$
b&#95;n-a&#95;n\longrightarrow0,
$$

则存在唯一 $L\in\mathbb R$ 满足

$$
L\in\bigcap&#95;{n=1}^{\infty}I&#95;n.
$$

符号 $\bigcap&#95;{n=1}^{\infty}I&#95;n$ 表示同时属于所有区间 $I&#95;n$ 的实数集合。

证明从左端点集合

$$
\lbrace a&#95;n:n\in\mathbb N\rbrace
$$

的上确界出发。嵌套关系保证每个右端点都是所有左端点的上界；区间长度趋于零保证公共点唯一。完整证明见附录 E.1。

### 7.1 子列

从数列 $(x&#95;n)$ 中按原顺序选出无穷多项，得到子列

$$
(x&#95;{n&#95;k})&#95;{k\geq1},
$$

其中

$$
n&#95;1<n&#95;2<n&#95;3<\cdots.
$$

**定理 7.2（Bolzano–Weierstrass）**　每个有界实数数列都有收敛子列。

证明采用区间二分。先把所有项放进一个有界闭区间；把区间分成左右两半，其中至少一半包含数列的无穷多项。重复选择这样的半区间，得到长度趋于零的闭区间套。区间套定理给出唯一公共点 $L$，再从第 $k$ 个区间中选取下标严格增加的一项。所选子列最终落入 $L$ 的任意小邻域。附录 E.2 将每一步展开。

## 8. Cauchy 列与完备性

极限定义需要预先知道候选值 $L$。Cauchy 条件只比较数列后面的项。

称 $(x&#95;n)$ 是 **Cauchy 列**，如果

$$
(\forall\varepsilon>0)(\exists N\in\mathbb N)
(\forall m,n\geq N)
\quad
|x&#95;m-x&#95;n|<\varepsilon.
$$

它表示数列的尾部可以被压进任意短的区间。

**命题 8.1**　每个收敛数列都是 Cauchy 列。

**证明。** 设 $x&#95;n\to L$。给定 $\varepsilon>0$，选择 $N$ 使 $n\geq N$ 时

$$
|x&#95;n-L|<\frac\varepsilon2.
$$

若 $m,n\geq N$，则

$$
\begin{aligned}
|x&#95;m-x&#95;n|
&\leq|x&#95;m-L|+|x&#95;n-L|\\\\
&<\varepsilon.
\end{aligned}
$$

所以 $(x&#95;n)$ 是 Cauchy 列。证毕。

反方向需要实数完备性。

**定理 8.2（Cauchy 完备性）**　每个实数 Cauchy 列都收敛于某个实数。

证明分三步。首先，Cauchy 列必有界；其次，Bolzano–Weierstrass 定理从中取出收敛子列；最后，Cauchy 条件迫使原数列与这条子列收敛到同一个极限。完整证明见附录 F。

### 8.1 为什么同一句话在 $\mathbb Q$ 中失败

由有理数稠密性，对每个 $n\in\mathbb N$，可以选择 $q&#95;n\in\mathbb Q$ 使

$$
0<\sqrt2-q&#95;n<10^{-n}.
$$

对任意 $m,n\geq N$，

$$
\begin{aligned}
|q&#95;m-q&#95;n|
&\leq|q&#95;m-\sqrt2|+|q&#95;n-\sqrt2|\\\\
&<2\cdot10^{-N}.
\end{aligned}
$$

给定 $\varepsilon>0$，由 Archimedean 性质选择 $N>2/\varepsilon$。不等式 $10^N\geq N$ 可以由归纳法得到：$N=1$ 时成立；若 $10^N\geq N$，则

$$
10^{N+1}
\geq10N
\geq N+1.
$$

因此

$$
2\cdot10^{-N}
\leq\frac2N
{}<\varepsilon.
$$

于是 $(q&#95;n)$ 是有理数中的 Cauchy 列。

同一个估计还能直接验证它在实数中收敛到 $\sqrt2$。给定 $\varepsilon>0$，选择 $N>1/\varepsilon$。当 $n\geq N$ 时，

$$
|q&#95;n-\sqrt2|
{}<10^{-n}
\leq\frac1n
\leq\frac1N
{}<\varepsilon.
$$

若它还收敛到某个有理数 $q$，极限唯一性将给出 $q=\sqrt2$，与 $\sqrt2\notin\mathbb Q$ 矛盾。因此 $\mathbb Q$ 不是完备的。

## 9. 实数怎样容纳这些边界

上确界公理把 $\mathbb R$ 作为已经存在的数系，并规定它没有上述缺口。1872 年，Dedekind 从有理数的次序出发构造实数，Cantor 则从有理 Cauchy 列出发；两条道路分别对应前面反复出现的集合语言与数列语言。

### 9.1 Dedekind 分割：把边界写成集合

一个 **Dedekind 分割**是满足下列三条性质的有理数子集 $D\subseteq\mathbb Q$：

1. $D\neq\varnothing$ 且 $D\neq\mathbb Q$；
2. 若 $r\in D$ 且 $q<r$，则 $q\in D$；
3. 对每个 $q\in D$，都存在 $r\in D$ 使 $q<r$。

第二条表示 $D$ 包含其中每个元素左侧的全部有理数，第三条表示 $D$ 没有最大元。这样的 $D$ 记录某条分界线左侧的所有有理数。

每个有理数 $a$ 都给出分割

$$
D&#95;a
:=
\lbrace q\in\mathbb Q:q<a\rbrace.
$$

分割 $D&#95;a$ 的边界就是 $a$。方程 $x^2=2$ 所确定的分割为

$$
D&#95;2
:=
\lbrace q\in\mathbb Q:q<0\rbrace
\cup
\lbrace q\in\mathbb Q:q\geq0,\ q^2<2\rbrace.
$$

符号 $\cup$ 表示两个集合的并。下标 $2$ 指向等式 $x^2=2$。分割 $D&#95;2$ 不等于任何 $D&#95;a$；否则它的边界将是某个平方等于 $2$ 的有理数。附录 G.1 会逐条验证 $D&#95;2$ 确实满足分割定义，并证明它不来自任何有理数。

用包含关系给分割排序：

$$
D\leq E
\quad\Longleftrightarrow\quad
D\subseteq E.
$$

符号 $\Longleftrightarrow$ 表示左右两个陈述互相推出。记所有 Dedekind 分割组成的集合为 $\mathcal D$。设非空分割族 $\mathcal F\subseteq\mathcal D$ 有上界，也就是存在 $H\in\mathcal D$ 使每个 $D\in\mathcal F$ 都满足 $D\subseteq H$。定义

$$
U
:=
\bigcup&#95;{D\in\mathcal F}D.
$$

符号 $\bigcup&#95;{D\in\mathcal F}D$ 表示把分割族中所有集合的元素合在一起。附录 G.2 将证明 $U$ 仍是 Dedekind 分割，并且

$$
U=\sup\mathcal F.
$$

在这个构造中，上确界有一个具体的集合表达式：它就是所有较小分割的并。每个分割保存一条边界左侧的信息，取并把整族边界共同确定的最小上界完整地收集起来。

### 9.2 Cauchy 列：把逼近过程变成一个点

Dedekind 分割从次序出发。另一种构造从有理数列之间的距离出发。

记正有理数集为

$$
\mathbb Q&#95;{>0}
:=
\lbrace q\in\mathbb Q:q>0\rbrace.
$$

记所有有理 Cauchy 列组成的集合为

$$
\mathcal C&#95;{\mathbb Q}
:=
\left\lbrace
(a&#95;n)&#95;{n\geq1}:
(\forall n\in\mathbb N)\ a&#95;n\in\mathbb Q,\
(\forall\varepsilon\in\mathbb Q&#95;{>0})
(\exists N\in\mathbb N)
(\forall m,n\geq N)\
|a&#95;m-a&#95;n|<\varepsilon
\right\rbrace.
$$

在 $\mathcal C&#95;{\mathbb Q}$ 上定义关系

$$
(a&#95;n)\sim(b&#95;n)
\quad\Longleftrightarrow\quad
a&#95;n-b&#95;n\longrightarrow0.
$$

这里 $a&#95;n-b&#95;n\to0$ 完全在 $\mathbb Q$ 内定义，含义是

$$
(\forall\varepsilon\in\mathbb Q&#95;{>0})
(\exists N\in\mathbb N)
(\forall n\geq N)
\quad
|a&#95;n-b&#95;n|<\varepsilon.
$$

符号 $\sim$ 表示两个有理 Cauchy 列被视为等价。它们可以逐项不同，只要二者的差趋于 $0$，就代表同一个极限位置。数列 $(a&#95;n)$ 的等价类记作

$$
[(a&#95;n)]
:=
\lbrace(b&#95;n)\in\mathcal C&#95;{\mathbb Q}:
(b&#95;n)\sim(a&#95;n)\rbrace.
$$

把所有等价类组成的商集记为

$$
\widehat{\mathbb Q}
:=
\mathcal C&#95;{\mathbb Q}/\mathord{\sim}.
$$

有理数 $q$ 通过常数列嵌入这个商集：

$$
\iota(q)
:=
[(q,q,q,\ldots)].
$$

希腊字母 $\iota$ 在这里表示把有理数送入商集的映射。

现在只用有理数二分出一个新元素。令

$$
\ell&#95;1:=1,
\qquad
u&#95;1:=2.
$$

字母 $\ell$ 表示区间的左端点，$u$ 表示右端点。已经得到有理数 $\ell&#95;n,u&#95;n$ 后，取中点

$$
m&#95;n:=\frac{\ell&#95;n+u&#95;n}{2}.
$$

若 $m&#95;n^2<2$，令 $\ell&#95;{n+1}:=m&#95;n$、$u&#95;{n+1}:=u&#95;n$；若 $m&#95;n^2>2$，令 $\ell&#95;{n+1}:=\ell&#95;n$、$u&#95;{n+1}:=m&#95;n$。附录 C.1 保证有理数 $m&#95;n$ 的平方不会等于 $2$，所以每一步恰好落入上述一种情形。

归纳法给出

$$
\ell&#95;n^2<2<u&#95;n^2,
\qquad
u&#95;n-\ell&#95;n
=
\frac1{2^{n-1}}.
$$

区间逐次嵌套。若 $k\geq n$，则

$$
\ell&#95;n
\leq\ell&#95;k
\leq u&#95;n,
$$

从而

$$
|\ell&#95;k-\ell&#95;n|
\leq u&#95;n-\ell&#95;n
=\frac1{2^{n-1}}
\longrightarrow0.
$$

给定 $\varepsilon\in\mathbb Q&#95;{>0}$，把它写成正整数之比 $\varepsilon=p/q$。选择 $N>q/p$。由 $2^{N-1}\geq N$ 可知

$$
\frac1{2^{N-1}}
\leq\frac1N
{}<\frac pq
=\varepsilon.
$$

任取 $j,k\geq N$，交换二者后可以设 $k\geq j$，于是

$$
|\ell&#95;k-\ell&#95;j|
\leq\frac1{2^{j-1}}
\leq\frac1{2^{N-1}}
{}<\varepsilon.
$$

所以 $(\ell&#95;n)$ 是有理 Cauchy 列，可以定义

$$
\xi
:=
[(\ell&#95;n)]
\in\widehat{\mathbb Q}.
$$

希腊字母 $\xi$ 在这里表示二分过程产生的等价类。附录 H.5 将证明 $\xi$ 不等于任何 $\iota(q)$，其中 $q\in\mathbb Q$。这个等价类为原先没有有理数落点的逼近过程提供了一个新元素；在与 Dedekind 构造的对应之下，它正是由分割 $D&#95;2$ 表示的正平方根二。

不同的有理逼近列只要相差一个趋零数列，就落入同一个等价类。

加法与乘法由代表元逐项定义：

$$
[(a&#95;n)]+[(b&#95;n)]
:=
[(a&#95;n+b&#95;n)],
$$

$$
[(a&#95;n)]\,[(b&#95;n)]
:=
[(a&#95;nb&#95;n)].
$$

附录 H 将证明 $\sim$ 是等价关系，逐项相加与相乘仍得到有理 Cauchy 列，并且运算结果不依赖代表元的选择。这种不依赖代表元选择的性质称为运算是**良定义的**。[上一篇关于等价关系与商结构的文章](/2026/08/30/2026-08-30-sets-maps-equivalence-quotients/)先由等价关系形成等价类，再把等价类组成商集；这里的商集把指向同一极限位置的无穷多种逼近方法合成一个数。

### 9.3 两种构造中的完备性

Dedekind 分割把实数理解为有理数轴上的边界，上确界由集合的并给出。Cauchy 商构造把实数理解为有理逼近过程的等价类，原本缺少落点的数列本身参与定义新的点。

前一种构造直接控制有界集合的边界，后一种构造直接控制逼近过程的极限。建立相应的次序与运算后，两者产生同构的完备有序域。“同构”表示存在保持加法、乘法和次序的一一对应；从有序域内部看，两种构造得到的是同一个实数系。

## 10. 完备性在分析学中的位置

上确界公理首先填补数轴中的缺口，随后保证单调有界数列的极限存在。闭区间套定理把完备性转成区间的公共点，Bolzano–Weierstrass 定理再把它转成有界数列中的收敛子列，最终得到 Cauchy 完备性。

后续实变函数会在更一般的函数空间中重复这个问题。向量空间若带有衡量元素大小的函数 $\lVert\cdot\rVert$，这个函数称为**范数**，并由

$$
d(f,g):=\lVert f-g\rVert
$$

定义元素 $f$ 与 $g$ 之间的距离。每个 Cauchy 列都在空间内收敛的赋范向量空间称为 **Banach 空间**。对 $p\geq1$，Lebesgue 空间 $L^p$ 由 $p$ 次方可积的函数类组成；$L^p$ 的完备性保证 Cauchy 函数列仍在这个空间内部取得极限。

复数域记作 $\mathbb C$。复变函数中的幂级数与一致收敛依赖 $\mathbb C$ 的完备性，解析延拓再建立在这些局部收敛结果之上。

下一篇分析文章将研究数列与无穷级数，证明极限的四则运算、夹逼定理、单调判别、上极限与下极限，并从几何级数进入收敛判别。

## 参考文献

1. Walter Rudin, *Principles of Mathematical Analysis*, Third Edition, McGraw–Hill, 1976.
2. Stephen Abbott, *Understanding Analysis*, Second Edition, Springer, 2015.
3. Terence Tao, *Analysis I*, Third Edition, Springer, 2016.
4. Tom M. Apostol, *Mathematical Analysis*, Second Edition, Addison-Wesley, 1974.
5. Charles C. Pugh, *Real Mathematical Analysis*, Second Edition, Springer, 2015.
6. Richard Dedekind, *Stetigkeit und irrationale Zahlen*, Friedrich Vieweg und Sohn, 1872.
7. Georg Cantor, “Über die Ausdehnung eines Satzes aus der Theorie der trigonometrischen Reihen,” *Mathematische Annalen*, 5, 1872, pp. 123–132.
8. Augustin-Louis Cauchy, *Cours d’Analyse de l’École Royale Polytechnique*, Debure frères, 1821.

---

## 附录

如下为正文附录补充。

## A. 绝对值与三角不等式

### A.1 基本界

对任意实数 $x$，绝对值定义立即给出

$$
-|x|\leq x\leq|x|.
$$

同理，

$$
-|y|\leq y\leq|y|.
$$

对应两边相加，得到

$$
-(|x|+|y|)
\leq x+y
\leq |x|+|y|.
$$

若 $a\geq0$ 且 $-a\leq z\leq a$，则 $|z|\leq a$。取 $a=|x|+|y|$ 与 $z=x+y$，得到三角不等式

$$
|x+y|\leq|x|+|y|.
$$

把三角不等式中的两项分别取为 $x-y$ 与 $y-z$，还得到

$$
|x-z|\leq|x-y|+|y-z|.
$$

这就是数轴距离的三角不等式。

### A.2 反三角不等式

由

$$
x=(x-y)+y
$$

和三角不等式，

$$
|x|\leq|x-y|+|y|,
$$

所以

$$
|x|-|y|\leq|x-y|.
$$

交换 $x,y$，得到

$$
|y|-|x|\leq|x-y|.
$$

两式合并为

$$
\bigl||x|-|y|\bigr|
\leq|x-y|.
$$

## B. 下确界与上确界的对偶

设非空集合 $E\subseteq\mathbb R$ 有下界。定义

$$
-E:=\lbrace-x:x\in E\rbrace.
$$

若 $m$ 是 $E$ 的下界，则对每个 $x\in E$ 都有 $m\leq x$。乘以 $-1$ 后不等号反向：

$$
-x\leq-m.
$$

因此 $-m$ 是 $-E$ 的上界。反过来，$-E$ 的每个上界 $M$ 都对应 $E$ 的下界 $-M$。

由完备性，$-E$ 有上确界。令

$$
\alpha:=\sup(-E).
$$

因为 $\alpha$ 是 $-E$ 的上界，$-\alpha$ 是 $E$ 的下界。若 $m$ 是 $E$ 的任意下界，则 $-m$ 是 $-E$ 的上界。上确界的最小性给出

$$
\alpha\leq-m.
$$

乘以 $-1$，得到

$$
m\leq-\alpha.
$$

所以 $-\alpha$ 是最大的下界，即

$$
\inf E=-\sup(-E).
$$

## C. 有理数缺口与稠密性

### C.1 不存在平方为 $2$ 的有理数

假设存在 $s\in\mathbb Q$ 满足 $s^2=2$。由于 $|s|>0$，可以取互素正整数 $p,q$，使

$$
|s|=\frac pq.
$$

“互素”表示 $p$ 与 $q$ 没有大于 $1$ 的公因数。两边平方得到

$$
p^2=2q^2.
$$

所以 $p^2$ 是偶数。若 $p$ 是奇数，可以写成 $p=2k+1$，于是

$$
p^2=4k^2+4k+1
$$

仍是奇数。因此 $p$ 必为偶数，写成 $p=2r$。代回后

$$
4r^2=2q^2,
$$

从而

$$
q^2=2r^2.
$$

同理 $q$ 也是偶数。于是 $p,q$ 都被 $2$ 整除，与二者互素矛盾。因此不存在 $s\in\mathbb Q$ 满足 $s^2=2$。

正文第 4.1 节构造了满足 $\alpha\geq0$ 与 $\alpha^2=2$ 的实数，并把它记为 $\sqrt2$。由刚才的结论，

$$
\sqrt2\notin\mathbb Q.
$$

### C.2 整数部分引理

对每个 $x\in\mathbb R$，存在 $m\in\mathbb Z$ 使

$$
m-1\leq x<m.
$$

先由 Archimedean 性质选择 $N\in\mathbb N$ 使 $N>|x|+1$。于是 $-N<x<N$。集合

$$
S:=\lbrace k\in\mathbb Z:k>x\rbrace
$$

非空，因为 $N\in S$；它在整数中被 $-N$ 从下方控制。把所有候选整数平移 $N$ 后，可以使用自然数的良序性：每个非空自然数子集都有最小元。因此 $S$ 有最小元，记为 $m$。

由定义 $x<m$。若 $m-1>x$，则 $m-1\in S$，与 $m$ 的最小性矛盾。所以 $m-1\leq x$，引理成立。

### C.3 有理数稠密性的完整证明

给定 $x<y$。因为 $y-x>0$，由 Archimedean 性质，可以选择 $n\in\mathbb N$ 使

$$
n>\frac1{y-x}.
$$

因此

$$
n(y-x)>1.
$$

对实数 $nx$ 使用整数部分引理，存在 $m\in\mathbb Z$ 使

$$
m-1\leq nx<m.
$$

令

$$
q:=\frac mn\in\mathbb Q.
$$

由 $nx<m$ 得 $x<q$。另一方面，

$$
m\leq nx+1<ny,
$$

所以 $q<y$。最终得到

$$
x<q<y.
$$

## D. 单调递减数列的收敛

设 $(x&#95;n)$ 单调递减且有下界。令

$$
E:=\lbrace x&#95;n:n\in\mathbb N\rbrace,
\qquad
L:=\inf E.
$$

任取 $\varepsilon>0$。数 $L+\varepsilon$ 严格大于最大下界，所以它不是 $E$ 的下界。于是存在 $N\in\mathbb N$ 使

$$
x&#95;N<L+\varepsilon.
$$

又因为 $L$ 是下界，$L\leq x&#95;N$。当 $n\geq N$ 时，单调递减性给出

$$
L
\leq x&#95;n
\leq x&#95;N
{}<L+\varepsilon.
$$

因此 $|x&#95;n-L|<\varepsilon$，即

$$
x&#95;n\longrightarrow L.
$$

## E. 区间套与 Bolzano–Weierstrass 定理

### E.1 闭区间套定理的完整证明

设

$$
I&#95;n=[a&#95;n,b&#95;n]
$$

非空，并满足 $I&#95;{n+1}\subseteq I&#95;n$。嵌套关系给出

$$
a&#95;n\leq a&#95;{n+1}
\leq b&#95;{n+1}
\leq b&#95;n.
$$

考虑左端点集合

$$
E:=\lbrace a&#95;n:n\in\mathbb N\rbrace.
$$

它非空。固定任意 $m\in\mathbb N$。若 $n\geq m$，因为 $I&#95;n\subseteq I&#95;m$，有 $a&#95;n\leq b&#95;m$；若 $n<m$，则 $a&#95;n\leq a&#95;m\leq b&#95;m$。所以 $b&#95;m$ 是整个 $E$ 的上界。

由完备性，令

$$
L:=\sup E.
$$

因为 $a&#95;m\in E$，有 $a&#95;m\leq L$；因为 $b&#95;m$ 是 $E$ 的上界，有 $L\leq b&#95;m$。因此

$$
L\in[a&#95;m,b&#95;m]=I&#95;m.
$$

$m$ 任意，所以

$$
L\in\bigcap&#95;{m=1}^{\infty}I&#95;m.
$$

最后证明唯一性。若 $L'$ 也属于每个 $I&#95;n$，则

$$
|L-L'|\leq b&#95;n-a&#95;n
$$

对每个 $n$ 成立。右侧趋于 $0$。若 $|L-L'|>0$，取

$$
\varepsilon:=\frac{|L-L'|}{2},
$$

当 $n$ 足够大时会有 $b&#95;n-a&#95;n<\varepsilon<|L-L'|$，矛盾。因此 $L=L'$。

### E.2 Bolzano–Weierstrass 定理的完整证明

设 $(x&#95;n)$ 有界。存在 $u&#95;1<v&#95;1$，使所有项都落在

$$
I&#95;1=[u&#95;1,v&#95;1]
$$

中。把 $I&#95;1$ 从中点分成两个闭区间。至少有一半包含数列的无穷多项；否则两半都只含有限多项，它们的并也只能包含有限多项。选取含无穷多项的那一半，记为 $I&#95;2$。

重复这一过程。得到闭区间套

$$
I&#95;1\supseteq I&#95;2\supseteq I&#95;3\supseteq\cdots,
$$

其中每个 $I&#95;k$ 都包含原数列的无穷多项，并且

$$
\operatorname{length}(I&#95;k)
=
\frac{v&#95;1-u&#95;1}{2^{k-1}}.
$$

符号 $\operatorname{length}([a,b])=b-a$ 表示区间长度。下面先说明分母确实趋于无穷。对每个 $k\in\mathbb N$，归纳法给出

$$
2^{k-1}\geq k.
$$

当 $k=1$ 时等式成立；若 $2^{k-1}\geq k$，则

$$
2^k\geq2k\geq k+1.
$$

因此

$$
0
{}<\operatorname{length}(I&#95;k)
=\frac{v&#95;1-u&#95;1}{2^{k-1}}
\leq\frac{v&#95;1-u&#95;1}{k}
\longrightarrow0.
$$

闭区间套定理给出唯一

$$
L\in\bigcap&#95;{k=1}^{\infty}I&#95;k.
$$

现在递归选择下标。先从 $I&#95;1$ 中选一项 $x&#95;{n&#95;1}$。已经选好 $n&#95;k$ 后，因为 $I&#95;{k+1}$ 包含无穷多项，可以选择 $n&#95;{k+1}>n&#95;k$，使

$$
x&#95;{n&#95;{k+1}}\in I&#95;{k+1}.
$$

于是得到子列 $(x&#95;{n&#95;k})$。因为 $x&#95;{n&#95;k}$ 与 $L$ 都属于 $I&#95;k$，

$$
|x&#95;{n&#95;k}-L|
\leq
\operatorname{length}(I&#95;k)
\longrightarrow0.
$$

所以

$$
x&#95;{n&#95;k}\longrightarrow L.
$$

## F. Cauchy 完备性的完整证明

### F.1 Cauchy 列有界

设 $(x&#95;n)$ 是 Cauchy 列。在定义中取 $\varepsilon=1$，存在 $N\in\mathbb N$，使得当 $m,n\geq N$ 时

$$
|x&#95;m-x&#95;n|<1.
$$

固定 $n=N$。对每个 $m\geq N$，

$$
|x&#95;m|
\leq|x&#95;m-x&#95;N|+|x&#95;N|
{}<1+|x&#95;N|.
$$

再令

$$
M
:=
1+\max\lbrace
|x&#95;1|,\ldots,|x&#95;N|
\rbrace.
$$

则所有 $m\in\mathbb N$ 都满足 $|x&#95;m|\leq M$，所以数列有界。

### F.2 从收敛子列回到原数列

由 Bolzano–Weierstrass 定理，有界数列 $(x&#95;n)$ 存在收敛子列

$$
x&#95;{n&#95;k}\longrightarrow L.
$$

任取 $\varepsilon>0$。Cauchy 条件给出 $N&#95;1\in\mathbb N$，使得当 $m,n\geq N&#95;1$ 时

$$
|x&#95;m-x&#95;n|<\frac\varepsilon2.
$$

子列收敛给出 $K\in\mathbb N$，使得当 $k\geq K$ 时

$$
|x&#95;{n&#95;k}-L|<\frac\varepsilon2.
$$

因为下标 $n&#95;k$ 严格增加，可以再增大 $K$，使 $n&#95;K\geq N&#95;1$。固定这样的 $K$。对任意 $n\geq N&#95;1$，

$$
\begin{aligned}
|x&#95;n-L|
&\leq|x&#95;n-x&#95;{n&#95;K}|
+|x&#95;{n&#95;K}-L|\\\\
&<\frac\varepsilon2+\frac\varepsilon2\\\\
&=\varepsilon.
\end{aligned}
$$

因此 $x&#95;n\to L$。每个实数 Cauchy 列都在 $\mathbb R$ 中收敛。

## G. Dedekind 分割与上确界

### G.1 平方根二所确定的分割

考虑

$$
D&#95;2
:=
\lbrace q\in\mathbb Q:q<0\rbrace
\cup
\lbrace q\in\mathbb Q:q\geq0,\ q^2<2\rbrace.
$$

先验证它是 Dedekind 分割。

数 $-1$ 属于 $D&#95;2$，所以 $D&#95;2$ 非空。数 $2$ 不属于 $D&#95;2$，所以 $D&#95;2\neq\mathbb Q$。

再验证向下封闭。设 $r\in D&#95;2$ 且 $q<r$。若 $q<0$，则定义直接给出 $q\in D&#95;2$。若 $q\geq0$，则 $r>q\geq0$。因为 $r\in D&#95;2$，有 $r^2<2$，从而

$$
q^2<r^2<2.
$$

所以 $q\in D&#95;2$。

最后证明 $D&#95;2$ 没有最大元。任取 $q\in D&#95;2$。若 $q<0$，令

$$
r:=\frac q2.
$$

因为 $q<q/2<0$，所以 $r\in D&#95;2$ 且 $q<r$。

若 $q\geq0$，则 $q^2<2$。取有理数

$$
\delta
:=
\frac{2-q^2}{2q+2}>0.
$$

由 $q\geq0$ 可知

$$
0<\delta\leq1.
$$

因此

$$
\begin{aligned}
(q+\delta)^2-q^2
&=2q\delta+\delta^2\\\\
&\leq(2q+1)\delta\\\\
&<(2q+2)\delta\\\\
&=2-q^2.
\end{aligned}
$$

于是 $(q+\delta)^2<2$，所以 $q+\delta\in D&#95;2$，并且 $q<q+\delta$。三条条件全部成立，$D&#95;2$ 是 Dedekind 分割。

下面证明 $D&#95;2$ 不等于任何有理数产生的分割

$$
D&#95;a
=
\lbrace q\in\mathbb Q:q<a\rbrace.
$$

假设存在 $a\in\mathbb Q$ 使 $D&#95;2=D&#95;a$。由于 $0\in D&#95;2$，必有 $a>0$。附录 C.1 已经证明 $a^2\neq2$。

若 $a^2<2$，取有理数

$$
\delta
:=
\frac{2-a^2}{2a+2}>0.
$$

与前面的计算相同，$(a+\delta)^2<2$。因此 $a+\delta\in D&#95;2$。但 $a+\delta>a$，所以 $a+\delta\notin D&#95;a$，与两个分割相等矛盾。

若 $a^2>2$，取有理数

$$
\delta
:=
\frac{a^2-2}{2a}>0.
$$

令 $c:=a-\delta$。由

$$
\begin{aligned}
c
&=\frac{a^2+2}{2a}\\\\
&>0
\end{aligned}
$$

以及

$$
\begin{aligned}
c^2
&=(a-\delta)^2\\\\
&=2+\delta^2\\\\
&>2
\end{aligned}
$$

可知 $c\notin D&#95;2$。再取有理数

$$
r:=\frac{a+c}{2}.
$$

此时 $c<r<a$，所以 $r\in D&#95;a$；又因为 $r>c>0$，有 $r^2>c^2>2$，所以 $r\notin D&#95;2$。这同样与 $D&#95;2=D&#95;a$ 矛盾。

两种情形都不可能，因此 $D&#95;2$ 的边界不由任何有理数给出。

### G.2 分割族的上确界

设 $\varnothing\neq\mathcal F\subseteq\mathcal D$，并且 $\mathcal F$ 有上界 $H\in\mathcal D$。这表示

$$
(\forall D\in\mathcal F)\quad D\subseteq H.
$$

定义

$$
U
:=
\bigcup&#95;{D\in\mathcal F}D.
$$

先证明 $U$ 是 Dedekind 分割。

因为 $\mathcal F$ 非空，可以选择 $D&#95;0\in\mathcal F$。分割 $D&#95;0$ 非空，且 $D&#95;0\subseteq U$，所以 $U$ 非空。另一方面，每个 $D\in\mathcal F$ 都包含于 $H$，因此

$$
U\subseteq H.
$$

分割 $H$ 不等于 $\mathbb Q$，所以 $U\neq\mathbb Q$。

设 $r\in U$ 且 $q<r$。按照集合并的定义，存在 $D\in\mathcal F$ 使 $r\in D$。分割 $D$ 向下封闭，所以 $q\in D$，进而 $q\in U$。因此 $U$ 向下封闭。

再取任意 $q\in U$。存在 $D\in\mathcal F$ 使 $q\in D$。因为 $D$ 没有最大元，可以选择 $r\in D$ 使 $q<r$。由 $D\subseteq U$ 可知 $r\in U$，所以 $U$ 也没有最大元。至此，$U\in\mathcal D$。

对每个 $D\in\mathcal F$，集合并的定义给出 $D\subseteq U$，所以 $U$ 是 $\mathcal F$ 的上界。

设 $K\in\mathcal D$ 是 $\mathcal F$ 的任意上界。于是每个 $D\in\mathcal F$ 都满足 $D\subseteq K$。若 $q\in U$，则存在 $D\in\mathcal F$ 使 $q\in D$，从而 $q\in K$。因此

$$
U\subseteq K.
$$

这说明 $U$ 小于或等于每个上界。故 $U$ 是最小上界，即

$$
\sup\mathcal F
=
\bigcup&#95;{D\in\mathcal F}D.
$$

## H. 有理 Cauchy 列的商构造

### H.1 等价关系

在 $\mathcal C&#95;{\mathbb Q}$ 上定义

$$
(a&#95;n)\sim(b&#95;n)
\quad\Longleftrightarrow\quad
a&#95;n-b&#95;n\longrightarrow0.
$$

先验证自反性。对每个 $(a&#95;n)\in\mathcal C&#95;{\mathbb Q}$，

$$
a&#95;n-a&#95;n=0\longrightarrow0,
$$

所以 $(a&#95;n)\sim(a&#95;n)$。

再验证对称性。若 $(a&#95;n)\sim(b&#95;n)$，则 $a&#95;n-b&#95;n\to0$。因为

$$
|b&#95;n-a&#95;n|
=
|a&#95;n-b&#95;n|,
$$

也有 $b&#95;n-a&#95;n\to0$，所以 $(b&#95;n)\sim(a&#95;n)$。

最后验证传递性。若 $(a&#95;n)\sim(b&#95;n)$ 且 $(b&#95;n)\sim(c&#95;n)$，则

$$
a&#95;n-b&#95;n\longrightarrow0,
\qquad
b&#95;n-c&#95;n\longrightarrow0.
$$

任取 $\varepsilon\in\mathbb Q&#95;{>0}$。存在 $N&#95;1,N&#95;2\in\mathbb N$，使得当 $n\geq N&#95;1$ 时

$$
|a&#95;n-b&#95;n|<\frac\varepsilon2,
$$

当 $n\geq N&#95;2$ 时

$$
|b&#95;n-c&#95;n|<\frac\varepsilon2.
$$

若 $n\geq\max\lbrace N&#95;1,N&#95;2\rbrace$，三角不等式给出

$$
\begin{aligned}
|a&#95;n-c&#95;n|
&\leq|a&#95;n-b&#95;n|
+|b&#95;n-c&#95;n|\\\\
&<\varepsilon.
\end{aligned}
$$

因此 $a&#95;n-c&#95;n\to0$，也就是 $(a&#95;n)\sim(c&#95;n)$。关系 $\sim$ 具有自反性、对称性和传递性，所以它是等价关系。

### H.2 逐项运算保持 Cauchy 性

设 $(a&#95;n),(b&#95;n)\in\mathcal C&#95;{\mathbb Q}$。

先考虑逐项和。任取 $\varepsilon\in\mathbb Q&#95;{>0}$。存在 $N&#95;1,N&#95;2\in\mathbb N$，使得当 $m,n\geq N&#95;1$ 时

$$
|a&#95;m-a&#95;n|<\frac\varepsilon2,
$$

当 $m,n\geq N&#95;2$ 时

$$
|b&#95;m-b&#95;n|<\frac\varepsilon2.
$$

若 $m,n\geq\max\lbrace N&#95;1,N&#95;2\rbrace$，则

$$
\begin{aligned}
|(a&#95;m+b&#95;m)-(a&#95;n+b&#95;n)|
&\leq|a&#95;m-a&#95;n|
+|b&#95;m-b&#95;n|\\\\
&<\varepsilon.
\end{aligned}
$$

因为有理数的和仍是有理数，所以 $(a&#95;n+b&#95;n)$ 是有理 Cauchy 列。

再考虑逐项积。Cauchy 列有界，因此存在有理数 $M\geq1$ 使所有 $n\in\mathbb N$ 都满足

$$
|a&#95;n|\leq M,
\qquad
|b&#95;n|\leq M.
$$

给定 $\varepsilon\in\mathbb Q&#95;{>0}$，选择 $N$，使 $m,n\geq N$ 时

$$
|a&#95;m-a&#95;n|<\frac{\varepsilon}{2M},
\qquad
|b&#95;m-b&#95;n|<\frac{\varepsilon}{2M}.
$$

利用分解

$$
a&#95;mb&#95;m-a&#95;nb&#95;n
=
a&#95;m(b&#95;m-b&#95;n)
+b&#95;n(a&#95;m-a&#95;n),
$$

得到

$$
\begin{aligned}
|a&#95;mb&#95;m-a&#95;nb&#95;n|
&\leq
|a&#95;m|\,|b&#95;m-b&#95;n|
+|b&#95;n|\,|a&#95;m-a&#95;n|\\\\
&\leq
M\frac{\varepsilon}{2M}
+M\frac{\varepsilon}{2M}\\\\
&=\varepsilon.
\end{aligned}
$$

因为有理数的积仍是有理数，所以 $(a&#95;nb&#95;n)$ 也是有理 Cauchy 列。

### H.3 运算不依赖代表元

设

$$
(a&#95;n)\sim(a'&#95;n),
\qquad
(b&#95;n)\sim(b'&#95;n).
$$

先处理加法。任取 $\varepsilon\in\mathbb Q&#95;{>0}$。由 $a&#95;n-a'&#95;n\to0$ 与 $b&#95;n-b'&#95;n\to0$，存在 $N&#95;1,N&#95;2\in\mathbb N$，使 $n\geq N&#95;1$ 时

$$
|a&#95;n-a'&#95;n|<\frac\varepsilon2,
$$

$n\geq N&#95;2$ 时

$$
|b&#95;n-b'&#95;n|<\frac\varepsilon2.
$$

因此，当 $n\geq\max\lbrace N&#95;1,N&#95;2\rbrace$ 时，

$$
\begin{aligned}
|(a&#95;n+b&#95;n)-(a'&#95;n+b'&#95;n)|
&\leq|a&#95;n-a'&#95;n|
+|b&#95;n-b'&#95;n|\\\\
&<\varepsilon.
\end{aligned}
$$

所以

$$
(a&#95;n+b&#95;n)\sim(a'&#95;n+b'&#95;n).
$$

下面把乘法逐步分解：

$$
\begin{aligned}
a&#95;nb&#95;n-a'&#95;nb'&#95;n
&=a&#95;nb&#95;n-a&#95;nb'&#95;n
+a&#95;nb'&#95;n-a'&#95;nb'&#95;n\\\\
&=a&#95;n(b&#95;n-b'&#95;n)
+b'&#95;n(a&#95;n-a'&#95;n).
\end{aligned}
$$

数列 $(a&#95;n)$ 与 $(b'&#95;n)$ 有界。取有理数 $M\geq1$ 使

$$
|a&#95;n|\leq M,
\qquad
|b'&#95;n|\leq M
$$

对所有 $n$ 成立。任取 $\varepsilon\in\mathbb Q&#95;{>0}$。因为 $b&#95;n-b'&#95;n\to0$ 且 $a&#95;n-a'&#95;n\to0$，存在 $N$ 使 $n\geq N$ 时

$$
|b&#95;n-b'&#95;n|<\frac{\varepsilon}{2M},
\qquad
|a&#95;n-a'&#95;n|<\frac{\varepsilon}{2M}.
$$

于是

$$
\begin{aligned}
|a&#95;nb&#95;n-a'&#95;nb'&#95;n|
&\leq
|a&#95;n|\,|b&#95;n-b'&#95;n|
+|b'&#95;n|\,|a&#95;n-a'&#95;n|\\\\
&<
M\frac{\varepsilon}{2M}
+M\frac{\varepsilon}{2M}\\\\
&=\varepsilon.
\end{aligned}
$$

因此

$$
(a&#95;nb&#95;n)\sim(a'&#95;nb'&#95;n).
$$

逐项和与逐项积的等价类只由输入的等价类决定，所以商集上的加法与乘法是良定义的。

### H.4 有理数的嵌入

定义

$$
\iota(q)
:=
[(q,q,q,\ldots)].
$$

若 $\iota(p)=\iota(q)$，则常数列 $(p-q,p-q,\ldots)$ 收敛到 $0$。假设 $p\neq q$，取

$$
\varepsilon:=\frac{|p-q|}{2}>0.
$$

这里 $\varepsilon\in\mathbb Q&#95;{>0}$。常数列的每一项都满足

$$
|p-q|>\varepsilon,
$$

与它收敛到 $0$ 矛盾。因此 $p=q$，映射 $\iota$ 是单射。“单射”表示不同的有理数不会被送到同一个等价类。

对任意 $p,q\in\mathbb Q$，

$$
\iota(p)+\iota(q)=\iota(p+q),
$$

$$
\iota(p)\iota(q)=\iota(pq).
$$

所以 $\iota$ 保持有理数的加法与乘法。$\mathbb Q$ 可以通过这些常数列等价类被看作 $\widehat{\mathbb Q}$ 的一部分。

### H.5 二分列产生的新元素

正文从

$$
\ell&#95;1=1,
\qquad
u&#95;1=2
$$

开始，对区间进行二分。先证明递归过程始终满足

$$
\ell&#95;n^2<2<u&#95;n^2
$$

以及

$$
u&#95;n-\ell&#95;n
=
\frac1{2^{n-1}}.
$$

当 $n=1$ 时，

$$
\ell&#95;1^2=1<2<4=u&#95;1^2
$$

且 $u&#95;1-\ell&#95;1=1$，两式成立。

假设两式对 $n$ 成立，并令

$$
m&#95;n
:=
\frac{\ell&#95;n+u&#95;n}{2}.
$$

若 $m&#95;n^2<2$，递归定义给出

$$
\ell&#95;{n+1}=m&#95;n,
\qquad
u&#95;{n+1}=u&#95;n.
$$

于是

$$
\ell&#95;{n+1}^2<2<u&#95;{n+1}^2
$$

并且

$$
\begin{aligned}
u&#95;{n+1}-\ell&#95;{n+1}
&=u&#95;n-\frac{\ell&#95;n+u&#95;n}{2}\\\\
&=\frac{u&#95;n-\ell&#95;n}{2}\\\\
&=\frac1{2^n}.
\end{aligned}
$$

若 $m&#95;n^2>2$，则

$$
\ell&#95;{n+1}=\ell&#95;n,
\qquad
u&#95;{n+1}=m&#95;n.
$$

这时同样有

$$
\ell&#95;{n+1}^2<2<u&#95;{n+1}^2
$$

以及

$$
\begin{aligned}
u&#95;{n+1}-\ell&#95;{n+1}
&=\frac{\ell&#95;n+u&#95;n}{2}-\ell&#95;n\\\\
&=\frac{u&#95;n-\ell&#95;n}{2}\\\\
&=\frac1{2^n}.
\end{aligned}
$$

归纳完成。递归定义还给出

$$
[\ell&#95;{n+1},u&#95;{n+1}]
\subseteq
[\ell&#95;n,u&#95;n],
$$

所以左端点单调递增，右端点单调递减。正文已经据此证明 $(\ell&#95;n)$ 是有理 Cauchy 列。

令

$$
\xi=[(\ell&#95;n)].
$$

下面证明 $\xi$ 不等于任何有理数的常数列等价类。假设存在 $q\in\mathbb Q$ 使

$$
\xi=\iota(q).
$$

按照等价类相等的定义，

$$
\ell&#95;n-q\longrightarrow0.
$$

同时，

$$
u&#95;n-\ell&#95;n
=\frac1{2^{n-1}}
\longrightarrow0.
$$

任取 $\varepsilon\in\mathbb Q&#95;{>0}$。存在 $N&#95;1,N&#95;2\in\mathbb N$，使 $n\geq N&#95;1$ 时

$$
|u&#95;n-\ell&#95;n|<\frac\varepsilon2,
$$

$n\geq N&#95;2$ 时

$$
|\ell&#95;n-q|<\frac\varepsilon2.
$$

若 $n\geq\max\lbrace N&#95;1,N&#95;2\rbrace$，则

$$
\begin{aligned}
|u&#95;n-q|
&\leq|u&#95;n-\ell&#95;n|
+|\ell&#95;n-q|\\\\
&<\varepsilon.
\end{aligned}
$$

所以 $u&#95;n-q\to0$。

所有 $\ell&#95;n,u&#95;n$ 都属于 $[1,2]$。令

$$
M:=2+|q|>0.
$$

利用平方差分解，

$$
\begin{aligned}
|\ell&#95;n^2-q^2|
&=|\ell&#95;n-q|\,|\ell&#95;n+q|\\\\
&\leq M|\ell&#95;n-q|,
\end{aligned}
$$

$$
\begin{aligned}
|u&#95;n^2-q^2|
&=|u&#95;n-q|\,|u&#95;n+q|\\\\
&\leq M|u&#95;n-q|.
\end{aligned}
$$

给定 $\varepsilon\in\mathbb Q&#95;{>0}$。由 $\ell&#95;n-q\to0$，当 $n$ 足够大时，

$$
|\ell&#95;n-q|<\frac\varepsilon M,
$$

所以 $|\ell&#95;n^2-q^2|<\varepsilon$。同理，由 $u&#95;n-q\to0$，当 $n$ 足够大时 $|u&#95;n^2-q^2|<\varepsilon$。因此

$$
\ell&#95;n^2\longrightarrow q^2,
\qquad
u&#95;n^2\longrightarrow q^2.
$$

附录 C.1 给出 $q^2\neq2$。若 $q^2<2$，取

$$
\eta:=\frac{2-q^2}{2}
\in\mathbb Q&#95;{>0}.
$$

字母 $\eta$ 表示这个固定的正误差。因为 $u&#95;n^2\to q^2$，当 $n$ 足够大时，

$$
|u&#95;n^2-q^2|<\eta.
$$

于是

$$
u&#95;n^2
{}<q^2+\eta
=\frac{q^2+2}{2}
{}<2,
$$

与 $u&#95;n^2>2$ 矛盾。

若 $q^2>2$，取

$$
\eta:=\frac{q^2-2}{2}
\in\mathbb Q&#95;{>0}.
$$

因为 $\ell&#95;n^2\to q^2$，当 $n$ 足够大时，

$$
|\ell&#95;n^2-q^2|<\eta.
$$

于是

$$
\ell&#95;n^2
{}>q^2-\eta
=\frac{q^2+2}{2}
{}>2,
$$

与 $\ell&#95;n^2<2$ 矛盾。

两种情形都不成立，所以 $\xi\neq\iota(q)$ 对每个 $q\in\mathbb Q$ 成立。商集 $\widehat{\mathbb Q}$ 已经包含 $\mathbb Q$ 中不存在的新元素。
