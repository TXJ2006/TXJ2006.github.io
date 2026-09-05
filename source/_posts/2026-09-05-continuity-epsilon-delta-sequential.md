---
title: "连续性：从 ε-δ 定义到序列判别"
date: 2026-09-05 08:00:00
categories:
  - 基础数学
  - 分析学
tags:
  - 数学基础
  - 分析学
  - 函数连续性
  - ε-δ 定义
  - 序列判别法
  - 一致连续
  - 介值定理
mathjax: true
toc: true
toc_number: false
comments: true
---

<a href="/2026/08/31/2026-08-31-limit-arithmetic-asymptotic-estimates/">上一篇文章</a>讨论了数列的极限和极限运算。现在把同样的“靠近”放到函数上：输入靠近一个点时，输出是否也靠近相应的函数值？

给定函数 $f$ 和点 $x_0$，我们想知道的是：当输入 $x$ 足够靠近 $x_0$ 时，输出 $f(x)$ 是否也足够靠近 $f(x_0)$。这句话听起来像图像上的“没有断开”，但分析学真正使用的是一条量词精确、可以逐行检验的定义：给定输出误差 $\varepsilon$，能否找到输入误差 $\delta$，使所有满足 $|x-x_0|<\delta$ 的输入都满足 $|f(x)-f(x_0)|<\varepsilon$。

曲线“没有断开”只是帮助理解的图像。证明时要用的是上面那条关于误差的量词陈述。我先从定义做几个估计，再证明序列判别法；四则运算、一致连续和介值定理都建立在这些估计上。

<!--more-->

## 1. 连续性的定义

### 1.1 先把对象说清楚

设 $D\subseteq\mathbb R$，函数写作

$$
f:D\longrightarrow\mathbb R.
$$

这里 $D$ 是定义域；只有 $x\in D$ 时，$f(x)$ 才有意义。固定一个点 $x_0\in D$。我们只研究 $x_0$ 附近的输入，不要求函数在定义域之外有定义。

如果 $x\in D$，输入误差是

$$
|x-x_0|,
$$

输出误差是

$$
|f(x)-f(x_0)|.
$$

绝对值把实数误差变成数轴上的距离。连续性的目标就是把“小的输入误差”传递为“小的输出误差”。

### 1.2 ε-δ 定义

称 $f$ 在 $x_0$ 处**连续**，如果

$$
(\forall\varepsilon>0)(\exists\delta>0)(\forall x\in D)
\quad
|x-x_0|<\delta
\Longrightarrow
|f(x)-f(x_0)|<\varepsilon.
$$

这里的 $\varepsilon$ 是先给出的输出误差，$\delta$ 是随后选出的输入范围，$x$ 则代表这个范围内的任意点。量词的顺序不能交换：必须先知道要把输出控制到多精确，再选择一个适用的 $\delta$，最后对所有满足 $|x-x_0|<\delta$ 的点检查不等式。$\delta$ 可以依赖于 $\varepsilon$、函数 $f$ 和点 $x_0$，但不能在选定之后又依赖正在检查的那个 $x$；否则控制的只是一个点，而不是整个邻域。

### 1.3 连续与极限的关系

如果 $x_0$ 是定义域 $D$ 的聚点，函数在 $x_0$ 处连续可以写成

$$
\lim_{x\to x_0}f(x)=f(x_0).
$$

这里 $x\to x_0$ 的意思是 $x$ 在定义域内趋近 $x_0$；它不要求 $x$ 与 $x_0$ 相等。连续性还要求函数在点上的实际取值 $f(x_0)$ 与这个极限相同。

因此要区分三个问题：

1. $f(x)$ 在 $x\to x_0$ 时是否有极限；
2. 函数值 $f(x_0)$ 是否已经定义；
3. 如果二者都有，它们是否相等。

例如

$$
f(x)=
\begin{cases}
\dfrac{x^2-1}{x-1},&x\neq1,\\\\
0,&x=1
\end{cases}
$$

在 $x\neq1$ 时可以约去因子得到 $f(x)=x+1$，所以 $\lim_{x\to1}f(x)=2$，但 $f(1)=0$。极限存在，函数却在 $1$ 处不连续。若把 $f(1)$ 改成 $2$，同一个函数就会在 $1$ 处连续。

## 2. 从定义做估计

连续性证明不是猜一个神奇的 $\delta$，而是从目标不等式反向整理。先写出

$$
|f(x)-f(x_0)|<\varepsilon,
$$

再把它化成一个只涉及 $|x-x_0|$ 的条件。最后让 $\delta$ 足够小，使这个条件由 $|x-x_0|<\delta$ 自动保证。

### 2.1 线性函数

设 $f(x)=ax+b$，其中 $a,b\in\mathbb R$。固定 $x_0\in\mathbb R$，给定 $\varepsilon>0$。误差直接化为

$$
\begin{aligned}
|f(x)-f(x_0)|
&=|ax+b-(ax_0+b)|\\\\
&=|a|\cdot|x-x_0|.
\end{aligned}
$$

若 $a\neq0$，取

$$
\delta:=\frac{\varepsilon}{|a|}.
$$

当 $|x-x_0|<\delta$ 时，

$$
|f(x)-f(x_0)|
=|a|\cdot|x-x_0|
<|a|\frac{\varepsilon}{|a|}
=\varepsilon.
$$

若 $a=0$，输出误差恒为 $0$，任意 $\delta>0$ 都可以。于是每一个线性函数在每一个实数点连续。

### 2.2 绝对值函数

考虑 $f(x)=|x|$。反三角不等式给出

$$
\bigl||x|-|x_0|\bigr|\leq|x-x_0|.
$$

给定 $\varepsilon>0$，直接取 $\delta=\varepsilon$。当 $|x-x_0|<\delta$ 时，

$$
|f(x)-f(x_0)|
=\bigl||x|-|x_0|\bigr|
\leq|x-x_0|
<\varepsilon.
$$

这里的输入误差以系数 $1$ 传到输出误差，因此不需要额外缩小邻域。

### 2.3 倒数函数：先让分母远离零

考虑 $f(x)=1/x$，定义域为 $D=\mathbb R\setminus\lbrace 0\rbrace$。固定 $x_0\neq0$，直接计算

$$
\left|\frac1x-\frac1{x_0}\right|
=\frac{|x-x_0|}{|x|\cdot|x_0|}.
$$

分子已经是输入误差，但分母含有变化的 $|x|$。因此证明必须分两步。

第一步，先把 $x$ 限制在 $x_0$ 附近。要求

$$
|x-x_0|<\frac{|x_0|}{2}.
$$

由反三角不等式，

$$
|x|
\geq |x_0|-|x-x_0|
\displaystyle >\frac{|x_0|}{2}.
$$

这说明只要输入落在半径 $|x_0|/2$ 的邻域中，分母就不会靠近零。

第二步，在这个邻域内估计输出误差：

$$
\left|\frac1x-\frac1{x_0}\right|
<\frac{|x-x_0|}{(|x_0|/2)|x_0|}
=\frac{2}{|x_0|^2}|x-x_0|.
$$

给定 $\varepsilon>0$，同时要求

$$
|x-x_0|<\frac{\varepsilon|x_0|^2}{2}
$$

和

$$
|x-x_0|<\frac{|x_0|}{2}.
$$

因此取

$$
\delta:=\min\left\lbrace\frac{\varepsilon|x_0|^2}{2},\frac{|x_0|}{2}\right\rbrace
$$

即可保证 $|1/x-1/x_0|<\varepsilon$。倒数函数在自己的定义域上连续。

这个证明揭示了商的连续性为什么必须排除分母为零：不是形式上的规定，而是因为误差估计需要一个正的分母下界。

## 3. 序列判别法

### 3.1 定理

**定理 3.1（序列判别法）。** 设 $f:D\to\mathbb R$，$x_0\in D$。则 $f$ 在 $x_0$ 处连续，当且仅当对每一个取值于 $D$ 且满足

$$
x_n\longrightarrow x_0
$$

的数列 $(x_n)$，都有

$$
f(x_n)\longrightarrow f(x_0).
$$

这个定理把“任意足够靠近的输入”转换成“任意趋于 $x_0$ 的数列”。一方面，连续性给出统一的邻域控制，所以任何这样的数列都会被送到 $f(x_0)$；另一方面，如果连续性失败，就能从每一个越来越小的邻域中挑出一个坏点，把这些坏点组成反例数列。

### 3.2 连续性推出序列极限

假设 $f$ 在 $x_0$ 处连续。取任意 $x_n\in D$ 且 $x_n\to x_0$。要证明 $f(x_n)\to f(x_0)$，给定任意 $\varepsilon>0$。

由连续性的 $\varepsilon$-$\delta$ 定义，存在 $\delta>0$，使得对所有 $x\in D$，

$$
|x-x_0|<\delta
\quad\Longrightarrow\quad
|f(x)-f(x_0)|<\varepsilon.
$$

另一方面，$x_n\to x_0$，所以存在 $N\in\mathbb N$，使得当 $n\geq N$ 时，

$$
|x_n-x_0|<\delta.
$$

把 $x=x_n$ 代入连续性的邻域控制，得到

$$
|f(x_n)-f(x_0)|<\varepsilon
\qquad(n\geq N).
$$

这正是 $f(x_n)\to f(x_0)$ 的定义。

### 3.3 反方向

现在假设对每一个 $x_n\in D$ 且 $x_n\to x_0$ 的数列，都有 $f(x_n)\to f(x_0)$。反设 $f$ 在 $x_0$ 处不连续。

“不连续”就是否定连续性的量词：存在某个 $\varepsilon_0>0$，使得对每一个 $\delta>0$，都能找到 $x\in D$ 满足

$$
|x-x_0|<\delta,
\qquad
|f(x)-f(x_0)|\geq\varepsilon_0.
$$

对每一个正整数 $n$，令 $\delta=1/n$。于是可以选择 $x_n\in D$，使得

$$
|x_n-x_0|<\frac1n,
\qquad
|f(x_n)-f(x_0)|\geq\varepsilon_0.
$$

由 $1/n\to0$ 和夹逼定理，

$$
0\leq|x_n-x_0|<\frac1n\longrightarrow0,
$$

因此 $x_n\to x_0$。按照假设，应该有 $f(x_n)\to f(x_0)$。但这不可能，因为所有 $n$ 都满足

$$
|f(x_n)-f(x_0)|\geq\varepsilon_0,
$$

它们永远不能进入半径 $\varepsilon_0$ 的输出邻域。矛盾。因此 $f$ 必须在 $x_0$ 处连续。

### 3.4 什么时候用哪一种说法

直接的 $\varepsilon$-$\delta$ 定义适合构造一个明确的邻域半径；序列判别适合发现不连续性。要证明连续，通常从输入误差出发寻找 $\delta$；要证明不连续，通常寻找一列 $x_n\to x_0$，但 $f(x_n)$ 不趋于 $f(x_0)$。

例如阶跃函数

$$
h(x)=
\begin{cases}
0,&x<0,\\\\
1,&x\geq0
\end{cases}
$$

在 $0$ 处不连续。取 $x_n=-1/n$，则 $x_n\to0$，但 $h(x_n)=0$，而 $h(0)=1$，所以 $h(x_n)$ 不可能趋于 $h(0)$。

## 4. 连续函数的运算

序列判别法允许我们直接调用上一篇已经证明的极限运算法则。若 $f,g$ 在 $x_0$ 处连续，且 $x_n\to x_0$，那么

$$
f(x_n)\to f(x_0),
\qquad
g(x_n)\to g(x_0).
$$

于是数列极限的四则运算可以直接用来证明函数运算的连续性。分别看和、积、商与复合。

### 4.1 和、差与常数倍

对任意 $x\in D$，定义

$$
(f+g)(x):=f(x)+g(x),
\qquad
(f-g)(x):=f(x)-g(x),
$$

以及 $(cf)(x):=cf(x)$。若 $x_n\to x_0$，由序列判别法和数列的和、差、常数倍定理，

$$
\begin{aligned}
(f+g)(x_n)
&=f(x_n)+g(x_n)\\\\
&\longrightarrow f(x_0)+g(x_0)\\\\
&=(f+g)(x_0).
\end{aligned}
$$

因此 $f+g$ 在 $x_0$ 处连续。差和常数倍完全相同：

$$
(f-g)(x_n)\to(f-g)(x_0),
\qquad
(cf)(x_n)\to(cf)(x_0).
$$

### 4.2 乘积

仍取任意 $x_n\to x_0$。由连续性，

$$
f(x_n)\to f(x_0),
\qquad
g(x_n)\to g(x_0).
$$

上一篇的乘积极限定理给出

$$
f(x_n)g(x_n)\longrightarrow f(x_0)g(x_0).
$$

而

$$
(fg)(x_n)=f(x_n)g(x_n),
\qquad
(fg)(x_0)=f(x_0)g(x_0).
$$

所以 $fg$ 在 $x_0$ 处连续。

### 4.3 商

设 $g(x_0)\neq0$，并把商定义在 $g(x)\neq0$ 的点上。由 $g$ 在 $x_0$ 处连续，

$$
g(x_n)\longrightarrow g(x_0)\neq0.
$$

上一篇的倒数定理给出

$$
\frac1{g(x_n)}\longrightarrow\frac1{g(x_0)}.
$$

再用乘积极限，

$$
\frac{f(x_n)}{g(x_n)}
=f(x_n)\frac1{g(x_n)}
\longrightarrow
f(x_0)\frac1{g(x_0)}
=\frac{f(x_0)}{g(x_0)}.
$$

因此 $f/g$ 在 $x_0$ 处连续。分母不为零是定义域条件，也是倒数误差估计能够成立的条件。

### 4.4 复合函数

设 $f:D\to\mathbb R$ 在 $x_0$ 处连续，$g:E\to\mathbb R$ 在 $f(x_0)$ 处连续，并且 $f(D)\subseteq E$。则复合函数

$$
(g\circ f)(x):=g(f(x))
$$

在 $x_0$ 处连续。

证明仍从任意数列开始。若 $x_n\to x_0$，由 $f$ 的连续性，

$$
f(x_n)\longrightarrow f(x_0).
$$

再把这个数列作为 $g$ 的输入，由 $g$ 在 $f(x_0)$ 处连续，

$$
g(f(x_n))\longrightarrow g(f(x_0)).
$$

也就是

$$
(g\circ f)(x_n)\longrightarrow(g\circ f)(x_0).
$$

序列判别法完成证明。

## 5. 从基本函数得到多项式和有理函数

### 5.1 恒等函数和幂函数

恒等函数 $\operatorname{id}(x)=x$ 在每一点连续，因为

$$
|\operatorname{id}(x)-\operatorname{id}(x_0)|=|x-x_0|.
$$

由乘积的连续性，若 $x\mapsto x^m$ 连续，则

$$
x\longmapsto x^{m+1}=x^m\cdot x
$$

也连续。归纳可知，对每个正整数 $m$，幂函数 $x\mapsto x^m$ 在整个实数轴上连续。常数函数 $x\mapsto c$ 则直接连续，因为对任意 $x,x_0$ 都有

$$
|c-c|=0.
$$

### 5.2 多项式

多项式写成

$$
p(x)=a_0+a_1x+a_2x^2+\cdots+a_mx^m.
$$

每一项 $a_jx^j$ 都连续；有限次相加仍连续。因此每个多项式在 $\mathbb R$ 的每一点连续。

这里“有限次”很重要。连续函数的无穷和需要额外的收敛控制，不能仅凭每一项连续就交换极限与无穷求和。这个问题会在函数列与级数中单独出现。

### 5.3 有理函数

有理函数是两个多项式的商：

$$
r(x)=\frac{p(x)}{q(x)}.
$$

在满足 $q(x)\neq0$ 的定义域上，$p$ 和 $q$ 连续；由商的连续性，$r$ 在每个 $q(x_0)\neq0$ 的点连续。

例如

$$
r(x)=\frac{x^2+1}{x-2}
$$

在 $\mathbb R\setminus\lbrace 2\rbrace$ 上连续。$x=2$ 不是一个可以用“取更小 $\delta$”修复的连续点，因为函数在这里根本没有定义。

## 6. 点态连续与一致连续

### 6.1 点态连续

“$f$ 在每一点连续”是一个点一个点的陈述：对每个 $x_0\in D$，给定 $\varepsilon$ 后，可以选择一个适合该点的 $\delta$。这个 $\delta$ 可以随着 $x_0$ 改变。

把量词写出来就是

$$
(\forall x_0\in D)(\forall\varepsilon>0)(\exists\delta>0)(\forall x\in D)
\quad
|x-x_0|<\delta
\Longrightarrow
|f(x)-f(x_0)|<\varepsilon.
$$

### 6.2 一致连续

称 $f:D\to\mathbb R$ 在 $D$ 上**一致连续**，如果

$$
(\forall\varepsilon>0)(\exists\delta>0)(\forall x,y\in D)
\quad
|x-y|<\delta
\Longrightarrow
|f(x)-f(y)|<\varepsilon.
$$

与点态连续相比，这里选出的 $\delta$ 不再随 $x_0$ 改变，而是一次控制整个定义域中的任意两点。因此一致连续必然推出逐点连续：固定一个 $x_0$，在一致连续定义中取 $y=x_0$，便得到

$$
|x-x_0|<\delta
\Longrightarrow
|f(x)-f(x_0)|<\varepsilon.
$$

反方向一般不成立。

### 6.3 例：$x^2$ 在 $[0,1]$ 上

在区间 $[0,1]$ 上，

$$
|x^2-y^2|=|x-y||x+y|.
$$

由于 $x,y\in[0,1]$，有 $|x+y|\leq2$，所以

$$
|x^2-y^2|\leq2|x-y|.
$$

给定 $\varepsilon>0$，取 $\delta=\varepsilon/2$。当 $|x-y|<\delta$ 时，

$$
|x^2-y^2|\leq2|x-y|<2\delta=\varepsilon.
$$

这个证明中 $\delta$ 只依赖于 $\varepsilon$，不依赖于 $x,y$，所以 $x^2$ 在 $[0,1]$ 上一致连续。

### 6.4 $x^2$ 在整个实数轴上不是一致连续

取两列

$$
x_n=n,
\qquad
y_n=n+\frac1n.
$$

则

$$
|x_n-y_n|=\frac1n\longrightarrow0,
$$

但

$$
\begin{aligned}
|x_n^2-y_n^2|
&=\left|n^2-\left(n+\frac1n\right)^2\right|\\\\
&=2+\frac1{n^2}\geq2.
\end{aligned}
$$

输出距离没有趋于零。若 $x^2$ 在 $\mathbb R$ 上一致连续，取任意 $\varepsilon=1$，就存在一个固定 $\delta>0$，使所有输入距离小于 $\delta$ 时输出距离都小于 $1$。但 $1/n<\delta$ 对足够大的 $n$ 成立，同时输出距离仍至少为 $2$，矛盾。因此 $x^2$ 虽然在每一点连续，却不是整个实数轴上的一致连续函数。

这正是点态和一致两个概念的差别：在点态连续中，远处的函数图像可以使用越来越小的局部尺度；一致连续要求一个尺度控制整个定义域。

## 7. 连续函数在附近不变号

**命题 7.1（符号稳定性）。** 若 $f$ 在 $x_0$ 处连续且 $f(x_0)>0$，则存在 $\delta>0$，使得

$$
x\in D,\quad |x-x_0|<\delta
\quad\Longrightarrow\quad
f(x)>0.
$$

**证明。** 取

$$
\varepsilon:=\frac{f(x_0)}2>0.
$$

由连续性，存在 $\delta>0$，使 $|x-x_0|<\delta$ 时

$$
|f(x)-f(x_0)|<\frac{f(x_0)}2.
$$

于是

$$
f(x)
\displaystyle >f(x_0)-\frac{f(x_0)}2
=\frac{f(x_0)}2>0.
$$

证毕。若 $f(x_0)<0$，对 $-f$ 使用同样的结论即可得到 $f(x)<0$ 的邻域。

这个小结论在介值定理中会再次用到：连续函数在一点严格为正或严格为负时，足够靠近这一点，符号不会改变。

## 8. 介值定理

### 8.1 定理陈述

**定理 8.1（介值定理）。** 设 $f:[a,b]\to\mathbb R$ 在闭区间 $[a,b]$ 上连续。若

$$
f(a)\leq y\leq f(b)
$$

或

$$
f(b)\leq y\leq f(a),
$$

则存在 $c\in[a,b]$，使得

$$
f(c)=y.
$$

换句话说，连续函数从一个端点值走到另一个端点值时，会取得两个端点之间的每一个数值。

证明只需处理

$$
f(a)<y<f(b).
$$

如果 $y=f(a)$ 或 $y=f(b)$，端点本身就是所需的 $c$；如果端点顺序相反，可以把 $-f$ 换成 $f$，或交换 $a,b$。

### 8.2 把目标值移到零

定义

$$
g(x):=f(x)-y.
$$

由连续函数减去常数仍连续，并且

$$
g(a)=f(a)-y<0,
\qquad
g(b)=f(b)-y>0.
$$

因此只需要证明：如果连续函数在 $a$ 处为负、在 $b$ 处为正，那么它在中间某点取值 $0$。

### 8.3 用上确界定位分界点

考虑集合

$$
E:=\lbrace x\in[a,b]:g(x)\leq0\rbrace.
$$

集合 $E$ 非空，因为 $a\in E$；它有上界 $b$，因为 $E\subseteq[a,b]$。由实数完备性，$E$ 有上确界。记

$$
c:=\sup E.
$$

显然 $a\leq c\leq b$。我们证明 $g(c)=0$。

### 8.4 不能有 $g(c)>0$

反设 $g(c)>0$。取

$$
\varepsilon:=\frac{g(c)}2>0.
$$

由 $g$ 在 $c$ 处连续，存在 $\delta>0$，使得

$$
|x-c|<\delta
\quad\Longrightarrow\quad
|g(x)-g(c)|<\frac{g(c)}2.
$$

于是这个邻域中的每个 $x$ 都满足

$$
g(x)>g(c)-\frac{g(c)}2=\frac{g(c)}2>0.
$$

另一方面，$c=\sup E$ 意味着任意小于 $c$ 的数都不能作为 $E$ 的上界。取一个满足 $c-\delta<x\leq c$ 的 $x\in E$。如果 $c=a$，则 $g(a)<0$ 与 $g(c)>0$ 已经矛盾，所以可以取这样的左侧点。对这个 $x$，一方面 $x\in E$ 给出 $g(x)\leq0$，另一方面 $|x-c|<\delta$ 给出 $g(x)>0$，矛盾。

因此

$$
g(c)\not>0.
$$

### 8.5 不能有 $g(c)<0$

反设 $g(c)<0$。取

$$
\varepsilon:=\frac{-g(c)}2>0.
$$

由连续性，存在 $\delta_1>0$，使得

$$
|x-c|<\delta_1
\quad\Longrightarrow\quad
|g(x)-g(c)|<\frac{-g(c)}2.
$$

于是

$$
g(x)<g(c)+\frac{-g(c)}2=\frac{g(c)}2<0.
$$

先注意到 $c<b$。如果 $c=b$，那么 $g(c)=g(b)>0$，这与当前假设 $g(c)<0$ 直接矛盾。因此可以在 $c$ 的右侧仍留出一小段区间。

取

$$
0<\delta<\min\lbrace\delta_1,b-c\rbrace.
$$

令 $z=c+\delta/2$。则 $z\in[a,b]$，且 $|z-c|<\delta_1$，所以 $g(z)<0$，即 $z\in E$。但 $z>c$，这与 $c$ 是 $E$ 的上界矛盾。

因此

$$
g(c)\not<0.
$$

结合上一节的 $g(c)\not>0$，得到 $g(c)=0$。于是

$$
f(c)-y=g(c)=0,
$$

即 $f(c)=y$。介值定理得证。

### 8.6 介值定理与完备性的关系

证明中真正决定位置的不是图像直觉，而是集合 $E$ 的上确界 $c$。如果数系缺少上确界，分界点可能不在数系中，证明就无法完成。连续性负责说明分界点两侧的函数值不能留下间隙；实数完备性负责保证这个分界点存在。

这与上一篇文章中用上确界构造 $\sqrt2$ 的思路完全同型：先收集所有满足某个不等式的点，再取它们的最小上界，最后用连续性或代数估计验证上确界正好满足等式。

## 9. 同一个条件的几种写法

同一个条件可以换几种方式来读。它们的内容相同，但适合处理的问题不同。

### 9.1 邻域语言

对每个输出误差 $\varepsilon$，存在输入半径 $\delta$，使

$$
f\bigl(D\cap(x_0-\delta,x_0+\delta)\bigr)
\subseteq
(f(x_0)-\varepsilon,f(x_0)+\varepsilon).
$$

它强调“一个输入邻域被送入一个输出邻域”。

### 9.2 极限语言

当 $x\to x_0$ 时，

$$
f(x)\to f(x_0).
$$

它把连续性接到极限运算上。

### 9.3 序列语言

对每个 $x_n\to x_0$，都有

$$
f(x_n)\to f(x_0).
$$

它特别适合构造不连续性的反例。

### 9.4 用反例说明不连续

不连续意味着存在一个固定的输出误差 $\varepsilon_0>0$，无论输入邻域缩得多小，都能找到一个输入点把输出误差留在 $\varepsilon_0$ 之外。用 $\delta=1/n$ 逐次选择这些点，就得到坏点数列。

这几种说法只是同一组量词的不同读法：邻域语言便于做估计，序列语言便于找反例。

## 10. 小结

连续性把“数列趋近”变成了函数在一点附近的控制。证明时真正反复使用的只有两件事：先固定输出误差，再把它分配给输入误差；遇到乘积或商，先给变化中的因子找一个界。序列判别法把同一件事换成数列语言，介值定理则说明这种局部控制如何依靠实数完备性产生一个确切的取值点。后面的导数、积分和一致收敛，都会继续使用这套误差语言。

## 参考文献

1. Augustin-Louis Cauchy, *Cours d'analyse de l'École Royale Polytechnique*, 1821.
2. Karl Weierstrass, lectures on the arithmetization of analysis, nineteenth century.
3. Tom M. Apostol, *Mathematical Analysis*, 2nd ed., Addison-Wesley, 1974.
4. Walter Rudin, *Principles of Mathematical Analysis*, 3rd ed., McGraw-Hill, 1976.
5. Terence Tao, *Analysis I*, 3rd ed., Hindustan Book Agency, 2016.

---

## 附录

如下为正文附录补充。

## 附录 A：符号与量词

### A.1 定义域、陪域和像

写作

$$
f:D\to\mathbb R
$$

表示 $f$ 的定义域是 $D$，陪域是 $\mathbb R$。对每个 $x\in D$，函数指定唯一的 $f(x)\in\mathbb R$。函数值 $f(x)$ 的集合

$$
f(D):=\lbrace f(x):x\in D\rbrace
$$

称为像集；像集是陪域的子集，不一定等于陪域。

### A.2 连续性的量词否定

连续性写成

$$
(\forall\varepsilon>0)(\exists\delta>0)(\forall x\in D)
\quad
|x-x_0|<\delta
\Longrightarrow
|f(x)-f(x_0)|<\varepsilon.
$$

逐层否定后，不连续就是

$$
(\exists\varepsilon_0>0)(\forall\delta>0)(\exists x\in D)
\quad
|x-x_0|<\delta
\quad\text{且}\quad
|f(x)-f(x_0)|\geq\varepsilon_0.
$$

第二个式子中的 $\varepsilon_0$ 可以固定不变；变化的是输入邻域半径 $\delta$ 和从该邻域中挑出的坏点 $x$。正因为 $\varepsilon_0$ 固定，令 $\delta=1/n$ 才能得到一个输出误差始终不小于固定正数的数列。

### A.3 点态连续与一致连续的量词差别

点态连续：

$$
(\forall x_0\in D)(\forall\varepsilon>0)(\exists\delta>0)(\forall x\in D)\cdots
$$

一致连续：

$$
(\forall\varepsilon>0)(\exists\delta>0)(\forall x,y\in D)\cdots
$$

点态连续中的 $\delta$ 可以依赖 $x_0$；一致连续中的 $\delta$ 必须在所有点之间共用。把 $\delta$ 从 $x_0$ 后面移到 $x_0$ 前面，就是从局部性质变成全局统一控制的关键。

## 附录 B：四则运算的直接 ε-δ 证明

这里不调用序列判别，直接从定义验证和、积、商的连续性；这样每个估计的来源都留在纸面上。

### B.1 和

设 $f,g$ 在 $x_0$ 处连续。给定 $\varepsilon>0$。

由 $f$ 连续，存在 $\delta_1>0$，使

$$
|x-x_0|<\delta_1
\Longrightarrow
|f(x)-f(x_0)|<\frac\varepsilon2.
$$

由 $g$ 连续，存在 $\delta_2>0$，使

$$
|x-x_0|<\delta_2
\Longrightarrow
|g(x)-g(x_0)|<\frac\varepsilon2.
$$

取

$$
\delta:=\min\lbrace\delta_1,\delta_2\rbrace.
$$

当 $|x-x_0|<\delta$ 时，两项同时成立。于是

$$
\begin{aligned}
|(f+g)(x)-(f+g)(x_0)|
&=|(f(x)-f(x_0))+(g(x)-g(x_0))|\\\\
&\leq |f(x)-f(x_0)|+|g(x)-g(x_0)|\\\\
&<\frac\varepsilon2+\frac\varepsilon2\\\\
&=\varepsilon.
\end{aligned}
$$

因此 $f+g$ 在 $x_0$ 处连续。差的证明只把加号换成减号，三角不等式仍然给出同一个上界。

### B.2 积：先控制一个因子的大小

设 $f,g$ 在 $x_0$ 处连续。目标误差为

$$
|f(x)g(x)-f(x_0)g(x_0)|.
$$

插入中间项 $f(x)g(x_0)$：

$$
\begin{aligned}
f(x)g(x)-f(x_0)g(x_0)
&=f(x)(g(x)-g(x_0))\\\\
&\quad+g(x_0)(f(x)-f(x_0)).
\end{aligned}
$$

于是

$$
\begin{aligned}
|f(x)g(x)-f(x_0)g(x_0)|
&\leq |f(x)|\cdot|g(x)-g(x_0)|\\\\
&\quad+|g(x_0)|\cdot|f(x)-f(x_0)|.
\end{aligned}
$$

第一项中的 $|f(x)|$ 还随 $x$ 变化。先用 $f$ 的连续性取一个半径 $\delta_0>0$，使

$$
|x-x_0|<\delta_0
\Longrightarrow
|f(x)-f(x_0)|<1.
$$

于是

$$
|f(x)|\leq |f(x_0)|+|f(x)-f(x_0)|<|f(x_0)|+1.
$$

记 $M:=|f(x_0)|+1$。再由 $g$ 的连续性选择 $\delta_1>0$，使

$$
|x-x_0|<\delta_1
\Longrightarrow
|g(x)-g(x_0)|<\frac{\varepsilon}{2M}.
$$

由 $f$ 的连续性选择 $\delta_2>0$，使

$$
|x-x_0|<\delta_2
\Longrightarrow
|f(x)-f(x_0)|<
\frac{\varepsilon}{2(|g(x_0)|+1)}.
$$

取

$$
\delta:=\min\lbrace\delta_0,\delta_1,\delta_2\rbrace.
$$

则

$$
\begin{aligned}
|f(x)g(x)-f(x_0)g(x_0)|
&<M\frac{\varepsilon}{2M}
 +|g(x_0)|\frac{\varepsilon}{2(|g(x_0)|+1)}\\\\
&<\frac\varepsilon2+\frac\varepsilon2\\\\
&=\varepsilon.
\end{aligned}
$$

所以 $fg$ 连续。证明中先控制 $|f(x)|$，正是乘积极限证明中“收敛数列有界”在函数局部版本中的对应物。

### B.3 商：先保证分母有正下界

设 $f,g$ 在 $x_0$ 处连续且 $g(x_0)\neq0$。先由 $g$ 连续，取 $\delta_0>0$，使

$$
|x-x_0|<\delta_0
\Longrightarrow
|g(x)-g(x_0)|<\frac{|g(x_0)|}{2}.
$$

反三角不等式给出

$$
|g(x)|
\geq |g(x_0)|-|g(x)-g(x_0)|
\displaystyle >\frac{|g(x_0)|}{2}.
$$

因此在这个邻域内分母不会为零。再写出

$$
\frac{f(x)}{g(x)}-\frac{f(x_0)}{g(x_0)}
=\frac{g(x_0)(f(x)-f(x_0))-f(x_0)(g(x)-g(x_0))}{g(x)g(x_0)}.
$$

从而

$$
\begin{aligned}
\left|\frac{f(x)}{g(x)}-\frac{f(x_0)}{g(x_0)}\right|
&\leq
\frac{2}{|g(x_0)|^2}
\bigl(|g(x_0)|\cdot|f(x)-f(x_0)|\\\\
&\qquad+|f(x_0)|\cdot|g(x)-g(x_0)|\bigr).
\end{aligned}
$$

给定 $\varepsilon>0$，由 $f,g$ 的连续性分别选择 $\delta_1,\delta_2$，使

$$
|f(x)-f(x_0)|<\frac{\varepsilon|g(x_0)|}{4},
$$

以及

$$
|g(x)-g(x_0)|<
\frac{\varepsilon|g(x_0)|^2}{4(|f(x_0)|+1)}.
$$

再取 $\delta=\min\lbrace\delta_0,\delta_1,\delta_2\rbrace$。代入上面的估计即可得到输出误差小于 $\varepsilon$。因此 $f/g$ 在 $x_0$ 处连续。

## 附录 C：序列反例的构造

设要证明 $f$ 在 $x_0$ 处不连续。由量词否定，先固定一个 $\varepsilon_0>0$，然后对每个 $n$ 在半径 $1/n$ 的邻域中选取一个坏点 $x_n$，使

$$
|x_n-x_0|<\frac1n,
\qquad
|f(x_n)-f(x_0)|\geq\varepsilon_0.
$$

于是 $x_n\to x_0$，而输出误差始终留在 $\varepsilon_0$ 之外，故 $f(x_n)$ 不可能收敛到 $f(x_0)$。

例如对

$$
f(x)=\begin{cases}
1,&x>0,\\\\
0,&x\leq0,
\end{cases}
$$

在 $0$ 处取 $x_n=1/n$，则 $x_n\to0$，但 $f(x_n)=1$，而 $f(0)=0$。取 $\varepsilon_0=1/2$，所有 $n$ 都有

$$
|f(x_n)-f(0)|=1\geq\frac12.
$$

跳跃、振荡和可去间断点都可以用同样的办法处理，差别只在于坏点 $x_n$ 的选法。

## 附录 D：介值定理证明中的上确界细节

在主文中令

$$
E=\lbrace x\in[a,b]:g(x)\leq0\rbrace,
\qquad c=\sup E.
$$

这里每一步都有明确依据。

### D.1 $E$ 非空且有上界

因为 $g(a)<0$，所以 $g(a)\leq0$，从而 $a\in E$，故 $E\neq\varnothing$。

又因为 $E\subseteq[a,b]$，所以对每个 $x\in E$ 都有 $x\leq b$，即 $b$ 是 $E$ 的上界。

实数完备性保证非空有上界集合存在上确界，因此 $c=\sup E$ 存在。

### D.2 上确界的逼近性质

对任意 $\eta>0$，数 $c-\eta$ 不可能是 $E$ 的上界；否则它会小于最小上界 $c$。所以存在 $x_\eta\in E$，满足

$$
c-\eta<x_\eta\leq c.
$$

这正是主文中从左侧逼近 $c$ 所用的点。注意这里并没有断言 $c\in E$；$c$ 是否属于 $E$ 正是连续性需要帮助我们判断的事情。

### D.3 为什么 $g(c)>0$ 会矛盾

若 $g(c)>0$，连续性给出某个 $\eta>0$，使

$$
|x-c|<\eta\Longrightarrow g(x)>0.
$$

由 D.2 取 $x_\eta\in E$ 满足 $c-\eta<x_\eta\leq c$。若 $x_\eta<c$，则 $|x_\eta-c|<\eta$；若 $x_\eta=c$，则直接有 $g(c)>0$。两种情况都与 $x_\eta\in E$ 的 $g(x_\eta)\leq0$ 矛盾。

### D.4 为什么 $g(c)<0$ 会矛盾

若 $g(c)<0$，连续性给出 $\eta>0$，使

$$
|x-c|<\eta\Longrightarrow g(x)<0.
$$

因为 $g(b)>0$，所以 $b\notin E$，从而 $c<b$。取 $0<h<\min\lbrace\eta,b-c\rbrace$，令 $z=c+h$。则 $z\in[a,b]$ 且 $|z-c|<\eta$，所以 $g(z)<0$，即 $z\in E$。但 $z>c$，与 $c$ 是 $E$ 的上界矛盾。

所以 $g(c)$ 既不大于零，也不小于零，只能等于零。

## 附录 E：Lipschitz 条件是统一连续性的直接来源

如果存在常数 $L\geq0$，使得对任意 $x,y\in D$，

$$
|f(x)-f(y)|\leq L|x-y|,
$$

则称 $f$ 满足 Lipschitz 条件。若 $L=0$，函数在 $D$ 上为常数；若 $L>0$，给定 $\varepsilon>0$，取

$$
\delta:=\frac\varepsilon L.
$$

当 $|x-y|<\delta$ 时，

$$
|f(x)-f(y)|\leq L|x-y|<L\frac\varepsilon L=\varepsilon.
$$

因此 Lipschitz 条件推出一致连续。线性函数、绝对值函数和有界区间上的 $x^2$ 都可以通过这个条件处理。

反过来，一致连续不一定满足某个固定的 Lipschitz 斜率。例如 $f(x)=\sqrt x$ 在 $[0,1]$ 上一致连续。对 $x,y\geq0$，不妨设 $x\geq y$，则

$$
|\sqrt x-\sqrt y|^2
=(\sqrt x-\sqrt y)^2
\leq(\sqrt x-\sqrt y)(\sqrt x+\sqrt y)
=x-y
=|x-y|.
$$

给定 $\varepsilon>0$，取 $\delta=\varepsilon^2$。当 $|x-y|<\delta$ 时，

$$
|\sqrt x-\sqrt y|
<\sqrt{\delta}
=\varepsilon.
$$

所以平方根在 $[0,1]$ 上一致连续。它在 $0$ 附近没有一个固定的 Lipschitz 斜率，但这不妨碍它拥有统一的平方根尺度控制。

## 附录 F：几处容易混淆的地方

### F.1 把 $\delta$ 选成依赖于 $x$

错误做法是：给定 $x$ 后，再令 $\delta=|x-x_0|+1$。这样得到的半径不是在控制一个邻域，而是在为已经选定的点寻找一个宽松条件。连续性要求先选好一个 $\delta$，然后所有满足 $|x-x_0|<\delta$ 的点同时通过检验。

### F.2 只验证 $x\to x_0$ 的一条路径

在实函数中，$x$ 是一维变量，但仍然有无穷多种趋近方式。验证 $x_n\to x_0$ 的一列只能说明这一列没有制造问题；要证明连续，必须处理所有趋近数列，或直接完成 $\varepsilon$-$\delta$ 证明。

### F.3 把函数值和极限混为一谈

连续要求

$$
\lim_{x\to x_0}f(x)=f(x_0).
$$

只知道左边存在，不足以得到连续；只知道右边有定义，也不足以得到连续。可去间断点正是这两个量不相等的情形。

### F.4 忽略定义域

定义域为 $D$ 时，连续性的量词是 $\forall x\in D$，不是对所有 $x\in\mathbb R$。例如 $\sqrt x$ 的定义域是 $[0,\infty)$；在 $0$ 处连续性只需要考虑定义域内的 $x\geq0$。
