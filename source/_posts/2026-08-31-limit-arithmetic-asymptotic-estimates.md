---
title: "极限如何运算：从 ε-N 定义到渐近估计"
date: 2026-08-31 10:00:00
categories:
  - 基础数学
  - 分析学
tags:
  - 数学基础
  - 分析学
  - 数列极限
  - 渐近估计
  - 大 O 记号
  - 数学证明
mathjax: true
toc: true
toc_number: false
comments: true
---

[上一篇关于实数完备性的文章](/2026/08/30/2026-08-30-real-number-completeness-supremum-cauchy/)解决了一个存在性问题：一个无限逼近过程在什么条件下真的会到达某个实数。当极限已经存在，接下来的问题是：极限能否参与代数运算，又应该怎样估计误差。

设

$$
x&#95;n\longrightarrow x,
\qquad
y&#95;n\longrightarrow y.
$$

人们很快会写出

$$
x&#95;n+y&#95;n\longrightarrow x+y,
\qquad
x&#95;ny&#95;n\longrightarrow xy.
$$

真正需要理解的是等式背后的决定：目标误差为什么要分成两份？乘积的差应该添加再减去哪一项？为什么乘法会突然用到有界性？除法中的分母如何保证不会靠近零？

这些问题把 $\varepsilon$-$N$ 定义变成了一套可以反复使用的推理方法。它的核心是将一个尚未受控的总误差，分解成已知会趋于零的局部误差。

<!--more-->

## 1. 极限定义中的两层逻辑

本篇中，$\mathbb N=\lbrace1,2,3,\ldots\rbrace$ 表示正整数集，$\mathbb R$ 表示实数集。符号 $:=$ 表示“定义为”，$\Longrightarrow$ 表示前者成立时后者也成立。

数列 $(x&#95;n)$ 收敛到实数 $x$ 的定义是：对每一个 $\varepsilon>0$，都存在正整数 $N$，使得

$$
n\geq N
\quad\Longrightarrow\quad
|x&#95;n-x|<\varepsilon.
$$

$\varepsilon$ 是先给定的误差容许量，$N$ 是为了达到这个精度而选择的起始位置。因此一个极限证明有两层逻辑：

1. 先观察目标式子，找到足以使它小于 $\varepsilon$ 的条件；
2. 再回到已知的收敛条件，为这些局部条件分别选择起始位置。

这个顺序值得特别留意。写成定稿时，证明往往从“给定 $\varepsilon>0$”开始，然后直接给出各个 $N$。实际思考通常是反向的：先把总误差拆开，再看每一块需要多小。

### 1.1 “最终成立”的含义

如果某个命题 $P(n)$ 对所有足够大的 $n$ 成立，就说 $P(n)$ **最终成立**。准确地说，这表示存在 $N\in\mathbb N$，使得每个 $n\geq N$ 都满足 $P(n)$。

两个最终成立的条件可以同时使用。如果第一个条件从 $N&#95;1$ 开始成立，第二个从 $N&#95;2$ 开始成立，取

$$
N:=\max\lbrace N&#95;1,N&#95;2\rbrace
$$

即可。符号 $\max\lbrace N&#95;1,N&#95;2\rbrace$ 表示两个数中较大的一个。当 $n\geq N$ 时，同时有 $n\geq N&#95;1$ 与 $n\geq N&#95;2$。有限多个条件也可以用同样的方法合并。

## 2. 加法：把误差预算分成两份

设

$$
x&#95;n\longrightarrow x,
\qquad
y&#95;n\longrightarrow y.
$$

要证明 $x&#95;n+y&#95;n\to x+y$，目标误差是

$$
|(x&#95;n+y&#95;n)-(x+y)|.
$$

先整理括号，再使用三角不等式：

$$
\begin{aligned}
|(x&#95;n+y&#95;n)-(x+y)|
&=|(x&#95;n-x)+(y&#95;n-y)|\\\\
&\leq |x&#95;n-x|+|y&#95;n-y|.
\end{aligned}
$$

右边已经出现两个已知趋于零的误差。如果希望它们的和小于 $\varepsilon$，一个自然的安排是

$$
|x&#95;n-x|<\frac{\varepsilon}{2},
\qquad
|y&#95;n-y|<\frac{\varepsilon}{2}.
$$

这就是误差预算。$\varepsilon/2$ 没有特殊的神秘性；任意正数 $\varepsilon&#95;1,\varepsilon&#95;2$ 只要满足

$$
\varepsilon&#95;1+\varepsilon&#95;2\leq\varepsilon
$$

都可以。平分的好处是不需要再引入额外选择。

同一个分解立即给出差的极限：

$$
x&#95;n-y&#95;n\longrightarrow x-y.
$$

对固定实数 $c$，还有

$$
cx&#95;n\longrightarrow cx.
$$

因为它的误差正好是 $|c|\cdot|x&#95;n-x|$。当 $c\neq0$ 时，只需把原数列的误差控制到 $\varepsilon/|c|$；$c=0$ 时结论直接成立。完整证明见附录 B。

## 3. 乘法：中间项怎样想到

乘积的目标误差是

$$
|x&#95;ny&#95;n-xy|.
$$

它还没有显示出 $x&#95;n-x$ 或 $y&#95;n-y$，因此无法直接使用已知条件。需要在 $x&#95;ny&#95;n$ 与 $xy$ 之间放入一个中间项。

一种选择是添加再减去 $x&#95;ny$：

$$
\begin{aligned}
x&#95;ny&#95;n-xy
&=x&#95;ny&#95;n-x&#95;ny+x&#95;ny-xy\\\\
&=x&#95;n(y&#95;n-y)+y(x&#95;n-x).
\end{aligned}
$$

这样选的原因是，第一项包含 $y&#95;n-y$，第二项包含 $x&#95;n-x$，每一项都有一个已知的小量。取绝对值后，

$$
|x&#95;ny&#95;n-xy|
\leq
|x&#95;n|\cdot|y&#95;n-y|
+|y|\cdot|x&#95;n-x|.
$$

此时多出了一个问题：$|x&#95;n|$ 随 $n$ 变化，似乎可能放大 $|y&#95;n-y|$。收敛恰好排除了这种可能。

**收敛数列必有界。** 由 $x&#95;n\to x$，取误差 $1$，可以找到 $N&#95;0$，使 $n\geq N&#95;0$ 时

$$
|x&#95;n-x|<1.
$$

于是

$$
|x&#95;n|
\leq |x|+|x&#95;n-x|
{}<|x|+1.
$$

数列的尾部被 $|x|+1$ 控制，前面只有有限多项，因此整个数列有界。设一个正数 $M$ 满足 $|x&#95;n|\leq M$，上面的乘积误差就变成

$$
|x&#95;ny&#95;n-xy|
\leq
M|y&#95;n-y|+|y|\cdot|x&#95;n-x|.
$$

现在可以分配误差预算。第一项控制到 $\varepsilon/2$，第二项也控制到 $\varepsilon/2$，总和便小于 $\varepsilon$。

![乘积极限中的误差分解与预算](/images/notes/assets/mathematical-foundations/limit-error-budget.svg)

也可以添加再减去 $xy&#95;n$，得到

$$
x&#95;ny&#95;n-xy
=y&#95;n(x&#95;n-x)+x(y&#95;n-y).
$$

这时需要使用 $(y&#95;n)$ 的有界性。两种分解的本质相同：在两个乘积之间只改变一个因子，再改变另一个因子。附录 C 与 D 分别证明收敛数列有界和乘积极限。

## 4. 倒数与商：先保证分母远离零

设 $x&#95;n\to x$ 且 $x\neq0$。倒数的差可以通分：

$$
\left|
\frac1{x&#95;n}-\frac1x
\right|
=
\frac{|x&#95;n-x|}{|x&#95;n|\cdot|x|}.
$$

分子趋于零，分母中却有一个随 $n$ 变化的 $|x&#95;n|$。因此要为 $|x&#95;n|$ 建立一个正的下界。

因为 $x\neq0$，数 $|x|/2$ 严格大于零。由 $x&#95;n\to x$，当 $n$ 足够大时，

$$
|x&#95;n-x|<\frac{|x|}{2}.
$$

利用反三角不等式，

$$
\begin{aligned}
|x&#95;n|
&\geq |x|-|x&#95;n-x|\\\\
&>\frac{|x|}{2}.
\end{aligned}
$$

所以 $x&#95;n$ 最终不为零，而且不会任意靠近零。代回倒数的误差，得

$$
\left|
\frac1{x&#95;n}-\frac1x
\right|
\leq
\frac{2}{|x|^2}|x&#95;n-x|.
$$

右边是一个固定常数乘以趋于零的误差，因此

$$
\frac1{x&#95;n}\longrightarrow\frac1x.
$$

设 $x&#95;n\to x$、$y&#95;n\to y$ 且 $y\neq0$。商可以写成乘积

$$
\frac{x&#95;n}{y&#95;n}
=x&#95;n\frac1{y&#95;n}.
$$

结合乘积与倒数的结论，得

$$
\frac{x&#95;n}{y&#95;n}
\longrightarrow
\frac{x}{y}.
$$

完整的 $\varepsilon$-$N$ 证明见附录 E。

## 5. 次序怎样通过极限

假设 $x&#95;n\leq y&#95;n$ 最终成立，并且

$$
x&#95;n\longrightarrow x,
\qquad
y&#95;n\longrightarrow y.
$$

那么

$$
x\leq y.
$$

这是极限的**保序性**。它保留的是非严格不等式 $\leq$。严格不等式可以在极限时变成等式：对每个 $n\in\mathbb N$，

$$
-\frac1n<0,
$$

而两边的极限都是 $0$。

反过来，如果极限之间存在严格间隔 $x<y$，那么这道间隔会在数列尾部保留下来：最终有 $x&#95;n<y&#95;n$。证明时可以把间隔 $y-x$ 分成三份，取

$$
\varepsilon:=\frac{y-x}{3}.
$$

当 $x&#95;n$ 与 $x$ 的距离、$y&#95;n$ 与 $y$ 的距离都小于 $\varepsilon$ 时，

$$
x&#95;n
{}<x+\varepsilon
{}<y-\varepsilon
{}<y&#95;n.
$$

中间的严格不等式来自 $2\varepsilon<y-x$。选择三等分留出了一份富余，使两个可能摇动的数列仍然无法相遇。

### 5.1 夹逼定理

设三个实数列满足

$$
a&#95;n\leq x&#95;n\leq b&#95;n
$$

最终成立。如果

$$
a&#95;n\longrightarrow L,
\qquad
b&#95;n\longrightarrow L,
$$

那么

$$
x&#95;n\longrightarrow L.
$$

上界与下界同时向 $L$ 收紧，留给 $x&#95;n$ 的活动区间会被压缩到任意小。给定 $\varepsilon>0$，只需让

$$
L-\varepsilon<a&#95;n
\leq x&#95;n
\leq b&#95;n<L+\varepsilon.
$$

于是 $|x&#95;n-L|<\varepsilon$。附录 F 将保序性、严格间隔的保留与夹逼定理分别证明。

## 6. 绝对值、最大值与可传递的估计

反三角不等式给出

$$
\bigl||x&#95;n|-|x|\bigr|
\leq |x&#95;n-x|.
$$

所以 $x&#95;n\to x$ 时，

$$
|x&#95;n|\longrightarrow |x|.
$$

对两个实数 $u,v$，较大者可以写成

$$
\max\lbrace u,v\rbrace
=
\frac{u+v+|u-v|}{2}.
$$

因此若 $x&#95;n\to x$ 且 $y&#95;n\to y$，利用已经证明的加法、减法、常数倍与绝对值运算，可得

$$
\max\lbrace x&#95;n,y&#95;n\rbrace
\longrightarrow
\max\lbrace x,y\rbrace.
$$

这里展示了一种重要的证明组织方式：已经建立的极限运算可以作为引理被继续组合，每次只处理新出现的结构。附录 G 给出详细证明。

## 7. 从“趋于零”到“趋于零的速度”

极限

$$
a&#95;n\longrightarrow0
$$

只说明误差最终可以任意小，没有说它与 $1/n$、$1/n^2$ 或其他尺度相比有多快。渐近记号用来比较这些速度。

以下所有关系都在 $n\to\infty$ 时理解。

### 7.1 大 $O$：给出一个最终的尺度上界

设 $(a&#95;n)$ 与 $(b&#95;n)$ 是实数列。如果存在常数 $C>0$ 与正整数 $N$，使得对所有 $n\geq N$，

$$
|a&#95;n|\leq C|b&#95;n|,
$$

就记作

$$
a&#95;n=O(b&#95;n).
$$

$O$ 是大写字母 O，这个符号记录的是一种最终上界关系。它不要求 $a&#95;n/b&#95;n$ 存在极限，也不要求两个数列同号。例如

$$
\frac{(-1)^n}{n}=O\left(\frac1n\right).
$$

记号 $a&#95;n=O(1)$ 表示 $(a&#95;n)$ 最终有界。由于数列前面只有有限多项，它等价于整个数列有界。

### 7.2 小 $o$：相对于参考尺度可以忽略

如果对每个 $\varepsilon>0$，都存在 $N\in\mathbb N$，使得对所有 $n\geq N$，

$$
|a&#95;n|\leq\varepsilon |b&#95;n|,
$$

就记作

$$
a&#95;n=o(b&#95;n).
$$

小写字母 $o$ 表示：无论允许的比例 $\varepsilon$ 多小，$|a&#95;n|$ 最终都不超过 $\varepsilon|b&#95;n|$。如果 $b&#95;n$ 最终不为零，这个定义等价于

$$
\frac{a&#95;n}{b&#95;n}\longrightarrow0.
$$

特别地，$a&#95;n=o(1)$ 就是 $a&#95;n\to0$。对任意 $p>0$，

$$
\frac1{n^{p+1}}
=o\left(\frac1{n^p}\right),
$$

因为两者相除后等于 $1/n\to0$。

### 7.3 渐近等价：比值趋于一

如果 $b&#95;n$ 最终不为零，并且

$$
\frac{a&#95;n}{b&#95;n}\longrightarrow1,
$$

就记作

$$
a&#95;n\sim b&#95;n.
$$

符号 $\sim$ 读作“渐近等价于”。它说明 $a&#95;n$ 与 $b&#95;n$ 的相对误差趋于零：

$$
\frac{a&#95;n-b&#95;n}{b&#95;n}
\longrightarrow0.
$$

因此

$$
a&#95;n\sim b&#95;n
\quad\Longleftrightarrow\quad
a&#95;n=b&#95;n\bigl(1+o(1)\bigr).
$$

这里 $1+o(1)$ 表示形如 $1+r&#95;n$ 的数列，其中 $r&#95;n\to0$。例如

$$
n+1\sim n,
$$

因为

$$
\frac{n+1}{n}=1+\frac1n\longrightarrow1.
$$

## 8. 渐近记号的运算规则

大 $O$ 与小 $o$ 把误差预算压缩成了更简洁的语言。设

$$
a&#95;n=O(b&#95;n),
\qquad
c&#95;n=O(d&#95;n).
$$

则

$$
a&#95;nc&#95;n=O(b&#95;nd&#95;n).
$$

因为两个最终上界相乘，仍然是一个固定常数倍的上界。如果 $a&#95;n=O(b&#95;n)$ 且 $c&#95;n=O(b&#95;n)$，那么

$$
a&#95;n+c&#95;n=O(b&#95;n).
$$

对小 $o$，一个可忽略量乘以一个有控制的量，仍是可忽略量：

$$
a&#95;n=o(b&#95;n),
\quad
c&#95;n=O(d&#95;n)
\quad\Longrightarrow\quad
a&#95;nc&#95;n=o(b&#95;nd&#95;n).
$$

其中一个常用特例是

$$
o(1)O(1)=o(1).
$$

若 $u&#95;n=o(1)$，则 $1+u&#95;n\to1$，因此它最终远离零。于是

$$
\frac1{1+u&#95;n}
=1+o(1).
$$

这条规则还可以精确一阶。由恒等式

$$
\frac1{1+u}
=1-u+\frac{u^2}{1+u},
$$

当 $u&#95;n\to0$ 时，

$$
\frac1{1+u&#95;n}
=1-u&#95;n+O(u&#95;n^2).
$$

这个展开完全来自代数恒等式与分母最终远离零，尚不需要微分。所有这些运算规则都在附录 H 中回到定义证明。

## 9. 渐近等价何时能代换

渐近等价对乘法很稳定。如果

$$
a&#95;n\sim b&#95;n,
\qquad
c&#95;n\sim d&#95;n,
$$

并且相关分母最终不为零，那么

$$
a&#95;nc&#95;n\sim b&#95;nd&#95;n,
\qquad
\frac{a&#95;n}{c&#95;n}
\sim
\frac{b&#95;n}{d&#95;n}.
$$

原因是比值会分别趋于 $1$，它们的乘积或商仍趋于 $1$。

加减法需要检查是否发生了相消。虽然

$$
n+1\sim n,
$$

但在两边同时减去 $n$ 后，得到的是 $1$ 与 $0$，二者不渐近等价。原来的等价关系控制相对误差

$$
\frac{(n+1)-n}{n}=\frac1n,
$$

而减去主要项 $n$ 之后，原先很小的误差 $1$ 变成了全部结果。

一个对称的例子更能看出相消的作用。设

$$
a&#95;n=n+1,
\quad
b&#95;n=n,
\quad
c&#95;n=-n,
\quad
d&#95;n=-n.
$$

此时 $a&#95;n\sim b&#95;n$ 且 $c&#95;n\sim d&#95;n$，但

$$
a&#95;n+c&#95;n=1,
\qquad
b&#95;n+d&#95;n=0.
$$

因此，在加减法中使用渐近等价时，应当先找出组合后的主要尺度，然后确认被忽略的误差相对于这个新尺度仍然趋于零。

## 10. 两个完整的渐近计算

### 10.1 有理式：极限之外还有多少误差

考虑

$$
r&#95;n
:=
\frac{3n^2-2n+5}{2n^2+n}.
$$

分子分母同除以 $n^2$，并令 $t&#95;n:=1/n$，得

$$
r&#95;n
=
\frac{3-2t&#95;n+5t&#95;n^2}{2+t&#95;n}.
$$

符号 $:=$ 表示左边的量由右边定义。因为 $t&#95;n\to0$，极限运算给出

$$
r&#95;n\longrightarrow\frac32.
$$

若还想知道它以多快的速度趋于 $3/2$，将分母写成

$$
\frac1{2+t&#95;n}
=
\frac12
\frac1{1+t&#95;n/2}.
$$

使用上一节的恒等式，

$$
\frac1{2+t&#95;n}
=
\frac12-\frac{t&#95;n}{4}+O(t&#95;n^2).
$$

与分子相乘，并按 $t&#95;n$ 的次数收集：

$$
\begin{aligned}
r&#95;n
&=(3-2t&#95;n+5t&#95;n^2)
\left(
\frac12-\frac{t&#95;n}{4}+O(t&#95;n^2)
\right)\\\\
&=\frac32
-\left(\frac34+1\right)t&#95;n
+O(t&#95;n^2)\\\\
&=\frac32-\frac{7}{4n}
+O\left(\frac1{n^2}\right).
\end{aligned}
$$

极限 $3/2$ 只给出终点；这个渐近式还给出了主要误差 $-7/(4n)$，剩余误差的量级不超过 $1/n^2$。

### 10.2 根式：相消之后先有理化

考虑

$$
s&#95;n:=\sqrt{n^2+n}-n.
$$

两项都与 $n$ 同阶，直接用 $\sqrt{n^2+n}\sim n$ 无法判断它们的差。这正是上一节中的相消。与共轭式相乘，

$$
\begin{aligned}
s&#95;n
&=\frac{(n^2+n)-n^2}{\sqrt{n^2+n}+n}\\\\
&=\frac{n}{\sqrt{n^2+n}+n}\\\\
&=\frac1{\sqrt{1+1/n}+1}.
\end{aligned}
$$

最后一步的分子分母同除以 $n$。由于 $n\in\mathbb N$，所以 $n>0$，并且

$$
\frac{\sqrt{n^2+n}}{n}
=\sqrt{\frac{n^2+n}{n^2}}
=\sqrt{1+\frac1n}.
$$

此时分子、分母都有有限极限，所以

$$
s&#95;n\longrightarrow\frac12.
$$

还可以在不使用微分的情况下找到一阶误差。令 $t&#95;n:=1/n$，则

$$
\begin{aligned}
\frac1{\sqrt{1+t&#95;n}+1}-\frac12
&=
\frac{1-\sqrt{1+t&#95;n}}
{2(\sqrt{1+t&#95;n}+1)}\\\\
&=
-\frac{t&#95;n}
{2(\sqrt{1+t&#95;n}+1)^2}.
\end{aligned}
$$

分母趋于 $8$，因此

$$
s&#95;n
=\frac12-\frac1{8n}
+o\left(\frac1n\right).
$$

这个计算展示了相消问题的常见处理顺序：先用恒等变形暴露真正的小量，再进入极限或渐近估计。

## 11. 从极限运算到分析学语言

$\varepsilon$-$N$ 证明的表面形式是不等式，其内部结构是误差的分解、缩放与合并。加法把误差直接相加；乘法需要先插入中间项，再用有界性控制放大因子；除法需要先建立分母的正下界；渐近记号则把这些反复出现的估计压缩成可以组合的符号。

当这套语言建立后，后续的连续性、微分、积分与级数都可以把它作为基本工具。极限将从单个数列的尾部行为，逐步扩展到函数在一点附近的局部行为。

## 参考文献

1. Bernard Bolzano, *Rein analytischer Beweis des Lehrsatzes, dass zwischen je zwey Werthen, die ein entgegengesetztes Resultat gewähren, wenigstens eine reelle Wurzel der Gleichung liege*, 1817.
2. Augustin-Louis Cauchy, *Cours d'analyse de l'École Royale Polytechnique*, 1821.
3. G. H. Hardy, *Orders of Infinity: The Infinitärcalcül of Paul du Bois-Reymond*, Cambridge University Press, 1910.
4. Tom M. Apostol, *Mathematical Analysis*, 2nd ed., Addison-Wesley, 1974.
5. Walter Rudin, *Principles of Mathematical Analysis*, 3rd ed., McGraw-Hill, 1976.
6. N. G. de Bruijn, *Asymptotic Methods in Analysis*, Dover Publications, 1981.
7. Terence Tao, *Analysis I*, 3rd ed., Hindustan Book Agency, 2016.

---

## 附录

如下为正文附录补充。

## A. 极限运算的两个基本工具

### A.1 有限多个“最终”条件如何合并

设 $P&#95;1(n),\ldots,P&#95;k(n)$ 是 $k$ 个关于正整数 $n$ 的命题，其中 $k\in\mathbb N$。假设对每个 $j\in\lbrace1,\ldots,k\rbrace$，都存在 $N&#95;j\in\mathbb N$，使得

$$
n\geq N&#95;j
\quad\Longrightarrow\quad
P&#95;j(n).
$$

取

$$
N:=\max\lbrace N&#95;1,\ldots,N&#95;k\rbrace.
$$

任取 $n\geq N$。按最大值的定义，对每个 $j$都有 $N\geq N&#95;j$，从而

$$
n\geq N\geq N&#95;j.
$$

所以 $P&#95;j(n)$ 对每个 $j$ 同时成立。

### A.2 数列 $1/n$ 趋于零

给定 $\varepsilon>0$。由实数的 Archimedean 性质，存在 $N\in\mathbb N$，使得

$$
N>\frac1\varepsilon.
$$

当 $n\geq N$ 时，$n\geq N>1/\varepsilon$。因为这些数都是正数，取倒数会改变不等号方向，从而

$$
0<\frac1n
\leq\frac1N
{}<\varepsilon.
$$

因此 $|1/n-0|<\varepsilon$，即 $1/n\to0$。这里使用的 Archimedean 性质已在上一篇文章中从实数完备性推出。

## B. 和、差与常数倍的完整证明

### B.1 和的极限

设 $x&#95;n\to x$ 且 $y&#95;n\to y$。给定 $\varepsilon>0$。

由 $x&#95;n\to x$，存在 $N&#95;1\in\mathbb N$，使得

$$
n\geq N&#95;1
\quad\Longrightarrow\quad
|x&#95;n-x|<\frac\varepsilon2.
$$

由 $y&#95;n\to y$，存在 $N&#95;2\in\mathbb N$，使得

$$
n\geq N&#95;2
\quad\Longrightarrow\quad
|y&#95;n-y|<\frac\varepsilon2.
$$

取 $N:=\max\lbrace N&#95;1,N&#95;2\rbrace$。当 $n\geq N$ 时，两个不等式同时成立，并且

$$
\begin{aligned}
|(x&#95;n+y&#95;n)-(x+y)|
&=|(x&#95;n-x)+(y&#95;n-y)|\\\\
&\leq|x&#95;n-x|+|y&#95;n-y|\\\\
&<\frac\varepsilon2+\frac\varepsilon2\\\\
&=\varepsilon.
\end{aligned}
$$

这正是 $x&#95;n+y&#95;n\to x+y$ 的定义。

### B.2 差的极限

给定 $\varepsilon>0$。由两个数列的收敛性，可以选择 $N&#95;1,N&#95;2$，使得 $n\geq N&#95;1$ 时 $|x&#95;n-x|<\varepsilon/2$，$n\geq N&#95;2$ 时 $|y&#95;n-y|<\varepsilon/2$。

取 $N:=\max\lbrace N&#95;1,N&#95;2\rbrace$。当 $n\geq N$ 时，

$$
\begin{aligned}
|(x&#95;n-y&#95;n)-(x-y)|
&=|(x&#95;n-x)-(y&#95;n-y)|\\\\
&\leq |x&#95;n-x|+|y&#95;n-y|\\\\
&<\varepsilon.
\end{aligned}
$$

因此 $x&#95;n-y&#95;n\to x-y$。

### B.3 常数倍的极限

设 $c\in\mathbb R$。若 $c=0$，则 $cx&#95;n=cx=0$，所以结论成立。

现设 $c\neq0$。给定 $\varepsilon>0$。因为 $\varepsilon/|c|>0$，由 $x&#95;n\to x$，存在 $N\in\mathbb N$，使得 $n\geq N$ 时

$$
|x&#95;n-x|<\frac\varepsilon{|c|}.
$$

于是

$$
\begin{aligned}
|cx&#95;n-cx|
&=|c|\cdot|x&#95;n-x|\\\\
&<|c|\frac\varepsilon{|c|}\\\\
&=\varepsilon.
\end{aligned}
$$

因此 $cx&#95;n\to cx$。

## C. 收敛数列有界的完整证明

设 $x&#95;n\to x$。在极限定义中取 $\varepsilon=1$。存在 $N&#95;0\in\mathbb N$，使得当 $n\geq N&#95;0$ 时，

$$
|x&#95;n-x|<1.
$$

由三角不等式，

$$
\begin{aligned}
|x&#95;n|
&=|(x&#95;n-x)+x|\\\\
&\leq |x&#95;n-x|+|x|\\\\
&<1+|x|.
\end{aligned}
$$

这控制了从第 $N&#95;0$ 项开始的尾部。前面的

$$
|x&#95;1|,\ldots,|x&#95;{N&#95;0-1}|
$$

只有有限多个实数。取

$$
M:=\max\lbrace
|x&#95;1|,\ldots,|x&#95;{N&#95;0-1}|,1+|x|
\rbrace.
$$

则对每个 $n\in\mathbb N$都有 $|x&#95;n|\leq M$，所以 $(x&#95;n)$ 有界。若 $N&#95;0=1$，前面没有需要另行处理的项，直接取 $M:=1+|x|$ 即可。

## D. 乘积极限的完整证明

设 $x&#95;n\to x$ 且 $y&#95;n\to y$。附录 C 表明 $(x&#95;n)$ 有界，所以存在 $M>0$，使得

$$
|x&#95;n|\leq M
$$

对所有 $n\in\mathbb N$ 成立。

给定 $\varepsilon>0$。为了统一处理 $y=0$ 与 $y\neq0$，使用正数 $|y|+1$。由 $y&#95;n\to y$，存在 $N&#95;1\in\mathbb N$，使得

$$
n\geq N&#95;1
\quad\Longrightarrow\quad
|y&#95;n-y|<\frac{\varepsilon}{2M}.
$$

由 $x&#95;n\to x$，存在 $N&#95;2\in\mathbb N$，使得

$$
n\geq N&#95;2
\quad\Longrightarrow\quad
|x&#95;n-x|<\frac{\varepsilon}{2(|y|+1)}.
$$

取 $N:=\max\lbrace N&#95;1,N&#95;2\rbrace$。当 $n\geq N$ 时，

$$
\begin{aligned}
|x&#95;ny&#95;n-xy|
&=|x&#95;n(y&#95;n-y)+y(x&#95;n-x)|\\\\
&\leq |x&#95;n|\cdot|y&#95;n-y|
+|y|\cdot|x&#95;n-x|\\\\
&\leq M\frac{\varepsilon}{2M}
+|y|\frac{\varepsilon}{2(|y|+1)}\\\\
&<\frac\varepsilon2+\frac\varepsilon2\\\\
&=\varepsilon.
\end{aligned}
$$

因此 $x&#95;ny&#95;n\to xy$。

## E. 倒数与商的完整证明

### E.1 分母最终远离零

设 $x&#95;n\to x$ 且 $x\neq0$。因为 $|x|/2>0$，存在 $N&#95;0\in\mathbb N$，使得 $n\geq N&#95;0$ 时

$$
|x&#95;n-x|<\frac{|x|}{2}.
$$

反三角不等式 $|u|\geq |v|-|u-v|$ 给出

$$
\begin{aligned}
|x&#95;n|
&\geq |x|-|x&#95;n-x|\\\\
&>|x|-\frac{|x|}{2}\\\\
&=\frac{|x|}{2}.
\end{aligned}
$$

因此 $x&#95;n\neq0$ 对所有 $n\geq N&#95;0$ 成立。

### E.2 倒数的极限

给定 $\varepsilon>0$。由 $x&#95;n\to x$，存在 $N&#95;1\in\mathbb N$，使得 $n\geq N&#95;1$ 时

$$
|x&#95;n-x|<\frac{\varepsilon|x|^2}{2}.
$$

取 $N:=\max\lbrace N&#95;0,N&#95;1\rbrace$，其中 $N&#95;0$ 来自 E.1。当 $n\geq N$ 时，$x&#95;n\neq0$ 且 $|x&#95;n|>|x|/2$，从而

$$
\begin{aligned}
\left|
\frac1{x&#95;n}-\frac1x
\right|
&=\frac{|x&#95;n-x|}{|x&#95;n|\cdot|x|}\\\\
&<\frac{2}{|x|^2}|x&#95;n-x|\\\\
&<\frac{2}{|x|^2}
\frac{\varepsilon|x|^2}{2}\\\\
&=\varepsilon.
\end{aligned}
$$

因此 $1/x&#95;n\to1/x$。

### E.3 商的极限

设 $x&#95;n\to x$、$y&#95;n\to y$ 且 $y\neq0$。由 E.1，$y&#95;n$ 最终不为零，所以商 $x&#95;n/y&#95;n$ 在数列尾部有定义。由 E.2，

$$
\frac1{y&#95;n}\longrightarrow\frac1y.
$$

再用附录 D 的乘积极限，

$$
\frac{x&#95;n}{y&#95;n}
=x&#95;n\frac1{y&#95;n}
\longrightarrow
x\frac1y
=\frac xy.
$$

这就证明了商的极限。

## F. 保序性与夹逼定理

### F.1 极限的保序性

设 $x&#95;n\leq y&#95;n$ 最终成立，$x&#95;n\to x$，$y&#95;n\to y$。假设结论 $x\leq y$ 不成立，则 $x>y$。取

$$
\varepsilon:=\frac{x-y}{3}>0.
$$

由 $x&#95;n\to x$，存在 $N&#95;1$，使得 $n\geq N&#95;1$ 时

$$
x&#95;n>x-\varepsilon.
$$

由 $y&#95;n\to y$，存在 $N&#95;2$，使得 $n\geq N&#95;2$ 时

$$
y&#95;n<y+\varepsilon.
$$

再设 $x&#95;n\leq y&#95;n$ 对所有 $n\geq N&#95;3$ 成立。取

$$
N:=\max\lbrace N&#95;1,N&#95;2,N&#95;3\rbrace.
$$

当 $n\geq N$ 时，因为 $2\varepsilon< x-y$，

$$
x&#95;n
{}>x-\varepsilon
{}>y+\varepsilon
{}>y&#95;n.
$$

这与 $x&#95;n\leq y&#95;n$ 矛盾。因此 $x\leq y$。

### F.2 严格间隔最终保留

设 $x&#95;n\to x$、$y&#95;n\to y$ 且 $x<y$。取

$$
\varepsilon:=\frac{y-x}{3}>0.
$$

存在 $N&#95;1,N&#95;2$，使得 $n\geq N&#95;1$ 时 $|x&#95;n-x|<\varepsilon$，$n\geq N&#95;2$ 时 $|y&#95;n-y|<\varepsilon$。取 $N:=\max\lbrace N&#95;1,N&#95;2\rbrace$。当 $n\geq N$ 时，

$$
x&#95;n
{}<x+\varepsilon
{}<y-\varepsilon
{}<y&#95;n.
$$

所以 $x&#95;n<y&#95;n$ 最终成立。

### F.3 夹逼定理

设 $a&#95;n\leq x&#95;n\leq b&#95;n$ 对所有 $n\geq N&#95;0$ 成立，并且 $a&#95;n\to L$、$b&#95;n\to L$。给定 $\varepsilon>0$。

存在 $N&#95;1$，使得 $n\geq N&#95;1$ 时

$$
|a&#95;n-L|<\varepsilon,
$$

所以 $a&#95;n>L-\varepsilon$。同理，存在 $N&#95;2$，使得 $n\geq N&#95;2$ 时 $b&#95;n<L+\varepsilon$。

取

$$
N:=\max\lbrace N&#95;0,N&#95;1,N&#95;2\rbrace.
$$

当 $n\geq N$ 时，

$$
L-\varepsilon
{}<a&#95;n
\leq x&#95;n
\leq b&#95;n
{}<L+\varepsilon.
$$

因此 $|x&#95;n-L|<\varepsilon$，即 $x&#95;n\to L$。

## G. 绝对值与最大值的极限

### G.1 绝对值

对任意实数 $u,v$，反三角不等式给出

$$
\bigl||u|-|v|\bigr|\leq|u-v|.
$$

设 $x&#95;n\to x$。给定 $\varepsilon>0$，存在 $N$，使得 $n\geq N$ 时 $|x&#95;n-x|<\varepsilon$。于是

$$
\bigl||x&#95;n|-|x|\bigr|
\leq |x&#95;n-x|
{}<\varepsilon.
$$

因此 $|x&#95;n|\to|x|$。

### G.2 最大值公式

先验证恒等式

$$
\max\lbrace u,v\rbrace
=\frac{u+v+|u-v|}{2}.
$$

若 $u\geq v$，则 $|u-v|=u-v$，右边等于 $u$，正是 $\max\lbrace u,v\rbrace$。若 $u<v$，则 $|u-v|=v-u$，右边等于 $v$，结论同样成立。

设 $x&#95;n\to x$、$y&#95;n\to y$。由和、差与绝对值的极限，

$$
\begin{aligned}
\max\lbrace x&#95;n,y&#95;n\rbrace
&=\frac{x&#95;n+y&#95;n+|x&#95;n-y&#95;n|}{2}\\\\
&\longrightarrow
\frac{x+y+|x-y|}{2}\\\\
&=\max\lbrace x,y\rbrace.
\end{aligned}
$$

## H. 渐近运算的完整证明

### H.1 大 $O$ 的加法与乘法

设 $a&#95;n=O(b&#95;n)$ 且 $c&#95;n=O(b&#95;n)$。按定义，存在 $C&#95;1,C&#95;2>0$ 与 $N&#95;1,N&#95;2\in\mathbb N$，使得

$$
n\geq N&#95;1
\quad\Longrightarrow\quad
|a&#95;n|\leq C&#95;1|b&#95;n|,
$$

$$
n\geq N&#95;2
\quad\Longrightarrow\quad
|c&#95;n|\leq C&#95;2|b&#95;n|.
$$

取 $N:=\max\lbrace N&#95;1,N&#95;2\rbrace$。当 $n\geq N$ 时，

$$
\begin{aligned}
|a&#95;n+c&#95;n|
&\leq |a&#95;n|+|c&#95;n|\\\\
&\leq(C&#95;1+C&#95;2)|b&#95;n|.
\end{aligned}
$$

因为 $C&#95;1+C&#95;2$ 是固定正常数，所以 $a&#95;n+c&#95;n=O(b&#95;n)$。

再设 $a&#95;n=O(b&#95;n)$ 且 $c&#95;n=O(d&#95;n)$。用同样的共同起点 $N$，当 $n\geq N$ 时，

$$
\begin{aligned}
|a&#95;nc&#95;n|
&=|a&#95;n|\cdot|c&#95;n|\\\\
&\leq C&#95;1C&#95;2|b&#95;nd&#95;n|.
\end{aligned}
$$

所以 $a&#95;nc&#95;n=O(b&#95;nd&#95;n)$。

### H.2 小 $o$ 与大 $O$ 的乘法

设 $a&#95;n=o(b&#95;n)$ 且 $c&#95;n=O(d&#95;n)$。按大 $O$ 的定义，存在 $C>0$ 和 $N&#95;0$，使得 $n\geq N&#95;0$ 时

$$
|c&#95;n|\leq C|d&#95;n|.
$$

要证明 $a&#95;nc&#95;n=o(b&#95;nd&#95;n)$，给定任意 $\varepsilon>0$。由 $a&#95;n=o(b&#95;n)$，对正数 $\varepsilon/C$，存在 $N&#95;1$，使得 $n\geq N&#95;1$ 时

$$
|a&#95;n|\leq\frac\varepsilon C|b&#95;n|.
$$

取 $N:=\max\lbrace N&#95;0,N&#95;1\rbrace$。当 $n\geq N$ 时，

$$
\begin{aligned}
|a&#95;nc&#95;n|
&\leq
\frac\varepsilon C|b&#95;n|\cdot C|d&#95;n|\\\\
&=\varepsilon|b&#95;nd&#95;n|.
\end{aligned}
$$

这正是 $a&#95;nc&#95;n=o(b&#95;nd&#95;n)$ 的定义。

### H.3 $1+o(1)$ 的倒数

设 $u&#95;n=o(1)$，即 $u&#95;n\to0$。因此 $1+u&#95;n\to1$。由附录 E.1，$1+u&#95;n$ 最终不为零，并且

$$
|1+u&#95;n|>\frac12
$$

最终成立。使用恒等式

$$
\frac1{1+u&#95;n}-1
=-\frac{u&#95;n}{1+u&#95;n},
$$

得

$$
\left|
\frac1{1+u&#95;n}-1
\right|
\leq2|u&#95;n|
$$

最终成立。右边趋于零，所以

$$
\frac1{1+u&#95;n}=1+o(1).
$$

再使用代数恒等式

$$
\frac1{1+u&#95;n}
-(1-u&#95;n)
=\frac{u&#95;n^2}{1+u&#95;n}.
$$

因为 $1/|1+u&#95;n|\leq2$ 最终成立，

$$
\left|
\frac{u&#95;n^2}{1+u&#95;n}
\right|
\leq2|u&#95;n|^2.
$$

按大 $O$ 的定义，这就是

$$
\frac1{1+u&#95;n}
=1-u&#95;n+O(u&#95;n^2).
$$

### H.4 渐近等价的乘法与商

设 $a&#95;n\sim b&#95;n$ 且 $c&#95;n\sim d&#95;n$。按定义，

$$
\frac{a&#95;n}{b&#95;n}\longrightarrow1,
\qquad
\frac{c&#95;n}{d&#95;n}\longrightarrow1.
$$

由乘积极限，

$$
\frac{a&#95;nc&#95;n}{b&#95;nd&#95;n}
=
\frac{a&#95;n}{b&#95;n}
\frac{c&#95;n}{d&#95;n}
\longrightarrow1.
$$

所以 $a&#95;nc&#95;n\sim b&#95;nd&#95;n$。

在 $c&#95;n,d&#95;n$ 以及所需分母最终不为零时，

$$
\frac{a&#95;n/c&#95;n}{b&#95;n/d&#95;n}
=
\frac{a&#95;n}{b&#95;n}
\frac{d&#95;n}{c&#95;n}.
$$

第一个因子趋于 $1$。由倒数极限，

$$
\frac{d&#95;n}{c&#95;n}
=
\frac1{c&#95;n/d&#95;n}
\longrightarrow1.
$$

因此

$$
\frac{a&#95;n}{c&#95;n}
\sim
\frac{b&#95;n}{d&#95;n}.
$$

### H.5 根式例子中的一阶估计

令 $t&#95;n:=1/n$，并设

$$
q&#95;n
:=
\frac1{\sqrt{1+t&#95;n}+1}-\frac12.
$$

正文已经通过有理化得到

$$
q&#95;n
{}=-
\frac{t&#95;n}
{2(\sqrt{1+t&#95;n}+1)^2}.
$$

现在把主要项 $-t&#95;n/8$ 分离出来。计算

$$
\begin{aligned}
\frac{q&#95;n-(-t&#95;n/8)}{t&#95;n}
&=
-\frac1{2(\sqrt{1+t&#95;n}+1)^2}
+\frac18.
\end{aligned}
$$

先验证这里需要的根式极限。因为 $t&#95;n=1/n>0$，

$$
\begin{aligned}
\left|\sqrt{1+t&#95;n}-1\right|
&=\frac{t&#95;n}{\sqrt{1+t&#95;n}+1}\\\\
&\leq t&#95;n.
\end{aligned}
$$

由 A.2，$t&#95;n\to0$。夹逼定理因此给出

$$
\sqrt{1+t&#95;n}\longrightarrow1.
$$

再使用极限的和、乘积与商的运算规则，上面的比值趋于

$$
-\frac1{2(1+1)^2}+\frac18=0.
$$

因此

$$
q&#95;n+\frac{t&#95;n}{8}
=o(t&#95;n),
$$

也就是

$$
\frac1{\sqrt{1+t&#95;n}+1}
=\frac12-\frac{t&#95;n}{8}+o(t&#95;n).
$$

代入 $t&#95;n=1/n$，得到正文中的展开式。

### H.6 小 $o$ 与比值趋零的等价性

设 $b&#95;n$ 最终不为零。先假设 $a&#95;n=o(b&#95;n)$。给定任意 $\varepsilon>0$，按小 $o$ 的定义，存在 $N$，使得当 $n\geq N$ 时

$$
|a&#95;n|\leq\varepsilon|b&#95;n|.
$$

由 $b&#95;n\neq0$，可以除以 $|b&#95;n|$，得到

$$
\left|\frac{a&#95;n}{b&#95;n}\right|\leq\varepsilon.
$$

这正是 $a&#95;n/b&#95;n\to0$ 的 $\varepsilon$-$N$ 形式。

反过，假设 $a&#95;n/b&#95;n\to0$。给定 $\varepsilon>0$，存在 $N$，使得当 $n\geq N$ 时

$$
\left|\frac{a&#95;n}{b&#95;n}\right|<\varepsilon.
$$

乘以正数 $|b&#95;n|$，就有

$$
|a&#95;n|<\varepsilon|b&#95;n|.
$$

这是 $a&#95;n=o(b&#95;n)$ 的定义。因此，在分母最终不为零时，

$$
a&#95;n=o(b&#95;n)
\quad\Longleftrightarrow\quad
\frac{a&#95;n}{b&#95;n}\longrightarrow0.
$$

取 $b&#95;n=1$，就得到 $a&#95;n=o(1)$ 与 $a&#95;n\to0$ 的等价性。

### H.7 渐近等价与 $1+o(1)$

假设 $b&#95;n$ 最终不为零。如果 $a&#95;n\sim b&#95;n$，则按定义

$$
\frac{a&#95;n}{b&#95;n}\longrightarrow1.
$$

令

$$
r&#95;n:=\frac{a&#95;n}{b&#95;n}-1.
$$

则 $r&#95;n\to0$，且逐项重新整理得

$$
a&#95;n=b&#95;n(1+r&#95;n).
$$

根据 H.6，$r&#95;n=o(1)$，所以 $a&#95;n=b&#95;n(1+o(1))$。

反过，如果存在 $r&#95;n=o(1)$ 使得

$$
a&#95;n=b&#95;n(1+r&#95;n),
$$

则除以 $b&#95;n$，得

$$
\frac{a&#95;n}{b&#95;n}=1+r&#95;n\longrightarrow1.
$$

这就是 $a&#95;n\sim b&#95;n$的定义。因此

$$
a&#95;n\sim b&#95;n
\quad\Longleftrightarrow\quad
a&#95;n=b&#95;n\bigl(1+o(1)\bigr).
$$
