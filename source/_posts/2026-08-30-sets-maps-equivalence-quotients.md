---
title: "从相等到等价：集合、映射与商结构"
date: 2026-08-30 21:10:00
categories:
  - 理论数学
tags:
  - 数学基础
  - 集合
  - 映射
  - 等价关系
  - 商集
  - 泛性质
mathjax: true
toc: true
toc_number: false
comments: true
---

现代数学中有一个反复出现的动作：先规定哪些对象应当被视为相同，再把每一批彼此等价的对象压缩成一个新对象。整数的同余类、群的陪集、环路的同伦类以及分析学中的函数商空间，都使用这套做法。

这项构造从集合开始。映射记录元素怎样被送往另一个集合；等价关系把元素分成互不相交的类；商集再把每一个等价类看成一个新的点。它们共同导向商集的泛性质：只要一个映射在每个等价类上取值相同，它就会唯一地下降为商集上的映射。抽象代数、拓扑学和分析学会沿着不同方向反复使用这个结论。

<!--more-->

## 1. 集合：先确定我们在谈论哪些对象

集合是若干对象组成的整体。若对象 $x$ 属于集合 $A$，写作

$$
x\in A;
$$

若不属于，则写作 $x\notin A$。集合中的对象称为它的**元素**。

集合由其元素完全确定。两个集合 $A$ 与 $B$ 相等，记作 $A=B$，意思是它们拥有完全相同的元素：

$$
A=B
\quad\Longleftrightarrow\quad
(\forall x)\bigl(x\in A\Longleftrightarrow x\in B\bigr).
$$

符号 $\forall$ 表示“对每一个”，$\Longleftrightarrow$ 表示左右两个命题互相推出。这是以后证明两个集合相等时使用的标准。

### 1.1 子集与真子集

如果 $A$ 的每个元素也属于 $B$，称 $A$ 是 $B$ 的子集，写作

$$
A\subseteq B.
$$

这一定义允许 $A=B$。若 $A\subseteq B$ 且 $A\neq B$，则 $A$ 是 $B$ 的真子集，可以写作 $A\subsetneq B$。

集合相等可以改写为两次包含：

$$
A=B
\quad\Longleftrightarrow\quad
\begin{cases}
A\subseteq B,\\\\
B\subseteq A.
\end{cases}
$$

这称为证明集合相等的**双包含法**。例如，为了证明 $A\cap B=B\cap A$，只需分别证明任取 $x\in A\cap B$ 都有 $x\in B\cap A$，以及反方向的包含关系。

不含任何元素的集合称为空集，记作 $\varnothing$。对任意集合 $A$，都有

$$
\varnothing\subseteq A.
$$

这里没有任何需要检查的元素：不存在一个 $x\in\varnothing$ 会违反子集定义。

只含一个元素 $a$ 的集合写作 $\lbrace a\rbrace$，称为单点集。要区分元素 $a$ 与单点集 $\lbrace a\rbrace$：前者是一个对象，后者是以该对象为唯一元素的集合。

### 1.2 并、交与差

给定集合 $A$ 与 $B$，它们的并集定义为

$$
A\cup B
:=
\lbrace x:(x\in A)\lor(x\in B)\rbrace,
$$

交集定义为

$$
A\cap B
:=
\lbrace x:(x\in A)\land(x\in B)\rbrace.
$$

符号 $\lor$ 表示“或”，$\land$ 表示“且”。这里的“或”包含两者同时成立的情形。若 $A\cap B=\varnothing$，称 $A$ 与 $B$ 不相交。

$A$ 去掉属于 $B$ 的元素后得到差集

$$
A\setminus B
:=
\lbrace x:(x\in A)\land(x\notin B)\rbrace.
$$

如果事先固定一个全集 $U$，并且 $A\subseteq U$，则 $A$ 相对于 $U$ 的补集为

$$
A^{\mathsf c}:=U\setminus A.
$$

补集依赖于全集。若不说明 $U$，表达式 $A^{\mathsf c}$ 本身不能确定一个集合。

并、交与补集满足 De Morgan 律：

$$
(A\cup B)^{\mathsf c}=A^{\mathsf c}\cap B^{\mathsf c},
\qquad
(A\cap B)^{\mathsf c}=A^{\mathsf c}\cup B^{\mathsf c}.
$$

以第一式为例，任取 $x\in U$，则

$$
\begin{aligned}
x\in(A\cup B)^{\mathsf c}
&\Longleftrightarrow x\notin A\cup B\\\\
&\Longleftrightarrow (x\notin A)\land(x\notin B)\\\\
&\Longleftrightarrow (x\in A^{\mathsf c})\land(x\in B^{\mathsf c})\\\\
&\Longleftrightarrow x\in A^{\mathsf c}\cap B^{\mathsf c}.
\end{aligned}
$$

左右两边包含完全相同的元素，所以两个集合相等。第二式的证明放在附录 A。

### 1.3 幂集与笛卡尔积

集合 $A$ 的所有子集组成一个新集合，称为 $A$ 的幂集：

$$
\mathcal P(A):=\lbrace B:B\subseteq A\rbrace.
$$

例如，当 $A=\lbrace 0,1\rbrace$ 时，

$$
\mathcal P(A)
=
\lbrace
\varnothing,
\lbrace0\rbrace,
\lbrace1\rbrace,
\lbrace0,1\rbrace
\rbrace.
$$

注意 $\varnothing\in\mathcal P(A)$，同时 $\varnothing\subseteq\mathcal P(A)$。前一句说空集是幂集的一个元素，后一句说空集是任何集合的子集，两句话含义不同。

集合 $A$ 与 $B$ 的笛卡尔积定义为

$$
A\times B
:=
\lbrace(a,b):a\in A,\ b\in B\rbrace.
$$

元素 $(a,b)$ 是有序对，因此通常 $(a,b)\neq(b,a)$。例如，平面可以写成

$$
\mathbb R^2=\mathbb R\times\mathbb R,
$$

其中 $\mathbb R$ 表示实数集。后面定义二元关系时，关系本身就是笛卡尔积的一个子集。

## 2. 映射：记录对象如何被送往另一个集合

设 $X$ 与 $Y$ 是集合。一个映射

$$
f:X\longrightarrow Y
$$

为每个 $x\in X$ 指定唯一的元素 $f(x)\in Y$。集合 $X$ 称为定义域，$Y$ 称为陪域，$f(x)$ 称为 $x$ 在 $f$ 下的像。

映射不仅由公式决定，还包括定义域和陪域。例如，公式 $x\mapsto x^2$ 可以给出两个不同的映射：

$$
f:\mathbb R\to\mathbb R,
\qquad
g:\mathbb R\to[0,\infty),
$$

虽然二者都满足 $f(x)=g(x)=x^2$，它们的陪域不同，所以不是同一个映射。这个差别会直接影响“是否满射”的判断。

两个映射 $f:X\to Y$ 与 $g:X\to Y$ 相等，是指

$$
(\forall x\in X)\quad f(x)=g(x).
$$

定义域或陪域不同的映射，不能直接按这个标准称为同一个映射。

### 2.1 像与原像

若 $A\subseteq X$，$A$ 在 $f$ 下的像定义为

$$
f(A):=\lbrace f(a):a\in A\rbrace\subseteq Y.
$$

整个定义域的像

$$
f(X)=\lbrace f(x):x\in X\rbrace
$$

也记作 $\operatorname{im}f$，读作 $f$ 的像集。

若 $B\subseteq Y$，$B$ 在 $f$ 下的原像定义为

$$
f^{-1}(B)
:=
\lbrace x\in X:f(x)\in B\rbrace.
$$

这里的 $f^{-1}(B)$ 对任何映射都有定义，不要求 $f$ 存在逆映射。原像运算严格保持并、交与补：

$$
\begin{aligned}
f^{-1}(B\cup C)&=f^{-1}(B)\cup f^{-1}(C),\\\\
f^{-1}(B\cap C)&=f^{-1}(B)\cap f^{-1}(C),\\\\
f^{-1}(Y\setminus B)&=X\setminus f^{-1}(B).
\end{aligned}
$$

第一式可以逐点验证。任取 $x\in X$，

$$
\begin{aligned}
x\in f^{-1}(B\cup C)
&\Longleftrightarrow f(x)\in B\cup C\\\\
&\Longleftrightarrow (f(x)\in B)\lor(f(x)\in C)\\\\
&\Longleftrightarrow (x\in f^{-1}(B))\lor(x\in f^{-1}(C))\\\\
&\Longleftrightarrow x\in f^{-1}(B)\cup f^{-1}(C).
\end{aligned}
$$

其余两式在附录 B 中证明。

像对交集却未必保持等号。总有

$$
f(A\cap B)\subseteq f(A)\cap f(B),
$$

但反方向可能失败。取 $f:\mathbb R\to\mathbb R$，$f(x)=x^2$，以及

$$
A=\lbrace-1\rbrace,
\qquad
B=\lbrace1\rbrace.
$$

此时 $A\cap B=\varnothing$，所以 $f(A\cap B)=\varnothing$；然而

$$
f(A)\cap f(B)=\lbrace1\rbrace.
$$

失败的原因是两个不同元素 $-1$ 与 $1$ 被送到了同一个值。等价关系稍后正是用来系统记录这种“映射无法区分”的现象。

### 2.2 单射、满射与双射

映射 $f:X\to Y$ 称为**单射**，如果

$$
f(x&#95;1)=f(x&#95;2)
\quad\Longrightarrow\quad
x&#95;1=x&#95;2.
$$

换言之，不同输入不会得到同一个输出。

$f$ 称为**满射**，如果

$$
(\forall y\in Y)(\exists x\in X)\quad f(x)=y.
$$

符号 $\exists$ 表示“至少存在一个”。上式等价于 $f(X)=Y$。同时为单射和满射的映射称为**双射**。

映射 $x\mapsto x^2$ 从 $\mathbb R$ 到 $\mathbb R$ 既不是单射，也不是满射；从 $[0,\infty)$ 到 $[0,\infty)$ 则是双射。公式没有改变，定义域和陪域改变了结论。

### 2.3 复合与恒等映射

给定映射

$$
f:X\to Y,
\qquad
g:Y\to Z,
$$

它们的复合映射

$$
g\circ f:X\to Z
$$

定义为

$$
(g\circ f)(x):=g(f(x)).
$$

符号的阅读顺序从右到左：先应用 $f$，再应用 $g$。

每个集合 $X$ 都有恒等映射

$$
\operatorname{id}&#95;X:X\to X,
\qquad
\operatorname{id}&#95;X(x)=x.
$$

它满足

$$
f\circ\operatorname{id}&#95;X=f,
\qquad
\operatorname{id}&#95;Y\circ f=f.
$$

映射复合满足结合律。若还有 $h:Z\to W$，则

$$
h\circ(g\circ f)=(h\circ g)\circ f.
$$

证明只需在任意 $x\in X$ 上比较两边：

$$
\begin{aligned}
\bigl(h\circ(g\circ f)\bigr)(x)
&=h\bigl((g\circ f)(x)\bigr)\\\\
&=h(g(f(x)))\\\\
&=(h\circ g)(f(x))\\\\
&=\bigl((h\circ g)\circ f\bigr)(x).
\end{aligned}
$$

### 2.4 逆映射何时存在

若存在映射 $g:Y\to X$ 满足

$$
g\circ f=\operatorname{id}&#95;X,
\qquad
f\circ g=\operatorname{id}&#95;Y,
$$

则称 $g$ 是 $f$ 的逆映射，记作 $f^{-1}$。这个记号和集合原像使用同样的符号，但对象不同：$f^{-1}:Y\to X$ 是一个映射，而 $f^{-1}(B)$ 是 $X$ 的一个子集。

**定理 2.1**　映射 $f:X\to Y$ 存在逆映射，当且仅当 $f$ 是双射。

**证明。** 先设 $f$ 存在逆映射 $g$。若 $f(x&#95;1)=f(x&#95;2)$，在两边应用 $g$，得到

$$
x&#95;1
=g(f(x&#95;1))
=g(f(x&#95;2))
=x&#95;2,
$$

所以 $f$ 是单射。对任意 $y\in Y$，取 $x=g(y)$，则

$$
f(x)=f(g(y))=y,
$$

所以 $f$ 是满射，因而是双射。

反过来设 $f$ 是双射。对每个 $y\in Y$，满射性保证至少存在一个 $x\in X$ 使 $f(x)=y$，单射性保证这样的 $x$ 至多一个。把这个唯一的 $x$ 记为 $x&#95;y$，定义

$$
g(y):=x&#95;y,
\qquad
f(x&#95;y)=y.
$$

由定义，$g(f(x))=x$ 且 $f(g(y))=y$，所以

$$
g\circ f=\operatorname{id}&#95;X,
\qquad
f\circ g=\operatorname{id}&#95;Y.
$$

故 $g=f^{-1}$。证毕。

## 3. 关系：把元素之间的联系写成集合

集合 $X$ 上的一个二元关系 $R$ 是笛卡尔积 $X\times X$ 的一个子集：

$$
R\subseteq X\times X.
$$

如果 $(x,y)\in R$，通常写成 $xRy$。关系不要求每个 $x$ 对应唯一的 $y$，因此它比映射宽泛得多。

例如，实数上的“小于等于”关系满足

$$
x\leq y
\quad\Longleftrightarrow\quad
(x,y)\in R&#95;{\leq}.
$$

集合之间的包含关系、整数之间的整除关系、平面上两点之间的距离关系，也都可以按这种方式写成笛卡尔积的子集。

### 3.1 等价关系的三条条件

集合 $X$ 上的关系 $\sim$ 称为等价关系，如果它满足：

1. **自反性**：对每个 $x\in X$，都有 $x\sim x$；
2. **对称性**：若 $x\sim y$，则 $y\sim x$；
3. **传递性**：若 $x\sim y$ 且 $y\sim z$，则 $x\sim z$。

三条条件分别保证每个元素和自己同类、同类判断不依赖书写顺序，以及经过中间元素不会产生矛盾。

“相等”是最细的等价关系，因为只有同一个元素彼此等价。另一个极端是规定任意两个元素都等价，它是最粗的等价关系，因为整个集合只剩一个类。

距离很近通常不是等价关系。假设在实数上定义

$$
x\sim y
\quad\Longleftrightarrow\quad
|x-y|<1.
$$

虽然它满足自反性和对称性，但 $0\sim0.75$ 且 $0.75\sim1.5$，而 $0\not\sim1.5$，所以传递性失败。若直接把“彼此接近”当作同一个对象，就可能得到不一致的分组。

### 3.2 模 $n$ 同余

固定正整数 $n$。对整数 $a,b\in\mathbb Z$，定义

$$
a\equiv b\pmod n
\quad\Longleftrightarrow\quad
n\mid(a-b).
$$

符号 $n\mid(a-b)$ 表示存在整数 $k$ 使 $a-b=kn$。这是一个等价关系。

自反性来自 $a-a=0=0\cdot n$。若 $a-b=kn$，则

$$
b-a=(-k)n,
$$

所以对称性成立。若 $a-b=kn$ 且 $b-c=\ell n$，其中 $k,\ell\in\mathbb Z$，则

$$
a-c=(a-b)+(b-c)=(k+\ell)n,
$$

所以传递性成立。

当 $n=3$ 时，所有整数被分为三类：除以 $3$ 余数为 $0$、$1$ 或 $2$ 的整数。类中有无穷多个整数，但商集只有三个元素。

## 4. 等价类与划分

设 $\sim$ 是 $X$ 上的等价关系。元素 $x\in X$ 的等价类定义为

$$
[x]
:=
\lbrace y\in X:y\sim x\rbrace.
$$

方括号 $[x]$ 表示一个集合，$x$ 是其中的一个元素。元素 $x$ 称为这个等价类的一个代表元；同一个等价类通常有许多代表元。

等价类有一个决定性的性质：两个等价类要么完全相同，要么不相交。

**命题 4.1**　对任意 $x,y\in X$，下列三个条件等价：

$$
x\sim y,
\qquad
[x]=[y],
\qquad
[x]\cap[y]\neq\varnothing.
$$

**证明。**

先设 $x\sim y$。任取 $z\in[x]$，则 $z\sim x$。由 $x\sim y$ 和传递性，得到 $z\sim y$，所以 $z\in[y]$。因此 $[x]\subseteq[y]$。由对称性，$y\sim x$，同理得到 $[y]\subseteq[x]$，故 $[x]=[y]$。

若 $[x]=[y]$，由自反性知 $x\in[x]$，所以 $x\in[x]\cap[y]$，交集非空。

最后设存在 $z\in[x]\cap[y]$。于是 $z\sim x$ 且 $z\sim y$。由对称性，$x\sim z$；再与 $z\sim y$ 使用传递性，得到 $x\sim y$。三个条件因此互相等价。证毕。

一个集合 $X$ 的**划分**是一族非空子集 $\lbrace P&#95;i\rbrace&#95;{i\in I}$，满足

$$
X=\bigcup&#95;{i\in I}P&#95;i,
$$

并且当 $i\neq j$ 时，$P&#95;i\cap P&#95;j=\varnothing$。这里 $I$ 是给各个子集编号的指标集。划分的意思是：每个 $x\in X$ 恰好落在其中一块。

**定理 4.2**　集合 $X$ 上的等价关系与 $X$ 的划分一一对应。

**证明。** 给定等价关系 $\sim$，自反性说明 $x\in[x]$，所以所有等价类覆盖 $X$；命题 4.1 说明不同等价类不相交。因此互不相同的等价类构成一个划分。

反过来，给定划分 $\lbrace P&#95;i\rbrace&#95;{i\in I}$，定义

$$
x\sim y
\quad\Longleftrightarrow\quad
(\exists i\in I)\quad (x\in P&#95;i)\land(y\in P&#95;i).
$$

每个元素都与自己位于同一块，所以关系自反；“位于同一块”与顺序无关，所以关系对称；若 $x,y$ 位于同一块且 $y,z$ 位于同一块，由于划分中的不同块不相交，包含 $y$ 的两块必须相同，所以 $x,z$ 位于同一块，关系传递。

两个构造互为逆过程：从等价关系得到的块正是其等价类；从划分得到关系后，每个元素的等价类正是它所在的那一块。证毕。

这个定理给出了等价关系的准确含义：它在集合上确定一个无冲突的划分规则。

## 5. 商集：把每个等价类当作一个新元素

由等价关系 $\sim$ 得到的所有等价类组成商集

$$
X/\!\sim
:=
\lbrace[x]:x\in X\rbrace.
$$

斜杠表示我们按照关系 $\sim$ 对 $X$ 取商。商集以等价类作为元素。

自然映射

$$
q:X\longrightarrow X/\!\sim,
\qquad
q(x)=[x]
$$

称为商映射或典范投影。它一定是满射，因为商集中的每个元素按定义都是某个 $[x]$。并且

$$
q(x)=q(y)
\quad\Longleftrightarrow\quad
[x]=[y]
\quad\Longleftrightarrow\quad
x\sim y.
$$

因此，$q$ 恰好忘掉了关系 $\sim$ 要求我们忽略的差别。

### 5.1 在等价类上定义映射

假设想用公式

$$
F([x])=f(x)
$$

定义一个从 $X/\!\sim$ 出发的映射。这里必须检查**良定义性**：同一个等价类可以写成 $[x]$，也可以写成 $[y]$；若 $[x]=[y]$，公式右侧必须给出相同结果。

所以必要条件是

$$
x\sim y
\quad\Longrightarrow\quad
f(x)=f(y).
$$

这句话称为“$f$ 在每个等价类上为常值”。检查这一条件，就是检查公式 $F([x])=f(x)$ 的良定义性。

例如，在模 $3$ 同余类上尝试定义 $F([a])=a$ 不可行，因为 $[1]=[4]$，却有 $1\neq4$。定义

$$
F([a])=a^2\bmod3
$$

则可以成立，因为同余的整数平方后仍然同余。附录 D 会完整验证这一点。

### 5.2 商集的泛性质

商集最重要的性质可以精确写成下面的定理。

**定理 5.1（商集的泛性质）**　设 $q:X\to X/\!\sim$ 是商映射，$f:X\to Y$ 是一个映射。以下两件事等价：

1. 若 $x\sim y$，则 $f(x)=f(y)$；
2. 存在唯一映射 $\overline f:X/\!\sim\to Y$，使得

$$
f=\overline f\circ q.
$$

符号 $\overline f$ 读作“$f$ bar”，表示由 $f$ 在商集上诱导的映射。

![商映射的分解](/images/notes/assets/mathematical-foundations/quotient-factorization.svg)

**证明。** 先假设条件 1 成立。对任意等价类 $[x]\in X/\!\sim$，定义

$$
\overline f([x]):=f(x).
$$

首先检查良定义性。若 $[x]=[y]$，命题 4.1 给出 $x\sim y$，再由条件 1 得到 $f(x)=f(y)$。因此用 $x$ 或 $y$ 作为代表元，$\overline f$ 的值相同。

接着验证分解等式。对每个 $x\in X$，

$$
(\overline f\circ q)(x)
=\overline f(q(x))
=\overline f([x])
=f(x).
$$

所以 $f=\overline f\circ q$。

还要证明唯一性。假设 $g:X/\!\sim\to Y$ 也满足 $f=g\circ q$。商集的任意元素都可以写成 $[x]=q(x)$，于是

$$
g([x])
=g(q(x))
=f(x)
=\overline f([x]).
$$

因此 $g$ 与 $\overline f$ 在商集的每个元素上取值相同，只能有 $g=\overline f$。

反过来，假设存在 $\overline f$ 使 $f=\overline f\circ q$。若 $x\sim y$，则 $q(x)=q(y)$，从而

$$
f(x)
=\overline f(q(x))
=\overline f(q(y))
=f(y).
$$

故 $f$ 在每个等价类上为常值。两个条件等价，且诱导映射唯一。证毕。

“泛性质”刻画的是商集面对任意目标集合 $Y$ 和任意类内常值映射 $f$ 时的共同性质。商群、商环与商空间会保留同样的分解形式，同时要求诱导映射保持相应的代数或拓扑结构。

## 6. 每个映射都在制造一个商

给定任意映射 $f:X\to Y$，可以在 $X$ 上定义关系

$$
x\sim&#95;f y
\quad\Longleftrightarrow\quad
f(x)=f(y).
$$

下标 $f$ 表示这个关系由映射 $f$ 决定。它是等价关系：

- 因为 $f(x)=f(x)$，所以 $x\sim&#95;f x$；
- 若 $f(x)=f(y)$，则 $f(y)=f(x)$，所以关系对称；
- 若 $f(x)=f(y)$ 且 $f(y)=f(z)$，则 $f(x)=f(z)$，所以关系传递。

等价类

$$
[x]&#95;f
=
\lbrace x'\in X:f(x')=f(x)\rbrace
$$

正是单点 $\lbrace f(x)\rbrace$ 的原像，也称为 $f$ 在 $f(x)$ 上的**纤维**：

$$
[x]&#95;f=f^{-1}(\lbrace f(x)\rbrace).
$$

映射 $f$ 在每条纤维上取同一个值，因此按照定理 5.1，它通过商集 $X/\!\sim&#95;f$ 唯一分解。

**定理 6.1（映射的标准分解）**　映射 $f:X\to Y$ 诱导一个双射

$$
\widetilde f:X/\!\sim&#95;f
\longrightarrow
\operatorname{im}f,
\qquad
\widetilde f([x]&#95;f)=f(x).
$$

**证明。** 良定义性来自 $\sim&#95;f$ 的定义：若 $[x]&#95;f=[y]&#95;f$，则 $f(x)=f(y)$。

若

$$
\widetilde f([x]&#95;f)=\widetilde f([y]&#95;f),
$$

则 $f(x)=f(y)$，所以 $x\sim&#95;f y$，进而 $[x]&#95;f=[y]&#95;f$。因此 $\widetilde f$ 是单射。

任取 $z\in\operatorname{im}f$。像集的定义保证存在 $x\in X$ 使 $z=f(x)$，于是

$$
z=\widetilde f([x]&#95;f).
$$

所以 $\widetilde f$ 是满射，最终为双射。证毕。

这一定理把任意映射拆成三部分：

$$
X
\xrightarrow{\ q\ }
X/\!\sim&#95;f
\xrightarrow{\ \widetilde f\ }
\operatorname{im}f
\xrightarrow{\ \iota\ }
Y.
$$

$q$ 是满射，$\widetilde f$ 是双射，$\iota$ 是把像集包含进陪域的单射。因此原映射满足

$$
f=\iota\circ\widetilde f\circ q.
$$

后面学习群同态时，关系 $x\sim&#95;f y$ 会由一个正规子群描述，定理 6.1 会升级为第一同构定理。此处的集合论分解正是那个定理的骨架。

## 7. 三个商结构的例子

### 7.1 整数模 $n$

模 $n$ 同余关系给出商集

$$
\mathbb Z/n\mathbb Z
:=
\mathbb Z/\!\equiv&#95;n.
$$

它包含 $n$ 个等价类，可以写成

$$
[0],[1],\ldots,[n-1].
$$

任何整数 $a$ 除以 $n$ 后都有唯一余数 $r\in\lbrace0,1,\ldots,n-1\rbrace$，并且 $[a]=[r]$。符号 $[0]$ 表示完整的等价类

$$
[0]=\lbrace kn:k\in\mathbb Z\rbrace.
$$

可以在商集上定义

$$
[a]+[b]:=[a+b].
$$

这个公式只有在代表元改变时结果不变，才真正定义了运算。附录 D 会证明：若 $a\equiv a'\pmod n$ 且 $b\equiv b'\pmod n$，则

$$
a+b\equiv a'+b'\pmod n.
$$

因此加法良定义。下一篇讨论群时，$\mathbb Z/n\mathbb Z$ 会成为最基本的有限群之一。

### 7.2 把区间的两个端点视为同一点

在闭区间 $[0,1]$ 上定义等价关系：每个内部点只与自己等价，并额外规定 $0\sim1$。于是商集

$$
[0,1]/(0\sim1)
$$

把两个端点合并成一个等价类 $\lbrace0,1\rbrace$。

考虑映射

$$
p:[0,1]\to S^1,
\qquad
p(t)=(\cos 2\pi t,\sin 2\pi t),
$$

其中

$$
S^1
=
\lbrace(x,y)\in\mathbb R^2:x^2+y^2=1\rbrace
$$

是单位圆周。若 $p(s)=p(t)$ 且 $s,t\in[0,1]$，则要么 $s=t$，要么 $\lbrace s,t\rbrace=\lbrace0,1\rbrace$。所以 $p$ 的纤维恰好是上述等价类。定理 6.1 给出集合之间的双射

$$
[0,1]/(0\sim1)
\cong S^1.
$$

符号 $\cong$ 在这里只表示存在自然双射。要进一步断言两边是同胚的拓扑空间，还必须给商集配备商拓扑，并证明诱导双射及其逆映射连续。这部分将在点集拓扑篇中完成。

### 7.3 同伦类

设 $X,Y$ 是拓扑空间，所有从 $X$ 到 $Y$ 的连续映射组成集合

$$
\mathcal C(X,Y).
$$

若两个连续映射可以通过连续变形互相连接，就称它们同伦，记作 $f\simeq g$。同伦关系满足自反、对称和传递，因此可以形成商集

$$
[X,Y]
:=
\mathcal C(X,Y)/\!\simeq.
$$

这里 $[X,Y]$ 的元素是连续映射的同伦类。基本群也从同样的动作开始：先取所有基于同一点的环路，再按照固定端点同伦取商，最后证明环路拼接能在商集上给出良定义的群运算。

现有文章[《从环路到自由群：基本群与空间的拼接》](/2026/08/30/2026-08-30-fundamental-group-van-kampen/)已经使用了这套构造。这里建立的商集语言，是其中同伦类、自由群和 van Kampen 定理的集合论基础。

## 8. 后续论证会反复使用的规则

后续论证会反复使用几项判断方法。

证明集合相等时，可以逐点证明双包含；处理映射时，必须同时记住定义域与陪域；在代表元上写公式时，必须检查良定义性；把对象按等价关系取商后，类内常值映射会唯一地通过商集分解。

从这里可以分别进入抽象代数、拓扑学与分析学。下一篇先沿抽象代数方向，从集合上的二元运算开始定义群，并依次证明单位元与逆元的唯一性、消去律、子群判别法以及群同态的基本性质。模 $n$ 同余类会从一个商集变成带有运算的代数结构。

## 参考文献

1. Paul R. Halmos, *Naive Set Theory*, Springer, 1974.
2. Herbert B. Enderton, *Elements of Set Theory*, Academic Press, 1977.
3. Serge Lang, *Algebra*, Revised Third Edition, Springer, 2002.
4. Paolo Aluffi, *Algebra: Chapter 0*, American Mathematical Society, 2009.
5. James R. Munkres, *Topology*, Second Edition, Prentice Hall, 2000.
6. Allen Hatcher, *Algebraic Topology*, Cambridge University Press, 2002.

---

## 附录

如下为正文附录补充。

## A. 集合运算的逐步证明

### A.1 交集的交换律

要证明

$$
A\cap B=B\cap A,
$$

任取 $x\in A\cap B$。按照交集定义，$x\in A$ 且 $x\in B$。交换两个命题的书写顺序，得到 $x\in B$ 且 $x\in A$，所以 $x\in B\cap A$。因此

$$
A\cap B\subseteq B\cap A.
$$

同理，任取 $x\in B\cap A$ 可以推出 $x\in A\cap B$，所以

$$
B\cap A\subseteq A\cap B.
$$

双包含给出所需等式。

### A.2 第二条 De Morgan 律

固定全集 $U$，并设 $A,B\subseteq U$。任取 $x\in U$，则

$$
\begin{aligned}
x\in(A\cap B)^{\mathsf c}
&\Longleftrightarrow x\notin A\cap B\\\\
&\Longleftrightarrow (x\notin A)\lor(x\notin B)\\\\
&\Longleftrightarrow (x\in A^{\mathsf c})\lor(x\in B^{\mathsf c})\\\\
&\Longleftrightarrow x\in A^{\mathsf c}\cup B^{\mathsf c}.
\end{aligned}
$$

因此

$$
(A\cap B)^{\mathsf c}
=A^{\mathsf c}\cup B^{\mathsf c}.
$$

### A.3 分配律

证明

$$
A\cap(B\cup C)
=(A\cap B)\cup(A\cap C).
$$

任取元素 $x$。

先设 $x\in A\cap(B\cup C)$。于是 $x\in A$，并且 $x\in B\cup C$。后一条件分成两种情形。若 $x\in B$，则 $x\in A\cap B$；若 $x\in C$，则 $x\in A\cap C$。无论哪种情形，都有

$$
x\in(A\cap B)\cup(A\cap C).
$$

这证明了

$$
A\cap(B\cup C)
\subseteq
(A\cap B)\cup(A\cap C).
$$

反过来，设 $x\in(A\cap B)\cup(A\cap C)$。若 $x\in A\cap B$，则 $x\in A$ 且 $x\in B\subseteq B\cup C$；若 $x\in A\cap C$，同样得到 $x\in A$ 且 $x\in C\subseteq B\cup C$。两种情形都推出 $x\in A\cap(B\cup C)$，因此反向包含也成立。双包含给出分配律。

## B. 像与原像的运算

设 $f:X\to Y$，$A,B\subseteq X$，$C,D\subseteq Y$。

### B.1 原像保持交集

任取 $x\in X$，

$$
\begin{aligned}
x\in f^{-1}(C\cap D)
&\Longleftrightarrow f(x)\in C\cap D\\\\
&\Longleftrightarrow (f(x)\in C)\land(f(x)\in D).
\end{aligned}
$$

按照原像的定义，最后两个条件分别等价于

$$
x\in f^{-1}(C),
\qquad
x\in f^{-1}(D).
$$

因此 $x$ 同时属于两个原像，也就是

$$
x\in f^{-1}(C)\cap f^{-1}(D).
$$

每一步都可以反向推出前一步，因此

$$
f^{-1}(C\cap D)=f^{-1}(C)\cap f^{-1}(D).
$$

### B.2 原像保持补集

任取 $x\in X$，

$$
\begin{aligned}
x\in f^{-1}(Y\setminus C)
&\Longleftrightarrow f(x)\in Y\setminus C\\\\
&\Longleftrightarrow f(x)\notin C\\\\
&\Longleftrightarrow x\notin f^{-1}(C)\\\\
&\Longleftrightarrow x\in X\setminus f^{-1}(C).
\end{aligned}
$$

所以

$$
f^{-1}(Y\setminus C)=X\setminus f^{-1}(C).
$$

### B.3 像保持并集

任取 $y\in Y$。

先设 $y\in f(A\cup B)$。按照像集的定义，存在 $x\in A\cup B$ 使 $f(x)=y$。若 $x\in A$，则 $y\in f(A)$；若 $x\in B$，则 $y\in f(B)$。所以 $y\in f(A)\cup f(B)$，从而

$$
f(A\cup B)\subseteq f(A)\cup f(B).
$$

反过来，若 $y\in f(A)\cup f(B)$，则 $y$ 至少属于 $f(A)$ 与 $f(B)$ 之一。前一种情形给出某个 $x\in A\subseteq A\cup B$ 使 $f(x)=y$；后一种情形给出某个 $x\in B\subseteq A\cup B$ 使 $f(x)=y$。两种情形都说明 $y\in f(A\cup B)$。因此反向包含成立，最终得到

$$
f(A\cup B)=f(A)\cup f(B).
$$

### B.4 像对交集何时保持等号

正文已经证明一般只有

$$
f(A\cap B)\subseteq f(A)\cap f(B).
$$

若额外假设 $f$ 是单射，可以证明等号成立。任取 $y\in f(A)\cap f(B)$，存在 $a\in A$ 与 $b\in B$ 使

$$
f(a)=y=f(b).
$$

单射性给出 $a=b$。因此同一个元素 $a$ 同时属于 $A$ 与 $B$，即 $a\in A\cap B$，从而 $y=f(a)\in f(A\cap B)$。这证明反向包含，故

$$
f(A\cap B)=f(A)\cap f(B).
$$

## C. 单射、满射与复合

### C.1 单射的复合仍是单射

设 $f:X\to Y$ 与 $g:Y\to Z$ 都是单射。若

$$
(g\circ f)(x&#95;1)=(g\circ f)(x&#95;2),
$$

则

$$
g(f(x&#95;1))=g(f(x&#95;2)).
$$

由 $g$ 的单射性，$f(x&#95;1)=f(x&#95;2)$；再由 $f$ 的单射性，$x&#95;1=x&#95;2$。因此 $g\circ f$ 是单射。

### C.2 满射的复合仍是满射

设 $f:X\to Y$ 与 $g:Y\to Z$ 都是满射。任取 $z\in Z$。由 $g$ 满射，存在 $y\in Y$ 使 $g(y)=z$；由 $f$ 满射，存在 $x\in X$ 使 $f(x)=y$。于是

$$
(g\circ f)(x)=g(f(x))=g(y)=z.
$$

所以 $g\circ f$ 是满射。

### C.3 单侧逆能推出什么

若存在 $g:Y\to X$ 使

$$
g\circ f=\operatorname{id}&#95;X,
$$

则 $f$ 必为单射。因为若 $f(x&#95;1)=f(x&#95;2)$，应用 $g$ 后得到

$$
x&#95;1=g(f(x&#95;1))=g(f(x&#95;2))=x&#95;2.
$$

此时 $g$ 称为 $f$ 的左逆。

若存在 $h:Y\to X$ 使

$$
f\circ h=\operatorname{id}&#95;Y,
$$

则 $f$ 必为满射。对任意 $y\in Y$，取 $x=h(y)$，便有 $f(x)=y$。此时 $h$ 称为 $f$ 的右逆。

一个映射可能只有左逆或只有右逆。只有当同一个逆映射同时满足两侧等式时，定理 2.1 才保证 $f$ 是双射。

## D. 模 $n$ 运算的良定义性

### D.1 加法

假设

$$
a\equiv a'\pmod n,
\qquad
b\equiv b'\pmod n.
$$

则存在 $k,\ell\in\mathbb Z$ 使

$$
a-a'=kn,
\qquad
b-b'=\ell n.
$$

两式相加得到

$$
(a+b)-(a'+b')=(k+\ell)n.
$$

所以

$$
a+b\equiv a'+b'\pmod n.
$$

因此 $[a]+[b]=[a+b]$ 与代表元选择无关。

### D.2 乘法

在同样的假设下，写成

$$
a=a'+kn,
\qquad
b=b'+\ell n.
$$

于是

$$
\begin{aligned}
ab-a'b'
&=(a'+kn)(b'+\ell n)-a'b'\\\\
&=a'\ell n+b'kn+k\ell n^2\\\\
&=n(a'\ell+b'k+k\ell n).
\end{aligned}
$$

所以 $n\mid(ab-a'b')$，即

$$
ab\equiv a'b'\pmod n.
$$

因此还可以良定义地写出

$$
[a][b]:=[ab].
$$

加法将使 $\mathbb Z/n\mathbb Z$ 成为群；加法与乘法放在一起，则会使它成为环。群和环的公理将在后续文章分别介绍。

## E. 区间端点商集与圆周的双射

定义 $p:[0,1]\to S^1$ 为

$$
p(t)=(\cos2\pi t,\sin2\pi t).
$$

下面验证 $p(s)=p(t)$ 的准确条件。若 $p(s)=p(t)$，两个坐标分别相等：

$$
\begin{aligned}
\cos2\pi s&=\cos2\pi t,\\\\
\sin2\pi s&=\sin2\pi t.
\end{aligned}
$$

使用余弦差角公式，

$$
\begin{aligned}
\cos\bigl(2\pi(s-t)\bigr)
&=\cos2\pi s\cos2\pi t\\\\
&\quad+\sin2\pi s\sin2\pi t\\\\
&=\cos^2 2\pi t+\sin^2 2\pi t\\\\
&=1.
\end{aligned}
$$

实数角 $\theta$ 满足 $\cos\theta=1$，当且仅当 $\theta=2k\pi$，其中 $k\in\mathbb Z$。取 $\theta=2\pi(s-t)$，得到

$$
s-t=k\in\mathbb Z.
$$

反过来，若 $s-t=k\in\mathbb Z$，正弦和余弦以 $2\pi$ 为周期，所以 $p(s)=p(t)$。因此

$$
p(s)=p(t)
\quad\Longleftrightarrow\quad
s-t\in\mathbb Z.
$$

因为 $s,t\in[0,1]$，差 $s-t$ 只能落在 $[-1,1]$。其中的整数只有 $-1,0,1$：

- 若 $s-t=0$，则 $s=t$；
- 若 $s-t=1$，则 $s=1,t=0$；
- 若 $s-t=-1$，则 $s=0,t=1$。

所以 $p$ 只识别两个端点，其余点都保持区分。按照定理 6.1，诱导映射

$$
\widetilde p:[0,1]/(0\sim1)\to S^1,
\qquad
\widetilde p([t])=p(t)
$$

是双射。

目前得到的是集合之间的双射。给商集配备商拓扑，并证明诱导双射及其逆映射连续后，才能得到同胚结论。这一步将在商空间部分完成。
