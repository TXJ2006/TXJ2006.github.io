---
title: "Supplement II: Weak Derivatives and Sobolev Spaces"
date: 2026-10-14 10:10:00
series: Analysis Supplements
series_order: 2
subtitle: From integration by parts to weak solutions
description: Reading notes on test functions, distributions, weak derivatives, Sobolev spaces, and weak formulations
categories:
  - Analysis Supplements
tags:
  - Weak Derivatives
  - Sobolev Spaces
  - Partial Differential Equations
  - Reading Notes
mathjax: true
toc: true
toc_number: false
comments: true
---

After [Supplement I](/2026/10/14/2026-10-14-supplement-measure-lebesgue-lp/), we read John K. Hunter, *Notes on Partial Differential Equations*, Sections 3.1--3.6 and 4.1. The central construction is to move a derivative from a rough function onto a smooth test function by integration by parts. The resulting identity depends only on the almost-everywhere class of the original function. Our main examples will distinguish a corner, which admits an integrable weak derivative, from a jump, whose distributional derivative is a point mass.

<!--more-->

## 1. Why weaken the derivative?

Let $\Omega\subseteq\mathbb R^n$ be an open set, where the positive integer $n$ is the dimension of the ambient Euclidean space. A classical derivative is defined by a pointwise limit. This is too rigid for many functions that arise as limits of smooth approximations or as solutions of partial differential equations.

The function

$$
u(x)=|x|\qquad(-1<x<1)
$$

is not classically differentiable at $x=0$, but it has a natural derivative in the sense that

$$
u'(x)=
\begin{cases}
-1,&x<0,\\
1,&x>0.
\end{cases}
$$

The missing value at $0$ does not matter for integration: a single point has Lebesgue measure $0$. This suggests an integral definition of differentiation. It does not justify merely ignoring every point where classical differentiation fails; the jump example below will show why that inference is false. Throughout, $dx$ denotes Lebesgue measure, and $L^p(\Omega)$ identifies functions that agree a.e.

## 2. Test functions

Write

$$
C_c^\infty(\Omega)
$$

for the space of infinitely differentiable functions $\varphi:\Omega\to\mathbb R$ whose support is compact and contained in $\Omega$. The support is

$$
\operatorname{supp}\varphi=\overline{\lbrace x\in\Omega:\varphi(x)\neq0\rbrace}.
$$

The overline denotes closure in $\mathbb R^n$. Compact support contained in $\Omega$ means that the nonzero part stays away from the boundary, so boundary terms disappear when we integrate by parts. The subscript $c$ means compact support and the superscript $\infty$ means smoothness of all orders. In one dimension, an example is $\varphi(x)=\exp(-1/(1-x^2))$ for $|x|<1$ and $\varphi(x)=0$ for $|x|\geq1$, considered on $\mathbb R$.

For a multi-index $\alpha=(\alpha_1,\ldots,\alpha_n)\in\mathbb N_0^n$, define

$$
|\alpha|=\alpha_1+\cdots+\alpha_n,
\qquad
\partial^\alpha
=\frac{\partial^{|\alpha|}}
{\partial x_1^{\alpha_1}\cdots\partial x_n^{\alpha_n}}.
$$

Here $\mathbb N_0=\{0,1,2,\ldots\}$ and $|\alpha|$ is the total order of the derivative. For example, $\alpha=(2,1)$ gives $\partial^\alpha u=\partial_{x_1}^2\partial_{x_2}u$. The zero multi-index denotes $u$ itself.

## 3. Distributions and weak derivatives

A **distribution** on $\Omega$ assigns a real number linearly and continuously to each test function. For this definition, $\varphi_j\to\varphi$ means that all supports lie in one compact subset of $\Omega$, and every derivative $\partial^\alpha\varphi_j$ converges uniformly to $\partial^\alpha\varphi$. Continuity requires that the assigned numbers converge as well. We write the action as

$$
\langle T,\varphi\rangle.
$$

The brackets denote application of a functional, not an inner product. A function belongs to $L^1_{\mathrm{loc}}(\Omega)$ if $\int_K|u|\;dx<\infty$ for every compact subset $K\subset\Omega$. Every such function defines a distribution $T_u$ by

$$
\langle T_u,\varphi\rangle
=\int_\Omega u(x)\varphi(x)\;dx.
$$

The distributional derivative is defined by moving the derivative onto the test function:

$$
\langle \partial^\alpha T,\varphi\rangle
=(-1)^{|\alpha|}\langle T,\partial^\alpha\varphi\rangle.
$$

If there is a function $v\in L^1_{\mathrm{loc}}(\Omega)$ such that

$$
\int_\Omega v\varphi\;dx
=(-1)^{|\alpha|}\int_\Omega u\;\partial^\alpha\varphi\;dx
\qquad\text{for every }\varphi\in C_c^\infty(\Omega),
$$

then $v$ is called the **weak derivative** $\partial^\alpha u$.

Every distribution has derivatives by this definition; a **weak derivative as a function** exists only when the distributional derivative can be represented by a locally integrable function $v$. Such a $v$ is unique a.e. If $u$ is smooth, ordinary integration by parts gives exactly this identity, so weak derivatives extend classical derivatives.

<div class="lecture-example">
  <p><strong>The corner $u(x)=|x|$.</strong> For $\varphi\in C_c^\infty((-1,1))$, split the integral at $0$ and integrate by parts:
  $$-\int_{-1}^1|x|\varphi'(x)\;dx
  =-\int_{-1}^0\varphi(x)\;dx+\int_0^1\varphi(x)\;dx.$$
  The endpoint terms vanish: $\varphi$ vanishes near $-1$ and $1$, and $|x|$ vanishes at $0$. Thus the weak derivative is $\operatorname{sgn}(x)$, equal to $-1$ for $x<0$ and $1$ for $x>0$. Its value at $0$ may be chosen arbitrarily. Both $u$ and its weak derivative are bounded on this interval, so $u\in W^{1,p}((-1,1))$ for every $1\leq p\leq\infty$.</p>
</div>

Now let $h(x)=\mathbf 1_{(0,1)}(x)$ on $(-1,1)$. Its classical derivative is $0$ except at $0$, but

$$
-\int_{-1}^1h(x)\varphi'(x)\;dx
=-\int_0^1\varphi'(x)\;dx=\varphi(0).
$$

The map $\delta_0:\varphi\mapsto\varphi(0)$ is the **Dirac distribution** at $0$. It cannot be represented by a locally integrable function. Indeed, suppose $v$ represented it. Choose $\varphi_j(x)=\eta(jx)$, where $\eta$ is a smooth function supported in $(-1,1)$ with $0\leq\eta\leq1$ and $\eta(0)=1$. Then $\int v\varphi_j\;dx=1$, whereas $|\int v\varphi_j\;dx|\leq\int_{-1/j}^{1/j}|v|\;dx\to0$, a contradiction. Thus $h$ has distributional derivative $\delta_0$, but no weak derivative in $L^1_{\mathrm{loc}}$; in particular $h\notin W^{1,p}((-1,1))$. This is Hunter's Example 3.4 restricted to our interval.

![A corner has a bounded weak derivative, while a jump has a Dirac distribution as its derivative.](/images/notes/assets/analysis/corner-and-jump.svg)

<p class="figure-caption">Our diagram, following the distinction in Hunter's Examples 3.3--3.4. Both classical derivatives fail only at the origin. Read the lower panels as distributions: the corner gives the integrable function sign(x); the jump gives a point mass, drawn as an arrow of mass 1.</p>

## 4. Sobolev spaces

For $1\leq p\leq\infty$ and an integer $k\geq0$, define

$$
W^{k,p}(\Omega)
=\left\{u\in L^p(\Omega):
\partial^\alpha u\in L^p(\Omega)\text{ for every }|\alpha|\leq k\right\},
$$

where every derivative is understood in the weak sense. A standard norm is

$$
\lVert u\rVert_{W^{k,p}}
=\left(\sum_{|\alpha|\leq k}
\lVert\partial^\alpha u\rVert_p^p\right)^{1/p}
\qquad(1\leq p<\infty).
$$

For $p=\infty$, replace the sum by the maximum of the corresponding $L^\infty$ norms. The space

$$
H^k(\Omega)=W^{k,2}(\Omega)
$$

is the Hilbert-space case. In particular,

$$
H^1(\Omega)=\{u\in L^2(\Omega):\partial_{x_j}u\in L^2(\Omega),\ 1\leq j\leq n\}.
$$

Here $\nabla u=(\partial_{x_1}u,\ldots,\partial_{x_n}u)$ is the weak gradient. It is an $L^2$ vector field even when $u$ has no pointwise gradient everywhere. These Sobolev spaces are complete. For $p=2$, the inner product is the sum of $L^2$ inner products of derivatives up to order $k$; for $k=1$ it is $\int_\Omega(uv+\nabla u\cdot\nabla v)\;dx$. The dot is the Euclidean inner product of vectors.

For $1\leq p<\infty$, we also need

$$
W_0^{1,p}(\Omega)=\overline{C_c^\infty(\Omega)}^{\,W^{1,p}(\Omega)},
\qquad H_0^1(\Omega)=W_0^{1,2}(\Omega).
$$

The overline now means closure in the indicated norm: every element is a limit in $W^{1,p}$ of compactly supported smooth functions. On a Lipschitz domain, these are precisely the functions with zero boundary trace. A Lipschitz domain has a boundary locally represented, after rotation, as the graph of a Lipschitz function. A **trace** extends the boundary values of smooth functions continuously to Sobolev functions; it is not obtained by assigning arbitrary values on the measure-zero boundary.

## 5. Smooth approximation by mollifiers

Work first on all of $\mathbb R^n$. Choose a nonnegative function $\rho\in C_c^\infty(\mathbb R^n)$ supported in the unit ball, with

$$
\int_{\mathbb R^n}\rho(x)\;dx=1.
$$

For $\varepsilon>0$, set

$$
\rho_\varepsilon(x)=\varepsilon^{-n}\rho(x/\varepsilon).
$$

The **mollification** of $u$ is the convolution

$$
u_\varepsilon(x)=(u*\rho_\varepsilon)(x)
=\int_{\mathbb R^n}u(x-y)\rho_\varepsilon(y)\;dy.
$$

For $u\in W^{k,p}(\mathbb R^n)$ with $1\leq p<\infty$, the function $u_\varepsilon$ is smooth and

$$
\partial^\alpha u_\varepsilon
=(\partial^\alpha u)*\rho_\varepsilon.
$$

Moreover $\lVert u_\varepsilon-u\rVert_{W^{k,p}(\mathbb R^n)}\to0$ as $\varepsilon\downarrow0$. Thus both the function and its weak derivatives are approximated in the same norm. On a general open set the convolution is defined directly only where the distance to the boundary exceeds $\varepsilon$; global approximation needs additional work with cutoffs or an extension. We do not assume that zero extension across an arbitrary boundary preserves Sobolev regularity.

## 6. Poincare's inequality

Let $\Omega$ be a bounded connected Lipschitz domain and let $1\leq p<\infty$. If

$$
u_\Omega=\frac1{|\Omega|}\int_\Omega u(x)\;dx
$$

is the average of $u$, where $|\Omega|=\int_\Omega1\;dx$ is its volume, then Poincare's inequality gives

$$
\lVert u-u_\Omega\rVert_{L^p(\Omega)}
\leq C\lVert\nabla u\rVert_{L^p(\Omega)}
\qquad(u\in W^{1,p}(\Omega)).
$$

The constant $C$ depends on $\Omega$ and $p$, not on $u$. Subtracting the average is necessary because constant functions have zero gradient. For functions in $W_0^{1,p}(\Omega)$, the boundary condition removes this ambiguity and one obtains

$$
\lVert u\rVert_{L^p(\Omega)}
\leq C\lVert\nabla u\rVert_{L^p(\Omega)}.
$$

On $(0,1)$ this estimate has a short proof. For a smooth function with $u(0)=0$, $u(x)=\int_0^xu'(t)\;dt$, so $|u(x)|\leq\int_0^1|u'(t)|\;dt\leq\lVert u'\rVert_p$ by Holder's inequality. Integrating $|u(x)|^p$ over the unit interval gives $\lVert u\rVert_p\leq\lVert u'\rVert_p$. Approximation extends the inequality to $W_0^{1,p}(0,1)$. This is the estimate that will make gradient energy control the entire $H_0^1$ norm.

## 7. Sobolev embedding

Assume $1\leq p<n$ and define the critical exponent

$$
p^*=\frac{np}{n-p}.
$$

If $\Omega$ is a bounded Lipschitz domain,

$$
W^{1,p}(\Omega)\hookrightarrow L^{p^*}(\Omega),
$$

which means that every $W^{1,p}$ function belongs to $L^{p^*}$ and

$$
\lVert u\rVert_{L^{p^*}(\Omega)}
\leq C\lVert u\rVert_{W^{1,p}(\Omega)}.
$$

Here $\hookrightarrow$ means continuous inclusion, quantified by the displayed estimate; it does not mean a compact inclusion. The exponent is forced by scaling: for $u_r(x)=u(rx)$ on $\mathbb R^n$, $\lVert u_r\rVert_q=r^{-n/q}\lVert u\rVert_q$ and $\lVert\nabla u_r\rVert_p=r^{1-n/p}\lVert\nabla u\rVert_p$. A bound $\lVert u\rVert_q\leq C\lVert\nabla u\rVert_p$ for all compactly supported smooth functions and all $r>0$ requires $-n/q=1-n/p$, hence $q=p^*$. This is Hunter's scaling argument in Section 3.7.

For instance, $W^{1,2}$ embeds into $L^6$ in dimension $3$. For $n\geq2$ and $p=n$, the same bounded-domain hypotheses give inclusion in every finite $L^q$, but generally not $L^\infty$. For $n<p<\infty$, Morrey's inequality gives a representative that is Holder continuous with exponent $1-n/p$. These are different conclusions; the formula for $p^*$ applies only when $p<n$.

## 8. Weak formulations

Let $\Omega$ be a bounded Lipschitz domain. Consider the Dirichlet problem

$$
-\Delta u=f\quad\text{in }\Omega,
\qquad u=0\quad\text{on }\partial\Omega.
$$

Here $\Delta u=\sum_{j=1}^n\partial_{x_j}^2u$ is the Laplacian, $\partial\Omega$ is the boundary, and $f$ is the prescribed source. For a smooth solution and $\varphi\in C_c^\infty(\Omega)$, multiply by $\varphi$ and integrate:

$$
\int_\Omega(-\Delta u)\varphi\;dx
=\int_\Omega f\varphi\;dx.
$$

When $f\in L^2(\Omega)$, integration by parts gives

$$
\int_\Omega\nabla u\cdot\nabla\varphi\;dx
=\int_\Omega f\varphi\;dx.
$$

More generally, $H^{-1}(\Omega)=(H_0^1(\Omega))^*$ is the space of bounded linear functionals on $H_0^1(\Omega)$; the superscript $*$ denotes the continuous dual. For $f$ in this space, the right-hand side is the dual pairing

$$
\langle f,\varphi\rangle_{H^{-1},H_0^1}
$$

rather than an ordinary integral. Thus a weak solution $u\in H_0^1(\Omega)$ satisfies

$$
\int_\Omega\nabla u\cdot\nabla\varphi\;dx
=\langle f,\varphi\rangle_{H^{-1},H_0^1}
\qquad\text{for every }\varphi\in H_0^1(\Omega).
$$

This formulation makes sense even when the classical second derivatives of $u$ do not exist. If $f\in L^2$, the pairing reduces to the usual integral by Cauchy--Schwarz. The requirement $u\in H_0^1$ encodes the boundary condition; the integral identity encodes the equation. Testing only the equation would leave the boundary condition unspecified. Existence follows from the Hilbert-space representation theorem and Poincare's inequality; Hunter gives that argument in Section 4.5, which is a subsequent reading rather than a proof supplied here.

## 9. Why this supplement belongs here

Sobolev spaces are not prerequisites for the first definitions of homotopy, fundamental groups, or singular homology. They become relevant when topology meets analysis: differential forms in $L^2$, Hodge Laplacians, elliptic regularity, and geometric analysis.

The distinction worth retaining is that ignoring null-set values does not erase a jump: $h'=0$ classically a.e., yet its distributional derivative is $\delta_0$. Integral identities preserve information that pointwise differentiation can lose. Sobolev membership asks that this derivative be represented by a function in the same $L^p$ scale. The chain of definitions is

$$
\text{measure}\longrightarrow L^p
\longrightarrow\text{weak derivatives}
\longrightarrow W^{k,p}\text{ and }H^k
\longrightarrow\text{weak PDEs}.
$$

## 10. Exercises for this reading

1. Verify the two integrations by parts for $|x|$, then calculate its $W^{1,p}((-1,1))$ norm for $1\leq p<\infty$.
2. Show that a constant function has zero weak gradient and explain why Poincare's inequality uses $u-u_\Omega$.
3. Let $u_\varepsilon(x)=\sqrt{x^2+\varepsilon^2}$ on $(-1,1)$. Prove that $u_\varepsilon\to|x|$ in $W^{1,p}$ for every finite $p$, using dominated convergence for the derivatives. Explain why convergence in $W^{1,\infty}$ fails.
4. For $f=1$ on $(0,1)$, check that $u(x)=x(1-x)/2$ belongs to $H_0^1$ and satisfies the weak identity. Calculate $\int_0^1|u'|^2\;dx$ and $\int_0^1u\;dx$ and explain why they agree.

## Sources

- John K. Hunter, *Notes on Partial Differential Equations*, revised 18 June 2014: Sections 3.1--3.6 (pp. 47--59), 3.7--3.11 (pp. 59--76), and 4.1--4.5 (pp. 91--100); mollifiers in Section 1.9 (pp. 11--12). [Course page](https://www.math.ucdavis.edu/~hunter/pdes/pdes.html); [PDF](https://www.math.ucdavis.edu/~hunter/pdes/pde_notes.pdf). Printed page $r$ is PDF page $r+6$. The examples here adapt the argument of Examples 3.3--3.4; no source figure is reproduced.
- Sheldon Axler, *Measure, Integration & Real Analysis*, Sections 7A--7B: [author's page](https://measure.axler.net/) and [PDF](https://measure.axler.net/MIRA.pdf).
- Richard Melrose, *Functional Analysis: Lecture Notes for 18.102*: [course page](https://math.mit.edu/~rbm/18.102-S13/index.html) and [lecture notes](https://math.mit.edu/~rbm/18.102-S13/FunctAnal.pdf).
