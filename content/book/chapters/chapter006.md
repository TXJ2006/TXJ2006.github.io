---
title: "Chapter 6: The Riesz Representation Theorem"
layout: "single"
url: "/book/chapters/chapter006/"
summary: "Substantiating the observer: Riesz representation as the structural hinge that permits models to be stored as tensors, optimized by gradient descent, and interpreted as geometric objects."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 6
---

<style>
  .post-content,
  .post-content h1, .post-content h2, .post-content h3, .post-content h4,
  .post-content p, .post-content li, .post-content blockquote,
  .post-content td, .post-content th {
    font-family: 'Times New Roman', 'Times', 'Noto Serif', Georgia, serif;
  }
  .post-content { font-size: 12pt; line-height: 1.72; }
  .def-box  { border-left:4px solid #4a148c; background:#faf5ff; padding:1em 1.2em; margin:1.5em 0; border-radius:4px; }
  .prop-box { border-left:4px solid #1565c0; background:#f0f6ff; padding:1em 1.2em; margin:1.5em 0; border-radius:4px; }
  .proof-box{ border-left:4px solid #999;    background:#fafafa; padding:.8em 1.2em; margin:1em 0 1.5em; border-radius:4px; }
  .ml-box   { border-left:4px solid #e65100; background:#fff8f0; padding:.8em 1.2em; margin:1em 0 1.5em; border-radius:4px; }
  .scholium-box { border:2px solid #6a1b9a; background:#fdf5ff; padding:1em 1.2em; margin:1.5em 0; border-radius:6px; }
</style>

<div style="text-align:center; margin:1.5em 0 2.5em 0;">

# Volume I &mdash; Mathematical Foundations and Axiomatization

## Chapter 6 &mdash; The Riesz Representation Theorem: Substantiating the Observer and the Terminal Isomorphism of Subject&ndash;Object

*Xujiang Tang*

</div>

## Abstract

The previous chapters separated *states* (data vectors) from *measurements* (dual functionals), and then imposed geometry (norms, inner products) and stability laws (operator norms). This chapter isolates the structural hinge on which a large fraction of modern machine learning silently depends: in a Hilbert space, every continuous linear measurement is representable as an inner product with a unique vector. This is not a "computational convenience." It is the mathematical clause that permits models to be stored as tensors, optimized by gradient descent, visualized as patterns, and interpreted as geometric objects.

We proceed from machine-learning pain points: (i) why some "rules" cannot be learned without regularization, (ii) why treating gradients as vectors is type-dependent and geometry-dependent, (iii) why representer theorems collapse solutions into data spans, and (iv) why the Banach regime breaks uniqueness and, in extreme cases, breaks representability altogether. Every theorem is proved with complete steps and every proof is translated into operational ML consequences.

---

## 6.1 Paradigm Shift: Why Can We "See" the Model?

### 6.1.1 Standard statement (Hilbert setting)

Let $H$ be a real Hilbert space with inner product $\langle\cdot,\cdot\rangle$ and induced norm $\|x\|=\sqrt{\langle x,x\rangle}$. Let $H^{\ast}$ denote the continuous (bounded) dual.

<div class="prop-box">

**Theorem 6.1 (Riesz representation).** For every $\phi\in H^{\ast}$, there exists a unique $w\in H$ such that

$$
\phi(x)=\langle x,w\rangle \quad \text{for all } x\in H,
$$

and the correspondence is isometric:

$$
\|\phi\|_{H^{\ast}}=\|w\|.
$$

Equivalently, the map $R:H\to H^{\ast}$ defined by $R(w)=\langle\,\cdot\,,w\rangle$ is a linear isometric isomorphism.

</div>

<div class="ml-box">

**Machine-learning translation.** A linear "rule" (continuous measurement) is not an ethereal object living outside the space; it is a vector *inside* the space. This is the precise condition under which weights can be treated as data-shaped tensors without type confusion.

</div>

---

### 6.1.2 The suppressed type distinction (and why engineering convenience is not ontology)

In the axioms, an input $x$ belongs to a primal space $H$, whereas a rule $\phi$ belongs to $H^{\ast}$. These are different types. The widespread ML habit of writing "$w$ and $x$ are both tensors" is justified *only* after one has chosen a Hilbert structure and invoked Riesz.

A convolution "filter" is logically a functional: it maps an image patch $x$ to a scalar score $\phi(x)$. It becomes drawable as an image-like array because the ambient geometry identifies $\phi$ with a unique $w$ such that $\phi(x)=\langle x,w\rangle$. In other geometries (general Banach spaces), this identification may fail (no representer) or fail to be canonical (non-unique norm-attaining functionals), and the visualization narrative loses mathematical legitimacy.

<div class="ml-box">

**Defect in common understanding (precise statement).** Many presentations treat the identification $H\simeq H^{\ast}$ as a triviality of finite-dimensional linear algebra. In fact:

1. it depends on the inner product, not merely on dimension;
2. it depends on completeness (Hilbert, not merely pre-Hilbert);
3. it depends on continuity/boundedness;
4. it fails in important function spaces unless extra structure is imposed.

This is why "weights look like data" is not a primitive truth; it is an artifact of a specific geometric contract.

</div>

---

### 6.1.3 Collapse from process to state (connectionist substantiation)

A functional $\phi$ is a *process*: it consumes $x$ and emits a scalar. Riesz asserts that, in Hilbert geometry, every continuous such process has a unique "state" $w$ that encodes it. This is the mathematical core behind connectionist realizability: a rule is physically stored as a weight configuration because the rule is representable as an inner product with that configuration.

A key frontier implication is that any learning paradigm that relies on storing rules as parameter tensors is implicitly committing to a Riesz-type identification&mdash;either exactly (Hilbert/RKHS) or approximately (finite-dimensional Euclidean surrogates).

---

## 6.2 Continuity: The Legitimacy Boundary of Representation

### 6.2.1 Why some "rules" cannot be learned: a functional-analytic diagnosis

Riesz requires $\phi\in H^{\ast}$, i.e., continuity (boundedness). If the rule is not continuous, then no vector representer exists.

This is not a technical nuisance; it is the exact point where "the model cannot be instantiated as weights" becomes a theorem.

---

### 6.2.2 A canonical counterexample: point evaluation in $L^2$ is not continuous

Let $H=L^2([0,1])$ with inner product $\langle f,g\rangle=\int_0^1 f(t)g(t)\,dt$. Consider the "rule" $E_{t_0}(f)=f(t_0)$ (point evaluation at $t_0\in[0,1]$). In learning terms, this is a measurement of a function's value at a point.

<div class="prop-box">

**Proposition 6.2.** The point-evaluation functional $E_{t_0}$ is not continuous on $L^2([0,1])$. Hence it is not in $H^{\ast}$ and has no Riesz representer.

</div>

<div class="proof-box">

*Proof (explicit sequence construction).*

Fix $t_0\in[0,1]$. For $n\in\mathbb{N}$, define

$$
f_n(t) :=
\begin{cases}
\sqrt{n}, & t\in[t_0,\,t_0+\tfrac{1}{n}] \cap [0,1], \\\\
0, & \text{otherwise}.
\end{cases}
$$

Then $f_n(t_0)=\sqrt{n}$ (interpreting $t_0$ as belonging to the interval). Compute the $L^2$ norm:

$$
\|f_n\|_2^2=\int_0^1 |f_n(t)|^2\,dt = \int_{t_0}^{t_0+1/n} n\,dt = n\cdot \frac{1}{n}=1,
$$

so $\|f_n\|_2=1$ for all $n$. Yet $E_{t_0}(f_n)=f_n(t_0)=\sqrt{n}\to\infty$.

If $E_{t_0}$ were continuous, there would exist $C>0$ such that $|E_{t_0}(f)|\le C\|f\|_2$ for all $f$. Applying this to $f_n$ yields $\sqrt{n}\le C$, contradiction. Hence $E_{t_0}$ is not continuous. $\square$

</div>

<div class="ml-box">

**Machine-learning translation.**

In $L^2$, "probing a function at a point" is too singular to be represented by a finite-energy weight function. Any attempt to learn such a rule without regularization forces the system toward distributions (Dirac-like objects), which are not legitimate elements of the hypothesis Hilbert space. This explains, at a structural level, why certain tasks induce unstable training or require smoothing/regularization: you are trying to learn a rule that lies outside the representable dual.

</div>

---

### 6.2.3 Regularization as continuity repair (the correct statement)

In ML practice, regularization is often presented as "prevent overfitting." The functional-analytic statement is sharper:

<div class="prop-box">

**Claim (structural).** Regularization restricts hypothesis classes so that the induced measurement functionals remain bounded, hence representable (and optimizable) within the chosen geometry.

</div>

Concrete example: in finite-dimensional Euclidean space, any linear rule $x\mapsto w^\top x$ is continuous and $\|\phi_w\|=\|w\|_2$. But in function spaces, continuity is a nontrivial constraint. Regularization (e.g., RKHS norm penalties) does not merely discourage complexity; it places the learner in a space where evaluation and learning operators are bounded, restoring Riesz representability and enabling stable optimization.

---

## 6.3 The Riesz Theorem: Full Proof and Structural Reading

### 6.3.1 Proof with complete steps (Hilbert space, real case)

Let $H$ be a real Hilbert space and $\phi\in H^{\ast}$, $\phi\ne 0$.

<div class="proof-box">

**Step 1 &mdash; Kernel is closed, hence orthogonal decomposition exists.**

Let $N:=\ker\phi=\{x\in H:\phi(x)=0\}$. Since $\phi$ is continuous, $N$ is closed. Therefore, by Hilbert space geometry,

$$
H = N \oplus N^\perp,
$$

i.e., each $x\in H$ decomposes uniquely as $x=n+u$ with $n\in N$, $u\in N^\perp$.

**Step 2 &mdash; $N^\perp$ is one-dimensional.**

Pick $u_0\notin N$ (possible since $\phi\ne 0$). Let $u$ be the orthogonal projection of $u_0$ onto $N^\perp$. Then $u\ne 0$ and $u\notin N$, so $\phi(u)\ne 0$.

Let $v\in N^\perp$. Define $\alpha := \phi(v)/\phi(u)$. Then

$$
\phi(v-\alpha u)=\phi(v)-\alpha\phi(u)=0,
$$

so $v-\alpha u\in N$. But $v\in N^\perp$ and $\alpha u\in N^\perp$, hence $v-\alpha u\in N^\perp$. Thus $v-\alpha u\in N\cap N^\perp=\{0\}$. Therefore $v=\alpha u$.

Hence $N^\perp=\mathrm{span}\{u\}$.

**Step 3 &mdash; Define the representer $w$.**

Set

$$
w := \frac{\phi(u)}{\|u\|^2}\,u.
$$

**Step 4 &mdash; Verify representation.**

For any $x\in H$, write $x=n+\beta u$ with $n\in N$. Then $\phi(x)=\beta\phi(u)$. Also,

$$
\langle x,w\rangle
=\Big\langle n+\beta u,\frac{\phi(u)}{\|u\|^2}u\Big\rangle
=\frac{\phi(u)}{\|u\|^2}\big(\langle n,u\rangle+\beta\langle u,u\rangle\big)
=\frac{\phi(u)}{\|u\|^2}\big(0+\beta\|u\|^2\big)
=\beta\phi(u)
=\phi(x).
$$

**Step 5 &mdash; Uniqueness.**

If $\phi(x)=\langle x,w\rangle=\langle x,w'\rangle$ for all $x$, then $\langle x,w-w'\rangle=0$ for all $x$. Taking $x=w-w'$ yields $\|w-w'\|^2=0$, hence $w=w'$.

**Step 6 &mdash; Isometry.**

For $\|x\|\le 1$, by Cauchy&ndash;Schwarz,

$$
|\phi(x)|=|\langle x,w\rangle|\le \|x\|\|w\|\le \|w\|.
$$

So $\|\phi\|\le \|w\|$. Conversely, take $x=w/\|w\|$ (if $w\ne 0$):

$$
|\phi(x)|=\Big|\Big\langle \frac{w}{\|w\|},w\Big\rangle\Big|=\|w\|.
$$

Thus $\|\phi\|=\|w\|$. $\square$

</div>

---

### 6.3.2 The hidden clause: Riesz is not "dual = primal"; it is "dual = primal via a metric"

The representer $w$ depends on the inner product. If the same underlying vector space is equipped with a different inner product $\langle x,y\rangle_G=x^\top G y$ (with $G\succ 0$), the representer of the *same* functional changes. Therefore, "the gradient vector" is a geometry-dependent choice of representer, not an intrinsic object.

This is the precise point at which many ML derivations quietly commit a type conversion without acknowledging the metric that justifies it.

---

## 6.4 The Representer Mechanism: Why Optimal Weights Are Shadows of Data

### 6.4.1 Representer theorem (RKHS form) with full proof

Let $H$ be an RKHS on $\mathcal{X}$ with reproducing kernel $k$. For each $x\in\mathcal{X}$, denote $k_x(\cdot)=k(x,\cdot)\in H$. The reproducing property is

$$
f(x)=\langle f,k_x\rangle_H.
$$

Consider training data $(x_i,y_i)_{i=1}^n$. Let $L:\mathbb{R}^n\to \mathbb{R}\cup\{+\infty\}$ be any loss depending on the values $(f(x_1),\dots,f(x_n))$, and let $\Omega:[0,\infty)\to\mathbb{R}$ be strictly increasing. Consider the optimization problem

$$
\min_{f\in H}\ \mathcal{J}(f):= L\big(f(x_1),\dots,f(x_n)\big) + \Omega(\|f\|_H).
$$

<div class="prop-box">

**Theorem 6.3 (Representer theorem).** Any minimizer $f^{\ast}$ of $\mathcal{J}$ admits a representation

$$
f^{\ast} = \sum_{i=1}^n \alpha_i\,k_{x_i}
\quad \text{for some coefficients } \alpha_1,\dots,\alpha_n\in\mathbb{R}.
$$

</div>

<div class="proof-box">

*Proof (complete, orthogonal decomposition).*

Let $S:=\mathrm{span}\{k_{x_1},\dots,k_{x_n}\}\subseteq H$. Since $S$ is finite-dimensional, it is closed. Decompose any $f\in H$ uniquely as

$$
f = s + r,
\quad s\in S,\ r\in S^\perp.
$$

By orthogonality, $\|f\|_H^2=\|s\|_H^2+\|r\|_H^2$.

Now examine the data-fit term. For each $i$,

$$
f(x_i)=\langle f,k_{x_i}\rangle
=\langle s+r, k_{x_i}\rangle
=\langle s,k_{x_i}\rangle + \langle r,k_{x_i}\rangle.
$$

But $k_{x_i}\in S$ and $r\in S^\perp$, hence $\langle r,k_{x_i}\rangle=0$. Therefore

$$
f(x_i)=\langle s,k_{x_i}\rangle = s(x_i).
$$

So the loss term depends only on $s$, not on $r$:

$$
L(f(x_1),\dots,f(x_n)) = L(s(x_1),\dots,s(x_n)).
$$

Meanwhile, the penalty term satisfies

$$
\|f\|_H^2=\|s\|_H^2+\|r\|_H^2 \ge \|s\|_H^2,
\quad \text{with equality iff } r=0.
$$

Since $\Omega$ is strictly increasing, $\Omega(\|f\|_H)\ge \Omega(\|s\|_H)$, with equality iff $r=0$.

Hence for any $f=s+r$,

$$
\mathcal{J}(f)=L(s(x_1),\dots,s(x_n))+\Omega(\|f\|_H)
\ge L(s(x_1),\dots,s(x_n))+\Omega(\|s\|_H)=\mathcal{J}(s),
$$

and strict inequality holds if $r\ne 0$. Therefore any minimizer must satisfy $r=0$, i.e., must lie in $S$. Thus $f^{\ast}\in S$, so $f^{\ast}=\sum_{i=1}^n \alpha_i k_{x_i}$. $\square$

</div>

<div class="ml-box">

**Machine-learning translation.**

This is the mathematically correct version of "weights are shadows of data." The model does not emerge from a vacuum; under Hilbert/RKHS geometry, the optimizer selects a solution in the span generated by the data's representers. Random initialization is a numerical device; the solution space is, structurally, data-spanned.

**Defect in common understanding.**

Many accounts treat the representer theorem as "kernel trick folklore." The real content is geometric: the orthogonal complement is invisible to the data term but costly to the norm term, so it is eliminated. This is a precise, non-metaphorical mechanism for how regularization collapses infinite-dimensional problems into finite-dimensional spans.

</div>

---

### 6.4.2 Frontier reading: memory, generalization, and interpolation

In modern overparameterized models (including large language models), empirical risk often reaches near-zero while generalization remains nontrivial. The representer mechanism suggests a clean separation:

- The *data-visible* subspace governs fitting (memorization capacity).
- The *penalty geometry* governs which interpolant is selected (generalization bias).

Even when the model is not explicitly an RKHS, many training regimes behave as if an implicit Hilbert geometry is being imposed (by initialization, optimizer preconditioning, normalization, or architectural constraints). The empirical phenomenon "implicit bias" is, at its core, a Riesz/representer phenomenon: a hidden geometry converts functionals into vectors and selects solutions by norm.

---

## 6.5 Operators and Duality: Why Naïve Gradient Descent is Geometrically Non-Natural

### 6.5.1 The type error that is usually hidden

Let $F$ be a functional on a parameter space $H$. Its differential at $\theta$ is a bounded linear functional

$$
DF(\theta): H \to \mathbb{R}, \qquad h\mapsto DF(\theta)[h].
$$

Thus $DF(\theta)\in H^{\ast}$, not in $H$. Subtracting "the gradient" from $\theta$ requires a map $H^{\ast}\to H$. In Hilbert spaces, Riesz provides exactly this map.

<div class="def-box">

**Definition 6.4 (Riesz map).** $R:H\to H^{\ast}$, $R(w)=\langle\,\cdot\,,w\rangle$. Its inverse $R^{-1}:H^{\ast}\to H$ converts functionals into vectors.

</div>

Then the gradient $\nabla F(\theta)\in H$ is defined by

$$
DF(\theta)[h] = \langle h, \nabla F(\theta)\rangle
\quad \text{for all } h\in H,
$$

i.e., $\nabla F(\theta)=R^{-1}(DF(\theta))$.

<div class="ml-box">

**Defect in common understanding.**

Many ML derivations write $\nabla F(\theta)$ as if it existed independently of geometry. In fact, it exists as a canonical vector only after choosing a Hilbert inner product, and changing the inner product changes the gradient vector (even when the functional is the same).

</div>

---

### 6.5.2 Natural gradient as the correct statement about the Riesz map

Suppose we choose an inner product induced by a positive-definite operator $G(\theta)$:

$$
\langle a,b\rangle_{G(\theta)} := \langle a,\, G(\theta)b\rangle,
$$

where the right-hand side uses a fixed reference inner product. The Riesz map becomes $R_{G(\theta)}(v)=\langle\,\cdot\,,v\rangle_{G(\theta)}$, which corresponds (in coordinates) to multiplication by $G(\theta)$. Therefore

$$
\nabla_{G} F(\theta) = R_{G(\theta)}^{-1}(DF(\theta)) = G(\theta)^{-1}\nabla F(\theta),
$$

where $\nabla F(\theta)$ is the reference gradient. This is exactly the "preconditioned" direction.

<div class="ml-box">

**Machine-learning translation.**

Natural gradient, and more generally metric-aware optimization, is the explicit recognition that the Riesz identification must match the intrinsic geometry of the model manifold (or the information geometry of the predictive distribution). Euclidean gradient descent is "non-natural" precisely because it fixes a geometry that is typically not intrinsic to the learning problem.

</div>

---

## 6.6 The Banach Boundary: What Breaks When We Leave Hilbert Geometry

### 6.6.1 What survives: Hahn&ndash;Banach versus Riesz

In a general Banach space $X$, the dual $X^{\ast}$ is still rich, and Hahn&ndash;Banach guarantees extension and existence of separating functionals. But there is no canonical isometric isomorphism $X\simeq X^{\ast}$ without additional structure. Thus:

- "rules exist" (many functionals), but
- "rules have unique vector bodies" (Riesz) generally fails.

This clarifies the earlier warning: under sharp constraints (e.g., $\ell_1$, $\ell_\infty$), the learning geometry is Banach-like, and the comfortable identification "dual = primal" dissolves. One must then track dual objects explicitly or introduce a chosen mirror map (a deliberate identification) rather than relying on Hilbert symmetry.

---

### 6.6.2 Practical ML consequence: why non-smooth constraint regimes demand dual-aware algorithms

Sparse optimization, robust optimization, and constrained learning naturally live in Banach geometries with non-smooth unit balls. In such settings, gradients are dual objects, and primal updates require a chosen dual-to-primal map (proximal operators, mirror descent, Fenchel duality). The "vector gradient" is not canonical; it is constructed.

This is the structural reason why proximal methods, mirror descent, and dual averaging are not peripheral: they are the mathematically correct response to the loss of a Riesz identification.

---

<div class="scholium-box">

## 6.7 Summary: Riesz as the Contract of Substantiation

1. **Substantiation of the observer.** Continuous measurement rules become unique vectors; "process" collapses into "state."
2. **Continuity as learnability boundary.** Non-continuous rules have no representer; instability and divergence often signal an attempt to learn outside the representable dual.
3. **Weights as data shadows.** Under Hilbert/RKHS geometry, optimal solutions live in data-generated spans (representer theorem).
4. **Gradient descent is geometry-dependent.** The gradient is the Riesz representer of the differential; changing the inner product changes the gradient vector. Natural gradient is the principled correction.
5. **Beyond Hilbert.** In Banach regimes, the canonical identification fails; dual-aware algorithms are not optional.

</div>

---

## Transition to Chapter 7 (RKHS): The Algebra of Kernels as Infinite-Dimensional Substantiation

Riesz tells us that continuous measurements become vectors once a Hilbert geometry is specified. RKHS theory specifies a particularly powerful geometry: evaluation becomes continuous, hence representable. Kernels therefore engineer representability: they choose a Hilbert space in which "seeing at a point" becomes a bounded measurement. This is the correct mathematical explanation for why kernels convert local constraints into global, optimizable structure&mdash;an "alchemy" not of tricks, but of geometry.

*Next: [Chapter 7: RKHS and the Kernel Trick](/book/chapters/chapter007/)*
