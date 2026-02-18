---
title: "Chapter 1: Vector Spaces and Dual Spaces"
layout: "single"
url: "/book/chapters/chapter001/"
summary: "The minimum algebraic structure underlying all of machine learning."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
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

# Volume I &mdash; The Mathematical Principles of Machine Learning

## Chapter 1 &mdash; Vector Spaces and Dual Spaces

*Xujiang Tang*

</div>

## Abstract

Modern machine learning is an engineered theory of *iterated perturbations*: parameters, representations, and updates are repeatedly added, scaled, aggregated, and compared. The purpose of this chapter is to pin down the minimum algebraic structure that makes those operations well-defined and stable: **vector spaces** for states and increments, and **dual spaces** for measurement, loss, and gradient information.

The chapter then adds the machine-learning layer: why distributed training forces commutative addition; why learning-rate semantics are exactly scalar action; why expectations, mini-batches, and convex combinations presuppose linearity; why backpropagation is covector transport through dual maps; why gradients are canonically elements of the dual; and why any practical "move along the gradient" step quietly inserts a metric (explicitly in natural gradient and implicitly in Adam-like preconditioners).

The end result is a clean type discipline: *states live in primal spaces, measurements live in dual spaces, and optimization requires a principled bridge between them*.

## Notation and Conventions

- $\mathbb{F}$ denotes a field, typically $\mathbb{R}$.
- $V, W$ denote vector spaces over $\mathbb{F}$.
- $V^{\ast}$ denotes the **dual space** of $V$: the space of linear maps $V \to \mathbb{F}$.
- A *vector* means an element of $V$. A *covector* means an element of $V^{\ast}$.
- For finite-dimensional $V$, coordinates are written with **upper indices** for vectors $v^i$ and **lower indices** for covectors $\varphi_i$, reflecting their different transformation laws.
- An inner product is **not** assumed unless explicitly stated. When introduced, it is additional structure, not part of the definition of a vector space.

## 1.1 Vector Spaces as the Algebra of Learnable State

### 1.1.1 Learning as a calculus of increments

A training rule has the schematic form

$$\theta_{t+1} = \theta_t \oplus \Delta_t,$$

where $\theta_t$ is a state (parameters, features, latent variables) and $\Delta_t$ is an update. To make such an update rule meaningful, $\Delta_t$ must support:

1. **Superposition** of contributions (e.g., gradients from different samples, devices, or layers), and
2. **Rescaling** by a step size (learning rate, schedule, or trust-region scalar).

If the space of increments does not support these two operations consistently, "mini-batching," "gradient accumulation," "learning rate," and "distributed reduction" cannot be mathematically coherent. This motivates the axioms of a vector space.

### 1.1.2 Definition: field and vector space

<div class="def-box">

**Definition 1.1 (Field).** A field $\mathbb{F}$ is a set with addition and multiplication such that $(\mathbb{F},+)$ is an Abelian group, $(\mathbb{F}\setminus\lbrace 0\rbrace,\cdot)$ is an Abelian group, and multiplication distributes over addition.

</div>

<div class="def-box">

**Definition 1.2 (Vector Space).** A vector space over $\mathbb{F}$ is a set $V$ equipped with:

- addition $+: V \times V \to V$,
- scalar multiplication $\cdot : \mathbb{F} \times V \to V$,

satisfying for all $u, v, w \in V$, $\alpha, \beta \in \mathbb{F}$:

| Axiom | Statement | Name |
|:-----:|:----------|:-----|
| (A1) | $u + v \in V$ | Closure under addition |
| (A2) | $u + v = v + u$ | Commutativity |
| (A3) | $(u + v) + w = u + (v + w)$ | Associativity |
| (A4) | $\exists\, 0 \in V$ such that $u + 0 = u$ | Additive identity |
| (A5) | $\forall\, u \in V,\ \exists\, (-u) \in V$ such that $u + (-u) = 0$ | Additive inverse |
| (A6) | $\alpha u \in V$ | Closure under scalar multiplication |
| (A7) | $\alpha(u + v) = \alpha u + \alpha v$ | Distributivity over vector addition |
| (A8) | $(\alpha + \beta)u = \alpha u + \beta u$, $(\alpha\beta)u = \alpha(\beta u)$, $1u = u$ | Distributivity, compatibility, identity |

This is the *minimal algebra* underlying every linearization used in ML.

</div>

### 1.1.3 Uniqueness facts that ML silently relies on

<div class="prop-box">

**Lemma 1.3 (Uniqueness of additive identity).** The element $0 \in V$ is unique.

</div>

<div class="proof-box">

*Proof.* Suppose $0$ and $0'$ are both additive identities. Then

$$0 = 0 + 0' = 0'.$$

Hence $0 = 0'$. $\square$

</div>

<div class="prop-box">

**Lemma 1.4 (Uniqueness of additive inverse).** For each $u \in V$, the inverse $-u$ is unique.

</div>

<div class="proof-box">

*Proof.* Suppose $v$ and $w$ both satisfy $u + v = 0$ and $u + w = 0$. Then

$$\begin{aligned} v &= v + 0 \\\\  &= v + (u + w) \\\\  &= (v + u) + w  &\quad&\text{by (A3)} \\\\  &= (u + v) + w  &\quad&\text{by (A2)} \\\\  &= 0 + w = w. \end{aligned}$$

Thus $v = w$. $\square$

</div>

<div class="ml-box">

**ML Interpretation.** "No update," "cancelation," and "exact undoing" are not vague metaphors; they require unique identity and unique inverses to make update arithmetic unambiguous &mdash; especially when debugging or reasoning about optimizer algebra.

</div>

### 1.1.4 Distributed training: why commutativity and associativity are non-negotiable

Let $g_1, \dots, g_n \in V$ be increments computed on different shards (workers, microbatches). A distributed system aggregates them via a reduction tree and network-dependent arrival order. To define a training rule independent of system nondeterminism, the sum must be order-invariant.

<div class="prop-box">

**Proposition 1.5 (Order invariance of aggregation).** For any permutation $\pi$ of $\lbrace 1, \dots, n\rbrace$,

$$g_1 + g_2 + \cdots + g_n = g_{\pi(1)} + g_{\pi(2)} + \cdots + g_{\pi(n)}.$$

</div>

<div class="proof-box">

*Proof.* Any permutation is a product of adjacent transpositions. It suffices to show that swapping adjacent terms does not change the sum. For any $a, b \in V$,

$$a + b = b + a \quad \text{by (A2).}$$

Associativity (A3) ensures that inserting parentheses in repeated sums does not change the value, so adjacent swaps can be performed inside any longer sum without ambiguity. Hence every permutation yields the same result. $\square$

</div>

<div class="prop-box">

**Corollary 1.6 (Well-defined gradient reduction).** The aggregated gradient $\sum_i g_i$ computed by *any* reduction topology is identical, provided gradients live in a vector space.

</div>

<div class="ml-box">

**ML Interpretation.** This is the mathematical justification behind all-reduce determinism: the training rule depends on the algebra, not on the communication schedule.

</div>

### 1.1.5 Scalar multiplication is exactly learning-rate semantics

A learning rate $\eta$ is meaningful only if scaling commutes with aggregation. Concretely, the requirement is:

$$\eta\left(\sum_{i=1}^n g_i\right) = \sum_{i=1}^n \eta\, g_i.$$

<div class="prop-box">

**Proposition 1.7 (Scaling commutes with summation).** For any $\eta \in \mathbb{F}$ and $g_1, \dots, g_n \in V$,

$$\eta\left(\sum_{i=1}^n g_i\right) = \sum_{i=1}^n \eta\, g_i.$$

</div>

<div class="proof-box">

*Proof (by induction, no steps omitted).*

**Base case** ($n = 1$). Both sides equal $\eta\, g_1$.

**Inductive step.** Assume the statement holds for $n$. Then for $n + 1$:

$$\begin{aligned} \eta\left(\sum_{i=1}^{n+1} g_i\right) &= \eta\left(\sum_{i=1}^{n} g_i + g_{n+1}\right) \\\\  &= \eta\left(\sum_{i=1}^{n} g_i\right) + \eta\, g_{n+1}  &\quad&\text{by (A7)} \\\\  &= \left(\sum_{i=1}^{n} \eta\, g_i\right) + \eta\, g_{n+1}  &\quad&\text{by induction hypothesis} \\\\  &= \sum_{i=1}^{n+1} \eta\, g_i. \end{aligned}$$

Thus the claim holds for all $n \geq 1$. $\square$

</div>

<div class="ml-box">

**ML Interpretation.** Mini-batch SGD with step size $\eta$ can be implemented as "sum then scale" or "scale then sum"; they are identical *only because* scalar distributivity holds.

</div>

### 1.1.6 Expectation, mini-batching, and convex combinations are linearity statements

SGD is built on the idea that a batch gradient approximates an expectation:

$$\nabla L(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\bigl[\nabla \ell(\theta; x)\bigr].$$

This assumes the ability to form linear combinations of gradients and that expectations distribute over addition and scalar multiplication.

Given samples $x_1, \dots, x_m$ and per-sample gradients $g_k = \nabla \ell(\theta; x_k) \in V$, the empirical mean gradient is:

$$\bar{g} = \frac{1}{m}\sum_{k=1}^m g_k.$$

This expression only makes sense because:

- the sum $\sum_k g_k$ is defined in $V$, and
- scaling by $\frac{1}{m} \in \mathbb{F}$ is defined in $V$.

Similarly, label smoothing, mixup, and many data-augmentation objectives are convex mixtures:

$$\tilde{y} = \lambda\, y_1 + (1 - \lambda)\, y_2,$$

which presuppose linear structure in the output/target representation space.

<div class="ml-box">

**ML Interpretation.** The statistical machinery of learning (expectations, averages, interpolation) is algebraic before it is probabilistic: it requires a vector-space structure.

</div>

### 1.1.7 Basis, dimension, and embeddings: what representation learning actually chooses

If $V$ is finite-dimensional with basis $\lbrace e_1, \dots, e_n\rbrace$, each $v \in V$ has unique coordinates $v^i$ such that

$$v = \sum_{i=1}^n v^i\, e_i.$$

<div class="ml-box">

**ML Interpretation.** Learning an embedding can be read as learning a map $E: \mathcal{X} \to V$ and often also learning a *useful coordinate system* for downstream tasks. Linear probes and linear heads test whether task-relevant information is linearly accessible in those coordinates.

</div>

A linear layer is a linear map $T: V \to W$. In coordinates it is a matrix; but the intrinsic object is the map $T$, independent of the chosen basis. This distinction becomes operational when discussing invariances, conditioning, and reparameterization.

### 1.1.8 Linear maps are the primitives of neural layers

A linear layer is $T(v) = Av$ (coordinate form). A typical network composes affine maps and nonlinearities:

$$v \mapsto \sigma(Av + b).$$

Even for nonlinear layers, training uses local linearization: the derivative $D\sigma$ and Jacobians of compositions. Hence linear algebra is not restricted to linear models; it is the local language of *all* differentiable computation graphs.

## 1.2 Dual Spaces as the Algebra of Measurement, Loss, and Gradient

### 1.2.1 Definition: dual space and canonical pairing

A model outputs scalars (scores, energies, losses) from representations. The minimal algebra consistent with superposition is linear evaluation.

<div class="def-box">

**Definition 1.8 (Dual space).**

$$V^{\ast} := \bigl\lbrace\varphi: V \to \mathbb{F}\ \big\vert\ \varphi(u + v) = \varphi(u) + \varphi(v),\quad \varphi(\alpha u) = \alpha\,\varphi(u)\bigr\rbrace.$$

</div>

<div class="def-box">

**Definition 1.9 (Canonical pairing).** For $\varphi \in V^{\ast}$ and $v \in V$,

$$\langle \varphi,\, v \rangle := \varphi(v) \in \mathbb{F}.$$

This pairing needs no metric and no coordinates.

</div>

<div class="ml-box">

**ML Interpretation.** A linear classifier is exactly a covector $\varphi \in V^{\ast}$ acting on an embedding $x \in V$, producing a scalar logit $\varphi(x)$.

</div>

### 1.2.2 The dual is a vector space

Define $(\varphi + \psi)(v) := \varphi(v) + \psi(v)$ and $(\alpha\varphi)(v) := \alpha\,\varphi(v)$.

<div class="prop-box">

**Proposition 1.10.** $V^{\ast}$ is a vector space over $\mathbb{F}$.

</div>

<div class="proof-box">

*Proof.* Closure and linearity of these operations are verified as follows. Let $\varphi, \psi \in V^{\ast}$. For all $u, v \in V$:

$$(\varphi + \psi)(u + v) = \varphi(u + v) + \psi(u + v) = \bigl(\varphi(u) + \varphi(v)\bigr) + \bigl(\psi(u) + \psi(v)\bigr) = (\varphi + \psi)(u) + (\varphi + \psi)(v).$$

Also, for $\alpha \in \mathbb{F}$:

$$(\varphi + \psi)(\alpha u) = \varphi(\alpha u) + \psi(\alpha u) = \alpha\,\varphi(u) + \alpha\,\psi(u) = \alpha\,(\varphi + \psi)(u).$$

Thus $\varphi + \psi \in V^{\ast}$. Similar verification shows $\alpha\varphi \in V^{\ast}$. Remaining axioms hold pointwise because they hold in $\mathbb{F}$. $\square$

</div>

<div class="ml-box">

**ML Interpretation.** Averaging heads, ensembling logits, interpolating linear probes: all remain in $V^{\ast}$. This is why logit averaging is algebraically clean.

</div>

### 1.2.3 Dual basis and coordinate invariance (covariance vs. contravariance)

Let $\lbrace e_1, \dots, e_n\rbrace$ be a basis of $V$. Define $\lbrace\varepsilon^1, \dots, \varepsilon^n\rbrace \subset V^{\ast}$ by

$$\varepsilon^i(e_j) = \delta^i_{\ j}.$$

<div class="prop-box">

**Proposition 1.11 (Existence and uniqueness of the dual basis).** Such $\lbrace\varepsilon^i\rbrace$ exists, is unique, and forms a basis of $V^{\ast}$.

</div>

<div class="proof-box">

*Proof.* Define $\varepsilon^i$ on basis elements by the above rule and extend linearly: for $v = \sum_j v^j\, e_j$,

$$\varepsilon^i(v) = \varepsilon^i\!\left(\sum_{j=1}^n v^j\, e_j\right) = \sum_{j=1}^n v^j\,\varepsilon^i(e_j) = \sum_{j=1}^n v^j\,\delta^i_{\ j} = v^i.$$

Uniqueness follows because a linear functional is determined by its values on a basis. $\square$

</div>

Now under a change of basis $e'_i = \sum_j A^j_{\ i}\, e_j$ (with $A$ invertible), vector coordinates and covector coordinates transform differently. If $v = \sum_i v^i\, e_i = \sum_i v'^i\, e'_i$, then

$$v^j = \sum_{i=1}^n A^j_{\ i}\, v'^i.$$

In contrast, requiring invariance of the scalar $\varphi(v)$ forces

$$\varphi'_k = \sum_{i=1}^n \varphi_i\, A^i_{\ k}.$$

<div class="ml-box">

**ML Interpretation.** Reparameterizations change gradient coordinates and parameter coordinates differently. Any algorithm that ignores this distinction is not invariant under change of variables.

</div>

### 1.2.4 Backpropagation is covector transport (the chain rule in correct types)

Let $f: V \to W$ and $\ell: W \to \mathbb{F}$. Set $y = f(x)$. Then $Df(x)$ is a linear map $V \to W$, while $D\ell(y)$ is a covector in $W^{\ast}$. The derivative of the composite is

$$D(\ell \circ f)(x) = D\ell(y) \circ Df(x),$$

which is an element of $V^{\ast}$. This is exactly the formal statement of backpropagation: **covectors propagate backward through the dual of the local linearization**.

In coordinates (with Jacobian $J \in \mathbb{R}^{m \times n}$), represent $D\ell(y)$ as a row vector $\frac{\partial \ell}{\partial y} \in \mathbb{R}^{1 \times m}$. For any perturbation $h \in \mathbb{R}^n$:

$$D(\ell \circ f)(x)[h] = \frac{\partial \ell}{\partial y}\,(Jh) = \left(\frac{\partial \ell}{\partial y}\,J\right)h.$$

Therefore the covector (row) representing $\frac{\partial \ell}{\partial x}$ is

$$\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial y}\, J.$$

If one insists on column gradients, this becomes the familiar $J^\top$ rule, but the invariant content is the same: it is dual-map composition.

### 1.2.5 Gradients live in the dual: the foundational type statement

Let $J: V \to \mathbb{F}$ be differentiable at $\theta$. By definition of the Fr&eacute;chet derivative, there exists a linear map $DJ(\theta): V \to \mathbb{F}$ such that

$$\lim_{h \to 0} \frac{J(\theta + h) - J(\theta) - DJ(\theta)[h]}{\lVert h\rVert} = 0.$$

Hence $DJ(\theta) \in V^{\ast}$. This is not optional: it is the definition of derivative in linear-algebraic terms.

<div class="ml-box">

**ML Interpretation.** The gradient is *not* inherently a vector in parameter space. It is a covector (a linear functional on perturbations). Any update rule that subtracts a "gradient vector" is already assuming an identification $V^{\ast} \cong V$, which requires additional structure (a metric).

</div>

### 1.2.6 Turning covectors into vectors: metric transport and natural gradient

To produce an update $\Delta\theta \in V$ from a covector $DJ(\theta) \in V^{\ast}$, one must supply an isomorphism $\sharp_G: V^{\ast} \to V$. The standard mechanism is a positive-definite bilinear form (metric) $G(\theta)$.

Define $\flat_G: V \to V^{\ast}$ by

$$(\flat_G(v))[h] := v^\top G(\theta)\, h.$$

The inverse map $\sharp_G = \flat_G^{-1}$ is sought. Given $DJ(\theta)[h] = \nabla J(\theta)^\top h$ in coordinates, the equality $\flat_G(v) = DJ(\theta)$ means:

$$v^\top G(\theta)\, h = \nabla J(\theta)^\top h \quad \text{for all } h.$$

Therefore the row vectors coincide:

$$v^\top G(\theta) = \nabla J(\theta)^\top.$$

Transposing:

$$G(\theta)^\top v = \nabla J(\theta).$$

If $G(\theta)$ is symmetric, $G(\theta)^\top = G(\theta)$, hence

$$G(\theta)\, v = \nabla J(\theta), \qquad v = G(\theta)^{-1}\, \nabla J(\theta).$$

<div class="def-box">

**Definition 1.12 (Metric gradient).** The metric gradient is

$$\operatorname{grad}_G J(\theta) := G(\theta)^{-1}\, \nabla J(\theta),$$

and the corresponding update rule is

$$\theta_{t+1} = \theta_t - \eta\, G(\theta_t)^{-1}\, \nabla J(\theta_t).$$

</div>

<div class="ml-box">

**ML Interpretation.**

- If $G$ is the **Fisher information metric**, this is the *natural gradient* update (Amari, 1998).
- If $G^{-1}$ is approximated by diagonal or block-diagonal statistics, one obtains the conceptual form of **Adam / RMSProp / Shampoo**: online approximations of dual-to-primal transport.

</div>

### 1.2.7 Optimizers as implicit geometry: SGD, momentum, Adam viewed through the dual&ndash;primal bridge

**1) SGD (Euclidean identification).** Using $G = I$ identifies covectors with vectors via the standard dot product. Then $\operatorname{grad}_G J = \nabla J$, giving the usual update $\theta \leftarrow \theta - \eta\,\nabla J$. This is simple but coordinate-dependent: change variables and the update changes.

**2) Momentum.** Momentum introduces a state variable $v_t \in V$ that integrates past transported gradients:

$$v_{t+1} = \beta\, v_t + (1 - \beta)\, u_t, \qquad \theta_{t+1} = \theta_t - \eta\, v_{t+1},$$

where $u_t$ is the chosen primal update direction obtained from a covector (often via Euclidean identification). The type issue persists: what matters is how covectors are turned into $u_t \in V$.

**3) Adam / RMSProp (diagonal metric approximation).** Adam constructs per-coordinate scaling from second-moment estimates. Interpreted geometrically, it selects an approximate metric $G(\theta)$ whose inverse is diagonal, thus implementing an approximate natural-gradient-like transport:

$$u_t \approx G_t^{-1}\, \nabla J(\theta_t).$$

The algorithmic details differ, but the structural meaning is stable: **adaptive methods modify the covector-to-vector map.**

### 1.2.8 Attention and bilinear forms: where dot products hide a metric

Attention scores often take the form $s = q^\top k$. Strictly typed, one should interpret this as evaluation of a covector on a vector, or as a bilinear form induced by a metric. In practice, projection matrices and normalization layers define an implicit bilinear form that makes this operation numerically stable. The key point is: **dot products are not free; they encode geometric assumptions.**

## 1.3 Summary and the Next Mathematical Layer

<div class="scholium-box">

**Scholium (Summary of Chapter 1).**

This chapter establishes the type system required by machine learning:

1. **Vector spaces** formalize the algebra of state increments: aggregation, averaging, interpolation, and learning-rate scaling.

2. **Dual spaces** formalize the algebra of measurement: scoring, loss evaluation, and the correct mathematical nature of gradients.

3. **Backpropagation** is covector transport through compositions of derivatives.

4. **Optimization** requires a bridge $V^{\ast} \to V$; choosing or approximating a metric is exactly the act of defining this bridge.

</div>

The next layer studies linear operators $T: V \to V$ and $T: V \to W$ and their invariants. In deep learning, stability and trainability are governed by operator spectra: singular values of Jacobians (signal propagation), conditioning of curvature/preconditioners (optimization speed), and spectral radii in recurrent dynamics (explosion/vanishing). Hence the natural continuation is **operator theory and spectral analysis** as the quantitative language of stability and information transport.

<div style="text-align:center; margin-top:3em; color:#888; font-size:0.9em;">

*&mdash; End of Chapter 1 &mdash;*

*Next: [Chapter 2: Spectral Theory &mdash; Eigendecomposition and SVD](/book/chapters/chapter002/)*

</div>
