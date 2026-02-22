---
title: "Chapter 7: Reproducing Kernel Hilbert Spaces"
layout: "single"
url: "/book/chapters/chapter007/"
summary: "RKHS as the corrective to an epistemic mismatch: why pointwise observation must be continuous, how Riesz forces reproducing kernels, and how Moore–Aronszajn, Mercer spectra, and the representer theorem govern modern kernel and deep learning."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 7
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

## Chapter 7 &mdash; Reproducing Kernel Hilbert Spaces: Discrete Experience as a Continuous Constraint

*Xujiang Tang*

</div>

## Abstract

An RKHS is not a computational trick; it is a corrective to an epistemic mismatch. Machine learning starts from finitely many pointwise observations $\{(x_i,y_i)\}_{i=1}^n$ yet aims to infer a function on a continuum. The central question is therefore not "how to compute inner products cheaply," but: **under what topology does pointwise observation become a stable, continuous constraint on an infinite-dimensional hypothesis?**

This chapter proceeds by necessity: (i) we begin with the failure mode in standard Hilbert spaces such as $L^2$, where point evaluation is not continuous and empirical constraints are topologically illegitimate; (ii) we elevate "evaluation must be bounded" to an axiom motivated by learning objectives; (iii) Riesz then forces the existence of representers $k_x$, which produce the reproducing property; (iv) Moore&ndash;Aronszajn guarantees that specifying a positive definite kernel is exactly specifying such a geometry; (v) Mercer spectral theory explains why infinite-dimensionality does not imply uncontrolled complexity; (vi) the representer theorem is derived as a projection principle for regularized empirical risk; and (vii) modern ML phenomena&mdash;kernel ridge regression, SVM margins, Gaussian processes, NTK limits, random features, and benign/non-benign interpolation&mdash;are shown to be corollaries of the same geometry.

---

## 7.1 The ML Pain Point That Forces RKHS: Why "Data at Points" Must Be Continuous

### 7.1.1 Where learning actually begins

Supervised learning poses constraints of the form

$$
f(x_i)\approx y_i,\quad i=1,\dots,n,
$$

and optimizes a functional such as

$$
\min_{f\in \mathcal{F}}\ \frac{1}{n}\sum_{i=1}^n \ell\big(f(x_i),y_i\big) + \lambda \, \Omega(f).
$$

This template is universal: ridge regression, SVM, logistic regression in feature space, kernel methods, and many function-space viewpoints of deep learning all fit it.

The silent assumption is that "$f\mapsto f(x_i)$" is a meaningful and stable functional on $\mathcal{F}$. This is exactly where naive function spaces break.

---

### 7.1.2 The failure mode (example): $L^2$ makes pointwise experience topologically invisible

Let $H=L^2([0,1])$. Elements of $H$ are equivalence classes under equality almost everywhere; pointwise evaluation is not even canonical. Worse, even if representatives are chosen, the evaluation functional $\delta_{x_0}(f)=f(x_0)$ is not continuous.

<div class="prop-box">

**Proposition 7.1 (Unbounded evaluation in $L^2$).**

Fix $x_0\in[0,1]$. Define

$$
f_n(t)=
\begin{cases}
\sqrt{n}, & t\in[x_0,x_0+1/n]\cap[0,1], \\\\
0, & \text{otherwise}.
\end{cases}
$$

Then $\|f_n\|_2=1$ for all $n$ but $f_n(x_0)=\sqrt{n}\to\infty$. Hence $\delta_{x_0}$ is unbounded and not continuous.

</div>

<div class="proof-box">

*Proof.*

$$
\|f_n\|_2^2=\int_0^1 |f_n(t)|^2\,dt=\int_{x_0}^{x_0+1/n} n\,dt =1.
$$

But $\delta_{x_0}(f_n)=\sqrt{n}$. If $\delta_{x_0}$ were bounded, $|\delta_{x_0}(f_n)|\le C\|f_n\|_2=C$ would hold, contradiction. $\square$

</div>

<div class="ml-box">

**Machine-learning interpretation (natural).**

Empirical risk depends on $\{f(x_i)\}$. If evaluation is not continuous, then infinitesimally small perturbations in the hypothesis norm can cause arbitrarily large changes in training predictions at a point&mdash;making "generalization by stability" impossible to even state cleanly. In short: **$L^2$ is a Hilbert space, but it is not a learning space for pointwise data.**

This is the precise mathematical reason we must *design* the function space so that pointwise experience has continuity.

</div>

---

### 7.1.3 The learning axiom (stated as a necessity, not a definition)

We now impose the minimal condition that makes empirical risk well-posed in a Hilbert geometry:

<div class="def-box">

**Axiom (Empirical legitimacy).** For each $x\in\mathcal{X}$, the evaluation functional $\delta_x:f\mapsto f(x)$ is bounded (continuous) on $\mathcal{H}$.

</div>

This is exactly the RKHS axiom. It is forced by the learning objective, not chosen by taste.

---

## 7.2 Riesz Forces Reproduction: "Points" Become Vectors

### 7.2.1 From the axiom to representers (the logical step)

Let $\mathcal{H}$ be a Hilbert space of functions on $\mathcal{X}$ satisfying bounded evaluation. Then $\delta_x\in\mathcal{H}^{\ast}$. By Riesz (Chapter 6), there exists a unique $k_x\in\mathcal{H}$ such that

$$
f(x)=\delta_x(f)=\langle f,k_x\rangle_{\mathcal{H}}.
$$

Define the kernel

$$
k(x,y):=k_y(x)=\langle k_y,k_x\rangle_{\mathcal{H}}.
$$

This yields the reproducing property:

$$
f(x)=\langle f,k(\cdot,x)\rangle_{\mathcal{H}}.
$$

<div class="ml-box">

**Machine-learning example 1 (linear models recovered).**

If $\mathcal{H}=\mathbb{R}^d$ with $\langle u,v\rangle=u^\top v$, then evaluation at $x$ for linear functionals $f_w(\cdot)=w^\top(\cdot)$ corresponds to $k(x,y)=x^\top y$. Thus "ordinary linear regression" is already RKHS regression in the linear kernel geometry.

**Machine-learning example 2 (why kernel values are similarities).**

Since $k(x,y)=\langle k_x,k_y\rangle$, the kernel is literally an inner product between the *representers of observation at points*. Similarity is not a heuristic; it is a geometric statement about how two point-constraints correlate in the hypothesis space.

</div>

---

## 7.3 Moore&ndash;Aronszajn: A Kernel Is a Geometry, Not a Trick (Full Construction + ML Interpretation)

### 7.3.1 Why positive definiteness is the right admissibility condition

In learning we inevitably form Gram matrices $K_{ij}=k(x_i,x_j)$. For stability and convexity, we require $K\succeq 0$ for every finite sample set. This is exactly positive definiteness.

<div class="def-box">

**Definition 7.2 (Positive definite kernel).**

$k$ is positive definite if for any $x_1,\dots,x_n$ and $c\in\mathbb{R}^n$,

$$
c^\top K c = \sum_{i,j} c_i c_j k(x_i,x_j)\ge 0.
$$

</div>

This is the condition ensuring that the norm induced by $k$ is nonnegative and learning objectives remain convex (e.g., kernel ridge regression).

---

### 7.3.2 The theorem (and the construction)

<div class="prop-box">

**Theorem 7.3 (Moore&ndash;Aronszajn).** Every positive definite kernel $k$ generates a unique RKHS $\mathcal{H}_k$ with reproducing kernel $k$.

</div>

<div class="proof-box">

*Proof sketch with all essential steps (kept linear and natural).*

1. Let $\mathcal{H}_0 = \mathrm{span}\{k(\cdot,x): x\in\mathcal{X}\}$.
2. Define $\langle \sum_i \alpha_i k(\cdot,x_i), \sum_j \beta_j k(\cdot,y_j)\rangle_0 = \sum_{i,j}\alpha_i\beta_j k(x_i,y_j)$.
3. Mod out the null space (functions with zero seminorm).
4. Complete the resulting inner-product space to obtain a Hilbert space $\mathcal{H}_k$.
5. Verify reproduction on the dense span and extend by continuity. $\square$

</div>

<div class="ml-box">

**Machine-learning example 3 (why Gram matrices are inevitable).**

On data $\{x_i\}$, all computations in $\mathcal{H}_k$ reduce to the Gram matrix $K$, because inner products among representers are exactly $k(x_i,x_j)$. The "kernel trick" is simply the computational shadow of this construction.

</div>

---

## 7.4 Mercer Spectra: Why Infinite Dimensionality Does Not Mean Uncontrolled Fitting

### 7.4.1 The ML question

How can a hypothesis space with infinitely many degrees of freedom avoid memorizing arbitrary noise?

The answer is: **the RKHS norm is not a generic size measure; it is a spectral energy functional determined by the kernel.**

---

### 7.4.2 Mercer expansion and spectral hierarchy

Under standard compactness/continuity assumptions, the kernel induces a compact positive integral operator $T_k$ on $L^2(\mu)$ with eigenpairs $(\lambda_j,\phi_j)$, and

$$
k(x,y)=\sum_{j=1}^\infty \lambda_j \phi_j(x)\phi_j(y).
$$

Any $f\in\mathcal{H}_k$ has expansion $f=\sum_j a_j \phi_j$ with norm

$$
\|f\|_{\mathcal{H}_k}^2 = \sum_{j=1}^\infty \frac{a_j^2}{\lambda_j}.
$$

Small $\lambda_j$ modes are expensive. This is a precise smoothness bias.

<div class="ml-box">

**Machine-learning example 4 (Gaussian/RBF kernel as a high-frequency suppressor).**

For RBF kernels, eigenvalues decay quickly; high-frequency components correspond to very small $\lambda_j$, so the RKHS norm strongly penalizes oscillatory functions. This explains why kernel ridge regression often behaves as "interpolation with smoothness," not arbitrary memorization.

**Machine-learning example 5 (why kernels can fail on images without structure).**

An isotropic kernel on raw pixels does not encode translational locality; it assigns similar energy penalties to spatially meaningful and spatially meaningless directions. The failure is spectral misalignment: large-eigenvalue directions do not coincide with task-relevant invariances. Deep nets succeed largely because learned features reshape the spectrum so that high-eigenvalue directions match the data manifold's invariances.

</div>

---

## 7.5 Representer Theorem as a Projection Law (and How It Explains SVM/KRR Naturally)

### 7.5.1 The variational problem (put first, as in ML)

Consider the canonical regularized risk:

$$
\min_{f\in\mathcal{H}_k} \ \frac{1}{n}\sum_{i=1}^n \ell\big(f(x_i),y_i\big)+\lambda\|f\|_{\mathcal{H}_k}^2.
$$

### 7.5.2 The theorem (then the proof)

<div class="prop-box">

**Theorem 7.4 (Representer theorem).** Every minimizer $f^{\ast}$ satisfies

$$
f^{\ast}(\cdot)=\sum_{i=1}^n \alpha_i k(\cdot,x_i).
$$

</div>

<div class="proof-box">

*Proof (projection geometry, minimal steps, no gaps).*

Let $S=\mathrm{span}\{k_{x_1},\dots,k_{x_n}\}$. Decompose any $f$ as $f=s+r$ with $s\in S$, $r\in S^\perp$.

Reproducing property gives for each $i$:

$$
f(x_i)=\langle f,k_{x_i}\rangle=\langle s,k_{x_i}\rangle+\langle r,k_{x_i}\rangle
=\langle s,k_{x_i}\rangle = s(x_i),
$$

since $k_{x_i}\in S$ and $r\perp S$. Therefore the empirical loss depends only on $s$.

Meanwhile $\|f\|^2=\|s\|^2+\|r\|^2\ge \|s\|^2$, with strict inequality unless $r=0$.

Thus any minimizer must have $r=0$, hence $f^{\ast}\in S$, i.e. a finite kernel expansion. $\square$

</div>

<div class="ml-box">

**Machine-learning example 6 (KRR closed form).**

With squared loss, one obtains $(K+n\lambda I)\alpha=y$ and $f^{\ast}(x)=k_x^\top (K+n\lambda I)^{-1}y$. This is the natural output of the projection argument, not a separate "kernel trick."

**Machine-learning example 7 (SVM margin geometry).**

In hard-margin SVM within an RKHS, one minimizes $\|f\|_{\mathcal{H}_k}$ subject to $y_i f(x_i)\ge 1$. The RKHS norm is the inverse margin: minimizing $\|f\|$ maximizes the margin. This is the correct functional-analytic version of "large margin" and makes explicit that the margin is geometry-dependent.

</div>

---

## 7.6 Beyond Classical Kernels: Modern ML Regimes Where the Kernel Evolves

### 7.6.1 NTK: when deep learning reduces to a fixed RKHS

In the infinite-width "lazy training" regime, gradient descent on a neural network induces a fixed kernel (NTK) and training becomes kernel regression in the corresponding RKHS. This is a genuine theorem-level bridge between deep nets and kernel methods.

<div class="ml-box">

**Machine-learning example 8 (practical interpretation).**

This explains why very wide networks can train stably and predict smoothly even with near interpolation: the implicit bias is minimum RKHS norm under the NTK geometry.

</div>

### 7.6.2 The critical defect in common understanding: "deep learning = kernel method" is false in feature-learning regimes

In many successful modern systems, the representation changes significantly during training; empirically, learned features reorganize similarity. That is precisely the regime where the effective kernel is not static. Mathematically: the map $\theta\mapsto k_\theta(x,x')$ evolves, hence the RKHS itself evolves. The correct theory must therefore track *geometry learning*, not only function learning.

<div class="ml-box">

**Machine-learning example 9 (contrastive learning as geometry learning).**

Contrastive objectives reshape embedding geometry by increasing similarity for positives and decreasing it for negatives. This is effectively learning a data-dependent kernel on representations. The "kernel trick" viewpoint is inadequate unless it is upgraded to "kernel evolution under optimization."

</div>

---

## 7.7 Frontier Problems: Interpolation, Benign Overfitting, and the Role of the RKHS Norm

### 7.7.1 Minimum-norm interpolation is the right object, not "overfitting vs underfitting"

When $\lambda\to 0$ and the kernel is sufficiently rich, KRR approaches minimum-norm interpolation:

$$
\min_{f\in\mathcal{H}_k} \|f\|_{\mathcal{H}_k}
\quad\text{s.t.}\quad f(x_i)=y_i.
$$

Whether this generalizes depends on how $\|f\|_{\mathcal{H}_k}$ scales with $n$, on eigen-decay $\{\lambda_j\}$, and on noise geometry. The naive claim "interpolation implies overfitting" is not a theorem; RKHS theory clarifies the actual invariants.

<div class="ml-box">

**Machine-learning example 10 (why some kernels interpolate poorly).**

If the kernel's spectrum does not provide sufficient penalty to noise-aligned directions (slow eigen-decay or spectral mismatch to the data distribution), minimum-norm interpolants can exhibit poor generalization. Conversely, in high-dimensional regimes with favorable eigen-structure, interpolation can be benign.

</div>

---

<div class="scholium-box">

## 7.8 Summary (with the logic now explicitly ML-natural)

1. ML begins with point constraints $f(x_i)\approx y_i$. For these to define a well-posed optimization problem in a Hilbert geometry, evaluation must be continuous.
2. $L^2$ fails this requirement: point evaluation is not continuous, hence empirical constraints are topologically illegitimate.
3. Enforcing bounded evaluation yields an RKHS. Riesz then forces representers $k_x$, producing the reproducing property and a kernel-defined geometry.
4. Moore&ndash;Aronszajn shows that a positive definite kernel is exactly the specification of such a space; it is not a computational shortcut, it is a geometric choice.
5. Mercer's theorem explains how infinite-dimensional spaces are spectrally organized; the RKHS norm penalizes high-complexity modes.
6. The representer theorem is a projection law explaining why regularized ML solutions collapse to data spans, yielding KRR/SVM in a logically unavoidable way.
7. Modern deep learning connects to RKHS through NTK in lazy regimes; the frontier is feature learning, where the effective kernel (and thus the RKHS geometry) evolves.

</div>

---

## Transition to Chapter 8 (Matrix Calculus): From Existence to Dynamics

RKHS theory tells us what solutions must look like and why they are well-posed. To understand how learning trajectories *move*&mdash;in kernels, in neural nets, and in hybrid systems&mdash;we need a calculus that differentiates objectives with respect to matrices and operators without collapsing into coordinate bookkeeping. Chapter 8 develops matrix differentials as the correct language of backpropagation, and exposes precisely where and how "geometry" enters the chain rule in modern ML.
