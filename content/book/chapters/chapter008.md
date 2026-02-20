---
title: "Chapter 8: Matrix Calculus"
layout: "single"
url: "/book/chapters/chapter008/"
summary: "Differentials as the first-order linear operator, gradients as Riesz representers under Frobenius geometry, and backpropagation as systematic pullback of covectors by adjoint Jacobians."
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

# Volume I &mdash; Mathematical Foundations and Axiomatization

## Chapter 8 &mdash; Matrix Calculus: Differentials, Adjoint Geometry, and Reverse-Mode Learning Dynamics

*Xujiang Tang*

</div>

## Abstract

Matrix calculus is often taught as a catalog of identities. For machine learning, this is the wrong entry point. The fundamental object is not a table of partial derivatives but the **differential**: the first-order linear operator that best approximates a nonlinear map at a point. Once one commits to this viewpoint, three facts become unavoidable. First, "a matrix" is a linear map acting on a space, not an array of numbers. Second, a "gradient" is not an intrinsic vector; it is the Riesz representer of a differential under a chosen inner product. Third, backpropagation is not a mnemonic for the chain rule; it is the systematic propagation of covectors by **pullback**, implemented by the **adjoint** of Jacobians.

This chapter develops matrix calculus from these first principles and then derives the standard backpropagation rules as corollaries. Throughout, the objective is to keep the geometry explicit and the algebra minimal but complete.

---

## 8.1 Matrices and Calculus as Geometry: Linearization as the Local Law

### 8.1.1 Matrices are linear maps, not arrays

Let $V,W$ be finite-dimensional real inner-product spaces. A matrix $A\in\mathbb{R}^{m\times n}$ is a coordinate representation of a linear map

$$
A: \mathbb{R}^n \to \mathbb{R}^m,\qquad x\mapsto Ax,
$$

relative to chosen bases. The invariant notion is the linear operator; the array is secondary.

In learning, a "layer" is a map between representation spaces. Treating it as a matrix is a choice of coordinates, convenient but conceptually derivative.

---

### 8.1.2 Calculus is linearization: the differential as the local surrogate

Let $F: \mathbb{R}^d \to \mathbb{R}^p$ be differentiable at $x$. The **(Fr&eacute;chet) differential** $dF_x$ is the unique linear map $dF_x:\mathbb{R}^d\to\mathbb{R}^p$ such that

$$
F(x+\delta x) = F(x) + dF_x[\delta x] + o(\|\delta x\|).
$$

Equivalently, $dF_x$ is the best linear approximation to $F$ at scale $\delta x\to 0$. The Jacobian $J_F(x)$ is a coordinate representation of $dF_x$.

<div class="ml-box">

**Machine-learning translation.** Every training step relies on the claim that locally, the loss landscape is well approximated by a linear functional of the parameter perturbation. The "gradient step" is simply the most informative direction of this local linear model under a metric.

</div>

---

## 8.2 When Matrices Are Variables: Differentials in Matrix Spaces

### 8.2.1 Matrix spaces are inner-product spaces

Let $\mathbb{R}^{m\times n}$ carry the **Frobenius inner product**

$$
\langle A,B\rangle_F := \mathrm{tr}(A^\top B),
\qquad 
\|A\|_F^2 = \mathrm{tr}(A^\top A).
$$

This is the Hilbert structure that makes gradients in matrix spaces canonical (by Riesz) once chosen.

A scalar-valued function of a matrix,

$$
\mathcal{L}: \mathbb{R}^{m\times n}\to\mathbb{R},
$$

has differential at $W$,

$$
d\mathcal{L}_W:\mathbb{R}^{m\times n}\to\mathbb{R},
\qquad \Delta W \mapsto d\mathcal{L}_W[\Delta W],
$$

a linear functional on the matrix space.

---

### 8.2.2 Gradient as Riesz representer of the differential (no "shape matching" folklore)

Because $(\mathbb{R}^{m\times n},\langle\cdot,\cdot\rangle_F)$ is a Hilbert space, Riesz representation implies there exists a unique matrix $\nabla_W \mathcal{L}$ such that

$$
d\mathcal{L}_W[\Delta W] = \langle \nabla_W \mathcal{L}, \Delta W\rangle_F
= \mathrm{tr}\big((\nabla_W\mathcal{L})^\top \Delta W\big),
\quad\text{for all }\Delta W.
$$

This equation is the definition of the gradient in matrix calculus. It is not a mnemonic; it is a representation theorem.

---

## 8.3 The Trace Principle: A Canonical Device for Reading Off Gradients

### 8.3.1 Why trace appears: scalars are invariant under cyclic permutation

Two trace identities will be used repeatedly:

<div class="prop-box">

**1. Cyclic invariance (when products are defined):**

$$
\mathrm{tr}(ABC)=\mathrm{tr}(BCA)=\mathrm{tr}(CAB).
$$

**2. Scalar identification:**

If $s\in\mathbb{R}$, then $s=\mathrm{tr}(s)$.

</div>

These are not tricks; they are coordinate-free ways to express inner products and to expose the unique matrix multiplying a perturbation $\Delta W$.

---

### 8.3.2 Canonical example: $\mathcal{L}(W)=\tfrac12\|Wx-b\|_2^2$

Let $x\in\mathbb{R}^n$, $b\in\mathbb{R}^m$, and define

$$
\mathcal{L}(W)=\frac{1}{2}\|Wx-b\|_2^2
=\frac{1}{2}(Wx-b)^\top (Wx-b).
$$

Set $r:=Wx-b\in\mathbb{R}^m$. We compute the differential under $W\mapsto W+\Delta W$.

<div class="proof-box">

**Step 1 (perturbation of the residual).**

$$
r(W+\Delta W) = (W+\Delta W)x - b = r + (\Delta W)x.
$$

Hence

$$
\delta r := r(W+\Delta W)-r = (\Delta W)x.
$$

**Step 2 (differential of the quadratic form).**

$$
\mathcal{L}(W+\Delta W)
= \frac{1}{2}(r+\delta r)^\top (r+\delta r)
= \frac{1}{2}(r^\top r + 2r^\top \delta r + (\delta r)^\top \delta r).
$$

Thus the first-order term is

$$
d\mathcal{L}_W[\Delta W] = r^\top \delta r = r^\top (\Delta W)x.
$$

**Step 3 (convert to Frobenius pairing).**

The scalar $r^\top (\Delta W)x$ can be written as a trace:

$$
r^\top (\Delta W)x
= \mathrm{tr}\big(r^\top (\Delta W)x\big)
= \mathrm{tr}\big(x r^\top (\Delta W)\big)
= \mathrm{tr}\big((r x^\top)^\top \Delta W\big).
$$

Therefore, by uniqueness of the Riesz representer,

$$
\nabla_W \mathcal{L} = r x^\top = (Wx-b)x^\top.
$$

</div>

<div class="ml-box">

This derivation explains why deep learning libraries repeatedly produce outer products between "error signals" and activations: it is forced by the Frobenius Riesz identity.

</div>

---

## 8.4 Chain Rule as Pullback: The Adjoint is the Time-Reversal Operator

### 8.4.1 The correct type statement: differentials are covectors

Let $\mathcal{L}:\mathbb{R}^p\to\mathbb{R}$ be a scalar loss and $F:\mathbb{R}^d\to\mathbb{R}^p$ a model map. Define the composite

$$
\Phi = \mathcal{L}\circ F:\mathbb{R}^d\to\mathbb{R}.
$$

At a point $x$, the differential $d\mathcal{L}_{F(x)}$ is a linear functional on $\mathbb{R}^p$, i.e. a covector in $(\mathbb{R}^p)^{\ast}$. The differential $dF_x$ is a linear map $\mathbb{R}^d\to\mathbb{R}^p$.

The chain rule at the level of differentials is

$$
d\Phi_x = d\mathcal{L}_{F(x)} \circ dF_x.
$$

This is composition of a covector with a linear map, producing a covector on $\mathbb{R}^d$.

---

### 8.4.2 Pullback and adjoint: how "errors" move backward

Given a linear map $A:V\to W$, its **adjoint** $A^{\ast}:W\to V$ is defined by

$$
\langle Av, w\rangle_W = \langle v, A^{\ast} w\rangle_V,
\quad \forall\, v\in V,\ w\in W.
$$

In Euclidean coordinates, $A^{\ast}=A^\top$. The point is not the transpose; it is the adjoint as the geometry-preserving pullback operator.

Given a covector $\eta\in W^{\ast}$, the pullback $A^{\ast}\eta\in V^{\ast}$ is defined by

$$
(A^{\ast}\eta)(v) := \eta(Av).
$$

Under inner-product identification (Riesz), covectors correspond to vectors; the pullback of covectors becomes multiplication by the adjoint of the Jacobian.

<div class="ml-box">

**Backpropagation in one sentence.**

Backprop is the repeated pullback of the terminal covector $d\mathcal{L}$ through the composed map defining the network, implemented by successive adjoints of local Jacobians.

</div>

---

## 8.5 Reverse-Mode Automatic Differentiation: The Algebraic Core of Backprop

### 8.5.1 A layered network as a composition of maps

Consider a depth-$L$ network as a composition

$$
h_0 = x,\quad
z_\ell = A_\ell h_{\ell-1} + b_\ell,\quad
h_\ell = \sigma_\ell(z_\ell),\quad \ell=1,\dots,L,
$$

and output $h_L$. Let the loss be $\mathcal{J}(x;\theta)=\mathcal{L}(h_L)$, where $\theta$ collects $\{A_\ell,b_\ell\}$.

We compute gradients by differentials.

---

### 8.5.2 Differential of the affine map

Fix $\ell$. The affine pre-activation is

$$
z_\ell = A_\ell h_{\ell-1} + b_\ell.
$$

A perturbation $(\Delta A_\ell,\Delta b_\ell,\Delta h_{\ell-1})$ induces

$$
\delta z_\ell
= (\Delta A_\ell)h_{\ell-1} + A_\ell(\delta h_{\ell-1}) + \Delta b_\ell.
$$

This is an identity in $\mathbb{R}^{d_\ell}$.

---

### 8.5.3 Differential of the nonlinearity (coordinate-wise, but stated invariantly)

Assume $\sigma_\ell:\mathbb{R}^{d_\ell}\to\mathbb{R}^{d_\ell}$ acts elementwise with derivative $\sigma_\ell'(z_\ell)$ (a vector). Then

$$
\delta h_\ell = D_\ell\, \delta z_\ell,
\qquad D_\ell := \mathrm{Diag}(\sigma_\ell'(z_\ell)).
$$

This isolates the local Jacobian of the activation.

---

### 8.5.4 Backward recursion by adjoints

Define the "error covector" at layer $\ell$ as the gradient of the loss with respect to $z_\ell$:

$$
\delta_\ell := \nabla_{z_\ell}\mathcal{J} \in \mathbb{R}^{d_\ell}.
$$

(Here we identify covectors and vectors via the Euclidean inner product.)

Start at the output:

$$
\delta_L = \nabla_{z_L}\mathcal{J}
= D_L^\top \nabla_{h_L}\mathcal{L}
= D_L \nabla_{h_L}\mathcal{L},
$$

since $D_L$ is diagonal.

Now propagate to the previous layer. From $z_\ell = A_\ell h_{\ell-1}+b_\ell$ and $h_{\ell-1}=\sigma_{\ell-1}(z_{\ell-1})$, we obtain

$$
\nabla_{h_{\ell-1}}\mathcal{J} = A_\ell^\top \delta_\ell,
\qquad
\delta_{\ell-1} = D_{\ell-1}(A_\ell^\top \delta_\ell).
$$

This is the pullback statement: the covector $\delta_\ell$ is pulled back through the affine map by $A_\ell^\top$, then through the nonlinearity by $D_{\ell-1}$.

---

### 8.5.5 Parameter gradients (complete derivation via trace)

We derive $\nabla_{A_\ell}\mathcal{J}$ without heuristics.

From the affine differential,

$$
\delta z_\ell = (\Delta A_\ell)h_{\ell-1} + \cdots
$$

and the definition of $\delta_\ell$,

$$
d\mathcal{J} = \delta_\ell^\top \delta z_\ell + \cdots
= \delta_\ell^\top (\Delta A_\ell)h_{\ell-1} + \delta_\ell^\top \Delta b_\ell + \cdots.
$$

<div class="proof-box">

Write the first term as a Frobenius pairing:

$$
\delta_\ell^\top (\Delta A_\ell)h_{\ell-1}
= \mathrm{tr}\big(\delta_\ell^\top (\Delta A_\ell)h_{\ell-1}\big)
= \mathrm{tr}\big(h_{\ell-1}\delta_\ell^\top \Delta A_\ell\big)
= \mathrm{tr}\big((\delta_\ell h_{\ell-1}^\top)^\top \Delta A_\ell\big).
$$

Hence

$$
\nabla_{A_\ell}\mathcal{J} = \delta_\ell h_{\ell-1}^\top,
\qquad
\nabla_{b_\ell}\mathcal{J} = \delta_\ell,
$$

the latter since $d\mathcal{J}=\delta_\ell^\top \Delta b_\ell=\langle \delta_\ell, \Delta b_\ell\rangle$.

</div>

<div class="ml-box">

**Machine-learning consequence (stated objectively).**

Backprop does not "choose" transposes and outer products to match dimensions. These objects are uniquely determined by adjoint geometry and the Frobenius Riesz representation.

</div>

---

## 8.6 A Modern Example: Attention as a Differential System

### 8.6.1 Single-head scaled dot-product attention (minimal but complete)

Let $Q\in\mathbb{R}^{n\times d}$, $K\in\mathbb{R}^{n\times d}$, $V\in\mathbb{R}^{n\times d_v}$. Define

$$
S := \frac{1}{\sqrt{d}} QK^\top \in \mathbb{R}^{n\times n},\qquad
P := \mathrm{softmax}(S)\ \text{(row-wise)},\qquad
Y := PV \in \mathbb{R}^{n\times d_v}.
$$

Let the loss be $\mathcal{J}=\mathcal{L}(Y)$.

We compute differentials in stages.

---

### 8.6.2 Differential of the linear maps

From $Y=PV$,

$$
\delta Y = (\delta P)V + P(\delta V).
$$

Given $\nabla_Y\mathcal{J} = G_Y$, we have

$$
d\mathcal{J} = \langle G_Y, \delta Y\rangle_F
= \langle G_Y, (\delta P)V\rangle_F + \langle G_Y, P(\delta V)\rangle_F.
$$

<div class="proof-box">

For the $V$-term:

$$
\langle G_Y, P(\delta V)\rangle_F
= \mathrm{tr}(G_Y^\top P\,\delta V)
= \mathrm{tr}((P^\top G_Y)^\top \delta V),
$$

hence

$$
\nabla_V\mathcal{J} = P^\top G_Y.
$$

For the $P$-term:

$$
\langle G_Y, (\delta P)V\rangle_F
= \mathrm{tr}(G_Y^\top (\delta P)V)
= \mathrm{tr}(V G_Y^\top \delta P)
= \mathrm{tr}((G_Y V^\top)^\top \delta P),
$$

hence

$$
\nabla_P\mathcal{J} = G_Y V^\top.
$$

</div>

---

### 8.6.3 Differential of row-wise softmax (expressed without coordinate obscurity)

For a single row $p=\mathrm{softmax}(s)\in\mathbb{R}^n$, the Jacobian is

$$
J_{\mathrm{softmax}}(s) = \mathrm{Diag}(p) - p p^\top.
$$

Thus for a perturbation $\delta s$,

$$
\delta p = (\mathrm{Diag}(p)-pp^\top)\,\delta s.
$$

Row-wise, this means: for each row $i$,

$$
\delta P_{i:} = \Big(\mathrm{Diag}(P_{i:}) - P_{i:}^\top P_{i:}\Big)\, \delta S_{i:}.
$$

Therefore the pullback is, row-wise:

$$
\nabla_{S_{i:}}\mathcal{J}
= \Big(\mathrm{Diag}(P_{i:}) - P_{i:}^\top P_{i:}\Big)\, \nabla_{P_{i:}}\mathcal{J}.
$$

This is the correct covector propagation through softmax.

---

### 8.6.4 Back to $Q,K$ through $S=\frac{1}{\sqrt{d}}QK^\top$

$$
\delta S = \frac{1}{\sqrt{d}}\big((\delta Q)K^\top + Q(\delta K)^\top\big).
$$

Let $G_S := \nabla_S\mathcal{J}$. Then

$$
d\mathcal{J} = \langle G_S, \delta S\rangle_F = \frac{1}{\sqrt{d}}\langle G_S, (\delta Q)K^\top\rangle_F + \frac{1}{\sqrt{d}}\langle G_S, Q(\delta K)^\top\rangle_F.
$$

<div class="proof-box">

First term:

$$
\langle G_S, (\delta Q)K^\top\rangle_F
= \mathrm{tr}(G_S^\top (\delta Q)K^\top)
= \mathrm{tr}(K^\top G_S^\top \delta Q)
= \mathrm{tr}((G_S K)^\top \delta Q),
$$

so

$$
\nabla_Q\mathcal{J} = \frac{1}{\sqrt{d}}\, G_S K.
$$

Second term:

$$
\langle G_S, Q(\delta K)^\top\rangle_F
= \mathrm{tr}(G_S^\top Q(\delta K)^\top)
= \mathrm{tr}((\delta K)(G_S^\top Q)^\top)
= \mathrm{tr}((G_S^\top Q)^\top \delta K),
$$

so

$$
\nabla_K\mathcal{J} = \frac{1}{\sqrt{d}}\, G_S^\top Q.
$$

</div>

<div class="ml-box">

**Machine-learning consequence.**

The "transpose patterns" in attention backprop are not artifacts of tensor libraries; they are adjoint pullbacks of bilinear maps. Attention is therefore a concrete, contemporary instance where matrix calculus is best understood as the calculus of linear maps and their adjoints.

</div>

---

<div class="scholium-box">

## 8.7 The Structural Defect to Eliminate: Treating Gradients as Intrinsic Vectors

A recurrent conceptual error in ML exposition is to speak as if gradients live in the same space as parameters without specifying a metric. Formally, the differential $d\mathcal{J}_\theta$ is a covector in $T_\theta^{\ast}\Theta$. Turning it into a vector update direction requires a Riesz map induced by an inner product (equivalently, a local metric). In Euclidean training, the chosen metric is implicit and fixed; in modern optimization (natural gradient, preconditioning, adaptive methods), the metric is explicit or effectively learned.

This is the correct mathematical lens for why different optimizers behave differently: they implement different identifications between covectors and vectors.

</div>

---

## 8.8 Transition to Chapter 9: Random Matrix Theory as the Initialization Law of Learning Dynamics

The matrix-calculus viewpoint isolates what training must do: backprop repeatedly applies adjoint Jacobians to propagate covectors. Hence the stability of learning hinges on the spectral behavior of products of Jacobians (or their linearized surrogates) across depth.

The next chapter formalizes the following claim in precise terms:

<div class="ml-box">

**Claim (initialization stability problem).**

At initialization, deep networks induce random linear operators (Jacobians) whose singular value distributions govern (i) forward signal propagation and (ii) backward gradient propagation. Random matrix theory provides the correct asymptotic language for these spectra in high dimension, thereby explaining when training begins in a stable regime (neither exploding nor vanishing) and why certain initialization and normalization schemes enforce near-isometries of signal flow.

</div>

This is not metaphor. It is the analytic statement that the *first* condition for learnability is spectral regularity of the random operators defining the network's local linearizations.

Accordingly, Chapter 9 will study the spectral laws of large random matrices and random operator products, and then interpret those laws as constraints on trainable architectures: stability of signal, stability of gradient, and the onset of pathological curvature.
