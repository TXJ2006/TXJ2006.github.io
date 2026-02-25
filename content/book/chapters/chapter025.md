---
title: "Chapter 25: Riemannian Manifolds and Metric Tensors"
layout: "single"
url: "/book/chapters/chapter025/"
summary: "Smooth manifolds defined via atlases; tangent vectors as derivations; cotangent space and covariant/contravariant transformation laws; Riemannian metric as a smoothly varying inner product; length, energy, and geodesics; Levi-Civita connection and Christoffel symbols derived from first axioms; Riemannian gradient and the mechanism behind natural gradient; pullback metrics and the Gauss-Newton connection; Fisher-Rao as the statistical manifold metric; optimization as choosing a metric."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 25
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

## Chapter 25 &mdash; Riemannian Manifolds and Metric Tensors

*Xujiang Tang*

</div>

## Abstract

A Riemannian manifold is a smooth manifold equipped with a smoothly varying inner product on each tangent space. This chapter builds the structure from scratch: manifolds as locally Euclidean spaces defined by atlases; tangent vectors as derivations; cotangent vectors and the distinction between covariant and contravariant transformation laws; the Riemannian metric tensor, its coordinate components, and the explicit constraint they satisfy under coordinate changes. From the metric we derive length, energy, and geodesics; construct the unique torsion-free metric-compatible connection (Levi&ndash;Civita) and its Christoffel symbols; and define the Riemannian gradient as the metric-dual of the differential. The chapter concludes with pullback metrics, the statistical manifold, and a geometric reading of the whole family of gradient-based optimization algorithms as gradient descent under a chosen metric.

---

## 1. Smooth Manifolds: Why Coordinates Are Not the Object

<div class="def-box">

**Definition 25.1 (Smooth manifold).** A $d$-dimensional smooth manifold is a set $M$ equipped with an atlas $\{(U_\alpha, \varphi_\alpha)\}$ where:
- $U_\alpha \subset M$ are open sets covering $M$,
- $\varphi_\alpha : U_\alpha \to \varphi_\alpha(U_\alpha) \subset \mathbb{R}^d$ are homeomorphisms (coordinate charts),
- for overlapping charts $U_\alpha \cap U_\beta \ne \emptyset$, the transition maps $\varphi_\beta \circ \varphi_\alpha^{-1} : \varphi_\alpha(U_\alpha \cap U_\beta) \to \varphi_\beta(U_\alpha \cap U_\beta)$ are $C^\infty$ diffeomorphisms.

</div>

The central point: **the manifold is not any particular coordinate representation.** A chart is an auxiliary local description; consistency between overlapping charts is enforced by smooth transition maps. Two atlases define the same smooth structure if their union is still an atlas.

<div class="ml-box">

**ML consequence.** Whenever "reparameterization should not change the algorithm," the algorithm must be defined via intrinsic objects on $M$, not via coordinate components. An update rule $\theta \leftarrow \theta - \alpha\nabla_\theta J$ depends on coordinates; a step defined by $\mathrm{KL}(p_\theta \| p_{\theta+\delta}) \le \varepsilon$ is intrinsic.

</div>

---

## 2. Tangent Vectors: The Right Notion of Infinitesimal Direction

On $\mathbb{R}^d$, a tangent vector at $x$ is an element of $\mathbb{R}^d$. On a general manifold with no ambient linear structure, tangent vectors must be defined intrinsically.

### 2.1 Tangent vectors as derivations

<div class="def-box">

**Definition 25.2 (Tangent vector).** A tangent vector at $p \in M$ is a linear map $v : C^\infty(M) \to \mathbb{R}$ satisfying the Leibniz rule:
$$v(fg) = v(f)\,g(p) + f(p)\,v(g).$$
The tangent space $T_pM$ is the set of all such derivations; it is a $d$-dimensional real vector space.

</div>

This encodes "directional derivative at a point" without assuming any coordinates.

### 2.2 Coordinate basis and decomposition

Take a chart $(U, \varphi)$ with coordinates $x^1, \dots, x^d$ on $\varphi(U) \subset \mathbb{R}^d$. For $f \in C^\infty(M)$, let $\tilde{f} = f \circ \varphi^{-1}$. Define the coordinate derivations at $p$:

$$\left.\frac{\partial}{\partial x^i}\right|_p(f) := \left.\frac{\partial \tilde{f}}{\partial x^i}\right|_{\varphi(p)}.$$

These are derivations and form a basis of $T_pM$. Any $v \in T_pM$ decomposes as $v = v^i \partial_{x^i}|_p$. The coefficients $(v^i)$ depend on the chosen chart; the tangent vector $v$ does not.

### 2.3 Coordinate change: contravariant law

Under a transition map $y = y(x)$:

$$\frac{\partial}{\partial x^i} = \frac{\partial y^a}{\partial x^i}\frac{\partial}{\partial y^a}.$$

If $v = v^i\partial_{x^i} = w^a\partial_{y^a}$, then $w^a = \dfrac{\partial y^a}{\partial x^i}\,v^i$. Components transform by the Jacobian of the forward map. This is the **contravariant transformation law**.

---

## 3. Cotangent Vectors and the Dual Geometry

The cotangent space $T_p^*M$ consists of linear functionals on $T_pM$. Elements are called **covectors** or **1-forms**.

The dual basis $\{dx^i\}$ is defined by $dx^i(\partial_{x^j}) = \delta^i_j$. A covector $\omega = \omega_i\,dx^i$ transforms under $y = y(x)$ as:

$$\omega'_a = \frac{\partial x^i}{\partial y^a}\,\omega_i.$$

Components transform by the inverse Jacobian. This is the **covariant transformation law**.

<div class="ml-box">

**ML relevance.** Gradients are naturally covectors (differentials of scalar functions), while parameter updates are vectors. A metric is the algebraic bridge between the two&mdash;its absence is the geometric reason ordinary gradient descent is coordinate-dependent.

</div>

---

## 4. Riemannian Metric: A Smoothly Varying Inner Product

<div class="def-box">

**Definition 25.3 (Riemannian manifold).** A Riemannian manifold $(M, g)$ is a smooth manifold $M$ equipped with a Riemannian metric $g$: an assignment to each $p \in M$ of a bilinear map $g_p : T_pM \times T_pM \to \mathbb{R}$ that is:
1. **Symmetric:** $g_p(u,v) = g_p(v,u)$,
2. **Positive definite:** $g_p(v,v) > 0$ for all $v \ne 0$,
3. **Smooth:** for smooth vector fields $U, V$, the function $p \mapsto g_p(U_p, V_p)$ is smooth.

</div>

The metric is local inner products smoothly glued across the manifold.

---

## 5. Metric Tensor in Coordinates: Components and Transformation Law

In a chart with coordinates $x^1, \dots, x^d$, define the metric components:

$$g_{ij}(p) := g_p\!\left(\frac{\partial}{\partial x^i},\,\frac{\partial}{\partial x^j}\right).$$

For vectors $u = u^i\partial_{x^i}$ and $v = v^j\partial_{x^j}$:

$$g_p(u,v) = g_{ij}(p)\,u^i v^j.$$

The matrix $G(p) = [g_{ij}(p)]$ is symmetric positive definite.

### 5.1 Coordinate change: the (0,2)-tensor law

Under $y = y(x)$ with Jacobian $J = \partial y/\partial x$:

$$g_{ij} = \frac{\partial y^a}{\partial x^i}\frac{\partial y^b}{\partial x^j}\,g'_{ab}, \qquad\text{equivalently}\quad G(x) = J(x)^\top G'(y)\,J(x).$$

The metric is not a matrix; the matrix is its coordinate representation. The transformation law is the condition that guarantees coordinate independence.

---

## 6. Length, Energy, and the Induced Distance

Let $\gamma : [a,b] \to M$ be a smooth curve with velocity $\dot\gamma(t) \in T_{\gamma(t)}M$.

### 6.1 Length

$$L(\gamma) = \int_a^b \sqrt{g_{\gamma(t)}(\dot\gamma(t),\dot\gamma(t))}\,dt.$$

### 6.2 Energy

$$E(\gamma) = \frac{1}{2}\int_a^b g_{\gamma(t)}(\dot\gamma(t),\dot\gamma(t))\,dt.$$

Energy avoids the square root and is technically simpler. Among curves with fixed endpoints and fixed parameter interval, energy minimizers are **geodesics** with constant speed; they are also length minimizers up to reparameterization.

<div class="ml-box">

**ML interpretation.** A training trajectory is a curve in parameter space. A metric decides which paths are short, which deformations are small, and which updates are conservative. Choosing the right metric for an optimization problem is a geometric design decision.

</div>

---

## 7. Levi&ndash;Civita Connection: Intrinsic Differentiation of Vector Fields

On $\mathbb{R}^d$, differentiating a vector field componentwise is invariant. On a manifold, coordinate components depend on the chart, so naive differentiation is not tensorial. A **connection** provides an intrinsic derivative.

<div class="def-box">

**Definition 25.4 (Connection).** A connection $\nabla$ assigns to vector fields $X, Y$ a vector field $\nabla_X Y$ satisfying:
- $\nabla_{fX+gY}Z = f\nabla_XZ + g\nabla_YZ$,
- $\nabla_X(Y+Z) = \nabla_XY + \nabla_XZ$,
- $\nabla_X(fY) = X(f)Y + f\nabla_XY$.

</div>

<div class="def-box">

**Definition 25.5 (Levi&ndash;Civita connection).** The unique connection on $(M,g)$ satisfying:
1. **Torsion-free:** $\nabla_XY - \nabla_YX = [X,Y]$,
2. **Metric-compatible:** $X(g(Y,Z)) = g(\nabla_XY, Z) + g(Y, \nabla_XZ)$.

</div>

These two axioms are not arbitrary: torsion-free means no intrinsic twist; metric-compatibility means parallel transport preserves inner products.

### 7.1 Christoffel symbols derived from the axioms

In coordinates, write $\nabla_{\partial_i}\partial_j = \Gamma^k_{ij}\,\partial_k$. Applying metric-compatibility to coordinate basis fields and using torsion-freeness ($\Gamma^k_{ij} = \Gamma^k_{ji}$) gives the system:

$$\partial_i g_{jk} = \Gamma^m_{ij}g_{mk} + \Gamma^m_{ik}g_{jm}, \quad \partial_j g_{ik} = \Gamma^m_{ji}g_{mk} + \Gamma^m_{jk}g_{im}, \quad \partial_k g_{ij} = \Gamma^m_{ki}g_{mj} + \Gamma^m_{kj}g_{im}.$$

Adding the first two and subtracting the third, and using symmetry of $\Gamma$, cancellation yields:

$$\partial_i g_{jk} + \partial_j g_{ik} - \partial_k g_{ij} = 2\,\Gamma^m_{ij}\,g_{mk}.$$

Contracting with the inverse metric $g^{k\ell}$ (where $g^{k\ell}g_{\ell m} = \delta^k_m$):

<div class="prop-box">

**Proposition 25.6 (Christoffel symbols).** The Levi&ndash;Civita connection coefficients are uniquely determined by the metric:
$$\Gamma^\ell_{ij} = \frac{1}{2}\,g^{\ell k}\!\left(\partial_i g_{jk} + \partial_j g_{ik} - \partial_k g_{ij}\right).$$

</div>

The connection is completely determined by $g$ and its first derivatives. This is where "the metric determines the geometry" becomes a concrete formula.

---

## 8. Geodesics: Shortest Paths and the Euler&ndash;Lagrange Equation

<div class="def-box">

**Definition 25.7 (Geodesic).** A geodesic is a smooth curve $\gamma$ satisfying $\nabla_{\dot\gamma}\dot\gamma = 0$ (parallel transport of its own velocity). In local coordinates:
$$\ddot{x}^\ell + \Gamma^\ell_{ij}(x)\,\dot{x}^i\dot{x}^j = 0.$$

</div>

### 8.1 Variational derivation from the energy functional

Taking $\mathcal{L}(x,\dot{x}) = \tfrac{1}{2}g_{ij}(x)\dot{x}^i\dot{x}^j$ and applying the Euler&ndash;Lagrange equation:

$$\frac{d}{dt}\!\left(\frac{\partial\mathcal{L}}{\partial\dot{x}^\ell}\right) - \frac{\partial\mathcal{L}}{\partial x^\ell} = 0.$$

Computing: $\partial\mathcal{L}/\partial\dot{x}^\ell = g_{\ell j}\dot{x}^j$ and $\partial\mathcal{L}/\partial x^\ell = \tfrac{1}{2}(\partial_\ell g_{ij})\dot{x}^i\dot{x}^j$. Expanding the time derivative and multiplying by $g^{\ell m}$ recovers exactly:

$$\ddot{x}^m + \Gamma^m_{ij}\dot{x}^i\dot{x}^j = 0.$$

The metric, via Christoffel symbols, induces canonical dynamics: geodesics are the metric&rsquo;s own notion of straight lines.

---

## 9. Riemannian Gradient: Why Natural Gradient Is Not a Metaphor

Let $f : M \to \mathbb{R}$ be smooth. Its differential at $p$ is the covector $df_p \in T_p^*M$ defined by $df_p(v) = v(f)$.

<div class="def-box">

**Definition 25.8 (Riemannian gradient).** The Riemannian gradient $\operatorname{grad} f(p)$ is the unique vector in $T_pM$ satisfying:
$$g_p(\operatorname{grad} f(p),\, v) = df_p(v) \qquad \forall\, v \in T_pM.$$

</div>

In coordinates: $df = (\partial_i f)\,dx^i$, and writing $\operatorname{grad} f = (\operatorname{grad} f)^k\partial_k$, the definition gives $g_{ij}(\operatorname{grad} f)^i = \partial_j f$, hence:

$$(\operatorname{grad} f)^k = g^{kj}\partial_j f.$$

<div class="prop-box">

**Proposition 25.9.** In any coordinates, $\operatorname{grad} f = G^{-1}\nabla f$, where $\nabla f$ is the coordinate gradient vector and $G = [g_{ij}]$.

</div>

In Euclidean space $g_{ij} = \delta_{ij}$, this reduces to the standard gradient. On a general manifold, the inverse metric converts the covector $df$ into a tangent vector in the intrinsically correct way.

**This is the mechanism behind natural gradient:** in the statistical manifold with Fisher metric $\mathcal{I}(\theta)$,

$$\operatorname{grad} J = \mathcal{I}(\theta)^{-1}\nabla_\theta J.$$

The ordinary gradient $\nabla_\theta J$ is a covector (differential); the Fisher metric supplies the isomorphism $T^*_\theta \to T_\theta$ that converts it into the correct descent direction.

---

## 10. Pullback Metrics and Model-Induced Geometry

<div class="def-box">

**Definition 25.10 (Pullback metric).** Let $\Phi : (M, g) \to (N, h)$ be a smooth map. The pullback metric $\Phi^*h$ on $M$ is:
$$(\Phi^*h)_p(u,v) := h_{\Phi(p)}(d\Phi_p(u),\, d\Phi_p(v)), \qquad u,v \in T_pM.$$

</div>

In local coordinates, if $y = \Phi(x)$ and $J = \partial y/\partial x$:

$$G_{\text{pull}}(x) = J(x)^\top H(y)\, J(x).$$

This formula appears in three places:
1. **Fisher reparameterization** (Chapter 24): $\mathcal{I}_\eta = J^\top\mathcal{I}_\theta J$.
2. **Gauss&ndash;Newton**: a squared loss in output space pulled back to parameter space gives $J^\top J$ as curvature.
3. **Representation learning**: a metric on embedding space $\mathbb{R}^k$ pulled back through $f_\theta : \mathbb{R}^D \to \mathbb{R}^k$ by its Jacobian.

The Jacobian is the universal mechanism by which geometry transfers across smooth maps.

---

## 11. The Statistical Manifold: Fisher&ndash;Rao as a Riemannian Metric

The parametric family $\{p_\theta : \theta \in \Theta\}$ is a manifold with parameters as (local) coordinates. The Fisher information matrix

$$\mathcal{I}(\theta) = \mathbb{E}_{p_\theta}\!\left[\nabla_\theta\log p_\theta(X)\,\nabla_\theta\log p_\theta(X)^\top\right]$$

transforms as a $(0,2)$-tensor under reparameterization (Proposition 24.6), so it defines a Riemannian metric on this manifold. The cleanest intrinsic characterization is via local KL geometry:

$$\mathrm{KL}(p_\theta \,\|\, p_{\theta+\delta}) = \frac{1}{2}\delta^\top\mathcal{I}(\theta)\delta + o(\|\delta\|^2).$$

Fisher&ndash;Rao is the unique Riemannian metric that captures the second-order infinitesimal structure of KL divergence. This shifts the unit of "small step":

- **Euclidean step:** $\|\delta\|$ small in parameter coordinates.
- **Fisher step:** $\mathrm{KL}(p_\theta, p_{\theta+\delta})$ small in distribution space.

The difference is operationally significant whenever parameterization is ill-conditioned or redundant, which is the rule rather than the exception in deep networks.

---

## 12. Optimization as Choosing a Metric

The entire family of gradient-based optimization algorithms can be read geometrically as gradient descent under a chosen Riemannian metric $g_\theta$ on parameter space:

$$\theta \leftarrow \theta - \alpha\,G(\theta)^{-1}\nabla J(\theta), \qquad G(\theta) = [g_{ij}(\theta)].$$

| Algorithm | Metric $G(\theta)$ | Geometric meaning |
|---|---|---|
| Gradient descent | $I$ (identity) | Euclidean metric on parameters |
| Natural gradient | $\mathcal{I}(\theta)$ (Fisher) | Fisher&ndash;Rao on distributions |
| Mirror descent | $\nabla^2 F(\theta)$ (Hessian of potential) | Bregman geometry; KL for $F=-H$ |
| Newton&rsquo;s method | $\nabla^2 J(\theta)$ (loss Hessian) | Curvature of objective |
| TRPO / trust region | $\mathcal{I}_\pi(\theta)$ (policy Fisher) | Distribution change of policy |

A practical implication: an algorithm "works across reparameterizations" precisely when its update is defined via an intrinsic object (a Riemannian gradient), not via coordinate components.

---

## 13. Data Manifolds vs Parameter Manifolds

Two geometries are often conflated in ML discussions:

1. **Data manifold** (in representation space): the support of the data distribution in $\mathbb{R}^D$, approximated by a low-dimensional submanifold. This is an extrinsic geometry question: the manifold is embedded in a vector space and analyzed via tangent structure and curvature.

2. **Model manifold** (in distribution space): the family $\{p_\theta\}$ as $\theta$ varies. This is an intrinsic geometry question: distributions form a manifold with Fisher metric; parameters are local coordinates.

These interact in generative modeling:
- **Diffusion and flow models** build time-indexed distribution families $p_t$ and velocity fields transporting samples. The natural notion of closeness is distributional (KL, Wasserstein), hence a metric on the model manifold.
- **Representation learning** shapes the geometry of embeddings&mdash;angles, norms, geodesic distances&mdash;which is a data-manifold geometry.

A consistent stance is to keep track explicitly of which manifold is under discussion and what metric is in use. Many failures in ML geometry arguments are not algebraic errors but metric mismatches.

---

## 14. Summary

<div class="scholium-box">

**Chapter 25 in one paragraph.** A smooth manifold is defined by an atlas of charts with smooth transition maps; the manifold is not any coordinate system. Tangent vectors are derivations, transforming contravariantly; covectors transform covariantly. A Riemannian metric $g$ is a smooth field of inner products on tangent spaces; in coordinates it is a symmetric positive-definite matrix $G = [g_{ij}]$ satisfying $G(x) = J^\top G'(y) J$ under coordinate changes. The metric determines length, energy, and geodesics; it uniquely selects the Levi&ndash;Civita connection with Christoffel symbols $\Gamma^\ell_{ij} = \tfrac{1}{2}g^{\ell k}(\partial_i g_{jk} + \partial_j g_{ik} - \partial_k g_{ij})$. The Riemannian gradient is $\operatorname{grad} f = G^{-1}\nabla f$: the metric converts a covector (ordinary gradient) into a vector (descent direction). Pullback metrics transport geometry across smooth maps via the Jacobian. The Fisher information matrix is the canonical Riemannian metric on the statistical manifold, and every gradient-based optimization algorithm is gradient descent under a chosen metric.

</div>

---

### Transition to Chapter 26 (Geodesics and Exponential Maps)

Chapter 25 defined geodesics via the variational principle and the geodesic ODE. Chapter 26 develops the geodesic structure in more depth: the exponential map $\exp_p : T_pM \to M$ that turns tangent vectors into curves on the manifold; its inverse (the logarithmic map); the cut locus where geodesics stop being minimizing; and the Riemannian distance function. On the statistical manifold, exponential and logarithmic maps become tools for moving between distributions in a geodesically natural way, with direct applications to online learning algorithms on simplices and on the space of Gaussian distributions.

*Next: [Chapter 26: Geodesics and Exponential Maps](/book/chapters/chapter026/)*
