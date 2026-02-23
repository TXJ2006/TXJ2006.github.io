---
title: "Chapter 24: Fisher Information Matrix — The Foundation of Natural Gradient"
layout: "single"
url: "/book/chapters/chapter024/"
summary: "Score function and Fisher information defined as the second moment of the score; equivalence with negative expected Hessian; Fisher as the Riemannian metric induced by local KL expansion; reparameterization invariance; natural gradient as steepest descent under a KL trust-region constraint; ML instantiations in logistic regression, Gauss–Newton, K-FAC, and policy optimization."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 24
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

## Chapter 24 &mdash; Fisher Information Matrix: The Foundation of Natural Gradient

*Xujiang Tang*

</div>

## Abstract

Let $\{p_\theta(x)\}_{\theta\in\Theta}$ be a parametric family with $\Theta\subset\mathbb{R}^d$ open and $p_\theta(x)>0$ with sufficient differentiability to exchange differentiation and integration. The Fisher information matrix (FIM) is defined as the second moment of the score $\nabla_\theta\log p_\theta(x)$, and equals both the negative expected Hessian of the log-likelihood and the coefficient of the leading quadratic term in the local expansion of KL divergence. This last characterization identifies the FIM as a Riemannian metric on the statistical manifold $\{p_\theta\}$, and provides a clean derivation of the natural gradient as the steepest descent direction under a KL trust-region constraint. The chapter develops all three equivalent definitions, proves reparameterization invariance, and anchors the theory in ML applications: logistic regression, the Gauss&ndash;Newton connection, K-FAC, and the TRPO trust-region geometry.

---

## 1. Score Function and Fisher Information: Basic Definitions

### 1.1 Log-likelihood and score

<div class="def-box">

**Definition 24.1 (Score function).** For a parametric family $\{p_\theta\}$, the log-density is $\ell_\theta(x):=\log p_\theta(x)$. The **score** at $\theta$ for observation $x$ is
$$s_\theta(x) := \nabla_\theta \log p_\theta(x) \in \mathbb{R}^d.$$

</div>

The score measures how sensitive the log-probability of $x$ is to an infinitesimal change of the parameter $\theta$.

### 1.2 Fisher information matrix

<div class="def-box">

**Definition 24.2 (Fisher information matrix).** The FIM at $\theta$ is the second moment of the score under $p_\theta$:
$$\mathcal{I}(\theta) := \mathbb{E}_{X\sim p_\theta}\!\left[s_\theta(X)\,s_\theta(X)^\top\right] = \int s_\theta(x)\,s_\theta(x)^\top\,p_\theta(x)\,dx.$$

</div>

$\mathcal{I}(\theta)$ is a $d\times d$ positive semidefinite matrix. It captures how sensitive the distribution $p_\theta$ is to infinitesimal parameter changes, averaged under the model's own distribution. No estimators, Cramér&ndash;Rao bounds, or data appear in this definition; the FIM is a property of the model family alone.

---

## 2. The Score has Zero Mean

<div class="prop-box">

**Proposition 24.3.** Under standard regularity, $\mathbb{E}_{p_\theta}[s_\theta(X)] = 0$.

</div>

<div class="proof-box">

*Proof.* Using the identity $\nabla_\theta\log p_\theta(x) = \nabla_\theta p_\theta(x)/p_\theta(x)$:
$$\mathbb{E}_{p_\theta}[s_\theta(X)] = \int \frac{\nabla_\theta p_\theta(x)}{p_\theta(x)}\,p_\theta(x)\,dx = \int \nabla_\theta p_\theta(x)\,dx = \nabla_\theta\!\int p_\theta(x)\,dx = \nabla_\theta(1) = 0. \quad\square$$

</div>

**Consequence.** The FIM is the variance (covariance matrix) of the score, not merely its second moment:
$$\mathcal{I}(\theta) = \mathrm{Cov}_{p_\theta}[s_\theta(X)] = \mathbb{E}[s_\theta s_\theta^\top] - \mathbb{E}[s_\theta]\mathbb{E}[s_\theta]^\top = \mathbb{E}[s_\theta s_\theta^\top].$$

The score behaves like a centered noise-like direction in parameter space under the model&rsquo;s own distribution, and the FIM is its covariance.

---

## 3. Equivalent Form: Fisher as Negative Expected Hessian

<div class="prop-box">

**Proposition 24.4 (Hessian form).** Under regularity,
$$\mathcal{I}(\theta) = -\,\mathbb{E}_{p_\theta}\!\left[\nabla_\theta^2 \log p_\theta(X)\right].$$

</div>

<div class="proof-box">

*Proof.* Differentiate the identity $\mathbb{E}_{p_\theta}[s_\theta(X)] = 0$ with respect to $\theta$:
$$0 = \nabla_\theta \int s_\theta(x)\,p_\theta(x)\,dx = \int \nabla_\theta\!\left(s_\theta(x)\,p_\theta(x)\right)dx.$$
Apply the product rule, using $\nabla_\theta s_\theta = \nabla_\theta^2\log p_\theta$ and $\nabla_\theta p_\theta = p_\theta\,s_\theta$:
$$\nabla_\theta(s_\theta p_\theta) = (\nabla_\theta^2\log p_\theta)\,p_\theta + s_\theta(p_\theta s_\theta)^\top.$$
Integrating both sides and recognizing expectations:
$$0 = \mathbb{E}_{p_\theta}[\nabla_\theta^2\log p_\theta(X)] + \mathbb{E}_{p_\theta}[s_\theta(X)s_\theta(X)^\top],$$
which gives $\mathcal{I}(\theta) = -\mathbb{E}_{p_\theta}[\nabla_\theta^2\log p_\theta(X)]$. $\square$

</div>

**Interpretation.** The FIM equals the average curvature of the log-likelihood surface, averaged under the model rather than under an empirical dataset. This &ldquo;intrinsic&rdquo; averaging is the geometric reason the FIM defines a metric on the model family rather than depending on a particular dataset.

---

## 4. Fisher Information as the Riemannian Metric Induced by KL

The deepest characterization of the FIM is as the second-order coefficient in the local expansion of KL divergence between nearby distributions in the family.

<div class="prop-box">

**Proposition 24.5 (KL second-order expansion).** For small $\delta\in\mathbb{R}^d$,
$$\mathrm{KL}(p_\theta \,\|\, p_{\theta+\delta}) = \frac{1}{2}\,\delta^\top \mathcal{I}(\theta)\,\delta + o(\|\delta\|^2).$$

</div>

<div class="proof-box">

*Proof.* Define $D(\delta):=\mathrm{KL}(p_\theta\|p_{\theta+\delta})$.

**First derivative vanishes at zero.** Differentiating $D(\delta) = \int p_\theta(x)[\log p_\theta(x) - \log p_{\theta+\delta}(x)]\,dx$ with respect to $\delta$ at $\delta=0$:
$$\nabla_\delta D\big|_{\delta=0} = -\int p_\theta(x)\nabla_\theta\log p_\theta(x)\,dx = -\mathbb{E}_{p_\theta}[s_\theta(X)] = 0.$$

**Second derivative gives Fisher.** Differentiating again:
$$\nabla_\delta^2 D\big|_{\delta=0} = -\int p_\theta(x)\nabla_\theta^2\log p_\theta(x)\,dx = -\mathbb{E}_{p_\theta}[\nabla_\theta^2\log p_\theta(X)] = \mathcal{I}(\theta).$$

A Taylor expansion then gives $D(\delta) = \tfrac{1}{2}\delta^\top\mathcal{I}(\theta)\delta + o(\|\delta\|^2)$. $\square$

</div>

**Geometric meaning.** The FIM is the Riemannian metric tensor induced by KL divergence on the statistical manifold $\{p_\theta : \theta\in\Theta\}$. KL divergence, while not a metric globally (it is asymmetric and fails the triangle inequality), induces a genuine Riemannian metric locally to second order. That metric is the Fisher information matrix.

---

## 5. Reparameterization Invariance

Let $\eta=\eta(\theta)$ be a smooth reparameterization with invertible Jacobian $J(\eta)=\partial\theta/\partial\eta$. Writing the same family as $p_\eta(x)=p_{\theta(\eta)}(x)$, the score transforms as

$$\nabla_\eta\log p_\eta(x) = J(\eta)^\top\nabla_\theta\log p_\theta(x).$$

<div class="prop-box">

**Proposition 24.6 (Metric transformation).** $\mathcal{I}_\eta(\eta) = J(\eta)^\top \mathcal{I}_\theta(\theta)\,J(\eta)$.

</div>

<div class="proof-box">

*Proof.* Directly from the score transformation:
$$\mathcal{I}_\eta = \mathbb{E}[(\nabla_\eta\log p)(\nabla_\eta\log p)^\top] = \mathbb{E}[J^\top ss^\top J] = J^\top\mathcal{I}_\theta\,J. \quad\square$$

</div>

This is precisely the transformation law of a covariant rank-2 tensor (a metric tensor) under coordinate change. The FIM is not merely a matrix associated with a parameterization; it is the intrinsic geometric object on the statistical manifold. This also confirms that the KL-expansion characterization in Section 4 is coordinate-free.

---

## 6. Natural Gradient: Steepest Descent Under Fisher Geometry

Let $J(\theta)$ be an objective to minimize (e.g., expected negative log-likelihood). Ordinary gradient descent measures steps in Euclidean parameter space. Natural gradient measures steps by how much the model distribution changes, using the KL-induced Fisher metric.

### 6.1 Derivation via KL trust-region constraint

Consider the constrained descent problem:

$$\min_{\delta\in\mathbb{R}^d}\ \nabla J(\theta)^\top\delta \quad\text{subject to}\quad \mathrm{KL}(p_\theta\|p_{\theta+\delta}) \le \varepsilon.$$

Using the local approximation $\mathrm{KL}(p_\theta\|p_{\theta+\delta})\approx \tfrac{1}{2}\delta^\top\mathcal{I}(\theta)\delta$, the constraint becomes the Fisher-metric ball $\tfrac{1}{2}\delta^\top\mathcal{I}\delta\le\varepsilon$.

Form the Lagrangian $\mathcal{L}(\delta,\lambda) = g^\top\delta + \lambda(\tfrac{1}{2}\delta^\top\mathcal{I}\delta - \varepsilon)$ with $g=\nabla J(\theta)$. The stationarity condition $\nabla_\delta\mathcal{L}=0$ gives:

$$g + \lambda\mathcal{I}\delta = 0 \quad\Rightarrow\quad \delta = -\frac{1}{\lambda}\mathcal{I}(\theta)^{-1}g.$$

The multiplier $\lambda>0$ is set by the constraint to control step size. The update direction is therefore:

<div class="def-box">

**Definition 24.7 (Natural gradient).** The natural gradient of $J$ at $\theta$ is
$$\widetilde{\nabla} J(\theta) := \mathcal{I}(\theta)^{-1}\nabla J(\theta),$$
and the natural gradient descent update is $\theta \leftarrow \theta - \alpha\,\mathcal{I}(\theta)^{-1}\nabla J(\theta)$.

</div>

Natural gradient is the steepest descent direction when distance in parameter space is measured by KL divergence between distributions, not by Euclidean distance between parameter vectors.

### 6.2 Score-field inner product interpretation

For tangent directions $u,v\in\mathbb{R}^d$, the Fisher metric defines
$$\langle u,v\rangle_\theta := u^\top\mathcal{I}(\theta)\,v = \mathbb{E}_{p_\theta}\!\left[(u^\top s_\theta(X))(v^\top s_\theta(X))\right].$$

A tangent direction $u$ in parameter space corresponds to the scalar random variable $u^\top s_\theta(X)$ (the directional score), and the inner product measures the $L^2(p_\theta)$ correlation of directional scores. Natural gradient uses these score-induced inner products as the definition of orthogonality and step length&mdash;not coordinate axes in $\mathbb{R}^d$.

<div class="ml-box">

**Why ordinary gradient is "wrong" geometrically.** An ordinary gradient step of size $\alpha$ in two different parameterizations of the same family produces different distributions (different amounts of distribution change). A natural gradient step of size $\alpha$ produces approximately the same KL change $\varepsilon$ in any parameterization. This invariance is the key practical benefit.

</div>

---

## 7. ML Instantiations

### 7.1 Logistic regression: Fisher equals expected feature covariance

For logistic regression $p_\theta(y=1\mid x)=\sigma(\theta^\top x)$, the log-likelihood Hessian per sample is $-\sigma(1-\sigma)\,xx^\top$. The FIM is

$$\mathcal{I}(\theta) = \mathbb{E}_{p_\theta}[\sigma(\theta^\top X)(1-\sigma(\theta^\top X))\,XX^\top],$$

a weighted covariance of the input features. The natural gradient step $\mathcal{I}^{-1}\nabla J$ is thus an inverse-covariance-weighted update, performing implicit feature whitening.

### 7.2 Neural networks: Fisher equals Gauss&ndash;Newton matrix

For models trained with log-likelihood losses, the FIM under the model distribution equals the expected Gauss&ndash;Newton matrix:

$$\mathcal{I}(\theta) = \mathbb{E}_{p_\theta}\!\left[J_\theta(x,y)^\top J_\theta(x,y)\right],$$

where $J_\theta$ is the Jacobian of the model outputs with respect to $\theta$. This connects natural gradient to the family of curvature-aware optimizers without requiring exact Hessian computation.

In deep learning, practical approximations of $\mathcal{I}^{-1}g$ include:
- **Diagonal Fisher**: ignore off-diagonal entries; simple but ignores correlations.
- **Block-diagonal Fisher**: one block per layer; preserves within-layer correlations.
- **K-FAC (Kronecker-Factored Approximate Curvature)**: approximates each block as a Kronecker product of two smaller matrices using the layer structure, enabling efficient inversion.

### 7.3 Reinforcement learning: TRPO and the policy Fisher metric

For a stochastic policy $\pi_\theta(a\mid s)$, define the Fisher matrix of the policy as

$$\mathcal{I}_\pi(\theta) = \mathbb{E}_{s\sim\rho,a\sim\pi_\theta}\!\left[\nabla_\theta\log\pi_\theta(a\mid s)\,\nabla_\theta\log\pi_\theta(a\mid s)^\top\right].$$

Trust Region Policy Optimization (TRPO) constrains each policy update by $\mathbb{E}_s[\mathrm{KL}(\pi_\theta(\cdot\mid s)\|\pi_{\theta+\delta}(\cdot\mid s))]\le\varepsilon$. Locally, this constraint becomes $\tfrac{1}{2}\delta^\top\mathcal{I}_\pi(\theta)\delta\le\varepsilon$, and the optimal update direction is exactly the natural gradient $\mathcal{I}_\pi^{-1}\nabla J$.

<div class="ml-box">

**TRPO in one sentence.** TRPO is natural gradient descent for the policy objective, with the KL constraint made explicit: the policy should change little in distribution space (few nats of KL per update), not necessarily little in parameter space (small Euclidean step).

</div>

---

## 8. A Deeper Geometric Perspective: Fisher as Intrinsic Energy of Score Fields

The score $s_\theta(x)=\nabla_\theta\log p_\theta(x)$ is a family of random variables indexed by $\theta$, one scalar per parameter direction. The FIM is their $L^2(p_\theta)$ Gram matrix:

$$[\mathcal{I}(\theta)]_{ij} = \langle \partial_i\log p_\theta,\,\partial_j\log p_\theta\rangle_{L^2(p_\theta)},$$

where $\partial_i=\partial/\partial\theta_i$. The statistical manifold $\{p_\theta\}$ is thus equipped with the inner product structure of its tangent space of score fields. All optimization algorithms that respect this inner product (natural gradient, mirror descent with negative entropy, TRPO) are instances of Riemannian gradient descent on this manifold.

---

## 9. Summary

<div class="scholium-box">

**Chapter 24 in one paragraph.** The score $s_\theta(x)=\nabla_\theta\log p_\theta(x)$ has zero mean under $p_\theta$; Fisher information $\mathcal{I}(\theta)=\mathbb{E}[ss^\top]$ is its covariance. Three equivalent characterizations: (1) second moment of the score; (2) negative expected Hessian of log-likelihood; (3) coefficient of the quadratic term in the local KL expansion $\mathrm{KL}(p_\theta\|p_{\theta+\delta})=\tfrac{1}{2}\delta^\top\mathcal{I}(\theta)\delta+o(\|\delta\|^2)$. Characterization (3) identifies $\mathcal{I}(\theta)$ as the Riemannian metric tensor on the statistical manifold. Natural gradient $\widetilde\nabla J=\mathcal{I}^{-1}\nabla J$ is the steepest descent direction under a KL trust-region constraint, and is coordinate-invariant. In ML: natural gradient for logistic regression whitens features; for neural networks it corresponds to Gauss&ndash;Newton; for policies it is the geometric core of TRPO.

</div>

| Quantity | Formula | Geometric meaning |
|---|---|---|
| Score | $s_\theta(x) = \nabla_\theta\log p_\theta(x)$ | Log-density gradient; zero mean under $p_\theta$ |
| FIM (definition) | $\mathcal{I}(\theta) = \mathbb{E}[ss^\top]$ | Covariance of the score |
| FIM (Hessian form) | $\mathcal{I}(\theta) = -\mathbb{E}[\nabla^2\log p_\theta]$ | Average log-likelihood curvature |
| FIM (KL form) | $\mathrm{KL}(p_\theta\|p_{\theta+\delta})\approx\frac{1}{2}\delta^\top\mathcal{I}\delta$ | Riemannian metric on statistical manifold |
| Natural gradient | $\widetilde\nabla J = \mathcal{I}^{-1}\nabla J$ | Steepest descent in distribution space |
| Reparameterization | $\mathcal{I}_\eta = J^\top\mathcal{I}_\theta J$ | Metric tensor transformation law |

---

### Transition to Chapter 25 (Riemannian Manifolds and Metric Tensors)

Chapter 24 used the language of Riemannian metrics and metric tensors informally, grounded in the specific example of the statistical manifold. Chapter 25 develops the general theory: what a smooth manifold is, what a Riemannian metric tensor is, how geodesics and exponential maps arise, and how curvature is defined. The Fisher information matrix will reappear as the canonical example of a Riemannian metric on a manifold of probability distributions, and the natural gradient will be reinterpreted as Riemannian gradient descent.

*Next: [Chapter 25: Riemannian Manifolds and Metric Tensors](/book/chapters/chapter025/)*
