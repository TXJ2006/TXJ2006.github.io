---
title: "Chapter 23: The Maximum Entropy Principle"
layout: "single"
url: "/book/chapters/chapter023/"
summary: "Maximum-entropy variational principle; derivation of exponential families; dual geometry via log-partition function; MaxEnt as I-projection (minimum relative entropy); ML instantiations: log-linear models, entropy regularization, label smoothing; continuous relative-entropy variant."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 23
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

## Chapter 23 &mdash; The Maximum Entropy Principle

*Xujiang Tang*

</div>

## Abstract

The maximum entropy principle chooses, among all distributions satisfying given moment constraints, the one that commits to no additional structure beyond those constraints. This variational formulation yields exponential-family distributions, admits a dual convex geometry governed by the log-partition function, and admits the clean information-geometric interpretation of an I-projection (minimum relative entropy) onto the constraint set. In machine learning this principle underlies log-linear models, label smoothing, entropy-regularized policies, and connections between Gibbs distributions and energy-based models.

---

## 1. The variational problem (definition)

Let $\mathcal X$ be finite and let statistics $\phi_1,\dots,\phi_k:\mathcal X\to\mathbb R$ be fixed. Given target moments $c_i$, solve

$$p^*\in\arg\max_{p\in\Delta}\ H(p)\quad\text{s.t.}\quad \mathbb{E}_p[\phi_i]=c_i,\ i=1,\dots,k,$$

and normalization $\sum_x p(x)=1$. The feasible set is compact; $H$ is continuous and strictly concave on the interior, so a unique interior solution exists when the constraints are compatible with the interior of the simplex.

---

## 2. Solving MaxEnt: Lagrangian and the exponential family

Assume an interior optimizer and introduce multipliers $\lambda\in\mathbb{R}^k$ and scalar $\nu$ for normalization. The stationarity condition yields

$$\log p(x)=\sum_{i=1}^k\lambda_i\phi_i(x)+\nu-1,$$

hence the optimizer has the exponential form

$$p^*_\lambda(x)=\frac{1}{Z(\lambda)}\exp\!\left(\sum_{i=1}^k\lambda_i\phi_i(x)\right),\qquad Z(\lambda)=\sum_x\exp\!\left(\sum_i\lambda_i\phi_i(x)\right).$$

The multipliers $\lambda$ are chosen so that $\mathbb{E}_{p^*_\lambda}[\phi_i]=c_i$.

---

## 3. Dual problem and geometry: moments as coordinates

Define the log-partition function $A(\lambda)=\log Z(\lambda)$. Then

$$\frac{\partial A}{\partial\lambda_i}=\mathbb{E}_{p^*_\lambda}[\phi_i(X)],$$

so the map $\lambda\mapsto\mathbb{E}_{p^*_\lambda}[\phi(X)]$ is the gradient of a convex function. Natural parameters $\lambda$ live in the dual coordinate system to moment coordinates $c$, and $A$ induces a dually-flat geometry used throughout information geometry.

---

## 4. MaxEnt as the I-projection of a reference measure

Maximizing entropy over a constrained set $\mathcal C$ is equivalent to minimizing $\mathrm{KL}(p\|u)$ where $u$ is uniform. More generally, with a reference measure $m(x)$, MaxEnt becomes minimum relative entropy to $m$:

$$\min_{p\in\mathcal C}\ \mathrm{KL}(p\|m)\quad\Longrightarrow\quad p^*(x)\propto m(x)\,e^{\lambda^\top\phi(x)}.$$ 

This is the information-projection viewpoint: MaxEnt picks the distribution in the feasible set nearest to the reference in KL geometry.

---

## 5. Canonical ML example: maximum-entropy classification (log-linear models)

Conditional MaxEnt with features $f_j(x,y)$ yields the softmax/log-linear conditional model

$$p_\theta(y\mid x)=\frac{\exp(\theta^\top f(x,y))}{\sum_{y'}\exp(\theta^\top f(x,y'))},$$

and training by maximum likelihood enforces empirical moment matching $\nabla A(\theta)=\hat c$.

---

## 6. MaxEnt as "choose a distribution, not a point estimate"

MaxEnt selects a distribution on the affine face determined by constraints, avoiding unwarranted higher-order structure. When constraints are empirical moments, MaxEnt is the least-committal distribution consistent with observed summaries.

---

## 7. Entropy regularization as smoothing in learning dynamics

Entropy regularization biases solutions toward interior points of the simplex (smoothing). In RL, adding policy entropy favors exploration. In classification, label smoothing is a small MaxEnt-style correction that improves calibration and prevents infinite logits.

---

## 8. Continuous spaces: maximum relative entropy

For continuous sample spaces, the invariant statement is minimum relative entropy to a reference measure $m(x)$ under moment constraints. The solution takes the form

$$p^*(x)=\frac{1}{Z(\lambda)}\,m(x)\exp(\lambda^\top\phi(x)).$$

This form is coordinate-free and avoids the pathologies of differential entropy without a base measure.

---

## 9. Summary

MaxEnt is a constrained variational principle whose solution is an exponential-family distribution. It is geometrically the KL-projection to the feasible set and underlies many ML constructs: log-linear models, Gibbs energies, entropy-regularization, and principled label smoothing.

*Next: [Chapter 24: Fisher Information Matrix](/book/chapters/chapter024/)*
