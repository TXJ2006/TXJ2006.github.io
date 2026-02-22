---
title: "Chapter 10: Convex Sets & Cones"
layout: "single"
url: "/book/chapters/chapter010/"
summary: "Convexity as mixture-closure and certificate existence; Jensen's inequality, PSD cones, separation theorems, minimax duality, and mean-field convexification of deep learning."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 10
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

## Chapter 10 &mdash; Convex Sets &amp; Cones: The Geometry of Certainty and the Infinite-Dimensional Convexification of Deep Learning

*Xujiang Tang*

</div>

## Abstract

Deep learning is usually described as optimization in a finite-dimensional parameter space \(\Theta\subset\mathbb{R}^p\) with a nonconvex objective. That description is correct but not canonical. The canonical object is the *prediction functional* (or, in generative settings, the induced distribution), and convexity is a statement about whether mixtures in the correct object space remain admissible and whether linear certificates exist. This chapter isolates what convexity actually *buys*: (i) unique global notions of agreement (mixtures), (ii) separating hyperplanes as "proof objects," and (iii) duality that turns search into verification. We then show how modern ML repeatedly re-encounters convexity by changing coordinates—from parameters to measures, from finite neurons to distributions over neurons, and from point predictions to probability measures.

---

## 10.1 Convexity as the Topology of Mixtures and Conceptual Consensus

> **Core theorem-sentence 10.1.** Convexity is exactly closure under mixtures; it is the minimal geometry in which "interpolation" is an internal operation rather than an external approximation.

### 10.1.1 Convex sets are mixture-closed universes

Let \(V\) be a real vector space.

<div class="def-box">

**Definition 10.1 (Convex set).** \(C\subseteq V\) is convex if for all \(x,y\in C\) and \(\lambda\in[0,1]\),
\[
\lambda x+(1-\lambda)y\in C.
\]

</div>

The common "bowl-shaped function" intuition is secondary. The fundamental object is the *mixture operator* \((x,y,\lambda)\mapsto \lambda x+(1-\lambda)y\). In ML this operator appears as:
- mixture of hypotheses (ensembling),
- mixture of labels (label smoothing),
- mixture of samples (mixup),
- mixture of policies (RL),
- mixture of distributions (generative modeling).

Convexity is precisely the requirement that such mixtures do not leave the admissible universe.

### 10.1.2 Finite mixtures, proved explicitly

<div class="prop-box">

**Proposition 10.2 (Finite mixture closure).** If \(C\) is convex then for any \(x_1,\dots,x_k\in C\) and \(\alpha_i\ge0\) with \(\sum_i \alpha_i=1\),
\[
\sum_{i=1}^k \alpha_i x_i\in C.
\]

</div>

<div class="proof-box">

**Proof.** Induction on \(k\). Base \(k=2\) is Definition 10.1. Suppose true for \(k\). For \(k+1\), set \(s=\sum_{i=1}^k\alpha_i\) and \(\alpha_{k+1}=1-s\). If \(s=0\) the claim is trivial. If \(s>0\), define \(\beta_i=\alpha_i/s\), so \(\sum_{i=1}^k \beta_i=1\). By induction, \(z=\sum_{i=1}^k \beta_i x_i\in C\). Then
\[
\sum_{i=1}^{k+1}\alpha_i x_i = s z + (1-s) x_{k+1}\in C
\]
by convexity. \(\square\)

</div>

### 10.1.3 ML example where convexity is the hidden axiom: mixup as a convex-extension principle

<div class="ml-box">

In mixup, one replaces training pairs \((x_i,y_i)\), \((x_j,y_j)\) by
\[
\tilde{x}=\lambda x_i+(1-\lambda)x_j,\qquad
\tilde{y}=\lambda y_i+(1-\lambda)y_j.
\]
This assumes an implicit convex universe: that "intermediate inputs" and "intermediate labels" are meaningful training constraints. The practice works best when the data manifold is locally well-approximated by convex mixtures in representation space (often not in pixel space, but in learned feature space). Thus, mixup is not merely augmentation; it is an attempt to *force* the hypothesis class to behave as if the relevant semantic domain were convex.

A precise way to see this is through Jensen: if the model class is such that \(\ell(f(\lambda x+(1-\lambda)x'),\lambda y+(1-\lambda)y')\) upper-bounds the mixed loss, then training on mixtures enforces a convexity-like regularity constraint on \(f\).

</div>

---

## 10.2 Convex Functions: Jensen's Inequality as the Algebra of Generalization

> **Core theorem-sentence 10.2.** Jensen's inequality is the formal reason why averaging reduces risk for convex losses; it is the algebraic core behind ensembling, calibration, and stability.

### 10.2.1 Convexity of functions and epigraph geometry

A function \(f:V\to\mathbb{R}\cup\{+\infty\}\) is convex iff its epigraph
\[
\mathrm{epi}(f)=\{(x,t): t\ge f(x)\}
\]
is a convex set. This is the correct geometric encoding: convexity is a statement about the existence of global supporting hyperplanes (linear certificates), not about "roundness."

### 10.2.2 Jensen's inequality, proved from supporting hyperplanes

<div class="prop-box">

**Theorem 10.3 (Jensen).** Let \(f\) be convex on a convex set \(C\subseteq\mathbb{R}^d\), and let \(X\) be a random variable with \(\mathbb{E}[X]\in C\). Then
\[
f(\mathbb{E}[X]) \le \mathbb{E}[f(X)].
\]

</div>

<div class="proof-box">

**Proof.** A standard theorem of convex analysis states: for any \(x_0\) in the interior of \(C\), there exists a supporting hyperplane of the epigraph, equivalently a vector \(g\in\mathbb{R}^d\) and scalar \(b\) such that
\[
f(x) \ge f(x_0) + g^\top (x-x_0)\quad \forall x\in C.
\]
Apply this inequality with \(x_0=\mathbb{E}[X]\):
\[
f(X) \ge f(\mathbb{E}[X]) + g^\top (X-\mathbb{E}[X]).
\]
Take expectation:
\[
\mathbb{E}[f(X)] \ge f(\mathbb{E}[X]) + g^\top (\mathbb{E}[X]-\mathbb{E}[X]) = f(\mathbb{E}[X]).
\]
\(\square\)

</div>

### 10.2.3 ML consequences that are often stated but rarely justified

<div class="ml-box">

1) **Ensembling for convex losses.** If \(\ell(\cdot,y)\) is convex in prediction, then for models \(f_1,\dots,f_k\) and mixture \(\bar{f}=\sum_i \alpha_i f_i\),
\[
\ell(\bar{f}(x),y) \le \sum_i \alpha_i \ell(f_i(x),y),
\]
hence averaging predictors cannot increase pointwise loss for convex \(\ell\). This is the mathematical core of why linear ensembling improves MSE and often improves logistic loss.

2) **Stochastic prediction smoothing.** If one injects noise \(\xi\) and predicts \(f(x+\xi)\), then Jensen yields
\[
\ell\big(\mathbb{E}[f(x+\xi)],y\big) \le \mathbb{E}[\ell(f(x+\xi),y)],
\]
which is a formal explanation for why test-time averaging (e.g., MC dropout) can stabilize convex-risk objectives.

</div>

---

## 10.3 Convex Cones and the PSD Cone: Legality of Correlation, Collapse as Boundary Degeneracy

> **Core theorem-sentence 10.4.** The PSD cone is the legality domain of covariance; collapse is rank-degeneration to the boundary; anti-collapse losses are interior-point pressures.

### 10.3.1 PSD cone and covariance

\[
\mathbb{S}^d_+ := \{A\in\mathbb{S}^d:\ v^\top A v \ge 0\ \forall v\}.
\]

<div class="prop-box">

**Proposition 10.5.** If \(Z\in\mathbb{R}^d\) with \(\mu=\mathbb{E}[Z]\) and \(\Sigma=\mathbb{E}[(Z-\mu)(Z-\mu)^\top]\), then \(\Sigma\in\mathbb{S}^d_+\).

</div>

<div class="proof-box">

**Proof.** For any \(v\), \(v^\top \Sigma v=\mathbb{E}[(v^\top(Z-\mu))^2]\ge0\). \(\square\)

</div>

### 10.3.2 Self-supervised collapse understood as cone-boundary attraction

<div class="ml-box">

In modern SSL, one often trains \(\phi(x)\in\mathbb{R}^d\). Collapse corresponds to \(\phi(x)\approx c\), hence \(\Sigma\approx 0\) (boundary of \(\mathbb{S}^d_+\)). Losses that penalize small per-dimension variance or nonzero off-diagonal covariance implicitly enforce \(\Sigma\) to be well-conditioned, i.e., keep eigenvalues away from zero—geometrically, to remain in the interior strata of the PSD cone.

A mathematically explicit "interiority surrogate" is a log-determinant barrier:
\[
\Omega(\Sigma) = -\log\det(\Sigma+\varepsilon I),
\]
which diverges as eigenvalues approach \(0\). Even when not used directly, many anti-collapse regularizers are approximations to such barrier behavior.

</div>

---

## 10.4 Duality and Separation: Why Convex Problems Admit Certificates

> **Core theorem-sentence 10.6.** Convexity is the condition under which optimization becomes verification: dual variables are proofs.

### 10.4.1 Separation theorem as the primitive of duality

<div class="prop-box">

**Theorem 10.7 (Separation, finite-dimensional).** If \(C\subset\mathbb{R}^d\) is closed, convex, and \(x_0\notin C\), then there exist \(w\neq 0\) and \(b\) s.t.
\[
w^\top x \le b\ \forall x\in C,\qquad w^\top x_0 > b.
\]

</div>

The point is structural: if the feasible set is convex, then infeasibility can be witnessed by a linear functional. This is exactly what makes robust optimization and adversarial certification possible.

### 10.4.2 ML example: adversarial robustness as a support-function computation

<div class="ml-box">

Consider a linear classifier \(f_w(x)=\mathrm{sign}(w^\top x)\). Under an \(\ell_p\)-bounded perturbation \(\|\delta\|_p\le\varepsilon\), the worst-case margin is
\[
\min_{\|\delta\|_p\le\varepsilon} w^\top (x+\delta)
=
w^\top x + \min_{\|\delta\|_p\le\varepsilon} w^\top \delta.
\]
The inner term is \(-\varepsilon \|w\|_q\), where \(q\) is the dual exponent (\(1/p+1/q=1\)), because
\[
\sup_{\|\delta\|_p\le1} w^\top \delta = \|w\|_q
\]
(the support function of the \(\ell_p\) ball). Hence robust margin equals \(w^\top x - \varepsilon \|w\|_q\).

This is convex geometry, not heuristics: robustness reduces to a support-function evaluation, and dual norms are the exact dual certificates.

</div>

---

## 10.5 Minimax and GAN: Convexity as the Condition for Equilibrium

> **Core theorem-sentence 10.8.** Minimax equality is a convex-compact phenomenon; nonconvex parameterizations destroy guarantees; measure-level reformulations restore them.

### 10.5.1 Von Neumann minimax for bilinear games (proof sketched earlier, now used)

For mixed strategies in simplices \(\Delta_m,\Delta_n\) and payoff \(p^\top A q\), minimax equality holds by LP strong duality. This is the canonical template: convexity + compactness + bilinearity yields equilibrium.

### 10.5.2 ML translation: why WGAN changes the state space rather than "improves loss"

<div class="ml-box">

In a classical GAN, the game is played in parameter space \((\theta,\psi)\) and is nonconvex/nonconcave. The minimax theorem does not apply. WGAN shifts the object of optimization toward measures and Lipschitz critics:
\[
W_1(P_{\text{data}},P_\theta)=\sup_{\|f\|_{\mathrm{Lip}}\le1}\big(\mathbb{E}_{P_{\text{data}}}[f]-\mathbb{E}_{P_\theta}[f]\big),
\]
which is convex in \(P_\theta\) and linear in expectations. The "convexity recovery" is not complete at the parameter level, but it is structurally meaningful at the distribution level: the target object is now a convex set (probability measures), and the critic constraint is a convex set (Lipschitz ball).

</div>

---

## 10.6 Wide-Layer / Mean-Field Limits: Deep Learning as Convex Optimization over Measures

> **Core theorem-sentence 10.9.** In the wide-layer limit, a network is linear in a parameter measure; with convex loss in prediction, population risk becomes convex in that measure.

### 10.6.1 Two-layer networks as integrals over neuron measures

For
\[
f_\theta(x)=\frac{1}{m}\sum_{i=1}^m a_i \sigma(w_i^\top x),
\]
define the empirical measure
\[
\rho_\theta=\frac{1}{m}\sum_{i=1}^m \delta_{(a_i,w_i)}.
\]
Then exactly:
\[
f_\theta(x)=\int a\,\sigma(w^\top x)\,d\rho_\theta(a,w).
\]

### 10.6.2 Convexity of risk in \(\rho\), proved without gaps

Let \(\ell(y,\hat{y})\) be convex in \(\hat{y}\) and define
\[
\mathcal{R}(\rho)=\mathbb{E}\big[\ell(Y,f_\rho(X))\big].
\]

<div class="prop-box">

**Theorem 10.10.** \(\mathcal{R}\) is convex in \(\rho\).

</div>

<div class="proof-box">

**Proof.** For \(\rho_\lambda=\lambda\rho_1+(1-\lambda)\rho_2\),
\[
f_{\rho_\lambda}(x)=\lambda f_{\rho_1}(x)+(1-\lambda)f_{\rho_2}(x)
\]
by linearity of the integral in the measure. Then for each \((x,y)\),
\[
\ell\big(y,f_{\rho_\lambda}(x)\big)
\le
\lambda \ell\big(y,f_{\rho_1}(x)\big)+(1-\lambda)\ell\big(y,f_{\rho_2}(x)\big)
\]
by convexity of \(\ell(y,\cdot)\). Take expectation to obtain
\[
\mathcal{R}(\rho_\lambda)\le \lambda\mathcal{R}(\rho_1)+(1-\lambda)\mathcal{R}(\rho_2).
\]
\(\square\)

</div>

### 10.6.3 ML interpretation: overparameterization as convexification by lifting

<div class="ml-box">

Finite networks optimize over particles \((a_i,w_i)\). The lifted object \(\rho\) lives in a convex space (measures). The apparent nonconvexity is partly the artifact of representing a measure by finitely many atoms. Increasing width increases the resolution of that atomic approximation; the dynamics becomes closer to a deterministic gradient flow in \(\rho\)-space, where convexity can assert stability properties that are invisible in \(\theta\)-space.

</div>

---

## 10.7 Scholium: What Convexity Really Means in ML (and what it does not)

<div class="scholium-box">

1) Convexity is not "nice curvature." It is mixture-closure plus the existence of linear certificates (separation).  
2) Many "deep learning tricks" are covert convexity restorations: ensembling, smoothing, Wasserstein lifting, wide-layer lifting, PSD interior regularization.  
3) Convexification is often achieved not by changing the algorithm but by changing the state variable: from parameters to predictions, from predictions to measures, from features to covariances.

</div>

---

## 10.8 Closing the Volume

Part A has built the deterministic substrate: algebra (spaces), operator dynamics (spectra), interaction syntax (tensors), quantitative stability (norms), learnability constraints (Hilbert/Banach), initialization laws (RMT), and finally the geometry of certainty (convexity).

The next logical object is unavoidable: once the relevant state variables become distributions and measures, we must formalize observability, integration, and change of measure. This is not "probability as applied statistics"; it is the legal foundation that makes generalization, stochastic optimization, and modern generative learning mathematically well-typed.

**Next chapter (proposed): Chapter 011 &mdash; Measure Theory and Probability: What Can Be Observed, What Can Be Integrated, and What Data Can Legitimately Constrain.**

*Next: [Chapter 11: Measure Theory and Probability](/book/chapters/chapter011/)*
