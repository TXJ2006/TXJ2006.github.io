---
title: "Chapter 13: Radonâ€“Nikodym Derivatives and Density Ratios"
layout: "single"
url: "/book/chapters/chapter013/"
summary: "Radonâ€“Nikodym derivatives as the computable primitive of measure comparison; density ratios behind covariate shift, GANs, contrastive learning, and off-policy RL."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 13
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

## Part B &mdash; Probability &amp; Measure

## Chapter 13 &mdash; Radon&ndash;Nikodym Derivatives and Density Ratios: Relativity of Measures and the Statistical Physics of Classifiers

*Xujiang Tang*

</div>

## Abstract

Chapter 12 established that "expectation" is a Lebesgue integral with respect to an unknown environment measure. Modern machine learning rarely operates in a single measure universe: we compare training vs. test (covariate shift), real vs. generated (GANs), joint vs. product measures (contrastive learning), behavior vs. target policies (off-policy RL), and base vs. pushforward measures (normalizing flows). The common mathematical obstruction is that *absolute* densities are coordinate- and reference-measure dependent; what remains invariant, operational, and often computable is the *relative density* â€?the Radon&ndash;Nikodym derivative.

The thesis of this chapter is deliberately strict: in high dimensions, the quantity that can be estimated, optimized, and transported is not a density \(p\) in isolation, but a density ratio \(\frac{d\mu}{d\nu}\). Many "algorithms" can be re-read as different ways of parameterizing, estimating, or regularizing this ratio.

---

## 13.1 Epistemological Break: "Probability Density" Is Not Absolute

> **Theorem-sentence 13.1.** A probability density function is a Radon&ndash;Nikodym derivative with respect to a chosen reference measure; only density ratios are coordinate-free objects of comparison.

Let \((\Omega,\mathcal{F})\) be a measurable space and let \(\mu,\nu\) be \(\sigma\)-finite measures on \((\Omega,\mathcal{F})\). A "density" exists only relative to a reference measure. If \(\mu \ll \lambda\) (Lebesgue measure), then \(p = \frac{d\mu}{d\lambda}\) is a density; if we replace \(\lambda\) by another reference \(\tilde{\lambda}\), then the density changes by the chain rule:
\[
\frac{d\mu}{d\tilde{\lambda}} = \frac{d\mu}{d\lambda}\cdot \frac{d\lambda}{d\tilde{\lambda}},
\]
whenever these derivatives are well-defined. Hence "absolute probability density" is not a primitive notion; it is a derivative in a measure-theoretic calculus.

<div class="ml-box">

A practical ML translation appears immediately in continuous generative modeling: normalizing flows do not compute an absolute "probability of a point" in a vacuum; they compute the derivative of a pushforward measure with respect to Lebesgue measure, and the Jacobian determinant is precisely the density-change factor dictated by measure transformation.

</div>

---

## 13.2 The Radonâ€“Nikodym Theorem as the Legality Condition for Comparing Worlds

> **Theorem-sentence 13.2.** Two measure worlds can be compared by a measurable ratio \(\frac{d\mu}{d\nu}\) if and only if \(\mu\) is absolutely continuous with respect to \(\nu\); this derivative is unique (a.e.) and yields the change-of-measure identity.

### 13.2.1 Statement (Ïƒ-finite Radonâ€“Nikodym theorem)

<div class="prop-box">

**Theorem 13.2 (Radon&ndash;Nikodym).** Let \(\mu,\nu\) be \(\sigma\)-finite measures on \((\Omega,\mathcal{F})\). If \(\mu \ll \nu\), then there exists a \(\mathcal{F}\)-measurable function \(h:\Omega\to[0,\infty)\) such that for every \(A\in\mathcal{F}\),
\[
\mu(A) = \int_A h \, d\nu.
\]
The function \(h\) is unique \(\nu\)-almost everywhere and is denoted \(h = \frac{d\mu}{d\nu}\).

</div>

### 13.2.2 Consequence (change-of-measure formula)

For every \(\mathcal{F}\)-measurable \(f\ge 0\) (or integrable with sign),
\[
\int_\Omega f \, d\mu = \int_\Omega f \cdot \frac{d\mu}{d\nu}\, d\nu.
\]

<div class="ml-box">

This identity is the mathematical core behind importance weighting, covariate shift correction, and off-policy evaluation: all are instances of evaluating an integral in one measure using samples from another.

</div>

---

## 13.3 Proof of the Radonâ€“Nikodym Theorem (Step-by-step, finite case first)

> **Theorem-sentence 13.3.** The Radon&ndash;Nikodym derivative can be constructed by a maximality argument over simple functions using only absolute continuity, Ïƒ-additivity, and monotone convergence.

### 13.3.1 Finite measures: \(\mu(\Omega)<\infty\), \(\nu(\Omega)<\infty\), and \(\mu\ll\nu\)

<div class="proof-box">

**Step 1 (the admissible class).** Define
\[
\mathcal{C} := \left\{ g\ge 0 \text{ measurable} : \int_A g\, d\nu \le \mu(A)\;\text{for all } A\in\mathcal{F} \right\}.
\]
This class is nonempty because \(g\equiv 0\in\mathcal{C}\).

**Step 2 (maximize the total mass).** Let
\[
\alpha := \sup_{g\in\mathcal{C}} \int_\Omega g\, d\nu.
\]
Choose a sequence \((g_n)\subset\mathcal{C}\) such that \(\int g_n\,d\nu \uparrow \alpha\).

**Step 3 (form an increasing candidate).** Define \(h_n := \max\{g_1,\dots,g_n\}\). Then \((h_n)\) is increasing and
\[
\int_\Omega h_n\, d\nu \ge \int_\Omega g_n\, d\nu \uparrow \alpha.
\]
To verify \(h_n\in\mathcal{C}\): fix any \(A\). Since \(h_n=\max_{k\le n} g_k\), decompose \(A=\bigsqcup_{k=1}^n A_k\) where \(A_k=\{h_n=g_k\}\cap A\). Because each \(g_k\in\mathcal{C}\),
\[
\int_A h_n\, d\nu = \sum_{k=1}^n \int_{A_k} g_k\, d\nu \le \sum_{k=1}^n \mu(A_k) = \mu(A).
\]
Hence \(h_n\in\mathcal{C}\).

**Step 4 (take the limit).** Let \(h := \lim_{n\to\infty} h_n\) (pointwise). By monotone convergence,
\[
\int_\Omega h\, d\nu = \lim_{n\to\infty}\int_\Omega h_n\, d\nu = \alpha, \qquad \int_A h\, d\nu = \lim_{n\to\infty}\int_A h_n\, d\nu \le \mu(A),
\]
so \(h\in\mathcal{C}\).

**Step 5 (show equality).** Define \(\eta(A) := \mu(A) - \int_A h\, d\nu \ge 0\). Since \(\mu\ll\nu\), so \(\eta\ll\nu\). Suppose for contradiction \(\eta(\Omega)>0\). Define
\[
c := \inf\left\{ \frac{\eta(E)}{\nu(E)} : E\subseteq \Omega,\; \nu(E)>0 \right\}.
\]
Since \(\eta\ll\nu\) and \(\eta(\Omega)>0\), there exists a set \(A\) with \(\nu(A)>0\) and \(\eta(A)>0\). One can show \(c>0\) using the uniform absolute continuity of \(\eta\) w.r.t. \(\nu\). Define \(g := h + c\,\mathbf{1}_A\). For any measurable \(E\),
\[
\int_E g\, d\nu = \int_E h\, d\nu + c\,\nu(E\cap A) \le \mu(E) - \eta(E) + \eta(E\cap A) \le \mu(E),
\]
so \(g\in\mathcal{C}\). But \(\int g\,d\nu = \alpha + c\,\nu(A) > \alpha\), contradicting maximality. Therefore \(\eta\equiv 0\) and \(\mu(A)=\int_A h\,d\nu\) for all \(A\).

**Step 6 (uniqueness).** If \(h'\) also satisfies \(\mu(A)=\int_A h'\,d\nu\) for all \(A\), then \(\int_A (h-h')\,d\nu=0\) for all \(A\). Taking \(A=\{h>h'\}\) forces \(\nu(\{h>h'\})=0\), and symmetrically \(\nu(\{h'<h\})=0\). Hence \(h=h'\) \(\nu\)-a.e. \(\square\)

</div>

### 13.3.2 Reduction to Ïƒ-finite measures

If \(\nu\) is \(\sigma\)-finite, write \(\Omega_k\uparrow \Omega\) with \(\nu(\Omega_k)<\infty\). Apply the finite-case theorem on each \(\Omega_k\) to obtain \(h_k=\frac{d\mu_k}{d\nu_k}\). By uniqueness on overlaps, the \(h_k\) are consistent \(\nu\)-a.e. and define a global \(h\) satisfying \(\mu(A)=\int_A h\,d\nu\) for all \(A\).

---

## 13.4 Density Ratios as the Computable Primitive in High Dimensions

> **Theorem-sentence 13.4.** In high-dimensional models with intractable normalizers, density ratios eliminate partition functions and reduce comparison to energy differences.

A central obstruction in modern generative modeling is normalization. Energy-based models often take the form
\[
p_\theta(x)=\frac{\exp(-E_\theta(x))}{Z_\theta}, \qquad Z_\theta=\int \exp(-E_\theta(x))\,d\lambda(x),
\]
where \(Z_\theta\) is generally intractable in high dimension. Yet ratios often cancel \(Z_\theta\):
\[
\frac{p_\theta(x)}{p_\phi(x)} = \exp\big(-(E_\theta(x)-E_\phi(x))\big)\cdot \frac{Z_\phi}{Z_\theta}.
\]

<div class="ml-box">

If the ratio is only used up to a multiplicative constant (as in many classification-based estimators and contrastive objectives), the global normalizer ratio becomes irrelevant. This is the measure-theoretic reason "relative comparison" survives dimensionality where absolute normalization fails.

</div>

---

## 13.5 Classifiers Estimate Density Ratios (Exact, Not Metaphorical)

> **Theorem-sentence 13.5.** The Bayes-optimal classifier between two measures encodes the Radon&ndash;Nikodym derivative; its logit is the log density ratio up to an additive constant determined by class priors.

### 13.5.1 Setup: a binary mixture world

Let \(P\) and \(Q\) be probability measures on \((\Omega,\mathcal{F})\). Form a mixture experiment: sample \(Y\in\{0,1\}\) with \(\mathbb{P}(Y=1)=\pi\), then sample \(X\mid Y=1\sim P\), \(X\mid Y=0\sim Q\). Assume both measures are absolutely continuous w.r.t. a common \(\sigma\)-finite reference \(\rho\), with densities \(p=\frac{dP}{d\rho}\), \(q=\frac{dQ}{d\rho}\). Then the posterior is
\[
\mathbb{P}(Y=1\mid X=x) = \frac{\pi p(x)}{\pi p(x)+(1-\pi)q(x)}.
\]

### 13.5.2 Derivation of the logit identity

<div class="proof-box">

Define the Bayes-optimal discriminator \(D^*(x)=\mathbb{P}(Y=1\mid X=x)\). Then
\[
\frac{D^*(x)}{1-D^*(x)} = \frac{\pi p(x)}{(1-\pi)q(x)}.
\]
Taking logs:
\[
\log\frac{D^*(x)}{1-D^*(x)} = \log\frac{p(x)}{q(x)} + \log\frac{\pi}{1-\pi}.
\]
Up to the prior constant \(\log\frac{\pi}{1-\pi}\), the logit is the log RN derivative \(\log\frac{dP}{dQ}\). \(\square\)

</div>

### 13.5.3 A minimal numeric example

Let \(P=\mathcal{N}(0,1)\) and \(Q=\mathcal{N}(m,1)\) on \(\mathbb{R}\), equal priors \(\pi=\tfrac{1}{2}\). Then
\[
\log\frac{p(x)}{q(x)} = -\frac{x^2}{2}+\frac{(x-m)^2}{2} = mx-\frac{m^2}{2}.
\]

<div class="ml-box">

The optimal logit is linear in \(x\), and the decision boundary is \(x=\frac{m}{2}\). This illustrates an important structural point: *density ratio estimation can be easier than density estimation*, because the ratio may lie in a simpler function class even when each density is complex.

</div>

---

## 13.6 Covariate Shift as a Change-of-Measure Identity (Not a Heuristic)

> **Theorem-sentence 13.6.** Under covariate shift (\(P_{\text{train}}(y\mid x)=P_{\text{test}}(y\mid x)\)), the test risk equals a reweighted train risk with weights given by a Radon&ndash;Nikodym derivative on the feature marginal.

Let \(P\) and \(Q\) be train and test distributions on \(\mathcal{X}\times\mathcal{Y}\). Suppose \(P(y\mid x)=Q(y\mid x)\) and \(Q_X\ll P_X\). Define
\[
w(x):=\frac{dQ_X}{dP_X}(x), \qquad g(x):=\int_{\mathcal{Y}} \ell(f(x),y)\,P(dy\mid x).
\]
By the covariate-shift assumption and the change-of-measure identity,
\[
\mathcal{R}_Q(f) = \int_{\mathcal{X}} g(x)\, Q_X(dx) = \int_{\mathcal{X}} g(x)\, w(x)\, P_X(dx) = \mathbb{E}_{(X,Y)\sim P}[w(X)\,\ell(f(X),Y)].
\]

<div class="ml-box">

This is not an approximation; it is an identity. The only approximation in practice is the estimation of \(w\). If \(w\) is heavy-tailed or unbounded (weak overlap), reweighting has high variance and becomes statistically unstable â€?not "bad engineering," but the analytic consequence of comparing measures with insufficient absolute continuity on the relevant support.

</div>

---

## 13.7 GANs, NCE, and Contrastive Learning as Structured Density-Ratio Fitting

> **Theorem-sentence 13.7.** GAN discriminators, NCE objectives, and InfoNCE-style contrastive losses all implement density-ratio estimation between two explicit measures; the differences are which measures are compared and which variational family parameterizes the ratio.

### 13.7.1 GAN optimal discriminator (measure-theoretic reading)

Let \(P_{\text{data}}\) and \(P_\theta\) have densities \(p\) and \(q_\theta\) w.r.t. a common reference. The standard GAN discriminator maximizes
\[
\mathbb{E}_{X\sim P_{\text{data}}}[\log D(X)] + \mathbb{E}_{X\sim P_\theta}[\log(1-D(X))].
\]
Pointwise optimization gives \(D^*(x)=\frac{p(x)}{p(x)+q_\theta(x)}\), hence
\[
\frac{D^*(x)}{1-D^*(x)}=\frac{p(x)}{q_\theta(x)} = \frac{dP_{\text{data}}}{dP_\theta}(x).
\]

<div class="ml-box">

When supports barely overlap, this derivative becomes ill-behaved and gradients collapse â€?precisely the absolute-continuity pathology, not a mysterious optimization curse.

</div>

### 13.7.2 Noise-Contrastive Estimation (NCE)

In NCE, one compares the data measure \(P_{\text{data}}\) to a known noise measure \(P_n\). The classifier separating data vs. noise estimates \(\log\frac{dP_{\text{data}}}{dP_n}\). If a model is specified as an unnormalized density \(\tilde{p}_\theta(x)=\exp(-E_\theta(x))\), fitting the ratio against noise allows one to learn \(E_\theta\) without explicitly computing \(Z_\theta\). The "partition function problem" is bypassed because the objective is expressed via a ratio against a known \(P_n\).

### 13.7.3 Contrastive learning (InfoNCE as joint-vs-product ratio)

Let \(P_{XY}\) be a joint measure (positive pairs) and \(P_X\otimes P_Y\) the product of marginals (negatives). The optimal scoring function \(s^*(x,y)\) separating joint samples from product samples is, up to constants,
\[
s^*(x,y) = \log\frac{dP_{XY}}{d(P_X\otimes P_Y)}(x,y),
\]
i.e., the log density ratio defining mutual information. Contrastive objectives learn dependence structure by fitting an RN derivative in a lifted product space.

---

## 13.8 Off-Policy Reinforcement Learning: RN Derivatives on Trajectory Space

> **Theorem-sentence 13.8.** Importance sampling in off-policy RL is the Radon&ndash;Nikodym derivative between trajectory measures induced by two policies; variance explosion is a structural consequence of multiplying many local RN factors.

Consider a Markov decision process with trajectory \(\tau=(s_0,a_0,s_1,a_1,\dots,s_T)\). A policy \(\pi\) induces a measure \(\mathbb{P}^\pi\) on trajectory space. For a behavior policy \(\mu\) and target policy \(\pi\), under standard support conditions, the RN derivative factorizes:
\[
\frac{d\mathbb{P}^\pi}{d\mathbb{P}^\mu}(\tau) = \prod_{t=0}^{T-1} \frac{\pi(a_t\mid s_t)}{\mu(a_t\mid s_t)}.
\]
For any trajectory functional \(F(\tau)\),
\[
\mathbb{E}_{\tau\sim \mathbb{P}^\pi}[F(\tau)] = \mathbb{E}_{\tau\sim \mathbb{P}^\mu}\!\left[\frac{d\mathbb{P}^\pi}{d\mathbb{P}^\mu}(\tau)\,F(\tau)\right].
\]

<div class="ml-box">

The practical problem â€?variance blow-up â€?has a mathematical diagnosis: RN derivatives on long horizons are products of many random ratios, often heavy-tailed. Clipping, per-decision correction, and control variates are not ad hoc tricks; they are variance-regularization strategies for unstable density ratios in high-dimensional trajectory spaces.

</div>

---

## 13.9 Scholium: What This Chapter Adds

<div class="scholium-box">

1. Density is not absolute; it is a derivative relative to a reference measure.  
2. Comparability of worlds requires absolute continuity; without it, RN derivatives and KL-type quantities become ill-defined or infinite.  
3. The RN derivative is the correct object behind reweighting, domain adaptation, off-policy evaluation, adversarial learning, and contrastive learning.  
4. "Classifier = density-ratio estimator" is a theorem: the Bayes logit equals the log RN derivative (up to priors).  
5. Many modern objectives avoid intractable normalizers because ratios annihilate partition functions, reducing global integration to local energy differences.

</div>

This closes the measure-relativity layer of Part B: we can now compare, transport, and reweight between measures by a principled derivative calculus.

**Chapter 014: Information Geometry and Cross-Entropy &mdash; Divergences as Riemannian Structure on the Manifold of Measures.**
