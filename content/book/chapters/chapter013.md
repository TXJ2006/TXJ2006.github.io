---
title: "Chapter 13: Information Theory & Entropy"
layout: "single"
url: "/book/chapters/chapter013/"
summary: "Entropy, cross-entropy, KL divergence, and mutual information as the canonical log-additive geometry of probability measures; maximum likelihood as divergence minimization."
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

## Part B &mdash; Probability &amp; Measure

## Chapter 13 &mdash; Information Theory &amp; Entropy: Logarithmic Functionals, Learning as Compression, and the Geometry of Likelihood

*Xujiang Tang*

</div>

## Abstract

Part B has established that uncertainty is not "noise in numbers," but structure in a probability measure \((\Omega,\mathcal{F},\mathbb{P})\), and that expectation is a Lebesgue integral. This chapter identifies the unique functional that converts probability mass into additive, optimizable "cost": the logarithm. The central claim is not rhetorical but structural: once we accept that learning should aggregate independent evidence additively, the negative log-likelihood (hence cross-entropy) becomes the canonical risk. Entropy, KL divergence, and mutual information then appear as the intrinsic geometry of probability measures under this log-additive calculus.

---

## 13.1 The Epistemological Break: Why the Logarithm Is Forced

> **Core theorem-sentence 13.1.** If evidence aggregates by product of probabilities, then any additive loss must be (a constant multiple of) the negative logarithm; thus log-loss is not a "choice," but the unique lawful scoring rule under multiplicative uncertainty.

### 13.1.1 From multiplicative uncertainty to additive cost

Let \(p\in(0,1]\) denote the probability assigned to a realized event. Suppose we seek a cost \(\ell(p)\) satisfying the minimal rationality axiom:

<div class="def-box">

**Axiom (Additivity under independent evidence).** If two independent pieces of evidence have probabilities \(p\) and \(q\), then the combined probability is \(pq\), and the combined cost should satisfy
\[
\ell(pq) = \ell(p) + \ell(q).
\]

</div>

Assume \(\ell\) is measurable and not pathological (e.g., locally bounded). Then Cauchy's functional equation on \((0,1]\) yields
\[
\ell(p) = -c\log p \quad\text{for some } c>0.
\]
(Here \(c\) sets the unit: \(c=1\) for nats, \(c=1/\log 2\) for bits.)

This is the first "type discipline" in probabilistic learning: if your model outputs probabilities and you want a sum over samples as objective, the log is forced.

### 13.1.2 ML insertion: why training objectives sum over samples

<div class="ml-box">

In supervised learning with i.i.d. data \((X_i,Y_i)\sim P\), empirical risk is a sample average:
\[
\widehat{\mathcal{R}}(\theta)=\frac{1}{n}\sum_{i=1}^n L_\theta(X_i,Y_i).
\]
If the model assigns conditional probabilities \(p_\theta(y\mid x)\), and we require additive aggregation of evidence across independent samples, then \(L_\theta(x,y)\) must be proportional to \(-\log p_\theta(y\mid x)\). This directly explains why modern LLM training uses token-level cross-entropy: the training set is treated as (approximately) independent evidence at the token factorization level, and the objective must be additive.

</div>

---

## 13.2 Entropy: The Lebesgue Integral of Surprise

> **Core theorem-sentence 13.2.** Entropy is the expected log-cost of observing an outcome under its own law; it is the intrinsic "uncertainty mass" of a distribution, defined as a Lebesgue integral, not as a combinatorial heuristic.

### 13.2.1 Definition (discrete)

Let \(P\) be a distribution on a countable alphabet \(\mathcal{X}\) with mass function \(p(x)\). The Shannon entropy is
\[
H(P) = \mathbb{E}_{X\sim P}\big[-\log p(X)\big] = \sum_{x\in\mathcal{X}} p(x)\,(-\log p(x)).
\]
This is a Lebesgue integral with respect to the counting measure (equivalently, an expectation).

### 13.2.2 Definition (continuous) and the measure-theoretic warning

If \(P\ll \lambda\) (Lebesgue measure) with density \(p\), the differential entropy is
\[
h(P)=\int_{\mathbb{R}^d} p(x)\,(-\log p(x))\,dx.
\]
Unlike discrete entropy, \(h(P)\) is not invariant under reparameterizations; this is not a defect but a measure-theoretic fact: densities depend on the reference measure.

<div class="ml-box">

A practical ML corollary: whenever one speaks about "entropy of continuous embeddings," one must specify the reference measure or use invariant objects (e.g., KL divergences, mutual information under fixed measures, or relative entropies).

</div>

---

## 13.3 Cross-Entropy and Maximum Likelihood: Learning as Risk Minimization in Measure Space

> **Core theorem-sentence 13.3.** Maximum likelihood estimation is exactly empirical cross-entropy minimization; generalization is the statement that empirical Lebesgue integrals converge to population integrals under the data-generating measure.

### 13.3.1 Cross-entropy

For distributions \(P\) and \(Q\) on \(\mathcal{X}\) (with \(P\ll Q\) in the discrete sense \(q(x)=0\Rightarrow p(x)=0\)), define
\[
H(P,Q) = \mathbb{E}_{X\sim P}\big[-\log q(X)\big] = \sum_x p(x)\,(-\log q(x)).
\]
This is the expected log-loss incurred if the world is \(P\) but we code/predict with \(Q\).

### 13.3.2 Maximum likelihood is cross-entropy minimization (step-by-step)

Let data \(x_1,\dots,x_n\) be i.i.d. from \(P\). Consider a parametric model family \(\{Q_\theta\}\) with pmf/pdf \(q_\theta\). The log-likelihood is
\[
\log L(\theta)=\sum_{i=1}^n \log q_\theta(x_i).
\]
Maximizing likelihood is equivalent to minimizing negative log-likelihood:
\[
\arg\max_\theta \log L(\theta) = \arg\min_\theta \Big(-\sum_{i=1}^n \log q_\theta(x_i)\Big).
\]
Divide by \(n\):
\[
\arg\min_\theta \frac{1}{n}\sum_{i=1}^n \big(-\log q_\theta(x_i)\big).
\]
Define the empirical measure \(\widehat{P}_n=\frac{1}{n}\sum_i\delta_{x_i}\). Then
\[
\frac{1}{n}\sum_{i=1}^n \big(-\log q_\theta(x_i)\big) = \int \big(-\log q_\theta(x)\big)\,d\widehat{P}_n(x) = H(\widehat{P}_n,Q_\theta).
\]
Thus MLE is empirical cross-entropy minimization. Under suitable conditions, \(\widehat{P}_n\Rightarrow P\) and the empirical integral converges to the population integral:
\[
H(\widehat{P}_n,Q_\theta)\to H(P,Q_\theta).
\]
So MLE is the procedure "choose \(Q_\theta\) that minimizes the population cross-entropy," estimated by the empirical cross-entropy.

### 13.3.3 ML example: LLM next-token training as conditional cross-entropy

<div class="ml-box">

Let a sequence \(S=(s_1,\dots,s_T)\) be drawn from an unknown distribution. A common modeling choice factorizes
\[
q_\theta(S)=\prod_{t=1}^T q_\theta(s_t\mid s_{<t}).
\]
The negative log-likelihood decomposes additively:
\[
-\log q_\theta(S)=\sum_{t=1}^T -\log q_\theta(s_t\mid s_{<t}).
\]
Training by minimizing the dataset average of this quantity is precisely minimizing an empirical conditional cross-entropy. "Perplexity" is simply \(\exp\) of the average token-level cross-entropy, i.e., the exponential of an expected log-loss.

</div>

---

## 13.4 KL Divergence: The Only Correct Notion of Regret Under Log-Loss

> **Core theorem-sentence 13.4.** KL divergence is the excess cross-entropy (regret) of using \(Q\) when the truth is \(P\); its nonnegativity is a theorem (Gibbs inequality), not an assumption.

### 13.4.1 Definition and identity

Define the KL divergence
\[
D_{\mathrm{KL}}(P\|Q) = \mathbb{E}_{X\sim P}\Big[\log\frac{p(X)}{q(X)}\Big] = \sum_x p(x)\log\frac{p(x)}{q(x)}.
\]
Then
\[
H(P,Q)=H(P)+D_{\mathrm{KL}}(P\|Q).
\]

<div class="proof-box">

**Derivation.**
\[
H(P,Q) = \sum_x p(x)(-\log q(x)) = \sum_x p(x)(-\log p(x)) + \sum_x p(x)\log\frac{p(x)}{q(x)} = H(P)+D_{\mathrm{KL}}(P\|Q).
\]

</div>

Thus minimizing cross-entropy in \(Q\) is equivalent to minimizing \(D_{\mathrm{KL}}(P\|Q)\) since \(H(P)\) does not depend on \(Q\).

### 13.4.2 Gibbs inequality (full proof, no skipped steps)

<div class="prop-box">

**Theorem 13.5 (Nonnegativity of KL).** \(D_{\mathrm{KL}}(P\|Q)\ge 0\), with equality iff \(P=Q\) (a.e. on support).

</div>

<div class="proof-box">

**Proof.** Use the elementary inequality \(\log u \le u-1\) for all \(u>0\), with equality iff \(u=1\).  
Let \(u(x)=\frac{q(x)}{p(x)}\) wherever \(p(x)>0\). Then \(-\log u(x)\ge 1-u(x)\). Multiply by \(p(x)\ge 0\) and sum:
\[
\sum_x p(x)\big(-\log u(x)\big) \ge \sum_x p(x)\big(1-u(x)\big).
\]
Left side:
\[
\sum_x p(x)\log\frac{p(x)}{q(x)} = D_{\mathrm{KL}}(P\|Q).
\]
Right side:
\[
\sum_x p(x) - \sum_x p(x)\frac{q(x)}{p(x)} = 1 - \sum_x q(x) = 1-1 = 0.
\]
Hence \(D_{\mathrm{KL}}(P\|Q)\ge 0\). Equality requires \(-\log u(x)=1-u(x)\) for all \(x\) with \(p(x)>0\), hence \(u(x)=1\), i.e., \(q(x)=p(x)\) on the support. \(\square\)

</div>

### 13.4.3 ML consequence: label smoothing as "truth-mixing" in KL geometry

<div class="ml-box">

Label smoothing replaces a one-hot target \(P\) by \(\tilde{P}=(1-\varepsilon)P+\varepsilon U\). Training minimizes \(D_{\mathrm{KL}}(\tilde{P}\|Q_\theta)\), which penalizes overconfident (near-singular) \(Q_\theta\). Geometrically, it keeps the target away from the boundary of the probability simplex where log-loss gradients can become extremely stiff. This is not a heuristic; it is boundary regularization in the convex set of measures.

</div>

---

## 13.5 Mutual Information: Dependence as a Divergence in Product Measure Space

> **Core theorem-sentence 13.6.** Mutual information is KL divergence between the joint measure and the product of marginals; it is the canonical scalar measure of statistical dependence.

### 13.5.1 Definition

For joint \(P_{XY}\) with marginals \(P_X,P_Y\),
\[
I(X;Y)=D_{\mathrm{KL}}(P_{XY}\,\|\,P_X\otimes P_Y) = \mathbb{E}\Big[\log\frac{p_{XY}(X,Y)}{p_X(X)p_Y(Y)}\Big].
\]
Nonnegativity follows immediately from KL nonnegativity. Moreover, \(I(X;Y)=0\) iff \(P_{XY}=P_X\otimes P_Y\), i.e., independence.

### 13.5.2 ML insertion: contrastive learning as controlled MI estimation

<div class="ml-box">

In contrastive methods (e.g., InfoNCE-style objectives), one constructs positive pairs \((x,y)\sim P_{XY}\) and negatives approximating \(P_X\otimes P_Y\). The learning signal is precisely a divergence between "true joint coupling" and "independent coupling." The objective can be read as building a classifier that separates joint samples from product samples; in the limit, such separation is governed by the likelihood ratio \(\log \frac{p_{XY}}{p_Xp_Y}\), i.e., the same object defining mutual information.

The practical thesis: contrastive learning is not "magic representation learning," it is divergence minimization between induced couplings in measure space.

</div>

---

## 13.6 The Radon–Nikodym Lens Revisited: When Log-Loss Is Legal (and When It Is Not)

> **Core theorem-sentence 13.7.** Log-loss and KL are only defined where a Radon–Nikodym derivative exists; failures of absolute continuity are not numerical instabilities but measure-theoretic type errors.

Recall: if \(P\not\ll Q\), then \(\log\frac{dP}{dQ}\) is undefined on sets where \(Q\) assigns zero mass but \(P\) does not. In such cases, \(D_{\mathrm{KL}}(P\|Q)=+\infty\) by definition.

<div class="ml-box">

This is the precise reason the "density-ratio worldview" can break in high-dimensional generative modeling: if \(P_{\text{data}}\) and \(P_\theta\) live on low-dimensional manifolds that do not overlap, absolute continuity fails in both directions. The resulting KL-based objectives either explode or flatten into constant regions, producing gradient starvation. The resolution (e.g., Wasserstein objectives, noise injection, diffusion forward processes) can be understood as *restoring absolute continuity* by convolving with a full-dimensional noise measure so that densities exist and RN derivatives become legal.

</div>

---

## 13.7 Scholium: Information Geometry as the Metric Backbone of Learning

<div class="scholium-box">

1) **Entropy** measures intrinsic uncertainty: \(H(P)=\mathbb{E}_P[-\log p]\).  
2) **Cross-entropy** is the operational training risk: \(H(P,Q)=\mathbb{E}_P[-\log q]\).  
3) **KL divergence** is regret under log-loss: \(D_{\mathrm{KL}}(P\|Q)=H(P,Q)-H(P)\ge0\).  
4) **Mutual information** is dependence as divergence: \(I(X;Y)=D_{\mathrm{KL}}(P_{XY}\|P_X\otimes P_Y)\).

These objects are not optional vocabulary; they are the minimal invariant scalars generated by the triple \((\text{measure},\ \log,\ \text{Lebesgue integral})\).

</div>

---

## 13.8 Closing Transition

We now possess the lawful arithmetic of uncertainty: we can integrate, we can compare measures by divergences, and we can interpret training as compression. The next question is structural: how do these divergences and expectations behave under parameterization, under transformation, and under optimization dynamics?

The next chapter therefore introduces the variational principle that governs modern generative learning and approximate inference:

**Ch 014: Variational Inference &amp; ELBO &mdash; When Optimization Replaces Integration, and Why "Approximation" Must Be a Theorem.**
