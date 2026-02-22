---
title: "Chapter 17: Law of Large Numbers & Central Limit Theorem"
layout: "single"
url: "/book/chapters/chapter017/"
summary: "LLN as almost-sure measure collapse making ERM a lawful surrogate; CLT as the universal Gaussian geometry of aggregation at the 鈭歯 scale; minibatch gradient noise, wide-network Gaussian processes, and scaling laws as direct corollaries."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 17
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

## Chapter 17 &mdash; Law of Large Numbers &amp; Central Limit Theorem: Macroscopic Determinism from Measure Collapse, and the Gaussian Destiny of High-Dimensional Learning

*Xujiang Tang*

</div>

## Abstract

We leave "single events" and "finite-time fluctuations" and enter the regime that actually powers modern ML: **aggregation**. Deep learning is not a theory of one sample; it is a theory of billions of samples, minibatches, and overparameterized sums. Two theorems govern this regime:

- **LLN** is the unique rigidity that makes empirical averages legitimate surrogates for population integrals.
- **CLT** is the unique universality mechanism that turns aggregated randomness into a Gaussian geometry at the \(\sqrt{n}\) scale 鈥?the geometry that appears in minibatch gradient noise, wide-network preactivations, and random feature limits.

---

## 17.0 Setup: Measure-Theoretic Primitives and ML Objects

Let \((\Omega,\mathcal{F},\mathbb{P})\) be a probability space. A random variable \(X:\Omega\to\mathbb{R}\) is integrable if \(\mathbb{E}|X|<\infty\). For ML:

- A data point \(Z=(X,Y)\sim\mathcal{D}\) is a random variable under an unknown data-generating measure \(\mathbb{P}=\mathbb{P}_\mathcal{D}\).
- A loss \(\ell(\theta;Z)\) is measurable and typically integrable.
- The **population risk** is a Lebesgue integral:
\[
R(\theta) := \mathbb{E}[\ell(\theta;Z)] = \int \ell(\theta;z)\,\mathbb{P}(dz).
\]
- The **empirical risk** for i.i.d. sample \(Z_1,\dots,Z_n\) is
\[
\widehat{R}_n(\theta) := \frac{1}{n}\sum_{i=1}^n \ell(\theta;Z_i).
\]

The entire practical ML move "train by averaging losses over the dataset" is the claim that \(\widehat{R}_n(\theta)\) is a good proxy for \(R(\theta)\). LLN is the first reason this can be true; uniform LLN and concentration inequalities (Chapter 18) are the second.

---

## 17.1 Strong Law of Large Numbers: The Collapse of Empirical Measure

> **Theorem-sentence 17.1.** The SLLN is the almost-sure collapse of the empirical measure to the population measure: aggregation kills randomness, not approximately, but on a set of sample paths of probability exactly one.

<div class="prop-box">

**Theorem 17.1 (Strong Law of Large Numbers).** Let \(X_1,X_2,\dots\) be i.i.d. with \(\mathbb{E}|X_1|<\infty\) and \(\mu=\mathbb{E}[X_1]\). Then
\[
\frac{1}{n}\sum_{i=1}^n X_i \;\xrightarrow[n\to\infty]{\;a.s.\;}\; \mu.
\]
Equivalently, \(\mathbb{P}\!\left(\lim_{n\to\infty}\bar{X}_n = \mu\right) = 1\).

</div>

### 17.1.1 Read the formula literally: not "approach," but "a.s. collapse"

Define \(\bar{X}_n := \frac{1}{n}\sum_{i=1}^n X_i\). The SLLN is not "usually close." It states: all worlds except a null set are **forced** to collapse to \(\mu\). In measure language, LLN is the first appearance of the phenomenon:

- *Micro-level:* each \(X_i\) varies arbitrarily in its distribution.
- *Macro-level:* aggregation kills randomness in the topology of almost-sure convergence.

Contrast with the **Weak LLN** (Khinchin), which gives only convergence in probability: \(\bar{X}_n\xrightarrow{\mathbb{P}}\mu\). The strong law gives a stronger topology 鈥?almost surely 鈥?and is the basis for the "single trajectory" ergodic arguments of Chapter 15.

### 17.1.2 ML derivation: why ERM is a lawful surrogate

Set \(X_i := \ell(\theta;Z_i)\). If \(\mathbb{E}|\ell(\theta;Z)|<\infty\), the SLLN gives
\[
\widehat{R}_n(\theta) = \frac{1}{n}\sum_{i=1}^n \ell(\theta;Z_i) \;\xrightarrow[]{\;a.s.\;}\; R(\theta) \quad\text{for fixed }\theta.
\]
This is the minimal physical legality of ERM: for any fixed parameter \(\theta\), empirical risk is a consistent estimator of population risk.

<div class="ml-box">

**Important ML nuance 鈥?where na茂ve treatments cheat.**
Optimization chooses \(\hat\theta_n\in\arg\min_\theta \widehat{R}_n(\theta)\), which depends on the entire sample. The SLLN *alone* does not guarantee \(R(\hat\theta_n)\to\inf_\theta R(\theta)\); that requires **uniform** control over \(\theta\) (Glivenko鈥揅antelli classes, Rademacher complexity, VC dimension). SLLN is the non-negotiable base: if even pointwise convergence fails, no generalization theory can be built on top.

</div>

### 17.1.3 Empirical measure collapse: the correct ML geometric object

<div class="def-box">

**Definition 17.1 (empirical measure).** For i.i.d. sample \(Z_1,\dots,Z_n\), the empirical measure is
\[
\widehat{\mathbb{P}}_n := \frac{1}{n}\sum_{i=1}^n \delta_{Z_i},
\]
where \(\delta_{Z_i}\) is the Dirac mass at \(Z_i\). Then empirical risk is the integral under \(\widehat{\mathbb{P}}_n\):
\[
\widehat{R}_n(\theta) = \int \ell(\theta;z)\,\widehat{\mathbb{P}}_n(dz).
\]

</div>

The SLLN is the statement that \(\widehat{\mathbb{P}}_n\) converges to \(\mathbb{P}\) in a sense strong enough to make these integrals converge for integrable \(\ell(\theta;\cdot)\). The precise topology is weak convergence of measures, combined with integrability conditions (or Glivenko鈥揅antelli for uniform convergence over function classes).

<div class="ml-box">

**The correct mental model for deep learning.** Deep learning is not "fitting a function to points." It is "replacing an unknown measure \(\mathbb{P}\) by a discrete surrogate \(\widehat{\mathbb{P}}_n\) and doing variational calculus on that surrogate." LLN certifies that the surrogate is asymptotically valid; concentration theory (Chapter 18) certifies it at finite \(n\).

</div>

### 17.1.4 Martingale proof sketch of the SLLN

The SLLN can be proved using the Borel鈥揅antelli lemma or via a martingale argument. The martingale route (connecting to Chapter 16) defines \(M_n := \bar{X}_n - \mu\) and shows this is approximately a martingale with diminishing increments. The martingale convergence theorem (Theorem 16.6) then ensures a.s. convergence, provided the Lyapunov-type condition \(\sum_n \mathbb{E}[(X_n/n)^2] < \infty\) holds. This reveals the structural unity: SLLN is a consequence of the same operator-theoretic convergence that underlies SA in RL.

---

## 17.2 Central Limit Theorem: The Only Non-Trivial Microscope Is \(\sqrt{n}\)

> **Theorem-sentence 17.2.** The CLT is the unique universality theorem: at scale \(\sqrt{n}\), all distributions with finite variance converge to the same Gaussian geometry; this is not an approximation but a topological limit in the space of probability measures.

<div class="prop-box">

**Theorem 17.2 (Lindeberg鈥揕茅vy CLT).** Let \(X_1,X_2,\dots\) be i.i.d. with \(\mathbb{E}[X_1]=\mu\) and \(\mathrm{Var}(X_1)=\sigma^2\in(0,\infty)\). Then
\[
\sqrt{n}\,\frac{\bar{X}_n-\mu}{\sigma} \;\xRightarrow[n\to\infty]{}\; \mathcal{N}(0,1).
\]
Here \(\Rightarrow\) denotes convergence in distribution (weak convergence of probability measures).

</div>

### 17.2.1 Why exactly \(\sqrt{n}\)? The variance-preserving scale

Write the sum of centered variables \(S_n := \sum_{i=1}^n (X_i-\mu)\). Then \(\mathbb{E}[S_n]=0\) and \(\mathrm{Var}(S_n)=n\sigma^2\). The three possible normalizations:

- Divide by \(n\): yield \(\bar{X}_n - \mu\to 0\) (LLN scale 鈥?the limit is degenerate at \(0\)).
- No normalization: variance explodes to \(\infty\) (no limit exists in distribution).
- Divide by \(\sqrt{n}\sigma\): variance is pinned to \(1\), and the shape converges universally to Gaussian.

The \(\sqrt{n}\) scale is therefore the **unique non-degenerate scale** for aggregation. The CLT is the characterization of the limiting shape at that scale.

### 17.2.2 ML interpretation: "uncertainty of an average" is Gaussian at \(\sqrt{n}\)

For empirical risk at fixed \(\theta\), set \(X_i=\ell(\theta;Z_i)\). If \(\mathrm{Var}(\ell(\theta;Z))<\infty\),
\[
\sqrt{n}\,\big(\widehat{R}_n(\theta) - R(\theta)\big) \;\xRightarrow[]{}\; \mathcal{N}(0,\sigma_\theta^2), \qquad \sigma_\theta^2 = \mathrm{Var}(\ell(\theta;Z)).
\]

So LLN says "risk estimate becomes correct," and CLT says "its residual fluctuation is Gaussian with standard deviation \(\sim n^{-1/2}\)." This is the quantitative backbone behind:
- statistical error bars in evaluation,
- why doubling data reduces uncertainty by only \(1/\sqrt{2}\),
- why data-scaling laws exhibit power laws near \(-1/2\) in simple variance-dominated regimes.

### 17.2.3 Lindeberg condition: the correct generality

The Lindeberg鈥揕茅vy CLT requires identical distributions. The more general **Lindeberg CLT** allows non-identically distributed \(X_i\) (hence applying to heterogeneous batch samples, position-dependent token losses, etc.) under the condition that no single summand dominates the total variance:

\[
\frac{1}{s_n^2}\sum_{i=1}^n \mathbb{E}\!\left[(X_i-\mu_i)^2\,\mathbf{1}_{|X_i-\mu_i|>\varepsilon s_n}\right] \to 0 \quad\forall\,\varepsilon>0,
\]
where \(s_n^2=\sum_{i=1}^n \mathrm{Var}(X_i)\). This formalizes "no single data point dominates," which is a diversity condition on the training set.

---

## 17.3 CLT in Optimization: Minibatch Gradients Are Gaussian Perturbations

> **Theorem-sentence 17.3.** Minibatch stochastic gradients are CLT objects: LLN makes them unbiased in direction, and CLT characterizes their noise as an isotropic Gaussian perturbation at scale \(1/\sqrt{m}\) in parameter space.

Let the population objective be \(L(\theta):=\mathbb{E}[\ell(\theta;Z)]\). Define the per-sample gradient \(g(\theta;Z):=\nabla_\theta\ell(\theta;Z)\) and the true gradient \(\nabla L(\theta)=\mathbb{E}[g(\theta;Z)]\). For minibatch \(\mathcal{B}=\{Z_{i_1},\dots,Z_{i_m}\}\) (i.i.d.),
\[
\widehat{g}_m(\theta) := \frac{1}{m}\sum_{j=1}^m g(\theta;Z_{i_j}).
\]

### 17.3.1 LLN: unbiasedness and consistency of SGD direction

\[
\mathbb{E}[\widehat{g}_m(\theta)] = \nabla L(\theta), \qquad \widehat{g}_m(\theta) \;\xrightarrow[m\to\infty]{\;a.s.\;}\; \nabla L(\theta).
\]

### 17.3.2 CLT: the exact \(\sqrt{m}\) noise geometry

Assume \(\mathrm{Cov}(g(\theta;Z))=\Sigma(\theta)\) exists and is finite. Then
\[
\sqrt{m}\,\big(\widehat{g}_m(\theta) - \nabla L(\theta)\big) \;\xRightarrow[]{}\; \mathcal{N}(0,\Sigma(\theta)).
\]
For large \(m\), the approximate model is:
\[
\widehat{g}_m(\theta) \approx \nabla L(\theta) + \frac{1}{\sqrt{m}}\,\xi, \qquad \xi\sim\mathcal{N}(0,\Sigma(\theta)).
\]

### 17.3.3 Practical ML consequence: batch size is a temperature knob

The SGD update becomes:
\[
\theta_{t+1} = \theta_t - \eta\,\widehat{g}_m(\theta_t) \approx \theta_t - \eta\,\nabla L(\theta_t) - \frac{\eta}{\sqrt{m}}\,\xi_t.
\]
The noise scale is \(\eta/\sqrt{m}\). This is the analytic origin of several empirical observations:

<div class="ml-box">

1. **Batch size reduces noise like \(m^{-1/2}\).** Doubling the batch size halves the gradient noise standard deviation. This is CLT, not heuristic.

2. **Small-batch generalization advantage.** Larger stochastic noise (\(\eta/\sqrt{m}\) larger) induces more diffusion in parameter space, potentially exploring broader basins. The "flat minima" preference of small-batch SGD (Chapter 15) has a direct CLT interpretation: the effective diffusion coefficient is \(\eta^2\Sigma(\theta)/m\).

3. **Variance-preserving scaling.** Keeping \(\eta/\sqrt{m}\) constant as \(m\) changes preserves the noise scale under the CLT approximation. This is the rigorous version of the "linear scaling rule" (learning rate scales linearly with batch size) used in large-batch training.

</div>

---

## 17.4 CLT in Representation: Why Wide Networks Become Gaussian Processes

> **Theorem-sentence 17.4.** As network width grows, preactivations become jointly Gaussian by CLT; in the infinite-width limit, the random function converges to a Gaussian process whose covariance kernel is determined by the architecture and initialization.

### 17.4.1 One hidden layer: preactivation as a CLT sum

Consider a single hidden layer with width \(m\) and input \(x\in\mathbb{R}^d\):
\[
a_j(x) := \frac{1}{\sqrt{d}}\sum_{k=1}^d W_{jk}\,x_k,
\]
with \(W_{jk}\) i.i.d., mean \(0\), variance \(1\). For fixed \(x\), \(a_j(x)\) is a sum of \(d\) independent terms. By CLT, as \(d\to\infty\):
\[
a_j(x) \;\xRightarrow[]{}\; \mathcal{N}\!\left(0,\,\|x\|^2/d\right).
\]
For two inputs \(x,x'\), the pair \((a_j(x),a_j(x'))\) is jointly Gaussian with covariance:
\[
\mathrm{Cov}(a_j(x),\,a_j(x')) = \mathbb{E}[a_j(x)\,a_j(x')] = \frac{1}{d}\sum_{k=1}^d x_k x'_k = \frac{\langle x,x'\rangle}{d}.
\]

### 17.4.2 Multi-layer wide limit: random functions converge to a GP (NNGP)

In a deep fully-connected network of widths \(m_1,m_2,\dots\to\infty\) (taken sequentially from the first hidden layer upward), with i.i.d. initialization and \(1/\sqrt{m}\) weight scaling, the output \(f(x)\) for any finite set of inputs \(x^{(1)},\dots,x^{(n)}\) converges in distribution to a multivariate Gaussian. Therefore the random function \(f:\mathbb{R}^d\to\mathbb{R}\) converges to a **Gaussian process** with a kernel determined recursively by the nonlinearity \(\phi\):
\[
K^{(\ell)}(x,x') = \mathbb{E}_{z\sim\mathcal{N}(0,K^{(\ell-1)})}[\phi(z(x))\,\phi(z(x'))].
\]

<div class="ml-box">

**ML consequences of the NNGP / NTK limit.**

1. At infinite width, training dynamics with gradient descent linearize around initialization. The function learned lies in the reproducing kernel Hilbert space of the Neural Tangent Kernel (NTK), and the generalization theory reduces to kernel methods.

2. Bayesian uncertainty and predictive distributions are tractable because GP posteriors are Gaussian with closed-form mean and covariance (in principle).

3. The "prior over functions" induced by random initialization is a GP prior: what the network "believes before seeing data" is a Gaussian field. Training updates the GP posterior.

This is CLT as *function-space geometry*: repeated "sum of independent contributions" arguments, applied first to preactivations, then recursively through layers.

</div>

### 17.4.3 Finite-width corrections: beyond the Gaussian limit

At finite width \(m\), the deviation from Gaussianity is \(O(1/m)\). These corrections are studied via the \(1/m\) expansion and are responsible for feature learning (the ability to move representations during training), which is absent at infinite width. The tension between "Gaussian CLT limit" and "finite-width feature learning" is the current frontier of theoretical deep learning.

---

## 17.5 LLN and CLT as the Hidden Axioms Behind Modern Scaling Laws

> **Theorem-sentence 17.5.** Empirical scaling laws for model performance as a function of data and compute reflect the \(\sqrt{n}\) CLT variance scale at their simplest level; deviations from this rate signal the emergence of structural learning beyond variance reduction.

### 17.5.1 Dataset scaling: why variance shrinks like \(n^{-1/2}\)

Any evaluation metric that is an average over i.i.d. samples inherits a CLT-type fluctuation. This is why:
- test accuracy stabilizes slowly (requires quadrupling data to halve the standard error),
- empirical improvements below \(O(1/\sqrt{n})\) may be within statistical noise,
- confidence intervals in NLP benchmarks scale as \(1/\sqrt{n}\) when the test set is the bottleneck.

### 17.5.2 Token-level aggregation in LLM pretraining

Cross-entropy loss in language modeling is an average over tokens:
\[
\widehat{R}_n(\theta) = \frac{1}{n}\sum_{i=1}^n \big(-\log p_\theta(\text{token}_i\mid\text{context}_i)\big).
\]

LLN (ergodic version, from Chapter 15, since tokens are not strictly i.i.d.) says this converges to the true expected negative log-likelihood under the data measure. CLT (or its dependent-data variant via martingale methods, Chapter 16) says the remaining fluctuations are asymptotically Gaussian at the \(\sqrt{n}\) scale.

<div class="ml-box">

The "stability" of training loss curves is not engineering magic; it is aggregation physics. Each gradient step averages over a minibatch and the loss curve averages over steps. Two layers of averaging, two applications of LLN-type reasoning. The residual "noise" visible in loss curves is the CLT fluctuation at the minibatch scale, and its magnitude is \(\sigma_\theta/\sqrt{m}\).

</div>

### 17.5.3 Beyond i.i.d.: dependent CLTs and mixing conditions

Modern training data violates strict i.i.d.: documents have internal structure, tokens within a context window are correlated, and minibatch sampling may be without replacement. The CLT generalizes to these settings under **mixing conditions** (e.g., \(\phi\)-mixing, \(\alpha\)-mixing), which formalize how fast correlations decay. Under exponential mixing (which holds for ergodic Markov chains with spectral gap, Chapter 15), a CLT still holds with the same \(\sqrt{n}\) rate, replacing \(\sigma^2\) by the **long-run variance**:
\[
\sigma_{\mathrm{LR}}^2 = \sum_{k=-\infty}^{\infty}\mathrm{Cov}(X_0,X_k).
\]
This is the formal reason why standard error computations remain valid for correlated data, provided correlations decay fast enough.

---

## 17.6 Proof Sketch of the CLT via Characteristic Functions

> **Theorem-sentence 17.6.** The characteristic function proof of the CLT reduces the problem to pointwise convergence of analytic functions, and reveals why the Gaussian is the unique stable limit: it is characterized by the condition that its log-characteristic function is exactly quadratic.

<div class="proof-box">

**Proof sketch (Lindeberg鈥揕茅vy).** Let \(Y_i = (X_i-\mu)/\sigma\) so that \(\mathbb{E}[Y_i]=0\), \(\mathbb{E}[Y_i^2]=1\). Define \(T_n = \frac{1}{\sqrt{n}}\sum_{i=1}^n Y_i\). The characteristic function of \(T_n\) is
\[
\varphi_{T_n}(t) = \mathbb{E}[e^{itT_n}] = \left(\varphi_{Y_1}\!\left(\frac{t}{\sqrt{n}}\right)\right)^n.
\]
Taylor-expand \(\varphi_{Y_1}\) around \(0\): since \(\mathbb{E}[Y_1]=0\) and \(\mathbb{E}[Y_1^2]=1\),
\[
\varphi_{Y_1}(s) = 1 - \frac{s^2}{2} + o(s^2) \quad\text{as } s\to 0.
\]
Substituting \(s=t/\sqrt{n}\):
\[
\varphi_{T_n}(t) = \left(1 - \frac{t^2}{2n} + o\!\left(\frac{1}{n}\right)\right)^n \;\to\; e^{-t^2/2} \quad\text{as } n\to\infty.
\]
Since \(e^{-t^2/2}\) is the characteristic function of \(\mathcal{N}(0,1)\), by L茅vy's continuity theorem, \(T_n\xRightarrow{}\mathcal{N}(0,1)\). \(\square\)

</div>

**Why Gaussian and not something else?** The Gaussian is characterized by having a *quadratic* log-characteristic function. Any distribution with finite variance, when summed and renormalized, converges to the unique distribution with this quadratic log-cf. This is why the Gaussian is the "attractor" of aggregation: it is the fixed point of the renormalization group for finite-variance distributions.

---

## 17.7 Scholium: What the Formulas Really Say (and What ML Really Does)

<div class="scholium-box">

1. **LLN is measure collapse.**
\[
\widehat{\mathbb{P}}_n = \frac{1}{n}\sum_{i=1}^n \delta_{Z_i} \quad\Longrightarrow\quad \int f\,d\widehat{\mathbb{P}}_n \to \int f\,d\mathbb{P}.
\]
ERM is "optimize on \(\widehat{\mathbb{P}}_n\)" hoping it reflects \(\mathbb{P}\). LLN is the first legality certificate.

2. **CLT is the only non-trivial local microscope.**
At scale \(n\): collapse (LLN). At scale \(1\): explosion (no limit). At scale \(\sqrt{n}\): universal Gaussian geometry. There is no other scale.

3. **SGD noise is CLT noise.**
\[
\widehat{g}_m(\theta) \approx \nabla L(\theta) + \frac{1}{\sqrt{m}}\,\mathcal{N}(0,\Sigma(\theta)).
\]
This is the analytic origin of "temperature," "diffusion," and "flat minima" narratives 鈥?derived, not asserted.

4. **Wide networks are CLT objects.** As width grows, random feature sums become Gaussian; deep compositions inherit GP/NTK structure in suitable limits. The NNGP is CLT applied recursively to function space.

5. **Scaling laws reflect \(n^{-1/2}\) as a baseline.** Deviations from this rate 鈥?power laws with exponent \(\ne -1/2\) 鈥?indicate genuine structure beyond variance reduction. Identifying that structure is the open problem in empirical scaling laws.

</div>

LLN and CLT are asymptotic statements. Modern ML needs **finite-sample** control. The next mathematically forced step is concentration inequalities and uniform laws (Hoeffding, Bernstein, McDiarmid, Rademacher complexity), and their dependent-data counterparts (Azuma鈥揌oeffding, Freedman for martingales), because actual training data are not strictly i.i.d. and optimization is adaptive.

**Chapter 018: Concentration Inequalities 鈥?Large Deviation Bounds and the Finite-Sample Mathematics of Generalization.**


*Next: [Chapter 18: Concentration Inequalities](/book/chapters/chapter018/)*
