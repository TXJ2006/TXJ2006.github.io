---
title: "Chapter 18: Concentration Inequalities 鈥?The Boundary of Learning Theory"
layout: "single"
url: "/book/chapters/chapter018/"
summary: "The Chernoff鈥搈gf pipeline; Hoeffding's lemma and inequality; Bernstein's inequality and variance-aware tail switching; McDiarmid via Doob martingales; finite-class uniform bounds and the union bound; the boundary where concentration ends and complexity begins."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 18
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

## Chapter 18 &mdash; Concentration Inequalities: The Boundary of Learning Theory

*Why finite-sample learning is possible, and where the real difficulty begins*

*Xujiang Tang*

</div>

## Abstract

Chapter 17 established that LLN makes empirical risk converge to population risk *asymptotically*. Learning theory needs a finite-\(n\) certificate 鈥?an explicit exponential tail bound on the deviation \(|R_n(\theta)-R(\theta)|\). Concentration inequalities are exactly the machinery that upgrades "eventually true" into "true at this \(n\), with this confidence." This chapter builds the concentration engine from first principles through a single pipeline: exponentiate the event, apply Markov, factor over independent variables, bound the one-dimensional MGF. Everything else 鈥?Hoeffding, Bernstein, McDiarmid 鈥?is a different way of executing step 4.

---

## 18.0 The Setup and the Tension

Let \(\mathcal{D}\) be a fixed but unknown distribution. Observe
\[
Z_1,\dots,Z_n \stackrel{i.i.d.}{\sim} \mathcal{D}.
\]
For a loss \(\ell(\theta;z)\) and parameter \(\theta\in\Theta\), define:
\[
R(\theta) := \mathbb{E}_{Z\sim\mathcal{D}}[\ell(\theta;Z)], \qquad R_n(\theta) := \frac{1}{n}\sum_{i=1}^n \ell(\theta;Z_i).
\]

Learning lives inside one tension: we can compute \(R_n(\theta)\) but we care about \(R(\theta)\). The LLN gives pointwise convergence as \(n\to\infty\). Concentration theory gives a finite-\(n\) certificate:
\[
\mathbb{P}\!\Big(|R_n(\theta)-R(\theta)|\le\varepsilon\Big)\ge 1-\delta,
\]
with \(\varepsilon\) explicit as a function of \((n,\delta)\) and concrete distributional parameters.

**The phrase "boundary of learning theory" has a sharp meaning:**
- Concentration answers: "how random is the empirical world?"
- Learning theory proper begins when you ask: "how many hypotheses am I simultaneously controlling?" That is where complexity (union bounds, VC, Rademacher) enters. This chapter builds the concentration engine and reaches exactly that boundary.

---

## 18.1 The Universal Mechanism: Exponentiation and Markov (Chernoff's Spine)

> **Theorem-sentence 18.1.** Every concentration inequality is an instance of one pipeline: exponentiate the deviation event to convert it into a moment-generating function problem, apply Markov's inequality, then bound the MGF using structural properties of the summands.

To upper-bound \(\mathbb{P}(S\ge a)\), the only non-negotiable idea is:

**Step 1 (exponentiate the event).** For any \(t>0\),
\[
\{S\ge a\} \iff \{e^{tS}\ge e^{ta}\}.
\]

**Step 2 (Markov on the nonneg random variable \(e^{tS}\)).** Since \(e^{tS}>0\),
\[
\mathbb{P}(e^{tS}\ge e^{ta}) \le \frac{\mathbb{E}[e^{tS}]}{e^{ta}}.
\]

Therefore, for all \(t>0\):
\[
\mathbb{P}(S\ge a) \le e^{-ta}\,\mathbb{E}[e^{tS}].
\]
This is the **Chernoff bound template**. Nothing else is "the method." Everything else is: (i) how to bound \(\mathbb{E}[e^{tS}]\), and (ii) how to optimize \(t\).

**The crucial simplification under independence.** When \(S = \sum_{i=1}^n (X_i - \mathbb{E}[X_i])\) with \(X_i\) independent,
\[
\mathbb{E}[e^{tS}] = \prod_{i=1}^n \mathbb{E}\!\big[e^{t(X_i-\mathbb{E}[X_i])}\big].
\]
The whole problem reduces to a one-dimensional analytic question: for what class of distributions can we bound \(\mathbb{E}[e^{t(X-\mu)}]\)?

---

## 18.2 Tail Type Is MGF Control

Rather than a zoo of named inequalities, there are only two MGF regimes.

<div class="def-box">

**Definition 18.1 (sub-Gaussian).** A centered random variable \(X\) (\(\mathbb{E}[X]=0\)) is *sub-Gaussian* with proxy variance \(\sigma^2\) if for all \(t\in\mathbb{R}\),
\[
\mathbb{E}[e^{tX}] \le \exp\!\left(\frac{\sigma^2 t^2}{2}\right).
\]
The exponential moment grows at most like that of a \(\mathcal{N}(0,\sigma^2)\) variable, so tails decay at least as fast as Gaussian tails.

</div>

<div class="def-box">

**Definition 18.2 (sub-exponential).** A centered variable \(X\) is *sub-exponential* with parameters \((v,b)\) if for all \(|t|<1/b\),
\[
\mathbb{E}[e^{tX}] \le \exp\!\left(\frac{v t^2}{2}\right).
\]
The quadratic MGF bound is only *local* (\(|t|<1/b\)), encoding heavier-than-Gaussian tails. Bernstein inequalities emerge exactly from this regime.

</div>

Sub-Gaussianity is stable under sums of independent sub-Gaussian variables: if \(X_i\) are independent sub-Gaussian with proxy variances \(\sigma_i^2\), then \(\sum_i X_i\) is sub-Gaussian with proxy variance \(\sum_i \sigma_i^2\).

---

## 18.3 Hoeffding's Lemma: Boundedness Implies Sub-Gaussianity

> **Theorem-sentence 18.2.** A bounded centered random variable is sub-Gaussian with proxy variance \((b-a)^2/4\); this is Hoeffding's lemma, and it is the only analytic fact needed to derive all Hoeffding-type inequalities.

**Setup.** Assume \(X\in[a,b]\) a.s. and \(\mathbb{E}[X]=0\).

**Claim.** For all \(t\in\mathbb{R}\),
\[
\mathbb{E}[e^{tX}] \le \exp\!\left(\frac{(b-a)^2 t^2}{8}\right).
\]

<div class="proof-box">

**Proof (complete, no steps skipped).**

**Step 1 (convexity bound).** The map \(x\mapsto e^{tx}\) is convex. For \(x\in[a,b]\), write \(x = (1-\lambda)a + \lambda b\) with \(\lambda = \frac{x-a}{b-a}\in[0,1]\). Convexity gives
\[
e^{tx} \le (1-\lambda)e^{ta} + \lambda e^{tb} = \frac{b-x}{b-a}e^{ta} + \frac{x-a}{b-a}e^{tb}.
\]
Take expectations. Since \(\mathbb{E}[X]=0\):
\[
\mathbb{E}[e^{tX}] \le \frac{b}{b-a}e^{ta} + \frac{-a}{b-a}e^{tb}.
\]

**Step 2 (reduce to one function).** Let \(p := \frac{-a}{b-a}\in(0,1)\), so \(1-p = \frac{b}{b-a}\). Define
\[
\phi(t) := \log\!\big((1-p)e^{ta} + p\,e^{tb}\big),
\]
so \(\mathbb{E}[e^{tX}] \le e^{\phi(t)}\). We show \(\phi(t)\le \frac{(b-a)^2 t^2}{8}\).

**Step 3 (boundary values).** \(\phi(0) = \log 1 = 0\).

**Step 4 (first derivative at 0).** \(\phi'(t) = \frac{(1-p)\,a\,e^{ta} + p\,b\,e^{tb}}{(1-p)e^{ta}+p\,e^{tb}}\), so \(\phi'(0) = (1-p)a + pb = \mathbb{E}[X] = 0\).

**Step 5 (bound the second derivative).** Computing \(\phi''(t)\):
\[
\phi''(t) = \frac{(1-p)\,p\,(b-a)^2\,e^{t(a+b)}}{\big((1-p)e^{ta}+p\,e^{tb}\big)^2}.
\]
Apply \((u+v)^2\ge 4uv\) to \(u=(1-p)e^{ta}\), \(v=p\,e^{tb}\):
\[
\big((1-p)e^{ta}+p\,e^{tb}\big)^2 \ge 4(1-p)\,p\,e^{t(a+b)}.
\]
Therefore \(\phi''(t) \le \frac{(b-a)^2}{4}\) for all \(t\).

**Step 6 (Taylor integral form).** Since \(\phi(0)=0\) and \(\phi'(0)=0\),
\[
\phi(t) = \int_0^t (t-s)\,\phi''(s)\,ds \le \frac{(b-a)^2}{4}\int_0^t(t-s)\,ds = \frac{(b-a)^2\,t^2}{8}.
\]
Exponentiating: \(\mathbb{E}[e^{tX}] \le e^{\phi(t)} \le \exp\!\left(\frac{(b-a)^2 t^2}{8}\right)\). \(\square\)

</div>

**Meaning.** Boundedness \(\Rightarrow\) sub-Gaussian MGF control with proxy variance \((b-a)^2/4\). The inequality is tight in the sense that the constant \(1/8\) cannot be improved for all \([a,b]\)-bounded distributions.

---

## 18.4 Hoeffding's Inequality: Exponential Concentration of Empirical Means

> **Theorem-sentence 18.3.** Hoeffding's inequality is Chernoff's template applied to a sum of independent bounded variables, using Hoeffding's lemma to bound each factor of the MGF, then optimizing over \(t\).

**Setup.** Let \(X_1,\dots,X_n\) be independent with \(X_i\in[a_i,b_i]\) a.s. and \(\mu_i=\mathbb{E}[X_i]\). Let \(S = \sum_{i=1}^n(X_i-\mu_i)\) and \(V := \sum_{i=1}^n(b_i-a_i)^2\).

<div class="prop-box">

**Theorem 18.1 (Hoeffding).** For all \(\varepsilon>0\),
\[
\mathbb{P}\!\left(\frac{1}{n}\sum_{i=1}^n(X_i-\mu_i)\ge\varepsilon\right) \le \exp\!\left(-\frac{2n^2\varepsilon^2}{V}\right).
\]
Two-sided: \(\mathbb{P}\!\left(\left|\frac{1}{n}\sum_{i=1}^n(X_i-\mu_i)\right|\ge\varepsilon\right) \le 2\exp\!\left(-\frac{2n^2\varepsilon^2}{V}\right)\).

</div>

<div class="proof-box">

**Proof.** For any \(t>0\), by Chernoff and independence:
\[
\mathbb{P}(S\ge n\varepsilon) \le e^{-tn\varepsilon}\prod_{i=1}^n \mathbb{E}[e^{t(X_i-\mu_i)}] \le \exp\!\left(-tn\varepsilon + \frac{t^2 V}{8}\right).
\]
Minimize the exponent over \(t>0\): set \(g(t) = -tn\varepsilon + \frac{t^2 V}{8}\), so \(g'(t)=0\) gives \(t^{\ast} = \frac{4n\varepsilon}{V}\). Then
\[
g(t^{\ast}) = -\frac{4n^2\varepsilon^2}{V} + \frac{V}{8}\cdot\frac{16n^2\varepsilon^2}{V^2} = -\frac{2n^2\varepsilon^2}{V}.
\]
Applying the same to \(-S\) and combining yields the two-sided bound. \(\square\)

</div>

### 18.4.1 ML: pointwise generalization for a fixed hypothesis

For bounded loss \(\ell(\theta;Z)\in[0,1]\) and fixed \(\theta\), set \(X_i=\ell(\theta;Z_i)\). Then \(V=n\cdot 1^2=n\) and:
\[
\mathbb{P}\!\big(|R_n(\theta)-R(\theta)|\ge\varepsilon\big) \le 2e^{-2n\varepsilon^2}.
\]
Solving for the sample size to achieve confidence \(1-\delta\): \(n\ge \frac{1}{2\varepsilon^2}\log\frac{2}{\delta}\).

<div class="ml-box">

This is a complete finite-sample statement for a *fixed* \(\theta\). But learning does not keep \(\theta\) fixed 鈥?it chooses \(\hat\theta\) based on data. That shifts the problem from pointwise deviation to *uniform* deviation, which is precisely where concentration meets complexity.

</div>

---

## 18.5 Bernstein's Inequality: Variance Enters and Tails Switch Regime

> **Theorem-sentence 18.4.** Bernstein's inequality injects variance information into the Chernoff template; it produces a bound that transitions from a Gaussian regime (small deviations dominated by \(\sigma^2\)) to an exponential regime (large deviations dominated by the amplitude \(M\)), honestly reflecting the two distinct sources of fluctuation.

**Setup.** Let \(Y_1,\dots,Y_n\) be i.i.d. centered (\(\mathbb{E}[Y_i]=0\)), with \(|Y_i|\le M\) a.s. and \(\mathrm{Var}(Y_i)=\sigma^2\).

<div class="prop-box">

**Theorem 18.2 (Bernstein).** For all \(\varepsilon>0\),
\[
\mathbb{P}(\bar{X}_n - \mu \ge \varepsilon) \le \exp\!\left(-\frac{n\varepsilon^2}{2(\sigma^2 + M\varepsilon/3)}\right).
\]

</div>

### 18.5.1 Key analytic lemma: controlling the exponential remainder

For any real \(u\) with \(|u|\le 1\), the Taylor remainder bound gives:
\[
e^u \le 1 + u + \frac{u^2/2}{1 - |u|/3}.
\]
Set \(u = tY\) with \(0<t<3/M\) so that \(|tY|\le tM < 3\). Take expectations:
\[
\mathbb{E}[e^{tY}] \le 1 + \frac{t^2\sigma^2}{2(1-tM/3)} \le \exp\!\left(\frac{t^2\sigma^2}{2(1-tM/3)}\right).
\]

### 18.5.2 Chernoff + independence + explicit choice of \(t\)

<div class="proof-box">

For \(t\in(0,3/M)\), by Chernoff and independence:
\[
\mathbb{P}(S\ge n\varepsilon) \le \exp\!\left(-tn\varepsilon + \frac{nt^2\sigma^2}{2(1-tM/3)}\right).
\]
Choose \(t^{\ast} = \frac{\varepsilon}{\sigma^2 + M\varepsilon/3}\). Admissibility check: \(t^{\ast}M = \frac{M\varepsilon}{\sigma^2+M\varepsilon/3} < \frac{M\varepsilon}{M\varepsilon/3} = 3\). Compute:
\[
1 - \frac{t^{\ast}M}{3} = \frac{\sigma^2}{\sigma^2+M\varepsilon/3}, \qquad \frac{(t^{\ast})^2\sigma^2}{2(1-t^{\ast}M/3)} = \frac{\varepsilon^2}{2(\sigma^2+M\varepsilon/3)}, \qquad t^{\ast}\varepsilon = \frac{\varepsilon^2}{\sigma^2+M\varepsilon/3}.
\]
The exponent becomes:
\[
-\frac{\varepsilon^2}{\sigma^2+M\varepsilon/3} + \frac{\varepsilon^2}{2(\sigma^2+M\varepsilon/3)} = -\frac{\varepsilon^2}{2(\sigma^2+M\varepsilon/3)}.
\]
Hence \(\mathbb{P}(\bar{X}_n-\mu\ge\varepsilon)\le\exp\!\left(-\frac{n\varepsilon^2}{2(\sigma^2+M\varepsilon/3)}\right)\). \(\square\)

</div>

### 18.5.3 The two regimes and their ML meaning

The Bernstein exponent has two asymptotic regimes:
- **Small \(\varepsilon\) (Gaussian regime):** \(\exp\!\left(-\frac{n\varepsilon^2}{2\sigma^2}\right)\) 鈥?variance dominates.
- **Large \(\varepsilon\) (exponential regime):** \(\exp\!\left(-\frac{3n\varepsilon}{2M}\right)\) 鈥?amplitude dominates.

<div class="ml-box">

**"Low noise learns faster" is not folklore; it is Bernstein.**
Hoeffding uses only boundedness: it produces \(\exp(-2n\varepsilon^2)\) regardless of the actual noise level. Bernstein replaces worst-case scale by the variance:
- When \(\sigma^2\) is small (clean labels, easy data), the deviation shrinks like \(\sqrt{\sigma^2\log(1/\delta)/n}\), often far below Hoeffding's \(\sqrt{\log(1/\delta)/n}\).
- The quadratic-to-linear transition is an honest reflection of moderate vs. large deviations, not a technicality.

This is the first time "difficulty of learning" becomes a quantitative property of the *distribution*, not just the hypothesis set.

</div>

---

## 18.6 From Sums to Algorithms: McDiarmid via Doob Martingales

> **Theorem-sentence 18.5.** McDiarmid's inequality extends concentration from simple empirical averages to arbitrary functions of independent samples, using the Doob martingale decomposition to reduce to a sum of bounded martingale differences.

Many ML objects are not simple averages. The learned parameter \(\hat\theta\) depends on the entire dataset. Let \(F(Z_1,\dots,Z_n)\) be a generic function.

### 18.6.1 Bounded differences condition

<div class="def-box">

**Definition 18.3 (bounded differences).** The function \(F\) satisfies the *bounded differences condition* with constants \(c_1,\dots,c_n\) if for each \(i\),
\[
\sup_{z_{1:n},\,z_i'}\big|F(z_1,\dots,z_i,\dots,z_n) - F(z_1,\dots,z_i',\dots,z_n)\big| \le c_i.
\]

</div>

This is an abstract stability statement: single-point perturbations cannot move the value too much.

### 18.6.2 Doob martingale decomposition

Let \(\mathcal{F}_i = \sigma(Z_1,\dots,Z_i)\) and define the Doob martingale:
\[
M_i := \mathbb{E}[F\mid\mathcal{F}_i], \quad i=0,1,\dots,n.
\]
Then \(M_0=\mathbb{E}[F]\), \(M_n=F\), and \((M_i)\) is a martingale by the tower property. The martingale difference increments are \(\Delta_i := M_i - M_{i-1}\). Under bounded differences, \(|\Delta_i|\le c_i/2\) a.s.: the conditional expectation over \(Z_i\) cannot move more than half the range \(c_i\).

Now we are back in a "sum of bounded increments" world:
\[
F - \mathbb{E}[F] = M_n - M_0 = \sum_{i=1}^n \Delta_i.
\]

### 18.6.3 Azuma鈥揌oeffding \(\Rightarrow\) McDiarmid

<div class="prop-box">

**Theorem 18.3 (Azuma鈥揌oeffding).** If \((M_i)\) is a martingale with \(|\Delta_i|\le d_i\) a.s., then
\[
\mathbb{P}(M_n - M_0 \ge \varepsilon) \le \exp\!\left(-\frac{\varepsilon^2}{2\sum_{i=1}^n d_i^2}\right).
\]

</div>

<div class="prop-box">

**Theorem 18.4 (McDiarmid).** Under bounded differences with constants \(c_i\),
\[
\mathbb{P}\!\big(F - \mathbb{E}[F] \ge \varepsilon\big) \le \exp\!\left(-\frac{2\varepsilon^2}{\sum_{i=1}^n c_i^2}\right).
\]

</div>

*Derivation:* Apply Azuma鈥揌oeffding with \(d_i = c_i/2\): \(\sum d_i^2 = \sum c_i^2/4\), giving the stated bound.

### 18.6.4 ML connection: stability implies generalization

Let \(\hat\theta = \hat\theta(Z_{1:n})\) be a learning algorithm's output. Define the generalization gap
\[
G(Z_{1:n}) := R(\hat\theta) - R_n(\hat\theta).
\]
If the algorithm is *uniformly stable* in the sense that changing one training example changes the loss on any test point by at most \(\beta/n\), then \(G\) satisfies bounded differences with \(c_i \asymp \beta/n\). Thus \(\sum c_i^2 \asymp \beta^2/n\), and McDiarmid gives:
\[
\mathbb{P}(G - \mathbb{E}[G] \ge \varepsilon) \le \exp\!\left(-\frac{2n\varepsilon^2}{\beta^2}\right).
\]

<div class="ml-box">

**The conceptual pivot.** Concentration can be about the *data distribution* (Hoeffding, Bernstein) or about the *algorithm's sensitivity* (McDiarmid/stability). This is why noise injection, early stopping, and regularization often improve generalization: they reduce effective sensitivity, hence tighten concentration. The generalization gap is small not because the hypothesis class is small, but because the algorithm is stable.

</div>

---

## 18.7 From Pointwise to Uniform: Finite Hypothesis Classes and the Union Bound

> **Theorem-sentence 18.6.** For a finite hypothesis class, a uniform deviation bound follows from a union bound over pointwise Hoeffding bounds; the cost of searching over \(|\mathcal{H}|\) hypotheses appears as \(\log|\mathcal{H}|\) in the required sample size.

Assume a finite hypothesis set \(\mathcal{H}\) and bounded loss \(\ell(h;Z)\in[0,1]\). For each \(h\in\mathcal{H}\), Hoeffding gives:
\[
\mathbb{P}\!\big(|R_n(h)-R(h)|\ge\varepsilon\big) \le 2e^{-2n\varepsilon^2}.
\]

By the union bound over all \(|\mathcal{H}|\) hypotheses:
\[
\mathbb{P}\!\Big(\exists\, h\in\mathcal{H}:\,|R_n(h)-R(h)|\ge\varepsilon\Big) \le 2|\mathcal{H}|\,e^{-2n\varepsilon^2}.
\]

Setting the right side equal to \(\delta\) and solving for \(\varepsilon\):

<div class="prop-box">

**Theorem 18.5 (uniform bound, finite class).** With probability at least \(1-\delta\), *simultaneously for all* \(h\in\mathcal{H}\),
\[
|R_n(h) - R(h)| \le \sqrt{\frac{1}{2n}\log\frac{2|\mathcal{H}|}{\delta}}.
\]
In particular, if \(\hat{h} = \arg\min_{h\in\mathcal{H}} R_n(h)\), then
\[
R(\hat{h}) \le \min_{h\in\mathcal{H}} R(h) + 2\sqrt{\frac{1}{2n}\log\frac{2|\mathcal{H}|}{\delta}}.
\]

</div>

**Reading the formula:**
- The term \(\sqrt{\log(1/\delta)/n}\) is the randomness cost (CLT-scale, Chapter 17).
- The term \(\sqrt{\log|\mathcal{H}|/n}\) is the model-search cost: each additional hypothesis "uses up" an \(\log\) factor of the sample budget.

<div class="ml-box">

**The boundary.** When \(|\mathcal{H}|\) is infinite, the union bound breaks entirely: \(\log|\mathcal{H}|=\infty\). To proceed, one must replace \(\log|\mathcal{H}|\) by a genuine measure of the effective complexity of the class 鈥?VC dimension (combinatorial), covering numbers (metric), or Rademacher complexity (probabilistic). That is where Chapter 19 begins.

</div>

---

## 18.8 The Unified Pipeline

> **Theorem-sentence 18.7.** The entire theory of concentration inequalities is a single pipeline; the named inequalities are different ways of bounding the MGF in step 4.

<div class="scholium-box">

**The pipeline:**

1. Learning compares \(R(\theta)\) with \(R_n(\theta)\) 鈥?an expectation vs. an empirical average.
2. Any such comparison is a tail probability about a random deviation \(S\).
3. Tail control begins with Chernoff: $\mathbb{P}(S\ge a)\le e^{-ta}\mathbb{E}[e^{tS}]$.
4. The entire difficulty is bounding \(\mathbb{E}[e^{tS}]\):
   - Bounded summands \(\Rightarrow\) sub-Gaussian MGF \(\Rightarrow\) **Hoeffding**.
   - Bounded + variance \(\Rightarrow\) local quadratic MGF \(\Rightarrow\) **Bernstein** (regime switching).
   - Arbitrary function of i.i.d. \(\Rightarrow\) bounded martingale increments via Doob \(\Rightarrow\) **McDiarmid**.
5. Once the statement must hold for the *data-chosen* hypothesis, concentration pushes into complexity control: finite class uses union bound; infinite class requires covering numbers or Rademacher complexity.

Concentration inequalities are the last purely probabilistic step. After this, the work is no longer "probability of sums"; it is "geometry and combinatorics of function classes."

</div>

---

## 18.9 Scholium: What This Chapter Forces You to Admit

<div class="scholium-box">

1. **Finite-sample learning is possible** because bounded fluctuations produce exponentially thin tails. This is not obvious from first principles; it is a consequence of the Chernoff鈥揗GF pipeline.

2. **Variance is the correct noise measure.** Hoeffding is a worst-case bound; Bernstein reveals that low-variance problems are genuinely easier, not just heuristically so.

3. **Stability is a concentration property.** McDiarmid's inequality shows that algorithmic stability (bounded sensitivity to single samples) directly implies concentration of the generalization gap.

4. **The union bound is sharp for finite classes but fails for infinite ones.** The \(\log|\mathcal{H}|\) cost is not a loose bound; for finite classes under adversarial construction it can be tight. The need for a better complexity measure is not a technical annoyance 鈥?it is the true open problem.

5. **Deep learning generalization is not yet explained by this chapter.** Neural networks have billions of parameters (\(|\mathcal{H}|\) is astronomically large), yet generalize well. The correct explanation requires function-space complexity measures (Rademacher, spectral norms, margin bounds), not raw parameter counts. Chapter 19 begins that story.

</div>

**Chapter 019: Generalization Theory 鈥?Rademacher Complexity, Covering Numbers, and Why Deep Networks Generalize.**


*Next: [Chapter 19: SDE Foundations](/book/chapters/chapter019/)*
