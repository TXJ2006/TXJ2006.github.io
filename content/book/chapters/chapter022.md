---
title: "Chapter 22: KL Divergence and Relative Entropy — Geometry, Definitions, and ML Grounding"
layout: "single"
url: "/book/chapters/chapter022/"
summary: "KL divergence defined as expected log-density ratio; Gibbs' inequality via Jensen; KL as the Bregman divergence induced by negative entropy; cross-entropy decomposition and MLE as KL minimization; forward vs. reverse KL and the mode-covering/mode-seeking asymmetry; ELBO as reverse-KL minimization; diffusion models as KL between path measures."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 22
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

## Chapter 22 &mdash; KL Divergence and Relative Entropy: Geometry, Definitions, and ML Grounding

*Xujiang Tang*

</div>

## Abstract

Let $\mathcal{X}$ be a finite set and let $p, q$ be probability mass functions on $\mathcal{X}$. The standing condition throughout is $q(x) = 0 \Rightarrow p(x) = 0$&mdash;absolute continuity $p \ll q$&mdash;which encodes not a technical convenience but a fundamental constraint: without it the KL divergence is $+\infty$ and the log-density ratio $\log p(x)/q(x)$ is undefined. This chapter develops KL divergence from its definition as an expected log-density ratio, through its proof of nonnegativity via Jensen&rsquo;s inequality, to its identification as the Bregman divergence induced by negative entropy. The cross-entropy decomposition connects KL to maximum likelihood; the asymmetry between forward and reverse KL explains the qualitative difference between mode-covering and mode-seeking inference. Concrete ML instantiations include cross-entropy training, the evidence lower bound (ELBO), and&mdash;prospectively&mdash;diffusion models as KL minimization between path measures.

---

## 1. Definition: KL Divergence as an Expected Log-Density Ratio

<div class="def-box">

**Definition 22.1 (KL divergence / relative entropy).** The Kullback&ndash;Leibler divergence from $q$ to $p$ is
$$\mathrm{KL}(p \| q) := \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)},$$
with the conventions $0 \log 0 := 0$ and $p(x) \log(p(x)/0) := +\infty$ whenever $p(x) > 0$. Logarithms are natural.

</div>

Define the **pointwise log-likelihood ratio** $\Lambda(x) := \log p(x)/q(x)$. Then

$$\mathrm{KL}(p \| q) = \mathbb{E}_{X \sim p}[\Lambda(X)].$$

KL is the expected tilt between two distributions in log-coordinates, weighted by the source $p$. Two structural facts follow immediately from the definition.

**Non-symmetry.** In general $\mathrm{KL}(p \| q) \ne \mathrm{KL}(q \| p)$. The two divergences weight the log ratio by different distributions, producing genuinely different quantities with different ML consequences (Section 6).

**Not a metric.** KL fails symmetry and the triangle inequality. It is a divergence in the sense of information geometry: a separation function satisfying nonnegativity and $\mathrm{KL}(p \| q) = 0 \Leftrightarrow p = q$, but not a distance.

---

## 2. Nonnegativity: Gibbs&rsquo; Inequality from Jensen

<div class="prop-box">

**Proposition 22.2 (Gibbs&rsquo; inequality).** For any $p, q$ with $p \ll q$,
$$\mathrm{KL}(p \| q) \ge 0,$$
with equality if and only if $p = q$.

</div>

<div class="proof-box">

*Proof.* Write
$$\mathrm{KL}(p \| q) = -\sum_{x} p(x) \log \frac{q(x)}{p(x)}.$$
Define $U(x) := q(x)/p(x)$ on $\mathrm{supp}(p)$. Then $U(x) > 0$ and
$$\sum_{x} p(x)\, U(x) = \sum_{x} q(x) = 1.$$
Apply Jensen&rsquo;s inequality to the strictly convex function $\varphi(u) = -\log u$:
$$-\sum_{x} p(x) \log U(x) = \mathbb{E}_{p}[\varphi(U)] \ge \varphi\!\left(\mathbb{E}_{p}[U]\right) = -\log 1 = 0.$$
Equality in Jensen holds iff $U(x)$ is $p$-almost surely constant. Constant $U$ with $\sum_x q(x) = 1$ forces $U \equiv 1$, i.e. $p = q$. $\square$

</div>

The proof is concise but its geometry is significant: KL nonnegativity is exactly the statement that $-\log$ is convex. The entire edifice of information theory rests on this one inequality.

---

## 3. KL as a Bregman Divergence: Geometry on the Simplex

### 3.1 Bregman Divergences

<div class="def-box">

**Definition 22.3 (Bregman divergence).** Let $F : \mathcal{C} \to \mathbb{R}$ be a strictly convex, differentiable function on a convex set $\mathcal{C}$. The Bregman divergence generated by $F$ is
$$D_F(p, q) := F(p) - F(q) - \langle \nabla F(q),\, p - q \rangle.$$

</div>

$D_F(p,q)$ measures the gap between $F(p)$ and the linear (first-order) approximation of $F$ at $q$ evaluated at $p$. By strict convexity of $F$, $D_F(p,q) \ge 0$ with equality iff $p = q$.

### 3.2 KL is the Bregman Divergence of Negative Entropy

Define the **negative entropy** functional on $\Delta^{m-1}$:
$$F(p) := \sum_{x} p(x) \log p(x).$$

$F$ is strictly convex on the simplex interior (Hessian is diagonal with entries $1/p(x) > 0$). Its gradient at $q$ has components
$$\frac{\partial F}{\partial q(x)} = 1 + \log q(x).$$

<div class="prop-box">

**Proposition 22.4.** $D_F(p, q) = \mathrm{KL}(p \| q)$.

</div>

<div class="proof-box">

*Proof.* Expand the Bregman formula:
$$D_F(p,q) = \sum_x p(x)\log p(x) - \sum_x q(x)\log q(x) - \sum_x (1+\log q(x))(p(x)-q(x)).$$
The terms involving $\sum_x (p(x) - q(x)) = 0$ cancel, leaving
$$D_F(p,q) = \sum_x p(x)\log p(x) - \sum_x p(x)\log q(x) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathrm{KL}(p\|q). \quad \square$$

</div>

**Geometric meaning.** On $\Delta^{m-1}$, negative entropy is the convex &ldquo;height function,&rdquo; and KL is the curvature-induced gap between the function and its tangent plane at $q$. This is a precise notion of separation that is neither a norm nor a metric, but is well-adapted to the geometry of probability distributions.

<div class="ml-box">

**Why Bregman structure matters in ML.** Mirror descent and exponentiated-gradient (EG) algorithms are steepest-descent methods under the geometry induced by $F$. When $F$ is negative entropy, the proximal step produces the multiplicative-weights update, the natural algorithm for probability simplex constraints. Adagrad, Adam, and natural-gradient methods all inherit their structure from a choice of Bregman potential.

</div>

---

## 4. Cross-Entropy Decomposition and Maximum Likelihood

<div class="def-box">

**Definition 22.5 (Cross-entropy).** For distributions $p, q$ on $\mathcal{X}$,
$$H(p, q) := -\sum_{x} p(x) \log q(x).$$

</div>

<div class="prop-box">

**Proposition 22.6 (Decomposition).** $H(p, q) = H(p) + \mathrm{KL}(p \| q)$, where $H(p) = -\sum_x p(x) \log p(x)$.

</div>

<div class="proof-box">

*Proof.*
$$\mathrm{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \sum_x p(x) \log p(x) - \sum_x p(x) \log q(x) = -H(p) + H(p,q). \quad \square$$

</div>

**The spine of likelihood training.** In supervised learning:
- The training objective is $\mathcal{L}(\theta) = \mathbb{E}_{(X,Y) \sim p_{\mathrm{data}}}[-\log q_\theta(Y \mid X)]$, the expected cross-entropy.
- By the decomposition: $\mathcal{L}(\theta) = \mathbb{E}_{p(x)}[H(p(\cdot \mid x))] + \mathbb{E}_{p(x)}[\mathrm{KL}(p(\cdot \mid x) \| q_\theta(\cdot \mid x))]$.
- The first term is fixed (the irreducible entropy of the labels). Minimizing $\mathcal{L}(\theta)$ over $\theta$ is exactly minimizing the average conditional KL divergence to the true distribution.

So &ldquo;train by minimizing log-loss&rdquo; is not a heuristic: it is KL minimization toward the data-generating distribution, within the model class $\{q_\theta\}$.

---

## 5. The Typical-Set View: Why KL Is Called &ldquo;Relative Entropy&rdquo;

For $n$ i.i.d. draws $X^n \sim p$, the typical set $\mathcal{T}_\epsilon^{(n)}$ has cardinality $\approx \exp(nH(p))$ and captures probability $\to 1$ under $p$. The probability assigned by $q$ to $\mathcal{T}_\epsilon^{(n)}$ decays as

$$q^{\otimes n}(\mathcal{T}_\epsilon^{(n)}) \approx \exp\!\left(-n \cdot H(p,q)\right) = \exp\!\left(-n(H(p) + \mathrm{KL}(p\|q))\right).$$

This follows from the law of large numbers applied to $-n^{-1}\log q(X^n) = n^{-1}\sum_i (-\log q(X_i)) \to \mathbb{E}_p[-\log q(X)] = H(p,q)$.

**Interpretation.** $\mathrm{KL}(p \| q)$ is the additional codelength per symbol you pay when encoding $p$-generated data with a code optimized for $q$. Entropy $H(p)$ is the irreducible log-volume of the typical set; relative entropy $\mathrm{KL}(p \| q)$ is the log-volume mismatch penalty.

---

## 6. Asymmetry: Forward KL vs. Reverse KL

Because the two orderings of $p$ and $q$ weight the log ratio differently, minimizing $\mathrm{KL}(p \| q_\theta)$ versus $\mathrm{KL}(q_\theta \| p)$ produces qualitatively different solutions.

### 6.1 Forward KL: Mode-Covering

$\mathrm{KL}(p \| q_\theta)$ is weighted by $p$. Wherever $p(x) > 0$ and $q_\theta(x) \approx 0$, the contribution is $p(x) \log(p(x)/q_\theta(x)) \to +\infty$. The minimizer therefore cannot afford to assign zero mass anywhere $p$ has mass. This forces **coverage**: $q_\theta$ must spread probability across all modes of $p$.

### 6.2 Reverse KL: Mode-Seeking

$\mathrm{KL}(q_\theta \| p)$ is weighted by $q_\theta$. Wherever $q_\theta(x) \approx 0$, the contribution is small regardless of $p(x)$. The minimizer can safely ignore regions where $p$ has mass, as long as $q_\theta$ is small there too. This produces **mode-seeking**: $q_\theta$ concentrates on one high-density region of $p$.

### 6.3 Discrete Illustration

Let $\mathcal{X} = \{1, 2\}$, $p(1) = p(2) = 1/2$, and $q(1) = 1$, $q(2) = 0$.

$$\mathrm{KL}(p \| q) = \frac{1}{2}\log\frac{1/2}{1} + \frac{1}{2}\log\frac{1/2}{0} = +\infty.$$

Forward KL is infinite: $q$ misses the support of $p$, which is forbidden.

$$\mathrm{KL}(q \| p) = 1 \cdot \log\frac{1}{1/2} = \log 2 < \infty.$$

Reverse KL is finite: $q$ concentrates on one mode and ignores the other, incurring a bounded cost.

<div class="ml-box">

**Design implication.** Which divergence to minimize is a genuine modeling decision:
- **MLE / cross-entropy training**: minimizes forward KL $\mathrm{KL}(p_{\mathrm{data}} \| q_\theta)$ &mdash; tends toward over-dispersed models that cover all modes.
- **Variational inference / ELBO**: minimizes reverse KL $\mathrm{KL}(q_\phi \| p_{\mathrm{posterior}})$ &mdash; tends toward concentrated approximations that lock onto one mode.
- **Normalizing flows / diffusion models**: can target either direction depending on the training formulation.

The asymmetry is not a nuisance; it is the reason these approaches produce qualitatively different approximations.

</div>

---

## 7. Continuous Case: Radon&ndash;Nikodym and Absolute Continuity

For continuous variables with densities $p(x)$, $q(x)$ with respect to Lebesgue measure,

$$\mathrm{KL}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)}\, dx,$$

with $\mathrm{KL}(p \| q) = +\infty$ if $p$ is not absolutely continuous with respect to $q$.

In measure-theoretic language: if $P \ll Q$ (absolute continuity), the Radon&ndash;Nikodym derivative $dP/dQ$ exists $Q$-almost everywhere, and

$$\mathrm{KL}(P \| Q) = \int \log\!\left(\frac{dP}{dQ}\right) dP = \mathbb{E}_{P}\!\left[\log \frac{dP}{dQ}\right].$$

This is the coordinate-free definition. It explains why support mismatch creates infinities: if $P$ is not absolutely continuous with respect to $Q$, the log-density ratio is not a $Q$-a.e. well-defined object. Chapter 13&rsquo;s Radon&ndash;Nikodym theorem is precisely the tool that makes this definition rigorous.

---

## 8. ML Instantiations: KL as the True Object

### 8.1 Classification Cross-Entropy

Let $p(y \mid x)$ be the true conditional label distribution and $q_\theta(y \mid x)$ the model. The expected cross-entropy loss decomposes as:

$$\mathbb{E}_{p(x)}\!\left[H\!\left(p(\cdot \mid x),\, q_\theta(\cdot \mid x)\right)\right] = \mathbb{E}_{p(x)}\!\left[H\!\left(p(\cdot \mid x)\right)\right] + \mathbb{E}_{p(x)}\!\left[\mathrm{KL}\!\left(p(\cdot \mid x) \,\|\, q_\theta(\cdot \mid x)\right)\right].$$

The first term is irreducible (Bayes risk). Minimizing log-loss over $\theta$ is minimizing average conditional KL. Log-loss is not a surrogate for some other objective: it is a divergence, and its minimum is the Bayes optimal predictor.

### 8.2 Variational Inference and the ELBO

Let $p(z \mid x)$ be the true posterior and $q_\phi(z \mid x)$ an approximating family. The marginal log-likelihood satisfies

$$\log p(x) = \underbrace{\mathbb{E}_{q_\phi}[\log p(x,z) - \log q_\phi(z \mid x)]}_{\mathrm{ELBO}(\phi)} + \mathrm{KL}\!\left(q_\phi(z \mid x) \,\|\, p(z \mid x)\right).$$

<div class="proof-box">

*Derivation.* Write $\log p(x) = \log p(x,z) - \log p(z \mid x)$. Take expectation under $q_\phi(z \mid x)$:
$$\log p(x) = \mathbb{E}_{q_\phi}[\log p(x,z)] - \mathbb{E}_{q_\phi}[\log p(z \mid x)].$$
Add and subtract $\mathbb{E}_{q_\phi}[\log q_\phi(z \mid x)]$:
$$\log p(x) = \underbrace{\mathbb{E}_{q_\phi}[\log p(x,z) - \log q_\phi(z \mid x)]}_{\mathrm{ELBO}} + \underbrace{\mathbb{E}_{q_\phi}\!\left[\log \frac{q_\phi(z \mid x)}{p(z \mid x)}\right]}_{\mathrm{KL}(q_\phi \| p(\cdot \mid x))}. \quad \square$$

</div>

Since $\log p(x)$ is fixed and $\mathrm{KL} \ge 0$, maximizing ELBO is equivalent to minimizing $\mathrm{KL}(q_\phi \| p(\cdot \mid x))$&mdash;the **reverse** KL. This is the source of mean-field VI&rsquo;s tendency to underestimate posterior variance and collapse to a single mode.

### 8.3 Diffusion Models: KL Between Path Measures (Prospective)

In continuous-time diffusion, the training objective can be written as a KL divergence between two path measures on the space of stochastic processes: the path measure of the data-driven reverse SDE versus the path measure of a reference forward SDE. By Girsanov&rsquo;s theorem, this KL reduces to a time-integral of squared drift differences in $L^2$ under an appropriate expectation:

$$\mathrm{KL}(P_{\mathrm{reverse}} \| P_{\mathrm{forward}}) = \frac{1}{2}\int_0^T \mathbb{E}\!\left[\|s_\theta(X_t, t) - \nabla_x \log p_t(X_t)\|^2\right] dt.$$

Minimizing this KL is exactly score-matching. Chapter 19 (Itô calculus) and Chapter 20 (Wiener processes) provide the tools; the KL perspective reveals why score-matching is the correct objective at the level of path-space geometry.

<div class="ml-box">

**Practical implication.** Diffusion model training is not merely a denoising heuristic: it is KL minimization in path space between a model-defined reverse process and the data-defined reverse process. Every design choice in diffusion architectures&mdash;noise schedules, predictor-corrector steps, classifier-free guidance&mdash;can be analyzed in terms of how it affects this path-space KL.

</div>

---

## 9. Summary: The Geometric Picture

<div class="scholium-box">

**Chapter 22 in one paragraph.** Distributions are points on the simplex $\Delta^{m-1}$. The negative entropy $F(p) = \sum_x p(x)\log p(x)$ is a strictly convex potential on this simplex. KL divergence is the Bregman divergence generated by $F$: the gap between $F(p)$ and the tangent plane to $F$ at $q$, evaluated at $p$. This Bregman structure explains nonnegativity (strict convexity), the asymmetry between $\mathrm{KL}(p\|q)$ and $\mathrm{KL}(q\|p)$ (different linearization points), and the natural emergence of KL in mirror-descent algorithms. In machine learning, forward KL minimization (cross-entropy / MLE) forces coverage of the data distribution; reverse KL minimization (ELBO / VI) produces concentrated approximations. The absolute-continuity condition $p \ll q$ is not optional: it reflects the Radon&ndash;Nikodym structure that makes the log-density ratio well-defined in continuous settings.

</div>

| Object | Formula | ML role |
|---|---|---|
| KL divergence | $\mathrm{KL}(p\|q) = \mathbb{E}_p[\log p/q]$ | Core divergence in training objectives |
| Nonnegativity | $\mathrm{KL}(p\|q) \ge 0$, $= 0$ iff $p=q$ | Jensen on $-\log$; guarantees well-posedness |
| Bregman structure | $D_F(p,q)$ with $F = \sum p\log p$ | Mirror descent, natural gradient |
| Cross-entropy | $H(p,q) = H(p) + \mathrm{KL}(p\|q)$ | MLE $=$ forward KL minimization |
| ELBO | $\log p(x) = \mathrm{ELBO} + \mathrm{KL}(q\|p_{\mathrm{post}})$ | VI $=$ reverse KL minimization |
| Forward KL | Weighted by $p$ | Mode-covering |
| Reverse KL | Weighted by $q$ | Mode-seeking |

---

### Transition to Chapter 23 (Maximum Entropy Principle)

Chapter 22 established KL divergence as the canonical divergence on the simplex. Chapter 23 asks: among all distributions satisfying a given set of moment constraints, which one is closest to a reference distribution $q$ in reverse KL? The answer is the **maximum entropy distribution** (or minimum KL distribution), which takes the exponential-family form. This is the maximum entropy principle, and it provides the theoretical foundation for exponential families, Gibbs distributions, energy-based models, and the connection between statistical mechanics and machine learning.

*Next: [Chapter 23: The Maximum Entropy Principle](/book/chapters/chapter023/)*
