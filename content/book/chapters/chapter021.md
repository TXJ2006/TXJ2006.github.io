---
title: "Chapter 21: Shannon Entropy and Mutual Information — A Geometric View"
layout: "single"
url: "/book/chapters/chapter021/"
summary: "Probability distributions as points on a simplex; entropy as a concave potential and log-volume of the typical set; mutual information as KL divergence to the independence manifold; cross-entropy decomposition and the geometry of maximum likelihood; information bottleneck and contrastive learning as dependence-control problems."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 21
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

## Chapter 21 &mdash; Shannon Entropy and Mutual Information: A Geometric View

*Xujiang Tang*

</div>

## Abstract

Let $X$ be a discrete random variable taking values in a finite set $\mathcal{X}$ with probability mass function $p(x) = \mathbb{P}(X=x)$. The two objects of interest in this chapter are Shannon entropy $H(X)$, a scalar measuring the uncertainty or effective spread of a distribution, and mutual information $I(X;Y)$, a scalar measuring statistical dependence between two variables. The unifying perspective is geometric: probability distributions live on a simplex; entropy is a concave potential on that simplex; mutual information is a KL divergence from a joint distribution to the manifold of independent distributions. This geometric picture is not an aesthetic overlay&mdash;it directly explains why cross-entropy training minimizes KL divergence, why the information bottleneck is a constrained geometry problem on coupled simplices, and why contrastive learning can be interpreted as pushing the empirical joint away from the independence manifold.

---

## 1. Probability Simplex as the State Space

Fix $|\mathcal{X}| = m$. The set of all distributions on $\mathcal{X}$ is the **probability simplex**

$$\Delta^{m-1} = \Big\{p \in \mathbb{R}^m :\ p_i \ge 0,\ \sum_{i=1}^{m} p_i = 1\Big\}.$$

A distribution is a point in $\Delta^{m-1}$. Its vertices are the delta distributions $\delta_i$ (all mass on one outcome); its centroid is the uniform distribution $u(x) = 1/m$. Statistical functionals&mdash;entropy, KL divergence, mutual information&mdash;are functions on this convex set or on products thereof.

<div class="ml-box">

**ML context.** Learning a classifier, a language model, or a posterior approximator often means learning a map whose output is a point on a simplex (a categorical distribution over classes or tokens). The geometry of $\Delta^{m-1}$ therefore directly shapes the loss landscape of the learning problem.

</div>

---

## 2. Shannon Entropy: Definition and Immediate Consequences

### 2.1 Definition

<div class="def-box">

**Definition 21.1 (Shannon entropy).** The Shannon entropy of $X$ is
$$H(X) := -\sum_{x \in \mathcal{X}} p(x) \log p(x),$$
with the convention $0 \log 0 := 0$ (justified by $\lim_{t \to 0^+} t \log t = 0$). Logarithms are natural unless stated otherwise; base-2 logs rescale by $1/\log 2$.

</div>

### 2.2 Entropy as Expected Log-Loss

Define the **surprisal** $\ell(x) := -\log p(x)$. Then

$$H(X) = \mathbb{E}[\ell(X)].$$

Entropy is the expected log-loss under the true distribution. This is the basic bridge to machine learning: cross-entropy and negative log-likelihood are expected log-losses; entropy is simply the log-loss when the model is perfect (i.e., when $q = p$).

### 2.3 Bounds and Maximizer

<div class="prop-box">

**Proposition 21.2 (Entropy bounds).** For any $p \in \Delta^{m-1}$:

1. $H(X) \ge 0$, with equality if and only if $p$ is a delta distribution (a vertex of the simplex).
2. $H(X) \le \log m$, with equality if and only if $p$ is uniform (the centroid of the simplex).

</div>

<div class="proof-box">

*Proof of the upper bound via KL.* Let $u(x) = 1/m$ be the uniform distribution. Then

$$\mathrm{KL}(p \| u) = \sum_x p(x) \log \frac{p(x)}{u(x)} = \sum_x p(x) \log p(x) - \sum_x p(x) \log \frac{1}{m} = -H(X) + \log m.$$

Since $\mathrm{KL}(p \| u) \ge 0$, it follows that $H(X) \le \log m$, with equality iff $p = u$. $\square$

</div>

**Geometric interpretation.** Entropy increases as one moves from the vertices toward the centroid of the simplex. The uniform distribution is the maximizer because it is the most spread-out distribution compatible with the simplex constraints.

### 2.4 Concavity of Entropy

<div class="prop-box">

**Proposition 21.3 (Concavity).** For any $p, q \in \Delta^{m-1}$ and $\lambda \in [0,1]$,
$$H(\lambda p + (1-\lambda)q) \ge \lambda H(p) + (1-\lambda) H(q).$$

Equivalently, $-H$ is convex on $\Delta^{m-1}$.

</div>

This concavity matters in machine learning because the shape of optimization landscapes for objectives built from entropy terms&mdash;maximum entropy models, entropy regularization in RL, softmax temperature scaling&mdash;is governed by whether entropy appears as a concave reward or a convex penalty.

---

## 3. Conditional Entropy and the Chain Rule

Let $(X, Y)$ be a pair of discrete variables with joint pmf $p(x, y)$. Define the conditional distribution $p(x \mid y) = p(x,y)/p(y)$ for $p(y) > 0$.

### 3.1 Definition

<div class="def-box">

**Definition 21.4 (Conditional entropy).** The conditional entropy of $X$ given $Y$ is

$$H(X \mid Y) := \sum_{y} p(y)\, H(X \mid Y = y) = -\sum_{x,y} p(x,y) \log p(x \mid y).$$

</div>

### 3.2 Chain Rule (Derivation)

Start from the joint entropy $H(X,Y) = -\sum_{x,y} p(x,y) \log p(x,y)$. Using the factorization $\log p(x,y) = \log p(y) + \log p(x \mid y)$:

$$H(X,Y) = -\sum_{x,y} p(x,y) \log p(y) - \sum_{x,y} p(x,y) \log p(x \mid y).$$

The first term simplifies as $-\sum_y \big(\sum_x p(x,y)\big) \log p(y) = -\sum_y p(y) \log p(y) = H(Y)$. The second term is $H(X \mid Y)$ by definition. Therefore:

$$\boxed{H(X,Y) = H(Y) + H(X \mid Y) = H(X) + H(Y \mid X).}$$

**Geometric meaning.** Joint uncertainty decomposes into uncertainty in $Y$ plus residual uncertainty in $X$ after observing $Y$. Conditioning always reduces or preserves entropy: $H(X \mid Y) \le H(X)$, with equality iff $X$ and $Y$ are independent.

---

## 4. Mutual Information: Two Equivalent Definitions, One Geometric Meaning

### 4.1 Definition via Entropy Reduction

<div class="def-box">

**Definition 21.5 (Mutual information).** Mutual information is the reduction in uncertainty of $X$ after observing $Y$:

$$I(X;Y) := H(X) - H(X \mid Y).$$

By symmetry (from the chain rule), $I(X;Y) = H(Y) - H(Y \mid X) = I(Y;X)$.

</div>

### 4.2 Mutual Information as KL Divergence (Derivation)

Define the **product of marginals** $p(x)p(y)$, i.e., the joint distribution that would hold if $X$ and $Y$ were independent. Then:

$$\mathrm{KL}\big(p(x,y) \,\|\, p(x)p(y)\big) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}.$$

Expanding the right-hand side:

$$= \sum_{x,y} p(x,y) \log p(x,y) - \sum_{x,y} p(x,y) \log p(x) - \sum_{x,y} p(x,y) \log p(y).$$

Each term simplifies:
- $\sum_{x,y} p(x,y) \log p(x,y) = -H(X,Y)$,
- $\sum_{x,y} p(x,y) \log p(x) = \sum_x p(x) \log p(x) = -H(X)$,
- $\sum_{x,y} p(x,y) \log p(y) = -H(Y)$.

Therefore:

$$\mathrm{KL}\big(p(x,y) \,\|\, p(x)p(y)\big) = -H(X,Y) + H(X) + H(Y).$$

Using $H(X,Y) = H(Y) + H(X \mid Y)$:

$$= -H(Y) - H(X \mid Y) + H(X) + H(Y) = H(X) - H(X \mid Y) = I(X;Y).$$

<div class="prop-box">

**Proposition 21.6 (MI as KL).** $I(X;Y) = \mathrm{KL}\big(p(x,y) \,\|\, p(x)p(y)\big) \ge 0$, with equality if and only if $X$ and $Y$ are independent.

</div>

### 4.3 Geometric Interpretation: Distance to the Independence Manifold

Consider the simplex of all joint distributions on $\mathcal{X} \times \mathcal{Y}$. Inside it sits the set

$$\mathcal{M}_{\mathrm{ind}} := \{q(x,y) : q(x,y) = q_X(x)\, q_Y(y)\},$$

the **manifold of independent distributions**&mdash;a low-dimensional curved surface inside the full joint simplex.

Then $I(X;Y)$ is the KL divergence from the true joint $p$ to the particular independent distribution formed by its own marginals $p(x)p(y)$. It quantifies how far $p$ is from being independent, measured in the KL geometry.

<div class="scholium-box">

**Scholium (The geometric picture).** Mutual information is not merely an abstract formula. It is a divergence from a point (the joint distribution $p(x,y)$) to a structured submanifold ($\mathcal{M}_{\mathrm{ind}}$). Every statement about MI&mdash;nonnegativity, the data processing inequality, the information bottleneck&mdash;has a clean interpretation in this geometry.

</div>

---

## 5. ML Example I: Labels, Representations, and Feature Selection

Let $Y$ be a class label, $X$ an observed input, and $Z = f_\theta(X)$ a learned representation.

### 5.1 Feature Selection as Entropy Reduction

Maximizing $I(Z;Y) = H(Y) - H(Y \mid Z)$ is equivalent to minimizing $H(Y \mid Z)$, i.e., reducing label uncertainty given the representation.

The training objective is typically cross-entropy $\mathcal{L}(\theta) = \mathbb{E}[-\log q_\theta(Y \mid X)]$. When $q_\theta(\cdot \mid x)$ matches the true conditional $p(\cdot \mid x)$:

$$\mathbb{E}[-\log p(Y \mid X)] = H(Y \mid X).$$

So log-loss at optimality recovers a conditional entropy. Any gap between the achieved loss and $H(Y \mid X)$ is precisely $\mathrm{KL}(p(Y \mid X) \,\|\, q_\theta(Y \mid X))$, an average KL divergence over $X$.

### 5.2 Information Bottleneck

The **information bottleneck** principle seeks representations that compress $X$ while preserving label information:

$$\min_{Z}\ I(X;Z) - \beta\, I(Z;Y).$$

Geometrically:
- $I(X;Z)$ penalizes moving $p(x,z)$ away from $\mathcal{M}_{\mathrm{ind}}$ in the $(X,Z)$-simplex, encouraging compression (less dependence between input and representation).
- $I(Z;Y)$ rewards dependence between $Z$ and $Y$, pushing $p(z,y)$ away from $\mathcal{M}_{\mathrm{ind}}$ in the $(Z,Y)$-simplex.

The trade-off is a constrained geometry problem on coupled joint simplices: one term pulls toward independence (compression), the other pulls away (prediction).

<div class="ml-box">

**Practical role.** The IB objective motivates many modern regularization strategies: dropout and other noise injections can be interpreted as forcing $I(X;Z)$ to be small; cross-entropy training forces $I(Z;Y)$ to be large. The tension between these two pressures shapes the effective dimensionality of learned representations.

</div>

---

## 6. ML Example II: Contrastive Learning and the Joint vs. Product Separation

In self-supervised learning, one constructs "positive pairs" $(X, Y)$ (e.g., two augmentations of the same image) and trains an encoder to make positives similar and negatives dissimilar. The underlying objective is frequently interpreted as maximizing mutual information between paired views.

The identity $I(X;Y) = \mathrm{KL}(p(x,y) \| p(x)p(y))$ makes the geometric picture explicit: if a critic function can discriminate samples from the joint $p(x,y)$ versus samples from the product $p(x)p(y)$, then it is learning a statistic of the dependence&mdash;a lower bound on mutual information.

**Geometric reading.** Training the encoder is pushing the empirical joint away from the product manifold $\mathcal{M}_{\mathrm{ind}}$ in a direction detectable by the critic. The richer the critic class, the tighter the lower bound on $I(X;Y)$.

<div class="ml-box">

**InfoNCE structural view.** Given $N$ samples, InfoNCE uses one positive pair $(x, y^+)$ and $N-1$ negatives $(x, y^-)$ drawn from the marginal $p(y)$. The negative samples approximate the product distribution $p(x)p(y)$. The log-ratio learned by the critic estimates $\log p(x,y)/p(x)p(y)$, the pointwise mutual information. Averaging over the joint gives the mutual information itself. The bound tightens as $N \to \infty$ because the product approximation improves.

</div>

---

## 7. Entropy as Log-Volume: Typical-Set Geometry

For large $n$, i.i.d. samples $X^n = (X_1, \dots, X_n)$ concentrate on a **typical set** $\mathcal{T}_\epsilon^{(n)}$ whose cardinality is approximately $\exp(nH(X))$. This gives entropy a direct geometric meaning:

<div class="prop-box">

**Proposition 21.7 (Typical-set cardinality).** For any $\epsilon > 0$ and sufficiently large $n$:
$$|\mathcal{T}_\epsilon^{(n)}| \approx \exp(nH(X)).$$
All other sequences have probability converging to zero.

</div>

- $H(X)$ is the **exponential growth rate of the typical set size**, i.e., a log-volume in discrete sample space.
- Given $Y^n$, the conditional typical set for $X^n$ has size $\approx \exp(nH(X \mid Y))$.
- Therefore $I(X;Y) = H(X) - H(X \mid Y)$ is the **log-volume reduction** of the typical set for $X^n$ when $Y^n$ is known.

**ML interpretation.** Side information (labels, paired views, context tokens) collapses uncertainty, shrinking the set of plausible explanations by an exponential factor of $\exp(nI(X;Y))$.

---

## 8. KL Geometry: Entropy in Optimization

### 8.1 Cross-Entropy Decomposition

<div class="def-box">

**Definition 21.8 (Cross-entropy).** Let $q$ be a model distribution for $X$. The cross-entropy is
$$H(p, q) := -\sum_x p(x) \log q(x).$$

</div>

<div class="prop-box">

**Proposition 21.9 (Decomposition).** $H(p,q) = H(p) + \mathrm{KL}(p \| q).$

</div>

<div class="proof-box">

*Proof.*
$$\mathrm{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \sum_x p(x) \log p(x) - \sum_x p(x) \log q(x) = -H(p) + H(p,q). \quad \square$$

</div>

**Geometry of maximum likelihood.** Minimizing $H(p,q)$ over $q$ is equivalent to minimizing $\mathrm{KL}(p \| q)$, because $H(p)$ is fixed. Cross-entropy training is KL minimization toward the true data distribution.

### 8.2 Mutual Information as a KL Projection

The product of marginals $p(x)p(y)$ is the distribution obtained by "removing dependence while keeping marginals fixed." In information geometry, it is the **$m$-projection** of the joint $p(x,y)$ onto the independence family under the constraint that marginals are preserved.

Thus $I(X;Y)$ is the divergence incurred by enforcing independence while preserving first-order statistics. This viewpoint is used when designing regularizers that explicitly control dependence:
- **Fairness:** penalizing $I(Z; A)$ where $A$ is a protected attribute forces the representation $Z$ to be independent of $A$.
- **Disentanglement:** penalizing $I(Z_i; Z_j)$ for different latent dimensions encourages statistically independent factors.
- **Redundancy reduction (Barlow Twins):** the cross-correlation matrix objective approximates $I(Z_i; Z_j) \approx 0$ for all $i \ne j$.

---

## 9. Continuous Variables: What Changes, What Survives

For continuous $X$ with density $p(x)$, the **differential entropy** is

$$h(X) := -\int p(x) \log p(x)\, dx.$$

Unlike discrete entropy, $h(X)$ is not invariant under reparameterization and can be negative. The log-volume interpretation survives, but must be handled with care: $\exp(nh(X))$ is the approximate volume of the typical set in continuous space, which transforms under coordinate changes.

**Mutual information remains invariant:**

$$I(X;Y) = \int p(x,y) \log \frac{p(x,y)}{p(x)p(y)}\, dx\, dy \ge 0,$$

and the KL-based geometry (joint vs. product) is entirely unchanged. For machine learning, this is why mutual information is a more stable dependence measure than differential entropy in continuous representation learning: it does not depend on the choice of coordinates.

---

## 10. Summary

<div class="scholium-box">

**Chapter 21 in one paragraph.** Probability distributions are points on the simplex $\Delta^{m-1}$. Shannon entropy $H(X)$ is a concave potential on this simplex, equal to the log-volume of the typical set and maximized at the uniform distribution. Conditional entropy $H(X \mid Y)$ is the residual log-volume after side information. Mutual information $I(X;Y) = H(X) - H(X \mid Y) = \mathrm{KL}(p(x,y) \| p(x)p(y))$ measures how far the joint distribution is from the manifold of independent distributions $\mathcal{M}_{\mathrm{ind}}$, in KL geometry. In machine learning, cross-entropy training is KL minimization toward the data distribution; representation learning becomes dependence control, where $I(\cdot;\cdot)$ is the natural scalar capturing how much statistical structure is preserved or suppressed. The information bottleneck and contrastive learning are two facets of the same geometric problem: navigating the joint simplex relative to $\mathcal{M}_{\mathrm{ind}}$.

</div>

| Quantity | Formula | Geometric meaning |
|---|---|---|
| Entropy | $H(X) = -\sum_x p(x) \log p(x)$ | Log-volume of typical set; concave potential on simplex |
| Conditional entropy | $H(X\|Y) = -\sum_{x,y} p(x,y) \log p(x\|y)$ | Residual log-volume after observing $Y$ |
| Mutual information | $I(X;Y) = H(X) - H(X\|Y)$ | KL distance from joint to independence manifold |
| Cross-entropy | $H(p,q) = H(p) + \mathrm{KL}(p\|q)$ | Optimal log-loss plus model mismatch |

---

### Transition to Chapter 22 (KL Divergence and Relative Entropy)

This chapter treated KL divergence as a tool for proving entropy bounds and defining mutual information. Chapter 22 elevates KL divergence to a first-class object: its axiomatic characterization, its asymmetry and what that asymmetry means for learning, the Gibbs variational principle, its role in Bayesian inference (prior-to-posterior divergence), and its behavior in the exponential family. The geometry of Chapter 21 (simplex, independence manifold) will carry over directly; the new element is a systematic study of KL as a divergence&mdash;not a distance&mdash;and of the projections it defines on families of distributions.

*Next: [Chapter 22: KL Divergence and Relative Entropy](/book/chapters/chapter022/)*
