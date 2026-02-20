---
title: "Chapter 11: Measure Theory and Probability"
layout: "single"
url: "/book/chapters/chapter011/"
summary: "Measure theory as the foundation of probability in learning: σ-algebras, measures, Lebesgue integration, empirical measures, Radon–Nikodym, and divergences."
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

# Volume I — Mathematical Foundations

## Part B — Probability and Measure

## Chapter 11 — Measure Theory and Probability: What Can Be Observed, What Can Be Integrated, and What Data Can Legitimately Constrain

*Xujiang Tang*

</div>

## Chapter Abstract

Learning theory tacitly assumes that “data define constraints” and that “losses are expectations.” Both statements are meaningless until we specify (i) which subsets of the world are *observable* (a σ-algebra), (ii) how observation is *quantified* (a measure), and (iii) how quantities are *aggregated* over uncertainty (the Lebesgue integral). This chapter builds the probability space as the legal substrate of inference, then places modern machine learning inside it: empirical risk minimization as integration against the empirical measure, distribution shift as a change of measure, importance weighting as a Radon–Nikodym derivative identity, and cross-entropy training as the minimization of a divergence.

---

### 11.1 Observability: σ-Algebras as the Admissible Logic of Events

> **Core Theorem-Statement 11.1:** A learning objective is well-defined if and only if its loss is measurable with respect to an admissible σ-algebra.

#### 11.1.1 Definition and necessity (why “events” must be closed under limits)
Let \(\Omega\) denote the space of possible outcomes (inputs, labels, trajectories, or entire datasets, depending on the modeling level).

A collection \(\mathcal{F}\subseteq 2^\Omega\) is a **σ-algebra** if:  
1. \(\Omega\in\mathcal{F}\).  
2. If \(A\in\mathcal{F}\), then \(A^c\in\mathcal{F}\).  
3. If \(A_1,A_2,\dots \in \mathcal{F}\), then \(\bigcup_{n=1}^\infty A_n \in \mathcal{F}\).

From (2) and (3), \(\mathcal{F}\) is also closed under countable intersections.

**ML placement (natural, not decorative).**  
A classifier induces events of the form \(\{x: f_\theta(x)\ge 0\}\). Training or evaluation inevitably considers *countable* compositions of such events (threshold sweeps, limiting procedures, convergence of iterates, etc.). If \(\mathcal{F}\) is not closed under countable operations, one cannot even guarantee that the limit of an “increasingly refined” evaluation procedure is an event the theory can talk about.

---

### 11.2 Measures: Quantifying Events Without Smuggling in Geometry

> **Core Theorem-Statement 11.2:** A measure is the unique consistent extension of finite additivity to countable limits, and this countability is what makes “large sample” statements possible.

#### 11.2.1 Measure axioms
A **measure** is a map \(\mu:\mathcal{F}\to[0,\infty]\) such that:  
1. \(\mu(\varnothing)=0\).  
2. (Countable additivity) If \(A_i\in\mathcal{F}\) are pairwise disjoint, then
\[
\mu\!\left(\bigcup_{i=1}^\infty A_i\right)=\sum_{i=1}^\infty \mu(A_i).
\]
A **probability measure** additionally satisfies \(\mu(\Omega)=1\). A **probability space** is \((\Omega,\mathcal{F},\mathbb{P})\).

#### 11.2.2 Basic consequences (proved without skipping steps)
**Proposition 11.3 (Monotonicity).** If \(A\subseteq B\) with \(A,B\in\mathcal{F}\), then \(\mu(A)\le \mu(B)\).

*Proof.*  
Write \(B=A\cup(B\setminus A)\) and note the union is disjoint. By countable additivity (finite is a special case),
\[
\mu(B)=\mu(A)+\mu(B\setminus A)\ge \mu(A).
\]
\(\square\)

**Proposition 11.4 (Countable subadditivity).** For any \(A_1,A_2,\dots\in\mathcal{F}\),
\[
\mu\!\left(\bigcup_{n=1}^\infty A_n\right)\le \sum_{n=1}^\infty \mu(A_n).
\]

*Proof.*  
Define disjoint sets \(B_1=A_1\) and \(B_n=A_n\setminus\bigcup_{k=1}^{n-1}A_k\) for \(n\ge2\). Then \(B_n\subseteq A_n\) and \(\bigcup_n B_n=\bigcup_n A_n\), with the \(B_n\) disjoint. Hence
\[
\mu\!\left(\bigcup_{n=1}^\infty A_n\right)
=\sum_{n=1}^\infty \mu(B_n)
\le \sum_{n=1}^\infty \mu(A_n),
\]
using monotonicity for \(\mu(B_n)\le\mu(A_n)\). \(\square\)

---

### 11.3 Random Variables and Pushforward: Models as Measure-Transport Operators

> **Core Theorem-Statement 11.5:** A model defines a pushforward measure; generative learning is the geometry of matching pushforwards.

#### 11.3.1 Random variables as measurable maps
A **random variable** is a measurable function \(X:(\Omega,\mathcal{F})\to(\mathbb{R}^d,\mathcal{B}(\mathbb{R}^d))\); measurability means \(X^{-1}(B)\in\mathcal{F}\) for all Borel sets \(B\).

#### 11.3.2 Pushforward measures
Given a measurable map \(T:\Omega\to\mathcal{Z}\), define the **pushforward** \(T_\#\mathbb{P}\) by
\[
(T_\#\mathbb{P})(A)=\mathbb{P}(T^{-1}(A)).
\]
In particular, if \(Z=T(X)\) for \(X\sim\mathbb{P}\), then the distribution of \(Z\) is \(T_\#\mathbb{P}\).

**ML placement (generative modeling in one sentence).**  
A generator \(G_\theta:\mathcal{Z}\to\mathcal{X}\) turns a base law \(\nu\) (e.g., Gaussian) into a model distribution \(P_\theta = (G_\theta)_\#\nu\). The learning objective is to select \(\theta\) so that \(P_\theta\) approximates the data law \(P_{\text{data}}\) under a chosen discrepancy (f-divergence, Wasserstein distance, MMD, etc.).

---

### 11.4 The Lebesgue Integral: Expectation as the Only Coherent Aggregator

> **Core Theorem-Statement 11.6:** The Lebesgue integral is the unique extension of summation to measurable limits that preserves monotone convergence.

#### 11.4.1 Construction (from simple functions, no skipped steps)
A **simple function** is \(s=\sum_{k=1}^m a_k \mathbf{1}_{A_k}\) with \(A_k\in\mathcal{F}\), \(a_k\ge0\). Define
\[
\int s\,d\mu := \sum_{k=1}^m a_k\,\mu(A_k).
\]

For a nonnegative measurable \(f\), define
\[
\int f\,d\mu
:=
\sup\left\{\int s\,d\mu:\ 0\le s\le f,\ s\ \text{simple}\right\}.
\]
For general \(f\), write \(f=f^+-f^-\) with \(f^\pm\ge 0\), and define \(\int f\,d\mu\) when \(\int f^+d\mu<\infty\) and \(\int f^-d\mu<\infty\).  
A **probability expectation** is exactly \(\mathbb{E}[f(X)]=\int f\,d\mathbb{P}_X\).

#### 11.4.2 Monotone Convergence Theorem (full proof)
**Theorem 11.7 (MCT).** If \(0\le f_n\uparrow f\) pointwise, then
\[
\int f_n\,d\mu \uparrow \int f\,d\mu.
\]

*Proof.*  
**Step 1 (monotonicity of integrals).** Since \(f_n\le f_{n+1}\), by definition of the integral as a supremum over lower simple functions, \(\int f_n d\mu\le \int f_{n+1}d\mu\). Thus \(I:=\lim_n \int f_n d\mu\) exists in \([0,\infty]\).  

**Step 2 (upper bound).** Because \(f_n\le f\), we have \(\int f_n d\mu\le \int f d\mu\), so \(I\le \int f d\mu\).  

**Step 3 (lower bound via approximation).** Let \(s\) be any simple function with \(0\le s\le f\). Write \(s=\sum_{k=1}^m a_k\mathbf{1}_{A_k}\). For each \(k\), define
\[
A_{k,n} := A_k\cap\{f_n \ge a_k\}.
\]
Since \(f_n\uparrow f\) and \(a_k\le f\) on \(A_k\) (because \(s\le f\)), we have \(\mathbf{1}_{A_{k,n}}\uparrow \mathbf{1}_{A_k}\), hence \(\mu(A_{k,n})\uparrow \mu(A_k)\) by continuity from below (a consequence of countable additivity).  

Define the simple function
\[
s_n := \sum_{k=1}^m a_k\,\mathbf{1}_{A_{k,n}}.
\]
Then \(0\le s_n \le f_n\) (by construction \(f_n\ge a_k\) on \(A_{k,n}\)), and
\[
\int s_n\,d\mu = \sum_{k=1}^m a_k\,\mu(A_{k,n}) \uparrow \sum_{k=1}^m a_k\,\mu(A_k)=\int s\,d\mu.
\]
Because \(s_n\le f_n\), we have \(\int s_n d\mu \le \int f_n d\mu\). Taking limits,
\[
\int s\,d\mu \le \lim_{n\to\infty}\int f_n\,d\mu = I.
\]

**Step 4 (take supremum over all simple \(s\le f\)).** By definition,
\[
\int f\,d\mu = \sup_{0\le s\le f}\int s\,d\mu \le I.
\]
Combined with Step 2 (\(I\le \int f\,d\mu\)), we obtain \(I=\int f\,d\mu\). \(\square\)

---

### 11.5 The Empirical Measure: Data as a Measure, Not as a List

> **Core Theorem-Statement 11.9:** A dataset induces a probability measure; empirical risk is an integral with respect to that measure.

Given samples \(x_1,\dots,x_n\) in \(\mathcal{X}\), define the **empirical measure**
\[
\hat{\mathbb{P}}_n := \frac{1}{n}\sum_{i=1}^n \delta_{x_i}.
\]
Then for any measurable \(f\),
\[
\int f(x)\,d\hat{\mathbb{P}}_n(x) = \frac{1}{n}\sum_{i=1}^n f(x_i).
\]
Thus empirical risk minimization is literally:
\[
\hat{\theta}_n \in \arg\min_\theta \int \ell(\theta;x)\,d\hat{\mathbb{P}}_n(x).
\]
The population risk is:
\[
\theta^\star \in \arg\min_\theta \int \ell(\theta;x)\,d\mathbb{P}(x).
\]

---

### 11.6 Change of Measure and Importance Weighting: The Exact Algebra of Distribution Shift

> **Core Theorem-Statement 11.10:** Under absolute continuity, distribution shift is exactly multiplication by a Radon–Nikodym derivative.

#### 11.6.1 Absolute continuity and Radon–Nikodym derivative
Let \(\mathbb{P}\) and \(\mathbb{Q}\) be measures on \((\Omega,\mathcal{F})\). We say \(\mathbb{P}\ll \mathbb{Q}\) (absolute continuity) if \(\mathbb{Q}(A)=0 \Rightarrow \mathbb{P}(A)=0\).

**Theorem 11.11 (Radon–Nikodym, specialized).** If \(\mathbb{P}\ll \mathbb{Q}\) and both are σ-finite, then there exists a measurable \(w\ge 0\) such that
\[
\mathbb{P}(A)=\int_A w\,d\mathbb{Q}
\quad\forall A\in\mathcal{F}.
\]
We write \(w=\frac{d\mathbb{P}}{d\mathbb{Q}}\).

#### 11.6.2 Importance sampling identity (proved directly)
**Proposition 11.12.** If \(\mathbb{P}\ll\mathbb{Q}\) with \(w=\frac{d\mathbb{P}}{d\mathbb{Q}}\), then for any integrable \(f\),
\[
\mathbb{E}_{\mathbb{P}}[f] = \mathbb{E}_{\mathbb{Q}}[f\,w].
\]

*Proof.*  
By Radon–Nikodym, for any measurable set \(A\),
\[
\mathbb{P}(A)=\int_A w\,d\mathbb{Q}.
\]
First prove for simple \(f=\sum_k a_k \mathbf{1}_{A_k}\):
\[
\int f\,d\mathbb{P}
=
\sum_k a_k\,\mathbb{P}(A_k)
=
\sum_k a_k \int_{A_k} w\,d\mathbb{Q}
=
\int \left(\sum_k a_k\mathbf{1}_{A_k}\right)w\,d\mathbb{Q}
=
\int f w\,d\mathbb{Q}.
\]
Extend to nonnegative measurable \(f\) by monotone convergence using simple approximations, then to integrable \(f\) by decomposing into positive and negative parts. \(\square\)

---

### 11.7 Divergences as Geometry on Measures: Why Cross-Entropy Works

> **Core Theorem-Statement 11.13:** Maximum likelihood is divergence minimization; its correctness is a convex inequality, not an optimization superstition.

#### 11.7.1 KL divergence and nonnegativity (full proof)
Assume \(\mathbb{P}\ll\mathbb{Q}\) with density ratio \(r=\frac{d\mathbb{P}}{d\mathbb{Q}}\). Define
\[
\mathrm{KL}(\mathbb{P}\,\|\,\mathbb{Q})
=
\int \log r \, d\mathbb{P}
=
\int r\log r \, d\mathbb{Q}.
\]

**Theorem 11.14 (Gibbs inequality).** \(\mathrm{KL}(\mathbb{P}\,\|\,\mathbb{Q})\ge 0\), with equality iff \(\mathbb{P}=\mathbb{Q}\) (a.e.).

*Proof.*  
Consider the convex function \(\varphi(t)=t\log t\) on \(t>0\). By Jensen’s inequality applied to \(r\) under \(\mathbb{Q}\),
\[
\int r\log r \, d\mathbb{Q}
\ge
\left(\int r\,d\mathbb{Q}\right)\log\left(\int r\,d\mathbb{Q}\right).
\]
But \(\int r\,d\mathbb{Q}=\int d\mathbb{P}=1\). Hence
\[
\mathrm{KL}(\mathbb{P}\,\|\,\mathbb{Q})=\int r\log r\,d\mathbb{Q}\ge 1\cdot \log 1 = 0.
\]
Equality in Jensen holds iff \(r\) is constant \(\mathbb{Q}\)-a.e., thus \(r=1\) \(\mathbb{Q}\)-a.e., implying \(\mathbb{P}=\mathbb{Q}\). \(\square\)

---

### 11.8 Scholium: What This Chapter Removes from the Realm of Metaphor
1) “Data constrain truth” becomes: \(\hat{\mathbb{P}}_n\) approximates \(\mathbb{P}\) and losses are integrals against these measures.  
2) “Distribution shift” becomes: \(\mathbb{P}\) and \(\mathbb{Q}\) are different measures; transfer requires \(d\mathbb{P}/d\mathbb{Q}\).  
3) “Cross-entropy works” becomes: KL nonnegativity is a convex inequality.  
4) “Limits commute with expectations” becomes: MCT/DCT are the legal instruments that permit exchanging training limits with population expectations.

---

### Transition
We now possess a rigorous notion of observability (σ-algebra), uncertainty (probability measure), and aggregation (Lebesgue expectation). The next step is to introduce *conditional* structure and time: conditional expectation, filtrations, martingales, and the stochastic-process viewpoint underlying SGD, diffusion models, and modern generalization bounds.

**Next chapter (proposed): Chapter 012 — Conditional Expectation and Martingales: Information, Filtrations, and the Mathematics of Stochastic Optimization.**
