---
title: "Chapter 4: Hilbert Spaces and Norms"
layout: "single"
url: "/book/chapters/chapter004/"
summary: "Norms as complexity gravity, inner products as resonance, Hilbert completeness as the existence law for learning endpoints, and Riesz representation as the bridge from functionals to weights."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 4
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

## Chapter 4 &mdash; Hilbert Spaces and Norms (Revised: Frontier ML&ndash;Enriched)

*Xujiang Tang*

</div>

## Abstract

Algebra alone cannot support learning: without a quantitative geometry, there is no notion of error magnitude, stability of iterative updates, capacity control, or robustness to perturbations. This chapter installs that geometry in a strictly axiomatic way. Norms formalize "complexity gravity" and the cost of representational amplitude; inner products formalize resonance, correlation, projection, and the operational meaning of similarity; Hilbert completeness guarantees that learning trajectories and minimizing sequences have legitimate endpoints within the hypothesis space; and the Riesz representation theorem explains&mdash;precisely&mdash;when and why abstract measurements (functionals, differentials, gradients) can be represented by concrete "weights" (vectors). We integrate these structures with modern machine learning: overparameterized interpolation and implicit bias, Lipschitz control and spectral norms, adversarial robustness via dual norms, attention and contrastive learning as inner-product energy models, reinforcement learning as contraction-plus-projection in function Hilbert spaces, and infinite-width limits (RKHS/NTK) as Hilbert-space dynamics.

---

## 4.1 Norms: The Gravity of Information and the Cost of Complexity

### 4.1.1 ML motivation (problem first): why "length" is unavoidable

Training is the controlled accumulation of updates. If updates can grow without penalty, optimization becomes underdetermined (infinitely many interpolants) and unstable (small local steps can accumulate into uncontrolled global drift). What is missing from a bare vector space is a quantitative notion of (i) the size of parameters and representations, (ii) the size of perturbations (noise, adversaries, distribution shift), and (iii) the size of operator effects (amplification through depth). A norm supplies exactly this missing structure, and thereby turns "learning" into a well-posed dynamical system.

---

### 4.1.2 Definition and axioms

Let $V$ be a vector space over $\mathbb{R}$ or $\mathbb{C}$.

<div class="def-box">

**Definition 4.1 (Norm).** A function $\|\cdot\|:V\to[0,\infty)$ is a **norm** if for all $x,y\in V$ and scalars $\alpha$,

1. $\|x\|\ge 0$, and $\|x\|=0\iff x=0$.
2. $\|\alpha x\|=|\alpha|\|x\|$.
3. $\|x+y\|\le \|x\|+\|y\|$.

</div>

The induced metric $d(x,y)=\|x-y\|$ turns "approximation," "stability," and "convergence" into mathematically checkable statements.

---

### 4.1.3 Canonical norms as inductive biases (parameters, features, operators)

**Vector norms in $\mathbb{R}^d$.** For $1\le p<\infty$,
$$
\|x\|_p=\Big(\sum_{i=1}^d|x_i|^p\Big)^{1/p},\qquad \|x\|_\infty=\max_i|x_i|.
$$

<div class="ml-box">

**Machine-learning interpretations:**
- $\ell_2$: smooth energy control (weight decay, isotropic shrinkage).
- $\ell_1$: sparsity bias (feature selection; compressed representations).
- $\ell_\infty$: worst-case coordinate control (clipping; max-norm robustness).

</div>

**Matrix norms (operator stability).** For $W\in\mathbb{R}^{m\times n}$,
$$
\|W\|_F=\Big(\sum_{i,j}W_{ij}^2\Big)^{1/2},\quad
\|W\|_2=\sigma_{\max}(W),\quad
\|W\|_*=\sum_i\sigma_i(W).
$$
Here $\|W\|_2$ directly controls worst-case amplification $\|Wx\|_2\le \|W\|_2\|x\|_2$, and therefore is the natural quantity behind Lipschitz control and stability diagnostics in deep networks. The nuclear norm $\|\cdot\|_*$ is a convex proxy for low rank, hence a structural bias toward compressible operators.

---

### 4.1.4 Dual norms as the algebra of worst-case coupling (adversarial, margins, shifts)

Many frontier questions in ML are "worst-case" questions: the strongest possible coupling between a perturbation and a model, the maximum sensitivity of a decision under bounded disturbance, the extremal change under distribution shift. These are not probabilistic statements; they are geometric statements, and the correct tool is the dual norm.

To avoid symbol collision with the nuclear norm $\|\cdot\|_*$, we denote the dual norm by $\|\cdot\|^\star$.

<div class="def-box">

**Definition 4.2 (Dual norm).** On $\mathbb{R}^d$ with pairing $\langle w,x\rangle=w^\top x$, the dual norm of $\|\cdot\|$ is
$$
\|w\|^\star := \sup_{\|x\|\le 1}\langle w,x\rangle.
$$

</div>

<div class="prop-box">

**Proposition 4.3 (Generalized Hölder inequality).** For all $w,x\in\mathbb{R}^d$,
$$
\langle w,x\rangle \le \|w\|^\star\,\|x\|.
$$

</div>

<div class="proof-box">

*Proof.* If $x=0$, trivial. Otherwise let $u=x/\|x\|$, so $\|u\|=1$. Then $\langle w,u\rangle\le \sup_{\|z\|\le 1}\langle w,z\rangle=\|w\|^\star$. Multiply by $\|x\|$ to get $\langle w,x\rangle\le \|w\|^\star\|x\|$. $\square$

</div>

<div class="ml-box">

**Frontier ML consequence (robustness and margins).**
For a linear score $f(x)=\langle w,x\rangle$,
$$
\sup_{\|\delta\|\le \varepsilon}|f(x+\delta)-f(x)|
= \sup_{\|\delta\|\le \varepsilon}|\langle w,\delta\rangle|
= \varepsilon\|w\|^\star.
$$
Thus robustness to $\|\cdot\|$-bounded perturbations is exactly controlled by $\|w\|^\star$. This extends beyond "adversarial images": it is the correct geometric backbone for robustness to embedding drift, prompt perturbations in representation space, and stability of retrieval scores under small embedding noise in large-scale systems.

</div>

---

### 4.1.5 A concrete numerical example (dual norm geometry made explicit)

Let $x=(1,2)\in\mathbb{R}^2$. Then
$$
\|x\|_1=|1|+|2|=3,\qquad \|x\|_\infty=\max\{1,2\}=2.
$$
Take $w=(3,-1)$. The dual of $\ell_1$ is $\ell_\infty$, meaning
$$
\|w\|_\infty = \sup_{\|z\|_1\le 1} w^\top z.
$$
Compute $\|w\|_\infty=\max\{|3|,|{-1}|\}=3$. To see the supremum is achieved, choose $z=(1,0)$, which satisfies $\|z\|_1=1$, and yields $w^\top z=3$. No $z$ with $\|z\|_1\le 1$ can exceed 3 because for any such $z$,
$$
w^\top z = 3z_1 - z_2 \le |3||z_1| + |{-1}||z_2|
\le \max\{|3|,|{-1}|\}(|z_1|+|z_2|)
\le 3\cdot 1=3.
$$
This is dual-norm geometry in its simplest, fully explicit form: the worst-case coupling chooses the coordinate with maximal magnitude.

---

### 4.1.6 Norm regularization and modern overparameterization: beyond "weight decay"

In overparameterized regimes, many models can interpolate training data. The decisive question becomes: which interpolant does the training dynamics select? Norms enter in two ways.

<div class="ml-box">

**1) Explicit bias (regularization).** Minimize
$$
\frac{1}{N}\sum_{i=1}^N \ell(\theta;x_i,y_i)+\frac{\lambda}{2}\|\theta\|_2^2,
$$
giving gradient
$$
\nabla J(\theta)=\frac{1}{N}\sum_i \nabla_\theta \ell(\theta;x_i,y_i)+\lambda \theta,
$$
and update
$$
\theta\leftarrow (1-\eta\lambda)\theta-\eta\Big(\frac{1}{N}\sum_i\nabla_\theta\ell(\theta;x_i,y_i)\Big),
$$
a contraction-plus-data-driven term. This makes "complexity has weight" operational.

**2) Implicit bias (dynamics).** Even when $\lambda=0$, gradient-based optimization in certain settings selects solutions with small norm (e.g., minimum Euclidean norm interpolants in least-squares). The correct conceptual point for frontier ML is: training is not only minimizing loss; it is selecting a solution via the geometry induced by the optimizer and parameterization. Norms&mdash;and the inner products that generate them&mdash;are the latent selection mechanism.

</div>

---

### 4.1.7 Operator norms and depth: stability of large models is a norm statement

For a composition $f=f_L\circ\cdots\circ f_1$, local stability depends on Jacobians:
$$
\|J_f(x)\|_2 \le \prod_{\ell=1}^L \|J_{f_\ell}\|_2.
$$
Thus controlling operator norms (or at least their distribution) is the quantitative backbone behind "trainability at scale," including exploding/vanishing gradients, sensitivity to perturbations, and calibration drift. This is where norms cease to be "mathematical decoration" and become the governing constraints of large-scale optimization.

---

## 4.2 Inner Product Spaces: Quantifying Resonance, Similarity, and Projection

### 4.2.1 ML motivation (problem first): similarity must be a rule, not a metaphor

Modern representation learning is built around similarity scores: retrieval, contrastive learning, attention, clustering, and calibration. But "similarity" is undefined until an inner product (or a metric) is fixed. Inner products are the minimal structure that makes angles, projections, and correlation coherent.

---

### 4.2.2 Definition and induced norm

<div class="def-box">

**Definition 4.4 (Inner product).** An inner product on $V$ is a map $\langle\cdot,\cdot\rangle:V\times V\to\mathbb{R}$ such that for all $x,y,z\in V$ and $\alpha\in\mathbb{R}$,

1. $\langle x,x\rangle\ge 0$, equality iff $x=0$.
2. $\langle x,y\rangle=\langle y,x\rangle$.
3. $\langle \alpha x+y, z\rangle=\alpha\langle x,z\rangle+\langle y,z\rangle$.

</div>

Define $\|x\|=\sqrt{\langle x,x\rangle}$. This is not assumed to satisfy triangle inequality; it will be derived.

---

### 4.2.3 Cauchy&ndash;Schwarz inequality (full proof)

<div class="prop-box">

**Theorem 4.5 (Cauchy&ndash;Schwarz).** $|\langle x,y\rangle|\le \|x\|\|y\|$.

</div>

<div class="proof-box">

*Proof.* If $y=0$, done. Assume $y\neq 0$. Consider $\phi(t)=\|x-ty\|^2\ge 0$ for all $t\in\mathbb{R}$. Expand:
$$
\phi(t)=\langle x-ty,x-ty\rangle
=\|x\|^2-2t\langle x,y\rangle+t^2\|y\|^2.
$$
This quadratic is nonnegative for all $t$, so its discriminant is nonpositive:
$$
(-2\langle x,y\rangle)^2-4\|y\|^2\|x\|^2\le 0
\;\Rightarrow\;
\langle x,y\rangle^2\le \|x\|^2\|y\|^2.
$$
Taking square roots yields the claim. $\square$

</div>

---

### 4.2.4 Triangle inequality for inner-product norms (derived)

$$
\|x+y\|^2=\|x\|^2+2\langle x,y\rangle+\|y\|^2\le (\|x\|+\|y\|)^2
\Rightarrow \|x+y\|\le \|x\|+\|y\|.
$$
Thus Euclidean-like behavior is a theorem, not an axiom.

---

### 4.2.5 Projection theorem: least squares is geometry

Let $H$ be an inner-product space, $S\subseteq H$ a nonempty closed linear subspace.

<div class="prop-box">

**Theorem 4.6 (Projection).** For each $x\in H$ there exists a unique $s^{\ast}\in S$ such that
$$
\|x-s^{\ast}\|=\inf_{s\in S}\|x-s\|,
\quad\text{and}\quad
x-s^{\ast}\perp S.
$$

</div>

<div class="ml-box">

**Machine-learning meaning:** linear regression is projection of labels onto the feature span; more generally, many training problems are "projection onto a model class" under an inner-product geometry.

</div>

---

### 4.2.6 Frontier ML interpretation: attention and contrastive learning as inner-product energy geometry

Two modern primitives are, at core, inner-product constructions.

<div class="ml-box">

**1) Contrastive learning (InfoNCE-like) is a log-bilinear energy model.**
Scores are $s(x,y)=\langle \phi(x),\psi(y)\rangle/\tau$. The temperature $\tau$ rescales the inner product, effectively changing the sharpness of the induced energy landscape. Norm control of embeddings prevents pathological regimes in which similarity is dominated by amplitude rather than direction (a common failure mode in large-scale retrieval systems).

**2) Attention is an adaptive operator built from inner products.**
Given query $q_a$ and keys $k_b$, the logits $s_{ab}=\langle q_a,k_b\rangle/\sqrt{d}$ are inner-product energies. The softmax normalizes these energies into a stochastic operator on token indices. Stability of attention is therefore governed by norm bounds on $q_a$ and $k_b$ and by operator norm properties of the induced attention matrix. This connects Chapter 2 (spectral control) to Chapter 4 (norm control) in a mathematically direct way: spectral behavior is controlled through norm-bounded inner-product energies.

</div>

---

## 4.3 Hilbert Spaces: Completeness as the Existence Law for Learning Endpoints

### 4.3.1 Definition

<div class="def-box">

A **Hilbert space** is a complete inner-product space $(H,\langle\cdot,\cdot\rangle)$: every Cauchy sequence converges in $H$.

</div>

---

### 4.3.2 Why completeness is a learning axiom (not a technicality)

Optimization generates sequences $(\theta_t)$. Even when losses decrease and increments shrink, one can fail to converge within a non-complete space: the "limit object" can exist only in a completion. Completeness is therefore the minimal condition under which "approaching a solution" implies "arriving at a legitimate element of the hypothesis space." In learning terms: completeness is the formal clause that prevents "convergent training but non-representable truth."

---

### 4.3.3 RKHS and infinite-width limits: Hilbert spaces as modern model objects

In a reproducing kernel Hilbert space $H$ on $\mathcal{X}$, evaluation $f\mapsto f(x)$ is continuous, so by Riesz (proved next) there exists $k_x\in H$ such that
$$
f(x)=\langle f,k_x\rangle_H.
$$
This single equation is the analytical core of kernel learning: evaluation is an inner product. It also provides the clean conceptual bridge to infinite-width phenomena: when training dynamics admit a linearized description, the effective function space behaves like a Hilbert space with a problem-induced kernel. The practical frontier point is that large-scale training often lives closer to "function-space geometry" than to naive parameter-space intuition; completeness is what makes those function limits legitimate objects.

---

### 4.3.4 Reinforcement learning as contraction plus projection in Hilbert spaces (our analysis)

A value function $V$ is a function on states; it is naturally an element of a function space. The Bellman operator $T$ often has a contraction property under suitable norms (e.g., with discount $\gamma<1$, $T$ is a contraction in $\|\cdot\|_\infty$ in standard settings). In approximation regimes, one restricts to a subspace $S$ (e.g., linear function approximation). The algorithmic core becomes:
$$
V \approx \Pi_S(TV),
$$
where $\Pi_S$ is a projection (typically in an $L_2(\mu)$-type inner product). The stability and bias of RL algorithms are therefore governed by the interplay of:

- contraction (a norm statement), and
- projection (a Hilbert geometry statement).

This yields a precise conceptual diagnosis for frontier RL instability: divergence is frequently a mismatch between the norm in which the operator contracts and the norm in which the approximation projects. Hilbert structure is not optional; it is the mechanism by which function approximation becomes mathematically controlled.

---

## 4.4 Riesz Representation: When Functionals Become Weights

### 4.4.1 Statement

<div class="prop-box">

**Theorem 4.7 (Riesz representation).** Let $H$ be a Hilbert space and $\phi\in H^{\ast}$ a bounded linear functional. Then there exists a unique $w\in H$ such that
$$
\phi(x)=\langle x,w\rangle_H\quad(\forall x\in H),
$$
and $\|\phi\|_{H^{\ast}}=\|w\|_H$.

</div>

This is the theorem that justifies drawing "models" in the same space as "data": measurements are representable as inner products with unique vectors.

---

### 4.4.2 Proof with navigable step structure (no omitted steps)

<div class="proof-box">

**Step 1 &mdash; Kernel and orthogonal decomposition.**
Let $N=\ker\phi$. Continuity of $\phi$ implies $N$ is closed. Hence
$$
H=N\oplus N^\perp.
$$

**Step 2 &mdash; $N^\perp$ is one-dimensional.**
Choose $u_0\notin N$. Let $u$ be the orthogonal projection of $u_0$ onto $N^\perp$. Then $u\neq 0$ and $\phi(u)\neq 0$.
For any $v\in N^\perp$, define $\alpha=\phi(v)/\phi(u)$. Then $\phi(v-\alpha u)=0$, so $v-\alpha u\in N$. But also $v-\alpha u\in N^\perp$. Hence $v-\alpha u\in N\cap N^\perp=\{0\}$, implying $v=\alpha u$. Thus $N^\perp=\mathrm{span}\{u\}$.

**Step 3 &mdash; Construct the representer.**
Define
$$
w := \frac{\phi(u)}{\|u\|^2}\,u.
$$

**Step 4 &mdash; Verify representation.**
For any $x\in H$, write $x=n+\beta u$ with $n\in N$. Then $\phi(x)=\beta\phi(u)$. Also,
$$
\langle x,w\rangle
=\left\langle n+\beta u,\frac{\phi(u)}{\|u\|^2}u\right\rangle
=\frac{\phi(u)}{\|u\|^2}(\langle n,u\rangle+\beta\|u\|^2)
=\beta\phi(u)
=\phi(x).
$$

**Step 5 &mdash; Uniqueness.**
If $\langle x,w\rangle=\langle x,w'\rangle$ for all $x$, then $\langle x,w-w'\rangle=0$ for all $x$. Taking $x=w-w'$ yields $\|w-w'\|^2=0$, hence $w=w'$.

**Step 6 &mdash; Norm identity.**
For $\|x\|\le 1$, $|\phi(x)|=|\langle x,w\rangle|\le \|x\|\|w\|\le \|w\|$, so $\|\phi\|_{H^{\ast}}\le \|w\|$.
Conversely, take $x=w/\|w\|$ to achieve equality. Hence $\|\phi\|_{H^{\ast}}=\|w\|$. $\square$

</div>

---

### 4.4.3 Frontier ML consequences (our analysis): geometry of gradients, preconditioning, and function-space learning

<div class="ml-box">

**1) Why "the gradient is a vector" is not primitive.**
The differential $Df(\theta)$ is a functional. In a Hilbert space, Riesz provides the unique vector $\nabla f(\theta)$ such that
$$
Df(\theta)[h]=\langle h,\nabla f(\theta)\rangle.
$$
Therefore, every gradient-based method implicitly chooses an inner product (a geometry) in which gradients are represented.

**2) Natural gradient and adaptive optimizers as Riesz maps under non-Euclidean inner products.**
If we replace the inner product by $\langle a,b\rangle_G=a^\top G b$ with $G\succ 0$, then the Riesz representer of $Df$ changes accordingly and yields the metric-corrected update direction $G^{-1}\nabla f$. This is the conceptual spine behind natural gradient and the geometric interpretation of preconditioning: optimizers differ primarily in how they implement the dual-to-primal map.

**3) Kernel methods and infinite-width training.**
In an RKHS, evaluation and gradients live naturally as bounded functionals, hence admit Riesz representers. This is the cleanest mathematical setting in which "weights are functions" and "learning is geometry" become literal rather than metaphorical. It provides a principled lens for analyzing when large models behave like linearized function-space learners and when they depart from that regime.

</div>

---

## 4.5 Closing: From Deterministic Geometry to Stochastic Worlds (and beyond Euclid)

Norms and inner products convert algebra into analysis: they define magnitude, stability, projection, and the legitimacy of optimization endpoints. Hilbert completeness makes learning trajectories mathematically inhabitable, and Riesz representation explains how abstract measurement rules become concrete trainable objects.

The next step is probabilistic: real learning is driven by random sampling, noisy gradients, and distribution shift. Measure theory and probability introduce integration, expectations as linear operators on function spaces, and the formal machinery to quantify uncertainty on top of Hilbert geometry.

A further step&mdash;crucial for scalable intelligence&mdash;is geometric generalization beyond Euclidean settings: non-Euclidean norms, operator-induced metrics, and graph-structured or manifold-structured hypothesis spaces. The Hilbert layer built here is the quantitative foundation; subsequent chapters will lift it into geometries suited to large-scale, parallel, compositional computation where "space" is not merely $\mathbb{R}^d$ but a structured arena of interacting modules.
