---
title: "Chapter 9: Random Matrix Theory"
layout: "single"
url: "/book/chapters/chapter009/"
summary: "The Marchenko–Pastur law, Xavier/He initializations as Jacobian-stability conditions, dynamical isometry, and mean-field signal propagation �?why high-dimensional randomness is spectral determinism."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 9
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

## Chapter 9 &mdash; Random Matrix Theory: The Deterministic Geometry of High-Dimensional Initialization

*Xujiang Tang*

</div>

## Abstract

Random initialization is often described in informal terms ("draw weights from a Gaussian"), yet in high dimension this language is misleading: while individual entries are random, the *operator-level* behavior of weight matrices and Jacobians is sharply constrained by concentration phenomena. This chapter develops a principled account of initialization through random matrix theory (RMT). We (i)&nbsp;formalize why "randomness" in high dimension becomes rigid at the spectral level; (ii)&nbsp;derive the Marchenko&ndash;Pastur law as the canonical spectral description of Gaussian layers; (iii)&nbsp;explain Xavier/He initializations as *operator- and Jacobian-stability conditions* rather than mere variance heuristics; (iv)&nbsp;analyze products of random matrices to characterize gradient explosion/vanishing and the notion of dynamical isometry; and (v)&nbsp;connect these results to contemporary architectures (residual networks and Transformers) where normalization and residual scaling modify the Jacobian spectrum in essential ways.

---

## 9.1 High-Dimensional Randomness Is Spectral Determinism

### 9.1.1 The object that matters: operators, not entries

Let $W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}$ be a random weight matrix. Entrywise randomness does not describe the effect of $W$ on signals. The relevant quantities are operator-level:

- Forward amplification: $\|Wx\|$ relative to $\|x\|$.
- Backward amplification: $\|W^\top g\|$ relative to $\|g\|$.
- Worst-case sensitivity: $\|W\|\_{\mathrm{op}} = \sup\_{\|x\|=1}\|Wx\|$, the spectral norm.
- Typical amplification: the empirical singular value distribution of $W$.

In deep networks the central object is not a single $W$, but the *input–output Jacobian*

$$
J(x) \;=\; \frac{\partial f(x)}{\partial x},
$$

whose singular values govern both information propagation and gradient flow:

$$
\|J(x)\|\_{\mathrm{op}} \;=\; \sup\_{\|v\|=1}\|J(x)v\|\quad\text{and}\quad \|J(x)^\top u\| \le \|J(x)\|\_{\mathrm{op}}\|u\|.
$$

### 9.1.2 A precise sense in which "random becomes rigid"

For many ensembles (Gaussian, sub-Gaussian, orthogonally-invariant), spectral statistics concentrate. Informally: for large $d\_{\text{in}},d\_{\text{out}}$, random fluctuations of global quantities (e.g., $\|W\|\_{\mathrm{op}}$, the empirical singular value measure) are small compared with their means. Thus "random initialization" should be understood as choosing a *spectral law* (up to vanishing fluctuations) rather than choosing arbitrary noise.

A practical consequence for ML: two independently drawn initializations with the same scaling are, at the operator level, nearly interchangeable; pathologies arise not from rare entrywise events but from deterministic spectral mismatch (e.g., Jacobian singular values drifting far from 1 as depth grows).

---

## 9.2 Gaussian Layers and the Marchenko&ndash;Pastur Law

### 9.2.1 Setup: Wishart matrices as Gram operators

Let $G\in\mathbb{R}^{d\times n}$ have i.i.d. entries $G_{ij}\sim\mathcal{N}(0,1)$. Define the sample covariance (Gram operator)

$$
S \;=\; \frac{1}{n}GG^\top \in \mathbb{R}^{d\times d}.
$$

The eigenvalues of $S$ are the squared singular values of $\frac{1}{\sqrt{n}}G$.
In neural networks, if $W = \frac{\sigma}{\sqrt{d\_{\text{in}}}}G$ with $G\in\mathbb{R}^{d\_{\text{out}}\times d\_{\text{in}}}$, then

$$
WW^\top = \frac{\sigma^2}{d\_{\text{in}}}GG^\top,
$$

so eigenvalues of $WW^\top$ (and singular values of $W$) are governed by the same mechanism.

Let $\gamma = \lim_{n\to\infty} d/n \in (0,\infty)$. Define the empirical spectral distribution (ESD)

$$
\mu_S \;=\;\frac{1}{d}\sum_{i=1}^d \delta\_{\lambda_i(S)}.
$$

### 9.2.2 The Marchenko&ndash;Pastur theorem (statement)

As $d,n\to\infty$ with $d/n\to\gamma$, $\mu_S$ converges (weakly, in probability) to the Marchenko&ndash;Pastur distribution $\mu\_{\mathrm{MP}}^{(\gamma)}$ with density

$$
\rho\_{\mathrm{MP}}^{(\gamma)}(\lambda) \;=\; \frac{\sqrt{(\lambda\_+-\lambda)(\lambda-\lambda\_-)}}{2\pi\gamma\,\lambda}\,\mathbf{1}\_{[\lambda\_-,\lambda\_+]}(\lambda), \quad \lambda\_\pm=(1\pm\sqrt{\gamma})^2,
$$

plus an atom at 0 of mass $(1-1/\gamma)\_+$ when $\gamma>1$.
Equivalently, the singular values of $\frac{1}{\sqrt{n}}G$ concentrate in $[1-\sqrt{\gamma},\,1+\sqrt{\gamma}]$.

### 9.2.3 Derivation by the moment method (step-by-step core)

We outline a complete derivation of the limiting moments; the identification with $\rho\_{\mathrm{MP}}$ follows by standard uniqueness of measures with compact support.

Define the $k$-th moment of the ESD:

$$
m_k(d,n) \;=\; \int \lambda^k\,d\mu_S(\lambda) \;=\; \frac{1}{d}\operatorname{tr}(S^k).
$$

Take expectation:

$$
\mathbb{E}[m_k(d,n)] = \frac{1}{d}\mathbb{E}\left[\operatorname{tr}\left(\left(\frac{1}{n}GG^\top\right)^k\right)\right] = \frac{1}{d\,n^k}\mathbb{E}\left[\operatorname{tr}\left((GG^\top)^k\right)\right].
$$

**Step 1 (expand the trace).**
Write indices explicitly. For $S=\frac{1}{n}GG^\top$,

$$
\operatorname{tr}(S^k) = \sum_{i_1=1}^d \cdots \sum_{i_k=1}^d S_{i_1 i_2}S_{i_2 i_3}\cdots S_{i_k i_1}.
$$

Each term $S_{ab}=\frac{1}{n}\sum_{t=1}^n G_{a t}G_{b t}$. Substituting yields

$$
\operatorname{tr}(S^k) = \frac{1}{n^k} \sum_{i_1,\dots,i_k} \sum_{t_1,\dots,t_k} \prod_{\ell=1}^k \left(G_{i_\ell t_\ell}G_{i_{\ell+1} t_\ell}\right), \quad i_{k+1}=i_1.
$$

**Step 2 (take expectation using Gaussian Wick rule).**
Because entries are centered Gaussian and independent across $(i,t)$, the expectation of a product is zero unless each variable appears with even total multiplicity. Wick's formula reduces the expectation to a sum over pairings of factors. Each valid pairing imposes index equalities among $\{i_\ell\}$ and $\{t_\ell\}$.

**Step 3 (identify leading contributions as $d,n\to\infty$).**
Among all pairings, those producing the largest powers of $d$ and $n$ dominate after normalization by $d\,n^k$. One obtains that the leading pairings correspond to *noncrossing* structures, yielding Catalan-type counts (the same combinatorics as in free probability for Wishart operators).

Concretely, the limit of $\mathbb{E}[m_k(d,n)]$ exists and equals

$$
m_k(\gamma) = \sum_{r=1}^{k} \frac{1}{k}\binom{k}{r}\binom{k}{r-1}\gamma^{r-1}.
$$

(These are the Narayana polynomials; the coefficients count noncrossing partitions by number of blocks.)

**Step 4 (verify first moments explicitly).**

- For $k=1$:

$$
m_1(d,n)=\frac{1}{d}\operatorname{tr}\!\left(\frac{1}{n}GG^\top\right) =\frac{1}{dn}\sum_{i,t}G_{it}^2, \quad \mathbb{E}[m_1]=\frac{1}{dn}\cdot dn = 1.
$$

So $m_1(\gamma)=1$.

- For $k=2$:

$$
m_2(d,n)=\frac{1}{d}\operatorname{tr}(S^2)=\frac{1}{d}\sum_{a,b}S_{ab}S_{ba} =\frac{1}{d}\sum_{a,b}\left(\frac{1}{n}\sum_t G_{at}G_{bt}\right)\left(\frac{1}{n}\sum_s G_{bs}G_{as}\right).
$$

Thus

$$
\mathbb{E}[m_2] = \frac{1}{d\,n^2} \sum_{a,b}\sum_{t,s} \mathbb{E}\big[G_{at}G_{bt}G_{bs}G_{as}\big].
$$

Using independence and Wick pairings, the surviving terms yield

$$
\mathbb{E}[m_2] \to 1+\gamma.
$$

Indeed $m_2(\gamma)=1+\gamma$ from the closed form above.

**Step 5 (identify the limiting law).**
A compactly supported probability measure is uniquely determined by its moments under mild conditions; the above moment sequence matches exactly the moments of $\rho\_{\mathrm{MP}}^{(\gamma)}$. Therefore $\mu_S\Rightarrow \mu\_{\mathrm{MP}}^{(\gamma)}$.

### 9.2.4 Immediate ML interpretation: a single random layer is a "spectral filter"

For a Gaussian layer $W=\frac{\sigma}{\sqrt{d\_{\text{in}}}}G$ with aspect ratio $\gamma=d\_{\text{out}}/d\_{\text{in}}$, the singular values of $W$ concentrate near

$$
s \in \sigma\,[\,1-\sqrt{\gamma},\,1+\sqrt{\gamma}\,],
$$

and in particular

$$
\|W\|\_{\mathrm{op}} \approx \sigma(1+\sqrt{\gamma}).
$$

Hence even "pure noise" weights are not spectrally arbitrary; they impose a deterministic band of amplification and attenuation directions.

---

## 9.3 Xavier/He Initialization Revisited: From Variance Heuristics to Jacobian Control

### 9.3.1 The classical variance condition (and its limitation)

Consider one layer $h = Wx$ with $x\in\mathbb{R}^{d\_{\text{in}}}$.
Assume $x_i$ are i.i.d. with $\mathbb{E}[x_i]=0$, $\mathbb{E}[x_i^2]=q$, and $W_{ij}$ i.i.d. with $\mathbb{E}[W_{ij}]=0$, $\mathrm{Var}(W_{ij})=\sigma_w^2/d\_{\text{in}}$.

Then for each output coordinate:

$$
h_j = \sum_{i=1}^{d\_{\text{in}}} W_{ji}x_i,\quad \mathbb{E}[h_j]=0,
$$

and

$$
\mathbb{E}[h_j^2] = \sum_{i=1}^{d\_{\text{in}}}\mathbb{E}[W_{ji}^2]\mathbb{E}[x_i^2] = d\_{\text{in}}\cdot \frac{\sigma_w^2}{d\_{\text{in}}}\cdot q = \sigma_w^2 q.
$$

Thus "variance preservation" suggests $\sigma_w^2\approx 1$.
However, this scalar condition does *not* control $\|W\|\_{\mathrm{op}}$ nor the singular value spread (which dictates the conditioning of information propagation). RMT supplies the missing operator-scale constraints.

### 9.3.2 ReLU changes the stability criterion by modifying the Jacobian

For $x^{(\ell)}=\phi(h^{(\ell)})$ with $h^{(\ell)}=W^{(\ell)}x^{(\ell-1)}$, the Jacobian through one layer is

$$
J_\ell \;=\; D_\ell W^{(\ell)}, \quad D_\ell = \mathrm{diag}\big(\phi'(h^{(\ell)})\big).
$$

For ReLU, $\phi'(u)=\mathbf{1}\_{\{u>0\}}$, so $D_\ell$ is a random diagonal mask with roughly half ones (under symmetric pre-activation distributions). Consequently, the stability of backprop depends on the product

$$
J = J_L J_{L-1}\cdots J_1,
$$

and not merely on the forward variance.

### 9.3.3 The critical Jacobian condition (a non-negotiable quantitative constraint)

A minimal requirement for trainability is that typical gradient norms do not explode/vanish exponentially with depth. A tractable proxy is the mean squared singular value of $J$:

$$
\frac{1}{d}\mathbb{E}\,\mathrm{tr}(JJ^\top),
$$

or, for layerwise propagation, the multiplicative factor

$$
\chi \;=\; \sigma_w^2\,\mathbb{E}[\phi'(Z)^2], \quad Z\sim\mathcal{N}(0,q\_{\star}),
$$

where $q\_{\star}$ is the stationary pre-activation variance (derived in §9.5).

Under standard mean-field approximations, one obtains

$$
\mathbb{E}\| \delta^{(\ell)} \|^2 \approx \chi^{L-\ell}\,\mathbb{E}\|\delta^{(L)}\|^2.
$$

Therefore:

- $\chi<1$: gradients vanish exponentially (ordered phase).
- $\chi>1$: gradients explode exponentially (chaotic phase).
- $\chi=1$: critical regime (edge of chaos), the only regime compatible with very deep optimization absent architectural corrections.

For ReLU, $\phi'(Z)^2=\mathbf{1}\_{\{Z>0\}}$, so $\mathbb{E}[\phi'(Z)^2]=1/2$. Hence criticality forces

$$
\chi=1 \quad\Longrightarrow\quad \sigma_w^2\cdot \frac{1}{2} = 1 \quad\Longrightarrow\quad \sigma_w^2=2,
$$

which yields the He scaling $\mathrm{Var}(W_{ij})=2/d\_{\text{in}}$.

This derivation is not a "rule of thumb": it is the Jacobian-stability constraint expressed in the only algebra available at initialization time.

---

## 9.4 Products of Random Matrices and Dynamical Isometry

### 9.4.1 Why depth is qualitatively different: multiplicative spectra

Let $A_1,\dots,A_L$ be random matrices representing layer Jacobians (including activation derivatives). Even if each $\|A_\ell\|\_{\mathrm{op}}$ is close to 1, the product

$$
J = A_L\cdots A_1
$$

can be badly conditioned because singular values multiply in a nontrivial way. A useful invariant is the log-singular spectrum:

$$
\log s_i(J) \;\;\text{behaves like an additive process over layers.}
$$

Thus small per-layer biases in log-singular values accumulate linearly in $L$, yielding exponential effects in singular values.

### 9.4.2 Dynamical isometry: the correct operator goal

Define dynamical isometry as the condition that the singular values of $J$ concentrate near 1:

$$
s_i(J)\approx 1\quad\text{for most }i.
$$

This is strictly stronger than $\mathbb{E}\|Jv\|^2\approx \|v\|^2$ for random $v$; it requires the *entire spectrum* to be well-conditioned. The reason is algorithmic: SGD is sensitive to the worst-conditioned directions through curvature and gradient transport.

### 9.4.3 Deep linear networks: exact isometry from orthogonality

If $f(x)=W_L\cdots W_1 x$ and each $W_\ell$ is orthogonal (square case) or appropriately scaled semi-orthogonal (rectangular case), then

$$
J = W_L\cdots W_1
$$

is orthogonal (or an isometry on the relevant subspace). Hence

$$
J^\top J = I \quad\Rightarrow\quad s_i(J)=1\ \ \forall\, i.
$$

This is the mathematically clean baseline: isometry is achievable in the linear setting by controlling the singular values of each factor.

### 9.4.4 Nonlinear networks: why ReLU obstructs perfect isometry

With ReLU, $A_\ell=D_\ell W_\ell$ includes a diagonal Bernoulli mask. Even if $W_\ell$ is orthogonal, $D_\ell$ destroys exact isometry by randomly deleting coordinates, broadening the singular spectrum. The lesson is conceptual: "good initialization" must be formulated as a statement about the spectrum of *Jacobian products*, not about the distribution of weights alone.

---

## 9.5 Mean-Field Signal Propagation and the Edge of Chaos

### 9.5.1 Forward variance recursion (complete derivation)

Consider the fully-connected recursion

<p>

$$
h^{(\ell)} = W^{(\ell)}x^{(\ell-1)} + b^{(\ell)},\quad x^{(\ell)}=\phi\!\left(h^{(\ell)}\right),
$$

</p>

with $W^{(\ell)}\_{ij}\sim \mathcal{N}(0,\sigma_w^2/d)$, $b^{(\ell)}\_i\sim\mathcal{N}(0,\sigma_b^2)$, independent across layers.

Assume (mean-field hypothesis) that for large width $d$, the coordinates of $h^{(\ell)}$ are approximately i.i.d. Gaussian with mean 0 and variance $q_\ell=\mathbb{E}[(h^{(\ell)}\_i)^2]$. Then

$$
h^{(\ell)}\_i = \sum_{j=1}^d W^{(\ell)}\_{ij}x^{(\ell-1)}\_j + b^{(\ell)}\_i.
$$

Taking expectation of the square and using independence:

<p>

$$
\mathbb{E}\big[(h^{(\ell)}\_i)^2\big] = \mathbb{E}\left[\left(\sum_{j=1}^d W_{ij}^{(\ell)}x_j^{(\ell-1)}\right)^2\right] + \mathbb{E}\big[(b^{(\ell)}\_i)^2\big],
$$

</p>

and expand the square:

$$
\mathbb{E}\left[\left(\sum_{j=1}^d W_{ij}^{(\ell)}x_j^{(\ell-1)}\right)^2\right] = \sum_{j=1}^d \mathbb{E}\big[(W_{ij}^{(\ell)})^2\big]\mathbb{E}\big[(x_j^{(\ell-1)})^2\big],
$$

since cross-terms vanish by independence and centering. Substitute $\mathbb{E}[(W_{ij}^{(\ell)})^2]=\sigma_w^2/d$ and $\mathbb{E}[(x_j^{(\ell-1)})^2]=\mathbb{E}[\phi(Z)^2]$ for $Z\sim\mathcal{N}(0,q\_{\ell-1})$:

<p>

$$
q_\ell = \sigma_w^2\,\mathbb{E}\_{Z\sim\mathcal{N}(0,q\_{\ell-1})}[\phi(Z)^2] + \sigma_b^2.
$$

</p>

A fixed point $q\_{\star}$ satisfies

<p>

$$
q\_{\star} = \sigma_w^2\,\mathbb{E}\_{Z\sim\mathcal{N}(0,q\_{\star})}[\phi(Z)^2] + \sigma_b^2.
$$

</p>

### 9.5.2 Backward sensitivity recursion and the critical parameter $\chi$

Let $\delta^{(\ell)} = \partial\mathcal{L}/\partial h^{(\ell)}$ denote backpropagated error at layer $\ell$. Then

$$
\delta^{(\ell-1)} = \left(W^{(\ell)}\right)^\top D_\ell\,\delta^{(\ell)}, \quad D_\ell=\mathrm{diag}\big(\phi'(h^{(\ell)})\big).
$$

Under the same mean-field assumptions and isotropy, one derives the approximate squared-norm recursion:

$$
\mathbb{E}\|\delta^{(\ell-1)}\|^2 \approx \chi\;\mathbb{E}\|\delta^{(\ell)}\|^2, \quad \chi = \sigma_w^2\,\mathbb{E}[\phi'(Z)^2],\quad Z\sim\mathcal{N}(0,q\_{\star}).
$$

Thus gradient stability across depth requires $\chi\approx 1$.

### 9.5.3 Concrete example: ReLU and the He critical scaling

For ReLU, $\phi(z)=\max\{0,z\}$ and $\phi'(z)=\mathbf{1}\_{\{z>0\}}$. If $Z$ is symmetric, $\mathbb{P}(Z>0)=1/2$, hence

$$
\mathbb{E}[\phi'(Z)^2]=\mathbb{E}[\mathbf{1}\_{\{Z>0\}}]=\frac{1}{2}.
$$

The criticality condition $\chi=1$ implies $\sigma_w^2=2$, i.e.

$$
\mathrm{Var}(W_{ij})=\frac{2}{d}.
$$

This is the mathematically correct reading: "He initialization" enforces critical Jacobian transport in the mean-field limit.

### 9.5.4 A modern caution: dropout and the destruction of criticality

If multiplicative noise is inserted (e.g., dropout masks), the recursion for $\chi$ changes; the critical point can shift or disappear depending on noise structure. This provides a theoretical explanation for why naive dropout can sharply reduce the trainable depth unless compensated by architectural normalization or residual pathways.

---

## 9.6 Beyond MLPs: Residual Networks, Normalization, and Transformers

### 9.6.1 Residual connections as spectral anchoring

A residual block has the form

<p>

$$
x_{\ell+1} = x_\ell + \alpha\,F_\ell(x_\ell).
$$

</p>

Its Jacobian is

<p>

$$
J_{\ell} = I + \alpha\,\nabla F_\ell(x_\ell).
$$

</p>

Even when $\nabla F_\ell$ is random-like, the identity term anchors singular values near 1 if $\alpha$ is scaled appropriately with depth. This changes the regime from multiplicative explosion/vanishing to controlled deviation from isometry.

Mean-field analyses show that residual connections can convert exponential depth scales into polynomial ones, thereby expanding the trainable depth without requiring perfect critical tuning of $\sigma_w^2$.

### 9.6.2 Transformers: why normalization and residual scaling are not "engineering details"

Transformer blocks combine residual pathways with normalization (LayerNorm) and attention/MLP sublayers. The relevant stability object remains Jacobian singular values, but LayerNorm introduces data-dependent rescaling that can suppress or amplify gradients depending on activation magnitude.

A key modern observation: extremely deep Transformers require explicit control of update magnitudes and Jacobian conditioning via depth-dependent residual scaling and normalization design, not just Xavier/He at the layer level. This is visible in architectures that modify residual branches and prescribe depth-dependent constants to bound update norms.

### 9.6.3 Spectral diagnostics as a practical corollary

RMT predicts that trainability correlates with the Jacobian singular value distribution. Hence one can empirically monitor:

- the spectral norm of layer Jacobians,
- the spread (condition number) of $J^\top J$ over batches,
- "spectral entropy" of singular values as an indicator of collapse into low-dimensional channels.

This reframes initialization and early training as *spectral-shaping problems*.

---

## 9.7 Wigner's Semicircle Law and the Geometry of the Loss Landscape at Initialization

### 9.7.1 The Hessian as a random-plus-structure operator

Let $\theta$ be parameters and $\mathcal{L}(\theta)$ the training loss. The Hessian

$$
H(\theta) = \nabla^2\_\theta \mathcal{L}(\theta)
$$

governs local curvature. At random initialization, many contributions to $H$ behave like sums of weakly dependent random terms; thus RMT gives a first-order description of its eigenvalue bulk.

### 9.7.2 The semicircle law (core moment derivation sketch)

For a Wigner matrix $A\in\mathbb{R}^{n\times n}$ with symmetric entries, mean 0, variance $1/n$, the ESD of eigenvalues converges to the semicircle density

$$
\rho\_{\mathrm{sc}}(\lambda) = \frac{1}{2\pi}\sqrt{4-\lambda^2}\,\mathbf{1}\_{[-2,2]}(\lambda).
$$

The moment method parallels §9.2: one expands

$$
\frac{1}{n}\mathbb{E}\mathrm{tr}(A^k) = \frac{1}{n}\sum_{i_1,\dots,i_k}\mathbb{E}[A_{i_1 i_2}\cdots A_{i_k i_1}],
$$

and counts the leading closed-walk contributions, again dominated by noncrossing pairings, yielding Catalan moments.

### 9.7.3 ML implication: saddles dominate in high dimension

Because the bulk spectrum under semicircle-like behavior is roughly symmetric about zero, a random point in a high-dimensional nonconvex landscape typically has both positive and negative curvature directions. Thus "optimization difficulty" is not primarily the multiplicity of poor local minima, but the prevalence of saddle geometry and ill-conditioned directions. SGD's noise can help traverse such regions, but only if gradient transport (Jacobians) is stable �?linking loss-landscape geometry back to initialization.

---

## 9.8 Frontier Perspective: Scaling Laws, $\mu$-Parametrization, and Initialization as a Limit Theorem

### 9.8.1 The modern target: stable behavior across model size

In contemporary large-scale training (e.g., very wide or very deep models), the objective is not merely to initialize *one* network well, but to ensure a family of networks behaves predictably under scaling (width, depth, batch size). This pushes initialization from "pick a variance" to "choose a parametrization that yields a non-degenerate limit."

### 9.8.2 A unifying principle

RMT and infinite-width theory jointly suggest a unifying design principle:

> Choose scaling so that the relevant random operators (layer maps, Jacobians, update operators) converge to nontrivial limiting laws with bounded spectra.

In this view, initialization is a statement about limits: we design the parameter distribution so that (i)&nbsp;forward signal statistics converge, (ii)&nbsp;backward transport is stable, and (iii)&nbsp;optimization dynamics does not degenerate as dimension increases.

---

## 9.9 Summary: What RMT Actually Explains in Deep Learning

1. Randomness at initialization is *spectrally deterministic* in high dimension: operator norms and spectral measures concentrate.
2. The Marchenko&ndash;Pastur law provides the canonical spectral template for Gaussian layers, fixing singular-value support and hence signal amplification bounds.
3. Xavier/He initializations are best understood as enforcing *critical transport* (especially of Jacobians), not as superficial variance-matching recipes.
4. Depth creates multiplicative spectral phenomena; dynamical isometry explains why controlling the *entire singular spectrum* matters, and why orthogonal structure can accelerate learning.
5. Modern architectures (ResNets, Transformers) succeed in part because residual and normalization mechanisms reshape Jacobian spectra �?an operator-theoretic, not merely heuristic, explanation.

The next mathematical step is to understand optimization regimes where geometry is controlled globally (e.g., convexity, cones, and monotone operators), and to identify precisely which parts of deep learning behave "as if convex" due to spectral regularities established at initialization.

*Next: [Chapter 10: Convexity, Duality, and Optimization Geometry](/book/chapters/chapter010/)*

*Next: [Chapter 10: Convexity, Duality, and Optimization Geometry](/book/chapters/chapter010/)*
