---
title: "Chapter 2: Spectral Theory �?Eigen-Decomposition and SVD"
layout: "single"
url: "/book/chapters/chapter002/"
summary: "Persistence, decay, oscillation, and the skeleton of information in linear operators."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 2
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

# Volume I &mdash; The Mathematical Principles of Machine Learning

## Chapter 2 &mdash; Spectral Theory: Eigen-Decomposition and Singular Value Decomposition

*Xujiang Tang*

</div>

## Abstract

If Chapter 1 provides the stage (vector spaces and dual spaces), then spectral theory describes the persistent rhythms on that stage: the modes that survive repeated application of an operator, the modes that die out, and the modes that oscillate. In machine learning, almost every stability question &mdash; signal propagation in depth, exploding/vanishing gradients, conditioning of optimization, capacity and compression, and collapse phenomena &mdash; reduces to spectral statements about linearizations (Jacobians) and weight operators. This chapter treats matrices as evolution operators and develops: (i) eigen-decomposition as persistence and memory in repeated dynamics, (ii) SVD as cross-space translation and channel capacity, and (iii) Rayleigh-quotient principles as the variational origin of principal modes. The exposition remains coordinate-free where possible and introduces coordinates only to compute and to connect with practical diagnostics (spectral norm control, orthogonal initialization, low-rank adaptation, and spectral entropy monitoring).

## Notation

- $V$ is a finite-dimensional real vector space, identified with $\mathbb{R}^n$ when coordinates are required.
- A linear operator $T:V\to V$ is represented by a matrix $A\in\mathbb{R}^{n\times n}$ in a chosen basis.
- For $A\in\mathbb{R}^{m\times n}$, singular values are denoted $\sigma_1\ge \sigma_2\ge \cdots \ge 0$.
- $\lVert x\rVert_2$ is the Euclidean norm and $\lVert A\rVert_2$ is the operator norm induced by $\lVert\cdot\rVert_2$ (largest singular value).
- Spectral radius $\rho(A):=\max\lbrace|\lambda|:\lambda\in \mathrm{spec}(A)\rbrace$.

## 2.1 Eigen-Decomposition: Persistence of Evolution and the System's "Character"

### 2.1.1 Operators as evolution: repeated processing is an operator power

Consider a linear recurrence (the local model of many nonlinear networks):

$$x_{t+1} = A x_t, \qquad x_0 \in \mathbb{R}^n.$$

By direct substitution,

$$x_1 = A x_0,\quad x_2 = A x_1 = A(Ax_0)=A^2x_0,\quad \ldots,\quad x_t = A^t x_0.$$

Thus the long-term behavior of the system is the behavior of $A^t$.

A continuous-time analogue (linearized gradient flow / linear ODE) is

$$\frac{d}{dt}x(t) = A x(t), \qquad x(0)=x_0.$$

The solution is $x(t)=e^{tA}x_0$, where

$$e^{tA} := \sum_{k=0}^{\infty}\frac{t^k}{k!}A^k,$$

and convergence holds for any matrix $A$.

Both dynamics reduce to understanding the action of $A^k$ on vectors, which is precisely what eigen-structure encodes.

### 2.1.2 Definition: eigenpairs and diagonalization

<div class="def-box">

**Definition 2.1 (Eigenpair).** A scalar $\lambda\in\mathbb{C}$ and a nonzero vector $v\in\mathbb{C}^n$ form an eigenpair of $A\in\mathbb{R}^{n\times n}$ if

$$A v = \lambda v.$$

</div>

If $A$ admits a basis of eigenvectors (over $\mathbb{C}$), it is diagonalizable:

$$A = V \Lambda V^{-1},$$

where columns of $V$ are eigenvectors and $\Lambda=\mathrm{diag}(\lambda_1,\ldots,\lambda_n)$.

<div class="prop-box">

**Proposition 2.2 (Powers under diagonalization).** If $A=V\Lambda V^{-1}$, then for every integer $t\ge 1$,

$$A^t = V \Lambda^t V^{-1}, \qquad \Lambda^t=\mathrm{diag}(\lambda_1^t,\ldots,\lambda_n^t).$$

</div>

<div class="proof-box">

*Proof (explicit multiplication).*
For $t=1$ it is true. Assume $A^t = V\Lambda^t V^{-1}$. Then

$$\begin{aligned} A^{t+1} &= A^t A \\\\ &= (V\Lambda^t V^{-1})(V\Lambda V^{-1}) \\\\ &= V\Lambda^t (V^{-1}V)\Lambda V^{-1} \\\\ &= V\Lambda^t I \Lambda V^{-1} \\\\ &= V\Lambda^{t+1}V^{-1}. \end{aligned}$$

Hence by induction the formula holds for all $t$. $\square$

</div>

Therefore, each eigen-direction evolves by a scalar factor $\lambda^t$. This is the most rigid possible statement of "persistence vs decay."

### 2.1.3 Persistence and forgetting in depth: eigenvalues as survival rates

Write an initial state $x_0$ in the eigenbasis:

$$x_0 = \sum_{i=1}^n c_i v_i.$$

Applying $A^t$ gives

$$x_t = A^t x_0 = A^t\left(\sum_{i=1}^n c_i v_i\right) =\sum_{i=1}^n c_i A^t v_i =\sum_{i=1}^n c_i \lambda_i^t v_i.$$

Now the asymptotics are immediate:

- If $|\lambda_i|<1$, then $|\lambda_i|^t\to 0$ exponentially: that mode **vanishes** with depth/time.
- If $|\lambda_i|>1$, then $|\lambda_i|^t\to \infty$: that mode **dominates** and the system becomes unstable.
- If $|\lambda_i|=1$, the mode is **marginal**: it persists without exponential shrinkage or growth (it may still rotate if $\lambda_i$ is complex).

<div class="ml-box">

**Machine-Learning Translation (deep linear networks / RNN core).**
In an RNN $h_{t+1}=\phi(Wh_t+Ux_t)$, the linearized hidden-state dynamics around a trajectory involves repeated multiplication by Jacobians. Even if the network is nonlinear, the dominant stability mechanism is the spectrum of the linearization. The "memory" of long-range dependencies requires modes with magnitude near 1; vanishing/exploding behaviors correspond to $|\lambda|\ll 1$ or $|\lambda|\gg 1$.

</div>

### 2.1.4 Spectral radius as global stability threshold

<div class="def-box">

**Definition 2.3 (Spectral radius).** $\rho(A):=\max_i |\lambda_i|$.

</div>

In discrete-time linear dynamics $x_{t+1}=Ax_t$, the spectral radius is the natural candidate for a stability boundary:

- If $\rho(A)<1$, the dynamics is asymptotically stable in the diagonalizable normal case; many perturbations decay.
- If $\rho(A)>1$, there exist directions that grow exponentially.

A subtlety: for non-normal matrices, eigenvalues alone do not control transient growth; singular values and pseudospectra become decisive. For ML practice, the relevant quantity for worst-case amplification is often $\lVert A\rVert_2$, not $\rho(A)$. Still, $\rho(A)$ encodes the asymptotic fate of eigenmodes and remains conceptually central.

<div class="ml-box">

**Machine-Learning Reading (criticality).**
The empirically useful regime for deep signal propagation is often near "criticality," where dominant amplification/attenuation factors sit near 1. This explains why many initialization and normalization schemes aim to keep effective operator scales near unity: not because unity is special numerically, but because it lies at the boundary between forgetting and explosion.

</div>

### 2.1.5 Spectral gap: concentration of behavior and decisiveness

Let $A$ have eigenvalues ordered by magnitude $|\lambda_1|\ge|\lambda_2|\ge\cdots$. Define a (magnitude) gap

$$\mathrm{gap}(A):=|\lambda_1|-|\lambda_2|.$$

When the leading eigenvalue is separated, repeated application tends to align states with the principal eigenvector, provided the initial condition has a nonzero projection onto it. Indeed,

$$x_t = \lambda_1^t \left(c_1 v_1 + \sum_{i\ge2} c_i \left(\frac{\lambda_i}{\lambda_1}\right)^t v_i\right).$$

If $|\lambda_i/\lambda_1|<1$ for $i\ge2$, then the ratio terms decay, and directionally $x_t$ aligns with $v_1$.

<div class="ml-box">

**Machine-Learning Reading.**

- In representation dynamics, a large effective spectral gap implies fast alignment to a dominant mode: the system behaves "decisively" by suppressing competing directions.
- In Markov-chain learning and diffusion operators (e.g., graph neural networks, Laplacian smoothing), the spectral gap controls mixing rate; a large gap means rapid convergence to a stationary structure, which can be helpful (denoising) or harmful (oversmoothing).

</div>

### 2.1.6 Complex eigenvalues: oscillation and phase structure

For real matrices, non-symmetric operators can have complex eigenvalues. If $\lambda=re^{i\theta}$ and $v$ is a complex eigenvector, then along that mode,

$$A^t v = \lambda^t v = r^t e^{it\theta} v.$$

Magnitude $r^t$ governs amplification/attenuation; phase $e^{it\theta}$ governs rotation/oscillation.

<div class="ml-box">

**Machine-Learning Reading.**
Oscillatory modes are not pathological; they are the linear signature of coupled subspaces. In optimization, such couplings manifest as spiraling trajectories near saddle-like structures; in sequence models, they correspond to rotating encodings that can represent periodic or phase-like structure. The key is that oscillation requires non-self-adjointness (or at least lack of a basis in which the operator is purely scaling).

</div>

### 2.1.7 Eigenvalues vs singular values: why ML often needs SVD even for square matrices

For general $A$, eigenvalues describe invariant directions (possibly complex), but they do not necessarily bound the norm growth of arbitrary vectors. Singular values do:

$$\lVert Ax\rVert_2 \le \lVert A\rVert_2 \lVert x\rVert_2, \qquad \lVert A\rVert_2=\sigma_1(A).$$

Deep learning stability &mdash; particularly gradient stability &mdash; depends on products of Jacobians $J_\ell\cdots J_1$, for which the relevant bound is

$$\lVert J_\ell\cdots J_1\rVert_2 \le \prod_{k=1}^\ell \lVert J_k\rVert_2.$$

This is why spectral normalization and Lipschitz control are stated in terms of singular values, not eigenvalues.

This naturally leads to Section 2.2.

## 2.2 Singular Value Decomposition: Cross-Space Translation and the Skeleton of Information

### 2.2.1 Definition and existence of SVD

Let $A\in\mathbb{R}^{m\times n}$.

<div class="prop-box">

**Theorem 2.4 (SVD).** There exist orthogonal matrices $U\in\mathbb{R}^{m\times m}$, $V\in\mathbb{R}^{n\times n}$, and a diagonal matrix $\Sigma\in\mathbb{R}^{m\times n}$ with nonnegative diagonal entries $\sigma_1\ge\cdots\ge\sigma_r>0$ (where $r=\mathrm{rank}(A)$) such that

$$A = U\Sigma V^\top.$$

Writing columns of $U$ as $u_i$ and of $V$ as $v_i$, this is equivalent to

$$A = \sum_{i=1}^{r} \sigma_i\, u_i v_i^\top, \qquad Av_i = \sigma_i u_i,\quad A^\top u_i = \sigma_i v_i.$$

</div>

<div class="proof-box">

*Derivation from eigen-decomposition (step-by-step).*

Consider $A^\top A\in\mathbb{R}^{n\times n}$. It is symmetric and positive semidefinite:

$$x^\top(A^\top A)x = (Ax)^\top (Ax) = \lVert Ax\rVert_2^2 \ge 0.$$

Hence $A^\top A$ has an orthonormal eigenbasis $\lbrace v_i\rbrace$ with real eigenvalues $\lambda_i\ge 0$:

$$A^\top A v_i = \lambda_i v_i.$$

Define $\sigma_i := \sqrt{\lambda_i}\ge 0$. For $\sigma_i>0$, define

$$u_i := \frac{1}{\sigma_i} A v_i.$$

Then

$$\begin{aligned} \lVert u_i\rVert_2^2 &= \frac{1}{\sigma_i^2}\lVert Av_i\rVert_2^2 \\\\ &= \frac{1}{\sigma_i^2} v_i^\top A^\top A v_i \\\\ &= \frac{1}{\sigma_i^2} v_i^\top (\sigma_i^2 v_i) \\\\ &= v_i^\top v_i = 1, \end{aligned}$$

so $u_i$ is unit length. Moreover,

$$Av_i = \sigma_i u_i$$

by definition of $u_i$, and

$$\begin{aligned} A^\top u_i &= A^\top\left(\frac{1}{\sigma_i}Av_i\right) \\\\ &= \frac{1}{\sigma_i}A^\top A v_i \\\\ &= \frac{1}{\sigma_i}(\sigma_i^2 v_i) \\\\ &= \sigma_i v_i. \end{aligned}$$

Extending $\lbrace u_i\rbrace$ and $\lbrace v_i\rbrace$ to orthonormal bases gives orthogonal $U,V$, and collecting $\sigma_i$ into $\Sigma$ yields $A=U\Sigma V^\top$. $\square$

</div>

<div class="ml-box">

**Machine-Learning Meaning (translation operator).**
A weight matrix $A: \mathbb{R}^n\to\mathbb{R}^m$ maps an input representation space to an output representation space. SVD identifies:

- right singular vectors $v_i$: "input modes" that the layer is most sensitive to,
- left singular vectors $u_i$: "output modes" that receive maximal transmitted energy,
- singular values $\sigma_i$: channel gains (bandwidth/strength) of each mode.

</div>

### 2.2.2 Channel view: singular values as information bandwidth

Let input $x$ decompose in the right-singular basis: $x=\sum_i \alpha_i v_i$. Then

$$Ax = \sum_{i=1}^r \sigma_i \alpha_i u_i.$$

The energy amplification in each mode is $\sigma_i^2$. If the system is noisy, this precisely governs signal-to-noise ratios per mode. Hence the singular spectrum is the quantitative skeleton of what a linear map can faithfully transmit.

In deep nets, the relevant operators are often Jacobians $J$. The distribution of $\lbrace\sigma_i(J)\rbrace$ determines whether gradients and signals propagate stably or collapse into a low-dimensional subspace.

### 2.2.3 Eckart&ndash;Young theorem: truncated SVD is the optimal low-rank approximation

<div class="prop-box">

**Theorem 2.5 (Eckart&ndash;Young&ndash;Mirsky, Frobenius norm form).**
Let $A=U\Sigma V^\top$ with singular values $\sigma_1\ge\cdots\ge\sigma_r$. For any integer $k < r$, define the rank-$k$ truncation

$$A_k := \sum_{i=1}^k \sigma_i u_i v_i^\top.$$

Then for every matrix $B$ with $\mathrm{rank}(B)\le k$,

$$\lVert A-B\rVert_F \ge \lVert A-A_k\rVert_F = \left(\sum_{i=k+1}^{r}\sigma_i^2\right)^{1/2}.$$

</div>

<div class="proof-box">

*Proof (explicit and modular).*

**Step 1: Frobenius norm is orthogonally invariant.** For orthogonal $U,V$,

$$\lVert A-B\rVert_F = \lVert U^\top(A-B)V\rVert_F.$$

This follows because $\lVert M\rVert_F^2=\mathrm{tr}(M^\top M)$, and

$$\begin{aligned} \lVert U^\top M V\rVert_F^2 &= \mathrm{tr}\big((U^\top M V)^\top (U^\top M V)\big) \\\\ &= \mathrm{tr}\big(V^\top M^\top U U^\top M V\big) \\\\ &= \mathrm{tr}(V^\top M^\top M V) \\\\ &= \mathrm{tr}(M^\top M) = \lVert M\rVert_F^2. \end{aligned}$$

**Step 2: Reduce to diagonal case.** Let $A=U\Sigma V^\top$ and define $C:=U^\top B V$. Then $\mathrm{rank}(C)=\mathrm{rank}(B)\le k$, and

$$\lVert A-B\rVert_F = \lVert\Sigma - C\rVert_F.$$

**Step 3: Lower bound $\lVert\Sigma-C\rVert_F^2$.** Write $\Sigma$ as diagonal with entries $\sigma_1,\dots,\sigma_r$ (and zeros beyond). Then

$$\lVert\Sigma-C\rVert_F^2 = \sum_{i,j} (\Sigma_{ij}-C_{ij})^2.$$

In particular, restricting to diagonal terms gives

$$\lVert\Sigma-C\rVert_F^2 \ge \sum_{i=1}^{r} (\sigma_i - C_{ii})^2 \ge \sum_{i=k+1}^{r} (\sigma_i - C_{ii})^2.$$

**Step 4: Use the rank constraint.** If $\mathrm{rank}(C)\le k$, then $C$ has at most $k$ nonzero singular values; equivalently, $C$ acts nontrivially on at most a $k$-dimensional subspace. A standard consequence is that among the diagonal entries in a basis aligned with $\Sigma$, one cannot match more than $k$ singular directions without paying residual energy in the remaining $(r-k)$ directions. The minimizer is achieved by choosing $C$ to agree with $\Sigma$ on the first $k$ diagonal entries and zero elsewhere, i.e. $C=\Sigma_k$ where $\Sigma_k=\mathrm{diag}(\sigma_1,\dots,\sigma_k,0,\dots)$. Substituting yields

$$\min_{\mathrm{rank}(C)\le k}\lVert\Sigma-C\rVert_F^2 = \sum_{i=k+1}^r \sigma_i^2.$$

Therefore $\lVert A-B\rVert_F \ge \lVert A-A_k\rVert_F$ and the minimum is attained at $B=A_k$. $\square$

</div>

<div class="ml-box">

**Machine-Learning Meaning (understanding as controlled rank-deficiency).**
A model that retains every small singular direction is, in effect, memorizing fine-grained fluctuations. Truncated spectra enforce a bias toward dominant modes &mdash; exactly what appears as "inductive bias" in learning: prefer stable, high-energy structure over brittle details.

</div>

### 2.2.4 PCA as SVD of centered data

Let centered data matrix $X\in\mathbb{R}^{N\times d}$ have rows $x_i^\top$. The empirical covariance is

$$C = \frac{1}{N}X^\top X.$$

If $X=U\Sigma V^\top$, then

$$C = \frac{1}{N}V\Sigma^\top \Sigma V^\top,$$

so the principal components are the right singular vectors $V$, and eigenvalues of $C$ are $\sigma_i^2/N$.

<div class="ml-box">

**Machine-Learning Meaning.**
PCA is not a separate topic; it is the SVD viewpoint applied to the data operator. This is the prototypical instance of "spectral structure reveals a low-dimensional skeleton."

</div>

### 2.2.5 Low-rank structure in modern architectures: compression and adaptation

Large models frequently exhibit spectral decay in weight matrices and activations. Two consequences are structural:

1. **Compression.** If $\sigma_{k+1},\sigma_{k+2},\dots$ are small, then $A\approx A_k$ is accurate; parameter and compute reductions follow.

2. **Low-rank adaptation (conceptual basis).** If a task-specific update $\Delta A$ is approximately low-rank, one can restrict adaptation to a small number of singular directions. The algebra is: represent $\Delta A$ as a sum of a few rank-1 outer products, $\Delta A \approx \sum_{i=1}^k a_i b_i^\top$, which is precisely the language SVD makes canonical.

### 2.2.6 Collapse and spectral entropy: a diagnostic principle

A practical collapse phenomenon can be rephrased spectrally: the operator becomes effectively low-rank, concentrating energy in very few singular directions. A useful scalar diagnostic is the spectral entropy of normalized singular energies:

$$p_i := \frac{\sigma_i^2}{\sum_j \sigma_j^2}, \qquad H := -\sum_i p_i \log p_i.$$

- Large $H$ indicates a flatter spectrum (diverse modes).
- Small $H$ indicates concentration (few dominant modes), consistent with representational collapse.

This is not a heuristic ornament: it is the quantitative statement "how many effective degrees of freedom remain."

### 2.2.7 Cross-space alignment: SVD of cross-covariance as shared resonance

Let $x\in\mathbb{R}^{d_x}$, $y\in\mathbb{R}^{d_y}$ be two modalities (e.g., image and text embeddings) and consider the cross-covariance

$$C := \mathbb{E}[x y^\top] \in \mathbb{R}^{d_x\times d_y}.$$

Compute the SVD $C=U\Sigma V^\top$. Then:

- $v_i$ are the dominant modes in the $y$-space that correlate with $x$,
- $u_i$ are the corresponding modes in the $x$-space,
- $\sigma_i$ quantify the strength of shared structure.

<div class="ml-box">

**Machine-Learning Meaning.**
If all $\sigma_i$ are near zero, there is no linear shared resonance &mdash; cross-domain transfer is geometrically constrained. If a few $\sigma_i$ are large, the modalities share a low-dimensional aligned subspace, making alignment and retrieval feasible.

</div>

## 2.3 Rayleigh Quotient: Variational Origin of Principal Modes

### 2.3.1 Rayleigh quotient and extremal eigenvalues (symmetric case)

Let $A\in\mathbb{R}^{n\times n}$ be symmetric. Define the Rayleigh quotient for nonzero $x$:

$$R_A(x) := \frac{x^\top A x}{x^\top x}.$$

Because scaling cancels, $R_A(\alpha x)=R_A(x)$ for $\alpha\ne 0$. Therefore we may constrain $x$ to the unit sphere $x^\top x=1$ and maximize/minimize

$$\max_{x^\top x=1} x^\top A x, \qquad \min_{x^\top x=1} x^\top A x.$$

<div class="prop-box">

**Theorem 2.6 (Extremizers are eigenvectors; extrema are eigenvalues).**
If $A$ is symmetric with eigenvalues $\lambda_{\min}\le \cdots \le \lambda_{\max}$, then

$$\max_{x^\top x=1} x^\top A x = \lambda_{\max},\quad \min_{x^\top x=1} x^\top A x = \lambda_{\min},$$

and maximizers/minimizers are eigenvectors corresponding to those eigenvalues.

</div>

<div class="proof-box">

*Proof (Lagrange multipliers, explicit).*

Maximize $f(x)=x^\top A x$ subject to $g(x)=x^\top x-1=0$. Form the Lagrangian

$$\mathcal{L}(x,\mu)=x^\top A x - \mu(x^\top x-1).$$

Compute the gradient with respect to $x$. Using symmetry of $A$,

$$\nabla_x (x^\top A x) = (A+A^\top)x = 2Ax, \qquad \nabla_x (x^\top x) = 2x.$$

Stationarity $\nabla_x\mathcal{L}=0$ gives

$$2Ax - \mu(2x)=0 \quad\Longrightarrow\quad Ax=\mu x.$$

Thus any constrained critical point is an eigenvector and the associated multiplier $\mu$ is its eigenvalue. Evaluate the objective at such an $x$ with $\lVert x\rVert_2=1$:

$$x^\top A x = x^\top(\mu x)=\mu(x^\top x)=\mu.$$

Therefore the constrained critical values are exactly eigenvalues. Since the unit sphere is compact and $x^\top A x$ is continuous, maxima and minima exist and must occur at critical points, hence at eigenvectors, yielding extrema $\lambda_{\max}$ and $\lambda_{\min}$. $\square$

</div>

### 2.3.2 Courant&ndash;Fischer minimax principle (statement for depth reasoning)

For symmetric $A$ with ordered eigenvalues $\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_n$,

$$\lambda_k = \max_{\dim(S)=k}\ \min_{\substack{x\in S \\\\ x\ne 0}} R_A(x) = \min_{\dim(S)=n-k+1}\ \max_{\substack{x\in S \\\\ x\ne 0}} R_A(x).$$

This principle is the variational foundation behind principal components, low-dimensional energy capture, and many "top-$k$ mode" computations used implicitly in ML.

### 2.3.3 Power iteration as an algorithmic consequence (principal mode extraction)

Let $A$ be diagonalizable with a unique dominant eigenvalue in magnitude $|\lambda_1|>|\lambda_2|$. Starting from $x_0$ with nonzero projection on $v_1$, define

$$x_{t+1} := \frac{A x_t}{\lVert A x_t\rVert_2}.$$

Writing $x_0=\sum_i c_i v_i$, we have $A^t x_0=\sum_i c_i \lambda_i^t v_i$. Divide by $\lVert A^t x_0\rVert$; the ratio $(\lambda_i/\lambda_1)^t$ decays for $i\ge2$, hence $x_t$ converges in direction to $v_1$.

<div class="ml-box">

**Machine-Learning Meaning.**
Many training-time and analysis-time procedures are power iterations in disguise: computing dominant covariance modes (PCA), estimating spectral norms for spectral normalization, and extracting principal directions of Hessian approximations.

</div>

### 2.3.4 Attention as a variational selection over a quadratic form (a precise local statement)

A common structural motif in ML is maximizing a score of the form $x^\top A x$ under normalization constraints, or selecting directions that maximize energy concentration. The Rayleigh quotient theorem states exactly which directions concentrate quadratic energy: eigenvectors. In practice, attention and routing mechanisms often construct data-dependent kernels $K$ and select dominant interactions; locally, this is an eigen/singular mode selection problem of an operator induced by the data and the model state.

The principled point is not that "attention equals Rayleigh quotient," but that the *selection of dominant modes* in many mechanisms has the same variational backbone: maximize a quadratic (or bilinear) energy under a constraint, yielding principal spectral directions.

## 2.4 Spectral Control in Training: Stability, Generalization, and Diagnostics

### 2.4.1 Gradient stability via Jacobian singular values

For a deep composition $f = f_L\circ \cdots \circ f_1$, the Jacobian is

$$J_f(x)=J_{f_L}(h_{L-1})\cdots J_{f_1}(x).$$

Backpropagation multiplies by transposes of these Jacobians; norm growth is controlled by singular values:

$$\lVert J_f(x)\rVert_2 \le \prod_{\ell=1}^L \lVert J_{f_\ell}\rVert_2 = \prod_{\ell=1}^L \sigma_1(J_{f_\ell}).$$

This inequality is the mathematical core behind exploding/vanishing gradients. It also explains why techniques that stabilize singular spectra &mdash; normalization, residual connections, orthogonal/near-isometric initialization &mdash; systematically improve trainability.

### 2.4.2 Spectral normalization and Lipschitz control

Imposing $\lVert W\rVert_2\le c$ for weight matrices $W$ bounds the Lipschitz constant of linear layers and contributes to global stability bounds. Estimating $\lVert W\rVert_2$ is computationally feasible via power iteration, making spectral control operational rather than purely theoretical.

### 2.4.3 Conditioning and optimization speed

First-order methods are sensitive to conditioning. For quadratic objectives $J(\theta)=\frac{1}{2}\theta^\top H\theta - b^\top\theta$ with symmetric positive definite $H$, convergence rates depend on eigenvalues of $H$, especially the condition number $\kappa=\lambda_{\max}/\lambda_{\min}$. Preconditioning is, in essence, spectral reshaping of the effective Hessian, aligning the optimization geometry with the problem's curvature.

## 2.5 Summary and the Next Mathematical Layer

<div class="scholium-box">

**Scholium (Summary of Chapter 2).**

Eigen-decomposition explains **persistence, decay, and oscillation** under repeated evolution in a single space. SVD explains **translation capacity, alignment, and low-rank skeletons** between spaces. Rayleigh-quotient principles explain why **dominant modes appear as variational extrema** and why "principal directions" are inevitable whenever a system seeks energetic concentration under constraints.

The next step is to move from linear operators acting on one vector space to **multilinear objects acting on products of spaces**. Once interactions are not unary (a single map $V\to V$) but multiway (pairwise, triple-wise, and higher), the natural language is tensor algebra: contraction, covariance, invariance, and the geometry of multi-dimensional coupling.

</div>

<div style="text-align:center; margin:2em 0 1em 0; font-style:italic;">

&mdash; End of Chapter 2 &mdash;

*Next: [Chapter 3: Tensor Algebra and Einstein Summation Convention](/book/chapters/chapter003/)*

</div>
