---
title: "Chapter 20: Wiener Processes and Brownian Motion 鈥?The Canonical Noise Source"
layout: "single"
url: "/book/chapters/chapter020/"
summary: "The Wiener process defined axiomatically and via Gaussian-process covariance; multidimensional Brownian motion; random-walk scaling limit and Donsker's principle; Langevin viewpoint; pure diffusion and the heat equation via Fourier; quadratic variation computed from first principles; scores under Gaussian smoothing and the denoising bridge."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 20
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

## Chapter 20 &mdash; Wiener Processes and Brownian Motion: The Canonical Noise Source

*Definition, Gaussian-process characterization, scaling limits, quadratic variation, heat equation, and the denoising bridge*

*Xujiang Tang*

</div>

## Abstract

Chapter 19 used Brownian motion as a primitive. This chapter unpacks the primitive. The Wiener process is defined axiomatically, characterized as a Gaussian process via its covariance function \(\min(s,t)\), and derived as the scaling limit of discrete random walks. Its quadratic variation is computed from first principles 鈥?this is the analytic fact that forces It么's correction term. Pure Brownian diffusion is connected to the heat equation by a Fourier derivation. Finally, scores under Gaussian smoothing are computed explicitly, providing the analytical bridge that makes denoising score matching possible.

---

## 20.1 Definition: The Wiener Process

In the modern probabilistic literature, "Brownian motion" is the sample-path object; "Wiener process" often refers to the same object viewed as a canonical measure on path space. The two perspectives coincide once the underlying probability space and filtration are fixed.

### 20.1.1 Axiomatic Definition (One Dimension)

<div class="def-box">

**Definition 20.1 (standard Brownian motion).** A one-dimensional standard Brownian motion \((W_t)_{t\ge 0}\) relative to a filtration \((\mathcal{F}_t)_{t\ge 0}\) is an \(\mathbb{R}\)-valued process such that:

1. \(W_0 = 0\) almost surely.
2. **(Independent increments)** For \(0\le t_0 < t_1 < \cdots < t_n\), the increments \[W_{t_1}-W_{t_0},\quad W_{t_2}-W_{t_1},\quad \ldots,\quad W_{t_n}-W_{t_{n-1}}\] are independent.
3. **(Gaussian increments)** For \(0\le s < t\), \[W_t - W_s \sim \mathcal{N}(0,\; t-s).\]
4. **(Path regularity)** \(t\mapsto W_t(\omega)\) is continuous for almost every \(\omega\).

</div>

The filtration condition includes that \(W_t\) is \(\mathcal{F}_t\)-measurable and that increments are independent of \(\mathcal{F}_s\) for \(s<t\). This "adapted + independent increments" structure is the probabilistic counterpart of causality: the future evolution is independent of the past given the present.

### 20.1.2 Gaussian-Process Characterization via Covariance \(\min(s,t)\)

A centered Gaussian process is fully determined by its covariance function. For Brownian motion:

<div class="prop-box">

**Proposition 20.1.** If \((W_t)\) is a standard Brownian motion, then for all \(s,t\ge 0\),
\[
\mathbb{E}[W_s W_t] = \min(s,t).
\]

</div>

<div class="proof-box">

**Proof.** Assume \(0\le s\le t\). Write \(W_t = W_s + (W_t - W_s)\). Then:
\[
\mathbb{E}[W_s W_t] = \mathbb{E}[W_s^2] + \mathbb{E}[W_s(W_t-W_s)].
\]
By independent increments, \(W_t - W_s\) is independent of \(W_s\) and has mean zero, so:
\[
\mathbb{E}[W_s(W_t-W_s)] = \mathbb{E}[W_s]\cdot\mathbb{E}[W_t-W_s] = 0.
\]
Also, \(\mathbb{E}[W_s^2] = \mathrm{Var}(W_s) = s\). Therefore \(\mathbb{E}[W_s W_t] = s = \min(s,t)\). \(\square\)

</div>

**Converse direction.** If \((W_t)\) is a centered Gaussian process with covariance \(\min(s,t)\) and continuous paths, then it has stationary independent increments with the correct Gaussian law, hence is Brownian motion. The independence of increments follows because for a Gaussian vector, zero cross-covariance implies independence: for disjoint intervals \([s_1,t_1]\) and \([s_2,t_2]\) with \(t_1\le s_2\),
\[
\mathrm{Cov}(W_{t_1}-W_{s_1},\;W_{t_2}-W_{s_2}) = \min(t_1,t_2)-\min(t_1,s_2)-\min(s_1,t_2)+\min(s_1,s_2) = 0.
\]

### 20.1.3 Multidimensional Wiener Process

<div class="def-box">

**Definition 20.2 (standard \(d\)-dimensional Brownian motion).** A standard \(d\)-dimensional Brownian motion \(W_t\in\mathbb{R}^d\) consists of \(d\) independent one-dimensional Brownian motions \((W_t^{(1)},\dots,W_t^{(d)})\). For \(0\le s < t\):
\[
W_t - W_s \sim \mathcal{N}(0,\;(t-s)I_d), \qquad \mathbb{E}[W_s W_t^\top] = \min(s,t)\,I_d.
\]

</div>

This object is the canonical noise source in diffusion models: its increments are Gaussian with variance proportional to elapsed time, which matches the design goal of continuously increasing uncertainty with a known closed form at every time.

---

## 20.2 The Physical View: Brownian Motion as the Scaling Limit of Microscopic Agitation

The physics of Brownian motion predates its measure-theoretic formulation by decades. Two classical routes lead to the same mathematics.

### 20.2.1 Random Walk Scaling and Mean-Squared Displacement

Consider a symmetric random walk with step size \(\Delta x\) and time step \(\Delta t\):
\[
S_n = \sum_{k=1}^n \xi_k, \qquad \xi_k\in\{+\Delta x,-\Delta x\},\quad \mathbb{P}(\xi_k=\pm\Delta x)=\tfrac{1}{2}.
\]
Then \(\mathbb{E}[S_n]=0\) and \(\mathrm{Var}(S_n)=n(\Delta x)^2\). At physical time \(t=n\Delta t\):
\[
\mathbb{E}\!\left[S_{t/\Delta t}^2\right] = \frac{t}{\Delta t}(\Delta x)^2.
\]
Define the diffusion constant \(D := \frac{(\Delta x)^2}{2\Delta t}\). Then \(\mathbb{E}[S_{t/\Delta t}^2] = 2Dt\) 鈥?the **linear-in-time mean-squared displacement**, the macroscopic signature of diffusion.

<div class="proof-box">

**Scaling to Brownian motion.** Define the rescaled process:
\[
X_t^{(\Delta)} := \frac{S_{\lfloor t/\Delta t\rfloor}}{\sqrt{t/\Delta t}} \cdot \sqrt{t}.
\]
More concisely, the scaling is chosen so that \(\mathrm{Var}(X_t^{(\Delta)})=t\) and increments over disjoint time blocks are sums of disjoint \(\xi_k\)'s, hence independent. As \(\Delta t\to 0\) with \(D\) fixed:
- Finite-dimensional distributions converge to those of \(\sqrt{2D}\,W_t\) by the CLT.
- Tightness (hence path-level convergence) follows from the Kolmogorov鈥揅hentsov criterion.

This is Donsker's invariance principle: Brownian motion is the universal scaling limit of centered random walks with finite variance steps.

</div>

**Practical implication for diffusion models.** Brownian increment simulation at discretization step \(\Delta t\):
\[
W_{t+\Delta t} - W_t \approx \sqrt{\Delta t}\,\varepsilon, \qquad \varepsilon\sim\mathcal{N}(0,1).
\]
This is not an approximation of the process 鈥?it is the exact distribution of the increment. Euler鈥揗aruyama discretization of SDEs rests entirely on this fact.

### 20.2.2 Langevin Viewpoint: Friction and Noise

A physical model for a particle's velocity \(V_t\) balances friction and random molecular kicks:
\[
dV_t = -\gamma V_t\,dt + \sigma\,dW_t \qquad \text{(Ornstein鈥揢hlenbeck SDE).}
\]
Integrating velocity gives position \(dX_t = V_t\,dt\). In the **overdamped regime** (fast velocity relaxation, \(\gamma\to\infty\) with \(\sigma^2/(2\gamma)=D\) fixed), position dynamics reduce directly to:
\[
dX_t = \sqrt{2D}\,dW_t.
\]
This is the simplest forward diffusion: noise injection with variance growing linearly in time, the \(D=\tfrac{1}{2}\) version of which is pure Brownian motion.

<div class="ml-box">

The VP SDE in diffusion models is an Ornstein鈥揢hlenbeck process with time-varying coefficients:
\[
dX_t = -\tfrac{1}{2}\beta(t)X_t\,dt + \sqrt{\beta(t)}\,dW_t.
\]
In the limit of constant \(\beta\) and large \(\beta t\), the stationary distribution is \(\mathcal{N}(0,I)\). The Langevin viewpoint makes transparent why OU processes are the natural choice: they combine mean reversion (driving toward the prior) with noise injection (erasing data information).

</div>

---

## 20.3 Pure Diffusion and the Heat Equation

Consider the pure diffusion SDE in \(\mathbb{R}^d\) (no drift):
\[
dX_t = \sqrt{2D}\,dW_t, \qquad X_0\sim p_0.
\]
This has the explicit solution \(X_t = X_0 + \sqrt{2D}\,W_t\).

### 20.3.1 Conditional Law and Gaussian Smoothing

Conditioned on \(X_0 = x_0\),
\[
X_t\mid X_0=x_0 \;\sim\; \mathcal{N}(x_0,\;2Dt\,I_d).
\]
The marginal density is therefore a Gaussian convolution:
\[
p_t(x) = \int p_0(x_0)\,\varphi_{2Dt}(x-x_0)\,dx_0 = (p_0 * \varphi_{2Dt})(x),
\]
where the heat kernel is
\[
\varphi_{2Dt}(u) := \frac{1}{(4\pi Dt)^{d/2}}\exp\!\left(-\frac{\|u\|^2}{4Dt}\right).
\]

<div class="prop-box">

**Proposition 20.2 (forward noising as Gaussian smoothing).** Under pure Brownian diffusion with coefficient \(\sqrt{2D}\), the density at time \(t\) is the \(2Dt\)-Gaussian smoothing of the initial density:
\[
p_t = p_0 * \varphi_{2Dt}.
\]

</div>

This identity is fundamental for diffusion models: the forward noising process smooths the data distribution at progressively larger scales, making the marginals progressively more Gaussian.

### 20.3.2 Deriving the Heat Equation via Fourier Analysis

<div class="proof-box">

**Proof.** Let \(\widehat{p_t}(k) := \int_{\mathbb{R}^d} e^{-ik\cdot x}\,p_t(x)\,dx\) be the Fourier transform. Since convolution becomes pointwise multiplication:
\[
\widehat{p_t}(k) = \widehat{p_0}(k)\cdot\widehat{\varphi_{2Dt}}(k).
\]
The Fourier transform of the Gaussian kernel is:
\[
\widehat{\varphi_{2Dt}}(k) = \exp(-Dt\|k\|^2).
\]
Hence \(\widehat{p_t}(k) = \widehat{p_0}(k)\,\exp(-Dt\|k\|^2)\). Differentiating in \(t\):
\[
\partial_t\,\widehat{p_t}(k) = -D\|k\|^2\,\widehat{p_t}(k).
\]
Multiplying by \(-\|k\|^2\) in Fourier space corresponds to applying the Laplacian \(\Delta\) in physical space. Inverting the Fourier transform: \(\partial_t p_t(x) = D\,\Delta p_t(x)\). \(\square\)

</div>

<div class="prop-box">

**Theorem 20.1 (heat equation).** Under pure Brownian diffusion \(dX_t = \sqrt{2D}\,dW_t\), the density satisfies:
\[
\partial_t p_t(x) = D\,\Delta p_t(x).
\]

</div>

This is the PDE counterpart of the SDE. In diffusion models this PDE describes the "forward smoothing" mechanism. The reverse process must undo it 鈥?and the score field specifies precisely how the density contracts back.

---

## 20.4 Quadratic Variation: Why Stochastic Calculus Needs New Rules

Classical calculus relies on \((\Delta x)^2 \ll \Delta x\) as \(\Delta x\to 0\). Brownian increments violate this: \((\Delta W)^2\) accumulates to order \(1\), not order \(2\).

Let \(\pi = \{0 = t_0 < t_1 < \cdots < t_n = t\}\) be a partition of \([0,t]\) with mesh \(|\pi| = \max_k(t_{k+1}-t_k)\). Define:
\[
Q_\pi(W;\,t) = \sum_{k=0}^{n-1}(W_{t_{k+1}} - W_{t_k})^2.
\]

<div class="prop-box">

**Theorem 20.2 (quadratic variation of Brownian motion).** As \(|\pi|\to 0\), \(Q_\pi(W;t)\to t\) in \(L^2\) (and hence in probability). That is, the quadratic variation of Brownian motion satisfies \([W]_t = t\).

</div>

<div class="proof-box">

**Proof.**

**Step 1 (expectation).** By independent increments and \(\mathrm{Var}(W_{t_{k+1}}-W_{t_k}) = t_{k+1}-t_k\):
\[
\mathbb{E}[Q_\pi(W;t)] = \sum_{k=0}^{n-1}(t_{k+1}-t_k) = t.
\]

**Step 2 (variance).** Each squared increment \((W_{t_{k+1}}-W_{t_k})^2\) has variance \(2(t_{k+1}-t_k)^2\) (since if \(Z\sim\mathcal{N}(0,\sigma^2)\) then \(\mathrm{Var}(Z^2)=2\sigma^4\)). By independence of increments:
\[
\mathrm{Var}(Q_\pi(W;t)) = \sum_{k=0}^{n-1} 2(t_{k+1}-t_k)^2 \le 2|\pi|\sum_{k=0}^{n-1}(t_{k+1}-t_k) = 2|\pi|\,t.
\]

**Step 3 (convergence).** As \(|\pi|\to 0\): \(\mathrm{Var}(Q_\pi(W;t))\to 0\), so \(Q_\pi(W;t)\to t\) in \(L^2\). \(\square\)

</div>

**Differential mnemonic.** The quadratic variation identity is the rigorous content behind the It么 bookkeeping rules:
\[
(dW_t)^2 = dt, \qquad dt\,dW_t = 0, \qquad (dt)^2 = 0.
\]
These are not formal conventions 鈥?they are consequences of the \(L^2\) convergence above, made precise in the derivation of It么's formula (Chapter 19, 搂19.4).

**Contrast with smooth functions.** For a smooth function \(f\) with \(f'\) bounded, the quadratic variation of \(f(t)\) satisfies
\[
\sum_{k}(f(t_{k+1})-f(t_k))^2 \le \|f'\|_\infty^2 \sum_k (t_{k+1}-t_k)^2 \le \|f'\|_\infty^2 |\pi|\,t \to 0.
\]
Brownian paths are different: their quadratic variation is not zero, it is time itself. This is why Brownian paths are nowhere differentiable (differentiability would force zero quadratic variation) and why ordinary calculus fails.

---

## 20.5 Machine Learning Connections: Why Wiener Noise Is the Correct Forward Corruption

Diffusion models require a forward corruption family \((p_t)_{t\in[0,T]}\) satisfying three conditions simultaneously:
1. \(p_0 = p_{\text{data}}\),
2. \(p_T\) is approximately \(\mathcal{N}(0,I)\),
3. \(p_t\) is tractable to sample at arbitrary \(t\) given a data sample \(x_0\).

Brownian-based constructions satisfy all three by design.

### 20.5.1 Brownian Motion as a Gaussian Channel Indexed by Time

For pure diffusion \(X_t = X_0 + \sqrt{2D}\,W_t\), the channel is:
\[
p(x_t\mid x_0,t) = \mathcal{N}(x_0,\;2Dt\,I).
\]
This is the continuous-time analog of the discrete Gaussian perturbation used in denoising training: data plus Gaussian noise, with the time parameter being exactly the noise scale. Condition (3) is satisfied because sampling \(x_t\mid x_0\) at any \(t\) requires only a single Gaussian draw.

### 20.5.2 Scores under Gaussian Smoothing and the Denoising Bridge

<div class="prop-box">

**Proposition 20.3 (conditional score for Gaussian corruption).** For pure Brownian diffusion,
\[
\nabla_{x_t}\log p(x_t\mid x_0,t) = -\frac{x_t - x_0}{2Dt}.
\]

</div>

<div class="proof-box">

**Proof.** Since \(p(x_t\mid x_0,t) = \mathcal{N}(x_0,2DtI)\), and the score of \(\mathcal{N}(m,\sigma^2 I)\) with respect to \(x\) is \(-\frac{x-m}{\sigma^2}\), the result follows with \(m=x_0\) and \(\sigma^2=2Dt\). \(\square\)

</div>

The marginal score is related to the conditional score by the DSM identity (Chapter 19, 搂19.8.1):
\[
\nabla\log p_t(x) = \mathbb{E}\!\big[\nabla_x\log p(X_t\mid X_0,t)\;\big|\;X_t=x\big] = -\frac{1}{2Dt}\,\mathbb{E}[X_t - X_0\mid X_t=x].
\]

<div class="ml-box">

**The denoising interpretation.** The marginal score at \((x,t)\) equals \(-\frac{1}{2Dt}\) times the expected residual \(\mathbb{E}[X_t-X_0\mid X_t=x]\). But \(\mathbb{E}[X_0\mid X_t=x]\) is the minimum mean-squared error (MMSE) denoiser: the best prediction of the clean data given the noisy observation. Therefore:
\[
\nabla\log p_t(x) = -\frac{x - \mathbb{E}[X_0\mid X_t=x]}{2Dt}.
\]
Learning the score is equivalent to learning the MMSE denoiser. This is why score networks in practice often output a "noise prediction" or "clean image prediction" rather than the score directly 鈥?all three are equivalent up to known rescaling by \(2Dt\) or \(\sigma(t)\).

</div>

### 20.5.3 Beyond Generative Modeling: Langevin Sampling

Brownian motion also appears in Langevin-type samplers for Bayesian inference and energy-based modeling:
\[
dX_t = \nabla\log\pi(X_t)\,dt + \sqrt{2}\,dW_t,
\]
whose stationary distribution is \(\pi\) under suitable conditions (Poincar茅 inequality, log-Sobolev inequality). Diffusion models generalize this to a time-inhomogeneous family: the reverse drift involves \(\nabla\log p_t\) rather than a single stationary \(\nabla\log\pi\). The forward noising creates the trajectory of targets \((p_t)\); the reverse SDE samples from this trajectory in reverse.

---

## 20.6 Summary of the Primitives

<div class="scholium-box">

**What Brownian motion is, precisely:**

1. A centered Gaussian process with covariance \(\mathbb{E}[W_s W_t] = \min(s,t)\).
2. Equivalently: adapted, continuous paths, independent Gaussian increments with \(\mathrm{Var}(W_t - W_s) = t-s\).
3. The universal scaling limit of symmetric random walks with finite variance (Donsker's principle).

**Its key analytic property:**

4. Quadratic variation \([W]_t = t\) 鈥?not zero, not random. This is why classical calculus fails and It么's formula needs the correction term \(\frac{1}{2}g^2\partial_{xx}\varphi\,dt\).

**Its role in diffusion models:**

5. Pure diffusion \(dX_t = \sqrt{2D}\,dW_t\) implements Gaussian smoothing at the density level: \(p_t = p_0 * \varphi_{2Dt}\), satisfying the heat equation \(\partial_t p_t = D\,\Delta p_t\).
6. The conditional score of the Gaussian corruption kernel is explicit: \(\nabla_{x_t}\log p(x_t\mid x_0,t) = -(x_t-x_0)/(2Dt)\).
7. Learning the score is equivalent to learning the MMSE denoiser; the three formulations (score, noise prediction, clean-image prediction) differ only by a known rescaling.

**The conceptual pivot for Chapter 21:**

The VP SDE replaces \(\sqrt{2D}\,dW_t\) by \(-\frac{1}{2}\beta(t)X_t\,dt + \sqrt{\beta(t)}\,dW_t\), adding mean reversion to the noise injection. The analysis of this Ornstein鈥揢hlenbeck dynamics 鈥?and of how the reverse-time SDE reconstructs data from the \(\mathcal{N}(0,I)\) prior 鈥?is the next step.

</div>

**Chapter 021: The Ornstein鈥揢hlenbeck Process, the VP SDE, and Closed-Form Marginals 鈥?From Mean Reversion to the DDPM Corruption Kernel.**
