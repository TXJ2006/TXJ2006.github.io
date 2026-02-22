---
title: "Chapter 19: Stochastic Differential Equations 鈥?The Calculus Under Diffusion Models"
layout: "single"
url: "/book/chapters/chapter019/"
summary: "It么 calculus from quadratic variation; It么's formula and the chain-rule correction; Fokker鈥揚lanck for density evolution; forward VP/VE SDEs; reverse-time SDE and the score; denoising score matching from first principles; probability flow ODE; drifting models as training-time distribution evolution."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 19
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

## Chapter 19 &mdash; Stochastic Differential Equations: The Calculus Under Diffusion Models

*It么 calculus, forward noising, reverse-time dynamics, and what "drift" really means in modern generative modeling*

*Xujiang Tang*

</div>

## Abstract

A diffusion model is, at its core, a statement about how a probability distribution evolves in time under a stochastic dynamics. The stochastic dynamics is an SDE. The "model" is typically not the forward SDE (noise injection is chosen, not learned), but the inverse mechanism that reconstructs data from noise. The aim here is to put the foundational objects on the table and then connect them to diffusion models at the level where the mathematics is doing the work: filtration, Brownian motion, It么 integral, quadratic variation, It么's formula, Fokker鈥揚lanck, and time reversal.

---

## 19.0 Scope and Notational Ground

Throughout, let \((\Omega,\mathcal{F},\mathbb{P})\) be a probability space and \((\mathcal{F}_t)_{t\ge 0}\) a filtration satisfying the usual conditions (right-continuity and completeness). Random processes are assumed adapted to \((\mathcal{F}_t)\) unless stated otherwise.

---

## 19.1 Brownian Motion as the Primitive (and Why Classical Calculus Breaks)

### 19.1.1 Definition: Wiener Process

<div class="def-box">

**Definition 19.1 (standard Brownian motion).** A one-dimensional standard Brownian motion \((W_t)_{t\ge 0}\) is a stochastic process satisfying:

1. \(W_0 = 0\) almost surely,
2. for \(0\le s < t\), the increment \(W_t - W_s \sim \mathcal{N}(0,\,t-s)\),
3. increments over disjoint intervals are independent,
4. \(t\mapsto W_t\) has continuous sample paths almost surely.

</div>

The key analytic feature is not Gaussianity by itself; it is path regularity:

- Brownian paths are almost surely **nowhere differentiable**,
- they have **infinite total variation** on any interval,
- but they have a **finite and deterministic quadratic variation**.

### 19.1.2 Quadratic Variation

Fix a partition \(\pi = \{0 = t_0 < t_1 < \cdots < t_n = t\}\). Consider
\[
Q_\pi(W;\,t) := \sum_{k=0}^{n-1}(W_{t_{k+1}} - W_{t_k})^2.
\]
For Brownian motion, as the mesh \(|\pi|\to 0\),
\[
Q_\pi(W;\,t)\ \longrightarrow\ t \qquad \text{in probability (and a.s. along refining partitions).}
\]
This limit is written as \([W]_t = t\). In differential notation:
\[
(dW_t)^2 = dt.
\]
In ordinary smooth calculus, \((dt)^2\) vanishes at first order; here the square of the noise increment contributes at first order. That single fact forces a new calculus.

---

## 19.2 The It么 Integral: Definition Before Intuition

An SDE uses integrals of the form \(\int_0^t H_s\,dW_s\). This is not a Riemann鈥揝tieltjes integral; Brownian motion has unbounded variation. The definition is a limit in \(L^2\), built from simple adapted processes.

### 19.2.1 Simple Adapted Processes

A process \(H\) is *simple and adapted* if
\[
H_s(\omega) = \sum_{k=0}^{n-1} H_k(\omega)\,\mathbf{1}_{(t_k,\,t_{k+1}]}(s),
\]
where each \(H_k\) is \(\mathcal{F}_{t_k}\)-measurable. For such \(H\), define
\[
\int_0^t H_s\,dW_s := \sum_{k=0}^{n-1} H_k\,(W_{t_{k+1}} - W_{t_k}).
\]
The measurability condition (use the left-endpoint information) is the adaptation constraint that matches causality in stochastic dynamics.

### 19.2.2 Extension by \(L^2\) Completion and It么 Isometry

If \(H\) is progressively measurable and square-integrable, meaning
\[
\mathbb{E}\!\left[\int_0^t H_s^2\,ds\right] < \infty,
\]
then there exists a sequence of simple adapted processes \(H^{(m)}\) approximating \(H\) in the sense that
\[
\mathbb{E}\!\left[\int_0^t (H_s^{(m)} - H_s)^2\,ds\right] \to 0.
\]
The It么 integral is defined by the \(L^2\) limit:
\[
\int_0^t H_s\,dW_s := \lim_{m\to\infty}\int_0^t H_s^{(m)}\,dW_s.
\]
This limit is independent of the approximating sequence. The essential identity is the **It么 isometry**:

<div class="prop-box">

**Proposition 19.1 (It么 isometry).** For any progressively measurable square-integrable \(H\),
\[
\mathbb{E}\!\left[\left(\int_0^t H_s\,dW_s\right)^2\right] = \mathbb{E}\!\left[\int_0^t H_s^2\,ds\right].
\]

</div>

This is the rigorous form of \((dW)^2 = dt\): the second moment of the stochastic integral equals the \(L^2\) energy of the integrand.

---

## 19.3 It么 Processes and SDEs: Definition, Existence, and Meaning of Coefficients

### 19.3.1 It么 Process

<div class="def-box">

**Definition 19.2 (It么 process).** A process \(X_t\) is an *It么 process* if it can be written as
\[
X_t = X_0 + \int_0^t f(s)\,ds + \int_0^t g(s)\,dW_s,
\]
for adapted processes \(f, g\) satisfying suitable integrability conditions.

</div>

In differential form: \(dX_t = f(t)\,dt + g(t)\,dW_t\). In multiple dimensions, with \(W_t\in\mathbb{R}^m\) and \(X_t\in\mathbb{R}^d\),
\[
dX_t = f(X_t,t)\,dt + G(X_t,t)\,dW_t,
\]
where \(G\in\mathbb{R}^{d\times m}\). The *diffusion matrix* is \(a(x,t) = G(x,t)G(x,t)^\top\).

### 19.3.2 SDE as a Fixed-Point Statement

An SDE
\[
dX_t = f(X_t,t)\,dt + G(X_t,t)\,dW_t, \qquad X_0\sim p_0
\]
is shorthand for the integral equation
\[
X_t = X_0 + \int_0^t f(X_s,s)\,ds + \int_0^t G(X_s,s)\,dW_s.
\]
Existence and uniqueness of a strong solution is guaranteed under standard global Lipschitz and linear growth conditions on the coefficients.

### 19.3.3 What "Drift" and "Diffusion" Mean Geometrically

- The **drift** \(f\) is the deterministic instantaneous velocity field: it transports mass in state space.
- The **diffusion coefficient** \(G\) injects local randomness; its covariance \(a = GG^\top\) determines how mass spreads.

In diffusion models, this separation becomes structural: the forward process is designed by choosing \(f, G\), and the reverse process introduces a learned correction depending on the score of intermediate marginals.

---

## 19.4 It么's Formula: The Calculus Rule That Replaces the Chain Rule

Let \(X_t\) satisfy \(dX_t = f(X_t,t)\,dt + g(X_t,t)\,dW_t\) in one dimension, and let \(\varphi(x,t)\) be \(C^{2,1}\). The classical chain rule fails because \((dX_t)^2\) contains a first-order piece.

### 19.4.1 Derivation from Taylor Expansion and Quadratic Variation

Write a second-order Taylor expansion:
\[
\varphi(X_{t+dt},\,t+dt) - \varphi(X_t,t) \approx \partial_t\varphi\,dt + \partial_x\varphi\,dX_t + \frac{1}{2}\partial_{xx}\varphi\,(dX_t)^2.
\]
Insert \(dX_t = f\,dt + g\,dW_t\). Applying the bookkeeping rules justified by quadratic variation:
\[
(dt)^2 = 0,\qquad dt\,dW_t = 0,\qquad (dW_t)^2 = dt,
\]
we get \((dX_t)^2 = g(X_t,t)^2\,dt\). Therefore:

<div class="prop-box">

**Theorem 19.1 (It么's formula, one dimension).** For \(\varphi\in C^{2,1}\),
\[
d\varphi(X_t,t) = \Big(\partial_t\varphi + f\,\partial_x\varphi + \frac{1}{2}g^2\,\partial_{xx}\varphi\Big)\,dt + g\,\partial_x\varphi\,dW_t.
\]

</div>

<div class="prop-box">

**Theorem 19.2 (It么's formula, \(d\) dimensions).** With diffusion matrix \(a = GG^\top\),
\[
d\varphi(X_t,t) = \Big(\partial_t\varphi + \nabla\varphi^\top f + \frac{1}{2}\mathrm{tr}(a\,\nabla^2\varphi)\Big)\,dt + \nabla\varphi^\top G\,dW_t.
\]

</div>

The extra term \(\frac{1}{2}g^2\,\partial_{xx}\varphi\) (or \(\frac{1}{2}\mathrm{tr}(a\,\nabla^2\varphi)\) in \(d\) dimensions) is the It么 correction 鈥?the second-order contribution forced by the non-vanishing quadratic variation of Brownian motion.

---

## 19.5 Generator and Fokker鈥揚lanck: From Sample Paths to Density Evolution

Diffusion models are distributional machines. The bridge from SDE pathwise dynamics to density evolution is the Kolmogorov forward equation (Fokker鈥揚lanck).

### 19.5.1 Infinitesimal Generator

<div class="def-box">

**Definition 19.3 (generator).** The *infinitesimal generator* \(\mathcal{L}\) acts on smooth test functions \(\varphi\) by
\[
\mathcal{L}\varphi(x,t) := f(x,t)\cdot\nabla\varphi(x,t) + \frac{1}{2}\mathrm{tr}\!\big(a(x,t)\,\nabla^2\varphi(x,t)\big).
\]

</div>

It么's formula implies that
\[
\varphi(X_t,t) - \varphi(X_0,0) - \int_0^t \big(\partial_s\varphi(X_s,s) + \mathcal{L}\varphi(X_s,s)\big)\,ds
\]
is a martingale (the stochastic integral term). Taking expectations kills the martingale term:
\[
\mathbb{E}[\varphi(X_t,t)] = \mathbb{E}[\varphi(X_0,0)] + \int_0^t \mathbb{E}\big[\partial_s\varphi(X_s,s) + \mathcal{L}\varphi(X_s,s)\big]\,ds.
\]

### 19.5.2 Deriving the Fokker鈥揚lanck Equation

Let \(p_t(x)\) be the density of \(X_t\). Differentiating \(\mathbb{E}[\varphi(X_t,t)] = \int\varphi(x,t)\,p_t(x)\,dx\) in \(t\) and using the generator identity, then canceling the \(\partial_t\varphi\) term, integrating by parts to move \(\mathcal{L}\) onto \(p_t\) via the adjoint \(\mathcal{L}^\ast\):

<div class="prop-box">

**Theorem 19.3 (Fokker鈥揚lanck / Kolmogorov forward equation).**
\[
\partial_t p_t(x) = \mathcal{L}^\ast p_t(x) = -\nabla\cdot\!\big(f(x,t)\,p_t(x)\big) + \frac{1}{2}\sum_{i,j}\partial_{x_i}\partial_{x_j}\!\big(a_{ij}(x,t)\,p_t(x)\big).
\]

</div>

This PDE is the distribution-level statement of the SDE: where the SDE describes individual paths, Fokker鈥揚lanck describes the evolution of the entire ensemble density.

---

## 19.6 The Diffusion-Model Forward Process as an SDE

Diffusion models choose a forward noising process that maps the data distribution \(p_{\text{data}}\) to a tractable prior (typically Gaussian).

### 19.6.1 General Forward SDE

A common setup for score-based generative modeling is:
\[
dX_t = f(X_t,t)\,dt + g(t)\,dW_t, \qquad X_0\sim p_{\text{data}},
\]
with scalar diffusion \(g(t)\) and drift \(f\) chosen so that \(p_T \approx \mathcal{N}(0,I)\) for large \(T\). This framing unifies discrete DDPMs and earlier score-based constructions.

### 19.6.2 Variance Preserving (VP) SDE: Continuous Analog of DDPM

The VP SDE is an Ornstein鈥揢hlenbeck type process:
\[
dX_t = -\frac{1}{2}\beta(t)X_t\,dt + \sqrt{\beta(t)}\,dW_t,
\]
with noise schedule \(\beta(t)>0\). Define
\[
\alpha(t) := \exp\!\left(-\frac{1}{2}\int_0^t \beta(s)\,ds\right), \qquad \sigma^2(t) := 1 - \alpha(t)^2.
\]
The conditional marginal has the closed form:
\[
X_t = \alpha(t)X_0 + \sigma(t)\,\varepsilon,\qquad \varepsilon\sim\mathcal{N}(0,I),
\]
so that \(p(X_t\mid X_0) = \mathcal{N}(\alpha(t)X_0,\;\sigma^2(t)I)\). This closed form is the computational reason diffusion training is feasible: one can sample \(X_t\) in a single step for any \(t\), without simulating the full path.

<div class="ml-box">

**Why the VP marginal is tractable.** The VP SDE is linear in \(X_t\), so its solution is an affine function of the Gaussian noise \(\varepsilon\). Linearity + Gaussian noise = Gaussian marginal, with explicitly computable mean and variance. This is what makes the denoising score matching objective (搂19.8) computable without approximation.

</div>

### 19.6.3 Variance Exploding (VE) SDE

A pure-noise inflation process:
\[
dX_t = g(t)\,dW_t,
\]
with \(g(t)\) increasing so that the variance "explodes." The VE SDE corresponds to SMLD-style noise perturbations in the score-SDE framework.

---

## 19.7 Reverse-Time SDE: Where the Score Appears

Forward noising is easy. Sampling requires reversing it.

### 19.7.1 The Score as the Missing Term

Let the forward SDE be \(dX_t = f(X_t,t)\,dt + g(t)\,dW_t\) and let \(p_t(x)\) denote the density of \(X_t\). Under mild regularity, the time-reversal of a diffusion is again a diffusion. Written in forward-in-time notation with a reverse-time Brownian motion \(\bar{W}_t\):

<div class="prop-box">

**Theorem 19.4 (reverse-time SDE).** The time-reversal of the forward SDE, running from \(T\) down to \(0\), satisfies
\[
dX_t = \Big(f(X_t,t) - g(t)^2\,\nabla_x\log p_t(X_t)\Big)\,dt + g(t)\,d\bar{W}_t, \qquad t:\;T\to 0.
\]

</div>

The only unknown object is \(\nabla_x\log p_t(x)\), the *score* of the marginal distribution at time \(t\). Score-based generative modeling takes this as the modeling target: train a neural network \(s_\theta(x,t)\approx \nabla_x\log p_t(x)\), then plug it into the reverse SDE and simulate to generate samples.

### 19.7.2 Why the Score Is the Correct Object

From the forward SDE, \(p_t\) satisfies Fokker鈥揚lanck. Time reversal is a statement about constructing a dynamics whose marginals evolve as \(p_{T-t}\). The score term appears because reversing diffusion requires "undoing" the entropy increase induced by the Laplacian spreading term.

The score \(\nabla_x\log p_t(x) = -\nabla_x U_t(x)\), where \(U_t(x) = -\log p_t(x)\) is the instantaneous energy. The reverse drift contracts mass in the direction of increasing density, exactly countering forward diffusion's dispersal. This is the same mechanism as Langevin dynamics, generalized to a time-inhomogeneous family of targets \(p_t\).

---

## 19.8 Training Objective: Denoising Score Matching from First Principles

The score \(\nabla\log p_t(x)\) is not directly available, but for VP/VE setups the conditional \(p(x_t\mid x_0)\) is closed-form, yielding a learnable surrogate.

### 19.8.1 The Identity Behind Denoising Score Matching

For fixed \(t\), the perturbed marginal is
\[
p_t(x) = \int p(x\mid x_0,t)\,p_{\text{data}}(x_0)\,dx_0.
\]
Differentiating:
\[
\nabla_x\log p_t(x) = \frac{\int \nabla_x p(x\mid x_0,t)\,p_{\text{data}}(x_0)\,dx_0}{p_t(x)}.
\]
Writing \(\nabla_x p(x\mid x_0,t) = p(x\mid x_0,t)\,\nabla_x\log p(x\mid x_0,t)\) and recognizing the posterior weight:
\[
\nabla_x\log p_t(x) = \mathbb{E}\!\Big[\nabla_x\log p(X_t\mid X_0,t)\ \Big|\ X_t = x\Big].
\]
The marginal score is the conditional expectation of the *conditional score*. This identity converts "unknown marginal" into "known Gaussian corruption + data samples."

### 19.8.2 Specialization to VP Gaussian Corruption

For VP corruption \(X_t = \alpha(t)X_0 + \sigma(t)\varepsilon\), the conditional is \(p(x_t\mid x_0,t) = \mathcal{N}(\alpha(t)x_0,\,\sigma^2(t)I)\). For a Gaussian \(\mathcal{N}(m,\sigma^2 I)\), the score with respect to \(x\) is \(-\frac{x-m}{\sigma^2}\). Therefore:
\[
\nabla_{x_t}\log p(x_t\mid x_0,t) = -\frac{x_t - \alpha(t)x_0}{\sigma^2(t)}.
\]

### 19.8.3 The Denoising Score Matching Objective

Let \(s_\theta(x,t)\) be a neural field. The ideal loss is
\[
\mathbb{E}_{t}\;\mathbb{E}_{x_t\sim p_t}\!\left[\|s_\theta(x_t,t) - \nabla\log p_t(x_t)\|^2\right].
\]
Substituting via the DSM identity: sample \(x_0\sim p_{\text{data}}\), \(t\), \(\varepsilon\sim\mathcal{N}(0,I)\), form \(x_t = \alpha(t)x_0+\sigma(t)\varepsilon\), and minimize

<p>
\[
\mathbb{E}\!\left[\left\|s_\theta(x_t,t) + \frac{x_t - \alpha(t)x_0}{\sigma^2(t)}\right\|^2\right],
\]
</p>

up to time-dependent weighting. This is the denoising score matching objective underlying modern diffusion training, and it reveals the DDPM noise-prediction objective as equivalent score estimation (up to rescaling by \(\sigma(t)\)).

<div class="ml-box">

**What is actually being trained.** The network \(s_\theta(x_t,t)\) learns to predict, given a noisy observation \(x_t\), which direction to move to increase the log-density of the current marginal. During sampling, this direction guides the reverse drift that reconstructs data from noise. The training set never contains the score labels directly; they are implicitly encoded in the noise realization \(\varepsilon\).

</div>

---

## 19.9 Sampling: Reverse SDE and Probability Flow ODE

With an estimated score field \(s_\theta(x,t)\), sampling proceeds by integrating the reverse-time dynamics.

### 19.9.1 Stochastic Sampling (Reverse SDE)

Plug \(s_\theta\) into the reverse-time drift:
\[
dX_t = \Big(f(X_t,t) - g(t)^2\,s_\theta(X_t,t)\Big)\,dt + g(t)\,d\bar{W}_t, \qquad T\to 0.
\]
Discretizations 鈥?Euler鈥揗aruyama, stochastic Runge鈥揔utta, predictor鈥揷orrector 鈥?produce practical samplers with different accuracy and compute trade-offs.

### 19.9.2 Deterministic Sampling (Probability Flow ODE)

A key observation is that there exists a deterministic ODE whose solution trajectories share the same time-marginals \(p_t\) as the stochastic reverse-time SDE:

<div class="prop-box">

**Theorem 19.5 (probability flow ODE).** The ODE
\[
\frac{dX_t}{dt} = f(X_t,t) - \frac{1}{2}g(t)^2\,s_\theta(X_t,t)
\]
has marginals identical to those of the reverse-time SDE at every \(t\).

</div>

This provides a deterministic sampling path and connects diffusion models to continuous normalizing flows via change-of-variables along ODE trajectories.

<div class="scholium-box">

**Structural comparison:**

| | Reverse SDE | Probability Flow ODE |
|---|---|---|
| **Trajectory** | Stochastic (noise at every step) | Deterministic (transport) |
| **Marginals** | \(p_t\) for all \(t\) | Same \(p_t\) for all \(t\) |
| **Governing object** | \(s_\theta(x,t)\) | Same \(s_\theta(x,t)\) |
| **Inference cost** | Many small stochastic steps | ODE solver (often fewer steps) |
| **Connection** | 鈥?| Continuous normalizing flow |

Both are governed by the same learned object \(s_\theta\). That is the central unification of the score-SDE framework.

</div>

---

## 19.10 A Distribution-First Perspective: Diffusion as "Learn the Correction to Entropy Production"

Forward diffusion adds noise, increasing entropy. The reverse process must decrease entropy while matching a specific evolving family of marginals.

In the reverse drift \(f(x,t) - g(t)^2\nabla\log p_t(x)\), the score term is exactly the contraction mechanism: \(-\nabla\log p_t\) points in the direction that increases density. This makes diffusion models naturally interpretable as a time-indexed family of instantaneous energy landscapes
\[
U_t(x) := -\log p_t(x),
\]
with score \(\nabla\log p_t(x) = -\nabla U_t(x)\). Sampling becomes a controlled evolution on these landscapes where the control is learned.

This perspective remains stable when the surface form changes 鈥?DDPM sampling, predictor鈥揷orrector, ODE samplers 鈥?because it is attached to the reverse drift structure, not to discretization details.

---

## 19.11 Beyond Diffusion: "Drift" as a General Design Variable

Diffusion and flow models both generate by iterating a pushforward map at inference time: start from a known prior and apply many small updates (stochastic or deterministic) so that the distribution moves toward data.

### 19.11.1 The Shared Core: Evolving a Pushforward Distribution

Many generative methods can be phrased as learning a map \(f_\theta\) such that the pushforward \((f_\theta)_{\#}\pi\) of a simple base \(\pi\) matches the data distribution \(p\). Diffusion and flow-based models realize this by composing many small transformations at inference time.

### 19.11.2 Drifting Models: Training-Time Distribution Evolution

A recent alternative (Deng, He et al.) proposes a different allocation of iteration: evolve the distribution during *training* so that inference becomes a single step. The core idea is a "drifting field" \(V\) that:
- moves samples in the direction that makes the generated distribution closer to data,
- vanishes at equilibrium (when generated distribution matches data exactly),
- is estimated from mini-batches and used to construct a "drifted target" for the network to regress to.

After training, inference is a single forward pass.

### 19.11.3 Situating Drifting Models in the SDE Framework

<div class="ml-box">

**Design axis comparison:**

SDE diffusion performs distribution evolution at inference time 鈥?the distribution \(p_t\) evolves along a trajectory of physical time \(t\in[0,T]\) driven by a known forward SDE and a learned reverse correction. Integrating this trajectory costs \(N\) network evaluations.

Drifting Models perform distribution evolution during *training* 鈥?the optimizer's updates, guided by the drifting field \(V\), move the generated distribution toward data over the course of training steps. After training, the distribution evolution has already happened; inference collapses to one evaluation.

From the SDE perspective: both are manipulating drift fields. The difference is which dynamical system serves as the "time axis":

- Diffusion: physical time of an SDE, requiring numerical integration at inference.
- Drifting: training-time evolution, requiring no inference-time integration.

</div>

This connects to a broader design space: generative modeling as choosing (i) a path of intermediate distributions and (ii) a mechanism that realizes the path 鈥?stochastic dynamics, deterministic transport, or training-driven drift.

---

## 19.12 Scholium: What This Chapter Forces You to Admit

<div class="scholium-box">

1. **Classical calculus fails on Brownian paths.** The quadratic variation identity \((dW)^2 = dt\) is not a convenience notation; it is the precise mathematical statement that forces the second-order It么 correction term. Every formula in diffusion modeling that looks like a chain rule is actually It么's formula.

2. **The score is the only unknown.** The entire reverse-time dynamics 鈥?from the theoretical time-reversal theorem to the practical training objective 鈥?depends on one object: \(\nabla_x\log p_t(x)\). Score estimation is not a choice of method; it is the canonical form of the problem.

3. **Training computes the score implicitly.** The denoising score matching objective never directly labels the score. It exploits the identity that the marginal score equals the conditional expectation of the conditional score, and the conditional score for Gaussian corruption is available in closed form. The entire training loop is a Monte Carlo approximation of this expectation.

4. **The SDE and ODE samplers are dual representations of the same marginal flow.** The stochastic reverse-time SDE and the probability flow ODE produce the same time-marginals \(p_t\). The choice between them is about trajectories and computational trade-offs, not about which distribution is targeted.

5. **"Drift" is a design variable, not a fixed concept.** Different generative models 鈥?diffusion, flows, drifting models 鈥?can be understood as different choices of drift field and different "time axes" (physical time vs. training time) along which distribution evolution unfolds. The SDE framework provides the language to compare them at the structural level.

</div>

**Chapter 020: Rademacher Complexity and Generalization Theory 鈥?From Finite Hypothesis Classes to Deep Networks.**


*Next: [Chapter 20: Wiener Processes and Brownian Motion](/book/chapters/chapter020/)*

*Next: [Chapter 20: Wiener Processes and Brownian Motion](/book/chapters/chapter020/)*
