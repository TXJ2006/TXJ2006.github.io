---
title: "Chapter 15: Markov Chains & Ergodicity"
layout: "single"
url: "/book/chapters/chapter015/"
summary: "Markov property as orthogonal truncation of history; Chapman窶適olmogorov as semigroup identity; ergodic theorem as the license for replacing ensemble expectations by time averages; SGD as an ergodic Markov process on parameter space."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 15
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

## Chapter 15 &mdash; Markov Chains &amp; Ergodicity: Topological Truncation of the Time Manifold and Ensemble Evolution in Machine Learning

*Xujiang Tang*

</div>

## Abstract

This chapter formalizes a single idea: once "time" enters the mathematical phenomenology of machine intelligence, the relevant object is no longer a static geometry but a *filtration* of information and a *semigroup* of operators. Markovity is not a slogan ("depends only on the present"), but a precise truncation of the information \(\sigma\)-algebra; ergodicity is not a folklore miracle, but the theorem that licenses replacing an inaccessible ensemble expectation by a time-average along one trajectory.

---

## 15.1 The Markov Assumption as Orthogonal Stripping of History

> **Theorem-sentence 15.1.** The Markov assumption is a physically motivated orthogonal truncation of the infinite history information horizon \(\sigma(X_0,\dots,X_n)\) to the present horizon \(\sigma(X_n)\); it requires the "state" to be a sufficient holographic compression of the past for predicting the future.

### 15.1.1 Definitions under Kolmogorov's axioms

Let \((\Omega,\mathcal{F},\mathbb{P})\) be a probability space and \((\mathcal{S},\mathcal{B}(\mathcal{S}))\) a measurable state space. A discrete-time stochastic process \(\{X_n\}_{n\ge 0}\) is a family of measurable maps \(X_n:\Omega\to\mathcal{S}\).

The *natural filtration* (history information flow) is
\[
\mathcal{F}_n := \sigma(X_0,X_1,\dots,X_n).
\]
For any measurable set \(A\in\mathcal{B}(\mathcal{S})\), prediction of the next event \(\{X_{n+1}\in A\}\) conditioned on the full history is
\[
\mathbb{P}(X_{n+1}\in A\mid \mathcal{F}_n) = \mathbb{E}\!\left[\mathbf{1}_{\{X_{n+1}\in A\}}\mid \sigma(X_0,\dots,X_n)\right].
\]

**Hilbert-space reading (connecting to Chapter 14).**
Working in \(L^2(\Omega,\mathcal{F},\mathbb{P})\), the operator \(\mathbb{E}[\cdot\mid\mathcal{F}_n]\) is an orthogonal projection onto the closed subspace \(L^2(\mathcal{F}_n)\). Conditioning on history is a projection onto an ever-expanding subspace as \(n\) grows.

<div class="ml-box">

**The computational curse.** As \(n\to\infty\), \(\mathcal{F}_n\) becomes overwhelmingly rich: exact projections onto \(L^2(\mathcal{F}_n)\) require representing arbitrarily long dependencies. This is the measure-theoretic core of why long-horizon sequence prediction is intrinsically hard, and why naﾃｯve RNNs face long-term dependency failure: the information horizon grows faster than feasible representation.

</div>

### 15.1.2 The Markov truncation: formal statement and geometric meaning

<div class="def-box">

**Definition 15.1 (Markov property, measure-theoretic form).** The process \(\{X_n\}\) is *Markov* if for every \(n\ge 0\) and every measurable \(A\in\mathcal{B}(\mathcal{S})\),
\[
\mathbb{P}(X_{n+1}\in A\mid \mathcal{F}_n) = \mathbb{P}(X_{n+1}\in A\mid \sigma(X_n)).
\]
Equivalently, for every bounded measurable \(f:\mathcal{S}\to\mathbb{R}\),
\[
\mathbb{E}[f(X_{n+1})\mid \mathcal{F}_n] = \mathbb{E}[f(X_{n+1})\mid X_n] \quad \text{a.s.}
\]

</div>

**What this equality actually says.** The left-hand side is the orthogonal projection of \(f(X_{n+1})\) onto \(L^2(\mathcal{F}_n)\); the right-hand side is the projection onto the much smaller subspace \(L^2(\sigma(X_n))\). For the two projections to coincide for all \(f\), all "residual" information in \(\sigma(X_0,\dots,X_{n-1})\) must be irrelevant for predicting \(X_{n+1}\) once \(X_n\) is known.

<div class="prop-box">

**Proposition 15.1 (orthogonality condition implied by Markovity).** Assume \(f(X_{n+1})\in L^2\). If the chain is Markov, then for any \(Z\in L^2(\mathcal{F}_n)\) with \(\mathbb{E}[Z\mid X_n]=0\),
\[
\mathbb{E}\!\left[\big(f(X_{n+1})-\mathbb{E}[f(X_{n+1})\mid X_n]\big)\,Z\right] = 0.
\]
The unpredictable component of \(f(X_{n+1})\) given \(X_n\) is orthogonal to all history features carrying no information beyond \(X_n\).

</div>

### 15.1.3 Machine learning: state representation as an imposed Markov compression

Markovity is rarely given; in ML it is *manufactured* by learning a representation.

> **Core theorem 15.1 (ML corollary).** A learned "state" \(h_n\) is useful precisely to the extent that it makes the process approximately Markov: $p(X_{n+1}\mid X_{0:n}) \approx p(X_{n+1}\mid h_n)$.

Concrete instances:

**1. RNN / GRU / LSTM as approximate Markovization.**
An RNN defines \(h_{n+1}=F_\theta(h_n, x_{n+1})\) and predicts via \(p_\theta(y_{n+1}\mid h_n)\). Training by negative log-likelihood encourages \(h_n\) to retain predictive information. LSTM gating is not a "trick"; it is a mechanism to prevent the learned compression from discarding long-range information that would break approximate Markovity.

**2. Hidden Markov Models and latent state-space models.**
In an HMM, latent \(Z_n\) is Markov and observations \(X_n\) are conditionally independent given \(Z_n\). The *belief state* \(b_n(\cdot)=\mathbb{P}(Z_n\in\cdot\mid X_{0:n})\) is a canonical sufficient statistic making the process Markov in belief space. This is the exact measure-theoretic meaning of "state = all information relevant for the future."

**3. POMDPs and representation learning in RL.**
In partially observed RL, the environment is Markov in the latent state \(S_n\), but the agent sees \(O_n\). A learned encoder \(h_n=\phi_\theta(O_{0:n})\) is successful only if it reconstructs a Markov (or belief) state for control. Many RL failures are precisely failures of learned Markovization.

---

## 15.2 Transition Operators and the Chapman窶適olmogorov Equation: Sequential Models as Time Semigroups

> **Theorem-sentence 15.2.** The Chapman窶適olmogorov equation is the semigroup homomorphism of probability evolution in time; it is the geometric skeleton underlying sequential models, from Markov chains to RNNs and residual flows.

### 15.2.1 The transition kernel and two dual operators

<div class="def-box">

**Definition 15.2 (Markov kernel).** A transition probability kernel \(P\) on \((\mathcal{S},\mathcal{B}(\mathcal{S}))\) is a map \(P:\mathcal{S}\times\mathcal{B}(\mathcal{S})\to[0,1]\) such that: (i) \(A\mapsto P(x,A)\) is a probability measure for each fixed \(x\); (ii) \(x\mapsto P(x,A)\) is measurable for each fixed \(A\). Markovity implies
\[
\mathbb{P}(X_{n+1}\in A\mid X_n=x) = P(x,A).
\]

</div>

There are two equivalent operator viewpoints:

**Koopman (function) operator.** For bounded measurable \(f\),
\[
(Pf)(x) := \int_{\mathcal{S}} f(y)\,P(x,dy).
\]

**Perron窶擢robenius (measure) operator.** For a probability measure \(\mu\) on \(\mathcal{S}\),
\[
(\mu P)(A) := \int_{\mathcal{S}} P(x,A)\,\mu(dx).
\]

These are dual: \(\int f\,d(\mu P) = \int (Pf)\,d\mu\).

### 15.2.2 Chapman窶適olmogorov as semigroup structure

Let \(P^{n}(x,A)=\mathbb{P}(X_{k+n}\in A\mid X_k=x)\). Then:

<div class="prop-box">

**Theorem 15.2 (Chapman窶適olmogorov).** For all \(m,n\in\mathbb{N}\),
\[
P^{m+n}(x,A) = \int_{\mathcal{S}} P^{m}(x,dz)\,P^{n}(z,A).
\]
Equivalently, \(P^{m+n}=P^m P^n\) as operators, i.e., \(\{P^n\}_{n\ge 0}\) is a semigroup.

</div>

<div class="proof-box">

**Proof (tower property).** By the Markov property and the tower property of conditional expectation (Chapter 14),
\[
\mathbb{P}(X_{k+m+n}\in A\mid X_k=x) = \mathbb{E}\!\left[\mathbb{P}(X_{k+m+n}\in A\mid X_{k+m})\mid X_k=x\right] = \int P^{n}(z,A)\,P^{m}(x,dz). \quad\square
\]

</div>

This is not merely algebra: time evolution composes by operator multiplication. The semigroup structure means that no "hidden memory" can leak across time steps beyond what the kernel encodes.

### 15.2.3 ML: sequential networks as parametric kernels; deep composition as semigroup iteration

**1. RNN as kernel approximation.**
A stochastic RNN with noise \(\epsilon_n\) defines \(h_{n+1}=F_\theta(h_n,x_{n+1},\epsilon_{n+1})\), inducing a transition kernel on \(h\)-space:
\[
P_\theta(h,A) = \mathbb{P}(F_\theta(h,x,\epsilon)\in A).
\]
Multi-step prediction is governed by \(P_\theta^n\), exactly the Chapman窶適olmogorov structure.

**2. Why gradients vanish and explode (operator-theoretic statement).**
Backprop through time multiplies Jacobians; in linearized dynamics this is repeated application of a linear operator. If the spectral radius is \(<1\), iteration collapses (vanishing gradients); if \(>1\), it blows up (exploding gradients). This is the Banach-space content of temporal instability, and it is precisely why residual connections and gating act as spectral anchoring.

**3. Residual networks as a time-discretized semigroup.**
A residual block \(h_{n+1}=h_n+f_\theta(h_n)\) is an Euler discretization of \(\dot{h}=f_\theta(h)\). The flow map \(T_t\) of an ODE forms a continuous-time semigroup \(\{T_t\}_{t\ge 0}\) with \(T_{t+s}=T_t\circ T_s\). ResNets are finite-step approximations to this semigroup; stability corresponds to controlling the spectrum of the linearized generator.

---

## 15.3 Ergodicity: When Time Averages Become Space Averages

> **Theorem-sentence 15.3.** Ergodicity is the topological foundation allowing an intelligent system to replace an intractable ensemble expectation under an unknown stationary measure by a time average along a single trajectory.

### 15.3.1 Stationarity as an eigenvector condition

<div class="def-box">

**Definition 15.3 (stationary distribution).** A probability measure \(\pi\) on \(\mathcal{S}\) is *stationary* for kernel \(P\) if
\[
\pi P = \pi, \qquad\text{i.e.}\qquad \pi(A)=\int_{\mathcal{S}}P(x,A)\,\pi(dx) \quad \forall A.
\]
In operator language, \(\pi\) is a left eigenmeasure of \(P\) with eigenvalue \(1\). For finite \(\mathcal{S}\), this is the left eigenvector condition \(\pi^\top P=\pi^\top\).

</div>

### 15.3.2 The ergodic theorem: the actual license used by MCMC and RL

<div class="def-box">

**Definition 15.4 (ergodicity).** A Markov chain is *ergodic* if it admits a unique stationary distribution \(\pi\) and
\[
\mu P^n \Rightarrow \pi \quad\text{as } n\to\infty
\]
for any initial distribution \(\mu\). In general spaces this corresponds to Harris ergodicity; in classical finite settings, irreducibility together with aperiodicity and positive recurrence suffices.

</div>

<div class="prop-box">

**Theorem 15.3 (Markov ergodic theorem, strong law form).** If the chain is ergodic with stationary distribution \(\pi\), then for any \(f\in L^1(\pi)\),
\[
\frac{1}{T}\sum_{t=1}^T f(X_t) \;\xrightarrow[T\to\infty]{\;a.s.\;}\; \int f(x)\,\pi(dx).
\]

</div>

This is the rigorous statement behind "one long run approximates the expectation." The theorem requires: (i) a unique stationary distribution, (ii) convergence from any start (ergodicity / irreducibility + aperiodicity), (iii) enough integrability of \(f\).

### 15.3.3 ML: three places where ergodicity is the hidden assumption

**1. MCMC for Bayesian learning.**
When sampling from a posterior \(\pi(\theta)\propto p(\theta)\prod_i p(x_i\mid\theta)\), we construct a Markov chain with stationary distribution \(\pi\). Expectations (posterior means, predictive likelihoods) are estimated by time averages. Without ergodicity and adequate mixing, MCMC is merely a trajectory generator with unknown bias, not a sampler.

**2. Policy evaluation in RL as ergodic averaging.**
Fix a policy \(\pi(a\mid s)\). This induces a Markov chain over states \(S_t\) with stationary distribution \(\rho^\pi\) (when it exists). Many RL estimators implicitly assume that empirical averages along episodes approximate expectations under \(\rho^\pi\). If the chain is not ergodic (e.g., multiple recurrent classes), learning from experience becomes fundamentally ambiguous: the time average depends on the initial state and which recurrent class is entered.

**3. Contrastive learning with negative sampling.**
The negative pool in contrastive objectives is often generated by a moving queue or replay buffer. Stability depends on the buffer approximating a stationary distribution of representations. If the representation distribution drifts too fast, the effective sampling measure is non-stationary 窶?a time-inhomogeneous Markov phenomenon, not merely "training instability."

---

## 15.4 SGD as an Ergodic Markov Process on Parameter Space

> **Theorem-sentence 15.4.** In high dimensions, SGD is better modeled as an ergodic Markov process whose long-run behavior is a stationary measure over parameters; generalization is then governed by where this measure concentrates, not by a single point estimate.

### 15.4.1 Why the "SGD finds a minimizer" story is incomplete

SGD iterates
\[
\theta_{k+1} = \theta_k - \eta\,\nabla\widehat{L}(\theta_k;\mathcal{B}_k),
\]
where \(\mathcal{B}_k\) is a random minibatch. Even at a global minimizer of the population loss, minibatch gradients fluctuate. Thus \(\theta_k\) typically does not converge to a fixed point in the deterministic sense; it continues to move. The right object is a *stationary distribution* over \(\theta\), not a single \(\theta^{\ast}\).

### 15.4.2 Diffusion limit: SGD as a discretization of Langevin dynamics

Decompose the minibatch gradient:
\[
\nabla\widehat{L}(\theta;\mathcal{B}) = \nabla L(\theta) + \xi(\theta), \qquad \mathbb{E}[\xi(\theta)\mid\theta]=0, \qquad \mathrm{Cov}(\xi(\theta)\mid\theta)=\Sigma(\theta).
\]
The SGD update becomes
\[
\theta_{k+1} = \theta_k - \eta\,\nabla L(\theta_k) - \eta\,\xi(\theta_k).
\]
Under small-step scaling, the continuous-time approximation is the Itﾃｴ SDE:
\[
d\theta_t = -\nabla L(\theta_t)\,dt + \sqrt{\eta\,\Sigma(\theta_t)}\,dW_t.
\]
The parameter process is Markov by construction: the next state depends only on the current \(\theta_t\), not the full history. This is the same structural move as in ﾂｧ15.1窶?5.2: a time-evolution law induces a Markov kernel on parameter space.

### 15.4.3 Stationary measures: from point optima to Gibbs-like laws

For *Stochastic Gradient Langevin Dynamics* (SGLD), the injected noise is calibrated so that the stationary distribution is approximately Gibbs:
\[
\pi(\theta) \propto \exp\!\left(-\frac{L(\theta)}{T_{\mathrm{eff}}}\right),
\]
with suitable regularity and a confining term. Vanilla SGD has state-dependent anisotropic noise, and its stationary measure may be non-equilibrium, but the correct conceptual object remains a stationary law over parameters.

<div class="ml-box">

**Machine learning consequence 1 (why "flat minima" matter).**
If \(\pi(\theta)\propto e^{-L(\theta)/T}\), then the probability mass of a basin depends on both depth and volume. Wide basins ("flat minima") can dominate even if they are not the deepest pointwise minima. This supplies a principled mechanism connecting stochastic training to generalization: SGD preferentially samples regions with large *measure* in parameter space.

</div>

<div class="ml-box">

**Machine learning consequence 2 (why checkpoint averaging works).**
Stochastic Weight Averaging (SWA) and Polyak窶迭uppert averaging compute time-averages along a Markov trajectory:
\[
\bar{\theta}_T := \frac{1}{T}\sum_{t=1}^T \theta_t.
\]
Ergodic reasoning then suggests \(\bar{\theta}_T\) approximates an expectation under the stationary measure when mixing is adequate. This is a direct application of the ergodic theorem, not an ad-hoc heuristic.

</div>

<div class="ml-box">

**Machine learning consequence 3 (ensembles as empirical measures).**
Deep ensembles can be interpreted as drawing multiple approximate samples from the stationary distribution induced by stochastic training (different seeds, data orders, augmentation randomness). Ensemble prediction is then an empirical approximation to an ensemble expectation 窶?exactly the object ergodicity is designed to justify.

</div>

---

## 15.5 Mixing Times and Spectral Gap: Quantitative Ergodicity

> **Theorem-sentence 15.5.** The spectral gap of the transition operator controls the mixing time; all quantitative convergence rates 窶?for MCMC, for exploration in MDPs, for stability of replay-buffer training 窶?follow from this single spectral quantity.

### 15.5.1 Total variation mixing time

<div class="def-box">

**Definition 15.5 (total variation distance and mixing time).** The total variation distance between measures \(\mu\) and \(\nu\) on \((\mathcal{S},\mathcal{B})\) is
\[
\|\mu-\nu\|_{\mathrm{TV}} := \sup_{A\in\mathcal{B}}|\mu(A)-\nu(A)|.
\]
The mixing time of a chain with stationary distribution \(\pi\) is
\[
t_{\mathrm{mix}}(\varepsilon) := \min\!\left\{n : \sup_{x\in\mathcal{S}}\|P^n(x,\cdot)-\pi\|_{\mathrm{TV}} \le \varepsilon\right\}.
\]

</div>

### 15.5.2 Spectral gap and its implications

For a reversible chain on a finite state space (or a self-adjoint operator in \(L^2(\pi)\)), let eigenvalues of \(P\) be ordered \(1=\lambda_1\ge\lambda_2\ge\cdots\ge\lambda_{|\mathcal{S}|}\ge -1\). The *spectral gap* is
\[
\gamma := 1 - \lambda_2.
\]

<div class="prop-box">

**Theorem 15.4 (spectral gap bound on mixing).** For a reversible, ergodic chain,
\[
t_{\mathrm{mix}}(\varepsilon) \le \frac{1}{\gamma}\log\!\left(\frac{1}{\varepsilon\,\pi_{\min}}\right),
\]
where \(\pi_{\min}=\min_x \pi(x)>0\). A larger gap \(\gamma\) means faster mixing.

</div>

<div class="ml-box">

**Exploration in MDPs.** A policy that induces a chain with small spectral gap on the state space explores slowly: states far from the current distribution take exponential time to reach in TV distance. Curiosity-driven exploration and count-based bonuses are heuristic methods for increasing \(\gamma\) by forcing visits to underexplored states.

**Replay buffer stability.** If the policy and representation change so fast that the induced state-distribution chain cannot mix before the next update, the replay buffer no longer samples from anything close to a stationary measure. This is a mixing-time failure, and it is the structural reason why off-policy learning with rapidly changing encoders can become unstable.

**Convergence diagnostics for Langevin / SGLD.** The analogous object in continuous time is the Poincarﾃｩ inequality constant; MCMC convergence diagnostics (effective sample size, \(\hat{R}\)) are empirical proxies for spectral gap estimation. Poor mixing = small gap = slow ergodic averaging.

</div>

### 15.5.3 Irreducibility and aperiodicity: the two structural prerequisites

<div class="def-box">

**Definition 15.6 (irreducibility).** A Markov chain is \(\varphi\)-irreducible (w.r.t. a measure \(\varphi\)) if for every \(x\in\mathcal{S}\) and every set \(A\) with \(\varphi(A)>0\), there exists \(n\ge 1\) such that \(P^n(x,A)>0\). Irreducibility means every part of the state space is reachable from everywhere.

**Definition 15.7 (aperiodicity).** A state \(x\) has period \(d(x)=\gcd\{n\ge 1: P^n(x,x)>0\}\). The chain is aperiodic if \(d(x)=1\) for all \(x\) (equivalently, the chain does not cycle deterministically). Aperiodicity prevents the chain from oscillating between disjoint subsets and never converging to \(\pi\).

</div>

**Canonical result:** an irreducible, aperiodic, positive-recurrent Markov chain on a countable state space is ergodic with a unique stationary distribution \(\pi\), and \(\|P^n(x,\cdot)-\pi\|_{\mathrm{TV}}\to 0\) for all \(x\).

**Harris recurrence** extends this to general state spaces: the chain returns to any set of positive \(\pi\)-measure infinitely often a.s., which is the correct generalization of positive recurrence.

---

## 15.6 Scholium: What This Chapter Forces You to Admit

<div class="scholium-box">

1. **Markovity is an information constraint.** It demands that \(\sigma(X_n)\) be sufficient for future prediction 窶?an extreme truncation of the history \(\sigma\)-algebra. Learning an adequate state representation is learning to impose this constraint.

2. **Time evolution is operator theory.** The Chapman窶適olmogorov equation is the semigroup identity for probability evolution; deep sequential models are parametric approximations to this semigroup, and their stability is governed by the spectral properties of their linearized generators.

3. **Ergodicity is the license behind learning from trajectories.** RL, MCMC, replay buffers, and parts of modern optimization rely 窶?explicitly or implicitly 窶?on ergodic replacement of ensemble averages by time averages. When the hidden assumption fails (multi-modal stationary distribution, slow mixing, non-stationarity), the empirical estimates produced are structurally unreliable.

4. **SGD is better viewed as a Markov sampler than a point optimizer.** In high dimensions, the object of interest is the stationary measure over parameters, because that is what the algorithm actually produces. Flat minima, weight averaging, and ensemble diversity are consequences of this measure-theoretic reality.

5. **Quantitative ergodicity 窶?mixing time, spectral gap 窶?is the correct language** for analyzing exploration in MDPs, convergence of MCMC-based learning, and stability of experience-replay training. These are not separate engineering problems; they are instances of the same spectral theory.

</div>

The natural continuation is martingale theory (Chapter 16): martingales are the canonical object for analyzing the fluctuations along a Markov trajectory, and they supply the convergence proofs for stochastic approximation and RL value learning.

**Chapter 016: Martingale Theory 窶?Convergence of Stochastic Approximation and the Mathematics of Online Learning.**
