---
title: "Chapter 16: Martingales & Stopping Times"
layout: "single"
url: "/book/chapters/chapter016/"
summary: "Martingales as conservation of conditional expectation; TD error as Bellman drift plus martingale noise; convergence of stochastic approximation in RL; stopping times and episodic legality; deep RL as martingale repair."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
weight: 16
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

## Chapter 16 &mdash; Martingales &amp; Stopping Times: The Conservation Law of Conditional Expectation and the Convergence Geometry of Reinforcement Learning

*Xujiang Tang*

</div>

## Abstract

Chapter 15 compressed *time* by imposing the Markov property. This chapter compresses *learning*: it explains why value-function learning converges under stochastic rewards and transitions. The central claim is precise:

> **Thesis.** In reinforcement learning, the *learnable* part of the temporal-difference (TD) error is its **predictable drift** (a Bellman residual), while the remaining part is a **martingale difference** (pure innovation). Convergence is the disappearance of the drift under stochastic approximation.

All martingale statements are derived directly from MDP definitions, not imposed from outside.

---

## 16.0 Notation and Primitives (RL-First)

An MDP is a tuple \(\mathcal{M}=(\mathcal{S},\mathcal{A},P,r,\gamma)\), where \(P(\cdot\mid s,a)\) is a transition kernel on \(\mathcal{S}\), \(r(s,a)\) is the reward distribution (or its mean), and \(\gamma\in(0,1)\) is the discount factor.

A stationary policy \(\pi(a\mid s)\) together with \(\mathcal{M}\) induces a **trajectory probability measure** \(\mathbb{P}_\pi\) on sequences
\[
(S_0,A_0,R_1,S_1,A_1,R_2,\dots).
\]

The agent's **information flow** (natural filtration) is
\[
\mathcal{F}_t := \sigma(S_0,A_0,R_1,S_1,\dots,S_t).
\]

The discounted **return** is
\[
G_t := \sum_{k=0}^{\infty}\gamma^k R_{t+k+1}.
\]

Value functions are conditional expectations under \(\mathbb{P}_\pi\):
\[
V^\pi(s) := \mathbb{E}_\pi[G_t\mid S_t=s], \qquad Q^\pi(s,a) := \mathbb{E}_\pi[G_t\mid S_t=s,A_t=a].
\]

---

## 16.1 Martingales as Conservation of Conditional Expectation

> **Theorem-sentence 16.1.** A martingale is not a "random process without trend"; it is a process that conserves conditional expectation across the information horizon: each future value, once conditioned on the present, equals the present value.

<div class="def-box">

**Definition 16.1 (martingale).** A process \(\{X_t\}_{t\ge 0}\) adapted to \(\{\mathcal{F}_t\}\) is a *martingale* if: (i) \(\mathbb{E}[|X_t|]<\infty\) for all \(t\), and (ii) \(\mathbb{E}[X_{t+1}\mid\mathcal{F}_t]=X_t\) almost surely.

</div>

This is a conservation law over the information horizon: once \(\mathcal{F}_t\) is fixed, the future has no predictable drift. Variants: a *supermartingale* satisfies \(\mathbb{E}[X_{t+1}\mid\mathcal{F}_t]\le X_t\) (a.s.), and a *submartingale* satisfies the reverse.

### 16.1.1 RL instantiation: the best prediction process is always a martingale

Fix any integrable terminal variable \(Y\in L^1\) (e.g., an episode outcome, a final score, or the infinite-horizon return \(G_0\)). Define
\[
X_t := \mathbb{E}[Y\mid\mathcal{F}_t].
\]
Then \(\{X_t\}\) is a martingale.

<div class="proof-box">

**Proof.** By the tower property of conditional expectation (Chapter 14),
\[
\mathbb{E}[X_{t+1}\mid\mathcal{F}_t] = \mathbb{E}[\mathbb{E}[Y\mid\mathcal{F}_{t+1}]\mid\mathcal{F}_t] = \mathbb{E}[Y\mid\mathcal{F}_t] = X_t. \quad\square
\]

</div>

<div class="ml-box">

**RL meaning.** If \(X_t\) is the agent's posterior expectation of an eventual outcome \(Y\), then "learning more" (\(\mathcal{F}_t\subset\mathcal{F}_{t+1}\)) can change the realized value of \(X_{t+1}\), but it cannot introduce a *systematic* drift given current information. Any apparent "bias" in value estimates that persists across time steps is a signal of a *modeling error*, not inherent randomness.

</div>

### 16.1.2 Sub- and supermartingales in RL

If the current estimate of future reward is *pessimistic* (underestimates), then \(X_t\) is a submartingale: as more information arrives, estimates tend to rise. This structure appears in pessimistic off-policy algorithms (e.g., conservative Q-learning), where the value estimate is deliberately biased downward for safety, and the submartingale property certifies that estimates improve monotonically as experience accumulates.

---

## 16.2 Martingale Difference Sequences: The Formal Definition of Innovation

> **Theorem-sentence 16.2.** A martingale difference sequence is the measure-theoretic formalization of "pure innovation": a term that is integrable, \(\mathcal{F}_{t+1}\)-measurable, and has zero conditional mean given current information.

<div class="def-box">

**Definition 16.2 (martingale difference sequence, MDS).** A sequence \(\{\varepsilon_{t+1}\}\) is an *MDS* w.r.t. \(\{\mathcal{F}_t\}\) if \(\varepsilon_{t+1}\) is \(\mathcal{F}_{t+1}\)-measurable, integrable, and
\[
\mathbb{E}[\varepsilon_{t+1}\mid\mathcal{F}_t] = 0 \quad \text{a.s.}
\]

</div>

**Geometric interpretation in \(L^2\).** If \(\varepsilon_{t+1}\in L^2\), then \(\mathbb{E}[\varepsilon_{t+1}\mid\mathcal{F}_t]=0\) means \(\varepsilon_{t+1}\) is orthogonal (in the inner product \(\mathbb{E}[\cdot\,\cdot]\)) to every \(\mathcal{F}_t\)-measurable square-integrable function. It is new information that cannot be predicted from the past â€?the orthogonal complement of the history subspace in \(L^2\).

**Variance additivity.** A key consequence: if \(\varepsilon_1,\varepsilon_2,\dots\) is an MDS with \(\varepsilon_t\in L^2\), then
\[
\mathrm{Var}\!\left(\sum_{t=1}^T \varepsilon_t\right) = \sum_{t=1}^T \mathbb{E}[\varepsilon_t^2],
\]
analogous to independence. This is why MDS-driven random walks grow at rate \(\sqrt{T}\) rather than \(T\).

---

## 16.3 TD Error = Bellman Drift + Martingale Noise

> **Theorem-sentence 16.3.** The temporal-difference error decomposes exactly into a predictable Bellman residual (the learnable drift) and a martingale difference (irreducible environmental noise); at the true value function, the drift vanishes and only martingale noise remains.

### 16.3.1 Bellman operator from RL primitives

From the return identity \(G_t = R_{t+1} + \gamma G_{t+1}\), define the **Bellman expectation operator**:
\[
(T^\pi V)(s) := \mathbb{E}_\pi\!\big[R_{t+1} + \gamma V(S_{t+1})\mid S_t=s\big].
\]
Then \(V^\pi\) is the unique fixed point \(V^\pi = T^\pi V^\pi\). The operator \(T^\pi\) is a contraction in \(\|\cdot\|_\infty\) with modulus \(\gamma\):
\[
\|T^\pi V - T^\pi U\|_\infty \le \gamma\,\|V-U\|_\infty.
\]

### 16.3.2 Doob decomposition of the TD error

For any candidate \(V\), define the TD error:
\[
\delta_t(V) := R_{t+1} + \gamma V(S_{t+1}) - V(S_t).
\]

Computing its conditional expectation given \(\mathcal{F}_t\):
\[
\mathbb{E}_\pi[\delta_t(V)\mid\mathcal{F}_t] = \mathbb{E}_\pi[R_{t+1} + \gamma V(S_{t+1})\mid\mathcal{F}_t] - V(S_t) = (T^\pi V - V)(S_t).
\]

Hence the **Doob decomposition** of the TD error:

<p>
$$
\delta_t(V) = \underbrace{(T^\pi V - V)(S_t)}_{\text{predictable drift (Bellman residual)}} + \underbrace{\varepsilon_{t+1}(V)}_{\text{MDS noise}},
$$
</p>

where
\[
\varepsilon_{t+1}(V) := \delta_t(V) - \mathbb{E}[\delta_t(V)\mid\mathcal{F}_t], \qquad \mathbb{E}[\varepsilon_{t+1}(V)\mid\mathcal{F}_t] = 0.
\]

<div class="prop-box">

**Theorem 16.3 (TD innovation at the truth).** If \(V=V^\pi\), then \((T^\pi V^\pi - V^\pi)(S_t)=0\) and therefore
\[
\mathbb{E}_\pi[\delta_t(V^\pi)\mid\mathcal{F}_t] = 0,
\]
i.e., \(\delta_t(V^\pi)\) is a martingale difference sequence.

</div>

<div class="ml-box">

**Practical criterion.** When a critic is "correct," its TD residual is unpredictable given the past: what remains is irreducible stochasticity of the environment. Any persistent predictable structure in the TD error is evidence that the value function has not converged.

</div>

### 16.3.3 Worked micro-example: explicit drift vs. noise

Consider two states \(\{s_0,s_1\}\), one action, deterministic transition \(s_0\to s_1\to s_1\), and reward noise:
\[
R_{t+1} = \begin{cases} 1 + \xi_{t+1}, & S_t=s_0, \\ \xi_{t+1}, & S_t=s_1, \end{cases} \qquad \mathbb{E}[\xi_{t+1}\mid\mathcal{F}_t] = 0.
\]
For any \(V\), the TD error is:
\[
\delta_t(V) = \begin{cases} 1 + \xi_{t+1} + \gamma V(s_1) - V(s_0), & S_t=s_0, \\ \xi_{t+1} + (\gamma-1)V(s_1), & S_t=s_1. \end{cases}
\]
Taking conditional expectations strips \(\xi_{t+1}\) and reveals the drift:
\[
\mathbb{E}[\delta_t(V)\mid\mathcal{F}_t] = \begin{cases} 1 + \gamma V(s_1) - V(s_0), & S_t=s_0, \\ (\gamma-1)V(s_1), & S_t=s_1. \end{cases}
\]
"Learning" is exactly "killing these drifts." Setting both to zero gives \(V^\pi(s_0)=\frac{1}{1-\gamma}\), \(V^\pi(s_1)=0\). The leftover \(\xi_{t+1}\) is the unavoidable martingale noise.

---

## 16.4 Convergence Mechanism: Stochastic Approximation Driven by MDS Noise

> **Theorem-sentence 16.4.** Stochastic approximation converges to the Bellman fixed point because the martingale noise terms average out under Robbinsâ€“Monro step sizes, while the Bellman drift provides a contractive restoring force.

### 16.4.1 Tabular TD(0) policy evaluation

The TD(0) update for visited state \(S_t\):
\[
V_{t+1}(S_t) = V_t(S_t) + \alpha_t\,\delta_t(V_t).
\]
Substituting the Doob decomposition:
\[
V_{t+1}(S_t) = V_t(S_t) + \alpha_t\Big((T^\pi V_t - V_t)(S_t) + \varepsilon_{t+1}(V_t)\Big).
\]

Two roles:
- **Deterministic drift:** \((T^\pi V_t - V_t)\) is the contractive direction toward \(V^\pi\).
- **Martingale noise:** \(\varepsilon_{t+1}(V_t)\) has zero conditional mean and bounded variance under standard assumptions.

<div class="prop-box">

**Theorem 16.4 (SA convergence template).** If rewards are bounded, every state is visited infinitely often, and step sizes satisfy the Robbinsâ€“Monro conditions
\[
\sum_t \alpha_t = \infty, \qquad \sum_t \alpha_t^2 < \infty,
\]
then TD(0) tracks the ODE \(\dot{V} = T^\pi V - V\) and converges to \(V^\pi\) a.s.

</div>

**Why martingales matter for convergence.** The Robbinsâ€“Monro conditions make the *cumulative* martingale noise \(\sum_t \alpha_t \varepsilon_{t+1}\) almost surely finite, while the drift keeps integrating until the Bellman residual vanishes. The two conditions \(\sum\alpha_t=\infty\) and \(\sum\alpha_t^2<\infty\) are not heuristic choices; they are precisely the conditions that ensure: (i) enough total step size to overcome any initialization bias, and (ii) enough decay to prevent noise accumulation.

### 16.4.2 Rate of convergence and variance

Under the SA framework with bounded rewards and step size \(\alpha_t=\frac{c}{t}\), the asymptotic mean-squared error decays as \(O(1/t)\). The variance of the martingale noise \(\varepsilon_{t+1}\) enters directly:
\[
\mathbb{E}[(V_t(s) - V^\pi(s))^2] = O\!\left(\frac{\mathbb{E}[\varepsilon^2]}{t}\right).
\]
High-variance environments (large \(\mathbb{E}[\varepsilon^2]\)) slow convergence; lower discount \(\gamma\) shrinks the effective noise by reducing the horizon over which errors accumulate.

---

## 16.5 Control: SARSA and Q-Learning in the Same Martingale Geometry

> **Theorem-sentence 16.5.** Both on-policy SARSA and off-policy Q-learning decompose into Bellman drift plus MDS noise; the operators differ but the martingale structure is identical.

### 16.5.1 SARSA (on-policy) as SA with MDS noise

Define the SARSA TD error:
\[
\delta_t^{\mathrm{SARSA}}(Q) := R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t).
\]
Then
\[
\mathbb{E}[\delta_t^{\mathrm{SARSA}}(Q)\mid\mathcal{F}_t] = (T^\pi Q - Q)(S_t,A_t),
\]
where \((T^\pi Q)(s,a) := \mathbb{E}\!\big[R_{t+1} + \gamma\,\mathbb{E}_{a'\sim\pi(\cdot\mid S_{t+1})}Q(S_{t+1},a')\mid S_t=s,A_t=a\big]\). At \(Q=Q^\pi\), the drift vanishes and the SARSA TD error is an MDS.

### 16.5.2 Q-learning (off-policy control) and the optimality operator

The optimality (Bellman optimal) operator:
\[
(T^{\ast} Q)(s,a) := \mathbb{E}\!\big[R_{t+1} + \gamma\max_{a'}Q(S_{t+1},a')\mid S_t=s,A_t=a\big].
\]
Q-learning uses the error \(\delta_t^Q(Q) := R_{t+1} + \gamma\max_{a'}Q(S_{t+1},a') - Q(S_t,A_t)\), and
\[
\mathbb{E}[\delta_t^Q(Q)\mid\mathcal{F}_t] = (T^{\ast} Q - Q)(S_t,A_t).
\]
The same drift+MDS structure holds. Tabular convergence requires: (i) contraction of \(T^{\ast}\) with modulus \(\gamma\); (ii) diminishing step sizes; (iii) sufficient visitation of all \((s,a)\) pairs. Point (iii) is the mathematical condition underlying "exploration": without it, the SA drift is never applied to unvisited regions and convergence to \(Q^{\ast}\) fails.

---

## 16.6 Stopping Times: Episodic RL as Optional Stopping Under Admissible Truncation

> **Theorem-sentence 16.6.** A stopping time is a random time whose occurrence is determined by the filtration alone â€?not the future. Optional stopping theorems give conditions under which the martingale's expectation is preserved at the stopping time; these conditions formalize when episode truncation is statistically unbiased.

<div class="def-box">

**Definition 16.3 (stopping time).** A random variable \(\tau:\Omega\to\mathbb{N}\cup\{\infty\}\) is a *stopping time* w.r.t. \(\{\mathcal{F}_t\}\) if \(\{\tau=t\}\in\mathcal{F}_t\) for all \(t\).

</div>

The decision "stop now" can only depend on observed history, not on the future. Episode termination (game over, goal reached, timeout) is always a stopping time relative to the agent's filtration.

<div class="prop-box">

**Theorem 16.5 (optional stopping theorem, bounded case).** If \(\{X_t\}\) is a martingale, \(\tau\) is a stopping time, and either \(\tau\) is bounded a.s. or \(\{X_{t\wedge\tau}\}\) is uniformly integrable, then
\[
\mathbb{E}[X_\tau] = \mathbb{E}[X_0].
\]

</div>

<div class="ml-box">

**Concrete RL implication.** Monte Carlo return estimates \(G_\tau = \sum_{t=0}^{\tau-1}\gamma^t R_{t+1}\) are unbiased estimates of \(V^\pi(S_0)\) precisely when the episode stopping time \(\tau\) satisfies the integrability conditions above. If termination can be infinite with non-negligible probability (improper continuing tasks without discounting, or unstable dynamics), naÃ¯ve truncation introduces bias. Martingale theory precisely diagnoses when "cutting an episode" is statistically legal and when it is not.

</div>

### 16.6.1 Optional stopping in value estimation: two failure modes

**Failure mode 1 (biased returns from premature truncation).**
In practice, episodes are truncated at a finite horizon \(H\) even in continuing tasks. If \(V_H(s)\ne 0\) at truncation, bootstrapping from \(V_H\) replaces the missing tail \(\sum_{t=H}^{\infty}\gamma^t R_{t+1}\). This is formally an importance-weighted truncation, not a true optional stop, and introduces bias unless \(V_H\) is accurate.

**Failure mode 2 (biased returns from goal-conditioned truncation).**
If the stopping rule "episode ends when reward is high" is correlated with the return \(G_\tau\), the expectation \(\mathbb{E}[G_\tau]\) is no longer \(V^\pi(S_0)\). The optional stopping theorem's condition â€?that \(\tau\) is \(\mathcal{F}\)-measurable and independent of the future increments â€?is violated, and selection bias enters.

---

## 16.7 Martingale Convergence Theorems and Their Implications

> **Theorem-sentence 16.7.** Martingale convergence theorems are the analytic foundation for almost-sure convergence in stochastic approximation: they certify that bounded, monotone (super/sub) martingales converge a.s. to a limit.

<div class="prop-box">

**Theorem 16.6 (martingale convergence theorem).** Let \(\{X_t\}\) be a supermartingale with \(\sup_t \mathbb{E}[X_t^-]<\infty\). Then \(X_t\to X_\infty\) a.s. for some integrable \(X_\infty\).

</div>

**Application to RL convergence proofs.** Many tabular RL convergence proofs construct a Lyapunov function \(\Phi_t = \|V_t - V^\pi\|^2\) and show it is a (approximate) supermartingale: the drift term decreases it by at least \(-\alpha_t c\,\Phi_t\) and the martingale noise term adds \(+\alpha_t^2\,\sigma^2\). Summing over steps and applying the convergence theorem gives a.s. convergence once \(\sum\alpha_t^2<\infty\).

The key construction:

\[
\mathbb{E}[\Phi_{t+1}\mid\mathcal{F}_t] \le (1 - 2\alpha_t c)\,\Phi_t + \alpha_t^2\,\sigma^2.
\]

Under Robbinsâ€“Monro conditions this yields \(\Phi_t\to 0\) a.s., i.e., \(V_t\to V^\pi\) a.s.

---

## 16.8 Deep RL: Why Target Networks, Replay, and Importance Sampling Are Martingale Repairs

> **Theorem-sentence 16.8.** Deep RL engineering heuristics â€?target networks, experience replay, importance sampling corrections â€?are repairs to specific violations of the filtration and measure conditions required for martingale-structure convergence.

Deep RL violates the clean filtration assumptions in three common ways.

**1. Bootstrapping with moving targets.**
If the bootstrap target uses the same parameters being updated, the "drift" term changes at every step, oscillating rather than contracting. A target network \(Q_{\bar\theta}\) (updated slowly or periodically) approximates a fixed Bellman operator, restoring a stable drift toward a well-defined fixed point.

**2. Experience replay breaks the filtration.**
Replayed samples are not generated by the current filtration \(\mathcal{F}_t\). This invalidates the conditional expectation identity \(\mathbb{E}[\varepsilon_{t+1}\mid\mathcal{F}_t]=0\) for the replay-sampled noise, because the noise is now correlated with the *replay buffer* filtration, not the agent's current information state. Prioritized replay further distorts the sampling distribution. These are genuine measure-theoretic violations, not engineering imperfections.

**3. Off-policy learning and density-ratio correction.**
If behavior policy is \(\mu\) but the objective is under \(\pi\), the samples come from the wrong measure. The correction uses the discrete Radonâ€“Nikodym derivative (Chapter 13):
\[
\rho_t := \frac{\pi(A_t\mid S_t)}{\mu(A_t\mid S_t)}.
\]
It restores correct expectations:
\[
\mathbb{E}_\mu[\rho_t Z_t] = \mathbb{E}_\pi[Z_t]
\]
for suitable \(Z_t\). This is exactly the measure-theoretic condition needed to regain martingale-difference noise around the intended drift. Without it, the "drift" points toward the wrong fixed point.

---

## 16.9 Azumaâ€“Hoeffding and Freedman Inequalities: Concentration for MDS

> **Theorem-sentence 16.9.** The Azumaâ€“Hoeffding and Freedman inequalities provide exponential concentration bounds for sums of bounded martingale differences; these are the analytic tools behind finite-sample regret bounds in RL.

<div class="prop-box">

**Theorem 16.7 (Azumaâ€“Hoeffding).** Let \(\{\varepsilon_t\}\) be an MDS with \(|\varepsilon_t|\le c_t\) a.s. Then for any \(\lambda>0\),
\[
\mathbb{P}\!\left(\sum_{t=1}^T \varepsilon_t \ge \lambda\right) \le \exp\!\left(-\frac{\lambda^2}{2\sum_{t=1}^T c_t^2}\right).
\]

</div>

<div class="prop-box">

**Theorem 16.8 (Freedman).** Let \(\{\varepsilon_t\}\) be an MDS with \(|\varepsilon_t|\le c\) a.s. and conditional variance \(\mathbb{E}[\varepsilon_t^2\mid\mathcal{F}_{t-1}]\le \sigma_t^2\). Define \(V_T=\sum_{t=1}^T \sigma_t^2\). Then for any \(\lambda,b>0\),
\[
\mathbb{P}\!\left(\sum_{t=1}^T \varepsilon_t \ge \lambda,\; V_T\le b\right) \le \exp\!\left(-\frac{\lambda^2}{2(b + c\lambda/3)}\right).
\]

</div>

<div class="ml-box">

**RL regret bounds.** In bandit and RL settings, the regret after \(T\) steps decomposes into a deterministic bias term (Bellman drift contributions from suboptimal exploration) and a stochastic term (martingale sum over TD noise). Azumaâ€“Hoeffding bounds the stochastic term, yielding regret bounds of order \(O(\sqrt{T})\) or better. Freedman's inequality gives sharper bounds when the conditional variance is small â€?important in low-noise environments where exploration is the bottleneck rather than estimation noise.

</div>

---

## 16.10 Scholium: What Martingales Say About Learning in RL

<div class="scholium-box">

1. **Definition-level essence.** Value functions are conditional expectations under the policy-induced trajectory measure; learning is the act of aligning a parametric predictor with these conditional expectations. The martingale structure is not added on top of RL â€?it *is* RL's mathematical skeleton.

2. **TD error anatomy.** The TD error decomposes into a predictable drift (Bellman residual), which contains learnable structure, and a martingale difference (innovation), which contains irreducible randomness. No amount of computation can reduce the martingale component; only the drift can be eliminated by learning.

3. **Convergence meaning.** Convergence is not "the TD error becomes zero pointwise"; it is "the predictable drift vanishes," leaving only martingale noise.

4. **Episodic legality.** Stopping-time theory formalizes when truncating time (episodes) preserves expectation and when it introduces bias. The optional stopping conditions are not technicalities; they are the conditions under which Monte Carlo returns are valid training targets.

5. **Deep RL heuristics are principled repairs.** Target networks restore a stable drift; replay-buffer corrections address filtration violations; importance weights restore the correct measure. Understanding each as a martingale repair clarifies both why the heuristic works and what can go wrong when it fails.

6. **Concentration inequalities complete the picture.** Azumaâ€“Hoeffding and Freedman inequalities bound the cumulative martingale noise, yielding finite-sample guarantees. The Robbinsâ€“Monro conditions are the asymptotic counterpart of these bounds.

</div>

The natural continuation is concentration inequalities for dependent sequences in full generality (Chapter 17), covering Hoeffding, Bernstein, and their extensions to non-i.i.d. data â€?the tools that convert the martingale framework into quantitative learning-theoretic bounds.

**Chapter 017: Concentration Inequalities â€?Large Deviation Bounds and the Mathematics of Generalization.**
