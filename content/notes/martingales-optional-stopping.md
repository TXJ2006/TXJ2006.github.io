---
title: "Martingales and Optional Stopping"
subtitle: "Why Bandit Proofs Need Filtrations"
summary: "Martingale tools for adaptive processes and stopping-time arguments."
description: "Martingale tools for adaptive processes and stopping-time arguments."
date: 2026-06-18
lastmod: 2026-06-18
weight: 50
tags: ["Martingales", "Optional Stopping", "Bandit Theory"]
draft: false
ShowToc: false
hideMeta: true
---

## A Decision Rule Also Chooses When to Look

Suppose an online experiment compares two recommendation policies. Every few minutes the dashboard refreshes. The new policy is slightly ahead. The lead disappears, returns, and grows again. Someone says:


*"Let us stop as soon as the result looks convincing."*


Nothing about that sentence is unusual. In fact, it is exactly what a sequential learner should do: look at the data, decide whether enough has been learned, and either continue or stop.

The difficulty is more subtle. A confidence interval designed for a fixed sample size answers a fixed-time question: $\text{``What can happen at time }n\text{?''}$ A sequential procedure asks a different question:

<div class="display-equation">
$$
\text{``What can happen at any time at which the data persuade us to stop?''}
$$
</div>

 Those are not the same event.

A bandit algorithm is full of such random times. It may stop exploring an arm when its upper confidence bound becomes small. It may eliminate an action when the evidence is strong enough. It may end an experiment once one action is clearly best. The amount of data collected is therefore not fixed in advance. It is produced by the data themselves.

> **Key idea.**
>
> The word *adaptive* means more than "the next action depends on the past." It also means that the amount of data, the arm being sampled, and the time at which a conclusion is reached may all depend on the past.

Martingales are the language that keeps this dependence honest (Williams 1991; Lattimore and Szepesvari 2020). They do not make adaptivity disappear. They separate it into two pieces:

<div class="display-equation">
$$
\boxed{\text{a choice made from the past}}
\qquad+
\qquad
\boxed{\text{new noise whose conditional mean is zero}.}
$$
</div>

 Once this separation is visible, many proofs become simple bookkeeping.

### The first warning: peeking changes the event

Let $X_1,X_2,\ldots$ be Bernoulli observations with mean $1/2$. At a fixed time $n$, an exact one-sided test can be calibrated so that $\Pbb_{1/2}(\text{reject at time }n)\le 0.05.$ Now imagine applying a fresh $5\%$ test after every observation and stopping at the first rejection. The event is no longer $\{\text{reject at time }n\}.$ It is $\bigcup_{t=1}^{n}\{\text{reject at time }t\}.$ Even when each individual event is rare, their union need not be rare.

This is not a technicality. It is the central reason fixed-time probability statements cannot simply be reused after arbitrary monitoring.

> **Think.**
>
> Optional stopping is often summarized as "stopping does not create profit in a fair game." That slogan is useful, but incomplete. The theorem only works when the stopping rule and the process satisfy conditions that prevent rare, enormous outcomes from carrying hidden expectation.

## The Past as a Mathematical Object

Before defining a martingale, we need a clean way to say what is known at each moment.

At the beginning of round $t$, the learner has seen everything from the previous rounds. Call this information $\F_{t-1}$. Using only this information, the learner chooses an action $A_t$. Then the reward $X_t$ arrives, and the information grows to $\F_t$.


The sequence $\F_0\subseteq \F_1\subseteq \F_2\subseteq\cdots$ is called a *filtration*. The inclusion simply says that information accumulates. Yesterday's facts do not become unknown today.

### Adapted and predictable

A process $Y_t$ is *adapted* when $Y_t$ is known by time $t$. In symbols, $Y_t$ is $\F_t$-measurable.

A process $H_t$ is *predictable* when $H_t$ is known just before round $t$. In discrete time, this means $H_t$ is $\F_{t-1}$-measurable.

The distinction is small in notation and decisive in proofs:

<div class="display-equation">
$$
\underbrace{A_t}_{\text{chosen before reward}}
\quad\text{is predictable,}
\qquad
\underbrace{X_t}_{\text{revealed after action}}
\quad\text{is adapted but not predictable.}
$$
</div>


A legal bandit policy cannot choose $A_t$ using $X_t$, because $X_t$ has not yet happened. That simple chronological fact is exactly what predictability records.

### Conditional expectation means "average after using the past"

The ordinary mean $\E[X_t]$ averages over every possible history. The conditional mean $\E[X_t\mid \F_{t-1}]$ first freezes the past and then averages only over what remains random.

For a stochastic bandit with arm means $\mu_1,\ldots,\mu_K$, once $A_t$ has been chosen we have

<div class="display-equation">
$$
\E[X_t\mid \F_{t-1}]
=
\mu_{A_t}.
$$
</div>

 This equation does not say the reward equals $\mu_{A_t}$. It says that after all past information and the current action are known, the remaining uncertainty has mean $\mu_{A_t}$.

## Martingales: Fairness After Conditioning on the Past

A martingale is often introduced through gambling. The useful idea is simpler:


*After using everything already known, the next expected change is zero.*


Let $M_0,M_1,M_2,\ldots$ be an adapted process with finite expectations. It is a martingale when

<div class="display-equation">
$$
\E[M_t\mid \F_{t-1}]=M_{t-1}
\qquad\text{for every }t\ge 1.
$$
</div>

 Subtract $M_{t-1}$ from both sides: $\E[M_t-M_{t-1}\mid \F_{t-1}]=0.$ If we write $D_t=M_t-M_{t-1},$ then

<div class="display-equation">
$$
\boxed{\E[D_t\mid \F_{t-1}]=0.}
$$
</div>

 The sequence $D_t$ is called a *martingale difference sequence*.

> **Key idea.**
>
> The condition is not that the increments are independent. The condition is weaker and better suited to sequential learning: after conditioning on the entire past, the next increment has mean zero.

### The simple random walk

Let $\xi_t\in\{-1,+1\}$ be a fair coin step: $\Pbb(\xi_t=1)=\Pbb(\xi_t=-1)=\frac12.$ Define

<div class="display-equation">
$$
S_t=\sum_{s=1}^{t}\xi_s,
\qquad S_0=0.
$$
</div>

 Then

<div class="display-equation">
$$
\begin{align*}
\E[S_t\mid\F_{t-1}]
&=\E[S_{t-1}+\xi_t\mid\F_{t-1}]\\
&=S_{t-1}+\E[\xi_t\mid\F_{t-1}]\\
&=S_{t-1}+0\\
&=S_{t-1}.
\end{align*}
$$
</div>

 So $S_t$ is a martingale.

Notice what the equation does *not* say. It does not say the path stays near zero. A fair random walk may wander far upward or downward. Martingale fairness concerns conditional expectation, not pathwise calmness.

### The noise in a bandit is a martingale difference

Let $A_t$ be the arm selected at round $t$, and let $X_t$ be its reward. Define the centered reward $D_t=X_t-\mu_{A_t}.$ Then

<div class="display-equation">
$$
\begin{align*}
\E[D_t\mid\F_{t-1}]
&=\E[X_t-\mu_{A_t}\mid\F_{t-1}]\\
&=\E[X_t\mid\F_{t-1}]-\mu_{A_t}\\
&=\mu_{A_t}-\mu_{A_t}\\
&=0.
\end{align*}
$$
</div>

 Therefore $M_t=\sum_{s=1}^{t}(X_s-\mu_{A_s})$ is a martingale.

This is the first important research pattern:

> **Proof pattern.**
>
> Write an adaptive observation as

<div class="display-equation">
$$
\text{observation}
=
\text{conditional mean given the past}
+
\text{martingale noise}.
$$
</div>

 The policy may be highly adaptive. The centered noise can still have conditional mean zero.

## Predictable Multipliers: How Adaptivity Enters Safely

Suppose $D_t$ is a martingale difference and $H_t$ is chosen using only the past. Then $H_tD_t$ is also a martingale difference, provided it is integrable.

The proof is one line, but every symbol matters:

<div class="display-equation">
$$
\begin{align*}
\E[H_tD_t\mid\F_{t-1}]
&=H_t\E[D_t\mid\F_{t-1}]\\
&=H_t\cdot 0\\
&=0.
\end{align*}
$$
</div>

 We were allowed to move $H_t$ outside the conditional expectation because $H_t$ was already known at time $t-1$.

This is the mathematical version of a basic rule:


*You may decide how much to bet before the outcome arrives, not after seeing it.*


### Following one arm inside an adaptive experiment

Fix an arm $a$. Define $I_{a,t}=\one\{A_t=a\}.$ Because the learner chooses $A_t$ before observing $X_t$, the indicator $I_{a,t}$ is $\F_{t-1}$-measurable.

Now define the arm-specific centered increment $D_{a,t}=I_{a,t}(X_t-\mu_a).$ Its conditional mean is

<div class="display-equation">
$$
\begin{align*}
\E[D_{a,t}\mid\F_{t-1}]
&=\E[I_{a,t}(X_t-\mu_a)\mid\F_{t-1}]\\
&=I_{a,t}\E[X_t-\mu_a\mid\F_{t-1}]\\
&=I_{a,t}(\mu_{A_t}-\mu_a).
\end{align*}
$$
</div>

 There are two cases:

<div class="display-equation">
$$
A_t\neq a
\quad\Longrightarrow\quad
I_{a,t}=0,
$$
</div>

 and

<div class="display-equation">
$$
A_t=a
\quad\Longrightarrow\quad
\mu_{A_t}-\mu_a=0.
$$
</div>

 Hence $\E[D_{a,t}\mid\F_{t-1}]=0.$ Therefore $S_a(t)=\sum_{s=1}^{t}I_{a,s}(X_s-\mu_a)$ is a martingale.

The corresponding number of observations is $N_a(t)=\sum_{s=1}^{t}I_{a,s}.$ Whenever $N_a(t)>0$,

<div class="display-equation">
$$
\begin{align*}
S_a(t)
&=\sum_{s=1}^{t}I_{a,s}X_s
-\mu_a\sum_{s=1}^{t}I_{a,s}\\
&=N_a(t)\widehat\mu_a(t)-N_a(t)\mu_a\\
&=N_a(t)\bigl(\widehat\mu_a(t)-\mu_a\bigr).
\end{align*}
$$
</div>

 Thus

<div class="display-equation">
$$
\boxed{
\widehat\mu_a(t)-\mu_a
=
\frac{S_a(t)}{N_a(t)}.
}
$$
</div>

 The numerator is martingale noise. The denominator is a random sample size chosen by the algorithm.

> **Think.**
>
> The stopped sample mean need not be unbiased, even when the stopped sum has mean zero. Ratios are nonlinear:

<div class="display-equation">
$$
\E\left[\frac{S_\tau}{\tau}\right]
\neq
\frac{\E[S_\tau]}{\E[\tau]}.
$$
</div>

 Optional stopping controls the martingale itself. It does not automatically validate every statistic built from it.

### A second example: importance weighting

Suppose the learner chooses arm $a$ with known probability $p_t(a)>0$, where $p_t(a)$ is determined from $\F_{t-1}$. Consider $Y_t(a)=\frac{\one\{A_t=a\}X_t}{p_t(a)}.$ Then

<div class="display-equation">
$$
\begin{align*}
\E[Y_t(a)\mid\F_{t-1}]
&=\frac{1}{p_t(a)}
\E[\one\{A_t=a\}X_t\mid\F_{t-1}]\\
&=\frac{1}{p_t(a)}
\Pbb(A_t=a\mid\F_{t-1})
\E[X_t\mid A_t=a,\F_{t-1}]\\
&=\frac{1}{p_t(a)}\,p_t(a)\mu_a\\
&=\mu_a.
\end{align*}
$$
</div>

 So $Y_t(a)-\mu_a$ is a martingale difference. This calculation is the backbone of inverse-propensity estimators in contextual bandits and adaptive experiments.

## Stopping Times: Rules That Do Not See the Future

A random time $\tau$ is a stopping time when, at every time $t$, we can decide whether $\tau\le t$ using only $\F_t$.

Formally,

<div class="display-equation">
$$
\{\tau\le t\}\in\F_t
\qquad\text{for every }t.
$$
</div>


The definition is easier to understand through examples.

### Examples

The first time a random walk reaches level $b$, $\tau=\inf\{t\ge 0:S_t\ge b\},$ is a stopping time. At time $t$, we can inspect $S_0,\ldots,S_t$ and know whether the boundary has already been crossed.

The first time a confidence interval excludes zero is also a stopping time. So is the first time a bandit algorithm eliminates an arm.

The time of the final maximum over a future horizon, $\tau=\argmax_{0\le s\le T}S_s,$ is not generally a stopping time. At time $t<T$, we cannot know whether a larger value will appear later.

> **Key idea.**
>
> A stopping time may react to everything already observed. It may not use tomorrow's data to decide that it should have stopped today.

### Stopped processes

Given a process $M_t$ and a stopping time $\tau$, define

<div class="display-equation">
$$
M_{t\wedge\tau}
=
M_{\min\{t,\tau\}}.
$$
</div>

 Before stopping, this follows the original process. After stopping, it stays frozen:

<div class="display-equation">
$$
M_{t\wedge\tau}
=
\begin{cases}
M_t, & t&lt;\tau,\\
M_\tau, & t\ge\tau.
\end{cases}
$$
</div>

 The stopped process is the clean object used in optional-stopping proofs.

## Optional Stopping, Proved Without Magic

We begin with the safest version of Doob's optional-stopping principle (Doob 1953; Williams 1991).

> **Result.**
>
> Let $(M_t)$ be a martingale and let $\tau$ be a stopping time bounded by a deterministic integer $n$: $\tau\le n\qquad\text{almost surely}.$ Then $\E[M_\tau]=\E[M_0].$ For a supermartingale, the equality becomes $\E[M_\tau]\le\E[M_0]$.

The proof is worth learning because the same shape appears throughout sequential analysis.

### Step 1: write the process as increments

Let $D_t=M_t-M_{t-1}.$ Then $M_t=M_0+\sum_{s=1}^{t}D_s.$ At the random time $\tau$, $M_\tau=M_0+\sum_{s=1}^{\tau}D_s.$ Because $\tau\le n$, we can rewrite the random-length sum as a fixed-length sum:

<div class="display-equation">
$$
\sum_{s=1}^{\tau}D_s
=
\sum_{s=1}^{n}\one\{\tau\ge s\}D_s.
$$
</div>

 Why? The indicator keeps exactly the increments that occur before stopping.

Hence

<div class="display-equation">
$$
M_\tau-M_0
=
\sum_{s=1}^{n}\one\{\tau\ge s\}D_s.
$$
</div>


### Step 2: notice what is known before the next increment

The event $\{\tau\ge s\}$ is the complement of $\{\tau\le s-1\}$. Since $\tau$ is a stopping time, $\{\tau\le s-1\}\in\F_{s-1}.$ Therefore $\one\{\tau\ge s\}$ is $\F_{s-1}$-measurable. It is a predictable multiplier.

### Step 3: take expectations one increment at a time

For every $s$,

<div class="display-equation">
$$
\begin{align*}
\E\bigl[\one\{\tau\ge s\}D_s\bigr]
&=\E\left[
\E\bigl[\one\{\tau\ge s\}D_s\mid\F_{s-1}\bigr]
\right]\\
&=\E\left[
\one\{\tau\ge s\}\E[D_s\mid\F_{s-1}]
\right]\\
&=\E\left[
\one\{\tau\ge s\}\cdot 0
\right]\\
&=0.
\end{align*}
$$
</div>

 Now sum:

<div class="display-equation">
$$
\begin{align*}
\E[M_\tau-M_0]
&=\E\left[
\sum_{s=1}^{n}\one\{\tau\ge s\}D_s
\right]\\
&=\sum_{s=1}^{n}
\E\bigl[\one\{\tau\ge s\}D_s\bigr]\\
&=0.
\end{align*}
$$
</div>

 Therefore

<div class="display-equation">
$$
\boxed{\E[M_\tau]=\E[M_0].}
$$
</div>


> **Proof pattern.**
>
> The proof has three moves:

<div class="display-equation">
$$
\text{random stopping}
\to
\text{predictable indicators}
\to
\text{zero conditional means}.
$$
</div>

 This is the discrete-time core of optional stopping.

### What happens for an unbounded stopping time?

If $\tau$ can be arbitrarily large, the conclusion needs additional control. Standard sufficient conditions include:

- $\tau$ is bounded;

- the stopped process $M_{t\wedge\tau}$ is uniformly bounded;

- or $\E[\tau]<\infty$ together with suitable control of the increments.

The exact theorem has several versions. Their common purpose is the same: prevent a vanishingly rare event from carrying a huge amount of expectation far out in time.

## Why the Conditions Matter

A clean counterexample shows what can go wrong.

Flip a fair coin repeatedly. Let $H_t=\one\{\text{the first }t\text{ tosses are all heads}\}$ and define

<div class="display-equation">
$$
M_t=2^tH_t,
\qquad M_0=1.
$$
</div>

 If a tail has already occurred, then $M_t=0$ forever. If all first $t$ tosses were heads, then at the next toss

<div class="display-equation">
$$
M_{t+1}
=
\begin{cases}
2^{t+1}, & \text{with probability }1/2,\\
0, & \text{with probability }1/2.
\end{cases}
$$
</div>

 Therefore, on the all-heads history,

<div class="display-equation">
$$
\begin{align*}
\E[M_{t+1}\mid\F_t]
&=\frac12\,2^{t+1}+\frac12\,0\\
&=2^t\\
&=M_t.
\end{align*}
$$
</div>

 On every history containing a tail, both sides are zero. Hence $(M_t)$ is a nonnegative martingale.

Let $\tau=\inf\{t\ge1:\text{toss }t\text{ is tails}\}.$ A tail eventually occurs with probability one, so

<div class="display-equation">
$$
M_\tau=0
\qquad\text{almost surely}.
$$
</div>

 Thus

<div class="display-equation">
$$
\E[M_\tau]=0
\neq
1=\E[M_0].
$$
</div>


Where did the expectation go?

For every finite $n$,

<div class="display-equation">
$$
M_{\tau\wedge n}
=
\begin{cases}
2^n, & \text{if the first }n\text{ tosses are all heads},\\
0, & \text{otherwise}.
\end{cases}
$$
</div>

 Hence

<div class="display-equation">
$$
\begin{align*}
\E[M_{\tau\wedge n}]
&=2^n\Pbb(\text{first }n\text{ tosses are all heads})\\
&=2^n\left(\frac12\right)^n\\
&=1.
\end{align*}
$$
</div>

 But for almost every path, $M_{\tau\wedge n}\longrightarrow 0.$ So

<div class="display-equation">
$$
\lim_{n\to\infty}\E[M_{\tau\wedge n}]
=1,
\qquad
\E\left[\lim_{n\to\infty}M_{\tau\wedge n}\right]
=0.
$$
</div>

 The limit and expectation cannot be exchanged. Rare all-heads paths become exponentially large and keep the finite-time expectation equal to one.

![Typical paths die at the first tail. The rare surviving path doubles, carrying the expectation at every finite time. This is why an unbounded stopping theorem needs more than almost-sure convergence.](/images/notes/assets/martingales/jackpot_martingale.webp)

*Typical paths die at the first tail. The rare surviving path doubles, carrying the expectation at every finite time. This is why an unbounded stopping theorem needs more than almost-sure convergence.*

> **Key idea.**
>
> Optional stopping is not a slogan that applies to every martingale and every random time. It is a theorem about when stopping and expectation can be interchanged safely.

## Ville's Inequality: One Bound for Every Time

Optional stopping becomes especially powerful when the process is nonnegative.

> **Result.**
>
> Let $(L_t)$ be a nonnegative supermartingale with $\E[L_0]\le 1.$ Then for every $\alpha\in(0,1)$,

<div class="display-equation">
$$
\Pbb\left(\sup_{t\ge0}L_t\ge\frac1\alpha\right)
\le\alpha.
$$
</div>


This is Ville's inequality (Ville 1939; Howard et al. 2020). Its proof is optional stopping in its cleanest form.

### Step-by-step proof

Define the first crossing time $\tau=\inf\left\{t\ge0:L_t\ge\frac1\alpha\right\}.$ For a fixed integer $n$, the truncated time $\tau\wedge n$ is bounded. Since $L_t$ is a supermartingale,

<div class="display-equation">
$$
\E[L_{\tau\wedge n}]
\le
\E[L_0]
\le1.
$$
</div>

 On the event $\{\tau\le n\}$, $L_{\tau\wedge n}=L_\tau\ge\frac1\alpha.$ Because $L_{\tau\wedge n}\ge0$ everywhere,

<div class="display-equation">
$$
\begin{align*}
1
&\ge\E[L_{\tau\wedge n}]\\
&\ge\E\left[L_{\tau\wedge n}\one\{\tau\le n\}\right]\\
&\ge\frac1\alpha\Pbb(\tau\le n).
\end{align*}
$$
</div>

 Therefore $\Pbb(\tau\le n)\le\alpha.$ As $n\to\infty$, the events $\{\tau\le n\}$ increase to $\{\tau<\infty\}$. Hence $\Pbb(\tau<\infty)\le\alpha.$ Equivalently,

<div class="display-equation">
$$
\boxed{
\Pbb\left(\exists t\ge0:L_t\ge\frac1\alpha\right)
\le\alpha.
}
$$
</div>


The important word is *exists*. One threshold controls every possible monitoring time at once.

### The betting interpretation

Think of $L_t$ as the wealth of a skeptical bettor under a null model. Under the null, the bettor cannot increase expected wealth. If the wealth ever reaches $1/\alpha$, the data contain strong evidence against the null. Ville's inequality says the null probability of ever reaching that level is at most $\alpha$.

A nonnegative process with this property is often called an *e-process*. The rule $\text{stop and reject when }L_t\ge 1/\alpha$ remains valid no matter how often the process is inspected.

## A Likelihood-Ratio Martingale

Consider Bernoulli observations. Under the null hypothesis, $H_0:p=p_0.$ Choose a fixed alternative $p_1$. Define the likelihood ratio

<div class="display-equation">
$$
L_t
=
\prod_{s=1}^{t}
\frac{p_1^{X_s}(1-p_1)^{1-X_s}}
     {p_0^{X_s}(1-p_0)^{1-X_s}}.
$$
</div>

 If $C_t=\sum_{s=1}^{t}X_s,$ then

<div class="display-equation">
$$
L_t
=
\left(\frac{p_1}{p_0}\right)^{C_t}
\left(\frac{1-p_1}{1-p_0}\right)^{t-C_t}.
$$
</div>


Under $H_0$, this is a martingale. To see it, write $L_t=L_{t-1}R_t,$ where

<div class="display-equation">
$$
R_t
=
\left(\frac{p_1}{p_0}\right)^{X_t}
\left(\frac{1-p_1}{1-p_0}\right)^{1-X_t}.
$$
</div>

 Then

<div class="display-equation">
$$
\begin{align*}
\E_{p_0}[L_t\mid\F_{t-1}]
&=L_{t-1}\E_{p_0}[R_t\mid\F_{t-1}]\\
&=L_{t-1}\left[
 p_0\frac{p_1}{p_0}
 +(1-p_0)\frac{1-p_1}{1-p_0}
\right]\\
&=L_{t-1}[p_1+(1-p_1)]\\
&=L_{t-1}.
\end{align*}
$$
</div>

 Therefore Ville's inequality gives

<div class="display-equation">
$$
\Pbb_{p_0}\left(\exists t:L_t\ge\frac1\alpha\right)
\le\alpha.
$$
</div>


This is a sequential test with no fixed horizon. It can be checked after every observation.

> **Think.**
>
> A likelihood ratio is not merely a score. Under the null it is a fair betting process. That martingale structure is what makes continuous monitoring valid.

## Exponential Supermartingales

Likelihood ratios are one route to sequential evidence. Concentration inequalities use another route: exponential transforms of centered sums.

Let $D_t$ be a martingale difference satisfying the conditional sub-Gaussian bound

<div class="display-equation">
$$
\E\left[e^{\lambda D_t}\mid\F_{t-1}\right]
\le
\exp\left(\frac{\lambda^2\sigma^2}{2}\right)
\qquad\text{for every }\lambda\in\R.
$$
</div>

 Define $S_t=\sum_{s=1}^{t}D_s$ and

<div class="display-equation">
$$
L_t(\lambda)
=
\exp\left(
\lambda S_t-\frac{\lambda^2\sigma^2 t}{2}
\right).
$$
</div>


### Why this is a supermartingale

Because $S_t=S_{t-1}+D_t$,

<div class="display-equation">
$$
\begin{align*}
L_t(\lambda)
&=\exp\left(
\lambda S_{t-1}-\frac{\lambda^2\sigma^2(t-1)}{2}
\right)
\exp\left(
\lambda D_t-\frac{\lambda^2\sigma^2}{2}
\right)\\
&=L_{t-1}(\lambda)
\exp\left(
\lambda D_t-\frac{\lambda^2\sigma^2}{2}
\right).
\end{align*}
$$
</div>

 Take conditional expectation:

<div class="display-equation">
$$
\begin{align*}
\E[L_t(\lambda)\mid\F_{t-1}]
&=L_{t-1}(\lambda)e^{-\lambda^2\sigma^2/2}
\E[e^{\lambda D_t}\mid\F_{t-1}]\\
&\le L_{t-1}(\lambda)e^{-\lambda^2\sigma^2/2}
e^{\lambda^2\sigma^2/2}\\
&=L_{t-1}(\lambda).
\end{align*}
$$
</div>

 Thus $L_t(\lambda)$ is a nonnegative supermartingale with $L_0(\lambda)=1$.

Ville's inequality now gives

<div class="display-equation">
$$
\Pbb\left(
\exists t:\
\lambda S_t-\frac{\lambda^2\sigma^2t}{2}
\ge\log\frac1\alpha
\right)
\le\alpha.
$$
</div>

 For $\lambda>0$, rearrange:

<div class="display-equation">
$$
\boxed{
\Pbb\left(
\exists t:\
S_t
\ge
\frac{\log(1/\alpha)}{\lambda}
+
\frac{\lambda\sigma^2t}{2}
\right)
\le\alpha.
}
$$
</div>

 This is a time-uniform linear boundary.

### Recovering a fixed-time bound

For one fixed time $t$, minimize

<div class="display-equation">
$$
\frac{\log(1/\alpha)}{\lambda}
+
\frac{\lambda\sigma^2t}{2}
$$
</div>

 over $\lambda>0$. Differentiate:

<div class="display-equation">
$$
-\frac{\log(1/\alpha)}{\lambda^2}
+
\frac{\sigma^2t}{2}
=0.
$$
</div>

 Hence

<div class="display-equation">
$$
\lambda^*
=
\sqrt{
\frac{2\log(1/\alpha)}{\sigma^2t}
}.
$$
</div>

 Substitute the optimizer:

<div class="display-equation">
$$
\begin{align*}
\frac{\log(1/\alpha)}{\lambda^*}
&=
\sigma\sqrt{\frac{t\log(1/\alpha)}{2}},\\
\frac{\lambda^*\sigma^2t}{2}
&=
\sigma\sqrt{\frac{t\log(1/\alpha)}{2}}.
\end{align*}
$$
</div>

 Adding the two terms,

<div class="display-equation">
$$
S_t
\le
\sigma\sqrt{2t\log(1/\alpha)}.
$$
</div>

 So

<div class="display-equation">
$$
\Pbb\left(
S_t\ge\sigma\sqrt{2t\log(1/\alpha)}
\right)
\le\alpha.
$$
</div>


There is one subtle point. The optimizing $\lambda^*$ depends on $t$. A different $t$ gives a different supermartingale. We may optimize for a predetermined time, but we cannot choose $\lambda$ after seeing which random time looked most favorable and pretend it was fixed all along.

Modern confidence-sequence methods solve this by mixing many $\lambda$ values or stitching together ranges of time (Howard et al. 2020, 2021). A simpler, more conservative construction uses a union bound.

## An Anytime Confidence Sequence from a Union Bound

Suppose $X_1,X_2,\ldots\in[0,1]$ are independent with common mean $\mu$. Hoeffding gives, at any fixed time $t$,

<div class="display-equation">
$$
\Pbb\left(
|\widehat\mu_t-\mu|\ge r
\right)
\le
2e^{-2tr^2}.
$$
</div>

 We want one event that holds for every $t$.

Allocate a small failure budget to each time:

<div class="display-equation">
$$
\delta_t
=
\frac{6\delta}{\pi^2t^2}.
$$
</div>

 Because $\sum_{t=1}^{\infty}\frac1{t^2}=\frac{\pi^2}{6},$ we have $\sum_{t=1}^{\infty}\delta_t=\delta.$ Choose $r_t$ so that $2e^{-2tr_t^2}=\delta_t.$ Then

<div class="display-equation">
$$
\begin{align*}
-2tr_t^2
&=\log\frac{\delta_t}{2},\\
2tr_t^2
&=\log\frac{2}{\delta_t},\\
r_t^2
&=\frac{1}{2t}\log\frac{2}{\delta_t}\\
&=\frac{1}{2t}
\log\left(
\frac{\pi^2t^2}{3\delta}
\right).
\end{align*}
$$
</div>

 Therefore

<div class="display-equation">
$$
\boxed{
r_t
=
\sqrt{
\frac{1}{2t}
\log\left(
\frac{\pi^2t^2}{3\delta}
\right)
}.
}
$$
</div>

 Now apply the union bound:

<div class="display-equation">
$$
\begin{align*}
\Pbb\left(
\exists t\ge1:
|\widehat\mu_t-\mu|>r_t
\right)
&\le
\sum_{t=1}^{\infty}
\Pbb\left(
|\widehat\mu_t-\mu|>r_t
\right)\\
&\le
\sum_{t=1}^{\infty}\delta_t\\
&=\delta.
\end{align*}
$$
</div>

 Equivalently,

<div class="display-equation">
$$
\boxed{
\Pbb\left(
\forall t\ge1:
\mu\in[\widehat\mu_t-r_t,\widehat\mu_t+r_t]
\right)
\ge1-\delta.
}
$$
</div>

 This sequence of intervals can be inspected continuously. It is wider than a fixed-time interval because it protects against every possible inspection time.

> **Key idea.**
>
> A fixed-time interval spends the whole error budget at one time. A confidence sequence spreads the budget across time, or uses a martingale construction that controls all times directly.

## The Bandit Version: Random Counts Are Not a Problem

Return to arm $a$. Recall

<div class="display-equation">
$$
S_a(t)
=
\sum_{s=1}^{t}I_{a,s}(X_s-\mu_a),
\qquad
N_a(t)=\sum_{s=1}^{t}I_{a,s}.
$$
</div>

 For rewards in $[0,1]$, Hoeffding's lemma gives

<div class="display-equation">
$$
\E\left[
\exp\bigl(\lambda(X_t-\mu_a)\bigr)
\mid A_t=a,\F_{t-1}
\right]
\le e^{\lambda^2/8}.
$$
</div>

 Define

<div class="display-equation">
$$
L_{a,t}(\lambda)
=
\exp\left(
\lambda S_a(t)
-
\frac{\lambda^2}{8}N_a(t)
\right).
$$
</div>

 We now verify the supermartingale property directly.

Let $I_t=I_{a,t}$. The one-step ratio is

<div class="display-equation">
$$
\frac{L_{a,t}(\lambda)}{L_{a,t-1}(\lambda)}
=
\exp\left(
\lambda I_t(X_t-\mu_a)
-
\frac{\lambda^2}{8}I_t
\right).
$$
</div>

 Since $I_t$ is known before $X_t$, there are two cases.

If $I_t=0$, $\frac{L_{a,t}(\lambda)}{L_{a,t-1}(\lambda)}=1.$ If $I_t=1$,

<div class="display-equation">
$$
\begin{align*}
\E\left[
\frac{L_{a,t}(\lambda)}{L_{a,t-1}(\lambda)}
\biggm|\F_{t-1}
\right]
&=
e^{-\lambda^2/8}
\E\left[e^{\lambda(X_t-\mu_a)}\mid\F_{t-1},A_t=a\right]\\
&\le
e^{-\lambda^2/8}e^{\lambda^2/8}\\
&=1.
\end{align*}
$$
</div>

 Thus

<div class="display-equation">
$$
\E[L_{a,t}(\lambda)\mid\F_{t-1}]
\le L_{a,t-1}(\lambda).
$$
</div>

 The algorithm may decide adaptively when to pull arm $a$. The process remains valid because the decision is made before the new reward arrives.

Ville's inequality gives

<div class="display-equation">
$$
\Pbb\left(
\exists t:\
S_a(t)
\ge
\frac{\log(1/\alpha)}{\lambda}
+
\frac{\lambda}{8}N_a(t)
\right)
\le\alpha.
$$
</div>

 This is the martingale form of a confidence statement indexed by a random number of pulls.

### A simple UCB radius valid over arms and pull counts

For a transparent construction, allocate failure probability across arms and sample counts:

<div class="display-equation">
$$
\delta_{a,n}
=
\frac{6\delta}{K\pi^2n^2}.
$$
</div>

 Then

<div class="display-equation">
$$
\sum_{a=1}^{K}\sum_{n=1}^{\infty}\delta_{a,n}
=\delta.
$$
</div>

 Using the two-sided Hoeffding bound at the $n$th observation of arm $a$,

<div class="display-equation">
$$
\Pbb\left(
|\widehat\mu_{a,n}-\mu_a|>r_{a,n}
\right)
\le\delta_{a,n}
$$
</div>

 with

<div class="display-equation">
$$
\boxed{
r_{a,n}
=
\sqrt{
\frac{1}{2n}
\log\left(
\frac{K\pi^2n^2}{3\delta}
\right)
}.
}
$$
</div>

 Therefore, with probability at least $1-\delta$, simultaneously for every arm and every sample count,

<div class="display-equation">
$$
|\widehat\mu_{a,n}-\mu_a|
\le r_{a,n}.
$$
</div>

 At calendar time $t$, simply insert the random count $N_a(t)$:

<div class="display-equation">
$$
U_a(t)
=
\widehat\mu_a(t)
+
r_{a,N_a(t)}.
$$
</div>

 The random sample size causes no difficulty because the guarantee was built to hold at all counts at once.

> **Proof pattern.**
>
> A common bandit proof follows this route:
>
> 1.  build a martingale from centered adaptive observations;
>
> 2.  turn it into a nonnegative supermartingale;
>
> 3.  stop it at the first bad event;
>
> 4.  use Ville or a union bound to control that event uniformly over time;
>
> 5.  analyze the algorithm on the resulting good event.

## A Small Experiment: Peeking, Stopping, and Evidence

The experiment has two parts.

First, we simulate a fair random walk and stop it when it reaches either $+12$ or $-12$, or at time $500$ if neither boundary has been reached. This stopping time is bounded. Optional stopping predicts that the mean stopped value remains approximately zero.

Second, we test a Bernoulli null hypothesis $H_0:p=0.5$ against the alternative $H_1:p=0.6.$ We compare three procedures over at most $300$ observations:

1.  an exact one-sided binomial test used only at the final horizon;

2.  the same fixed-time test checked after every observation, stopping at the first rejection;

3.  a likelihood-ratio e-process, stopping when $L_t\ge20=1/0.05$.

### Stopped random walks

![Sample paths of a fair random walk. Each path stops at the first boundary crossing or at the finite horizon. The rule reacts to the past but never sees the future.](/images/notes/assets/martingales/stopped_random_walks.webp)

*Sample paths of a fair random walk. Each path stops at the first boundary crossing or at the finite horizon. The rule reacts to the past but never sees the future.*

Across $100{,}000$ independent runs, the empirical mean of the stopped value was $-0.0198,$ close to the theoretical value zero. The upper and lower boundaries were reached with nearly equal probability.
| Quantity                          |    Estimate |
|:----------------------------------|------------:|
| Mean stopped value $\E[S_\tau]$ | $-0.0198$ |
| Probability of reaching $+12$   | $0.49039$ |
| Probability of reaching $-12$   | $0.49209$ |
| Mean stopping time                |  $141.84$ |

<p class="table-caption">Bounded optional-stopping experiment.</p>

### Repeated fixed-time testing

Under the null $p=0.5$, the final-horizon exact test rejected in about $4.74\%$ of runs, close to the nominal $5\%$. When the same fixed-time test was checked repeatedly, the false-alarm probability rose to $27.27\%$.

The e-process was checked at every time as well, but its false-alarm probability remained $4.29\%$. The difference is not how often the dashboard was opened. The difference is whether the probability statement was designed for continuous monitoring.
| Procedure | Null $p=0.5$ | Alternative $p=0.6$ |
|:---|:--:|:--:|
| Exact test at final horizon only | $0.04740$ | $0.96430$ |
| Exact fixed-time test, checked repeatedly | $0.27270$ | $0.98614$ |
| Likelihood-ratio e-process | $0.04288$ | $0.89798$ |

<p class="table-caption">Probability of stopping and rejecting by time $300$ over $50{,}000$ Monte Carlo runs.</p>

![Repeated fixed-time testing gains a little speed under the alternative by spending far more than the advertised false-alarm budget. The e-process can also stop early, while preserving the null guarantee.](/images/notes/assets/martingales/peeking_false_alarms.webp)

*Repeated fixed-time testing gains a little speed under the alternative by spending far more than the advertised false-alarm budget. The e-process can also stop early, while preserving the null guarantee.*

### Evidence paths

![Likelihood-ratio e-processes under the null and alternative. Under the null, the process may fluctuate upward, but Ville’s inequality controls the chance of ever crossing 1/α. Under the alternative, evidence tends to grow.](/images/notes/assets/martingales/eprocess_paths.webp)

*Likelihood-ratio e-processes under the null and alternative. Under the null, the process may fluctuate upward, but Ville’s inequality controls the chance of ever crossing 1/α. Under the alternative, evidence tends to grow.*

The likelihood-ratio process is not forced to increase. Evidence can rise and fall. The guarantee concerns the probability of ever crossing the threshold under the null.

## Core Python Implementation

The experiment uses exact binomial thresholds for the fixed-time test and a likelihood-ratio martingale for the anytime-valid test.

    import math
    import numpy as np
    from scipy.stats import binom


    def exact_thresholds(horizon, alpha=0.05, p0=0.5):
        """k[t] satisfies P_{p0}(Bin(t,p0) >= k[t]) <= alpha."""
        k = np.empty(horizon + 1, dtype=int)
        k[0] = 1
        for t in range(1, horizon + 1):
            # isf returns x with P(X > x) <= alpha.
            k[t] = int(binom.isf(alpha, t, p0)) + 1
        return k


    def run_tests(rng, p, runs=50_000, horizon=300,
                  alpha=0.05, p0=0.5, p1=0.6):
        thresholds = exact_thresholds(horizon, alpha, p0)
        successes = np.zeros(runs, dtype=int)

        repeated_alarm = np.zeros(runs, dtype=bool)
        e_alarm = np.zeros(runs, dtype=bool)
        log_e = np.zeros(runs, dtype=float)

        log_success = math.log(p1 / p0)
        log_failure = math.log((1 - p1) / (1 - p0))
        log_boundary = math.log(1 / alpha)

        for t in range(1, horizon + 1):
            x = rng.random(runs) < p
            successes += x

            # Incorrect for continuous monitoring: repeatedly apply
            # a test calibrated only for one fixed time.
            repeated_alarm |= successes >= thresholds[t]

            # Correct for continuous monitoring under H0:
            # update the likelihood-ratio martingale.
            log_e += np.where(x, log_success, log_failure)
            e_alarm |= log_e >= log_boundary

        fixed_alarm = successes >= thresholds[horizon]

        return {
            "fixed horizon": fixed_alarm.mean(),
            "repeated fixed-time": repeated_alarm.mean(),
            "e-process": e_alarm.mean(),
        }

The bounded random-walk stopping rule is equally direct.

    def stopped_random_walk(rng, horizon=500, boundary=12):
        value = 0
        for t in range(1, horizon + 1):
            value += 1 if rng.random() < 0.5 else -1
            if abs(value) >= boundary:
                return value, t
        return value, horizon

The full reproducible script supplied with this note generates all tables and figures. Appendix C includes the complete version used for the blog build.

## What Optional Stopping Does and Does Not Say

The phrase "optional stopping" can create two opposite misunderstandings.

The first is that adaptive stopping is invalid. That is false. Sequential inference is entirely possible when the probability statement is built for random times.

The second is that every statistic remains valid after stopping. That is also false. The theorem applies to a martingale or supermartingale under specific conditions. It does not automatically preserve the distribution of a $z$-statistic, the coverage of a fixed-time confidence interval, or the unbiasedness of a stopped ratio.

A useful checklist is:

1.  What is the filtration?

2.  Is the decision made before the next random outcome?

3.  What is the martingale difference?

4.  Is the process nonnegative if Ville's inequality is used?

5.  Is the stopping time bounded, or is another optional-stopping condition available?

6.  Is the desired statement fixed-time or time-uniform?

If these questions have clear answers, the proof is usually on solid ground.

## Where This Pattern Reappears

The same structure appears throughout modern bandit theory.

### UCB and arm elimination

Confidence events must hold while the algorithm chooses which arm to sample and when to eliminate it. Time-uniform or pull-count-uniform bounds make the random sampling schedule harmless.

### Best-arm identification

The stopping time is the output. The algorithm ends when evidence separates one arm from the others. A fixed-time guarantee is not enough; the error probability must remain controlled at the data-dependent stopping time.

### Linear and contextual bandits

The scalar martingale becomes a vector martingale. The random denominator becomes a design matrix. Self-normalized martingale inequalities control (Abbasi-Yadkori et al. 2011)

<div class="display-equation">
$$
\left\|\sum_{t=1}^{T}x_t\eta_t\right\|_{V_T^{-1}},
\qquad
V_T=\lambda I+\sum_{t=1}^{T}x_tx_t^\top.
$$
</div>

 The contexts $x_t$ may be chosen adaptively, as in modern contextual-bandit models developed at ETH and elsewhere (Kirschner and Krause 2019), but they are predictable: they are known before the new noise $\eta_t$ arrives.

### Anytime algorithms

An anytime algorithm does not need the horizon in advance. This makes random-time reasoning unavoidable. Work on anytime batched bandits and finite-time Thompson-sampling analysis, including research by Tianyuan Jin and collaborators (Jin et al. 2021; Jin et al. 2022), lives in this broader proof culture: control adaptive histories, construct high-probability events, and keep guarantees valid without knowing in advance when the process will end.

> **Key idea.**
>
> The mature way to read a sequential proof is not to ask only "which inequality is used?" Ask instead:

<div class="display-equation">
$$
\begin{gathered}
\text{What is predictable?}\qquad
\text{What is the martingale noise?}\\
\text{At which random time is the process stopped?}
\end{gathered}
$$
</div>

 Those three questions reveal the architecture of the argument.

## What to Carry Forward

A filtration is simply the past, recorded carefully.

A martingale difference is new noise that remains mean-zero after the past is used.

A predictable multiplier is a decision made before that noise arrives.

A stopping time is a rule that may use the past but not the future.

Optional stopping turns random-length sums into predictable, fixed-length sums.

Ville's inequality turns a nonnegative supermartingale into a statement that is valid at every monitoring time.

The entire chain can be summarized as

<div class="display-equation">
$$
\boxed{
\begin{gathered}
\text{history}
\to
\text{predictable action}
\to
\text{martingale noise}\\
\to
\text{nonnegative supermartingale}
\to
\text{time-uniform guarantee}
\end{gathered}
}
$$
</div>

 That chain is one of the central proof paradigms of bandits, sequential testing, and adaptive data analysis.

## Appendix A. Formula Sheet
| Object | Formula |
|:---|:---|
| Filtration | $\F_0\subseteq\F_1\subseteq\cdots$ |
| Martingale | $\E[M_t\mid\F_{t-1}]=M_{t-1}$ |
| Martingale difference | $D_t=M_t-M_{t-1}$ and $\E[D_t\mid\F_{t-1}]=0$ |
| Predictable multiplier | $H_t\in\F_{t-1}$ implies $\E[H_tD_t\mid\F_{t-1}]=0$ |
| Stopping time | $\{\tau\le t\}\in\F_t$ for every $t$ |
| Stopped process | $M_{t\wedge\tau}=M_{\min\{t,\tau\}}$ |
| Bounded optional stopping | $\tau\le n\Rightarrow\E[M_\tau]=\E[M_0]$ |
| Ville’s inequality | $\Pbb(\sup_tL_t\ge1/\alpha)\le\alpha$ for nonnegative supermartingale $L_t$ with $\E L_0\le1$ |
| Exponential supermartingale | $L_t(\lambda)=\exp(\lambda S_t-\lambda^2\sigma^2t/2)$ |
| Bernoulli likelihood ratio | $L_t=(p_1/p_0)^{C_t}((1-p_1)/(1-p_0))^{t-C_t}$ |
| Arm-specific martingale | $S_a(t)=\sum_{s\le t}\one\{A_s=a\}(X_s-\mu_a)$ |
| Arm pull count | $N_a(t)=\sum_{s\le t}\one\{A_s=a\}$ |
| Anytime Hoeffding radius | $\sqrt{\log(\pi^2t^2/(3\delta))/(2t)}$ |

<p class="table-caption">Core formulas.</p>

## Appendix B. Notation Table
| Symbol | Meaning |
|:---|:---|
| $\F_t$ | all information available after round $t$ |
| $A_t$ | action chosen at round $t$ |
| $X_t$ | reward observed at round $t$ |
| $\mu_a$ | mean reward of arm $a$ |
| $M_t$ | martingale or supermartingale |
| $D_t$ | martingale difference $M_t-M_{t-1}$ |
| $H_t$ | predictable quantity known before round $t$ |
| $\tau$ | stopping time |
| $S_t$ | centered cumulative sum |
| $L_t$ | nonnegative test martingale or e-process |
| $\alpha$ | allowed probability of ever crossing the rejection threshold |
| $I_{a,t}$ | indicator $\one\{A_t=a\}$ |
| $N_a(t)$ | number of pulls of arm $a$ by time $t$ |
| $\widehat\mu_a(t)$ | empirical mean of arm $a$ |

<p class="table-caption">Notation.</p>

## Further Reading

The presentation follows the selective, example-first martingale style associated with Williams' Cambridge text and the bandit-oriented development in Lattimore and Szepesvari. For sharper time-uniform boundaries, confidence sequences, and mixture constructions, the work of Howard, Ramdas, McAuliffe, and Sekhon is the natural next step. For high-dimensional bandits, self-normalized vector martingales lead to the confidence ellipsoids used in linear UCB methods.

## Appendix C. Full Experiment Script

``` {.python style="blogcode" language="Python"}
"""Reproducible experiments for
'Martingales and Optional Stopping: Learning at a Random Time'.

The script creates:
  - stopped_random_walks.pdf/png
  - peeking_false_alarms.pdf/png
  - eprocess_paths.pdf/png
  - jackpot_martingale.pdf/png
  - martingale_results.csv

All simulations use a fixed random seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binom


OUT = Path(__file__).resolve().parent
SEED = 20260618


@dataclass(frozen=True)
class Config:
    alpha: float = 0.05
    p0: float = 0.50
    p1: float = 0.60
    horizon: int = 300
    runs: int = 50000
    random_walk_runs: int = 100000
    random_walk_horizon: int = 500
    random_walk_boundary: int = 12


def first_crossing_stopped_random_walk(
    rng: np.random.Generator,
    runs: int,
    horizon: int,
    boundary: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate symmetric random walks stopped at +/- boundary or horizon."""
    positions = np.zeros(runs, dtype=np.int32)
    stopped = np.zeros(runs, dtype=bool)
    tau = np.full(runs, horizon, dtype=np.int32)

    for t in range(1, horizon + 1):
        active = ~stopped
        if not np.any(active):
            break
        steps = np.where(rng.random(np.sum(active)) < 0.5, -1, 1)
        positions[active] += steps
        crossed = active & (np.abs(positions) >= boundary)
        tau[crossed] = t
        stopped[crossed] = True

    return positions, tau, stopped


def sample_stopped_paths(
    rng: np.random.Generator,
    n_paths: int,
    horizon: int,
    boundary: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    paths: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_paths):
        s = 0
        values = [0]
        times = [0]
        for t in range(1, horizon + 1):
            s += 1 if rng.random() < 0.5 else -1
            values.append(s)
            times.append(t)
            if abs(s) >= boundary:
                break
        paths.append((np.asarray(times), np.asarray(values)))
    return paths


def exact_one_sided_thresholds(horizon: int, alpha: float, p0: float) -> np.ndarray:
    """k_t such that P_{p0}(Bin(t,p0) >= k_t) <= alpha."""
    thresholds = np.empty(horizon + 1, dtype=np.int32)
    thresholds[0] = 1
    for t in range(1, horizon + 1):
        # scipy's isf returns x with P(X > x) <= alpha, so k = x + 1.
        thresholds[t] = int(binom.isf(alpha, t, p0)) + 1
    return thresholds


def sequential_testing_rates(
    rng: np.random.Generator,
    p: float,
    cfg: Config,
    thresholds: np.ndarray,
) -> dict[str, float]:
    """Compare fixed-horizon, repeated fixed-time, and e-process tests."""
    n = cfg.runs
    successes = np.zeros(n, dtype=np.int32)
    repeated_alarm = np.zeros(n, dtype=bool)
    e_alarm = np.zeros(n, dtype=bool)
    log_e = np.zeros(n, dtype=float)
    log_threshold = math.log(1.0 / cfg.alpha)
    log_success = math.log(cfg.p1 / cfg.p0)
    log_failure = math.log((1.0 - cfg.p1) / (1.0 - cfg.p0))

    for t in range(1, cfg.horizon + 1):
        x = rng.random(n) < p
        successes += x
        repeated_alarm |= successes >= thresholds[t]
        log_e += np.where(x, log_success, log_failure)
        e_alarm |= log_e >= log_threshold

    fixed_alarm = successes >= thresholds[cfg.horizon]
    return {
        "fixed_horizon": float(np.mean(fixed_alarm)),
        "repeated_fixed_time": float(np.mean(repeated_alarm)),
        "e_process": float(np.mean(e_alarm)),
    }


def make_eprocess_paths(
    rng: np.random.Generator,
    p: float,
    p0: float,
    p1: float,
    horizon: int,
    n_paths: int,
) -> np.ndarray:
    x = rng.random((n_paths, horizon)) < p
    increments = np.where(x, math.log(p1 / p0), math.log((1 - p1) / (1 - p0)))
    return np.exp(np.cumsum(increments, axis=1))


def jackpot_paths(rng: np.random.Generator, n_paths: int, horizon: int) -> np.ndarray:
    """M_t = 2^t 1{the first t tosses are all heads}."""
    values = np.ones((n_paths, horizon + 1), dtype=float)
    alive = np.ones(n_paths, dtype=bool)
    for t in range(1, horizon + 1):
        heads = rng.random(n_paths) < 0.5
        alive &= heads
        values[:, t] = np.where(alive, 2.0**t, 0.0)
    return values


def plot_stopped_random_walks(paths: list[tuple[np.ndarray, np.ndarray]], boundary: int) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for times, values in paths:
        ax.plot(times, values, linewidth=1.0, alpha=0.8)
        ax.scatter(times[-1], values[-1], s=15)
    ax.axhline(boundary, linestyle="--", linewidth=1.2)
    ax.axhline(-boundary, linestyle="--", linewidth=1.2)
    ax.axhline(0, linewidth=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("martingale value $S_t$")
    ax.set_title("A fair random walk stopped when it reaches a boundary")
    ax.text(0.985, 0.955, fr"boundaries $\pm {boundary}$", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "stopped_random_walks.pdf", bbox_inches="tight")
    fig.savefig(OUT / "stopped_random_walks.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_testing_rates(null_rates: dict[str, float], alt_rates: dict[str, float]) -> None:
    labels = ["fixed horizon", "peek repeatedly", "e-process"]
    keys = ["fixed_horizon", "repeated_fixed_time", "e_process"]
    x = np.arange(len(labels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.bar(x - width / 2, [null_rates[k] for k in keys], width, label="null: $p=0.5$")
    ax.bar(x + width / 2, [alt_rates[k] for k in keys], width, label="alternative: $p=0.6$")
    ax.axhline(0.05, linestyle="--", linewidth=1.1, label="nominal level $0.05$")
    ax.set_xticks(x, labels)
    ax.set_ylabel("probability of stopping and rejecting")
    ax.set_ylim(0, 1.12)
    ax.set_title("Repeated use of a fixed-time test creates false alarms", pad=34)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.105), fontsize=8)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "peeking_false_alarms.pdf", bbox_inches="tight")
    fig.savefig(OUT / "peeking_false_alarms.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_eprocess_paths(null_paths: np.ndarray, alt_paths: np.ndarray, alpha: float) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    t = np.arange(1, null_paths.shape[1] + 1)
    for i, path in enumerate(null_paths):
        ax.plot(t, path, linewidth=0.9, alpha=0.55, color="0.55", label="null $p=0.5$" if i == 0 else None)
    for i, path in enumerate(alt_paths):
        ax.plot(t, path, linewidth=1.1, alpha=0.75, color="C0", label="alternative $p=0.6$" if i == 0 else None)
    ax.axhline(1 / alpha, linestyle="--", linewidth=1.2, color="C1", label="$1/\\alpha$")
    ax.set_yscale("log")
    ax.set_xlabel("time")
    ax.set_ylabel("e-process $L_t$ (log scale)")
    ax.set_title("Evidence may be monitored continuously without changing the threshold")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "eprocess_paths.pdf", bbox_inches="tight")
    fig.savefig(OUT / "eprocess_paths.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_jackpot_martingale(paths: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.25))
    t = np.arange(paths.shape[1])
    for path in paths:
        ax.step(t, path, where="post", linewidth=1.2, alpha=0.8)
    ax.set_yscale("symlog", linthresh=0.5)
    ax.set_xlabel("time")
    ax.set_ylabel(r"$M_t = 2^t \,\mathbf{1}\{\mathrm{all\ heads\ so\ far}\}$")
    ax.set_title("A martingale whose infinite stopping limit loses its mean")
    ax.text(0.98, 0.92, "every finite-time mean is 1\nfirst tail sends the path to 0",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "jackpot_martingale.pdf", bbox_inches="tight")
    fig.savefig(OUT / "jackpot_martingale.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(SEED)

    positions, tau, hit = first_crossing_stopped_random_walk(
        rng,
        cfg.random_walk_runs,
        cfg.random_walk_horizon,
        cfg.random_walk_boundary,
    )
    paths = sample_stopped_paths(
        rng,
        n_paths=14,
        horizon=120,
        boundary=cfg.random_walk_boundary,
    )
    plot_stopped_random_walks(paths, cfg.random_walk_boundary)

    thresholds = exact_one_sided_thresholds(cfg.horizon, cfg.alpha, cfg.p0)
    null_rates = sequential_testing_rates(rng, cfg.p0, cfg, thresholds)
    alt_rates = sequential_testing_rates(rng, cfg.p1, cfg, thresholds)
    plot_testing_rates(null_rates, alt_rates)

    null_paths = make_eprocess_paths(rng, cfg.p0, cfg.p0, cfg.p1, 250, 8)
    alt_paths = make_eprocess_paths(rng, cfg.p1, cfg.p0, cfg.p1, 250, 8)
    plot_eprocess_paths(null_paths, alt_paths, cfg.alpha)

    jackpot = jackpot_paths(rng, n_paths=10, horizon=12)
    plot_jackpot_martingale(jackpot)

    upper_hits = np.mean(positions == cfg.random_walk_boundary)
    lower_hits = np.mean(positions == -cfg.random_walk_boundary)
    results = pd.DataFrame(
        [
            {"experiment": "bounded stopping", "metric": "mean stopped value", "value": float(np.mean(positions))},
            {"experiment": "bounded stopping", "metric": "upper-boundary probability", "value": float(upper_hits)},
            {"experiment": "bounded stopping", "metric": "lower-boundary probability", "value": float(lower_hits)},
            {"experiment": "bounded stopping", "metric": "mean stopping time", "value": float(np.mean(tau))},
            {"experiment": "null p=0.5", "metric": "fixed-horizon rejection", "value": null_rates["fixed_horizon"]},
            {"experiment": "null p=0.5", "metric": "repeated fixed-time rejection", "value": null_rates["repeated_fixed_time"]},
            {"experiment": "null p=0.5", "metric": "e-process rejection", "value": null_rates["e_process"]},
            {"experiment": "alternative p=0.6", "metric": "fixed-horizon rejection", "value": alt_rates["fixed_horizon"]},
            {"experiment": "alternative p=0.6", "metric": "repeated fixed-time rejection", "value": alt_rates["repeated_fixed_time"]},
            {"experiment": "alternative p=0.6", "metric": "e-process rejection", "value": alt_rates["e_process"]},
        ]
    )
    results.to_csv(OUT / "martingale_results.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
```


Abbasi-Yadkori, Yasin, Dávid Pál, and Csaba Szepesvári. 2011. "Improved Algorithms for Linear Stochastic Bandits." *Advances in Neural Information Processing Systems*.


Doob, Joseph L. 1953. *Stochastic Processes*. Wiley.


Freedman, David A. 1975. "On Tail Probabilities for Martingales." *The Annals of Probability* 3 (1): 100--118.


Howard, Steven R., Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. 2020. "Time-Uniform Chernoff Bounds via Nonnegative Supermartingales." *Probability Surveys* 17: 257--317.


Howard, Steven R., Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. 2021. "Time-Uniform, Nonparametric, Nonasymptotic Confidence Sequences." *The Annals of Statistics* 49 (2): 1055--80.


Jin, Tianyuan, Jing Tang, Pan Xu, Keli Huang, Xiaokui Xiao, and Quanquan Gu. 2021. "Almost Optimal Anytime Algorithm for Batched Multi-Armed Bandits." *Proceedings of the 38th International Conference on Machine Learning*.


Jin, T., P. Xu, X. Xiao, and A. Anandkumar. 2022. *Finite-Time Regret of Thompson Sampling Algorithms for Exponential Family Multi-Armed Bandits*.


Kirschner, Johannes, and Andreas Krause. 2019. "Stochastic Bandits with Context Distributions." *Advances in Neural Information Processing Systems*.


Lattimore, T., and C. Szepesvari. 2020. *Bandit Algorithms*. Cambridge University Press.


Ville, Jean. 1939. *Étude Critique de La Notion de Collectif*. Gauthier-Villars.


Williams, David. 1991. *Probability with Martingales*. Cambridge University Press.
