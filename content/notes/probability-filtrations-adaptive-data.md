---
title: "Probability, Filtrations, and Adaptive Data"
subtitle: "The Hidden Grammar of Bandit Proofs"
summary: "Filtrations, stopping, post-selection effects, and adaptive observations."
description: "Filtrations, stopping, post-selection effects, and adaptive observations."
date: 2026-06-18
lastmod: 2026-06-18
weight: 30
libraryFolder: "probability-statistics"
libraryFolderName: "概率与统计基础"
libraryFolderColor: 2
tags: ["Probability", "Filtrations", "Adaptive Data"]
draft: false
ShowToc: false
hideMeta: true
---

## The real problem: data is not waiting on a table

In supervised learning, the data set usually arrives first. We open the file, see rows of examples, and then fit a model. The model may be complicated, but the data has already been collected.

Bandit learning reverses the order. The algorithm acts first. The data appears because of the action.

That one reversal is the source of almost all probability in bandit theory.


The hard part is not that rewards are random. The hard part is that the learner chooses which randomness it will see. A bad early choice can starve an arm of data. A lucky early reward can make a bad arm look good. A confidence interval that is valid at one fixed time can become misleading if we keep peeking and stop when it looks favorable.

This note is about the grammar that prevents those mistakes.

> **Key idea.**
>
> A bandit proof does not start with "let $\Omega$ be a probability space." It starts with a simple rule of order:

<div class="display-equation">
$$
\text{past} \quad \longrightarrow \quad \text{action} \quad \longrightarrow \quad \text{reward}.
$$
</div>

 The formal probability notation is just a clean way to respect this order.

## Random variables: numbers before they are seen

A random variable is a number whose value has not yet been revealed.

For a Bernoulli reward, the number is either $0$ or $1$. If the click probability is $p$, then

<div class="display-equation">
$$
X=\begin{cases}
1, & \text{with probability }p,\\
0, & \text{with probability }1-p.
\end{cases}
$$
</div>


The expectation is the long-run average value. Here there are only two possible values, so the calculation is direct:

<div class="numbered-equation" id="eq:bernoulli_mean">
$$
\begin{align}
\E[X]
&= 1\cdot \Pp(X=1) + 0\cdot \Pp(X=0) \\
&= 1\cdot p + 0\cdot (1-p) \\
&= p.
\end{align}
$$
<span class="equation-number" aria-label="Equation 1">(1)</span>
</div>


If we observe $n$ independent rewards $X_1,\ldots,X_n$, the sample average is $\widehat p_n = \frac{1}{n}\sum_{s=1}^n X_s.$

The sample average is random before the experiment is run. Its expectation is easy to compute:

<div class="numbered-equation" id="eq:sample_mean_unbiased">
$$
\begin{align}
\E[\widehat p_n]
&= \E\left[\frac{1}{n}\sum_{s=1}^n X_s\right] \\
&= \frac{1}{n}\sum_{s=1}^n \E[X_s] \\
&= \frac{1}{n}\sum_{s=1}^n p \\
&= p.
\end{align}
$$
<span class="equation-number" aria-label="Equation 2">(2)</span>
</div>


This is the first comfort: the sample average points to the right target on average.

But an average can point in the right direction and still wobble. Probability inequalities measure the wobble.

## Error bars: turning randomness into a usable statement

For Bernoulli rewards, Hoeffding's inequality says

<div class="numbered-equation" id="eq:hoeffding">
$$
\Pp\left(\left|\widehat p_n-p\right|\geq \varepsilon\right)
\leq 2\exp(-2n\varepsilon^2).
$$
<span class="equation-number" aria-label="Equation 3">(3)</span>
</div>


This formula is not meant to be memorized as decoration. It answers a practical question: how large should the error bar be if we want failure probability at most $\delta$?

Start from the right-hand side and set it equal to $\delta$:

<div class="numbered-equation" id="eq:hoeffding_radius">
$$
\begin{align}
2\exp(-2n\varepsilon^2) &= \delta,\\
\exp(-2n\varepsilon^2) &= \frac{\delta}{2},\\
-2n\varepsilon^2 &= \log\left(\frac{\delta}{2}\right),\\
2n\varepsilon^2 &= \log\left(\frac{2}{\delta}\right),\\
\varepsilon^2 &= \frac{\log(2/\delta)}{2n},\\
\varepsilon &= \sqrt{\frac{\log(2/\delta)}{2n}}.
\end{align}
$$
<span class="equation-number" aria-label="Equation 4">(4)</span>
</div>


So we may write the fixed-time confidence statement as

<div class="numbered-equation" id="eq:fixed_time_ci">
$$
\Pp\left(
\left|\widehat p_n-p\right|
\leq
\sqrt{\frac{\log(2/\delta)}{2n}}
\right)
\geq 1-\delta.
$$
<span class="equation-number" aria-label="Equation 5">(5)</span>
</div>


> **Think.**
>
> The phrase "fixed time" is doing real work. The statement above is about one sample size $n$ chosen before looking at the data. It is not automatically a promise that the same error bar works after repeatedly looking at the data and stopping at a convenient time.

## A first warning: peeking changes the game

Suppose a coin is fair, so $p=1/2$. If we toss it exactly $200$ times and look once at the end, a standard error bar behaves roughly as expected.

But if we look after every toss and stop the moment the running average looks high, the final number is no longer an ordinary fixed-time average. We have selected a time because the noise looked favorable.

This is not a philosophical issue. It appears directly in simulation.

![The fixed-time mean is centered near the true value 0.5. The adaptively stopped mean is shifted upward because we stop when the noise is favorable.](/images/notes/assets/filtrations/figures/fixed_vs_stopped_means.webp)

*The fixed-time mean is centered near the true value 0.5. The adaptively stopped mean is shifted upward because we stop when the noise is favorable.*

The same idea appears when we keep checking an error bar many times.

![A normal-looking error bar is reasonable for one final look, but repeated peeking with the same bar creates many false alarms. A union-bound bar is crude but safe.](/images/notes/assets/filtrations/figures/false_alarm_comparison.webp)

*A normal-looking error bar is reasonable for one final look, but repeated peeking with the same bar creates many false alarms. A union-bound bar is crude but safe.*

The experiment used $20{,}000$ runs of a fair Bernoulli process with horizon $T=200$. The stopped rule begins checking after $20$ samples and stops when the running mean first exceeds $0.56$.
| Quantity                                                 | Value |
|:---------------------------------------------------------|------:|
| True Bernoulli mean                                      | 0.500 |
| Probability of stopping early                            | 0.555 |
| Average fixed-time mean                                  | 0.500 |
| Average stopped mean                                     | 0.548 |
| One final look false alarm with normal bar               | 0.057 |
| Many peeks false alarm with same normal bar              | 0.414 |
| Many peeks false alarm with union bar                    | 0.000 |
| Post-selection average of one fixed arm                  | 0.500 |
| Post-selection average of empirical winner among 20 arms | 0.705 |

<p class="table-caption">The important number is not the exact decimal. The important lesson is that adapting to the observed noise changes the distribution of what we report.</p>


## Histories: the learner's notebook

In a bandit problem, the learner accumulates a notebook.

At the beginning of round $t$, the notebook contains all past actions and rewards:

<div class="display-equation">
$$
H_{t-1}=(A_1,R_1,A_2,R_2,\ldots,A_{t-1},R_{t-1}).
$$
</div>


Then the learner chooses an action using only this notebook:

<div class="display-equation">
$$
A_t = \pi_t(H_{t-1}).
$$
</div>


Then the environment reveals one reward:

<div class="display-equation">
$$
R_t \sim \text{reward distribution of arm }A_t.
$$
</div>


Then the notebook becomes

<div class="display-equation">
$$
H_t=(H_{t-1},A_t,R_t).
$$
</div>


A filtration is just the mathematical name for this growing notebook. We write

<div class="display-equation">
$$
\F_t = \text{all information contained in }H_t.
$$
</div>


So $\F_{t-1}$ means "everything known before choosing and observing at time $t$."

> **Key idea.**
>
> A filtration is not a mysterious object. It is the learner's notebook, ordered by time. The only rule is that the algorithm may read old pages before acting, but it cannot read the next reward before choosing the next action.

## Conditional expectation: averaging after the past is known

The ordinary expectation $\E[X]$ averages before anything is known. The conditional expectation $\E[X\mid \F_{t-1}]$ averages after the past notebook has been opened.

For a bandit reward, suppose arm $i$ has mean $\mu_i$. If the learner chooses $A_t=i$, then

<div class="display-equation">
$$
\E[R_t \mid \F_{t-1}, A_t=i] = \mu_i.
$$
</div>


But $A_t$ itself is chosen using $\F_{t-1}$, so once the past is known, $A_t$ is also known. Hence

<div class="numbered-equation" id="eq:conditional_reward_mean">
$$
\E[R_t \mid \F_{t-1}] = \mu_{A_t}.
$$
<span class="equation-number" aria-label="Equation 6">(6)</span>
</div>


Now define the one-step noise

<div class="display-equation">
$$
\eta_t = R_t - \mu_{A_t}.
$$
</div>


Using [Eq. (6)](#eq:conditional_reward_mean),

<div class="numbered-equation" id="eq:martingale_difference">
$$
\begin{align}
\E[\eta_t\mid \F_{t-1}]
&= \E[R_t-\mu_{A_t}\mid \F_{t-1}] \\
&= \E[R_t\mid \F_{t-1}] - \mu_{A_t} \\
&= \mu_{A_t} - \mu_{A_t} \\
&= 0.
\end{align}
$$
<span class="equation-number" aria-label="Equation 7">(7)</span>
</div>


This tiny equation is everywhere in bandit theory. It says: after we condition on the past, the remaining reward noise is fair.

## Martingale differences: fair noise after conditioning

A sequence $\eta_t$ satisfying

<div class="display-equation">
$$
\E[\eta_t\mid \F_{t-1}]=0
$$
</div>

 is called a martingale difference sequence.

The name is less important than the picture. At time $t$, the learner may be very clever. It may choose $A_t$ using every previous reward. But after the action is fixed, the new noise still has mean zero.


This is why adaptive data is not hopeless. The actions are adaptive, but the reward noise remains conditionally centered.

## Adaptive sample means

For arm $i$, define the number of times it has been pulled by time $t$:

<div class="display-equation">
$$
N_i(t)=\sum_{s=1}^t \one\{A_s=i\}.
$$
</div>


Define the empirical mean of arm $i$:

<div class="display-equation">
$$
\widehat\mu_i(t)=
\frac{\sum_{s=1}^t \one\{A_s=i\}R_s}{N_i(t)}
\quad\text{when }N_i(t)>0.
$$
</div>


The numerator has two pieces:

<div class="display-equation">
$$
\begin{align}
\sum_{s=1}^t \one\{A_s=i\}R_s
&= \sum_{s=1}^t \one\{A_s=i\}(\mu_i + R_s-\mu_i) \\
&= \mu_i\sum_{s=1}^t \one\{A_s=i\}
+ \sum_{s=1}^t \one\{A_s=i\}(R_s-\mu_i) \\
&= \mu_i N_i(t)
+ \sum_{s=1}^t \one\{A_s=i\}(R_s-\mu_i).
\end{align}
$$
</div>


Divide by $N_i(t)$:

<div class="numbered-equation" id="eq:adaptive_empirical_error">
$$
\begin{align}
\widehat\mu_i(t)-\mu_i
&=
\frac{1}{N_i(t)}
\sum_{s=1}^t \one\{A_s=i\}(R_s-\mu_i).
\end{align}
$$
<span class="equation-number" aria-label="Equation 8">(8)</span>
</div>


The error is still an average of centered noise terms, but the number of terms is random and chosen adaptively. That is the reason for filtrations and anytime confidence bounds.

The key centering step is

<div class="numbered-equation" id="eq:adaptive_centering">
$$
\begin{align}
\E\left[\one\{A_s=i\}(R_s-\mu_i)\mid \F_{s-1}\right]
&= \one\{A_s=i\}\,\E\left[(R_s-\mu_i)\mid \F_{s-1},A_s=i\right] \\
&= \one\{A_s=i\}\cdot 0 \\
&= 0.
\end{align}
$$
<span class="equation-number" aria-label="Equation 9">(9)</span>
</div>


The indicator $\one\{A_s=i\}$ is allowed because the action is chosen from the past. It is known at the moment we average over the new reward.

> **Think.**
>
> This is the essence of adaptive data: the learner can choose which arm to sample, but it cannot choose the new random fluctuation after the arm has been chosen. That remaining fluctuation is still fair after conditioning on the past.

## The union bound: a simple way to be right many times

The union bound says that the probability that at least one bad event happens is at most the sum of the probabilities of the bad events.

For events $E_1,\ldots,E_m$,

<div class="numbered-equation" id="eq:union_bound">
$$
\Pp\left(\bigcup_{j=1}^m E_j\right)\leq \sum_{j=1}^m \Pp(E_j).
$$
<span class="equation-number" aria-label="Equation 10">(10)</span>
</div>


A one-line proof uses indicators. For every outcome,

<div class="display-equation">
$$
\one\left\{\bigcup_{j=1}^m E_j\right\}
\leq
\sum_{j=1}^m \one\{E_j\}.
$$
</div>


Take expectations:

<div class="display-equation">
$$
\begin{align}
\Pp\left(\bigcup_{j=1}^m E_j\right)
&= \E\left[\one\left\{\bigcup_{j=1}^m E_j\right\}\right] \\
&\leq \E\left[\sum_{j=1}^m \one\{E_j\}\right] \\
&= \sum_{j=1}^m \E[\one\{E_j\}] \\
&= \sum_{j=1}^m \Pp(E_j).
\end{align}
$$
</div>


Now suppose we want confidence intervals for $K$ arms and $T$ possible times. There are at most $KT$ things we want to be simultaneously correct about.

Give each one failure probability

<div class="display-equation">
$$
\delta' = \frac{\delta}{KT}.
$$
</div>


For each arm-time pair, Hoeffding gives

<div class="display-equation">
$$
\Pp\left(
|\widehat\mu_i(t)-\mu_i|>
\sqrt{\frac{\log(2/\delta')}{2N_i(t)}}
\right)
\leq \delta'.
$$
</div>


Substitute $\delta'=\delta/(KT)$:

<div class="numbered-equation" id="eq:union_radius">
$$
\begin{align}
\sqrt{\frac{\log(2/\delta')}{2N_i(t)}}
&= \sqrt{\frac{\log(2KT/\delta)}{2N_i(t)}}.
\end{align}
$$
<span class="equation-number" aria-label="Equation 11">(11)</span>
</div>


Then the union bound says that all these intervals are correct together with probability at least $1-\delta$.

This is crude. Modern analyses often use sharper time-uniform tools. But the union-bound version is the first clean proof pattern, and it already explains why logarithms appear in bandit regret bounds.

## From probability to UCB

UCB is built from one sentence:

> Choose the arm whose plausible optimistic value is largest.

At time $t$, define the radius

<div class="display-equation">
$$
\operatorname{rad}_i(t)
=\sqrt{\frac{\log(2KT/\delta)}{2N_i(t)}}.
$$
</div>


The optimistic index is

<div class="display-equation">
$$
\operatorname{UCB}_i(t)
=\widehat\mu_i(t)+\operatorname{rad}_i(t).
$$
</div>


The algorithm chooses

<div class="display-equation">
$$
A_{t+1}\in \argmax_i \operatorname{UCB}_i(t).
$$
</div>


The confidence event is

<div class="numbered-equation" id="eq:good_event">
$$
\mathcal{G}=
\left\{
\forall i,\forall t:\ |
\widehat\mu_i(t)-\mu_i|
\leq \operatorname{rad}_i(t)
\right\}.
$$
<span class="equation-number" aria-label="Equation 12">(12)</span>
</div>


On $\mathcal{G}$, every empirical mean is close to its truth at every relevant time. That single event drives the UCB proof.

### The pull-count argument, line by line

Let $i^*$ be an optimal arm and let $i$ be a suboptimal arm. Define the gap

<div class="display-equation">
$$
\Delta_i=\mu_{i^*}-\mu_i>0.
$$
</div>


Suppose UCB pulls arm $i$ at some time. Since UCB chose $i$ instead of $i^*$,

<div class="numbered-equation" id="eq:ucb_chose_i">
$$
\widehat\mu_i + \operatorname{rad}_i
\geq
\widehat\mu_{i^*}+\operatorname{rad}_{i^*}.
$$
<span class="equation-number" aria-label="Equation 13">(13)</span>
</div>


On the good event $\mathcal{G}$,

<div class="numbered-equation" id="eq:optimal_still_plausible">
$$
\widehat\mu_{i^*}+\operatorname{rad}_{i^*}
\geq \mu_{i^*}.
$$
<span class="equation-number" aria-label="Equation 14">(14)</span>
</div>


Combining [Eq. (13)](#eq:ucb_chose_i) and [Eq. (14)](#eq:optimal_still_plausible),

<div class="numbered-equation" id="eq:chosen_index_above_opt">
$$
\widehat\mu_i + \operatorname{rad}_i
\geq \mu_{i^*}.
$$
<span class="equation-number" aria-label="Equation 15">(15)</span>
</div>


Again on $\mathcal{G}$,

<div class="numbered-equation" id="eq:suboptimal_upper">
$$
\widehat\mu_i \leq \mu_i + \operatorname{rad}_i.
$$
<span class="equation-number" aria-label="Equation 16">(16)</span>
</div>


Substitute [Eq. (16)](#eq:suboptimal_upper) into [Eq. (15)](#eq:chosen_index_above_opt):

<div class="numbered-equation" id="eq:radius_large">
$$
\begin{align}
\mu_i + \operatorname{rad}_i + \operatorname{rad}_i
&\geq \mu_{i^*},\\
2\operatorname{rad}_i
&\geq \mu_{i^*}-\mu_i,\\
2\operatorname{rad}_i
&\geq \Delta_i,\\
\operatorname{rad}_i
&\geq \frac{\Delta_i}{2}.
\end{align}
$$
<span class="equation-number" aria-label="Equation 17">(17)</span>
</div>


Now plug in the radius:

<div class="numbered-equation" id="eq:ucb_pull_count_simple">
$$
\begin{align}
\sqrt{\frac{\log(2KT/\delta)}{2N_i(t)}}
&\geq \frac{\Delta_i}{2},\\
\frac{\log(2KT/\delta)}{2N_i(t)}
&\geq \frac{\Delta_i^2}{4},\\
4\log(2KT/\delta)
&\geq 2N_i(t)\Delta_i^2,\\
N_i(t)
&\leq \frac{2\log(2KT/\delta)}{\Delta_i^2}.
\end{align}
$$
<span class="equation-number" aria-label="Equation 18">(18)</span>
</div>


Depending on the exact convention for the radius, constants change. The message does not change: a bad arm can only be pulled many times if its uncertainty radius is still large. Once it has been sampled enough, its upper confidence bound drops below the optimal arm's plausible value.

> **Key idea.**
>
> The UCB proof is not magic. It is a bookkeeping argument:

<div class="display-equation">
$$
\text{bad arm chosen}
\Rightarrow
\text{its uncertainty must still be large}
\Rightarrow
\text{it has not been sampled too often}.
$$
</div>


## Post-selection bias: the quiet enemy

Another common mistake is to look at many random estimates and then trust the largest one as if it had been chosen in advance.

Imagine $20$ arms. In this experiment they are all identical: every arm has true mean $0.5$. We sample each arm $20$ times and select the one with the largest empirical mean.

The selected arm looks much better than $0.5$, not because it is truly better, but because it won a noise contest.

![When all arms have the same true mean, the best empirical arm is still biased upward. Selection turns noise into apparent signal.](/images/notes/assets/filtrations/figures/post_selection_bias.webp)

*When all arms have the same true mean, the best empirical arm is still biased upward. Selection turns noise into apparent signal.*

This is why bandit algorithms need explicit exploration rules. Greedy selection tends to chase the current winner. Confidence-based and posterior-based algorithms try to remember that the current winner may be winning only because the data is thin.

## Code implementation

The simulation is deliberately small. It is not a benchmark. It is a microscope.

The first part compares one fixed-time average with an adaptively stopped average.

    for r in range(cfg.runs):
        x = rng.binomial(1, cfg.p, size=cfg.fixed_n)
        s = np.cumsum(x)
        t = np.arange(1, cfg.fixed_n + 1)
        m = s / t

        fixed_means[r] = m[-1]

        normal_radius = 1.96 * np.sqrt(0.25 / t)
        union_r = union_radius(t, cfg.alpha, cfg.fixed_n)
        final_normal_false_alarm[r] = abs(m[-1] - cfg.p) > normal_radius[-1]
        anytime_normal_false_alarm[r] = np.any(np.abs(m - cfg.p) > normal_radius)
        anytime_union_false_alarm[r] = np.any(np.abs(m - cfg.p) > union_r)

        eligible = np.where((t >= cfg.stop_start) & (m >= cfg.threshold))[0]
        if eligible.size > 0:
            j = int(eligible[0])
            stopped_times[r] = j + 1
            stopped_means[r] = m[j]
            stopped_by_rule[r] = True
        else:
            stopped_times[r] = cfg.fixed_n
            stopped_means[r] = m[-1]
            stopped_by_rule[r] = False

The second part shows post-selection bias.

    # All arms have the same true mean.  Any winner is a winner only because of noise.
    samples = rng.binomial(1, cfg.p, size=(cfg.runs, cfg.k_arms, cfg.n_per_arm))
    means = samples.mean(axis=2)
    selected = means.max(axis=1)
    ordinary = means[:, 0]

The full script is included in the appendix and in the accompanying simulation file.

## What this chapter prepares

The next bandit algorithms will use the same grammar repeatedly.

For UCB, the key object is a confidence event that holds across arms and times.

For Thompson sampling, the key object is a posterior distribution updated by the same history $\F_t$.

For best-arm identification, the key object is a stopping time: a random time at which the algorithm decides it has enough evidence.

For batched bandits, the key object is restricted adaptivity: the notebook is updated only at batch boundaries.

So probability is not a decorative layer. It is the language that keeps time, information, and evidence in the correct order.

> **Key idea.**
>
> The deepest lesson is simple: in bandits, data has a timestamp. A proof is correct only if it respects what was known before the action and what was revealed after the action.

## Appendix A. Symbol Table

  -----------------------------------------------------------------------------------------------------
  Symbol                      Meaning
  --------------------------- -------------------------------------------------------------------------
  Symbol                      Meaning

  $A_t$                       action chosen at round $t$

  $R_t$                       reward observed after choosing $A_t$

  $H_t$                       history or notebook after round $t$

  $\F_t$                      filtration: all information contained in $H_t$

  $\mu_i$                     true mean reward of arm $i$

  $N_i(t)$                    number of pulls of arm $i$ by time $t$

  $\widehat\mu_i(t)$          empirical mean of arm $i$ by time $t$

  $\Delta_i$                  suboptimality gap $\mu_{i^*}-\mu_i$

  $\eta_t$                    centered reward noise $R_t-\mu_{A_t}$

  $\operatorname{rad}_i(t)$   confidence radius for arm $i$ at time $t$

  $\mathcal{G}$               good event on which all confidence intervals are simultaneously correct
  -----------------------------------------------------------------------------------------------------

## Appendix B. Core Formulas


<div class="display-equation">
$$
\begin{align}
\E[X] &= \sum_x x\,\Pp(X=x),\\
\widehat p_n &= \frac{1}{n}\sum_{s=1}^n X_s,\\
\Pp\left(|\widehat p_n-p|\geq \varepsilon\right) &\leq 2\exp(-2n\varepsilon^2),\\
\varepsilon_n(\delta) &= \sqrt{\frac{\log(2/\delta)}{2n}},\\
H_t &= (A_1,R_1,\ldots,A_t,R_t),\\
\E[R_t\mid \F_{t-1}] &= \mu_{A_t},\\
\E[R_t-\mu_{A_t}\mid \F_{t-1}] &=0,\\
N_i(t)&=\sum_{s=1}^t \one\{A_s=i\},\\
\widehat\mu_i(t)-\mu_i
&=\frac{1}{N_i(t)}\sum_{s=1}^t \one\{A_s=i\}(R_s-\mu_i),\\
\Pp\left(\bigcup_j E_j\right)&\leq \sum_j \Pp(E_j),\\
\operatorname{UCB}_i(t)&=\widehat\mu_i(t)+\sqrt{\frac{\log(2KT/\delta)}{2N_i(t)}}.
\end{align}
$$
</div>


## Appendix C. Full Simulation Code

``` {.python style="blogcode" language="Python" caption="Full simulation code."}
"""
Simulation for the lecture note:
Probability, Filtrations, and Adaptive Data.

The goal is not to build a large benchmark.  The goal is to make three
probability ideas visible:

1. A sample average is reliable at one fixed time.
2. Looking many times with the same fixed-time error bar creates false discoveries.
3. Selecting the largest empirical mean among many arms creates upward bias.

Run:
    python probability-filtrations-simulation.py
"""
from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

rng = np.random.default_rng(7)


@dataclass
class Config:
    p: float = 0.50
    runs: int = 20000
    fixed_n: int = 200
    stop_start: int = 20
    threshold: float = 0.56
    alpha: float = 0.05
    k_arms: int = 20
    n_per_arm: int = 20


def hoeffding_radius(n: np.ndarray | float, alpha: float) -> np.ndarray | float:
    """Fixed-time two-sided Hoeffding radius for Bernoulli rewards in [0,1]."""
    return np.sqrt(np.log(2.0 / alpha) / (2.0 * n))


def union_radius(n: np.ndarray | float, alpha: float, T: int) -> np.ndarray | float:
    """A simple anytime radius obtained by a union bound over t=1,...,T."""
    return np.sqrt(np.log(2.0 * T / alpha) / (2.0 * n))


def simulate_fixed_and_stopped(cfg: Config):
    fixed_means = np.empty(cfg.runs)
    stopped_means = np.empty(cfg.runs)
    stopped_times = np.empty(cfg.runs, dtype=int)
    stopped_by_rule = np.empty(cfg.runs, dtype=bool)
    final_normal_false_alarm = np.empty(cfg.runs, dtype=bool)
    anytime_normal_false_alarm = np.empty(cfg.runs, dtype=bool)
    anytime_union_false_alarm = np.empty(cfg.runs, dtype=bool)

    for r in range(cfg.runs):
        x = rng.binomial(1, cfg.p, size=cfg.fixed_n)
        s = np.cumsum(x)
        t = np.arange(1, cfg.fixed_n + 1)
        m = s / t

        fixed_means[r] = m[-1]

        normal_radius = 1.96 * np.sqrt(0.25 / t)
        union_r = union_radius(t, cfg.alpha, cfg.fixed_n)
        final_normal_false_alarm[r] = abs(m[-1] - cfg.p) > normal_radius[-1]
        anytime_normal_false_alarm[r] = np.any(np.abs(m - cfg.p) > normal_radius)
        anytime_union_false_alarm[r] = np.any(np.abs(m - cfg.p) > union_r)

        eligible = np.where((t >= cfg.stop_start) & (m >= cfg.threshold))[0]
        if eligible.size > 0:
            j = int(eligible[0])
            stopped_times[r] = j + 1
            stopped_means[r] = m[j]
            stopped_by_rule[r] = True
        else:
            stopped_times[r] = cfg.fixed_n
            stopped_means[r] = m[-1]
            stopped_by_rule[r] = False

    rad_fixed = hoeffding_radius(cfg.fixed_n, cfg.alpha)
    fixed_cover = np.mean(np.abs(fixed_means - cfg.p) <= rad_fixed)

    rad_stopped_naive = hoeffding_radius(stopped_times, cfg.alpha)
    stopped_cover_naive = np.mean(np.abs(stopped_means - cfg.p) <= rad_stopped_naive)

    rad_stopped_union = union_radius(stopped_times, cfg.alpha, cfg.fixed_n)
    stopped_cover_union = np.mean(np.abs(stopped_means - cfg.p) <= rad_stopped_union)

    return {
        "fixed_means": fixed_means,
        "stopped_means": stopped_means,
        "stopped_times": stopped_times,
        "stopped_by_rule": stopped_by_rule,
        "fixed_cover": fixed_cover,
        "stopped_cover_naive": stopped_cover_naive,
        "stopped_cover_union": stopped_cover_union,
        "final_normal_false_alarm": np.mean(final_normal_false_alarm),
        "anytime_normal_false_alarm": np.mean(anytime_normal_false_alarm),
        "anytime_union_false_alarm": np.mean(anytime_union_false_alarm),
        "stop_rate": np.mean(stopped_by_rule),
        "avg_stop_time": np.mean(stopped_times),
        "fixed_mean_avg": np.mean(fixed_means),
        "stopped_mean_avg": np.mean(stopped_means),
    }


def simulate_post_selection(cfg: Config):
    # All arms have the same true mean.  Any winner is a winner only because of noise.
    samples = rng.binomial(1, cfg.p, size=(cfg.runs, cfg.k_arms, cfg.n_per_arm))
    means = samples.mean(axis=2)
    selected = means.max(axis=1)
    ordinary = means[:, 0]
    return {
        "selected_means": selected,
        "ordinary_means": ordinary,
        "selected_mean_avg": float(np.mean(selected)),
        "ordinary_mean_avg": float(np.mean(ordinary)),
        "selection_bias": float(np.mean(selected) - cfg.p),
    }


def make_plots(cfg: Config, fs, ps):
    plt.figure(figsize=(6.6, 4.0))
    bins = np.linspace(0.35, 0.75, 41)
    plt.hist(fs["fixed_means"], bins=bins, alpha=0.6, density=True, label="fixed time average")
    plt.hist(fs["stopped_means"], bins=bins, alpha=0.6, density=True, label="average after adaptive stopping")
    plt.axvline(cfg.p, linewidth=2, label="true mean")
    plt.xlabel("observed sample mean")
    plt.ylabel("density")
    plt.title("Fixed sampling versus adaptive stopping")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "fixed_vs_stopped_means.pdf"))
    plt.savefig(os.path.join(FIG, "fixed_vs_stopped_means.png"), dpi=200)
    plt.close()

    labels = ["one final look\nnormal bar", "many peeks\nsame bar", "many peeks\nunion bar"]
    vals = [fs["final_normal_false_alarm"], fs["anytime_normal_false_alarm"], fs["anytime_union_false_alarm"]]
    plt.figure(figsize=(6.4, 3.8))
    plt.bar(labels, vals)
    plt.axhline(0.05, linestyle="--", linewidth=1.5, label="5% target")
    plt.ylim(0.0, 0.50)
    plt.ylabel("false alarm rate")
    plt.title("Peeking many times changes the meaning of an error bar")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "false_alarm_comparison.pdf"))
    plt.savefig(os.path.join(FIG, "false_alarm_comparison.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(6.6, 4.0))
    bins = np.linspace(0.2, 0.95, 41)
    plt.hist(ps["ordinary_means"], bins=bins, alpha=0.6, density=True, label="one fixed arm")
    plt.hist(ps["selected_means"], bins=bins, alpha=0.6, density=True, label="best empirical arm among 20")
    plt.axvline(cfg.p, linewidth=2, label="true mean")
    plt.xlabel("empirical mean")
    plt.ylabel("density")
    plt.title("Post-selection optimism when all arms are equal")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "post_selection_bias.pdf"))
    plt.savefig(os.path.join(FIG, "post_selection_bias.png"), dpi=200)
    plt.close()


def main():
    cfg = Config()
    fs = simulate_fixed_and_stopped(cfg)
    ps = simulate_post_selection(cfg)
    make_plots(cfg, fs, ps)

    rows = [
        {"quantity": "true Bernoulli mean", "value": cfg.p},
        {"quantity": "runs", "value": cfg.runs},
        {"quantity": "fixed sample size", "value": cfg.fixed_n},
        {"quantity": "adaptive stop threshold", "value": cfg.threshold},
        {"quantity": "probability of stopping early", "value": fs["stop_rate"]},
        {"quantity": "average stopping time", "value": fs["avg_stop_time"]},
        {"quantity": "average fixed-time mean", "value": fs["fixed_mean_avg"]},
        {"quantity": "average stopped mean", "value": fs["stopped_mean_avg"]},
        {"quantity": "fixed-time Hoeffding coverage", "value": fs["fixed_cover"]},
        {"quantity": "stopped-time naive Hoeffding coverage", "value": fs["stopped_cover_naive"]},
        {"quantity": "stopped-time union-bound Hoeffding coverage", "value": fs["stopped_cover_union"]},
        {"quantity": "one final look false alarm with normal bar", "value": fs["final_normal_false_alarm"]},
        {"quantity": "many peeks false alarm with same normal bar", "value": fs["anytime_normal_false_alarm"]},
        {"quantity": "many peeks false alarm with union bar", "value": fs["anytime_union_false_alarm"]},
        {"quantity": "post-selection average of one fixed arm", "value": ps["ordinary_mean_avg"]},
        {"quantity": "post-selection average of empirical winner", "value": ps["selected_mean_avg"]},
        {"quantity": "post-selection bias", "value": ps["selection_bias"]},
    ]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "results.csv"), index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
```

## References
Auer, P., N. Cesa-Bianchi, and P. Fischer. 2002. "Finite-Time Analysis of the Multiarmed Bandit Problem." *Machine Learning* 47: 235--56.


Bubeck, S., and N. Cesa-Bianchi. 2012. "Regret Analysis of Stochastic and Nonstochastic Multi-Armed Bandit Problems." *Foundations and Trends in Machine Learning* 5 (1): 1--122.


Howard, Steven R., Aaditya Ramdas, Jon McAuliffe, and Jasjeet Sekhon. 2020. "Time-Uniform Chernoff Bounds via Nonnegative Supermartingales." *Probability Surveys* 17: 257--317.


Lattimore, T., and C. Szepesvari. 2020. *Bandit Algorithms*. Cambridge University Press.
