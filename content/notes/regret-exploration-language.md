---
title: "Regret, Exploration, and Exploitation"
subtitle: "The Language of Bandit Theory"
summary: "Regret, exploration-exploitation tradeoffs, and analytical viewpoints."
description: "Regret, exploration-exploitation tradeoffs, and analytical viewpoints."
date: 2026-06-17
lastmod: 2026-06-17
weight: 20
libraryFolder: "bandit-learning"
libraryFolderName: "Bandit 与在线学习"
libraryFolderColor: 1
tags: ["Bandits", "Regret", "Exploration"]
draft: false
ShowToc: false
hideMeta: true
---

## The first word is regret

A bandit problem begins with a small annoyance.

You choose an action. You see the reward of that action. The rewards of the actions you did not choose disappear.

This is not a technical detail. It is the whole problem. In supervised learning, a training example usually tells us the correct label. In a bandit problem, one action gives one receipt. A receipt is not an answer key.

The word *regret* is the cleanest way to measure the cost of this missing answer key.

> **Idea.**
>
> Regret is not sadness. It is an accounting identity. It asks: after $T$ decisions, how much reward did we lose compared with always taking the best action?

Imagine three possible headlines for the same article. The true click probabilities are $0.30,\qquad 0.35,\qquad 0.42.$ The third headline is best, but the learner does not know this. If it shows the first headline, it observes only whether the first headline was clicked. It does not observe whether the second or third headline would have been clicked by the same reader.

That is the whole difficulty:

<div class="display-equation">
$$
\text{we need data to choose well, but choosing badly is how we get some of the data.}
$$
</div>


### The tiny laboratory

We use a $K$-armed Bernoulli bandit. There are $K$ actions, called arms. Pulling arm $a$ gives a random reward $X_{a,t}\in\{0,1\},$ with mean $\E[X_{a,t}] = \mu_a.$ For the running example, $(\mu_1,\mu_2,\mu_3)=(0.30,0.35,0.42).$ The best mean is $\mu_\star = \max_{a\in\{1,\ldots,K\}}\mu_a=0.42.$

At round $t$, the learner chooses an arm $A_t$ and observes only $X_{A_t,t}.$ It does not observe $X_{a,t}$ for $a\ne A_t$.


The dashed arrows are the counterfactual rewards. They exist in our mathematical imagination, but they are not observed by the learner.

### Regret as one-line bookkeeping

Suppose the learner chooses arms $A_1,A_2,\ldots,A_T.$ The oracle, who knows the best arm, receives expected reward $T\mu_\star.$ The learner receives expected reward $\E\left[\sum_{t=1}^T X_{A_t,t}\right].$ So the expected regret is

<div class="display-equation">
$$
R_T
=
T\mu_\star
-
\E\left[\sum_{t=1}^T X_{A_t,t}\right].
$$
</div>


Now write it in the useful form. The reward of the chosen arm has conditional mean $\mu_{A_t}$. Therefore

<div class="display-equation">
$$
\begin{aligned}
R_T
&=
T\mu_\star
-
\E\left[\sum_{t=1}^T X_{A_t,t}\right] \\[0.3em]
&=
\E\left[\sum_{t=1}^T \mu_\star\right]
-
\E\left[\sum_{t=1}^T X_{A_t,t}\right] \\[0.3em]
&=
\E\left[\sum_{t=1}^T \bigl(\mu_\star-X_{A_t,t}\bigr)\right] \\[0.3em]
&=
\sum_{t=1}^T
\E\left[\mu_\star-X_{A_t,t}\right].
\end{aligned}
$$
</div>

 The only probabilistic step is the next one: $\E[X_{A_t,t}\mid A_t]=\mu_{A_t}.$ Hence

<div class="display-equation">
$$
\begin{aligned}
\E\left[\mu_\star-X_{A_t,t}\right]
&=
\E\left[
\E\left[\mu_\star-X_{A_t,t}\mid A_t\right]
\right] \\[0.3em]
&=
\E\left[
\mu_\star-\E[X_{A_t,t}\mid A_t]
\right] \\[0.3em]
&=
\E\left[\mu_\star-\mu_{A_t}\right].
\end{aligned}
$$
</div>

 Therefore

<div class="display-equation">
$$
\boxed{
R_T=
\E\left[\sum_{t=1}^T(\mu_\star-\mu_{A_t})\right].
}
$$
</div>


This is the basic accounting formula. Define the gap of arm $a$ by $\Delta_a=\mu_\star-\mu_a.$ If arm $a$ is pulled $N_a(T)$ times, then

<div class="display-equation">
$$
\begin{aligned}
R_T
&=
\E\left[\sum_{t=1}^T \Delta_{A_t}\right] \\[0.3em]
&=
\E\left[\sum_{t=1}^T\sum_{a=1}^K \Delta_a\ind\{A_t=a\}\right] \\[0.3em]
&=
\sum_{a=1}^K
\Delta_a\,
\E\left[\sum_{t=1}^T \ind\{A_t=a\}\right] \\[0.3em]
&=
\boxed{
\sum_{a=1}^K \Delta_a\,\E[N_a(T)].
}
\end{aligned}
$$
</div>


> **Think.**
>
> This formula is the first real language of bandit theory. To control regret, we do not need to predict every reward. We need to control how often the algorithm pulls arms with positive gap.

## Exploration and exploitation are not slogans

People often describe bandits by saying "exploration versus exploitation." The phrase is correct, but too vague. Regret makes it precise.

If we pull a bad arm, we lose reward now. That is exploration cost.

If we avoid an uncertain arm too early, we may never learn that it is good. That is exploitation cost caused by premature certainty.

The real question is:

<div class="display-equation">
$$
\text{How many times should uncertainty be allowed to buy more information?}
$$
</div>


### The two costs in one toy calculation

Consider two arms with $\mu_1<\mu_2,\qquad \Delta=\mu_2-\mu_1>0.$ A simple strategy is *explore-then-commit*:

1.  Pull each arm $m$ times.

2.  Compare their sample means.

3.  Use the arm with larger sample mean for the remaining $T-2m$ rounds.

If the algorithm explores arm 1 exactly $m$ times, that part alone costs $m\Delta.$ The more dangerous event is choosing arm 1 after exploration. Let

<div class="display-equation">
$$
\widehat{\mu}_{1,m}
=
\frac1m\sum_{i=1}^m X_{1,i},
\qquad
\widehat{\mu}_{2,m}
=
\frac1m\sum_{i=1}^m X_{2,i}.
$$
</div>

 The algorithm commits to the wrong arm when $\widehat{\mu}_{1,m}\ge \widehat{\mu}_{2,m}.$ Since $\mu_2-\mu_1=\Delta$, this event implies at least one of the following two things:

<div class="display-equation">
$$
\widehat{\mu}_{1,m}-\mu_1\ge \frac{\Delta}{2},
\qquad
\mu_2-\widehat{\mu}_{2,m}\ge \frac{\Delta}{2}.
$$
</div>

 Indeed, if neither happened, then

<div class="display-equation">
$$
\widehat{\mu}_{1,m}
&lt;
\mu_1+\frac{\Delta}{2}
=
\frac{\mu_1+\mu_2}{2}
=
\mu_2-\frac{\Delta}{2}
&lt;
\widehat{\mu}_{2,m},
$$
</div>

 so the wrong commitment would be impossible.

Therefore, by the union bound,

<div class="display-equation">
$$
\begin{aligned}
\Prob(\widehat{\mu}_{1,m}\ge \widehat{\mu}_{2,m})
&\le
\Prob\left(\widehat{\mu}_{1,m}-\mu_1\ge \frac{\Delta}{2}\right)
+
\Prob\left(\mu_2-\widehat{\mu}_{2,m}\ge \frac{\Delta}{2}\right).
\end{aligned}
$$
</div>

 For rewards in $[0,1]$, Hoeffding's inequality says

<div class="display-equation">
$$
\Prob(\widehat{\mu}_{a,m}-\mu_a\ge \varepsilon)
\le
\exp(-2m\varepsilon^2).
$$
</div>

 Use $\varepsilon=\Delta/2$:

<div class="display-equation">
$$
\begin{aligned}
\Prob(\widehat{\mu}_{1,m}\ge \widehat{\mu}_{2,m})
&\le
\exp\left(-2m\frac{\Delta^2}{4}\right)
+
\exp\left(-2m\frac{\Delta^2}{4}\right) \\[0.3em]
&=
2\exp\left(-\frac{m\Delta^2}{2}\right).
\end{aligned}
$$
</div>

 Hence the expected regret of explore-then-commit is bounded by

<div class="display-equation">
$$
\boxed{
R_T
\le
m\Delta
+
(T-2m)\Delta
\cdot
2\exp\left(-\frac{m\Delta^2}{2}\right).
}
$$
</div>


This is the tradeoff in its simplest form: $\text{explore more} \Longrightarrow m\Delta \text{ grows},$ but

<div class="display-equation">
$$
\text{explore more} \Longrightarrow
2e^{-m\Delta^2/2} \text{ shrinks}.
$$
</div>


> **Idea.**
>
> Exploration is not curiosity. It is an insurance premium against choosing the wrong arm for a long time.

## The algorithms in a small experiment

We now run a small experiment. The environment has three Bernoulli arms: $(0.30,0.35,0.42).$ The horizon is $T=2000$, and we repeat the experiment over $500$ independent runs.

We compare four algorithms.

### Greedy

Greedy uses the current sample mean and always chooses the arm that looks best: $A_t\in \argmax_a \widehat{\mu}_a(t).$ It is natural, but fragile. A lucky first click can make a bad arm look good. Once that happens, greedy may keep feeding the same mistaken belief.

### $\varepsilon$-greedy

$\varepsilon$-greedy mostly behaves greedily, but sometimes chooses a random arm:

<div class="display-equation">
$$
A_t=
\begin{cases}
\text{a uniformly random arm}, & \text{with probability }\varepsilon,\\
\argmax_a \widehat{\mu}_a(t), & \text{with probability }1-\varepsilon.
\end{cases}
$$
</div>

 This is the simplest repair. It forces the algorithm to keep looking around.

### UCB

UCB chooses the arm with the largest upper confidence index:

<div class="display-equation">
$$
A_t\in
\argmax_a
\left\{
\widehat{\mu}_a(t)
+
\sqrt{\frac{2\log t}{N_a(t)}}
\right\}.
$$
</div>

 The first term is what we have seen. The second term is how uncertain we still are. An arm can be chosen either because it has performed well or because it has not been tested enough.

### Thompson sampling

For Bernoulli rewards, Thompson sampling keeps a Beta posterior for each arm: $\theta_a\sim \mathrm{Beta}(\alpha_a,\beta_a).$ At each round it samples one possible world from the posterior and acts greedily in that sampled world: $A_t\in\argmax_a \theta_a.$ A click updates $\alpha_a\leftarrow \alpha_a+1,$ and a non-click updates $\beta_a\leftarrow \beta_a+1.$

### The code that creates the comparison

The important part is the action rule. The environment is intentionally small, so that the algorithmic difference is not hidden behind engineering.

    if policy == "Greedy":
        means = sums / np.maximum(counts, 1)
        a = int(np.argmax(means))

    elif policy == "Epsilon-greedy":
        if rng.random() < epsilon:
            a = int(rng.integers(K))
        else:
            means = sums / np.maximum(counts, 1)
            a = int(np.argmax(means))

    elif policy == "UCB":
        means = sums / counts
        bonus = np.sqrt(2.0 * np.log(max(t + 1, 2)) / counts)
        a = int(np.argmax(means + bonus))

    elif policy == "Thompson sampling":
        theta = rng.beta(alpha, beta)
        a = int(np.argmax(theta))

### What the experiment says
| Algorithm              | Final regret | Arm 1 pulls | Arm 2 pulls | Arm 3 pulls |
|:-----------------------|-------------:|------------:|------------:|------------:|
| Greedy                 |       137.51 |      932.35 |      366.19 |      701.46 |
| $\varepsilon$-greedy |        39.85 |      140.61 |      328.29 |     1531.10 |
| UCB                    |        67.29 |      286.26 |      470.60 |     1243.14 |
| Thompson sampling      |        27.77 |       98.75 |      227.37 |     1673.88 |

<p class="table-caption">Results over 500 runs, horizon $T=2000$.</p>

![Mean cumulative regret. The bands show approximately two standard errors.](/images/notes/assets/regret-exploration/regret_curves.webp)

*Mean cumulative regret. The bands show approximately two standard errors.*

![Mean number of pulls of each arm. The best arm has mean 0.42.](/images/notes/assets/regret-exploration/pull_counts.webp)

*Mean number of pulls of each arm. The best arm has mean 0.42.*

The greedy algorithm is not bad because it is stupid. It is bad because it has no mechanism for doubting its early evidence.

UCB and Thompson sampling are different ways of putting doubt into the action rule. UCB adds an explicit error bar. Thompson sampling randomizes according to posterior uncertainty. Both are saying: an arm that has not been tested enough should not be declared dead too quickly.

## A little probability, used as a ruler

The probability needed for this note is modest. We use it as a ruler for sample averages.

If arm $a$ has mean $\mu_a$, then after $n$ pulls its sample mean is

<div class="display-equation">
$$
\widehat{\mu}_{a,n}
=
\frac1n\sum_{i=1}^n X_{a,i}.
$$
</div>

 This is a noisy ruler. It points near $\mu_a$, but not exactly at $\mu_a$.

Hoeffding's inequality says that for rewards in $[0,1]$,

<div class="display-equation">
$$
\Prob\left(
\left|\widehat{\mu}_{a,n}-\mu_a\right|\ge r
\right)
\le
2\exp(-2nr^2).
$$
</div>

 If we want the error probability to be at most $\delta$, solve $2\exp(-2nr^2)=\delta.$ Step by step:

<div class="display-equation">
$$
\begin{aligned}
2\exp(-2nr^2)&=\delta,\\
\exp(-2nr^2)&=\frac{\delta}{2},\\
-2nr^2&=\log\frac{\delta}{2},\\
2nr^2&=\log\frac{2}{\delta},\\
r^2&=\frac{\log(2/\delta)}{2n},\\
r&=\sqrt{\frac{\log(2/\delta)}{2n}}.
\end{aligned}
$$
</div>

 Thus a natural error bar is

<div class="display-equation">
$$
\boxed{
\widehat{\mu}_{a,n}
\pm
\sqrt{\frac{\log(2/\delta)}{2n}}.
}
$$
</div>


> **Think.**
>
> A confidence interval is not a magical theorem. It is just a ruler with a warning label: after $n$ samples, the ruler is wrong by more than $r$ with probability at most $\delta$.

### Why the logarithm appears

If we check one arm once, a small failure probability is enough. But a bandit algorithm checks many arms at many times. We want all of the rulers to be correct at once.

Suppose there are $K$ arms and $T$ rounds. There are at most $KT$ pairs $(a,n)$ to worry about. If each pair fails with probability at most $2/T^4$, then the probability that any pair fails is at most

<div class="display-equation">
$$
KT\cdot \frac{2}{T^4}
=
\frac{2K}{T^3}.
$$
</div>

 This is the union bound. It says:

<div class="display-equation">
$$
\Prob(\text{at least one bad event})
\le
\sum \Prob(\text{each bad event}).
$$
</div>

 Now choose the radius

<div class="display-equation">
$$
r_{a,n}
=
\sqrt{\frac{2\log T}{n}}.
$$
</div>

 Then

<div class="display-equation">
$$
\begin{aligned}
\Prob\left(
\left|\widehat{\mu}_{a,n}-\mu_a\right|
\ge
\sqrt{\frac{2\log T}{n}}
\right)
&\le
2\exp\left(
-2n\cdot \frac{2\log T}{n}
\right)\\
&=
2\exp(-4\log T)\\
&=
\frac{2}{T^4}.
\end{aligned}
$$
</div>

 So with high probability, for every arm and every sample size,

<div class="display-equation">
$$
\left|\widehat{\mu}_{a,n}-\mu_a\right|
\le
\sqrt{\frac{2\log T}{n}}.
$$
</div>


This is the mathematical reason for the UCB bonus.

## UCB proof, slowly

We now prove the standard idea behind UCB regret. The proof is not long. The hard part is knowing what each line is trying to say.

Let

<div class="display-equation">
$$
\mathrm{rad}_{a}(t)
=
\sqrt{\frac{2\log T}{N_a(t)}}.
$$
</div>

 UCB uses the index

<div class="display-equation">
$$
\mathrm{UCB}_a(t)
=
\widehat{\mu}_a(t)+\mathrm{rad}_a(t).
$$
</div>

 At time $t$, it chooses an arm with largest index.

### The good event

Define the good event

<div class="display-equation">
$$
\mathcal{G}
=
\left\{
\forall a,\forall n\le T:
\left|\widehat{\mu}_{a,n}-\mu_a\right|
\le
\sqrt{\frac{2\log T}{n}}
\right\}.
$$
</div>

 From the previous calculation, $\Prob(\mathcal{G}^c)\le \frac{2K}{T^3}.$

On $\mathcal{G}$, every empirical mean is close to its truth at every time. So the rest of the proof is deterministic bookkeeping.

### What must be true if UCB pulls a bad arm

Fix a suboptimal arm $a$, so $\Delta_a=\mu_\star-\mu_a>0.$ Suppose UCB pulls arm $a$ at some time $t$. Since UCB chooses the largest index,

<div class="display-equation">
$$
\widehat{\mu}_a(t)+\mathrm{rad}_a(t)
\ge
\widehat{\mu}_\star(t)+\mathrm{rad}_\star(t).
$$
</div>

 On the good event,

<div class="display-equation">
$$
\widehat{\mu}_\star(t)+\mathrm{rad}_\star(t)
\ge
\mu_\star.
$$
</div>

 Therefore,

<div class="display-equation">
$$
\widehat{\mu}_a(t)+\mathrm{rad}_a(t)
\ge
\mu_\star.
$$
</div>

 Again on the good event,

<div class="display-equation">
$$
\widehat{\mu}_a(t)
\le
\mu_a+\mathrm{rad}_a(t).
$$
</div>

 Combine the two inequalities:

<div class="display-equation">
$$
\begin{aligned}
\mu_\star
&\le
\widehat{\mu}_a(t)+\mathrm{rad}_a(t)\\
&\le
\mu_a+\mathrm{rad}_a(t)+\mathrm{rad}_a(t)\\
&=
\mu_a+2\mathrm{rad}_a(t).
\end{aligned}
$$
</div>

 Thus

<div class="display-equation">
$$
\Delta_a
=
\mu_\star-\mu_a
\le
2\mathrm{rad}_a(t).
$$
</div>

 Substitute the radius:

<div class="display-equation">
$$
\Delta_a
\le
2\sqrt{\frac{2\log T}{N_a(t)}}.
$$
</div>

 Square both sides:

<div class="display-equation">
$$
\Delta_a^2
\le
4\cdot \frac{2\log T}{N_a(t)}
=
\frac{8\log T}{N_a(t)}.
$$
</div>

 Rearrange:

<div class="display-equation">
$$
N_a(t)
\le
\frac{8\log T}{\Delta_a^2}.
$$
</div>


> **Idea.**
>
> This is the whole UCB proof in one sentence: once a bad arm has been pulled enough, its uncertainty bonus becomes too small to hide its gap.

### Turning pull count into regret

If arm $a$ is suboptimal, then on the good event it can be pulled at most about $\frac{8\log T}{\Delta_a^2}$ times. Each pull costs $\Delta_a$. Therefore its regret contribution is at most

<div class="display-equation">
$$
\Delta_a\cdot \frac{8\log T}{\Delta_a^2}
=
\frac{8\log T}{\Delta_a}.
$$
</div>

 Summing over suboptimal arms,

<div class="display-equation">
$$
\boxed{
R_T
\lesssim
\sum_{a:\Delta_a>0}
\frac{8\log T}{\Delta_a}.
}
$$
</div>

 This is called a gap-dependent bound. It is strong when the gaps are not too small.

### Why small gaps are different

If $\Delta_a$ is tiny, pulling arm $a$ is not very harmful. The previous bound becomes large because it tries to identify a tiny difference very accurately. But a tiny difference does not need to be identified quickly, because its regret cost is small.

A simple split captures this. Choose a threshold $\varepsilon>0$.

For arms with small gap $\Delta_a\le \varepsilon$, the regret is at most $T\varepsilon.$ For arms with larger gap $\Delta_a>\varepsilon$, use the UCB count:

<div class="display-equation">
$$
\sum_{\Delta_a>\varepsilon}
\frac{8\log T}{\Delta_a}
\le
\sum_{\Delta_a>\varepsilon}
\frac{8\log T}{\varepsilon}
\le
\frac{8K\log T}{\varepsilon}.
$$
</div>

 Thus

<div class="display-equation">
$$
R_T
\lesssim
T\varepsilon+\frac{8K\log T}{\varepsilon}.
$$
</div>

 Balance the two terms:

<div class="display-equation">
$$
T\varepsilon
=
\frac{8K\log T}{\varepsilon}.
$$
</div>

 Then

<div class="display-equation">
$$
\varepsilon^2
=
\frac{8K\log T}{T},
\qquad
\varepsilon
=
\sqrt{\frac{8K\log T}{T}}.
$$
</div>

 Substitute back:

<div class="display-equation">
$$
R_T
\lesssim
\sqrt{KT\log T}.
$$
</div>

 This is a gap-independent statement. It says that even if the gaps are tiny, regret grows much slower than $T$.

## Thompson sampling as probability matching

UCB says: use an optimistic upper bound. Thompson sampling says: sample a possible world and act as if that world were true.

For Bernoulli rewards, start with $\theta_a\sim \mathrm{Beta}(1,1).$ After $S_a$ successes and $F_a$ failures,

<div class="display-equation">
$$
\theta_a\mid \text{data}
\sim
\mathrm{Beta}(1+S_a,1+F_a).
$$
</div>

 At each round:

<div class="display-equation">
$$
\theta_1,\ldots,\theta_K
\text{ are sampled independently from their posteriors,}
$$
</div>

 and the learner chooses $A_t\in\argmax_a \theta_a.$

> **Think.**
>
> Thompson sampling does not add a bonus. It lets uncertainty itself randomize the decision. An uncertain arm still has a chance to look best in a plausible sampled world.

This is why Thompson sampling often feels less mechanical than UCB. It does not ask each arm for an explicit optimism certificate. It asks the posterior: how often is this arm the winner in worlds that still look plausible?

In symbols,

<div class="display-equation">
$$
\Prob(A_t=a\mid \mathcal{H}_{t-1})
=
\Prob\left(
\theta_a=\max_j\theta_j
\mid \mathcal{H}_{t-1}
\right),
$$
</div>

 where $\mathcal{H}_{t-1}$ is the data observed before round $t$.

This is sometimes called probability matching. The probability of choosing an arm is matched to the posterior probability that the arm is optimal.

## What this language buys us

The point of this note is not that UCB and Thompson sampling are the only algorithms worth knowing. The point is that they teach the basic grammar of sequential learning.


Once you see this grammar, many later topics become less mysterious:

- Linear bandits replace arm means by a parameterized reward model.

- Contextual bandits let the best action depend on observed context.

- Bayesian optimization uses posterior uncertainty over functions.

- Best-arm identification changes the objective from earning reward to finding the best arm.

- Reinforcement learning adds state, delayed reward, and credit assignment.

The same question keeps returning: what should an algorithm do with uncertainty when every experiment has a cost?

## Appendix A. Symbol Table

  Symbol                  Meaning
  ----------------------- ---------------------------------------------------------------------
  $K$                     number of arms
  $T$                     time horizon
  $A_t$                   action chosen at round $t$
  $X_{a,t}$               reward of arm $a$ at round $t$
  $\mu_a$                 mean reward of arm $a$
  $\mu_\star$             best mean reward
  $\Delta_a$              gap $\mu_\star-\mu_a$
  $N_a(T)$                number of pulls of arm $a$ up to time $T$
  $\widehat{\mu}_{a,n}$   sample mean of arm $a$ after $n$ pulls
  $R_T$                   expected cumulative regret
  $\mathcal{G}$           good event where all empirical means stay in their confidence bands

## Appendix B. Full Experiment Script

``` {.python style="blogcode" language="Python" caption="Full experiment script."}

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_once(policy, probs, horizon, rng, epsilon=0.10):
    K = len(probs)
    counts = np.zeros(K, dtype=int)
    sums = np.zeros(K, dtype=float)
    alpha = np.ones(K)
    beta = np.ones(K)
    mu_star = np.max(probs)
    cumulative_regret = np.zeros(horizon)

    def pull(a):
        r = rng.binomial(1, probs[a])
        counts[a] += 1
        sums[a] += r
        alpha[a] += r
        beta[a] += 1 - r
        return r

    regret = 0.0

    # give each policy one clean round of initial data, except Thompson, which can act from its prior.
    if policy in ["Greedy", "Epsilon-greedy", "UCB"]:
        for a in range(K):
            if a >= horizon:
                break
            r = pull(a)
            regret += mu_star - probs[a]
            cumulative_regret[a] = regret
        start = min(K, horizon)
    else:
        start = 0

    for t in range(start, horizon):
        if policy == "Greedy":
            means = np.divide(sums, np.maximum(counts, 1))
            a = int(np.argmax(means))
        elif policy == "Epsilon-greedy":
            if rng.random() < epsilon:
                a = int(rng.integers(K))
            else:
                means = np.divide(sums, np.maximum(counts, 1))
                a = int(np.argmax(means))
        elif policy == "UCB":
            means = sums / counts
            bonus = np.sqrt(2.0 * np.log(max(t + 1, 2)) / counts)
            a = int(np.argmax(means + bonus))
        elif policy == "Thompson sampling":
            theta = rng.beta(alpha, beta)
            a = int(np.argmax(theta))
        else:
            raise ValueError(policy)

        r = pull(a)
        regret += mu_star - probs[a]
        cumulative_regret[t] = regret

    return cumulative_regret, counts

def main():
    probs = np.array([0.30, 0.35, 0.42])
    horizon = 2000
    n_runs = 500
    policies = ["Greedy", "Epsilon-greedy", "UCB", "Thompson sampling"]

    all_regrets = {}
    all_counts = {}
    for policy in policies:
        regrets = []
        counts = []
        for seed in range(n_runs):
            rng = np.random.default_rng(202706 + 1009 * seed + len(policy))
            r, c = run_once(policy, probs, horizon, rng)
            regrets.append(r)
            counts.append(c)
        all_regrets[policy] = np.vstack(regrets)
        all_counts[policy] = np.vstack(counts)

    rows = []
    for policy in policies:
        final = all_regrets[policy][:, -1]
        cnt = all_counts[policy]
        rows.append({
            "policy": policy,
            "mean_final_regret": final.mean(),
            "std_final_regret": final.std(ddof=1),
            "arm0_mean_pulls": cnt[:, 0].mean(),
            "arm1_mean_pulls": cnt[:, 1].mean(),
            "arm2_mean_pulls": cnt[:, 2].mean(),
        })
    df = pd.DataFrame(rows)
    df.to_csv("results.csv", index=False)

    plt.figure(figsize=(7.2, 4.2))
    x = np.arange(1, horizon + 1)
    for policy in policies:
        mean = all_regrets[policy].mean(axis=0)
        se = all_regrets[policy].std(axis=0, ddof=1) / np.sqrt(n_runs)
        plt.plot(x, mean, label=policy)
        plt.fill_between(x, mean - 2 * se, mean + 2 * se, alpha=0.12)
    plt.xlabel("round")
    plt.ylabel("mean cumulative regret")
    plt.title("Regret in a three-arm Bernoulli bandit")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig("regret_curves.pdf")
    plt.savefig("regret_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7.2, 4.2))
    labels = [r"$p=0.30$", r"$p=0.35$", r"$p=0.42$"]
    xloc = np.arange(len(labels))
    width = 0.18
    for i, policy in enumerate(policies):
        means = all_counts[policy].mean(axis=0)
        plt.bar(xloc + (i - 1.5) * width, means, width, label=policy)
    plt.xticks(xloc, labels)
    plt.ylabel("mean number of pulls")
    plt.title("Where each algorithm spends its samples")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig("pull_counts.pdf")
    plt.savefig("pull_counts.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
```

## References
Auer, P., N. Cesa-Bianchi, and P. Fischer. 2002. "Finite-Time Analysis of the Multiarmed Bandit Problem." *Machine Learning* 47: 235--56.


Bubeck, S., and N. Cesa-Bianchi. 2012. "Regret Analysis of Stochastic and Nonstochastic Multi-Armed Bandit Problems." *Foundations and Trends in Machine Learning* 5 (1): 1--122.


Lattimore, T., and C. Szepesvari. 2020. *Bandit Algorithms*. Cambridge University Press.


Russo, D., B. Van Roy, A. Kazerouni, I. Osband, and Z. Wen. 2018. "A Tutorial on Thompson Sampling." *Foundations and Trends in Machine Learning* 11 (1): 1--96.


Srinivas, N., A. Krause, S. M. Kakade, and M. Seeger. 2010. "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design." *International Conference on Machine Learning*.
