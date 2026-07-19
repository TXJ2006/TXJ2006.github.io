---
title: "Concentration Bounds in Bandit Analysis"
subtitle: "Hoeffding, Bernstein, and Sub-Gaussian Tails"
summary: "Fixed-time bounds, confidence radii, and UCB-style reasoning."
description: "Fixed-time bounds, confidence radii, and UCB-style reasoning."
date: 2026-06-18
lastmod: 2026-06-18
weight: 40
tags: ["Concentration", "UCB", "Probability"]
draft: false
ShowToc: false
hideMeta: true
---

## The Error Bar Is the Algorithm

A bandit algorithm looks like a decision rule. It chooses an arm. It receives a reward. It chooses again. But inside almost every clean bandit proof there is a smaller object doing most of the work: an error bar.

If the empirical mean of an arm is the number we can see, the true mean is the number we want. The distance between them is invisible. Concentration inequalities are the tools that make this invisible distance usable.

> **Key idea.**
>
> A concentration bound is not just a probability lemma. In bandit theory it becomes an algorithmic object. It tells the learner how much optimism is still defensible after seeing a certain amount of data.

The whole story of UCB can be read as one sentence: $\text{choose the arm with the largest plausible mean.}$ The word "plausible" is where probability enters.

### A small picture

Consider four actions. Each has an unknown click probability. The learner sees only the reward of the action it chose.


The hard part is not that rewards are random. The hard part is that the data are produced by our past decisions. If an arm looks bad early, we may stop sampling it. If it looks good early, we may sample it more. The sample size of each arm is itself a random outcome of the algorithm.

This is why concentration in bandits must be slightly more careful than concentration in a fixed dataset.

## One Random Number, Then Many

We start from the smallest possible object.

A random variable is a number we do not know yet. If $X$ is a Bernoulli reward, then

<div class="display-equation">
$$
X =
\begin{cases}
1, & \text{click or success},\\
0, & \text{no click or failure}.
\end{cases}
$$
</div>

 Its mean is $\mu = \E[X].$ For a Bernoulli random variable with success probability $p$,

<div class="display-equation">
$$
\E[X]
= 1\cdot p + 0\cdot (1-p)
= p.
$$
</div>


Now suppose we pull the same arm $n$ times and obtain $X_1,X_2,\ldots,X_n.$ The empirical mean is $\widehat\mu_n = \frac{1}{n}\sum_{i=1}^{n}X_i.$ It is the most natural estimate of $\mu$ because

<div class="display-equation">
$$
\E[\widehat\mu_n]
= \E\left[\frac{1}{n}\sum_{i=1}^{n}X_i\right]
= \frac{1}{n}\sum_{i=1}^{n}\E[X_i]
= \frac{1}{n}\cdot n\mu
= \mu.
$$
</div>


> **Think.**
>
> The sample mean is unbiased, but unbiased does not mean correct. It means it is right on average over repeated worlds. In one actual run, it can still be too high or too low. Concentration asks: by how much?

### The event we care about

The basic bad event is $\left\{\widehat\mu_n - \mu \ge r\right\}.$ This event says: the empirical mean is too optimistic by at least $r$.

The two-sided bad event is

<div class="display-equation">
$$
\left\{|\widehat\mu_n - \mu| \ge r\right\}
= \left\{\widehat\mu_n - \mu \ge r\right\}
\cup
\left\{\mu - \widehat\mu_n \ge r\right\}.
$$
</div>

 Therefore

<div class="display-equation">
$$
\Pbb\!\left(|\widehat\mu_n - \mu| \ge r\right)
\le
\Pbb\!\left(\widehat\mu_n - \mu \ge r\right)
+
\Pbb\!\left(\mu - \widehat\mu_n \ge r\right).
$$
</div>

 This is the union bound. It says that if either of two things can go wrong, the chance that something goes wrong is at most the sum of the two chances.

## Hoeffding: The First Error Bar

Hoeffding's inequality is the first concentration bound one should learn for bandits. It is simple, robust, and often good enough.

> **Result.**
>
> Let $X_1,\ldots,X_n$ be independent random variables in $[0,1]$ with common mean $\mu$. Then for every $r>0$, $\Pbb\!\left(\widehat\mu_n-\mu\ge r\right) \le \exp(-2nr^2),$ and $\Pbb\!\left(|\widehat\mu_n-\mu|\ge r\right) \le 2\exp(-2nr^2).$

The statement says that the probability of a large error decays exponentially in $nr^2$. More data makes the error smaller. A larger error is less likely.

### The proof idea in one line

A probability of a large deviation is hard to bound directly. Hoeffding's proof turns the deviation into an exponential moment:

<div class="display-equation">
$$
\Pbb(S_n\ge nr)
\quad\longrightarrow\quad
\E[\exp(\lambda S_n)],
$$
</div>

 where $S_n = \sum_{i=1}^n (X_i-\mu).$ The exponential function is useful because products appear when independent variables are added.

### Step-by-step proof

Let $S_n = \sum_{i=1}^{n}(X_i-\mu).$ Then

<div class="display-equation">
$$
\widehat\mu_n-\mu
= \frac{1}{n}\sum_{i=1}^{n}(X_i-\mu)
= \frac{S_n}{n}.
$$
</div>

 Hence

<div class="display-equation">
$$
\left\{\widehat\mu_n-\mu\ge r\right\}
= \left\{S_n\ge nr\right\}.
$$
</div>

 For any $\lambda>0$,

<div class="display-equation">
$$
S_n\ge nr
\quad\Longleftrightarrow\quad
\exp(\lambda S_n)\ge \exp(\lambda nr).
$$
</div>

 By Markov's inequality,

<div class="display-equation">
$$
\Pbb\!\left(\exp(\lambda S_n)\ge \exp(\lambda nr)\right)
\le
\frac{\E[\exp(\lambda S_n)]}{\exp(\lambda nr)}.
$$
</div>

 Therefore

<div class="display-equation">
$$
\Pbb(S_n\ge nr)
\le
\exp(-\lambda nr)\E[\exp(\lambda S_n)].
$$
</div>

 Now expand $S_n$:

<div class="display-equation">
$$
\E[\exp(\lambda S_n)]
= \E\left[\exp\left(\lambda\sum_{i=1}^{n}(X_i-\mu)\right)\right].
$$
</div>

 Since the exponential of a sum is a product, $= \E\left[\prod_{i=1}^{n}\exp\left(\lambda(X_i-\mu)\right)\right].$ Since $X_i$ are independent, $= \prod_{i=1}^{n}\E\left[\exp\left(\lambda(X_i-\mu)\right)\right].$ Hoeffding's lemma for a centered random variable in an interval of length $1$ gives

<div class="display-equation">
$$
\E\left[\exp\left(\lambda(X_i-\mu)\right)\right]
\le \exp\left(\frac{\lambda^2}{8}\right).
$$
</div>

 Thus

<div class="display-equation">
$$
\E[\exp(\lambda S_n)]
\le
\prod_{i=1}^{n}\exp\left(\frac{\lambda^2}{8}\right)
= \exp\left(\frac{n\lambda^2}{8}\right).
$$
</div>

 Putting this back into the probability bound,

<div class="display-equation">
$$
\Pbb(S_n\ge nr)
\le
\exp(-\lambda nr)\exp\left(\frac{n\lambda^2}{8}\right)
= \exp\left(-\lambda nr+\frac{n\lambda^2}{8}\right).
$$
</div>

 This is true for every $\lambda>0$. We choose the best $\lambda$.

Let $\phi(\lambda)=-\lambda nr+\frac{n\lambda^2}{8}.$ Then $\phi'(\lambda)=-nr+\frac{n\lambda}{4}.$ Set $\phi'(\lambda)=0$:

<div class="display-equation">
$$
-nr+\frac{n\lambda}{4}=0
\quad\Longrightarrow\quad
\lambda=4r.
$$
</div>

 Substitute $\lambda=4r$:

<div class="display-equation">
$$
\phi(4r)=-(4r)nr+\frac{n(4r)^2}{8}
= -4nr^2+\frac{16nr^2}{8}
= -4nr^2+2nr^2
= -2nr^2.
$$
</div>

 Therefore

<div class="display-equation">
$$
\Pbb(\widehat\mu_n-\mu\ge r)
= \Pbb(S_n\ge nr)
\le \exp(-2nr^2).
$$
</div>

 The same argument applied to $-S_n$ gives

<div class="display-equation">
$$
\Pbb(\mu-\widehat\mu_n\ge r)
\le \exp(-2nr^2).
$$
</div>

 Finally,

<div class="display-equation">
$$
\Pbb(|\widehat\mu_n-\mu|\ge r)
\le
\exp(-2nr^2)+\exp(-2nr^2)
=2\exp(-2nr^2).
$$
</div>


> **Key idea.**
>
> Hoeffding turns "the sample mean might be wrong" into a number. That number is an error bar.

## From Tail Bound to Confidence Radius

The inequality

<div class="display-equation">
$$
\Pbb(|\widehat\mu_n-\mu|\ge r)
\le 2\exp(-2nr^2)
$$
</div>

 is not yet an algorithm. An algorithm needs a radius $r$.

Choose a failure probability $\delta\in(0,1)$. We want $2\exp(-2nr^2)\le \delta.$ Now solve for $r$ step by step: $2\exp(-2nr^2)\le \delta,$ $\exp(-2nr^2)\le \frac{\delta}{2},$ $-2nr^2 \le \log\left(\frac{\delta}{2}\right),$ $2nr^2 \ge \log\left(\frac{2}{\delta}\right),$ $r^2 \ge \frac{\log(2/\delta)}{2n},$ $r \ge \sqrt{\frac{\log(2/\delta)}{2n}}.$ Therefore the Hoeffding confidence radius is

<div class="display-equation">
$$
\boxed{
\operatorname{rad}_{\rm H}(n,\delta)
=\sqrt{\frac{\log(2/\delta)}{2n}}.
}
$$
</div>

 With probability at least $1-\delta$,

<div class="display-equation">
$$
\mu\in
\left[
\widehat\mu_n-\operatorname{rad}_{\rm H}(n,\delta),
\widehat\mu_n+\operatorname{rad}_{\rm H}(n,\delta)
\right].
$$
</div>


> **Think.**
>
> Do not remember only the formula. Remember the tradeoff: the radius shrinks like $1/\sqrt n$ and grows like $\sqrt{\log(1/\delta)}$. Confidence is not free. Asking for a very small failure probability makes the interval wider.

## Why Bandits Need Many Error Bars at Once

In supervised learning, one often analyzes a fixed estimator at a fixed sample size. In bandits, the learner keeps looking, choosing, and updating. At time $t$, each arm $a$ has its own random number of pulls $N_a(t)=\sum_{s=1}^{t}\one\{A_s=a\}.$ The sample mean of arm $a$ is

<div class="display-equation">
$$
\widehat\mu_a(t)
=\frac{1}{N_a(t)}\sum_{s=1}^{t}\one\{A_s=a\}X_s,
$$
</div>

 when $N_a(t)>0$.

The learner wants all confidence intervals to be correct at all relevant times. A simple way to do this is to buy many error bars with one failure budget.

Suppose there are $K$ arms and $T$ rounds. There are at most $KT$ arm-time pairs. Give each pair failure probability $\delta' = \frac{\delta}{KT}.$ For one pair $(a,t)$, Hoeffding gives

<div class="display-equation">
$$
\Pbb\left(
|\widehat\mu_a(t)-\mu_a|
\sqrt{\frac{\log(2/\delta')}{2N_a(t)}}
\right)
\le \delta'.
$$
</div>

 By the union bound,

<div class="display-equation">
$$
\Pbb\left(\text{at least one arm-time interval fails}\right)
\le
\sum_{a=1}^{K}\sum_{t=1}^{T}\delta'
=KT\cdot \frac{\delta}{KT}
=\delta.
$$
</div>

 Thus, with probability at least $1-\delta$, every displayed error bar is correct.

> **Key idea.**
>
> The union bound is not a crude technicality here. It is the mechanism that lets an adaptive algorithm look at many arms many times without pretending that each look is isolated.

## UCB from Hoeffding

For arm $a$, define the optimistic index $U_a(t)=\widehat\mu_a(t)+\sqrt{\frac{2\log t}{N_a(t)}}.$ The exact constant is not sacred. The structure is sacred:

<div class="display-equation">
$$
\text{index} = \text{what we have seen} + \text{how wrong it could still be}.
$$
</div>

 At each round, $A_t\in \argmax_{a\in[K]} U_a(t).$

### Algorithm design

    def ucb_hoeffding_action(t, sums, counts):
        means = sums / counts
        bonus = np.sqrt(2.0 * np.log(t + 1.0) / counts)
        return int(np.argmax(means + bonus))

The code is short because the theorem is doing the work. The square-root term is the confidence radius.

### The standard pull-count calculation

Let $a^*$ be an optimal arm and let $\Delta_a=\mu_{a^*}-\mu_a>0$ for a suboptimal arm $a$.

On a good event, all confidence intervals are correct. For every arm $b$ and time $t$,

<div class="display-equation">
$$
|\widehat\mu_b(t)-\mu_b|
\le
\sqrt{\frac{2\log t}{N_b(t)}}.
$$
</div>

 If UCB chooses a suboptimal arm $a$ at time $t$, then $U_a(t)\ge U_{a^*}(t).$ Since the optimal arm's index is optimistic,

<div class="display-equation">
$$
U_{a^*}(t)
=\widehat\mu_{a^*}(t)+\sqrt{\frac{2\log t}{N_{a^*}(t)}}
\ge \mu_{a^*}.
$$
</div>

 Therefore $U_a(t)\ge \mu_{a^*}.$ But on the good event, $\widehat\mu_a(t)\le \mu_a+\sqrt{\frac{2\log t}{N_a(t)}}.$ Hence

<div class="display-equation">
$$
U_a(t)
=\widehat\mu_a(t)+\sqrt{\frac{2\log t}{N_a(t)}}
\le
\mu_a+2\sqrt{\frac{2\log t}{N_a(t)}}.
$$
</div>

 Combining the lower and upper bounds on $U_a(t)$,

<div class="display-equation">
$$
\mu_{a^*}
\le
\mu_a+2\sqrt{\frac{2\log t}{N_a(t)}}.
$$
</div>

 Subtract $\mu_a$:

<div class="display-equation">
$$
\Delta_a
\le
2\sqrt{\frac{2\log t}{N_a(t)}}.
$$
</div>

 Divide by $2$:

<div class="display-equation">
$$
\frac{\Delta_a}{2}
\le
\sqrt{\frac{2\log t}{N_a(t)}}.
$$
</div>

 Square both sides:

<div class="display-equation">
$$
\frac{\Delta_a^2}{4}
\le
\frac{2\log t}{N_a(t)}.
$$
</div>

 Rearrange:

<div class="display-equation">
$$
N_a(t)
\le
\frac{8\log t}{\Delta_a^2}.
$$
</div>

 Thus a suboptimal arm can be pulled many times only if its uncertainty is still large. Once it has been sampled enough, optimism can no longer hide its gap.

The regret contribution of arm $a$ is approximately $\Delta_a \E[N_a(T)].$ The calculation above explains the familiar logarithmic shape:

<div class="display-equation">
$$
\Delta_a N_a(T)
\lesssim
\Delta_a\cdot \frac{8\log T}{\Delta_a^2}
=
\frac{8\log T}{\Delta_a}.
$$
</div>


## Bernstein: When Variance Matters

Hoeffding only uses the fact that rewards lie in $[0,1]$. It does not ask whether the arm is noisy or nearly deterministic.

But this matters. A Bernoulli arm with mean $0.5$ has variance $0.5(1-0.5)=0.25.$ A Bernoulli arm with mean $0.05$ has variance $0.05(1-0.05)=0.0475.$ Both rewards lie in $[0,1]$, but the second one fluctuates much less.

Bernstein-type bounds use this variance information.

> **Result.**
>
> A simple Bernstein-style confidence radius has the form

<div class="display-equation">
$$
\operatorname{rad}_{\rm B}(n,\delta,\sigma^2)
=
\sqrt{\frac{2\sigma^2\log(2/\delta)}{n}}
+
\frac{\log(2/\delta)}{3n}.
$$
</div>

 The first term is variance-sensitive. The second term is a boundedness correction.

### How to read the formula

Compare Hoeffding and Bernstein:

<div class="display-equation">
$$
\operatorname{rad}_{\rm H}(n,\delta)
=\sqrt{\frac{\log(2/\delta)}{2n}},
$$
</div>



<div class="display-equation">
$$
\operatorname{rad}_{\rm B}(n,\delta,\sigma^2)
=\sqrt{\frac{2\sigma^2\log(2/\delta)}{n}}
+\frac{\log(2/\delta)}{3n}.
$$
</div>

 If $\sigma^2$ is small, the square-root term is small. For large $n$, the $1/n$ correction becomes smaller than the $1/\sqrt n$ term, and the variance advantage becomes visible.

![For a low-variance Bernoulli arm with p = 0.05, a variance-aware radius can be substantially shorter than Hoeffding’s radius.](/images/notes/assets/concentration/confidence_radius_widths.webp)

*For a low-variance Bernoulli arm with p = 0.05, a variance-aware radius can be substantially shorter than Hoeffding’s radius.*

### Empirical Bernstein UCB

In a bandit problem, $\sigma_a^2$ is usually unknown. We estimate it.

For arm $a$, define

<div class="display-equation">
$$
\widehat\sigma_a^2(t)
=\frac{1}{N_a(t)}\sum_{s:A_s=a}\left(X_s-\widehat\mu_a(t)\right)^2.
$$
</div>

 Then an empirical Bernstein-style index is

<div class="display-equation">
$$
U_a^{\rm EB}(t)
=\widehat\mu_a(t)
+
\sqrt{\frac{2\widehat\sigma_a^2(t)\log t}{N_a(t)}}
+
\frac{3\log t}{N_a(t)}.
$$
</div>


    def ucb_bernstein_action(t, sums, sq_sums, counts):
        means = sums / counts
        variances = np.maximum(0.0, sq_sums / counts - means * means)
        log_term = np.log(t + 1.0)
        bonus = np.sqrt(2.0 * variances * log_term / counts) + 3.0 * log_term / counts
        return int(np.argmax(means + bonus))

This is the same design pattern as UCB-Hoeffding, but the error bar now listens to the observed variance.

## Sub-Gaussian Tails: A Cleaner Language

Hoeffding is excellent for bounded variables. But many models are not naturally bounded. A Gaussian reward, for example, can take any real value.

A random variable $X$ with mean $\mu$ is called $\sigma$-sub-Gaussian if, for every $\lambda\in\R$,

<div class="display-equation">
$$
\E\left[\exp\left(\lambda(X-\mu)\right)\right]
\le
\exp\left(\frac{\lambda^2\sigma^2}{2}\right).
$$
</div>


This definition says: the exponential moments of $X-\mu$ are no larger than those of a Gaussian with variance proxy $\sigma^2$.

### Tail bound from the definition

Let $X_1,\ldots,X_n$ be independent $\sigma$-sub-Gaussian random variables with mean $\mu$. Let $\widehat\mu_n=\frac{1}{n}\sum_{i=1}^{n}X_i.$ We derive a one-sided tail bound.

Set $S_n=\sum_{i=1}^{n}(X_i-\mu).$ Then $\widehat\mu_n-\mu=\frac{S_n}{n}.$ For $\lambda>0$,

<div class="display-equation">
$$
\Pbb(\widehat\mu_n-\mu\ge r)
=\Pbb(S_n\ge nr)
$$
</div>

 $=\Pbb(\exp(\lambda S_n)\ge \exp(\lambda nr)).$ By Markov's inequality,

<div class="display-equation">
$$
\Pbb(S_n\ge nr)
\le
\exp(-\lambda nr)\E[\exp(\lambda S_n)].
$$
</div>

 By independence,

<div class="display-equation">
$$
\E[\exp(\lambda S_n)]
=\prod_{i=1}^{n}\E[\exp(\lambda(X_i-\mu))].
$$
</div>

 By sub-Gaussianity,

<div class="display-equation">
$$
\le
\prod_{i=1}^{n}\exp\left(\frac{\lambda^2\sigma^2}{2}\right)
=
\exp\left(\frac{n\lambda^2\sigma^2}{2}\right).
$$
</div>

 Therefore

<div class="display-equation">
$$
\Pbb(\widehat\mu_n-\mu\ge r)
\le
\exp\left(-\lambda nr+\frac{n\lambda^2\sigma^2}{2}\right).
$$
</div>

 Choose the best $\lambda$.

Let $\psi(\lambda)=-\lambda nr+\frac{n\lambda^2\sigma^2}{2}.$ Then $\psi'(\lambda)=-nr+n\lambda\sigma^2.$ Set $\psi'(\lambda)=0$:

<div class="display-equation">
$$
-nr+n\lambda\sigma^2=0
\quad\Longrightarrow\quad
\lambda=\frac{r}{\sigma^2}.
$$
</div>

 Substitute:

<div class="display-equation">
$$
\psi\left(\frac{r}{\sigma^2}\right)
= -\frac{r}{\sigma^2}nr
+\frac{n}{2}\frac{r^2}{\sigma^4}\sigma^2
= -\frac{nr^2}{\sigma^2}+\frac{nr^2}{2\sigma^2}
= -\frac{nr^2}{2\sigma^2}.
$$
</div>

 Thus

<div class="display-equation">
$$
\Pbb(\widehat\mu_n-\mu\ge r)
\le
\exp\left(-\frac{nr^2}{2\sigma^2}\right).
$$
</div>

 The two-sided version is

<div class="display-equation">
$$
\Pbb(|\widehat\mu_n-\mu|\ge r)
\le
2\exp\left(-\frac{nr^2}{2\sigma^2}\right).
$$
</div>


Solving $2\exp\left(-\frac{nr^2}{2\sigma^2}\right)\le\delta$ gives

<div class="display-equation">
$$
\boxed{
\operatorname{rad}_{\rm SG}(n,\delta,\sigma)
=\sigma\sqrt{\frac{2\log(2/\delta)}{n}}.
}
$$
</div>


> **Key idea.**
>
> Sub-Gaussianity is the cleanest language for many bandit papers. It says that averages concentrate like Gaussian averages, even when the rewards are not exactly Gaussian.

## A Small Experiment

We now run a minimal experiment. The environment has four Bernoulli arms with means $(0.01,\,0.03,\,0.05,\,0.07).$ The best arm is only slightly better than the others, and all rewards are low-variance because clicks are rare.

We compare four policies:

<div class="display-equation">
$$
\text{Greedy},\quad \text{UCB-Hoeffding},\quad \text{UCB-Bernstein},\quad \text{Thompson sampling}.
$$
</div>

 The point is not to crown a universal winner. The point is to see how different uses of uncertainty change behavior.

### Fixed-time coverage

First, we verify the simplest message: Hoeffding intervals are conservative at a fixed time.
| $n$ | Hoeffding radius | Empirical miss probability |
|------:|:----------------:|:--------------------------:|
|    10 |      0.4295      |          0.00164           |
|    20 |      0.3037      |          0.00134           |
|    50 |      0.1921      |          0.00326           |
|   100 |      0.1358      |          0.00338           |
|   200 |      0.0960      |          0.00260           |
|   500 |      0.0607      |          0.00290           |
|  1000 |      0.0429      |          0.00358           |

<p class="table-caption">Empirical miss probability for Hoeffding intervals with nominal $\delta=0.05$. The true Bernoulli mean is $p=0.30$, and each row uses $50{,}000$ Monte Carlo repetitions.</p>

![The empirical probability of missing the true mean stays below the nominal failure level. The bound is deliberately safe.](/images/notes/assets/concentration/fixed_time_hoeffding.webp)

*The empirical probability of missing the true mean stays below the nominal failure level. The bound is deliberately safe.*

### Bandit results

The following table reports average final regret over $150$ independent runs and $T=3000$ rounds.
| Policy        | Final regret |   Arm 0 |  Arm 1 |  Arm 2 |   Arm 3 |
|:--------------|-------------:|--------:|-------:|-------:|--------:|
| Greedy        |       157.51 | 2557.59 |  61.11 |  80.71 |  300.60 |
| UCB-Hoeffding |        70.64 |  492.59 | 627.50 | 799.19 | 1080.73 |
| UCB-Bernstein |        51.74 |  291.75 | 475.73 | 760.27 | 1472.25 |
| Thompson      |        22.22 |   97.34 | 176.99 | 465.01 | 2260.65 |

<p class="table-caption">Average final regret and average number of pulls per arm.</p>

Greedy often gets trapped by early noise. UCB-Hoeffding keeps exploring because the error bars are wide. UCB-Bernstein uses the low observed variance to reduce unnecessary exploration. Thompson sampling, although not a concentration-index algorithm, also converts uncertainty into action probabilities and quickly allocates more mass to the best arm.

![Average cumulative regret over 150 runs. Concentration controls how long uncertainty can justify exploration.](/images/notes/assets/concentration/ucb_regret_curves.webp)

*Average cumulative regret over 150 runs. Concentration controls how long uncertainty can justify exploration.*

## Complete Experiment Code

The code below is the core of the simulation. The full script is included with this note.

    import numpy as np

    def run_bandit(policy, p, T, rng):
        K = len(p)
        counts = np.zeros(K, dtype=int)
        sums = np.zeros(K, dtype=float)
        sq_sums = np.zeros(K, dtype=float)
        regret = np.zeros(T, dtype=float)
        p_star = float(np.max(p))
        cumulative_regret = 0.0

        # Pull each arm once, so every empirical mean is defined.
        for t in range(min(K, T)):
            a = t
            r = rng.binomial(1, p[a])
            counts[a] += 1
            sums[a] += r
            sq_sums[a] += r * r
            cumulative_regret += p_star - p[a]
            regret[t] = cumulative_regret

        for t in range(K, T):
            means = sums / np.maximum(counts, 1)

            if policy == "Greedy":
                a = int(np.argmax(means))

            elif policy == "UCB-Hoeffding":
                bonus = np.sqrt(2.0 * np.log(t + 1.0) / counts)
                a = int(np.argmax(means + bonus))

            elif policy == "UCB-Bernstein":
                variances = np.maximum(0.0, sq_sums / counts - means * means)
                log_term = np.log(t + 1.0)
                bonus = np.sqrt(2.0 * variances * log_term / counts) \
                        + 3.0 * log_term / counts
                a = int(np.argmax(means + bonus))

            elif policy == "Thompson":
                alpha = sums + 1.0
                beta = counts - sums + 1.0
                a = int(np.argmax(rng.beta(alpha, beta)))

            r = rng.binomial(1, p[a])
            counts[a] += 1
            sums[a] += r
            sq_sums[a] += r * r
            cumulative_regret += p_star - p[a]
            regret[t] = cumulative_regret

        return regret, counts

## What to Remember

Concentration bounds are often introduced as probability facts. In bandit theory they become operational objects.

- Hoeffding gives a simple radius when rewards are bounded.

- Bernstein improves the radius when variance is small.

- Sub-Gaussian tails provide a clean language for unbounded but well-behaved noise.

- UCB is just empirical performance plus an error bar.

- The regret proof works because a suboptimal arm cannot remain plausibly optimal after enough samples.

The deepest point is not the square root. It is the conversion:

<div class="display-equation">
$$
\text{probability} \quad\longrightarrow\quad \text{confidence} \quad\longrightarrow\quad \text{action} \quad\longrightarrow\quad \text{regret control}.
$$
</div>


## Appendix A. Formula Sheet
| Object | Formula |
|:---|:---|
| Sample mean | $\widehat\mu_n = n^{-1}\sum_{i=1}^{n}X_i$ |
| Hoeffding tail | $\Pbb(|\widehat\mu_n-\mu|\ge r)\le 2e^{-2nr^2}$ |
| Hoeffding radius | $\sqrt{\log(2/\delta)/(2n)}$ |
| Sub-Gaussian MGF | $\E[e^{\lambda(X-\mu)}]\le e^{\lambda^2\sigma^2/2}$ |
| Sub-Gaussian radius | $\sigma\sqrt{2\log(2/\delta)/n}$ |
| Bernstein radius | $\sqrt{2\sigma^2\log(2/\delta)/n}+\log(2/\delta)/(3n)$ |
| Hoeffding-UCB index | $\widehat\mu_a(t)+\sqrt{2\log t/N_a(t)}$ |
| Empirical-Bernstein index | $\widehat\mu_a(t)+\sqrt{2\widehat\sigma_a^2(t)\log t/N_a(t)}+3\log t/N_a(t)$ |

<p class="table-caption">Core formulas.</p>

## Appendix B. Notation Table
| Symbol            | Meaning                                                 |
|:------------------|:--------------------------------------------------------|
| $X_i$           | reward sample                                           |
| $\mu$           | true mean of a reward distribution                      |
| $\widehat\mu_n$ | empirical mean after $n$ samples                      |
| $\delta$        | allowed failure probability                             |
| $r$             | confidence radius                                       |
| $K$             | number of arms                                          |
| $T$             | time horizon                                            |
| $A_t$           | arm chosen at time $t$                                |
| $N_a(t)$        | number of times arm $a$ has been pulled by time $t$ |
| $\mu_a$         | true mean of arm $a$                                  |
| $a^*$           | an optimal arm                                          |
| $\Delta_a$      | gap $\mu_{a^*}-\mu_a$                                 |
| $U_a(t)$        | optimistic index for arm $a$ at time $t$            |

<p class="table-caption">Notation.</p>

## Appendix C. Full Experiment Script

``` {.python style="blogcode" language="Python" caption="Full experiment script."}
"""Experiments for the lecture note
Concentration Bounds in Bandit Analysis.

The script generates three figures and one CSV file:
  1. fixed_time_hoeffding.pdf/png
  2. confidence_radius_widths.pdf/png
  3. ucb_regret_curves.pdf/png
  4. concentration_results.csv

Only numpy, pandas, and matplotlib are used.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent


def hoeffding_radius(n: np.ndarray | float, delta: float) -> np.ndarray | float:
    """Two-sided Hoeffding confidence radius for variables in [0,1]."""
    return np.sqrt(np.log(2.0 / delta) / (2.0 * np.asarray(n)))


def bernstein_radius_oracle(n: np.ndarray | float, delta: float, var: float) -> np.ndarray | float:
    """A simple variance-aware Bernstein-style radius."""
    x = np.log(2.0 / delta)
    n = np.asarray(n)
    return np.sqrt(2.0 * var * x / n) + x / (3.0 * n)


def fixed_time_coverage(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = 0.30
    delta = 0.05
    trials = 50_000
    ns = np.array([10, 20, 50, 100, 200, 500, 1000])
    rows = []
    empirical = []
    hoeffding_bounds = []
    for n in ns:
        samples = rng.binomial(1, p, size=(trials, n))
        means = samples.mean(axis=1)
        r = hoeffding_radius(n, delta)
        miss = np.mean(np.abs(means - p) > r)
        rows.append({
            "experiment": "fixed_time_coverage",
            "n": int(n),
            "delta": delta,
            "p": p,
            "hoeffding_radius": float(r),
            "empirical_miss_probability": float(miss),
        })
        empirical.append(miss)
        hoeffding_bounds.append(delta)

    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    ax.plot(ns, empirical, marker="o", label="Empirical miss probability")
    ax.plot(ns, hoeffding_bounds, linestyle="--", label="Nominal delta = 0.05")
    ax.set_xscale("log")
    ax.set_xlabel("sample size n")
    ax.set_ylabel("probability")
    ax.set_title("Fixed-time Hoeffding intervals are conservative")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fixed_time_hoeffding.pdf")
    fig.savefig(OUT / "fixed_time_hoeffding.png", dpi=220)
    plt.close(fig)
    return pd.DataFrame(rows)


def radius_width_plot() -> pd.DataFrame:
    delta = 0.05
    p = 0.05
    var = p * (1.0 - p)
    ns = np.arange(5, 1001)
    h = hoeffding_radius(ns, delta)
    b = bernstein_radius_oracle(ns, delta, var)

    fig, ax = plt.subplots(figsize=(6.5, 4.1))
    ax.plot(ns, h, label="Hoeffding radius")
    ax.plot(ns, b, label="Bernstein radius using variance")
    ax.set_xlabel("sample size n")
    ax.set_ylabel("radius")
    ax.set_title("Variance-aware error bars can be much shorter")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "confidence_radius_widths.pdf")
    fig.savefig(OUT / "confidence_radius_widths.png", dpi=220)
    plt.close(fig)

    return pd.DataFrame({
        "experiment": "radius_widths",
        "n": ns,
        "delta": delta,
        "p": p,
        "variance": var,
        "hoeffding_radius": h,
        "bernstein_radius": b,
    })


def run_bandit(policy: str, p: np.ndarray, T: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    K = len(p)
    counts = np.zeros(K, dtype=int)
    sums = np.zeros(K, dtype=float)
    sq_sums = np.zeros(K, dtype=float)
    regret = np.zeros(T, dtype=float)
    p_star = float(np.max(p))
    cumulative_regret = 0.0

    for t in range(min(K, T)):
        a = t
        r = rng.binomial(1, p[a])
        counts[a] += 1
        sums[a] += r
        sq_sums[a] += r * r
        cumulative_regret += p_star - p[a]
        regret[t] = cumulative_regret

    for t in range(K, T):
        means = sums / np.maximum(counts, 1)
        if policy == "Greedy":
            a = int(np.argmax(means))
        elif policy == "UCB-Hoeffding":
            bonus = np.sqrt(2.0 * np.log(t + 1.0) / counts)
            a = int(np.argmax(means + bonus))
        elif policy == "UCB-Bernstein":
            variances = np.zeros(K)
            for i in range(K):
                n = counts[i]
                if n <= 1:
                    variances[i] = 0.25
                else:
                    m = means[i]
                    variances[i] = max(0.0, sq_sums[i] / n - m * m)
            log_term = np.log(t + 1.0)
            bonus = np.sqrt(2.0 * variances * log_term / counts) + 3.0 * log_term / counts
            a = int(np.argmax(means + bonus))
        elif policy == "Thompson":
            alpha = sums + 1.0
            beta = counts - sums + 1.0
            a = int(np.argmax(rng.beta(alpha, beta)))
        else:
            raise ValueError(f"Unknown policy: {policy}")

        r = rng.binomial(1, p[a])
        counts[a] += 1
        sums[a] += r
        sq_sums[a] += r * r
        cumulative_regret += p_star - p[a]
        regret[t] = cumulative_regret

    return regret, counts


def bandit_experiment(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = np.array([0.01, 0.03, 0.05, 0.07])
    T = 3000
    trials = 150
    policies = ["Greedy", "UCB-Hoeffding", "UCB-Bernstein", "Thompson"]
    mean_regrets = {}
    rows = []

    for policy in policies:
        regrets = []
        counts_all = []
        for _ in range(trials):
            regret, counts = run_bandit(policy, p, T, rng)
            regrets.append(regret)
            counts_all.append(counts)
        regrets = np.asarray(regrets)
        counts_all = np.asarray(counts_all)
        mean_regrets[policy] = regrets.mean(axis=0)
        row = {"experiment": "bandit_regret", "policy": policy, "T": T, "trials": trials,
               "final_regret": float(mean_regrets[policy][-1])}
        for i, val in enumerate(counts_all.mean(axis=0)):
            row[f"mean_pulls_arm_{i}"] = float(val)
        rows.append(row)

    fig, ax = plt.subplots(figsize=(6.7, 4.3))
    x = np.arange(1, T + 1)
    for policy in policies:
        ax.plot(x, mean_regrets[policy], label=policy)
    ax.set_xlabel("round t")
    ax.set_ylabel("average cumulative regret")
    ax.set_title("Concentration turns uncertainty into action")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "ucb_regret_curves.pdf")
    fig.savefig(OUT / "ucb_regret_curves.png", dpi=220)
    plt.close(fig)

    return pd.DataFrame(rows)


def main() -> None:
    tables = [fixed_time_coverage(), radius_width_plot(), bandit_experiment()]
    out = pd.concat(tables, ignore_index=True, sort=False)
    out.to_csv(OUT / "concentration_results.csv", index=False)
    print(out.tail(12).to_string(index=False))


if __name__ == "__main__":
    main()
```

## References
Audibert, Jean-Yves, Rémi Munos, and Csaba Szepesvári. 2009. "Exploration-Exploitation Tradeoff Using Variance Estimates in Multi-Armed Bandits." *Theoretical Computer Science* 410 (19): 1876--902.


Auer, P., N. Cesa-Bianchi, and P. Fischer. 2002. "Finite-Time Analysis of the Multiarmed Bandit Problem." *Machine Learning* 47: 235--56.


Boucheron, Stéphane, Gábor Lugosi, and Pascal Massart. 2013. *Concentration Inequalities: A Nonasymptotic Theory of Independence*. Oxford University Press.


Bubeck, S., and N. Cesa-Bianchi. 2012. "Regret Analysis of Stochastic and Nonstochastic Multi-Armed Bandit Problems." *Foundations and Trends in Machine Learning* 5 (1): 1--122.


Lattimore, T., and C. Szepesvari. 2020. *Bandit Algorithms*. Cambridge University Press.
