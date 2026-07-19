---
title: "Change of Measure for Bandit Lower Bounds"
subtitle: "How Nearby Worlds Force Every Algorithm to Gather Evidence"
summary: "Alternative environments, indistinguishability, and information-theoretic lower bounds."
description: "Alternative environments, indistinguishability, and information-theoretic lower bounds."
date: 2026-06-20
lastmod: 2026-06-20
weight: 80
tags: ["Change of Measure", "Lower Bounds", "Best-Arm Identification"]
draft: false
ShowToc: false
hideMeta: true
---

## A Lower Bound Begins by Inventing Another World

An upper bound explains what one particular algorithm can achieve. A lower bound asks for something more stubborn:


*What must every algorithm pay before it can be reliably correct?*


The cleanest answer begins with two possible worlds.

In the first world, arm 1 is best. In the second, arm 2 is best. The learner runs the same code in both worlds. It sees rewards one at a time, changes its sampling choices after each observation, and eventually announces an answer.

If the two worlds produce almost the same data, the learner cannot behave very differently in them. If we nevertheless demand the correct answer in both worlds, then the learner must collect enough observations to make the worlds distinguishable.

That sentence is the entire philosophy of change-of-measure lower bounds.

> **Diagram.** A lower bound compares the same algorithm under two environments that require different answers.

The proof strategy has three moving parts:

1.  construct an alternative environment that changes the correct answer;

2.  measure how much statistical evidence each observation supplies against that alternative;

3.  show that reliable behavior in the two environments requires a minimum amount of total evidence.

The first step is model design. The second is information accounting. The third is hypothesis testing.

> **Key idea.**
>
> A lower bound does not argue that an algorithm is clumsy. It argues that two worlds are too similar for *any* algorithm to separate without paying for information.

## One Coin, Two Possible Biases

We begin without bandits. Suppose a coin has bias either $p$ or $q$, where $p\neq q$. Let

<div class="display-equation">
$$
P=\Ber(p),
\qquad
Q=\Ber(q).
$$
</div>

 One toss produces $X\in\{0,1\}$.

If $X=1$, the probability of the observation is $p$ under $P$ and $q$ under $Q$. If $X=0$, the probabilities are $1-p$ and $1-q$. The likelihood ratio is therefore

<div class="display-equation">
$$
\frac{P(X)}{Q(X)}
=
\begin{cases}
\dfrac{p}{q}, & X=1,\\[0.8em]
\dfrac{1-p}{1-q}, & X=0.
\end{cases}
$$
</div>

 Its logarithm can be written without cases:

<div class="numbered-equation" id="eq:one-step-llr">
$$
\ell(X)
=
X\log\frac{p}{q}
+(1-X)\log\frac{1-p}{1-q}.
$$
<span class="equation-number" aria-label="Equation 1">(1)</span>
</div>


A positive value favors $P$; a negative value favors $Q$.

### Evidence from several observations adds

For independent observations $X_1,\ldots,X_n$,

<div class="display-equation">
$$
\begin{align*}
\frac{P^n(X_1,\ldots,X_n)}{Q^n(X_1,\ldots,X_n)}
&=
\prod_{i=1}^{n}\frac{P(X_i)}{Q(X_i)},\\
L_n
:=
\log\frac{P^n(X_1,\ldots,X_n)}{Q^n(X_1,\ldots,X_n)}
&=
\sum_{i=1}^{n}\ell(X_i).
\end{align*}
$$
</div>


The log-likelihood ratio is a ledger. Every observation adds one entry.

Under $P$, the expected entry is

<div class="display-equation">
$$
\begin{align*}
\E_P[\ell(X)]
&=
\E_P\left[
X\log\frac{p}{q}
+(1-X)\log\frac{1-p}{1-q}
\right]\\
&=
\E_P[X]\log\frac{p}{q}
+\E_P[1-X]\log\frac{1-p}{1-q}\\
&=
p\log\frac{p}{q}
+(1-p)\log\frac{1-p}{1-q}\\
&=
\kl(p,q).
\end{align*}
$$
</div>

 Hence

<div class="numbered-equation" id="eq:iid-evidence-drift">
$$
\boxed{
\E_P[L_n]=n\,\kl(p,q).
}
$$
<span class="equation-number" aria-label="Equation 2">(2)</span>
</div>


KL divergence is not merely a distance-like quantity. In this experiment it is the expected evidence supplied by one observation.

![Eight evidence paths under $P=\Ber(0.55)$ against $Q=\Ber(0.45)$. Individual paths wander, while the expected drift is $t\kl(0.55,0.45)$.](/images/notes/assets/change-of-measure/evidence_paths.webp)

*Eight evidence paths under $P=\Ber(0.55)$ against $Q=\Ber(0.45)$. Individual paths wander, while the expected drift is $t\kl(0.55,0.45)$.*

Some paths can favor the wrong model for a long time. That is not a defect in the calculation. It is the reason a lower bound exists. Evidence arrives with noise even when its average direction is correct.

> **Think.**
>
> If $p$ and $q$ move closer, then $\kl(p,q)$ shrinks. The expected evidence per toss shrinks with it. A hard alternative is therefore not a wildly different world; it is a nearby world that changes the answer.

## Changing Measure, Without Mystery

Let $P$ and $Q$ be two probability laws with densities $p$ and $q$ on the same sample space. Define $L(x)=\log\frac{p(x)}{q(x)}.$ Then $q(x)=p(x)e^{-L(x)}.$ For any event $E$,

<div class="display-equation">
$$
\begin{align*}
Q(E)
&=\int_E q(x)\,\dd x\\
&=\int_E p(x)e^{-L(x)}\,\dd x\\
&=\E_P\left[\one_E e^{-L}\right].
\end{align*}
$$
</div>

 Thus

<div class="numbered-equation" id="eq:change-measure-event">
$$
\boxed{
Q(E)=\E_P\left[\one_E e^{-L}\right].
}
$$
<span class="equation-number" aria-label="Equation 3">(3)</span>
</div>


This is the change-of-measure identity. It says that a probability in the $Q$-world can be computed by averaging in the $P$-world, provided each outcome is reweighted by how much more or less plausible it is under $Q$.

There is no algorithm in this identity. It is a statement about two probability laws. Bandit lower bounds become possible when we apply it to the law of an entire adaptive history.

### Compressing the whole experiment to one event

Suppose the final decision is summarized by an event $E$. Let

<div class="display-equation">
$$
\alpha=P(E),
\qquad
\beta=Q(E).
$$
</div>

 The entire observation may be complicated, but the event records only one bit: did $E$ occur?

Information cannot increase when data are compressed. In this binary case,

<div class="numbered-equation" id="eq:binary-data-processing">
$$
\boxed{
\KL(P\Vert Q)\geq \kl(\alpha,\beta).
}
$$
<span class="equation-number" aria-label="Equation 4">(4)</span>
</div>


We now derive it directly.

On the event $E$, write $P_E$ and $Q_E$ for the conditional laws. Then

<div class="display-equation">
$$
P(\dd x)=\alpha P_E(\dd x),
\qquad
Q(\dd x)=\beta Q_E(\dd x),
\qquad x\in E.
$$
</div>

 Therefore, on $E$,

<div class="display-equation">
$$
\frac{\dd P}{\dd Q}
=
\frac{\alpha}{\beta}
\frac{\dd P_E}{\dd Q_E}.
$$
</div>

 The contribution of $E$ to KL is

<div class="display-equation">
$$
\begin{align*}
\int_E \log\frac{\dd P}{\dd Q}\,\dd P
&=
\alpha\int_E
\log\left(
\frac{\alpha}{\beta}
\frac{\dd P_E}{\dd Q_E}
\right)\dd P_E\\
&=
\alpha\log\frac{\alpha}{\beta}
+
\alpha\KL(P_E\Vert Q_E).
\end{align*}
$$
</div>

 The same calculation on $E^c$ gives

<div class="display-equation">
$$
(1-\alpha)\log\frac{1-\alpha}{1-\beta}
+
(1-\alpha)\KL(P_{E^c}\Vert Q_{E^c}).
$$
</div>

 Adding both pieces,

<div class="display-equation">
$$
\begin{align*}
\KL(P\Vert Q)
&=
\alpha\log\frac{\alpha}{\beta}
+(1-\alpha)\log\frac{1-\alpha}{1-\beta}\\
&\quad
+
\alpha\KL(P_E\Vert Q_E)
+(1-\alpha)\KL(P_{E^c}\Vert Q_{E^c})\\
&\geq
\alpha\log\frac{\alpha}{\beta}
+(1-\alpha)\log\frac{1-\alpha}{1-\beta}\\
&=
\kl(\alpha,\beta).
\end{align*}
$$
</div>

 The inequality uses only the nonnegativity of KL divergence.

> **Key idea.**
>
> A lower-bound proof does not need to understand every feature of the final history. It can compress the history to the one event on which the two worlds must disagree.

### A useful testing inequality

Another common form is the Bretagnolle--Huber inequality. For every event $E$,

<div class="numbered-equation" id="eq:bh">
$$
\boxed{
P(E)+Q(E^c)
\geq
\frac{1}{2}\exp\{-\KL(P\Vert Q)\}.
}
$$
<span class="equation-number" aria-label="Equation 5">(5)</span>
</div>


Interpret $E$ as the decision "choose $Q$." Then $P(E)$ is an error under $P$, while $Q(E^c)$ is an error under $Q$. Equation [Eq. (5)](#eq:bh) says that the two errors cannot both be tiny unless the two laws have accumulated substantial KL divergence.

The complete proof is given in Appendix [the appendix](#proof-of-the-bretagnolle-huber-inequality). For now, the important shape is

<div class="display-equation">
$$
\text{testing error}
\gtrsim
e^{-\text{information}}.
$$
</div>


![The exact error of the optimal likelihood-ratio test and the Bretagnolle–Huber lower bound for $\Ber(0.4)$ versus $\Ber(0.6)$. The bound is not exact, but it captures the unavoidable exponential scale.](/images/notes/assets/change-of-measure/testing_error_bound.webp)

*The exact error of the optimal likelihood-ratio test and the Bretagnolle–Huber lower bound for $\Ber(0.4)$ versus $\Ber(0.6)$. The bound is not exact, but it captures the unavoidable exponential scale.*

The bound is deliberately simple and therefore not always numerically tight. Its value is portability: the same inequality can be attached to an adaptive history, a stopping rule, or a recommendation event without redesigning the test from scratch.

## The Law of an Adaptive Bandit History

Return to a $K$-armed bandit. Let $\nu=(\nu_1,\ldots,\nu_K)$ be one environment and $\nu'=(\nu'_1,\ldots,\nu'_K)$ be another. The policy is the same in both environments.

At round $t$, the policy chooses an arm $A_t$ from the past history $H_{t-1}=(A_1,X_1,\ldots,A_{t-1},X_{t-1}).$ Write $\pi_t(a\mid h_{t-1})$ for the probability that the policy chooses arm $a$ after history $h_{t-1}$. If arm $a$ is chosen, its reward density is $p_a$ under $\nu$ and $p'_a$ under $\nu'$.

For a realized history $h_T=(a_1,x_1,\ldots,a_T,x_T),$ the density under $\nu$ factors as

<div class="numbered-equation" id="eq:history-factorization-nu">
$$
P_\nu(h_T)
=
\prod_{t=1}^{T}
\pi_t(a_t\mid h_{t-1})p_{a_t}(x_t).
$$
<span class="equation-number" aria-label="Equation 6">(6)</span>
</div>

 Under $\nu'$,

<div class="numbered-equation" id="eq:history-factorization-nup">
$$
P_{\nu'}(h_T)
=
\prod_{t=1}^{T}
\pi_t(a_t\mid h_{t-1})p'_{a_t}(x_t).
$$
<span class="equation-number" aria-label="Equation 7">(7)</span>
</div>


### The policy terms cancel

Divide [Eq. (6)](#eq:history-factorization-nu) by [Eq. (7)](#eq:history-factorization-nup):

<div class="display-equation">
$$
\begin{align*}
\frac{P_\nu(h_T)}{P_{\nu'}(h_T)}
&=
\frac{\prod_{t=1}^{T}\pi_t(a_t\mid h_{t-1})p_{a_t}(x_t)}
{\prod_{t=1}^{T}\pi_t(a_t\mid h_{t-1})p'_{a_t}(x_t)}\\
&=
\prod_{t=1}^{T}\frac{p_{a_t}(x_t)}{p'_{a_t}(x_t)}.
\end{align*}
$$
</div>

 Therefore the history log-likelihood ratio is

<div class="numbered-equation" id="eq:bandit-llr">
$$
L_T
=
\sum_{t=1}^{T}
\log\frac{p_{A_t}(X_t)}{p'_{A_t}(X_t)}.
$$
<span class="equation-number" aria-label="Equation 8">(8)</span>
</div>


This cancellation is the central algebraic fact. The policy may be adaptive, randomized, and highly nonlinear. None of those policy probabilities appears in the final likelihood ratio, because the same decision rule is used in both worlds.

> **Diagram.** The learner’s adaptivity changes which reward factors appear, but not the likelihood-ratio algebra.

### Expected evidence equals pulls times armwise KL

Take expectations under $\nu$. From [Eq. (8)](#eq:bandit-llr),

<div class="display-equation">
$$
\begin{align*}
\E_\nu[L_T]
&=
\sum_{t=1}^{T}
\E_\nu\left[
\log\frac{p_{A_t}(X_t)}{p'_{A_t}(X_t)}
\right].
\end{align*}
$$
</div>

 Condition on the selected arm. If $A_t=a$, then $X_t\sim\nu_a$ under $\nu$, and therefore

<div class="display-equation">
$$
\begin{align*}
\E_\nu\left[
\left.
\log\frac{p_{A_t}(X_t)}{p'_{A_t}(X_t)}
\right|A_t=a, H_{t-1}
\right]
&=
\int p_a(x)\log\frac{p_a(x)}{p'_a(x)}\,\dd x\\
&=
\KL(\nu_a\Vert\nu'_a).
\end{align*}
$$
</div>

 Hence

<div class="display-equation">
$$
\begin{align*}
\E_\nu[L_T]
&=
\sum_{t=1}^{T}
\E_\nu\left[\KL(\nu_{A_t}\Vert\nu'_{A_t})\right]\\
&=
\sum_{t=1}^{T}
\sum_{a=1}^{K}
\Pbb_\nu(A_t=a)\KL(\nu_a\Vert\nu'_a)\\
&=
\sum_{a=1}^{K}
\E_\nu\left[
\sum_{t=1}^{T}\one\{A_t=a\}
\right]
\KL(\nu_a\Vert\nu'_a).
\end{align*}
$$
</div>

 Define the number of pulls $N_a(T)=\sum_{t=1}^{T}\one\{A_t=a\}.$ Then

<div class="numbered-equation" id="eq:divergence-decomposition">
$$
\boxed{
\KL(P_\nu^T\Vert P_{\nu'}^T)
=
\E_\nu[L_T]
=
\sum_{a=1}^{K}
\E_\nu[N_a(T)]\KL(\nu_a\Vert\nu'_a).
}
$$
<span class="equation-number" aria-label="Equation 9">(9)</span>
</div>


This is often called the divergence decomposition. It turns a complicated adaptive experiment into a simple information ledger:


*expected pulls of arm $a$*$\times$*information supplied by one pull of arm $a$*.


> **Key idea.**
>
> Adaptivity changes the random pull counts. It does not create information from nowhere. Every unit of KL must still be paid for by observations from arms on which the two environments differ.

### A numerical audit of the identity

The supplied code runs UCB under $\nu=(\Ber(0.62),\Ber(0.50))$ and compares it with the alternative $\nu'=(\Ber(0.62),\Ber(0.68)).$ Only arm 2 changes. For each simulated history we compute the realized log-likelihood ratio and compare its average with $\E_\nu[N_2(T)]\kl(0.50,0.68).$

![The empirical mean log-likelihood ratio matches the pull-count information ledger, even though the sampling rule is adaptive.](/images/notes/assets/change-of-measure/adaptive_information_identity.webp)

*The empirical mean log-likelihood ratio matches the pull-count information ledger, even though the sampling rule is adaptive.*
| Horizon | $\E[N_2(T)]$ | empirical $\E[L_T]$ | $\E[N_2(T)]\kl(0.50,0.68)$ |
|--------:|---------------:|----------------------:|-----------------------------:|
|     100 |          37.05 |                 2.581 |                        2.572 |
|     400 |         114.50 |                 7.997 |                        7.946 |
|     800 |         187.81 |                13.040 |                       13.034 |
|    1200 |         244.16 |                16.966 |                       16.945 |

<p class="table-caption">Adaptive information accounting under UCB, based on 12,000 replications.</p>

The code performing the essential update is short:

    means = sums / counts
    bonus = np.sqrt(2.0 * np.log(t + 1.0) / counts)
    actions = np.argmax(means + bonus, axis=1)

    arm_means = true_means[actions]
    alt_means = alternative_means[actions]
    rewards = rng.binomial(1, arm_means).astype(float)

    counts[indices, actions] += 1
    sums[indices, actions] += rewards

    total_llr += (
        rewards * np.log(arm_means / alt_means)
        + (1.0 - rewards)
          * np.log((1.0 - arm_means) / (1.0 - alt_means))
    )

## Stopping When the Evidence Is Enough

Many bandit algorithms do not stop at a fixed horizon. They stop when the data appear decisive. Let $\tau$ be such a stopping time, and let $\calF_\tau$ contain everything known when the algorithm stops.

The stopped log-likelihood ratio is

<div class="display-equation">
$$
L_\tau
=
\sum_{t=1}^{\tau}
\log\frac{p_{A_t}(X_t)}{p'_{A_t}(X_t)}.
$$
</div>

 Under the usual integrability conditions, the same information ledger holds:

<div class="numbered-equation" id="eq:stopped-ledger">
$$
\E_\nu[L_\tau]
=
\sum_{a=1}^{K}
\E_\nu[N_a(\tau)]\KL(\nu_a\Vert\nu'_a).
$$
<span class="equation-number" aria-label="Equation 10">(10)</span>
</div>


Now take any event $E\in\calF_\tau$. It could be the event that the algorithm recommends arm 1, eliminates arm 4, or stops before a given time. Applying binary data processing to the stopped history gives

<div class="display-equation">
$$
\E_\nu[L_\tau]
\geq
\kl\bigl(\Pbb_\nu(E),\Pbb_{\nu'}(E)\bigr).
$$
</div>

 Combining with [Eq. (10)](#eq:stopped-ledger) yields the fundamental bandit change-of-measure inequality.

> **Bandit transportation inequality.**
>
> For two bandit environments $\nu$ and $\nu'$, an almost surely finite stopping time $\tau$, and any event $E\in\calF_\tau$,

<div class="numbered-equation" id="eq:transportation">
$$
\boxed{
\sum_{a=1}^{K}
\E_\nu[N_a(\tau)]\KL(\nu_a\Vert\nu'_a)
\geq
\kl\bigl(\Pbb_\nu(E),\Pbb_{\nu'}(E)\bigr).
}
$$
<span class="equation-number" aria-label="Equation 11">(11)</span>
</div>


The word "transportation" is helpful. The left side measures how much information the algorithm transports from observations into its history. The right side measures how far apart the algorithm's behavior must be in the two worlds.

### The proof in three transparent steps

**Step 1: the stopped change-of-measure identity.** For $E\in\calF_\tau$,

<div class="display-equation">
$$
\Pbb_{\nu'}(E)
=
\E_\nu\left[\one_E e^{-L_\tau}\right].
$$
</div>


**Step 2: compress the stopped history to $E$.** The laws of the whole stopped history satisfy

<div class="display-equation">
$$
\KL(P_\nu^{H_\tau}\Vert P_{\nu'}^{H_\tau})
\geq
\kl\bigl(\Pbb_\nu(E),\Pbb_{\nu'}(E)\bigr).
$$
</div>

 Since the log density ratio of the stopped histories is $L_\tau$,

<div class="display-equation">
$$
\KL(P_\nu^{H_\tau}\Vert P_{\nu'}^{H_\tau})
=
\E_\nu[L_\tau].
$$
</div>


**Step 3: expand the expected evidence arm by arm.** Write the observations from arm $a$ in their order of appearance as $Y_{a,1},Y_{a,2},\ldots.$ Then

<div class="display-equation">
$$
L_\tau
=
\sum_{a=1}^{K}
\sum_{s=1}^{N_a(\tau)}
\log\frac{p_a(Y_{a,s})}{p'_a(Y_{a,s})}.
$$
</div>

 The expected increment from arm $a$ is

<div class="display-equation">
$$
\E_\nu\left[
\log\frac{p_a(Y_{a,s})}{p'_a(Y_{a,s})}
\right]
=
\KL(\nu_a\Vert\nu'_a).
$$
</div>

 Wald's identity therefore gives

<div class="display-equation">
$$
\E_\nu[L_\tau]
=
\sum_{a=1}^{K}
\E_\nu[N_a(\tau)]\KL(\nu_a\Vert\nu'_a).
$$
</div>

 Putting the three steps together proves [Eq. (11)](#eq:transportation).

> **Proof pattern.**
>
> A change-of-measure lower bound usually has the same skeleton:

<div class="display-equation">
$$
\text{sample allocation}
\longrightarrow
\text{KL information}
\longrightarrow
\text{separation of a decision event}.
$$
</div>

 The creativity lies mainly in choosing the alternative environment and the event.

## The Simplest Sample-Complexity Lower Bound

Return once more to the two-coin problem. Suppose a decision rule observes $n$ tosses and must identify whether the coin is $P=\Ber(p)$ or $Q=\Ber(q)$. Let $E=\{\text{the rule declares }P\}.$ Assume the rule has error at most $\delta$ in both worlds:

<div class="display-equation">
$$
P^n(E)\geq 1-\delta,
\qquad
Q^n(E)\leq\delta.
$$
</div>

 By [Eq. (4)](#eq:binary-data-processing),

<div class="display-equation">
$$
\begin{align*}
\KL(P^n\Vert Q^n)
&\geq
\kl\bigl(P^n(E),Q^n(E)\bigr)\\
&\geq
\kl(1-\delta,\delta).
\end{align*}
$$
</div>

 Independence gives $\KL(P^n\Vert Q^n)=n\KL(P\Vert Q)=n\kl(p,q).$ Therefore

<div class="numbered-equation" id="eq:coin-lower-bound">
$$
\boxed{
 n
 \geq
 \frac{\kl(1-\delta,\delta)}{\kl(p,q)}.
}
$$
<span class="equation-number" aria-label="Equation 12">(12)</span>
</div>


Every part has an interpretation:


<div class="display-equation">
$$
\underbrace{n}_{\text{number of observations}}
\times
\underbrace{\kl(p,q)}_{\text{evidence per observation}}
\geq
\underbrace{\kl(1-\delta,\delta)}_{\text{evidence demanded by reliability}}.
$$
</div>


For $p=0.55$, $q=0.45$, and $\delta=0.05$,

<div class="display-equation">
$$
\kl(0.55,0.45)\approx0.02007,
\qquad
\kl(0.95,0.05)\approx2.650.
$$
</div>

 Thus $n\geq\frac{2.650}{0.02007}\approx132.1.$ A rule that promises five-percent error in both directions needs at least about 133 tosses according to this information bound.

As $\delta\downarrow0$,

<div class="display-equation">
$$
\kl(1-\delta,\delta)
=(1-2\delta)\log\frac{1-\delta}{\delta}
\sim
\log\frac{1}{\delta}.
$$
</div>

 So the familiar logarithm in fixed-confidence sample complexity is not an artifact of one algorithm. It is the amount of evidence required to drive an error probability down to $\delta$.

## Best-Arm Identification: Make the Answer Change

Consider a bandit environment $\nu$ with a unique best arm $a^*(\nu)=\argmax_a \mu_a.$ A fixed-confidence algorithm stops at $\tau$ and recommends $\widehat a$. It is $\delta$-correct if $\Pbb_\nu(\widehat a=a^*(\nu))\geq1-\delta$ for every environment in the model class.

Fix the true environment $\nu$. Now choose an alternative $\lambda$ whose best arm is different: $a^*(\lambda)\neq a^*(\nu).$ Define $E=\{\widehat a=a^*(\nu)\}.$ Correctness in the true world gives $\Pbb_\nu(E)\geq1-\delta.$ Correctness in the alternative world gives $\Pbb_\lambda(E)\leq\delta,$ because the arm recommended on $E$ is wrong under $\lambda$.

Insert this event into the transportation inequality:

<div class="numbered-equation" id="eq:bai-one-alternative">
$$
\boxed{
\sum_{a=1}^{K}
\E_\nu[N_a(\tau)]\KL(\nu_a\Vert\lambda_a)
\geq
\kl(1-\delta,\delta).
}
$$
<span class="equation-number" aria-label="Equation 13">(13)</span>
</div>


This inequality holds for *every* alternative that changes the best arm.

### A first, deliberately simple alternative

Suppose there are two Bernoulli arms with $\mu_1>\mu_2.$ Keep arm 1 unchanged and raise arm 2 to a mean $\lambda_2>\mu_1$. Then only arm 2 differs, and [Eq. (13)](#eq:bai-one-alternative) becomes

<div class="display-equation">
$$
\E_\nu[N_2(\tau)]\kl(\mu_2,\lambda_2)
\geq
\kl(1-\delta,\delta).
$$
</div>

 Hence

<div class="numbered-equation" id="eq:single-arm-bai-bound">
$$
\E_\nu[N_2(\tau)]
\geq
\frac{\kl(1-\delta,\delta)}{\kl(\mu_2,\lambda_2)}.
$$
<span class="equation-number" aria-label="Equation 14">(14)</span>
</div>


Let $\lambda_2$ approach $\mu_1$ from above. The alternative becomes as close as possible while still changing the best arm, yielding

<div class="display-equation">
$$
\E_\nu[N_2(\tau)]
\gtrsim
\frac{\kl(1-\delta,\delta)}{\kl(\mu_2,\mu_1)}.
$$
</div>


This already explains why a suboptimal arm cannot simply be ignored. If the learner rarely samples arm 2, it cannot rule out the nearby world in which arm 2 is actually best.

> **Think.**
>
> Why not raise arm 2 to $0.99$? Because that alternative is easy to distinguish. Its KL divergence is large, so the resulting lower bound is weak. Lower bounds become strong by finding the *closest answer-changing alternative*.

### The allocation game

Let

<div class="display-equation">
$$
\Alt(\nu)
=
\{\lambda:a^*(\lambda)\neq a^*(\nu)\}
$$
</div>

 be the set of answer-changing alternatives. Define the expected allocation proportions

<div class="display-equation">
$$
w_a
=
\frac{\E_\nu[N_a(\tau)]}{\E_\nu[\tau]},
\qquad
\sum_{a=1}^{K}w_a=1.
$$
</div>

 Then [Eq. (13)](#eq:bai-one-alternative) can be written

<div class="display-equation">
$$
\E_\nu[\tau]
\sum_{a=1}^{K}w_a\KL(\nu_a\Vert\lambda_a)
\geq
\kl(1-\delta,\delta).
$$
</div>

 Because this must hold for every $\lambda\in\Alt(\nu)$,

<div class="display-equation">
$$
\E_\nu[\tau]
\inf_{\lambda\in\Alt(\nu)}
\sum_{a=1}^{K}w_a\KL(\nu_a\Vert\lambda_a)
\geq
\kl(1-\delta,\delta).
$$
</div>

 Thus

<div class="numbered-equation" id="eq:allocation-specific-lb">
$$
\E_\nu[\tau]
\geq
\frac{\kl(1-\delta,\delta)}
{\displaystyle
\inf_{\lambda\in\Alt(\nu)}
\sum_{a=1}^{K}w_a\KL(\nu_a\Vert\lambda_a)}.
$$
<span class="equation-number" aria-label="Equation 15">(15)</span>
</div>


The algorithm chooses $w$. The lower-bound argument then chooses the hardest alternative against that allocation. This leads to the characteristic information value

<div class="numbered-equation" id="eq:t-star">
$$
\boxed{
\frac{1}{T^*(\nu)}
=
\sup_{w\in\Delta_K}
\inf_{\lambda\in\Alt(\nu)}
\sum_{a=1}^{K}w_a\KL(\nu_a\Vert\lambda_a),
}
$$
<span class="equation-number" aria-label="Equation 16">(16)</span>
</div>

 where

<div class="display-equation">
$$
\Delta_K
=
\left\{w\in[0,1]^K:\sum_{a=1}^{K}w_a=1\right\}.
$$
</div>

 Consequently,

<div class="numbered-equation" id="eq:tstar-lb">
$$
\E_\nu[\tau]
\geq
T^*(\nu)\,\kl(1-\delta,\delta).
$$
<span class="equation-number" aria-label="Equation 17">(17)</span>
</div>


> **Research connection.**
>
> Equation [Eq. (16)](#eq:t-star) is not only a lower bound. It is an algorithm-design target. Track-and-Stop and later best-arm identification methods attempt to learn and track the allocation that maximizes the worst-case information rate. The same information-allocation viewpoint also underlies modern batched BAI and regret-aware BAI, including research directions developed by Tianyuan Jin and collaborators.

## Two Gaussian Arms: Solve the Lower-Bound Game Completely

Consider two Gaussian arms with known common variance $\sigma^2$:

<div class="display-equation">
$$
\nu_1=\Normal(\mu_1,\sigma^2),
\qquad
\nu_2=\Normal(\mu_2,\sigma^2),
\qquad
\mu_1>\mu_2.
$$
</div>

 Let $\Delta=\mu_1-\mu_2>0.$ Assign a fraction $w$ of samples to arm 1 and $1-w$ to arm 2.

For an alternative mean vector $(\lambda_1,\lambda_2)$, Gaussian KL gives

<div class="display-equation">
$$
\KL\bigl(\Normal(\mu_a,\sigma^2)\Vert\Normal(\lambda_a,\sigma^2)\bigr)
=
\frac{(\mu_a-\lambda_a)^2}{2\sigma^2}.
$$
</div>

 The information rate is therefore

<div class="numbered-equation" id="eq:gaussian-rate">
$$
I_w(\lambda_1,\lambda_2)
=
\frac{w(\mu_1-\lambda_1)^2
+(1-w)(\mu_2-\lambda_2)^2}{2\sigma^2}.
$$
<span class="equation-number" aria-label="Equation 18">(18)</span>
</div>


The alternative must make arm 2 at least as good as arm 1: $\lambda_1\leq\lambda_2.$ The closest point in this half-space lies on the boundary $\lambda_1=\lambda_2=m.$ Thus we minimize

<div class="display-equation">
$$
I_w(m)
=
\frac{w(\mu_1-m)^2+(1-w)(\mu_2-m)^2}{2\sigma^2}.
$$
</div>


### Step 1: find the hardest common mean

Differentiate with respect to $m$:

<div class="display-equation">
$$
\begin{align*}
\frac{\dd}{\dd m}
\left[
 w(\mu_1-m)^2+(1-w)(\mu_2-m)^2
\right]
&=
2w(m-\mu_1)+2(1-w)(m-\mu_2).
\end{align*}
$$
</div>

 Set this equal to zero:

<div class="display-equation">
$$
\begin{align*}
0
&=
w(m-\mu_1)+(1-w)(m-\mu_2)\\
&=
m-w\mu_1-(1-w)\mu_2.
\end{align*}
$$
</div>

 Therefore

<div class="numbered-equation" id="eq:hardest-m">
$$
\boxed{
 m_w=w\mu_1+(1-w)\mu_2.
}
$$
<span class="equation-number" aria-label="Equation 19">(19)</span>
</div>


The hard alternative moves toward the arm receiving fewer samples. If arm 1 receives most of the budget, then its mean is well measured and the alternative moves close to $\mu_1$, leaving the poorly measured arm 2 to do most of the work.

![For each allocation w, the hardest boundary alternative is the minimum of the corresponding quadratic information curve.](/images/notes/assets/change-of-measure/gaussian_alternative_geometry.webp)

*For each allocation w, the hardest boundary alternative is the minimum of the corresponding quadratic information curve.*

### Step 2: substitute the minimizer

From [Eq. (19)](#eq:hardest-m),

<div class="display-equation">
$$
\begin{align*}
\mu_1-m_w
&=
\mu_1-w\mu_1-(1-w)\mu_2\\
&=
(1-w)(\mu_1-\mu_2)\\
&=
(1-w)\Delta,
\end{align*}
$$
</div>

 and

<div class="display-equation">
$$
\begin{align*}
\mu_2-m_w
&=
\mu_2-w\mu_1-(1-w)\mu_2\\
&=
-w(\mu_1-\mu_2)\\
&=
-w\Delta.
\end{align*}
$$
</div>

 Therefore

<div class="display-equation">
$$
\begin{align*}
\inf_{\lambda_1\leq\lambda_2}I_w(\lambda_1,\lambda_2)
&=
\frac{w(1-w)^2\Delta^2+(1-w)w^2\Delta^2}{2\sigma^2}\\
&=
\frac{w(1-w)\bigl((1-w)+w\bigr)\Delta^2}{2\sigma^2}\\
&=
\frac{w(1-w)\Delta^2}{2\sigma^2}.
\end{align*}
$$
</div>


### Step 3: choose the best allocation

Since

<div class="display-equation">
$$
w(1-w)
=
\frac14-\left(w-\frac12\right)^2,
$$
</div>

 the maximum occurs at $w^*=\frac12.$ The optimal information rate is

<div class="numbered-equation" id="eq:gaussian-tstar-rate">
$$
\frac{1}{T^*(\nu)}
=
\frac{\Delta^2}{8\sigma^2}.
$$
<span class="equation-number" aria-label="Equation 20">(20)</span>
</div>

 Hence

<div class="numbered-equation" id="eq:gaussian-bai-lb">
$$
\boxed{
\E_\nu[\tau]
\geq
\frac{8\sigma^2}{\Delta^2}
\kl(1-\delta,\delta).
}
$$
<span class="equation-number" aria-label="Equation 21">(21)</span>
</div>


![The worst-case information rate is maximized by equal allocation in the symmetric two-Gaussian problem.](/images/notes/assets/change-of-measure/gaussian_information_rate.webp)

*The worst-case information rate is maximized by equal allocation in the symmetric two-Gaussian problem.*

This conclusion is easy to say after the calculation: both sample means enter the comparison, so making one estimate precise while leaving the other noisy is wasteful. The lower-bound game turns that intuition into an exact constant.

### The same allocation appears in the actual error

With a fixed total budget $n$, let

<div class="display-equation">
$$
n_1=wn,
\qquad
n_2=(1-w)n.
$$
</div>

 The sample-mean difference satisfies

<div class="display-equation">
$$
\widehat\mu_1-\widehat\mu_2
\sim
\Normal\left(
\Delta,
\sigma^2\left(\frac1{n_1}+\frac1{n_2}\right)
\right).
$$
</div>

 If the learner recommends the larger sample mean, then

<div class="display-equation">
$$
\begin{align*}
\Pbb(\text{error})
&=
\Pbb(\widehat\mu_1-\widehat\mu_2\leq0)\\
&=
\Phi\left(
-\frac{\Delta}
{\sigma\sqrt{1/n_1+1/n_2}}
\right).
\end{align*}
$$
</div>

 The denominator is minimized at $n_1=n_2$. Thus the finite-sample testing calculation and the information lower bound select the same allocation.

![For n = 400, (μ1, μ2) = (0.2, 0), and σ = 1, both exact calculation and Monte Carlo show that equal allocation minimizes the recommendation error.](/images/notes/assets/change-of-measure/gaussian_allocation_error.webp)

*For n = 400, (μ1, μ2) = (0.2, 0), and σ = 1, both exact calculation and Monte Carlo show that equal allocation minimizes the recommendation error.*
| Fraction $w$ on arm 1 | exact error probability | information rate |
|------------------------:|------------------------:|-----------------:|
|                    0.10 |                  0.1151 |          0.00180 |
|                    0.30 |                  0.0334 |          0.00420 |
|                    0.50 |                  0.0228 |          0.00500 |
|                    0.70 |                  0.0334 |          0.00420 |
|                    0.90 |                  0.1151 |          0.00180 |

<p class="table-caption">Selected Gaussian allocation results.</p>

    n1 = max(1, int(round(total_budget * weight)))
    n2 = total_budget - n1

    mean1 = mu1 + sigma / np.sqrt(n1) * rng.standard_normal(replications)
    mean2 = mu2 + sigma / np.sqrt(n2) * rng.standard_normal(replications)
    empirical_error = np.mean(mean1 <= mean2)

    standard_error = sigma * np.sqrt(1.0 / n1 + 1.0 / n2)
    exact_error = norm.cdf(-(mu1 - mu2) / standard_error)

## A Glimpse of the Lai--Robbins Regret Lower Bound

The same argument also explains why logarithmic exploration is unavoidable in regret minimization.

Suppose arm $a$ is suboptimal under $\nu$: $\mu_a<\mu^*.$ Construct an alternative $\nu'$ that changes only arm $a$, making it slightly better than $\mu^*$. If the algorithm almost never samples arm $a$ under $\nu$, then the histories under $\nu$ and $\nu'$ remain too similar. But a good policy must behave differently:

- under $\nu$, it should pull arm $a$ rarely;

- under $\nu'$, it should pull arm $a$ almost all the time.

Choose an event such as $E_T=\left\{N_a(T)<\frac{T}{2}\right\}.$ For a sufficiently efficient policy,

<div class="display-equation">
$$
\Pbb_\nu(E_T)\to1,
\qquad
\Pbb_{\nu'}(E_T)\to0.
$$
</div>

 The transportation inequality gives

<div class="numbered-equation" id="eq:regret-preview">
$$
\E_\nu[N_a(T)]\KL(\nu_a\Vert\nu'_a)
\geq
\kl\bigl(\Pbb_\nu(E_T),\Pbb_{\nu'}(E_T)\bigr).
$$
<span class="equation-number" aria-label="Equation 22">(22)</span>
</div>

 Under the consistency conditions used by Lai and Robbins, the right side grows like $\log T$. Moving $\nu'_a$ down toward the boundary where its mean equals $\mu^*$ yields

<div class="numbered-equation" id="eq:lai-robbins-pull-preview">
$$
\liminf_{T\to\infty}
\frac{\E_\nu[N_a(T)]}{\log T}
\geq
\frac{1}{\KL(\nu_a\Vert\nu^*)}.
$$
<span class="equation-number" aria-label="Equation 23">(23)</span>
</div>

 Multiplying by the regret gap $\Delta_a=\mu^*-\mu_a$ and summing gives the classical shape

<div class="numbered-equation" id="eq:lai-robbins-preview">
$$
\liminf_{T\to\infty}
\frac{R_T(\nu)}{\log T}
\geq
\sum_{a:\Delta_a>0}
\frac{\Delta_a}{\KL(\nu_a\Vert\nu^*)}.
$$
<span class="equation-number" aria-label="Equation 24">(24)</span>
</div>


This section is only the proof skeleton. The next chapter will state the consistency assumptions carefully, construct the event with the correct polynomial probabilities, and derive the full Lai--Robbins lower bound without hiding the asymptotic steps.

> **Key idea.**
>
> The logarithmic exploration rate is the price of ruling out a nearby world in which a currently suboptimal arm is actually optimal.

## What the Experiments Are Checking

The numerical work in this chapter serves as a proof audit rather than as evidence for the theorem.

### Evidence paths

For $X_t\sim\Ber(p)$, the cumulative evidence against $q$ is

<div class="display-equation">
$$
L_t
=
\sum_{s=1}^{t}
\left[
X_s\log\frac{p}{q}
+(1-X_s)\log\frac{1-p}{1-q}
\right].
$$
</div>

 The experiment checks that paths are noisy while their mean slope is $\kl(p,q)$.

### Testing error

For each $n$, the script computes the exact sum of type-I and type-II errors of the equal-prior likelihood-ratio test between $\Ber(0.4)^n$ and $\Ber(0.6)^n$. It compares this with $\frac12e^{-n\kl(0.4,0.6)}.$
| $n$ | exact sum of errors |           lower bound |
|------:|--------------------:|----------------------:|
|    10 |              0.5331 |                0.2222 |
|    20 |              0.3722 |                0.0988 |
|    40 |              0.2041 |                0.0195 |
|    80 |              0.0716 |              0.000761 |
|   160 |              0.0107 | $1.16\times10^{-6}$ |

<p class="table-caption">Optimal testing error versus the Bretagnolle–Huber lower bound.</p>

The lower bound becomes loose in this example because it sacrifices constants for generality. It is still correct, and it preserves the central message that the error cannot decay independently of accumulated information.

### Adaptive information accounting

The UCB experiment checks

<div class="display-equation">
$$
\E_\nu[L_T]
=
\sum_a\E_\nu[N_a(T)]\KL(\nu_a\Vert\nu'_a)
$$
</div>

 under an adaptive policy. The near overlap of the two curves is a direct numerical check that the likelihood-ratio implementation, pull counts, and KL formula agree.

### Allocation and hard alternatives

The Gaussian experiment checks two linked predictions:


<div class="display-equation">
$$
\text{hardest alternative for allocation }w
\quad\Longrightarrow\quad
m_w=w\mu_1+(1-w)\mu_2,
$$
</div>

 and

<div class="display-equation">
$$
\text{best worst-case information rate}
\quad\Longrightarrow\quad
w^*=\frac12.
$$
</div>

 The exact finite-sample error confirms the same symmetry.

> **Think.**
>
> A simulation can reveal a coding error or a missing factor of two. It cannot prove a universal lower bound. The proof ranges over all algorithms and all admissible alternatives; a simulation samples only finitely many histories from one implementation.

## Common Mistakes in Change-of-Measure Proofs

### Choosing an alternative that does not change the answer

If $a^*(\lambda)=a^*(\nu)$, then the recommendation event need not have very different probabilities. The right side of [Eq. (11)](#eq:transportation) may be small, and no useful lower bound follows.

### Choosing an alternative that is too far away

A dramatic alternative has a large KL cost per observation. Since the lower bound divides by this cost, the result becomes weak. The right alternative usually sits near the decision boundary.

### Reversing the direction of KL

The expected log-likelihood ratio under $\nu$ is

<div class="display-equation">
$$
\E_\nu\left[\log\frac{\dd P_\nu}{\dd P_{\nu'}}\right]
=
\KL(P_\nu\Vert P_{\nu'}),
$$
</div>

 not the reverse divergence. The pull counts are also averaged under $\nu$. Both directions must match.

### Forgetting that counts are random

In an adaptive bandit, $N_a(T)$ depends on previous rewards. The correct identity uses $\E_\nu[N_a(T)]\KL(\nu_a\Vert\nu'_a),$ not a deterministic count inserted after the fact.

### Using an event unavailable at the stopping time

The event $E$ must belong to $\calF_\tau$. A lower-bound proof cannot use future observations that the algorithm did not possess when it stopped.

### Proving correctness only in the true environment

The argument needs the algorithm to be reliable in both $\nu$ and $\lambda$. This is why lower bounds assume uniform correctness over a model class. A procedure designed for one known instance could simply print the answer without observing anything.

### Stopping after one convenient alternative

One alternative gives one constraint. A sharp lower bound minimizes over every answer-changing alternative and then optimizes the sampling allocation. The optimization is the theorem, not decorative notation.

## What to Carry Forward

The chapter can be compressed to five equations.

For two laws,

<div class="display-equation">
$$
Q(E)=\E_P[\one_Ee^{-L}],
\qquad
L=\log\frac{\dd P}{\dd Q}.
$$
</div>


Compressing an experiment to an event cannot increase information:

<div class="display-equation">
$$
\KL(P\Vert Q)
\geq
\kl(P(E),Q(E)).
$$
</div>


For a fixed-horizon adaptive bandit,

<div class="display-equation">
$$
\KL(P_\nu^T\Vert P_{\nu'}^T)
=
\sum_a\E_\nu[N_a(T)]\KL(\nu_a\Vert\nu'_a).
$$
</div>


For a stopped experiment and $E\in\calF_\tau$,

<div class="display-equation">
$$
\sum_a\E_\nu[N_a(\tau)]\KL(\nu_a\Vert\nu'_a)
\geq
\kl(\Pbb_\nu(E),\Pbb_{\nu'}(E)).
$$
</div>


For fixed-confidence best-arm identification,

<div class="display-equation">
$$
\frac1{T^*(\nu)}
=
\sup_{w\in\Delta_K}
\inf_{\lambda\in\Alt(\nu)}
\sum_a w_a\KL(\nu_a\Vert\lambda_a).
$$
</div>


These equations express one idea in progressively richer settings:


*To behave differently in two nearby worlds, a learner must observe enough of the places where those worlds differ.*


> **Diagram.** The reusable research pattern behind many bandit lower bounds.

## Proof of the Bretagnolle--Huber Inequality

Let $p$ and $q$ be densities of $P$ and $Q$ with respect to a common measure. For any event $E$,

<div class="display-equation">
$$
\begin{align*}
P(E)+Q(E^c)
&=
\int_E p+\int_{E^c}q\\
&\geq
\int \min\{p,q\}.
\end{align*}
$$
</div>

 Define the affinity $\rho(P,Q)=\int\sqrt{pq}.$ By Cauchy--Schwarz,

<div class="display-equation">
$$
\begin{align*}
\rho(P,Q)^2
&=
\left(
\int
\sqrt{\min\{p,q\}}
\sqrt{\max\{p,q\}}
\right)^2\\
&\leq
\left(\int\min\{p,q\}\right)
\left(\int\max\{p,q\}\right).
\end{align*}
$$
</div>

 Since

<div class="display-equation">
$$
\int\max\{p,q\}
=2-\int\min\{p,q\}
\leq2,
$$
</div>

 we obtain

<div class="numbered-equation" id="eq:min-affinity">
$$
\int\min\{p,q\}
\geq
\frac{\rho(P,Q)^2}{2}.
$$
<span class="equation-number" aria-label="Equation 25">(25)</span>
</div>


Next relate affinity to KL. Under $P$,

<div class="display-equation">
$$
\begin{align*}
\rho(P,Q)
&=
\int p\sqrt{\frac{q}{p}}\\
&=
\E_P\left[\sqrt{\frac{q}{p}}\right].
\end{align*}
$$
</div>

 Because $-\log$ is convex, Jensen's inequality gives

<div class="display-equation">
$$
\begin{align*}
-\log\rho(P,Q)
&\leq
\E_P\left[
-\log\sqrt{\frac{q}{p}}
\right]\\
&=
\frac12\E_P\left[\log\frac{p}{q}\right]\\
&=
\frac12\KL(P\Vert Q).
\end{align*}
$$
</div>

 Therefore

<div class="display-equation">
$$
\rho(P,Q)
\geq
\exp\left\{-\frac12\KL(P\Vert Q)\right\}.
$$
</div>

 Squaring and substituting into [Eq. (25)](#eq:min-affinity),

<div class="display-equation">
$$
\int\min\{p,q\}
\geq
\frac12e^{-\KL(P\Vert Q)}.
$$
</div>

 Finally,

<div class="display-equation">
$$
P(E)+Q(E^c)
\geq
\int\min\{p,q\}
\geq
\frac12e^{-\KL(P\Vert Q)}.
$$
</div>


## A Direct Jensen Proof of Event Compression

The change-of-measure identity gives $Q(E)=\E_P[\one_Ee^{-L}].$ Condition on $E$:

<div class="display-equation">
$$
Q(E)
=P(E)\E_P[e^{-L}\mid E].
$$
</div>

 Jensen's inequality for the convex exponential function yields

<div class="display-equation">
$$
\E_P[e^{-L}\mid E]
\geq
e^{-\E_P[L\mid E]}.
$$
</div>

 Hence

<div class="display-equation">
$$
Q(E)
\geq
P(E)e^{-\E_P[L\mid E]},
$$
</div>

 which rearranges to

<div class="display-equation">
$$
\E_P[L\mid E]
\geq
\log\frac{P(E)}{Q(E)}.
$$
</div>

 The same argument on $E^c$ gives

<div class="display-equation">
$$
\E_P[L\mid E^c]
\geq
\log\frac{P(E^c)}{Q(E^c)}.
$$
</div>

 Therefore

<div class="display-equation">
$$
\begin{align*}
\E_P[L]
&=
P(E)\E_P[L\mid E]
+P(E^c)\E_P[L\mid E^c]\\
&\geq
P(E)\log\frac{P(E)}{Q(E)}
+P(E^c)\log\frac{P(E^c)}{Q(E^c)}\\
&=
\kl(P(E),Q(E)).
\end{align*}
$$
</div>

 Since $\E_P[L]=\KL(P\Vert Q)$, this proves [Eq. (4)](#eq:binary-data-processing).

## Formula Sheet


L0.34L0.57 Object & Formula\
Log-likelihood ratio & $L=\log(\dd P/\dd Q)$\
Change of measure & $Q(E)=\E_P[\one_Ee^{-L}]$\
Binary data processing & $\KL(P\Vert Q)\geq\kl(P(E),Q(E))$\
Bretagnolle--Huber & $P(E)+Q(E^c)\geq\tfrac12e^{-\KL(P\Vert Q)}$\
Bandit history LLR & $L_T=\sum_{t=1}^T\log[p_{A_t}(X_t)/p'_{A_t}(X_t)]$\
Divergence decomposition & $\KL(P_\nu^T\Vert P_{\nu'}^T)=\sum_a\E_\nu[N_a(T)]\KL(\nu_a\Vert\nu'_a)$\
Stopped transportation & $\sum_a\E_\nu[N_a(\tau)]\KL(\nu_a\Vert\nu'_a)\geq\kl(\Pbb_\nu(E),\Pbb_{\nu'}(E))$\
Fixed-confidence demand & $\kl(1-\delta,\delta)\sim\log(1/\delta)$\
BAI information value & $T^*(\nu)^{-1}=\sup_w\inf_{\lambda\in\Alt(\nu)}\sum_aw_a\KL(\nu_a\Vert\lambda_a)$\
Two-Gaussian rate & $T^*(\nu)^{-1}=\Delta^2/(8\sigma^2)$\


## Notation Table


L0.24L0.67 Symbol & Meaning\
$P,Q$ & two probability laws being compared\
$L$ & log-likelihood ratio $\log(\dd P/\dd Q)$\
$E$ & a measurable decision event\
$\kl(x,y)$ & Bernoulli or binary relative entropy\
$\nu,\nu'$ & two bandit environments\
$\pi_t$ & policy kernel at round $t$\
$H_t$ & observed history through round $t$\
$A_t,X_t$ & selected arm and observed reward at round $t$\
$N_a(T)$ & number of pulls of arm $a$ by time $T$\
$\tau$ & stopping time\
$\calF_\tau$ & information available when the algorithm stops\
$\Alt(\nu)$ & environments whose correct answer differs from that under $\nu$\
$w$ & vector of expected sampling proportions\
$T^*(\nu)$ & characteristic information time for best-arm identification\
$\Delta$ & gap between the two arm means\


## Implementation Notes

The supplied Python script regenerates all figures and the CSV table. Several details are worth checking when adapting it:

1.  Keep the direction of the likelihood ratio consistent: the simulation draws from the true environment and computes $\log(p_\nu/p_{\nu'})$.

2.  When an arm is unchanged between environments, its likelihood-ratio increment is exactly zero.

3.  Initialize every arm before evaluating a UCB index.

4.  In Gaussian allocation experiments, drawing the sample mean directly is exactly equivalent to simulating every individual observation and is much faster.

5.  Compare Monte Carlo output with an analytic identity whenever one is available. Here the exact Gaussian error and the divergence decomposition provide two independent audits.

## Complete Reproducible Code

``` {.python style="blogcode" language="Python" basicstyle="\\ttfamily\\scriptsize"}
"""Reproducible experiments for
Change of Measure Arguments in Bandit Lower Bounds.

The script creates seven figures and one CSV file in the same directory.
All random experiments use a fixed NumPy seed.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom, norm

OUTPUT_DIR = Path(__file__).resolve().parent
SEED = 20260620


def bernoulli_kl(p: np.ndarray | float, q: np.ndarray | float) -> np.ndarray | float:
    """KL(Bernoulli(p) || Bernoulli(q)), with stable clipping."""
    eps = np.finfo(float).eps
    p_arr = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    q_arr = np.clip(np.asarray(q, dtype=float), eps, 1.0 - eps)
    value = p_arr * np.log(p_arr / q_arr) + (1.0 - p_arr) * np.log(
        (1.0 - p_arr) / (1.0 - q_arr)
    )
    if np.ndim(value) == 0:
        return float(value)
    return value


def bernoulli_llr(reward: np.ndarray, p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """log p(X)/q(X) for Bernoulli rewards and possibly arm-dependent means."""
    return reward * np.log(p / q) + (1.0 - reward) * np.log((1.0 - p) / (1.0 - q))


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def experiment_evidence_paths(rng: np.random.Generator, rows: list[dict[str, object]]) -> None:
    p, q = 0.55, 0.45
    horizon = 400
    paths = 8
    rewards = rng.binomial(1, p, size=(paths, horizon)).astype(float)
    increments = bernoulli_llr(rewards, np.full_like(rewards, p), np.full_like(rewards, q))
    cumulative = np.cumsum(increments, axis=1)
    expected = np.arange(1, horizon + 1) * bernoulli_kl(p, q)

    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_subplot(111)
    t = np.arange(1, horizon + 1)
    for path in cumulative:
        ax.plot(t, path, linewidth=1.0, alpha=0.72)
    ax.plot(t, expected, linewidth=2.4, linestyle="--", label=r"$t\,\mathrm{kl}(p,q)$")
    ax.axhline(0.0, linewidth=0.8)
    ax.set_xlabel("number of observations")
    ax.set_ylabel("cumulative log-likelihood ratio")
    ax.set_title("Evidence is noisy, but its mean drift is KL")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure(fig, "evidence_paths")

    rows.append(
        {
            "experiment": "evidence_paths",
            "x": horizon,
            "setting": f"Bernoulli({p}) vs Bernoulli({q})",
            "metric": "mean_final_llr",
            "value": float(cumulative[:, -1].mean()),
        }
    )
    rows.append(
        {
            "experiment": "evidence_paths",
            "x": horizon,
            "setting": f"Bernoulli({p}) vs Bernoulli({q})",
            "metric": "theoretical_mean_final_llr",
            "value": float(expected[-1]),
        }
    )


def exact_lrt_error_sum(n: int, p: float, q: float) -> float:
    """Sum of the two errors for the equal-prior likelihood-ratio test."""
    successes = np.arange(n + 1)
    llr = successes * math.log(p / q) + (n - successes) * math.log((1.0 - p) / (1.0 - q))
    choose_p = llr >= 0.0
    error_under_p = binom.pmf(successes, n, p)[~choose_p].sum()
    error_under_q = binom.pmf(successes, n, q)[choose_p].sum()
    return float(error_under_p + error_under_q)


def experiment_testing_bound(rows: list[dict[str, object]]) -> None:
    p, q = 0.40, 0.60
    sample_sizes = np.array([5, 10, 20, 40, 80, 160, 240])
    exact_errors = np.array([exact_lrt_error_sum(int(n), p, q) for n in sample_sizes])
    bh_bounds = 0.5 * np.exp(-sample_sizes * bernoulli_kl(p, q))

    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_subplot(111)
    ax.semilogy(sample_sizes, exact_errors, marker="o", label="optimal likelihood-ratio test")
    ax.semilogy(sample_sizes, bh_bounds, marker="s", linestyle="--", label="Bretagnolle-Huber lower bound")
    ax.set_xlabel("sample size n")
    ax.set_ylabel("sum of the two error probabilities")
    ax.set_title("Nearby models cannot be separated faster than their information allows")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure(fig, "testing_error_bound")

    for n, exact, bound in zip(sample_sizes, exact_errors, bh_bounds):
        rows.extend(
            [
                {
                    "experiment": "testing_bound",
                    "x": int(n),
                    "setting": f"Bernoulli({p}) vs Bernoulli({q})",
                    "metric": "exact_lrt_error_sum",
                    "value": float(exact),
                },
                {
                    "experiment": "testing_bound",
                    "x": int(n),
                    "setting": f"Bernoulli({p}) vs Bernoulli({q})",
                    "metric": "bretagnolle_huber_bound",
                    "value": float(bound),
                },
            ]
        )


def simulate_ucb_information(
    rng: np.random.Generator,
    horizon: int,
    replications: int,
    true_means: np.ndarray,
    alternative_means: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run UCB under the true environment and return counts and total LLR."""
    arms = len(true_means)
    counts = np.zeros((replications, arms), dtype=np.int64)
    sums = np.zeros((replications, arms), dtype=float)
    total_llr = np.zeros(replications, dtype=float)
    indices = np.arange(replications)

    # Pull each arm once.
    for arm in range(arms):
        rewards = rng.binomial(1, true_means[arm], size=replications).astype(float)
        counts[:, arm] += 1
        sums[:, arm] += rewards
        total_llr += bernoulli_llr(
            rewards,
            np.full(replications, true_means[arm]),
            np.full(replications, alternative_means[arm]),
        )

    for t in range(arms, horizon):
        means = sums / counts
        bonus = np.sqrt(2.0 * np.log(t + 1.0) / counts)
        actions = np.argmax(means + bonus, axis=1)
        arm_means = true_means[actions]
        alt_means = alternative_means[actions]
        rewards = rng.binomial(1, arm_means).astype(float)
        counts[indices, actions] += 1
        sums[indices, actions] += rewards
        total_llr += bernoulli_llr(rewards, arm_means, alt_means)

    return counts, total_llr


def experiment_adaptive_information(rng: np.random.Generator, rows: list[dict[str, object]]) -> None:
    true_means = np.array([0.62, 0.50])
    alternative_means = np.array([0.62, 0.68])
    horizons = np.array([50, 100, 200, 400, 800, 1200])
    replications = 12000
    empirical_llr: list[float] = []
    accounted_information: list[float] = []
    avg_arm2_counts: list[float] = []
    arm_kls = np.asarray(bernoulli_kl(true_means, alternative_means))

    for horizon in horizons:
        counts, total_llr = simulate_ucb_information(
            rng,
            int(horizon),
            replications,
            true_means,
            alternative_means,
        )
        average_counts = counts.mean(axis=0)
        lhs = float(total_llr.mean())
        rhs = float(np.dot(average_counts, arm_kls))
        empirical_llr.append(lhs)
        accounted_information.append(rhs)
        avg_arm2_counts.append(float(average_counts[1]))
        rows.extend(
            [
                {
                    "experiment": "adaptive_information",
                    "x": int(horizon),
                    "setting": "UCB: true=(0.62,0.50), alternative=(0.62,0.68)",
                    "metric": "empirical_mean_log_likelihood_ratio",
                    "value": lhs,
                },
                {
                    "experiment": "adaptive_information",
                    "x": int(horizon),
                    "setting": "UCB: true=(0.62,0.50), alternative=(0.62,0.68)",
                    "metric": "sum_expected_pulls_times_arm_kl",
                    "value": rhs,
                },
                {
                    "experiment": "adaptive_information",
                    "x": int(horizon),
                    "setting": "UCB: true=(0.62,0.50), alternative=(0.62,0.68)",
                    "metric": "average_pulls_arm_2",
                    "value": float(average_counts[1]),
                },
            ]
        )

    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(horizons, empirical_llr, marker="o", label="empirical mean log-likelihood ratio")
    ax.plot(horizons, accounted_information, marker="s", linestyle="--", label=r"$\sum_a \mathbb{E}[N_a]D(\nu_a,\nu'_a)$")
    ax.set_xlabel("horizon")
    ax.set_ylabel("information")
    ax.set_title("Adaptive sampling changes the counts, not the information ledger")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure(fig, "adaptive_information_identity")

    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(horizons, avg_arm2_counts, marker="o")
    ax.set_xlabel("horizon")
    ax.set_ylabel("average pulls of the distinguishing arm")
    ax.set_title("UCB gathers evidence by revisiting the arm changed in the alternative")
    ax.grid(alpha=0.2)
    save_figure(fig, "distinguishing_arm_counts")


def experiment_gaussian_allocation(rng: np.random.Generator, rows: list[dict[str, object]]) -> None:
    mu1, mu2, sigma = 0.20, 0.00, 1.0
    total_budget = 400
    replications = 150000
    weights = np.arange(0.05, 1.00, 0.05)
    empirical_errors: list[float] = []
    exact_errors: list[float] = []
    information_rates: list[float] = []

    for weight in weights:
        n1 = max(1, int(round(total_budget * weight)))
        n2 = total_budget - n1
        mean1 = mu1 + sigma / math.sqrt(n1) * rng.standard_normal(replications)
        mean2 = mu2 + sigma / math.sqrt(n2) * rng.standard_normal(replications)
        empirical = float(np.mean(mean1 <= mean2))
        standard_error = sigma * math.sqrt(1.0 / n1 + 1.0 / n2)
        exact = float(norm.cdf(-(mu1 - mu2) / standard_error))
        effective_weight = n1 / total_budget
        information_rate = effective_weight * (1.0 - effective_weight) * (mu1 - mu2) ** 2 / (
            2.0 * sigma**2
        )
        empirical_errors.append(empirical)
        exact_errors.append(exact)
        information_rates.append(information_rate)
        rows.extend(
            [
                {
                    "experiment": "gaussian_allocation",
                    "x": effective_weight,
                    "setting": f"mu=({mu1},{mu2}), sigma={sigma}, budget={total_budget}",
                    "metric": "empirical_error_probability",
                    "value": empirical,
                },
                {
                    "experiment": "gaussian_allocation",
                    "x": effective_weight,
                    "setting": f"mu=({mu1},{mu2}), sigma={sigma}, budget={total_budget}",
                    "metric": "exact_error_probability",
                    "value": exact,
                },
                {
                    "experiment": "gaussian_allocation",
                    "x": effective_weight,
                    "setting": f"mu=({mu1},{mu2}), sigma={sigma}, budget={total_budget}",
                    "metric": "hardest_alternative_information_rate",
                    "value": information_rate,
                },
            ]
        )

    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(weights, exact_errors, marker="o", label="exact Gaussian error")
    ax.scatter(weights, empirical_errors, s=28, label="Monte Carlo")
    ax.axvline(0.5, linestyle="--", linewidth=1.0, label="equal allocation")
    ax.set_xlabel("fraction of samples assigned to arm 1")
    ax.set_ylabel("probability of recommending the wrong arm")
    ax.set_title("A poor allocation leaves one side of the comparison too noisy")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure(fig, "gaussian_allocation_error")

    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_subplot(111)
    ax.plot(weights, information_rates, marker="o")
    ax.axvline(0.5, linestyle="--", linewidth=1.0)
    ax.set_xlabel("fraction of samples assigned to arm 1")
    ax.set_ylabel("information rate against the hardest alternative")
    ax.set_title("The lower-bound game selects equal allocation in the symmetric Gaussian case")
    ax.grid(alpha=0.2)
    save_figure(fig, "gaussian_information_rate")


def experiment_alternative_geometry(rows: list[dict[str, object]]) -> None:
    mu1, mu2, sigma = 0.20, 0.00, 1.0
    means = np.linspace(-0.08, 0.28, 361)
    weights: Iterable[float] = (0.20, 0.50, 0.80)

    fig = plt.figure(figsize=(7.2, 4.5))
    ax = fig.add_subplot(111)
    for weight in weights:
        objective = (
            weight * (mu1 - means) ** 2 + (1.0 - weight) * (mu2 - means) ** 2
        ) / (2.0 * sigma**2)
        minimizer = weight * mu1 + (1.0 - weight) * mu2
        minimum = weight * (1.0 - weight) * (mu1 - mu2) ** 2 / (2.0 * sigma**2)
        ax.plot(means, objective, label=f"allocation w={weight:.1f}")
        ax.scatter([minimizer], [minimum], s=35)
        rows.extend(
            [
                {
                    "experiment": "alternative_geometry",
                    "x": weight,
                    "setting": f"Gaussian mu=({mu1},{mu2}), sigma={sigma}",
                    "metric": "hardest_common_mean",
                    "value": minimizer,
                },
                {
                    "experiment": "alternative_geometry",
                    "x": weight,
                    "setting": f"Gaussian mu=({mu1},{mu2}), sigma={sigma}",
                    "metric": "minimum_information_rate",
                    "value": minimum,
                },
            ]
        )
    ax.set_xlabel("common mean m in the boundary alternative")
    ax.set_ylabel(r"$wD(\mu_1,m)+(1-w)D(\mu_2,m)$")
    ax.set_title("The hardest alternative moves toward the least-observed arm")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_figure(fig, "gaussian_alternative_geometry")


def write_results(rows: list[dict[str, object]]) -> None:
    output = OUTPUT_DIR / "change_of_measure_results.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment", "x", "setting", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    experiment_evidence_paths(rng, rows)
    experiment_testing_bound(rows)
    experiment_adaptive_information(rng, rows)
    experiment_gaussian_allocation(rng, rows)
    experiment_alternative_geometry(rows)
    write_results(rows)
    print(f"Wrote figures and results to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

## Further Reading

The divergence-decomposition presentation in Lattimore and Szepesvari's *Bandit Algorithms* is an especially clear route from information theory to bandit lower bounds (Lattimore and Szepesvari 2020). Kaufmann, Cappe, and Garivier give a stopped change-of-distribution inequality tailored to best-arm identification and use it to derive instance-dependent complexity bounds (Kaufmann et al. 2016). Garivier and Kaufmann turn the resulting allocation game into the Track-and-Stop algorithm and a tight asymptotic characterization for one-parameter bandits (Garivier and Kaufmann 2016).

The regret direction begins with Lai and Robbins (Lai and Robbins 1985) and its multiparameter extension by Burnetas and Katehakis (Burnetas and Katehakis 1996). For the broader testing tradition behind these arguments, see Chernoff's sequential design work (Chernoff 1959), Wald's sequential analysis (Wald 1947), and modern lower-bound treatments by Tsybakov (Tsybakov 2009) and Le Cam and Yang (Le Cam and Yang 2000). Recent work by Yang, Tan, and Jin uses information-theoretic lower bounds to study best-arm identification with minimal cumulative regret (Yang et al. 2024), while Jin and collaborators study how near-optimal information allocation can be implemented under batching constraints (Jin et al. 2024).


Burnetas, Apostolos N., and Michael N. Katehakis. 1996. "Optimal Adaptive Policies for Sequential Allocation Problems." *Advances in Applied Mathematics* 17 (2): 122--42.


Chernoff, Herman. 1959. "Sequential Design of Experiments." *The Annals of Mathematical Statistics* 30 (3): 755--70.


Garivier, Aurélien, and Emilie Kaufmann. 2016. "Optimal Best Arm Identification with Fixed Confidence." *Proceedings of the 29th Annual Conference on Learning Theory*.


Jin, Tianyuan, Yuhang Yang, Jing Tang, Xiongjun Xiao, and Pan Xu. 2024. "Optimal Batched Best Arm Identification." *Advances in Neural Information Processing Systems 37*.


Kaufmann, Emilie, Olivier Cappé, and Aurélien Garivier. 2016. "On the Complexity of Best-Arm Identification in Multi-Armed Bandit Models." *Journal of Machine Learning Research* 17 (1): 1--42.


Lai, T. L., and H. Robbins. 1985. "Asymptotically Efficient Adaptive Allocation Rules." *Advances in Applied Mathematics* 6 (1): 4--22.


Lattimore, T., and C. Szepesvari. 2020. *Bandit Algorithms*. Cambridge University Press.


Le Cam, Lucien, and Grace Lo Yang. 2000. *Asymptotics in Statistics: Some Basic Concepts*. 2nd ed. Springer.


Mannor, Shie, and John N. Tsitsiklis. 2004. "The Sample Complexity of Exploration in the Multi-Armed Bandit Problem." *Journal of Machine Learning Research* 5: 623--48.


Tsybakov, Alexandre B. 2009. *Introduction to Nonparametric Estimation*. Springer.


Wald, Abraham. 1947. *Sequential Analysis*. Wiley.


Yang, J., V. Y. F. Tan, and T. Jin. 2024. *Best Arm Identification with Minimal Regret*.
