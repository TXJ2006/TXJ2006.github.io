---
title: "Why Bandits Matter"
subtitle: "Sequential Decision-Making Beyond Supervised Learning"
summary: "Sequential decisions under uncertainty and the basic bandit abstraction."
description: "Sequential decisions under uncertainty and the basic bandit abstraction."
date: 2026-06-15
lastmod: 2026-06-15
weight: 10
tags: ["Bandits", "Online Learning", "Sequential Decisions"]
draft: false
ShowToc: false
hideMeta: true
---

## Before the First Formula

The first question in a learning problem is not "which model should we use?" It is more primitive:

> Where did the data come from?

In supervised learning, the data usually arrive as a table. Someone has already collected the inputs and labels. The learner may be clever or foolish, but the table is there.

In bandit learning, the table is not there. The learner helps create it. Each action asks the world one question, and the world answers only that question.

This is the whole difference. A bandit algorithm is not merely predicting from data. It is choosing which data it will get to see.

> **The shortest useful definition.**
>
> A bandit problem is learning by trying one option at a time, while never seeing what would have happened under the options not tried.

That sentence is enough to begin. The mathematics enters only because we want to measure how costly this kind of learning is, and how a good algorithm keeps the cost under control.

A helpful mental picture is an ordinary recommendation slot. At 8:00 a.m., the system chooses one item to show. A user clicks or does not click. The system sees the outcome of the shown item, but it does not see the click outcomes of the hidden items. Tomorrow it must choose again, using yesterday's incomplete evidence.

This example is not special. The same shape appears when a teacher chooses one hint for a student, a doctor chooses one treatment, a lab chooses one experiment, or a prompt-selection system chooses one prompt for a language model. In all cases, the learner receives feedback from the chosen action, not from the unchosen alternatives.

The goal of this note is to keep that simple picture visible while building the main formulas. The style follows a useful tradition in modern bandit writing: start from the feedback pattern, use examples to make the pressure of the problem visible, and introduce proof machinery only when it pays rent. Lattimore and Szepesvari describe their book as mathematically focused but not a traditional chain of lemmas and theorems; they emphasize guiding principles, intuition, and depth over formal display (Lattimore and Szepesvari 2020). That is the standard I have in mind here.

> **Diagram.** Supervised learning starts from a table. Bandit learning builds a table by acting.

## The First Object Is the Feedback

### Do not start with a tuple

A common weak opening is to define a bandit problem as a list of symbols. That is technically correct and pedagogically backwards. The first object is not a tuple. The first object is the feedback.

Suppose there are several actions. At a given moment, you choose one. The world returns one number. The other numbers remain unobserved.

Only now do the symbols become useful. Let $A_t$ be the action chosen at time $t$, and let $X_t(A_t)$ be the reward observed after choosing it. If $a\neq A_t$, then $X_t(a)$ is not observed.

> **The feedback pattern.**
>
> \[
> \begin{aligned}
> \text{choose }A_t
> &\quad\Longrightarrow\quad \text{observe }X_t(A_t),\\
> \text{do not choose }a
> &\quad\Longrightarrow\quad \text{do not observe }X_t(a).
> \end{aligned}
> \]

The second line is the source of almost everything in bandit theory. The missing outcome is not a nuisance. It is the problem.

### The missing column

Imagine two possible actions, $A$ and $B$. For a single user, there are two potential outcomes: $X(A),\qquad X(B).$ One says what would happen if we chose $A$. The other says what would happen if we chose $B$. We can reveal only one.


\[
\begin{array}{c|cc|c}
\text{chosen action} & X(A) & X(B) & \text{observed outcome}\\
\midrule
A & \text{visible} & \text{hidden} & X(A)\\
B & \text{hidden} & \text{visible} & X(B)
\end{array}
\]


A supervised-learning dataset tries to fill the label column for every example. A bandit dataset has a hole in every row. The hole is exactly where the unchosen actions would have been.

> **A useful plain-language test.**
>
> If after acting you can ask, "what would have happened if I had chosen differently?" and the data cannot answer, you are in bandit territory.

### Why bad actions damage future data

In supervised learning, a bad prediction is painful, but the label still arrives. In bandit learning, a bad action can also damage the learner's future evidence.

If an algorithm always chooses the currently popular item, it keeps collecting data about that item. The less popular item may remain uncertain forever. The learner is not only using biased data; it is generating the bias.

This is why bandit algorithms need exploration. Exploration is not curiosity for its own sake. It is the repair mechanism for missing feedback.

> **Diagram.** The learner’s action controls the next piece of data. This feedback loop is the main difference from ordinary supervised learning.

## A Small Experiment Before More Theory

### The same world, two ways to learn

Before adding more probability, it is useful to run a tiny experiment. The point is not to make a benchmark. The point is to make the feedback pattern visible.

There are three arms. You may think of them as three possible thumbnails, three prompts, or three treatments in a toy clinical example. Their click probabilities are $\mu_1=0.10,\qquad \mu_2=0.12,\qquad \mu_3=0.16.$ Arm $3$ is best, but the learner does not know this.

We compare four learners in exactly the same world.

- **Full-feedback learner.** It chooses one arm but observes the reward of every arm. This is the supervised-learning spirit: labels are supplied by the data-collection process rather than purchased one action at a time.

- **Greedy bandit.** It tries each arm once, then always chooses the arm with the largest observed average.

- **UCB.** It chooses the arm with the largest empirical mean plus an uncertainty bonus.

- **Thompson sampling.** It samples a plausible click rate for every arm from its posterior and chooses the arm whose sampled value is largest.

The full-feedback learner is not meant to be a general supervised-learning algorithm. It is the cleanest possible full-information analogue. Its purpose is to isolate the difference between seeing all labels and seeing only the label of the action we picked.

> **What the experiment should reveal.**
>
> If every round gives labels for all arms, learning is mostly estimation. If every round gives only one label, learning is also data acquisition. The learner must decide which uncertainty is worth reducing.

### The algorithms in code

The code below is deliberately plain. No library hides the learning rule. Each algorithm keeps a few counters and chooses an arm.

    import numpy as np

    def run_once(p, T, rng):
        K = len(p)
        best = np.max(p)
        regret = {"full": [], "greedy": [], "ucb": [], "ts": []}

        # 1. Full feedback: choose one arm, observe all arms.
        sums = np.zeros(K)
        counts = np.zeros(K)
        for t in range(T):
            if t < K:
                a = t
            else:
                a = np.argmax(sums / np.maximum(counts, 1))
            regret["full"].append(best - p[a])

            rewards_all = rng.binomial(1, p)   # the full row is visible
            sums += rewards_all
            counts += 1

        # 2. Greedy bandit: observe only the chosen arm.
        sums = np.zeros(K)
        counts = np.zeros(K)
        for t in range(T):
            if t < K:
                a = t
            else:
                a = np.argmax(sums / np.maximum(counts, 1))
            r = rng.binomial(1, p[a])          # only this reward is visible
            regret["greedy"].append(best - p[a])
            sums[a] += r
            counts[a] += 1

        # 3. UCB: empirical mean plus an uncertainty bonus.
        sums = np.zeros(K)
        counts = np.zeros(K)
        for t in range(T):
            if t < K:
                a = t
            else:
                mean = sums / counts
                radius = np.sqrt(2 * np.log(t + 1) / counts)
                a = np.argmax(mean + radius)
            r = rng.binomial(1, p[a])
            regret["ucb"].append(best - p[a])
            sums[a] += r
            counts[a] += 1

        # 4. Thompson sampling for Bernoulli rewards.
        alpha = np.ones(K)
        beta = np.ones(K)
        for t in range(T):
            sample = rng.beta(alpha, beta)
            a = np.argmax(sample)
            r = rng.binomial(1, p[a])
            regret["ts"].append(best - p[a])
            alpha[a] += r
            beta[a] += 1 - r

        return {name: np.cumsum(vals) for name, vals in regret.items()}

Read the code slowly. The full-feedback learner has the line $\texttt{rewards\_all = rng.binomial(1, p)}.$ That line gives it a whole row of labels. The bandit algorithms have the line $\texttt{r = rng.binomial(1, p[a])}.$ That line gives them only the chosen label. The difference between these two lines is the difference between supervised-style feedback and bandit feedback.

### What the code produces

I ran the experiment for $T=2000$ rounds and averaged over $400$ independent runs. The table reports the final cumulative regret and the average number of pulls of each arm.

  Algorithm             Final regret   Pulls of arm 1   Pulls of arm 2   Pulls of arm 3
  ------------------- -------------- ---------------- ---------------- ----------------
  Full feedback               $4.57$           $26.2$           $74.8$         $1899.0$
  Greedy bandit              $98.08$         $1504.0$          $196.0$          $300.0$
  UCB                        $49.81$          $448.7$          $572.2$          $979.1$
  Thompson sampling          $21.23$          $160.6$          $289.8$         $1549.5$

  : Average behavior over 400 runs with true means $(0.10,0.12,0.16)$ and horizon $T=2000$.

The numbers tell the story without any theorem.

The full-feedback learner rapidly finds the best arm because every round teaches it about every arm. Greedy often gets trapped by early noise. UCB keeps checking uncertain arms because uncertainty itself becomes part of the index. Thompson sampling explores in a softer way: an arm is tried when its posterior still gives it a plausible chance of being best.

![Average cumulative regret. The greedy bandit can pay a large price for believing early noise. UCB and Thompson sampling reduce this cost by making uncertainty operational.](/images/notes/assets/bandits/regret_curves.webp)

*Average cumulative regret. The greedy bandit can pay a large price for believing early noise. UCB and Thompson sampling reduce this cost by making uncertainty operational.*

![Average pull counts. The best arm is arm 3. The greedy learner often spends many rounds on the wrong arm; the sampling and optimism-based methods move probability mass toward the best arm while still collecting evidence.](/images/notes/assets/bandits/pull_counts.webp)

*Average pull counts. The best arm is arm 3. The greedy learner often spends many rounds on the wrong arm; the sampling and optimism-based methods move probability mass toward the best arm while still collecting evidence.*

### The lesson hidden in the experiment

A supervised learner can be viewed as a reader of a completed table. A bandit learner is a writer of an incomplete table. This makes a qualitative difference.

> **The one-line difference.**
>
> \[
> \begin{aligned}
> \text{full feedback:}&\qquad (X_t(1),X_t(2),X_t(3))\text{ is observed},\\
> \text{bandit feedback:}&\qquad X_t(A_t)\text{ is observed, and }X_t(a)\text{ for }a\neq A_t\text{ is missing}.
> \end{aligned}
> \]

Now the later formulas have a concrete object to explain. Regret measures the cost of not knowing the best arm. Confidence intervals explain why UCB does not blindly trust early averages. Posterior sampling explains why Bayesian uncertainty can be converted into action.

## Probability, Introduced Only When Needed

### A reward is a number we have not seen yet

Probability can sound abstract when introduced as a formal system. In bandits, its first role is modest: it gives language for numbers we have not seen yet.

Before an item is shown, its click outcome is unknown. After it is shown, the outcome becomes a number. We model the before-state by a random variable.

For a click/no-click reward,

\[
X=\begin{cases}
1, & \text{click},\\
0, & \text{no click}.
\end{cases}
\]

 If the click probability is $\mu$, then

\[
\Pp(X=1)=\mu,
\qquad
\Pp(X=0)=1-\mu.
\]

 The average value of this random reward is

\[
\begin{aligned}
\E[X]
&=1\cdot \Pp(X=1)+0\cdot \Pp(X=0)\\
&=1\cdot \mu+0\cdot(1-\mu)\\
&=\mu.
\end{aligned}
\]


So in a click model, the mean reward is simply the click probability.

### The sample average is a noisy ruler

If we try the same arm $n$ times, we see rewards $Y_1,Y_2,\ldots,Y_n.$ The empirical mean is $\widehat\mu_n=\frac{1}{n}\sum_{i=1}^n Y_i.$ This is the observed click rate. It is our ruler for the unknown mean $\mu$.

But the ruler is noisy. If a thumbnail has true click probability $0.10$, it can still receive $2$ clicks in $10$ trials or $0$ clicks in $10$ trials. Small samples wobble.

The first probability question is therefore:

> How far can the observed average be from the true average?

For bounded rewards in $[0,1]$, Hoeffding's inequality says that large errors become exponentially unlikely:

\[
\Pp\left(\left|\widehat\mu_n-\mu\right|\geq r\right)
\leq
2\exp(-2nr^2).
\]


This line is not meant to be memorized as a magic spell. Read it from right to left:

\[
\begin{array}{ccl}
 n \text{ grows} &\Longrightarrow& \exp(-2nr^2) \text{ shrinks},\\
 r \text{ grows} &\Longrightarrow& \exp(-2nr^2) \text{ shrinks}.
\end{array}
\]

 More data makes a fixed error less likely. A larger error is also less likely.

### Turning a tail bound into an error bar

Algorithms usually want an error bar. That means choosing $r$ so that the probability of error is at most a small number $\delta$:

\[
\Pp\left(\left|\widehat\mu_n-\mu\right|\geq r\right)
\leq \delta.
\]

 Hoeffding gives the sufficient condition $2\exp(-2nr^2)=\delta.$ Solve this slowly:

> **From Hoeffding to a confidence radius.**
>
> \[
> \begin{aligned}
> 2\exp(-2nr^2)&=\delta\\
> \exp(-2nr^2)&=\frac{\delta}{2}\\
> \log\left(\exp(-2nr^2)\right)&=\log\left(\frac{\delta}{2}\right)\\
> -2nr^2&=\log\left(\frac{\delta}{2}\right)\\
> 2nr^2&=\log\left(\frac{2}{\delta}\right)\\
> r^2&=\frac{\log(2/\delta)}{2n}\\
> r&=\sqrt{\frac{\log(2/\delta)}{2n}}.
> \end{aligned}
> \]

Thus a clean error bar is

\[
\left|\widehat\mu_n-\mu\right|
\leq
\sqrt{\frac{\log(2/\delta)}{2n}}
\]

 with probability at least $1-\delta$.

> **The idea behind the formula.**
>
> An arm tried only a few times deserves a wide error bar. An arm tried many times deserves a narrow one. UCB is built from exactly this sentence.

> **Diagram.** The confidence radius shrinks like $1/\sqrt{n}$. This shrinking error bar is the engine inside UCB.

### The union bound is not mysterious

Many proofs need a statement to hold for many arms and many times. We first control one error event, then combine many such controls.

The only tool needed is the union bound:

\[
\Pp(E_1\cup E_2\cup\cdots\cup E_m)
\leq
\Pp(E_1)+\Pp(E_2)+\cdots+\Pp(E_m).
\]


Why is this true? If two alarms can ring, the chance that at least one rings cannot exceed the sum of their separate chances. The sum may double-count worlds where both alarms ring, so it is an upper bound.

For two events, the exact identity is

\[
\Pp(E_1\cup E_2)
=
\Pp(E_1)+\Pp(E_2)-\Pp(E_1\cap E_2).
\]

 Since probabilities are nonnegative, $\Pp(E_1\cap E_2)\geq 0,$ so

\[
\Pp(E_1\cup E_2)
\leq
\Pp(E_1)+\Pp(E_2).
\]

 For many events, repeat the same idea.

### History means what the learner already knows

Bandit data are adaptive: today's action depends on yesterday's observations. To talk about this without mysticism, we use the word history.

Let $\calF_t$ denote everything known before choosing at time $t$: $\calF_t=\{A_1,X_1(A_1),\ldots,A_{t-1},X_{t-1}(A_{t-1})\}.$ An algorithm chooses $A_t$ using this history. Then the new reward arrives and the history grows.

Conditional probability simply means probability after opening the notebook:

\[
\Pp(\text{event}\mid \calF_t)
=
\text{probability of the event given the current notebook}.
\]


> **A nontechnical reading of $\calF_t$.**
>
> The symbol $\calF_t$ is not there to impress the reader. It means the notebook of everything the learner has seen so far.

This is enough probability for the first bandit algorithms: random rewards, means, sample averages, error bars, union bounds, and history.

## Regret Is the Price of Learning

### The meaning before the formula

A bandit learner must sometimes choose an action that may not be best, because otherwise it may never learn what is best. The cost of these imperfect choices is called regret.

Think of regret as the price paid for not already knowing the best action. If the best thumbnail would have produced a $10\%$ click rate and the chosen thumbnail has an $8\%$ click rate, the expected loss for that decision is $2\%$ of a click.

This does not mean exploration is bad. Exploration is often the only way to stop making the same comfortable mistake. Regret is not a moral judgment. It is accounting.

### Now the notation earns its place

Suppose there are $K$ arms. Arm $a$ has mean reward $\mu_a=\E[X(a)].$ The best mean is $\mu_* = \max_{1\leq a\leq K}\mu_a.$ If the learner chooses $A_t$, then the expected loss at that time is $\mu_* - \mu_{A_t}.$ Adding over $T$ rounds gives pseudo-regret: $\Reg_T=\sum_{t=1}^T(\mu_* - \mu_{A_t}).$

This compares the learner with the best fixed arm, not with a magical oracle that knows every random outcome in advance.

### The key decomposition

For each arm define its gap $\Delta_a=\mu_* - \mu_a,$ and define the number of times arm $a$ is played by $N_a(T)=\sum_{t=1}^T \ind\{A_t=a\}.$ The indicator $\ind\{A_t=a\}$ is just a switch:

\[
\ind\{A_t=a\}=\begin{cases}
1,& A_t=a,\\
0,& A_t\neq a.
\end{cases}
\]


Now decompose regret:

> **Regret as gap times count.**
>
> \[
> \begin{aligned}
> \Reg_T
> &=\sum_{t=1}^T(\mu_* - \mu_{A_t})\\
> &=\sum_{t=1}^T\sum_{a=1}^K(\mu_* - \mu_a)\ind\{A_t=a\}\\
> &=\sum_{a=1}^K(\mu_* - \mu_a)\sum_{t=1}^T\ind\{A_t=a\}\\
> &=\sum_{a=1}^K\Delta_a N_a(T).
> \end{aligned}
> \]

This formula is the first serious piece of bandit analysis. It says that a bad arm hurts in two ways: how bad it is, and how often it is played.

### Checking the switch by hand

Suppose that at time $t$ the algorithm chose arm $j$. Then exactly one indicator is equal to $1$:

\[
\ind\{A_t=j\}=1,
\qquad
\ind\{A_t=a\}=0\quad(a\neq j).
\]

 Therefore

\[
\begin{aligned}
\sum_{a=1}^K(\mu_* - \mu_a)\ind\{A_t=a\}
&=(\mu_* - \mu_1)\ind\{j=1\}+\cdots+(\mu_* - \mu_K)\ind\{j=K\}\\
&=0+\cdots+0+(\mu_* - \mu_j)+0+\cdots+0\\
&=\mu_* - \mu_j\\
&=\mu_* - \mu_{A_t}.
\end{aligned}
\]


No probability is being used here. It is bookkeeping.

### Taking expectation

The count $N_a(T)$ is random because the algorithm's choices depend on random rewards. Its expectation is easy to interpret.

For any event $E$,

\[
\E[\ind\{E\}]
=1\cdot\Pp(E)+0\cdot\Pp(E^c)
=\Pp(E).
\]

 Therefore

\[
\begin{aligned}
\E[N_a(T)]
&=\E\left[\sum_{t=1}^T\ind\{A_t=a\}\right]\\
&=\sum_{t=1}^T\E[\ind\{A_t=a\}]\\
&=\sum_{t=1}^T\Pp(A_t=a).
\end{aligned}
\]

 Taking expectation in the regret decomposition gives

\[
\E[\Reg_T]
=
\sum_{a=1}^K\Delta_a\E[N_a(T)].
\]


> **Takeaway.**
>
> To prove a regret bound, we usually prove that bad arms are not played too often. Everything else is a way of making that sentence true.

## A Simple Failure: Explore-Then-Commit

### Taste first, then order forever

A natural first strategy is simple. Try each thumbnail for a while. Estimate its click rate. Then use the one that looks better.

This is called explore-then-commit. The name is plain because the algorithm is plain.

> **A restaurant analogy.**
>
> A friend says: try two restaurants twice, then pick one for every dinner this month. This may sound reasonable. But if two restaurants are close in quality, two visits are not enough. If one restaurant is clearly bad, too many test visits are wasteful. The fixed trial length is both too small and too large.

Let there be two arms. Arm 1 is better:

\[
\mu_1>\mu_2,
\qquad
\Delta=\mu_1-\mu_2>0.
\]

 Explore-then-commit pulls each arm $m$ times. Then it chooses the arm with larger empirical mean.

### The two costs

There are two possible sources of regret.

First, exploration pulls the bad arm $m$ times: $\text{exploration regret}=m\Delta.$ Second, after exploration, the algorithm may wrongly commit to arm 2. If this happens, it pays roughly $\Delta$ for every remaining round:

\[
\text{wrong-commit regret}\leq (T-2m)\Delta\cdot \Pp(\widehat\mu_2\geq \widehat\mu_1).
\]

 Thus

\[
\E[\Reg_T]
\leq
m\Delta+(T-2m)\Delta\Pp(\widehat\mu_2\geq \widehat\mu_1).
\]


> **Diagram.** Explore-then-commit separates learning and earning. The separation makes the analysis easy and exposes what later algorithms improve.

### Bounding the wrong commitment probability

The algorithm commits wrongly if $\widehat\mu_2\geq \widehat\mu_1.$ This event can happen only if at least one empirical mean makes a noticeable error.

Since $\mu_1-\mu_2=\Delta,$ the midpoint between the two means is $\frac{\mu_1+\mu_2}{2}=\mu_1-\frac{\Delta}{2}=\mu_2+\frac{\Delta}{2}.$ If both estimates are on the correct side of this midpoint, then arm 1 wins: $\widehat\mu_1>\frac{\mu_1+\mu_2}{2}>\widehat\mu_2.$ Therefore, the wrong event implies at least one failure:

\[
\{\widehat\mu_2\geq\widehat\mu_1\}
\subseteq
\left\{\widehat\mu_1\leq \mu_1-\frac{\Delta}{2}\right\}
\cup
\left\{\widehat\mu_2\geq \mu_2+\frac{\Delta}{2}\right\}.
\]


For bounded rewards in $[0,1]$, Hoeffding's inequality says

\[
\Pp(\widehat\mu-\mu\geq \varepsilon)
\leq
\exp(-2m\varepsilon^2),
\]

 and the same bound holds for downward deviations. With $\varepsilon=\Delta/2$,

\[
\exp\left(-2m\left(\frac{\Delta}{2}\right)^2\right)
=
\exp\left(-\frac{m\Delta^2}{2}\right).
\]

 Using a union bound: $\Pp(E_1\cup E_2)\leq \Pp(E_1)+\Pp(E_2).$ So

\[
\Pp(\widehat\mu_2\geq \widehat\mu_1)
\leq
2\exp\left(-\frac{m\Delta^2}{2}\right).
\]


> **Wrong commitment, line by line.**
>
> \[
> \begin{aligned}
> \Pp(\widehat\mu_2\geq\widehat\mu_1)
> &\leq
> \Pp\left(\widehat\mu_1\leq \mu_1-\frac{\Delta}{2}\right)
> +
> \Pp\left(\widehat\mu_2\geq \mu_2+\frac{\Delta}{2}\right)\\
> &\leq
> \exp\left(-2m\left(\frac{\Delta}{2}\right)^2\right)
> +
> \exp\left(-2m\left(\frac{\Delta}{2}\right)^2\right)\\
> &=
> 2\exp\left(-\frac{m\Delta^2}{2}\right).
> \end{aligned}
> \]

Plugging this into regret gives

\[
\E[\Reg_T]
\leq
m\Delta+2T\Delta\exp\left(-\frac{m\Delta^2}{2}\right).
\]


### What this simple algorithm teaches

Explore-then-commit is not the algorithm one should trust in a serious system. Its value is pedagogical. It shows the main bargain.

If $m$ is small, the second term can be large: the learner may commit to the wrong arm. If $m$ is large, the first term is large: the learner wastes too many trials on a known bad arm.

> **The bargain.**
>
> Exploration is like paying rent for information. Too little rent and you stay ignorant. Too much rent and you never earn enough from what you learned.

The next idea, optimism, improves this by not fixing the exploration budget in advance.

## Optimism: Trust the Unknown Until It Disappoints You

### The idea in one sentence

Optimism is not positivity. It is a rule for action under uncertainty.

> **Optimism.**
>
> If an arm has not been tried much, allow it to look better than its current average. If it is truly bad, enough trials will shrink this allowance and remove it from serious competition.

This is the idea behind UCB: upper confidence bound.

At time $t$, for each arm $a$, compute $\text{index}_a(t)=\widehat\mu_a(t)+r_a(t),$ where $\widehat\mu_a(t)$ is the observed average and $r_a(t)$ is a confidence radius. Choose the arm with the largest index.

$A_t\in\argmax_a\left\{\widehat\mu_a(t)+r_a(t)\right\}.$

> **Diagram.** UCB chooses the arm with the best plausible value, not necessarily the best current average.

### Where the radius comes from

Let an arm be pulled $n$ times. Let its rewards be $Y_1,\ldots,Y_n$, each in $[0,1]$, with mean $\mu$. The empirical mean is $\widehat\mu_n=\frac{1}{n}\sum_{i=1}^n Y_i.$ Hoeffding's inequality says

\[
\Pp\left(\widehat\mu_n-\mu\geq r\right)
\leq
\exp(-2nr^2).
\]

 We want the right side to be a tiny number, say $t^{-4}$. So choose $r$ by solving $\exp(-2nr^2)=t^{-4}.$ Now solve it slowly:

> **Solving for the UCB radius.**
>
> \[
> \begin{aligned}
> \exp(-2nr^2)&=t^{-4}\\
> \log\left(\exp(-2nr^2)\right)&=\log(t^{-4})\\
> -2nr^2&=-4\log t\\
> 2nr^2&=4\log t\\
> r^2&=\frac{2\log t}{n}\\
> r&=\sqrt{\frac{2\log t}{n}}.
> \end{aligned}
> \]

Thus a natural confidence radius is $r_a(t)=\sqrt{\frac{2\log t}{N_a(t)}}.$ It shrinks when the arm is sampled often. It grows slowly with time because the algorithm makes many decisions and we want confidence to hold across many of them.

### The good event

Here is the event we want:

\[
\left|\widehat\mu_a(t)-\mu_a\right|
\leq
r_a(t)
\quad\text{for all arms and relevant times.}
\]

 Call this event $G$. On $G$, every confidence interval contains its true mean.

Why can we hope $G$ holds? Because each individual failure is rare, and a union bound lets us add many rare failures.

> **Union bound in the UCB setting.**
>
> For one arm and one sample size $n$,

\[
\Pp\left(\widehat\mu_{a,n}-\mu_a>\sqrt{\frac{2\log T}{n}}\right)
\leq
\exp\left(-2n\cdot\frac{2\log T}{n}\right)
=T^{-4}.
\]

 The lower tail is the same:

\[
\Pp\left(\mu_a-\widehat\mu_{a,n}>\sqrt{\frac{2\log T}{n}}\right)
\leq T^{-4}.
\]

 Therefore

\[
\Pp\left(\left|\widehat\mu_{a,n}-\mu_a\right|>\sqrt{\frac{2\log T}{n}}\right)
\leq 2T^{-4}.
\]

 There are at most $KT$ pairs $(a,n)$, so

\[
\Pp(G^c)
\leq
\sum_{a=1}^K\sum_{n=1}^T 2T^{-4}
=2KT^{-3}.
\]


No mystery is hidden here. The good event is just the statement that all sample averages are close to their true means. The union bound is just the rule that if many bad things can happen, the chance that at least one happens is at most the sum of their chances.

### Why a bad arm eventually loses

Let $a$ be a suboptimal arm. Its gap is $\Delta_a=\mu_* - \mu_a>0.$ Suppose the good event $G$ holds and UCB chooses arm $a$ at time $t$. Since UCB chose $a$, its index must be at least the index of the best arm $*$:

\[
\widehat\mu_a(t)+r_a(t)
\geq
\widehat\mu_*(t)+r_*(t).
\]

 On $G$, $\widehat\mu_*(t)+r_*(t)\geq \mu_*.$ Also on $G$,

\[
\widehat\mu_a(t)
\leq \mu_a+r_a(t).
\]

 Thus

\[
\widehat\mu_a(t)+r_a(t)
\leq
\mu_a+2r_a(t).
\]

 Putting the inequalities together:

\[
\mu_*
\leq
\mu_a+2r_a(t).
\]

 Therefore $\Delta_a\leq 2r_a(t).$ Using $r_a(t)=\sqrt{2\log T/N_a(t)}$: $\Delta_a\leq 2\sqrt{\frac{2\log T}{N_a(t)}}.$ Now solve for $N_a(t)$:

> **A bad arm can be chosen only while its radius is large.**
>
> \[
> \begin{aligned}
> \Delta_a&\leq 2\sqrt{\frac{2\log T}{N_a(t)}}\\
> \frac{\Delta_a}{2}&\leq \sqrt{\frac{2\log T}{N_a(t)}}\\
> \frac{\Delta_a^2}{4}&\leq \frac{2\log T}{N_a(t)}\\
> N_a(t)&\leq \frac{8\log T}{\Delta_a^2}.
> \end{aligned}
> \]

This is the entire UCB proof in one picture: a bad arm is played only while it can still hide behind uncertainty. Once its confidence interval is narrow enough, it cannot beat the best arm's lower reality anymore.

### The regret bound

On the good event, each bad arm $a$ is pulled at most about $\frac{8\log T}{\Delta_a^2}$ times after initialization. Its regret contribution is gap times count:

\[
\Delta_a\cdot \frac{8\log T}{\Delta_a^2}
=
\frac{8\log T}{\Delta_a}.
\]

 Adding arms gives

\[
\E[\Reg_T]
\lesssim
\sum_{a:\Delta_a>0}\frac{\log T}{\Delta_a}.
\]


> **Takeaway.**
>
> UCB is not a complicated formula. It is a disciplined way to say: try uncertain things while they could plausibly be good; stop trying them once the data have made that hope too expensive to believe.

## Sampling a Possible World

### The Bayesian picture

Optimism says: choose the arm with the best plausible upper value.

Posterior sampling says something lighter:

> **Posterior sampling.**
>
> Imagine one possible world consistent with the data. Act optimally in that imagined world. Repeat this every round.

This is Thompson sampling. It is one of the cleanest ideas in sequential learning. It often looks almost too simple: keep a posterior distribution over the unknown rewards, sample from it, and choose the arm that is best under the sample.

### A coin-click model

Suppose each thumbnail has an unknown click probability $\theta_a$. A click is $1$, no click is $0$: $X_t(a)\sim\Ber(\theta_a).$ For each arm, start with a Beta prior: $\theta_a\sim\Beta(\alpha_a,\beta_a).$ If the arm receives a click, increase $\alpha_a$ by one. If it receives no click, increase $\beta_a$ by one.

> **Beta-Bernoulli update.**
>
> \[
> \begin{aligned}
> \text{prior density:}\quad
> p(\theta)&\propto \theta^{\alpha-1}(1-\theta)^{\beta-1},\\
> \text{one click:}\quad
> p(1\mid \theta)&=\theta,\\
> \text{posterior:}\quad
> p(\theta\mid 1)&\propto \theta\cdot \theta^{\alpha-1}(1-\theta)^{\beta-1}\\
> &=\theta^{\alpha}(1-\theta)^{\beta-1},\\
> \theta\mid 1&\sim \Beta(\alpha+1,\beta).
> \end{aligned}
> \]

\[
\begin{aligned}
\text{one no-click:}\quad
p(0\mid\theta)&=1-\theta,\\
p(\theta\mid 0)&\propto (1-\theta)\theta^{\alpha-1}(1-\theta)^{\beta-1}\\
&=\theta^{\alpha-1}(1-\theta)^{\beta},\\
\theta\mid 0&\sim\Beta(\alpha,\beta+1).
\end{aligned}
\]


After $S_a$ clicks and $F_a$ no-clicks on arm $a$, $\theta_a\mid\text{data}\sim\Beta(\alpha_a+S_a,\beta_a+F_a).$

### The decision rule

At time $t$: $\widetilde\theta_a(t)\sim\Beta(\alpha_a+S_a(t),\beta_a+F_a(t)),$ independently for each arm, and choose $A_t\in\argmax_a \widetilde\theta_a(t).$

> **Diagram.** Thompson sampling does not directly add a confidence bonus. It samples a plausible world and follows the best arm in that world.

### Why this explores

If an arm has little data, its posterior is wide. A wide posterior occasionally samples a high value. That gives the arm chances to be tried. If the arm is bad, the failures push its posterior down and make such lucky samples rarer.

So the exploration is not manually scheduled. It emerges from uncertainty.

> **Classical paradigm: probability matching.**
>
> Thompson sampling chooses an arm with probability equal to the posterior probability that the arm is optimal. Arms are tried according to how believable their optimality is under current evidence.

Let $A^*$ be the optimal arm under the unknown true parameter. Given the history $\calF_t$, posterior sampling satisfies

\[
\Pp(A_t=a\mid\calF_t)
=\Pp(A^*=a\mid\calF_t).
\]

 This identity is the clean algebraic soul of Thompson sampling.

### The identity slowly

Let $\theta$ be the unknown vector of arm means. Let $g(\theta)=\argmax_a \theta_a.$ The true optimal arm is $A^*=g(\theta).$ Thompson sampling draws $\widetilde\theta\sim \text{posterior of }\theta\text{ given }\calF_t,$ and plays $A_t=g(\widetilde\theta).$ Because $\theta$ and $\widetilde\theta$ have the same conditional distribution given $\calF_t$,

\[
\theta\mid\calF_t
\quad\overset{d}{=}\quad
\widetilde\theta\mid\calF_t.
\]

 Therefore applying the same map $g$ preserves the conditional distribution:

\[
g(\theta)\mid\calF_t
\quad\overset{d}{=}\quad
g(\widetilde\theta)\mid\calF_t.
\]

 Thus

\[
A^*\mid\calF_t
\quad\overset{d}{=}\quad
A_t\mid\calF_t.
\]

 So for each arm $a$, $\Pp(A^*=a\mid\calF_t)=\Pp(A_t=a\mid\calF_t).$

This proof uses no heavy probability. It says: if two hidden worlds are drawn from the same bag, then the winner in the first world and the winner in the second world have the same chance of being any given arm.

### Information accounting

Many modern Thompson sampling analyses, including work in the broader line of information-ratio proofs, ask a simple question:

> How much regret do we pay per unit of information gained?

The technical details can become intricate, but the idea is almost everyday reasoning. If a trial is expensive but teaches a lot, it may be worth it. If a trial is expensive and teaches little, a good algorithm should avoid repeating it.

This way of thinking connects cleanly to recent theoretical work on Thompson sampling, best-arm identification with regret control, and multi-agent bandits, including work by Jin and collaborators (Jin et al. 2022, 2024; Yang et al. 2024). The common flavor is not just to design a rule, but to prove that the rule pays for uncertainty in a controlled way.

## Why This Toy Model Keeps Reappearing

### Context changes the arms

The homepage example was too simple: every visitor was treated the same. In practice, users arrive with context: time, device, language, past behavior, location, or topic preference.

A contextual bandit sees a context $x_t$, chooses an action $A_t$, and receives the reward of that action. The missing counterfactual remains:

\[
\text{observe }X_t(A_t,x_t),
\qquad
\text{miss }X_t(a,x_t),\ a\neq A_t.
\]

 The setting becomes richer, but the wound is the same.

> **Diagram.** Contextual bandits add side information before the action. They do not restore the missing outcomes of unchosen actions.

### Linear bandits: the geometry version

In a linear bandit, each action has a feature vector $x_a\in\R^d$, and the mean reward is assumed to be approximately linear: $\mu_a=x_a^\top\theta_*.$ The unknown object is now a vector $\theta_*$. Each action gives one noisy measurement of this vector in the direction $x_a$.

A helpful picture is this: learning a linear bandit is like trying to locate a hidden point in space by shining flashlights from different directions. Some directions reveal new information. Others repeat what is already known.

The UCB idea becomes geometric: $\text{score}(x)=x^\top\widehat\theta+\beta\sqrt{x^\top V^{-1}x}.$ The first term is what we currently believe. The second term is how uncertain we are in direction $x$.

> **Linear confidence in one line.**
>
> \[
> \begin{aligned}
> |x^\top(\widehat\theta-\theta_*)|
> &=|\langle x,\widehat\theta-\theta_*\rangle|\\
> &=|\langle V^{-1/2}x,V^{1/2}(\widehat\theta-\theta_*)\rangle|\\
> &\leq \|V^{-1/2}x\|_2\cdot\|V^{1/2}(\widehat\theta-\theta_*)\|_2\\
> &\leq \sqrt{x^\top V^{-1}x}\cdot \beta.
> \end{aligned}
> \]

The last inequality is just Cauchy-Schwarz plus a confidence ellipsoid. This is the geometric heart of linear UCB.

### Gaussian-process bandits: smooth unknown landscapes

Sometimes actions are not a small list. They may be continuous: temperature in a lab, a hyperparameter, a drug dosage, a robotics controller, or a design choice. The reward function may be smooth but unknown.

Gaussian-process bandits model the unknown function as a random smooth landscape. The algorithm samples or optimizes where the landscape might be high, while learning the shape of the landscape from noisy evaluations.

The same old idea appears again: $\text{mean prediction} + \text{uncertainty bonus}.$ For GP-UCB this takes the form

\[
A_t\in\argmax_{x\in\calX}\left\{m_{t-1}(x)+\sqrt{\beta_t}\sigma_{t-1}(x)\right\}.
\]

 This line connects finite-armed bandits to Bayesian optimization, a major line of work including Srinivas, Krause, Kakade, and Seeger (Srinivas et al. 2010), the Cambridge probabilistic modelling tradition (Ghahramani 2015), and Oxford work on Gaussian-process global optimization (Osborne et al. 2009).

> **Diagram.** In GP bandits, the confidence interval becomes a band over a function. The old finite-arm picture survives, but the arms now live in a space.

### Reliable AI and the same old missing column

The same obstruction appears in modern systems. A recommender only observes the item it showed. A tutoring system only observes the hint it gave. A medical decision support tool may only observe the outcome under the chosen treatment. An LLM steering policy only observes the response produced under the chosen prompt or intervention.

The names change. The missing counterfactual remains.

This is why bandits are not just a topic. They are a microscope. They strip sequential decision-making down to its smallest difficult piece.

## Epilogue: The Essence

The whole note can be compressed into a few sentences.

Supervised learning receives examples. Bandit learning creates examples by acting. Because only the chosen action is observed, the learner must spend some reward to buy information. Regret is the receipt for that purchase.

Explore-then-commit buys information on a fixed schedule. UCB buys it adaptively by giving uncertain arms temporary credit. Thompson sampling buys it by sampling possible worlds and acting according to their winners. Linear and Gaussian-process bandits carry the same ideas into structured spaces.

The deepest lesson is not a formula. It is a research habit: before designing an algorithm, ask what feedback is actually visible. Most of the mathematics follows from that one question.

\[heading=bibintoc,title=References\]


Ghahramani, Z. 2015. "Probabilistic Machine Learning and Artificial Intelligence." *Nature* 521: 452--59.


Jin, T., H.-L. Hsu, W. Chang, and P. Xu. 2024. "Finite-Time Frequentist Regret Bounds of Multi-Agent Thompson Sampling on Sparse Hypergraphs." *AAAI Conference on Artificial Intelligence*.


Jin, T., P. Xu, X. Xiao, and A. Anandkumar. 2022. *Finite-Time Regret of Thompson Sampling Algorithms for Exponential Family Multi-Armed Bandits*.


Lattimore, T., and C. Szepesvari. 2020. *Bandit Algorithms*. Cambridge University Press.


Osborne, M. A., R. Garnett, and S. J. Roberts. 2009. "Gaussian Processes for Global Optimization." *International Conference on Learning and Intelligent Optimization*.


Srinivas, N., A. Krause, S. M. Kakade, and M. Seeger. 2010. "Gaussian Process Optimization in the Bandit Setting: No Regret and Experimental Design." *International Conference on Machine Learning*.


Yang, J., V. Y. F. Tan, and T. Jin. 2024. *Best Arm Identification with Minimal Regret*.
