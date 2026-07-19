---
title: "Exponential Families for Bandit Models"
subtitle: "The Natural Language of Modern Bandit Models"
summary: "Canonical parameters, sufficient statistics, and information geometry."
description: "Canonical parameters, sufficient statistics, and information geometry."
date: 2026-06-20
lastmod: 2026-06-20
weight: 70
tags: ["Exponential Families", "KL-UCB", "Thompson Sampling"]
draft: false
ShowToc: false
hideMeta: true
---

## Three Reward Models, One Repeated Calculation

A bandit can return very different kinds of rewards.

A recommendation is clicked or ignored. A sensor reports a real number with measurement noise. A server receives a count of requests during the next second. The first observation is binary, the second is continuous, and the third is a nonnegative integer.

At first these look like three unrelated statistical problems. Yet after a few observations, the learner keeps asking the same questions:

- What single number summarizes what this arm has shown so far?

- Which parameter value makes those observations most plausible?

- How quickly can a wrong parameter value be ruled out?

- How should a prior belief be updated after one more reward?

For Bernoulli, Gaussian, and Poisson rewards, the algebra behind all four questions has the same shape. That shared shape is the exponential family.

The name can make the subject sound more exotic than it is. The central idea is modest:


*Put the unknown parameter next to a summary of the data, and place everything needed for normalization in one convex function.*


We will first see the pattern inside familiar distributions. Only then will we give it a name.

### Binary rewards: the data become a count

Let $X\in\{0,1\}$ and let $p$ be the probability of a success. Then $\Pbb_p(X=x)=p^x(1-p)^{1-x}.$ For observations $x_1,\ldots,x_n$,

<div class="display-equation">
$$
\begin{align*}
L(p;x_1,\ldots,x_n)
&=\prod_{i=1}^{n}p^{x_i}(1-p)^{1-x_i}\\
&=p^{\sum_{i=1}^{n}x_i}(1-p)^{n-\sum_{i=1}^{n}x_i}.
\end{align*}
$$
</div>

 Define $S_n=\sum_{i=1}^{n}x_i.$ Then $L(p;x_1,\ldots,x_n)=p^{S_n}(1-p)^{n-S_n}.$ The order of clicks and non-clicks has disappeared. The likelihood remembers only how many clicks occurred.

### Gaussian rewards: the data become a sum

Suppose $X\sim\Normal(\mu,\sigma^2)$ and the variance $\sigma^2$ is known. Its density is

<div class="display-equation">
$$
p_\mu(x)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left\{-\frac{(x-\mu)^2}{2\sigma^2}\right\}.
$$
</div>

 For $n$ independent observations,

<div class="display-equation">
$$
\begin{align*}
\log L(\mu)
&=
-\frac{n}{2}\log(2\pi\sigma^2)
-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2\\
&=
-\frac{n}{2}\log(2\pi\sigma^2)
-\frac{1}{2\sigma^2}\sum_{i=1}^{n}x_i^2
+\frac{\mu}{\sigma^2}\sum_{i=1}^{n}x_i
-\frac{n\mu^2}{2\sigma^2}.
\end{align*}
$$
</div>

 As a function of $\mu$, the sample enters through $S_n=\sum_{i=1}^{n}x_i.$ Again, the full history can be compressed to a running total.

### Count rewards: the data become a sum again

Suppose $X\sim\Poi(\lambda)$. Then

<div class="display-equation">
$$
\Pbb_\lambda(X=x)
=
\frac{e^{-\lambda}\lambda^x}{x!},
\qquad x=0,1,2,\ldots.
$$
</div>

 For $n$ independent observations,

<div class="display-equation">
$$
\begin{align*}
L(\lambda)
&=\prod_{i=1}^{n}\frac{e^{-\lambda}\lambda^{x_i}}{x_i!}\\
&=\frac{1}{\prod_{i=1}^{n}x_i!}
\exp\left\{-n\lambda+\left(\sum_{i=1}^{n}x_i\right)\log\lambda\right\}.
\end{align*}
$$
</div>

 The parameter-dependent part again uses only $S_n=\sum_{i=1}^{n}x_i.$

> **Key idea.**
>
> The exponential-family idea is already visible. A potentially long dataset is replaced by a short running summary. The model then compares that summary with what each parameter value predicts.

> **Diagram.** Different observations can feed the same statistical workflow.

## The Template Appears

A one-parameter exponential family has probability mass or density

<div class="numbered-equation" id="eq:canonical-family">
$$
\boxed{
 p_\eta(x)
 =
 h(x)\exp\{\eta T(x)-A(\eta)\}.
}
$$
<span class="equation-number" aria-label="Equation 1">(1)</span>
</div>


The notation is easiest to understand one piece at a time.

- $x$ is the observation.

- $T(x)$ is the part of the observation that talks directly to the parameter. It is called the sufficient statistic for one observation.

- $\eta$ is the natural parameter.

- $h(x)$ contains everything that depends on $x$ but not on $\eta$.

- $A(\eta)$ is the log-partition function. It makes the total probability equal to one.

The word "natural" does not mean that $\eta$ is always the parameter a practitioner would report. A practitioner reports a click probability $p$, a mean $\mu$, or a rate $\lambda$. The natural parameter is the coordinate that makes the log-likelihood linear in the data summary.

Taking logs in [Eq. (1)](#eq:canonical-family),

<div class="display-equation">
$$
\log p_\eta(x)
=
\log h(x)+\eta T(x)-A(\eta).
$$
</div>

 The unknown parameter and the observed statistic meet through the simple product $\eta T(x).$ Everything else has a separate job.

> **Think.**
>
> Why not place an arbitrary constant in $A(\eta)$? Because changing $A$ changes the total mass of the density. Once $h$, $T$, and the reference measure are fixed, normalization determines $A$.

## The Log-Partition Function Is the Engine

Let

<div class="display-equation">
$$
Z(\eta)
=
\int h(x)e^{\eta T(x)}\,\dd\nu(x),
$$
</div>

 where the integral means a sum for a discrete model and an ordinary integral for a continuous model. To make [Eq. (1)](#eq:canonical-family) integrate to one, we need

<div class="display-equation">
$$
\begin{align*}
1
&=\int p_\eta(x)\,\dd\nu(x)\\
&=\int h(x)e^{\eta T(x)-A(\eta)}\,\dd\nu(x)\\
&=e^{-A(\eta)}\int h(x)e^{\eta T(x)}\,\dd\nu(x)\\
&=e^{-A(\eta)}Z(\eta).
\end{align*}
$$
</div>

 Therefore $e^{A(\eta)}=Z(\eta),$ and hence

<div class="numbered-equation" id="eq:log-partition">
$$
\boxed{
A(\eta)
=
\log\int h(x)e^{\eta T(x)}\,\dd\nu(x).
}
$$
<span class="equation-number" aria-label="Equation 2">(2)</span>
</div>


At first, $A$ looks like a normalization term that we must carry around. In fact, it stores almost everything we want to know.

### First derivative: the mean

Write $Z(\eta)=e^{A(\eta)}.$ Differentiate $A(\eta)=\log Z(\eta)$:

<div class="display-equation">
$$
\begin{align*}
A'(\eta)
&=\frac{Z'(\eta)}{Z(\eta)}\\
&=\frac{\int T(x)h(x)e^{\eta T(x)}\,\dd\nu(x)}
        {\int h(x)e^{\eta T(x)}\,\dd\nu(x)}\\
&=\int T(x)h(x)e^{\eta T(x)-A(\eta)}\,\dd\nu(x)\\
&=\int T(x)p_\eta(x)\,\dd\nu(x)\\
&=\E_\eta[T(X)].
\end{align*}
$$
</div>

 Thus

<div class="numbered-equation" id="eq:first-derivative">
$$
\boxed{A'(\eta)=\E_\eta[T(X)].}
$$
<span class="equation-number" aria-label="Equation 3">(3)</span>
</div>


The derivative of the normalizer is the mean of the statistic.

### Second derivative: the variance

Differentiate once more:

<div class="display-equation">
$$
\begin{align*}
A''(\eta)
&=\frac{Z''(\eta)Z(\eta)-[Z'(\eta)]^2}{[Z(\eta)]^2}\\
&=\frac{Z''(\eta)}{Z(\eta)}-
\left(\frac{Z'(\eta)}{Z(\eta)}\right)^2\\
&=\E_\eta[T(X)^2]-\E_\eta[T(X)]^2\\
&=\Var_\eta(T(X)).
\end{align*}
$$
</div>

 Therefore

<div class="numbered-equation" id="eq:second-derivative">
$$
\boxed{A''(\eta)=\Var_\eta(T(X))\ge 0.}
$$
<span class="equation-number" aria-label="Equation 4">(4)</span>
</div>


So $A$ is convex. This convexity is not a decorative theorem. It gives uniqueness of the mean map, concavity of the likelihood, the local form of KL divergence, and the curvature used in concentration bounds.

> **Result.**
>
> For a regular one-parameter exponential family,

<div class="display-equation">
$$
A(\eta)\quad\longrightarrow\quad
A'(\eta)=\text{mean statistic}
\quad\longrightarrow\quad
A''(\eta)=\text{variance of the statistic}.
$$
</div>

 One function stores normalization, location, and uncertainty.

### The vector version

If the statistic and natural parameter are vectors,

<div class="display-equation">
$$
p_{\bm\eta}(x)
=
h(x)\exp\{\bm\eta^\top\bm T(x)-A(\bm\eta)\},
$$
</div>

 then the same calculation gives $\nabla A(\bm\eta)=\E_{\bm\eta}[\bm T(X)]$ and $\nabla^2A(\bm\eta)=\Cov_{\bm\eta}(\bm T(X)).$ The Hessian is positive semidefinite because every covariance matrix is positive semidefinite.

The one-dimensional case is enough for the present bandit chapter, but the vector identity is one reason exponential families sit at the center of graphical models and variational inference (Wainwright and Jordan 2008; Ghahramani 2015).

![The same derivative identities hold for three familiar reward families.](/images/notes/assets/exponential-families/log_partition_map.webp)

*The same derivative identities hold for three familiar reward families.*

## Three Canonical Examples, Derived Slowly

### Bernoulli rewards

Start from

<div class="display-equation">
$$
p_p(x)=p^x(1-p)^{1-x},
\qquad x\in\{0,1\}.
$$
</div>

 Take logs inside the exponential:

<div class="display-equation">
$$
\begin{align*}
p_p(x)
&=\exp\{x\log p+(1-x)\log(1-p)\}\\
&=\exp\{x\log p+\log(1-p)-x\log(1-p)\}\\
&=\exp\left\{x\log\frac{p}{1-p}+\log(1-p)\right\}.
\end{align*}
$$
</div>

 Define the natural parameter $\eta=\log\frac{p}{1-p}.$ Solving for $p$,

<div class="display-equation">
$$
\begin{align*}
e^\eta&=\frac{p}{1-p},\\
e^\eta(1-p)&=p,\\
e^\eta&=p(1+e^\eta),\\
p&=\frac{e^\eta}{1+e^\eta}.
\end{align*}
$$
</div>

 Also, $1-p=\frac{1}{1+e^\eta},$ so $\log(1-p)=-\log(1+e^\eta).$ Therefore

<div class="display-equation">
$$
p_\eta(x)
=
\exp\{\eta x-\log(1+e^\eta)\}.
$$
</div>

 We can now read off

<div class="display-equation">
$$
T(x)=x,
\qquad
h(x)=1,
\qquad
A(\eta)=\log(1+e^\eta).
$$
</div>

 Differentiate:

<div class="display-equation">
$$
\begin{align*}
A'(\eta)
&=\frac{e^\eta}{1+e^\eta}\\
&=p,
\end{align*}
$$
</div>

 and

<div class="display-equation">
$$
\begin{align*}
A''(\eta)
&=\frac{e^\eta(1+e^\eta)-e^{2\eta}}{(1+e^\eta)^2}\\
&=\frac{e^\eta}{(1+e^\eta)^2}\\
&=p(1-p).
\end{align*}
$$
</div>

 The derivative identities reproduce the Bernoulli mean and variance.

### Gaussian rewards with known variance

Let

<div class="display-equation">
$$
p_\mu(x)
=
\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left\{-\frac{(x-\mu)^2}{2\sigma^2}\right\}.
$$
</div>

 Expand the square:

<div class="display-equation">
$$
\begin{align*}
-\frac{(x-\mu)^2}{2\sigma^2}
&=-\frac{x^2-2\mu x+\mu^2}{2\sigma^2}\\
&=-\frac{x^2}{2\sigma^2}
+\frac{\mu x}{\sigma^2}
-\frac{\mu^2}{2\sigma^2}.
\end{align*}
$$
</div>

 Hence

<div class="display-equation">
$$
p_\mu(x)
=
\underbrace{\frac{1}{\sqrt{2\pi\sigma^2}}
\exp\left\{-\frac{x^2}{2\sigma^2}\right\}}_{h(x)}
\exp\left\{\frac{\mu}{\sigma^2}x-\frac{\mu^2}{2\sigma^2}\right\}.
$$
</div>

 Set $\eta=\frac{\mu}{\sigma^2}.$ Then $\mu=\sigma^2\eta$ and

<div class="display-equation">
$$
\frac{\mu^2}{2\sigma^2}
=
\frac{\sigma^2\eta^2}{2}.
$$
</div>

 Thus

<div class="display-equation">
$$
p_\eta(x)
=
h(x)\exp\left\{\eta x-\frac{\sigma^2\eta^2}{2}\right\},
$$
</div>

 so

<div class="display-equation">
$$
T(x)=x,
\qquad
A(\eta)=\frac{\sigma^2\eta^2}{2}.
$$
</div>

 Differentiate: $A'(\eta)=\sigma^2\eta=\mu$ and $A''(\eta)=\sigma^2.$ Again the mean and variance are recovered immediately.

### Poisson rewards

Start from $p_\lambda(x)=\frac{e^{-\lambda}\lambda^x}{x!}.$ Write it as one exponential:

<div class="display-equation">
$$
\begin{align*}
p_\lambda(x)
&=\frac{1}{x!}\exp\{-\lambda+x\log\lambda\}.
\end{align*}
$$
</div>

 Set

<div class="display-equation">
$$
\eta=\log\lambda,
\qquad
\lambda=e^\eta.
$$
</div>

 Then

<div class="display-equation">
$$
p_\eta(x)
=
\frac{1}{x!}\exp\{\eta x-e^\eta\}.
$$
</div>

 Therefore

<div class="display-equation">
$$
T(x)=x,
\qquad
h(x)=\frac{1}{x!},
\qquad
A(\eta)=e^\eta.
$$
</div>

 Differentiate: $A'(\eta)=e^\eta=\lambda$ and $A''(\eta)=e^\eta=\lambda.$ The Poisson mean and variance are both equal to the rate.

### A compact dictionary


L0.17L0.17L0.10L0.19L0.18 Family & Natural parameter & $T(x)$ & $A(\eta)$ & Mean $A'(\eta)$\
Bernoulli & $\eta=\log[p/(1-p)]$ & $x$ & $\log(1+e^\eta)$ & $e^\eta/(1+e^\eta)$\
Gaussian, known $\sigma^2$ & $\eta=\mu/\sigma^2$ & $x$ & $\sigma^2\eta^2/2$ & $\sigma^2\eta$\
Poisson & $\eta=\log\lambda$ & $x$ & $e^\eta$ & $e^\eta$\
Exponential & $\eta=-\lambda<0$ & $x$ & $-\log(-\eta)$ & $-1/\eta$\
Gamma, fixed shape $\alpha$ & $\eta=-\beta<0$ & $x$ & $-\alpha\log(-\eta)$ & $-\alpha/\eta$\


The family is broad enough to cover many reward models, but not every parameterization of every distribution is a one-parameter exponential family. The support must not change with the parameter, and the parameter must enter in the canonical linear form.

## A Dataset Collapses to a Running Statistic

Let $X_1,\ldots,X_n$ be independent observations from [Eq. (1)](#eq:canonical-family). Their joint density is

<div class="display-equation">
$$
\begin{align*}
p_\eta(x_1,\ldots,x_n)
&=\prod_{i=1}^{n}h(x_i)e^{\eta T(x_i)-A(\eta)}\\
&=\left(\prod_{i=1}^{n}h(x_i)\right)
\exp\left\{\eta\sum_{i=1}^{n}T(x_i)-nA(\eta)\right\}.
\end{align*}
$$
</div>

 Define $S_n=\sum_{i=1}^{n}T(X_i).$ Then

<div class="numbered-equation" id="eq:joint-family">
$$
\boxed{
p_\eta(x_1,\ldots,x_n)
=H(x_1,\ldots,x_n)
\exp\{\eta S_n-nA(\eta)\}.
}
$$
<span class="equation-number" aria-label="Equation 5">(5)</span>
</div>


Once $n$ and $S_n$ are known, the rest of the sample no longer changes the likelihood ratio between two parameter values. This is the operational meaning of sufficiency here.

![Two Bernoulli sequences with the same number of successes produce exactly the same likelihood curve.](/images/notes/assets/exponential-families/sufficient_statistic_likelihood.webp)

*Two Bernoulli sequences with the same number of successes produce exactly the same likelihood curve.*

### Maximum likelihood becomes moment matching

Ignoring terms that do not depend on $\eta$, the log-likelihood is $\ell_n(\eta)=\eta S_n-nA(\eta).$ Differentiate: $\ell_n'(\eta)=S_n-nA'(\eta).$ At an interior maximum,

<div class="display-equation">
$$
\begin{align*}
0&=S_n-nA'(\widehat\eta_n),\\
A'(\widehat\eta_n)&=\frac{S_n}{n}.
\end{align*}
$$
</div>

 But $A'(\eta)=\E_\eta[T(X)]$. Therefore

<div class="numbered-equation" id="eq:moment-matching">
$$
\boxed{
\E_{\widehat\eta_n}[T(X)]
=
\frac{1}{n}\sum_{i=1}^{n}T(X_i).
}
$$
<span class="equation-number" aria-label="Equation 6">(6)</span>
</div>


The fitted model chooses the parameter whose expected statistic matches the observed average statistic.

The second derivative is

<div class="display-equation">
$$
\begin{align*}
\ell_n''(\eta)
&=-nA''(\eta)\\
&=-n\Var_\eta(T(X))\\
&\le 0.
\end{align*}
$$
</div>

 So the log-likelihood is concave. In a regular nondegenerate family, the stationary point is the unique maximum.

### The familiar estimators fall out immediately

For Bernoulli rewards,

<div class="display-equation">
$$
A'(\eta)=p,
\qquad
T(X)=X,
$$
</div>

 so $\widehat p_n=\frac{1}{n}\sum_{i=1}^{n}X_i.$

For Gaussian rewards with known variance,

<div class="display-equation">
$$
A'(\eta)=\mu,
\qquad
T(X)=X,
$$
</div>

 so $\widehat\mu_n=\frac{1}{n}\sum_{i=1}^{n}X_i.$

For Poisson rewards,

<div class="display-equation">
$$
A'(\eta)=\lambda,
\qquad
T(X)=X,
$$
</div>

 so $\widehat\lambda_n=\frac{1}{n}\sum_{i=1}^{n}X_i.$

These estimators look identical because the three models use the same statistic $T(X)=X$. Their uncertainty is not identical; that difference is stored in $A''$.

> **Research connection.**
>
> The reusable state of a one-parameter exponential-family arm is often just two numbers:

<div class="display-equation">
$$
N_a(t)
\quad\text{and}\quad
S_a(t)=\sum_{s\le t:A_s=a}T(X_s).
$$
</div>

 This is more than a coding convenience. It is the exact likelihood state needed by maximum likelihood, KL-UCB, and conjugate Bayesian updates.

## Conjugate Priors Are Closed Bookkeeping Rules

A Bayesian learner begins with a prior and multiplies it by the likelihood. A conjugate prior is chosen so that this multiplication changes only a small set of hyperparameters.

For the canonical family, consider a prior on $\eta$ of the form

<div class="numbered-equation" id="eq:generic-conjugate">
$$
\pi(\eta\mid\xi,\nu)
\propto
\exp\{\xi\eta-\nu A(\eta)\}\ind\{\eta\in\mathcal H\}.
$$
<span class="equation-number" aria-label="Equation 7">(7)</span>
</div>

 Here $\mathcal H$ is the natural-parameter space. The constants $\xi$ and $\nu$ must be chosen so that the prior is integrable.

After observing $x_1,\ldots,x_n$, Bayes' rule gives

<div class="display-equation">
$$
\begin{align*}
\pi(\eta\mid x_1,\ldots,x_n)
&\propto
\pi(\eta)\prod_{i=1}^{n}p_\eta(x_i)\\
&\propto
\exp\{\xi\eta-\nu A(\eta)\}
\exp\{\eta S_n-nA(\eta)\}\\
&=\exp\{(\xi+S_n)\eta-(\nu+n)A(\eta)\}.
\end{align*}
$$
</div>

 Thus

<div class="numbered-equation" id="eq:generic-update">
$$
\boxed{
\xi\leftarrow\xi+S_n,
\qquad
\nu\leftarrow\nu+n.
}
$$
<span class="equation-number" aria-label="Equation 8">(8)</span>
</div>


The posterior has the same form as the prior. The old pseudo-total and new total are added; the old pseudo-count and new count are added.

### Beta-Bernoulli, including the change of coordinates

For Bernoulli rewards,

<div class="display-equation">
$$
\eta=\log\frac{p}{1-p},
\qquad
A(\eta)=\log(1+e^\eta).
$$
</div>

 The canonical prior in $\eta$ is $\pi_\eta(\eta)\propto e^{\xi\eta-\nu A(\eta)}.$ Since

<div class="display-equation">
$$
p=\frac{e^\eta}{1+e^\eta},
\qquad
\frac{\dd\eta}{\dd p}=\frac{1}{p(1-p)},
$$
</div>

 the induced density in $p$ is

<div class="display-equation">
$$
\begin{align*}
\pi_p(p)
&=\pi_\eta(\eta(p))\left|\frac{\dd\eta}{\dd p}\right|\\
&\propto
\exp\left\{\xi\log\frac{p}{1-p}-\nu\log\frac{1}{1-p}\right\}
\frac{1}{p(1-p)}\\
&=p^{\xi-1}(1-p)^{\nu-\xi-1}.
\end{align*}
$$
</div>

 This is a Beta distribution with

<div class="display-equation">
$$
\alpha=\xi,
\qquad
\beta=\nu-\xi.
$$
</div>

 If $S_n$ successes are observed in $n$ trials, then

<div class="display-equation">
$$
\alpha\leftarrow\alpha+S_n,
\qquad
\beta\leftarrow\beta+n-S_n.
$$
</div>


### Normal-Normal with known observation variance

For $X\sim\Normal(\mu,\sigma^2)$,

<div class="display-equation">
$$
\eta=\frac{\mu}{\sigma^2},
\qquad
A(\eta)=\frac{\sigma^2\eta^2}{2}.
$$
</div>

 The generic prior is

<div class="display-equation">
$$
\begin{align*}
\pi(\eta)
&\propto
\exp\left\{\xi\eta-\frac{\nu\sigma^2\eta^2}{2}\right\}\\
&\propto
\exp\left\{-\frac{\nu\sigma^2}{2}
\left(\eta-\frac{\xi}{\nu\sigma^2}\right)^2\right\}.
\end{align*}
$$
</div>

 Hence

<div class="display-equation">
$$
\eta\sim\Normal\left(\frac{\xi}{\nu\sigma^2},\frac{1}{\nu\sigma^2}\right).
$$
</div>

 Because $\mu=\sigma^2\eta$, $\mu\sim\Normal\left(\frac{\xi}{\nu},\frac{\sigma^2}{\nu}\right).$ After $n$ observations,

<div class="display-equation">
$$
\mu\mid X_{1:n}
\sim
\Normal\left(
\frac{\xi+\sum_iX_i}{\nu+n},
\frac{\sigma^2}{\nu+n}
\right).
$$
</div>

 The posterior mean is a weighted average:

<div class="display-equation">
$$
\begin{align*}
\frac{\xi+\sum_iX_i}{\nu+n}
&=
\frac{\nu}{\nu+n}\frac{\xi}{\nu}
+
\frac{n}{\nu+n}\frac{1}{n}\sum_iX_i.
\end{align*}
$$
</div>


### Gamma-Poisson

For Poisson rewards,

<div class="display-equation">
$$
\eta=\log\lambda,
\qquad
A(\eta)=e^\eta.
$$
</div>

 The prior in $\eta$ is $\pi_\eta(\eta)\propto e^{\xi\eta-\nu e^\eta}.$ Since $\lambda=e^\eta$ and $\dd\eta/\dd\lambda=1/\lambda$,

<div class="display-equation">
$$
\begin{align*}
\pi_\lambda(\lambda)
&\propto
\lambda^\xi e^{-\nu\lambda}\frac{1}{\lambda}\\
&=\lambda^{\xi-1}e^{-\nu\lambda}.
\end{align*}
$$
</div>

 This is a Gamma distribution with shape $\xi$ and rate $\nu$. After observing counts $X_1,\ldots,X_n$,

<div class="display-equation">
$$
\xi\leftarrow\xi+\sum_{i=1}^{n}X_i,
\qquad
\nu\leftarrow\nu+n.
$$
</div>


![The posterior keeps the same shape because sufficient statistics add. Dashed lines mark empirical means.](/images/notes/assets/exponential-families/conjugate_updates.webp)

*The posterior keeps the same shape because sufficient statistics add. Dashed lines mark empirical means.*

> **Key idea.**
>
> Conjugacy is not a mysterious Bayesian coincidence. The likelihood contributes $\eta S_n-nA(\eta)$, so a prior built from the same two terms can absorb new data by addition.

## KL Divergence Becomes Convex Geometry

The previous chapter treated KL divergence as average log-evidence. Inside an exponential family, it has a closed form.

Take two members of the same family, $P_\eta$ and $P_\zeta$. The log-likelihood ratio is

<div class="display-equation">
$$
\begin{align*}
\log\frac{p_\eta(x)}{p_\zeta(x)}
&=\log\frac{h(x)e^{\eta T(x)-A(\eta)}}
                 {h(x)e^{\zeta T(x)-A(\zeta)}}\\
&=(\eta-\zeta)T(x)-A(\eta)+A(\zeta).
\end{align*}
$$
</div>

 Take expectation under $P_\eta$:

<div class="display-equation">
$$
\begin{align*}
\KL(P_\eta\Vert P_\zeta)
&=(\eta-\zeta)\E_\eta[T(X)]-A(\eta)+A(\zeta)\\
&=(\eta-\zeta)A'(\eta)-A(\eta)+A(\zeta)\\
&=A(\zeta)-A(\eta)-A'(\eta)(\zeta-\eta).
\end{align*}
$$
</div>

 Therefore

<div class="numbered-equation" id="eq:kl-bregman-natural">
$$
\boxed{
\KL(P_\eta\Vert P_\zeta)
=
A(\zeta)-A(\eta)-A'(\eta)(\zeta-\eta).
}
$$
<span class="equation-number" aria-label="Equation 9">(9)</span>
</div>


The right side is the gap between the convex function $A(\zeta)$ and the tangent line to $A$ at $\eta$. Convexity makes the gap nonnegative.


### The local quadratic and Fisher information

Let $\zeta=\eta+h$. Taylor expand $A(\eta+h)$:

<div class="display-equation">
$$
A(\eta+h)
=
A(\eta)+A'(\eta)h+\frac{1}{2}A''(\eta)h^2+O(h^3).
$$
</div>

 Insert this into [Eq. (9)](#eq:kl-bregman-natural):

<div class="display-equation">
$$
\begin{align*}
\KL(P_\eta\Vert P_{\eta+h})
&=A(\eta+h)-A(\eta)-A'(\eta)h\\
&=\frac{1}{2}A''(\eta)h^2+O(h^3).
\end{align*}
$$
</div>

 The score is

<div class="display-equation">
$$
\begin{align*}
\frac{\partial}{\partial\eta}\log p_\eta(X)
&=T(X)-A'(\eta).
\end{align*}
$$
</div>

 Therefore the Fisher information is

<div class="display-equation">
$$
\begin{align*}
I(\eta)
&=\E_\eta\left[
\left(\frac{\partial}{\partial\eta}\log p_\eta(X)\right)^2
\right]\\
&=\E_\eta[(T(X)-A'(\eta))^2]\\
&=\Var_\eta(T(X))\\
&=A''(\eta).
\end{align*}
$$
</div>

 Thus

<div class="display-equation">
$$
\boxed{
\KL(P_\eta\Vert P_{\eta+h})
=
\frac{1}{2}I(\eta)h^2+O(h^3).
}
$$
</div>


### The mean coordinate and the convex dual

Define the mean parameter $\mu=A'(\eta).$ The convex conjugate of $A$ is $A^*(\mu)=\sup_\eta\{\eta\mu-A(\eta)\}.$ At the matching pair $\mu=A'(\eta)$, $A^*(\mu)=\eta\mu-A(\eta).$ Let $\mu_\eta=A'(\eta)$ and $\mu_\zeta=A'(\zeta)$. The Bregman divergence generated by $A^*$ is

<div class="display-equation">
$$
\begin{align*}
D_{A^*}(\mu_\eta,\mu_\zeta)
&=A^*(\mu_\eta)-A^*(\mu_\zeta)
-\zeta(\mu_\eta-\mu_\zeta)\\
&=[\eta\mu_\eta-A(\eta)]-[\zeta\mu_\zeta-A(\zeta)]
-\zeta\mu_\eta+\zeta\mu_\zeta\\
&=(\eta-\zeta)\mu_\eta-A(\eta)+A(\zeta)\\
&=\KL(P_\eta\Vert P_\zeta).
\end{align*}
$$
</div>

 So KL is a Bregman divergence in either coordinate system, with the order changing through convex duality. This link underlies a broad connection between exponential families and Bregman geometry (Banerjee et al. 2005; Wainwright and Jordan 2008).

## Concentration Comes from the Same Function

The moment-generating function of $T(X)$ is almost free once $A$ is known. For any $\lambda$ such that $\eta+\lambda$ remains in the natural-parameter space,

<div class="display-equation">
$$
\begin{align*}
\E_\eta[e^{\lambda T(X)}]
&=\int e^{\lambda T(x)}h(x)e^{\eta T(x)-A(\eta)}\,\dd\nu(x)\\
&=e^{-A(\eta)}\int h(x)e^{(\eta+\lambda)T(x)}\,\dd\nu(x)\\
&=e^{-A(\eta)}e^{A(\eta+\lambda)}\\
&=\exp\{A(\eta+\lambda)-A(\eta)\}.
\end{align*}
$$
</div>

 Hence, with $\mu=A'(\eta)$,

<div class="numbered-equation" id="eq:centered-mgf">
$$
\boxed{
\E_\eta[e^{\lambda(T(X)-\mu)}]
=
\exp\{A(\eta+\lambda)-A(\eta)-\lambda\mu\}.
}
$$
<span class="equation-number" aria-label="Equation 10">(10)</span>
</div>


This identity turns the log-partition function into a concentration engine.

### Chernoff's method in the family

Let $T_1,\ldots,T_n$ be independent copies of $T(X)$, and let $\overline T_n=\frac{1}{n}\sum_{i=1}^{n}T_i.$ For $x>\mu$ and $\lambda>0$,

<div class="display-equation">
$$
\begin{align*}
\Pbb_\eta(\overline T_n\ge x)
&=\Pbb_\eta\left(e^{\lambda\sum_iT_i}\ge e^{\lambda nx}\right)\\
&\le e^{-\lambda nx}\E_\eta\left[e^{\lambda\sum_iT_i}\right]\\
&=e^{-\lambda nx}\prod_{i=1}^{n}\E_\eta[e^{\lambda T_i}]\\
&=\exp\{-n[\lambda x-A(\eta+\lambda)+A(\eta)]\}.
\end{align*}
$$
</div>

 We choose the best $\lambda$:

<div class="display-equation">
$$
\Pbb_\eta(\overline T_n\ge x)
\le
\exp\left\{-n\sup_{\lambda>0}
[\lambda x-A(\eta+\lambda)+A(\eta)]\right\}.
$$
</div>

 Let $\eta_x$ satisfy $A'(\eta_x)=x.$ Differentiate the expression inside the supremum:

<div class="display-equation">
$$
\begin{align*}
\frac{\dd}{\dd\lambda}
[\lambda x-A(\eta+\lambda)+A(\eta)]
&=x-A'(\eta+\lambda).
\end{align*}
$$
</div>

 The optimum is reached at

<div class="display-equation">
$$
\eta+\lambda^*=\eta_x,
\qquad
\lambda^*=\eta_x-\eta.
$$
</div>

 Substitute:

<div class="display-equation">
$$
\begin{align*}
\lambda^*x-A(\eta+\lambda^*)+A(\eta)
&=(\eta_x-\eta)x-A(\eta_x)+A(\eta)\\
&=(\eta_x-\eta)A'(\eta_x)-A(\eta_x)+A(\eta)\\
&=\KL(P_{\eta_x}\Vert P_\eta).
\end{align*}
$$
</div>

 Therefore

<div class="numbered-equation" id="eq:family-chernoff">
$$
\boxed{
\Pbb_\eta(\overline T_n\ge x)
\le
\exp\{-n\KL(P_{\eta_x}\Vert P_\eta)\}.
}
$$
<span class="equation-number" aria-label="Equation 11">(11)</span>
</div>


The same KL divergence that measures evidence also controls rare sample means.

> **Proof pattern.**
>
> A recurring proof pattern in bandit theory is

<div class="display-equation">
$$
\text{log-partition}
\longrightarrow
\text{moment-generating function}
\longrightarrow
\text{convex optimization}
\longrightarrow
\text{KL exponent}.
$$
</div>

 Once the family is identified, many concentration calculations become instances of this one pattern.

![The rate functions differ in shape, but all arise by the same exponential-family calculation.](/images/notes/assets/exponential-families/rate_functions.webp)

*The rate functions differ in shape, but all arise by the same exponential-family calculation.*

## An Exponential-Family Bandit Keeps One Ledger per Arm

Consider $K$ arms. Arm $a$ has parameter $\eta_a$ and mean $\mu_a=A'(\eta_a).$ At round $t$, the learner chooses $A_t$ using the past and observes a reward $X_t$ from the chosen arm.

The adaptive choice of arms changes which observations are collected. It does not change the form of the reward likelihood. Up to terms independent of the parameters,

<div class="display-equation">
$$
\begin{align*}
\log p_{\bm\eta}(H_T)
&=\sum_{t=1}^{T}
\left[\eta_{A_t}T(X_t)-A(\eta_{A_t})\right]+\text{constant}\\
&=\sum_{a=1}^{K}
\left[
\eta_a\sum_{t=1}^{T}\ind\{A_t=a\}T(X_t)
-
A(\eta_a)\sum_{t=1}^{T}\ind\{A_t=a\}
\right]+    ext{constant}.
\end{align*}
$$
</div>

 Define $N_a(T)=\sum_{t=1}^{T}\ind\{A_t=a\}$ and $S_a(T)=\sum_{t=1}^{T}\ind\{A_t=a\}T(X_t).$ Then

<div class="numbered-equation" id="eq:bandit-ledger">
$$
\boxed{
\log p_{\bm\eta}(H_T)
=
\sum_{a=1}^{K}\left[\eta_aS_a(T)-N_a(T)A(\eta_a)\right]
+\text{constant}.
}
$$
<span class="equation-number" aria-label="Equation 12">(12)</span>
</div>


Every arm needs a count and a sufficient-statistic total. The algorithm decides where the next line is written; the exponential family decides how the ledger is interpreted.

### The empirical mean parameter

For arm $a$, the maximum-likelihood equation is $A'(\widehat\eta_a)=\frac{S_a}{N_a}.$ Define $\widehat\mu_a=\frac{S_a}{N_a}.$ When $T(X)=X$, this is the ordinary sample mean. The meaning of its uncertainty still depends on the reward family.

### Distribution-aware optimism

Let $d(\mu,q)=\KL(P_\mu\Vert P_q)$ be the KL divergence written in mean coordinates. A generic KL upper confidence index is

<div class="numbered-equation" id="eq:generic-kl-ucb">
$$
U_a(t)
=
\sup\left\{
q:\ N_a(t)d(\widehat\mu_a(t),q)\le\beta(t)
\right\}.
$$
<span class="equation-number" aria-label="Equation 13">(13)</span>
</div>

 The policy chooses $A_t\in\argmax_a U_a(t).$

The same line of code means different confidence shapes in different families:
| Family | $d(\mu,q)=\KL(P_\mu\Vert P_q)$ |
|:---|:---|
| Bernoulli | $\mu\log(\mu/q)+(1-\mu)\log[(1-\mu)/(1-q)]$ |
| Gaussian, known $\sigma^2$ | $(\mu-q)^2/(2\sigma^2)$ |
| Poisson | $\mu\log(\mu/q)+q-\mu$ |
| Exponential, mean $\mu$ | $\log(q/\mu)+\mu/q-1$ |

<p class="table-caption">KL divergence between two mean parameters.</p>

For the Gaussian family, [Eq. (13)](#eq:generic-kl-ucb) can be solved directly:

<div class="display-equation">
$$
\begin{align*}
N_a\frac{(\widehat\mu_a-q)^2}{2\sigma^2}
&\le\beta(t),\\
(q-\widehat\mu_a)^2
&\le\frac{2\sigma^2\beta(t)}{N_a},\\
q
&\le\widehat\mu_a+\sqrt{\frac{2\sigma^2\beta(t)}{N_a}}.
\end{align*}
$$
</div>

 Thus

<div class="display-equation">
$$
U_a(t)
=
\widehat\mu_a(t)+\sqrt{\frac{2\sigma^2\beta(t)}{N_a(t)}}.
$$
</div>

 For Bernoulli and Poisson rewards, a one-dimensional binary search finds the endpoint.

### Posterior sampling uses the same ledger

With independent conjugate priors, the posterior of arm $a$ updates as

<div class="display-equation">
$$
\xi_a(t)=\xi_{a,0}+S_a(t),
\qquad
\nu_a(t)=\nu_{a,0}+N_a(t).
$$
</div>

 Thompson sampling draws one parameter from each posterior and acts as if the draw were true:

<div class="display-equation">
$$
\begin{align*}
\widetilde\eta_a(t)&\sim\pi_a(\eta_a\mid H_{t-1}),\\
A_t&\in\argmax_a A'(\widetilde\eta_a(t)).
\end{align*}
$$
</div>


For Bernoulli, this is Beta sampling. For Gaussian rewards with known variance, it is Normal sampling. For Poisson rewards, it is Gamma sampling.

Exponential-family structure has therefore supported both optimistic and posterior-sampling analyses. Korda, Kaufmann, and Munos used its closed-form KL and Fisher information to analyze Thompson sampling in one-dimensional families (Korda et al. 2013). Jin, Xu, Xiao, and Anandkumar later developed finite-time and asymptotic guarantees for exponential-family bandits through ExpTS and ExpTS$^+$ (Jin et al. 2022). The point of the abstraction is not merely to shorten notation; it lets one proof strategy travel across reward models.

> **Research connection.**
>
> A useful research question is often not "Which named distribution should I use?" but:
>
> ::: center
> What are the sufficient statistic, log-partition function, mean map, KL geometry, and conjugate update of the reward model?
> :::
>
> Once these five objects are known, much of the bandit machinery becomes visible.

## From Mathematics to an Algorithm Interface

The implementation can mirror the theory. A model only needs to provide a few operations:

- convert accumulated statistics into an empirical mean;

- compute or invert the family KL divergence;

- update and sample from a conjugate posterior;

- generate rewards for simulation.

The decision loop then stays almost unchanged.

    def kl_ucb_step(model, counts, totals, t):
        empirical_mean = totals / counts
        beta = np.log(t) + 3.0 * np.log(max(np.log(t), 1.0))
        upper_index = model.kl_upper(empirical_mean, counts, beta)
        return int(np.argmax(upper_index))

For Gaussian rewards, the model-specific inversion is closed form:

    def gaussian_kl_upper(empirical_mean, counts, beta, sigma):
        return empirical_mean + np.sqrt(
            2.0 * sigma**2 * beta / counts
        )

For Bernoulli or Poisson rewards, monotonicity makes binary search sufficient:

    def kl_upper_by_bisection(empirical_mean, counts, beta,
                              kl_divergence, upper_bracket,
                              steps=24):
        lo = empirical_mean.copy()
        hi = upper_bracket(empirical_mean, counts, beta)

        for _ in range(steps):
            mid = (lo + hi) / 2.0
            feasible = counts * kl_divergence(empirical_mean, mid) <= beta
            lo = np.where(feasible, mid, lo)
            hi = np.where(feasible, hi, mid)

        return lo

The Thompson-sampling skeleton is even shorter:

    def thompson_step(model, counts, totals, rng):
        sampled_means = model.sample_posterior_mean(
            counts=counts,
            totals=totals,
            rng=rng,
        )
        return int(np.argmax(sampled_means))

The abstraction is useful only if the model methods remain mathematically transparent. A large software hierarchy that hides the likelihood would defeat the purpose.

## Reproducible Experiment: One Skeleton, Three Families

The supplied script builds three four-armed environments:

- Bernoulli means $(0.18,0.24,0.31,0.36)$;

- Gaussian means $(0.00,0.10,0.18,0.25)$ with known $\sigma=0.35$;

- Poisson means $(1.70,2.00,2.30,2.60)$.

For each environment, it runs a family-aware KL-UCB policy and conjugate Thompson sampling for $T=5000$ rounds over $240$ independent runs. The performance measure is pseudo-regret,

<div class="display-equation">
$$
R_T
=
\sum_{t=1}^{T}(\mu^*-\mu_{A_t}).
$$
</div>


![The algorithmic skeleton is unchanged; the family determines the KL and posterior operations.](/images/notes/assets/exponential-families/exponential_family_regret.webp)

*The algorithmic skeleton is unchanged; the family determines the KL and posterior operations.*

![Both methods eventually allocate most observations to the arm with the largest mean.](/images/notes/assets/exponential-families/exponential_family_pull_counts.webp)

*Both methods eventually allocate most observations to the arm with the largest mean.*
| Family    | Algorithm         | Final pseudo-regret | Pulls of best arm |
|:----------|:------------------|--------------------:|------------------:|
| Bernoulli | KL-UCB            |    $92.98\pm1.32$ |        $3806.2$ |
| Bernoulli | Thompson sampling |    $44.66\pm1.49$ |        $4402.2$ |
| Gaussian  | KL-UCB            |    $54.74\pm0.76$ |        $4470.0$ |
| Gaussian  | Thompson sampling |    $23.74\pm0.67$ |        $4766.5$ |
| Poisson   | KL-UCB            |   $265.43\pm3.69$ |        $4388.0$ |
| Poisson   | Thompson sampling |   $101.14\pm3.01$ |        $4761.7$ |

<p class="table-caption">Simulation results at $T=5000$. Standard errors are across 240 runs.</p>

The numerical ranking is not a universal benchmark. It depends on the selected instances, priors, and exploration schedule. The experiment is designed to demonstrate a structural point: one decision loop can operate across reward types when the model supplies the right sufficient statistic, KL divergence, and posterior update.

### What the code makes visible

The Bernoulli policy stores successes and failures. The Gaussian policy stores a count and reward sum. The Poisson policy stores a count and event total. In all three cases, the posterior or confidence index is computed from those same two running quantities.

This is the computational meaning of the exponential family:

<div class="display-equation">
$$
\boxed{
\text{raw history}
\longrightarrow
(N_a,S_a)
\longrightarrow
\text{evidence or posterior}
\longrightarrow
\text{next action}.
}
$$
</div>


## What the Definition Does Not Say

### It does not say that every parameter is natural

The reported mean and the natural parameter may be very different:

<div class="display-equation">
$$
\eta=\log\frac{p}{1-p},
\qquad
\eta=\frac{\mu}{\sigma^2},
\qquad
\eta=\log\lambda.
$$
</div>

 The natural coordinate simplifies likelihood algebra. The mean coordinate simplifies decisions and regret statements. Good proofs move between them deliberately.

### It does not say that the base measure is irrelevant

The factor $h(x)$ cancels in likelihood ratios between members of the same family, but it is part of the distribution. For Poisson rewards, $1/x!$ distinguishes the model from other count models. For Gaussian rewards, the $e^{-x^2/(2\sigma^2)}$ term carries the fixed shape of the noise.

### It does not say that variance is constant

The variance is $A''(\eta).$ For Bernoulli rewards it is $p(1-p)$; for Poisson rewards it is $\lambda$; for a known-variance Gaussian it is constant. A distribution-aware algorithm uses this changing curvature instead of pretending all arms have the same noise geometry.

### It does not automatically give anytime validity

The fixed-$n$ Chernoff bound in [Eq. (11)](#eq:family-chernoff) is not automatically valid after arbitrary repeated peeking. The filtration and martingale tools from the previous chapter are still needed to construct time-uniform confidence sequences.

### It does not remove modeling risk

A clean exponential-family likelihood may still be wrong. Clicks may be delayed, counts may be overdispersed, Gaussian noise may be heavy-tailed, and reward distributions may drift. The framework clarifies the consequences of a model; it does not certify that the model matches reality.

## What to Carry Forward

An exponential family is a compact statistical machine.

The density has the form $p_\eta(x)=h(x)e^{\eta T(x)-A(\eta)}.$ The log-partition function normalizes the model: $A(\eta)=\log\int h(x)e^{\eta T(x)}\,\dd\nu(x).$ Its first two derivatives give the mean and variance:

<div class="display-equation">
$$
A'(\eta)=\E_\eta[T(X)],
\qquad
A''(\eta)=\Var_\eta(T(X)).
$$
</div>

 A sample is compressed to $S_n=\sum_{i=1}^{n}T(X_i).$ Maximum likelihood matches the empirical statistic to the model mean: $A'(\widehat\eta_n)=\frac{S_n}{n}.$ Conjugate Bayes adds counts and sufficient statistics: $(\xi,\nu)\leftarrow(\xi+S_n,\nu+n).$ KL divergence is the convexity gap of $A$:

<div class="display-equation">
$$
\KL(P_\eta\Vert P_\zeta)
=A(\zeta)-A(\eta)-A'(\eta)(\zeta-\eta).
$$
</div>

 Concentration follows from the shifted log-partition function:

<div class="display-equation">
$$
\log\E_\eta[e^{\lambda T(X)}]
=A(\eta+\lambda)-A(\eta).
$$
</div>

 And an adaptive bandit needs only one likelihood ledger per arm: $(N_a,S_a).$


This is why exponential families are a natural language for bandit theory. They do not make every model identical. They reveal which calculations are identical, which quantities remain model-specific, and where statistical evidence enters the decision rule.

## Appendix A. Formula Sheet
| Object | Formula |
|:---|:---|
| Canonical family | $p_\eta(x)=h(x)e^{\eta T(x)-A(\eta)}$ |
| Log-partition | $A(\eta)=\log\int h(x)e^{\eta T(x)}\,\dd\nu(x)$ |
| Mean statistic | $A'(\eta)=\E_\eta[T(X)]$ |
| Variance statistic | $A''(\eta)=\Var_\eta(T(X))$ |
| Vector mean | $\nabla A(\bm\eta)=\E_{\bm\eta}[\bm T(X)]$ |
| Vector covariance | $\nabla^2A(\bm\eta)=\Cov_{\bm\eta}(\bm T(X))$ |
| Joint sufficient statistic | $S_n=\sum_{i=1}^{n}T(X_i)$ |
| MLE equation | $A'(\widehat\eta_n)=S_n/n$ |
| Conjugate prior | $\pi(\eta\mid\xi,\nu)\propto e^{\xi\eta-\nu A(\eta)}$ |
| Conjugate update | $(\xi,\nu)\mapsto(\xi+S_n,\nu+n)$ |
| Family KL | $\KL(P_\eta\Vert P_\zeta)=A(\zeta)-A(\eta)-A'(\eta)(\zeta-\eta)$ |
| Fisher information | $I(\eta)=A''(\eta)$ |
| Moment-generating function | $\E_\eta[e^{\lambda T(X)}]=e^{A(\eta+\lambda)-A(\eta)}$ |
| Chernoff exponent | $\Pbb_\eta(\overline T_n\ge x)\le e^{-n\KL(P_{\eta_x}\Vert P_\eta)}$ |
| Bandit ledger | $N_a(t),\ S_a(t)=\sum_{s\le t:A_s=a}T(X_s)$ |
| KL-UCB index | $\sup\{q:N_a d(\widehat\mu_a,q)\le\beta(t)\}$ |

<p class="table-caption">Core formulas.</p>

## Appendix B. Notation Table
| Symbol           | Meaning                                                 |
|:-----------------|:--------------------------------------------------------|
| $X$            | one observation or reward                               |
| $T(X)$         | canonical statistic of one observation                  |
| $h(x)$         | base density or mass independent of the parameter       |
| $\eta$         | natural parameter                                       |
| $\mathcal H$   | natural-parameter space                                 |
| $A(\eta)$      | log-partition function                                  |
| $\mu=A'(\eta)$ | mean parameter for the statistic                        |
| $S_n$          | sum of sufficient statistics through $n$ observations |
| $\xi,\nu$      | conjugate-prior hyperparameters                         |
| $A^*$          | convex conjugate of $A$                               |
| $d(\mu,q)$     | family KL divergence in mean coordinates                |
| $A_t$          | arm selected at round $t$                             |
| $N_a(t)$       | number of pulls of arm $a$ by time $t$              |
| $S_a(t)$       | sufficient-statistic total for arm $a$                |
| $U_a(t)$       | upper confidence index                                  |

<p class="table-caption">Notation.</p>

## Appendix C. Minimal Implementation Notes

The supplied Python script reproduces all figures and tables. A few details matter in numerical work:

1.  Clip Bernoulli probabilities away from exactly $0$ and $1$ before taking logarithms, while preserving the limiting convention $0\log0=0$.

2.  Use shape-rate consistently for Gamma distributions. NumPy and SciPy accept a scale, which is the reciprocal of the rate.

3.  Invert one-sided Bernoulli or Poisson KL balls by monotone bisection. A fixed iteration count gives predictable numerical cost.

4.  Initialize every arm before dividing by $N_a(t)$.

5.  Treat the simulation table as an audit of the implementation, not a general ranking of algorithms.

## Further Reading

Barndorff-Nielsen's monograph develops the statistical theory of exponential families in depth (Barndorff-Nielsen 1978). Wainwright and Jordan give the modern convex-analytic view that connects exponential families, marginal polytopes, and variational inference (Wainwright and Jordan 2008). MacKay's Cambridge text is an unusually readable route through likelihood, Bayesian updating, and information geometry (MacKay 2003); Bishop's treatment from Microsoft Research Cambridge is a useful machine-learning companion (Bishop 2006). Lattimore and Szepesvari place these ideas directly inside bandit theory (Lattimore and Szepesvari 2020). For Thompson sampling in one-dimensional exponential families, see Korda, Kaufmann, and Munos (Korda et al. 2013) and the finite-time development of Jin and collaborators (Jin et al. 2022).

## Appendix D. Full Experiment Script

``` {.python style="blogcode" language="Python" basicstyle="\\ttfamily\\scriptsize"}
"""Reproducible experiments for
'Exponential Families: The Natural Language of Modern Bandit Models'.

The script creates:
  - log_partition_map.pdf/png
  - sufficient_statistic_likelihood.pdf/png
  - conjugate_updates.pdf/png
  - rate_functions.pdf/png
  - exponential_family_regret.pdf/png
  - exponential_family_pull_counts.pdf/png
  - exponential_families_results.csv

All simulations use fixed random seeds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, gammaln
from scipy.stats import beta as beta_dist
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm


OUT = Path(__file__).resolve().parent
SEED = 20260620


@dataclass(frozen=True)
class Config:
    horizon: int = 5000
    runs: int = 240
    binary_steps: int = 22

    bernoulli_means: tuple[float, ...] = (0.18, 0.24, 0.31, 0.36)
    gaussian_means: tuple[float, ...] = (0.00, 0.10, 0.18, 0.25)
    gaussian_sigma: float = 0.35
    poisson_means: tuple[float, ...] = (1.70, 2.00, 2.30, 2.60)


def bernoulli_kl(p: np.ndarray | float, q: np.ndarray | float) -> np.ndarray:
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    q_safe = np.clip(q_arr, 1e-14, 1.0 - 1e-14)
    p_safe = np.clip(p_arr, 1e-14, 1.0 - 1e-14)
    term1 = np.where(p_arr > 0.0, p_safe * np.log(p_safe / q_safe), 0.0)
    term2 = np.where(
        p_arr < 1.0,
        (1.0 - p_safe) * np.log((1.0 - p_safe) / (1.0 - q_safe)),
        0.0,
    )
    return term1 + term2


def poisson_kl(p: np.ndarray | float, q: np.ndarray | float) -> np.ndarray:
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    q_safe = np.clip(q_arr, 1e-14, None)
    p_safe = np.clip(p_arr, 1e-14, None)
    return np.where(p_arr > 0.0, p_safe * np.log(p_safe / q_safe), 0.0) + q_safe - p_arr


def bernoulli_kl_upper(emp: np.ndarray, counts: np.ndarray, beta: float, steps: int) -> np.ndarray:
    lo = emp.copy()
    hi = np.ones_like(emp)
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        feasible = counts * bernoulli_kl(emp, mid) <= beta
        lo = np.where(feasible, mid, lo)
        hi = np.where(feasible, hi, mid)
    return lo


def poisson_kl_upper(emp: np.ndarray, counts: np.ndarray, beta: float, steps: int) -> np.ndarray:
    radius = beta / counts
    lo = np.maximum(emp, 0.0)
    hi = np.maximum(1.0, emp + 2.0 * np.sqrt(2.0 * (emp + 1.0) * radius) + 4.0 * radius + 2.0)
    # The bracket above is deliberately generous for the means used here.
    for _ in range(steps):
        mid = (lo + hi) / 2.0
        feasible = counts * poisson_kl(emp, mid) <= beta
        lo = np.where(feasible, mid, lo)
        hi = np.where(feasible, hi, mid)
    return lo


def plot_log_partition_map() -> pd.DataFrame:
    eta_b = np.linspace(-5.0, 5.0, 500)
    A_b = np.logaddexp(0.0, eta_b)
    m_b = expit(eta_b)
    v_b = m_b * (1.0 - m_b)

    eta_g = np.linspace(-3.0, 3.0, 500)
    A_g = 0.5 * eta_g**2
    m_g = eta_g
    v_g = np.ones_like(eta_g)

    eta_p = np.linspace(-2.5, 1.7, 500)
    A_p = np.exp(eta_p)
    m_p = np.exp(eta_p)
    v_p = np.exp(eta_p)

    fig, axes = plt.subplots(3, 3, figsize=(10.0, 8.0))
    families = [
        ("Bernoulli", eta_b, A_b, m_b, v_b),
        ("Gaussian, $\\sigma^2=1$", eta_g, A_g, m_g, v_g),
        ("Poisson", eta_p, A_p, m_p, v_p),
    ]
    for row, (name, eta, A, mean, var) in enumerate(families):
        axes[row, 0].plot(eta, A, linewidth=1.9)
        axes[row, 0].set_ylabel(name)
        axes[row, 1].plot(eta, mean, linewidth=1.9)
        axes[row, 2].plot(eta, var, linewidth=1.9)
        for col in range(3):
            axes[row, col].axhline(0.0, linewidth=0.6, alpha=0.45)
            axes[row, col].grid(True, linewidth=0.3, alpha=0.25)
            if row == 2:
                axes[row, col].set_xlabel(r"natural parameter $\eta$")
    axes[0, 0].set_title(r"$A(\eta)$")
    axes[0, 1].set_title(r"$A'(\eta)=\mathbb{E}[T(X)]$")
    axes[0, 2].set_title(r"$A''(\eta)=\mathrm{Var}(T(X))$")
    fig.suptitle("One function stores normalization, mean, and variance", y=1.01)
    fig.tight_layout()
    fig.savefig(OUT / "log_partition_map.pdf", bbox_inches="tight")
    fig.savefig(OUT / "log_partition_map.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(
        [
            {"experiment": "log_partition", "quantity": "Bernoulli mean at eta=0", "value": 0.5},
            {"experiment": "log_partition", "quantity": "Bernoulli variance at eta=0", "value": 0.25},
            {"experiment": "log_partition", "quantity": "Gaussian variance for sigma2=1", "value": 1.0},
            {"experiment": "log_partition", "quantity": "Poisson mean at eta=log(2.5)", "value": 2.5},
        ]
    )


def plot_sufficient_statistic_likelihood() -> pd.DataFrame:
    n = 12
    sequences = {
        "sequence A: 7 clicks": np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1]),
        "sequence B: same 7 clicks": np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1]),
        "sequence C: 4 clicks": np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0]),
    }
    p = np.linspace(0.01, 0.99, 600)

    fig, ax = plt.subplots(figsize=(7.3, 4.55))
    rows: list[dict[str, float | str]] = []
    styles = ["-", "--", "-."]
    for (label, seq), style in zip(sequences.items(), styles):
        s = int(seq.sum())
        log_lik = s * np.log(p) + (n - s) * np.log(1.0 - p)
        log_lik -= log_lik.max()
        ax.plot(p, np.exp(log_lik), linestyle=style, linewidth=2.0, label=label)
        rows.append({"experiment": "sufficiency", "quantity": f"success count for {label}", "value": float(s)})
    ax.set_xlabel(r"candidate click probability $p$")
    ax.set_ylabel("relative likelihood")
    ax.set_title("Order disappears: the likelihood remembers only the success count")
    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "sufficient_statistic_likelihood.pdf", bbox_inches="tight")
    fig.savefig(OUT / "sufficient_statistic_likelihood.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_conjugate_updates() -> pd.DataFrame:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.55))
    rows: list[dict[str, float | str]] = []

    # Beta-Bernoulli
    p = np.linspace(0.001, 0.999, 600)
    a0, b0 = 1.0, 1.0
    successes, failures = 14, 6
    a1, b1 = a0 + successes, b0 + failures
    axes[0].plot(p, beta_dist.pdf(p, a0, b0), linewidth=1.7, label="prior")
    axes[0].plot(p, beta_dist.pdf(p, a1, b1), linewidth=2.0, label="posterior")
    axes[0].axvline(successes / (successes + failures), linewidth=0.9, linestyle="--")
    axes[0].set_title("Bernoulli mean")
    axes[0].set_xlabel(r"$p$")
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False)
    rows.append({"experiment": "conjugacy", "quantity": "Beta posterior mean", "value": a1 / (a1 + b1)})

    # Gaussian known variance
    mu = np.linspace(-0.8, 1.2, 600)
    prior_mean, prior_sd = 0.0, 0.7
    sigma = 0.5
    n, xbar = 12, 0.45
    prior_precision = 1.0 / prior_sd**2
    data_precision = n / sigma**2
    post_var = 1.0 / (prior_precision + data_precision)
    post_mean = post_var * (prior_precision * prior_mean + data_precision * xbar)
    axes[1].plot(mu, norm.pdf(mu, prior_mean, prior_sd), linewidth=1.7, label="prior")
    axes[1].plot(mu, norm.pdf(mu, post_mean, math.sqrt(post_var)), linewidth=2.0, label="posterior")
    axes[1].axvline(xbar, linewidth=0.9, linestyle="--")
    axes[1].set_title("Gaussian mean")
    axes[1].set_xlabel(r"$\mu$")
    axes[1].legend(frameon=False)
    rows.append({"experiment": "conjugacy", "quantity": "Gaussian posterior mean", "value": post_mean})

    # Gamma-Poisson (shape-rate)
    lam = np.linspace(0.001, 5.5, 600)
    shape0, rate0 = 1.0, 1.0
    n_pois, total = 12, 30
    shape1, rate1 = shape0 + total, rate0 + n_pois
    axes[2].plot(lam, gamma_dist.pdf(lam, a=shape0, scale=1.0 / rate0), linewidth=1.7, label="prior")
    axes[2].plot(lam, gamma_dist.pdf(lam, a=shape1, scale=1.0 / rate1), linewidth=2.0, label="posterior")
    axes[2].axvline(total / n_pois, linewidth=0.9, linestyle="--")
    axes[2].set_title("Poisson rate")
    axes[2].set_xlabel(r"$\lambda$")
    axes[2].legend(frameon=False)
    rows.append({"experiment": "conjugacy", "quantity": "Gamma posterior mean", "value": shape1 / rate1})

    for ax in axes:
        ax.grid(True, linewidth=0.3, alpha=0.25)
    fig.suptitle("Conjugacy is a bookkeeping rule: old pseudo-data plus new data", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "conjugate_updates.pdf", bbox_inches="tight")
    fig.savefig(OUT / "conjugate_updates.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_rate_functions() -> pd.DataFrame:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.55))
    rows: list[dict[str, float | str]] = []

    mu_b = 0.35
    x_b = np.linspace(0.005, 0.995, 600)
    rate_b = bernoulli_kl(x_b, mu_b)
    axes[0].plot(x_b, rate_b, linewidth=2.0)
    axes[0].axvline(mu_b, linewidth=0.9, linestyle="--")
    axes[0].set_title("Bernoulli")
    axes[0].set_xlabel(r"sample mean $x$")
    axes[0].set_ylabel("rate function")

    mu_g, sigma = 0.25, 0.35
    x_g = np.linspace(-0.8, 1.3, 600)
    rate_g = (x_g - mu_g) ** 2 / (2.0 * sigma**2)
    axes[1].plot(x_g, rate_g, linewidth=2.0)
    axes[1].axvline(mu_g, linewidth=0.9, linestyle="--")
    axes[1].set_title("Gaussian")
    axes[1].set_xlabel(r"sample mean $x$")

    mu_p = 2.6
    x_p = np.linspace(0.01, 6.5, 600)
    rate_p = poisson_kl(x_p, mu_p)
    axes[2].plot(x_p, rate_p, linewidth=2.0)
    axes[2].axvline(mu_p, linewidth=0.9, linestyle="--")
    axes[2].set_title("Poisson")
    axes[2].set_xlabel(r"sample mean $x$")

    for ax in axes:
        ax.set_ylim(bottom=0.0)
        ax.grid(True, linewidth=0.3, alpha=0.25)
    fig.suptitle("Different reward models, the same concentration story", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "rate_functions.pdf", bbox_inches="tight")
    fig.savefig(OUT / "rate_functions.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    rows.extend(
        [
            {"experiment": "rate_function", "quantity": "Bernoulli rate at x=0.45", "value": float(bernoulli_kl(0.45, mu_b))},
            {"experiment": "rate_function", "quantity": "Gaussian rate at x=0.45", "value": float((0.45 - mu_g) ** 2 / (2.0 * sigma**2))},
            {"experiment": "rate_function", "quantity": "Poisson rate at x=3.2", "value": float(poisson_kl(3.2, mu_p))},
        ]
    )
    return pd.DataFrame(rows)


def _beta_time(t: int) -> float:
    if t <= 2:
        return 1.0
    return math.log(t) + 3.0 * math.log(max(math.log(t), 1.0))


def simulate_bernoulli(cfg: Config, algorithm: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = np.asarray(cfg.bernoulli_means, dtype=float)
    runs, horizon, k = cfg.runs, cfg.horizon, means.size
    counts = np.zeros((runs, k), dtype=float)
    sums = np.zeros((runs, k), dtype=float)
    regrets = np.zeros((runs, horizon), dtype=float)
    best = float(means.max())
    rr = np.arange(runs)

    for a in range(k):
        reward = (rng.random(runs) < means[a]).astype(float)
        counts[:, a] += 1.0
        sums[:, a] += reward
        regrets[:, a] = (best - means[a]) + (regrets[:, a - 1] if a > 0 else 0.0)

    for t in range(k, horizon):
        emp = sums / counts
        if algorithm == "KL-UCB":
            index = bernoulli_kl_upper(emp, counts, _beta_time(t + 1), cfg.binary_steps)
        elif algorithm == "Thompson sampling":
            index = rng.beta(1.0 + sums, 1.0 + counts - sums)
        else:
            raise ValueError(f"unknown algorithm: {algorithm}")
        action = np.argmax(index, axis=1)
        reward = (rng.random(runs) < means[action]).astype(float)
        counts[rr, action] += 1.0
        sums[rr, action] += reward
        regrets[:, t] = regrets[:, t - 1] + best - means[action]
    return regrets, counts


def simulate_gaussian(cfg: Config, algorithm: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = np.asarray(cfg.gaussian_means, dtype=float)
    sigma = cfg.gaussian_sigma
    runs, horizon, k = cfg.runs, cfg.horizon, means.size
    counts = np.zeros((runs, k), dtype=float)
    sums = np.zeros((runs, k), dtype=float)
    regrets = np.zeros((runs, horizon), dtype=float)
    best = float(means.max())
    rr = np.arange(runs)

    for a in range(k):
        reward = rng.normal(means[a], sigma, size=runs)
        counts[:, a] += 1.0
        sums[:, a] += reward
        regrets[:, a] = (best - means[a]) + (regrets[:, a - 1] if a > 0 else 0.0)

    prior_mean, prior_var = 0.0, 1.0
    for t in range(k, horizon):
        emp = sums / counts
        if algorithm == "KL-UCB":
            index = emp + np.sqrt(2.0 * sigma**2 * _beta_time(t + 1) / counts)
        elif algorithm == "Thompson sampling":
            precision = 1.0 / prior_var + counts / sigma**2
            post_var = 1.0 / precision
            post_mean = post_var * (prior_mean / prior_var + sums / sigma**2)
            index = rng.normal(post_mean, np.sqrt(post_var))
        else:
            raise ValueError(f"unknown algorithm: {algorithm}")
        action = np.argmax(index, axis=1)
        reward = rng.normal(means[action], sigma)
        counts[rr, action] += 1.0
        sums[rr, action] += reward
        regrets[:, t] = regrets[:, t - 1] + best - means[action]
    return regrets, counts


def simulate_poisson(cfg: Config, algorithm: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    means = np.asarray(cfg.poisson_means, dtype=float)
    runs, horizon, k = cfg.runs, cfg.horizon, means.size
    counts = np.zeros((runs, k), dtype=float)
    sums = np.zeros((runs, k), dtype=float)
    regrets = np.zeros((runs, horizon), dtype=float)
    best = float(means.max())
    rr = np.arange(runs)

    for a in range(k):
        reward = rng.poisson(means[a], size=runs).astype(float)
        counts[:, a] += 1.0
        sums[:, a] += reward
        regrets[:, a] = (best - means[a]) + (regrets[:, a - 1] if a > 0 else 0.0)

    for t in range(k, horizon):
        emp = sums / counts
        if algorithm == "KL-UCB":
            index = poisson_kl_upper(emp, counts, _beta_time(t + 1), cfg.binary_steps)
        elif algorithm == "Thompson sampling":
            index = rng.gamma(shape=1.0 + sums, scale=1.0 / (1.0 + counts))
        else:
            raise ValueError(f"unknown algorithm: {algorithm}")
        action = np.argmax(index, axis=1)
        reward = rng.poisson(means[action]).astype(float)
        counts[rr, action] += 1.0
        sums[rr, action] += reward
        regrets[:, t] = regrets[:, t - 1] + best - means[action]
    return regrets, counts


def run_bandit_experiments(cfg: Config) -> pd.DataFrame:
    families = {
        "Bernoulli": (simulate_bernoulli, np.asarray(cfg.bernoulli_means)),
        "Gaussian": (simulate_gaussian, np.asarray(cfg.gaussian_means)),
        "Poisson": (simulate_poisson, np.asarray(cfg.poisson_means)),
    }
    algorithms = ("KL-UCB", "Thompson sampling")
    curves: dict[tuple[str, str], np.ndarray] = {}
    counts_map: dict[tuple[str, str], np.ndarray] = {}
    rows: list[dict[str, float | str]] = []

    for f_idx, (family, (simulator, means)) in enumerate(families.items()):
        for a_idx, algorithm in enumerate(algorithms):
            regrets, counts = simulator(cfg, algorithm, SEED + 1000 * f_idx + 100 * a_idx)
            curves[(family, algorithm)] = regrets.mean(axis=0)
            counts_map[(family, algorithm)] = counts.mean(axis=0)
            final = regrets[:, -1]
            rows.extend(
                [
                    {"experiment": "bandit", "family": family, "algorithm": algorithm, "quantity": "mean final pseudo-regret", "value": float(final.mean())},
                    {"experiment": "bandit", "family": family, "algorithm": algorithm, "quantity": "standard error final pseudo-regret", "value": float(final.std(ddof=1) / math.sqrt(cfg.runs))},
                    {"experiment": "bandit", "family": family, "algorithm": algorithm, "quantity": "mean pulls of best arm", "value": float(counts[:, int(np.argmax(means))].mean())},
                ]
            )

    time = np.arange(1, cfg.horizon + 1)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.55), sharex=True)
    for ax, family in zip(axes, families.keys()):
        for algorithm in algorithms:
            ax.plot(time, curves[(family, algorithm)], linewidth=1.9, label=algorithm)
        ax.set_title(family)
        ax.set_xlabel("round")
        ax.grid(True, linewidth=0.3, alpha=0.25)
    axes[0].set_ylabel("mean pseudo-regret")
    axes[-1].legend(frameon=False)
    fig.suptitle("The policy skeleton stays the same; the family supplies the right evidence scale", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "exponential_family_regret.pdf", bbox_inches="tight")
    fig.savefig(OUT / "exponential_family_regret.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.55))
    x = np.arange(4)
    width = 0.36
    for ax, (family, (_, means)) in zip(axes, families.items()):
        for j, algorithm in enumerate(algorithms):
            ax.bar(x + (j - 0.5) * width, counts_map[(family, algorithm)], width=width, label=algorithm)
        ax.set_title(family)
        ax.set_xticks(x, [f"arm {j+1}" for j in x])
        ax.set_xlabel("arm")
        ax.grid(True, axis="y", linewidth=0.3, alpha=0.25)
    axes[0].set_ylabel("mean number of pulls")
    axes[-1].legend(frameon=False)
    fig.suptitle("Most samples eventually flow to the best mean", y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "exponential_family_pull_counts.pdf", bbox_inches="tight")
    fig.savefig(OUT / "exponential_family_pull_counts.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    return pd.DataFrame(rows)


def main() -> None:
    cfg = Config()
    frames = [
        plot_log_partition_map(),
        plot_sufficient_statistic_likelihood(),
        plot_conjugate_updates(),
        plot_rate_functions(),
        run_bandit_experiments(cfg),
    ]
    results = pd.concat(frames, ignore_index=True, sort=False)
    results.to_csv(OUT / "exponential_families_results.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
```


Banerjee, Arindam, Srujana Merugu, Inderjit S. Dhillon, and Joydeep Ghosh. 2005. "Clustering with Bregman Divergences." *Journal of Machine Learning Research* 6: 1705--49.


Barndorff-Nielsen, Ole E. 1978. *Information and Exponential Families in Statistical Theory*. Wiley.


Bishop, Christopher M. 2006. *Pattern Recognition and Machine Learning*. Springer.


Brown, Lawrence D. 1986. *Fundamentals of Statistical Exponential Families with Applications in Statistical Decision Theory*. Institute of Mathematical Statistics.


Garivier, Aurélien, and Olivier Cappé. 2011. "The KL-UCB Algorithm for Bounded Stochastic Bandits and Beyond." *Proceedings of the 24th Annual Conference on Learning Theory*.


Ghahramani, Z. 2015. "Probabilistic Machine Learning and Artificial Intelligence." *Nature* 521: 452--59.


Jin, T., P. Xu, X. Xiao, and A. Anandkumar. 2022. *Finite-Time Regret of Thompson Sampling Algorithms for Exponential Family Multi-Armed Bandits*.


Korda, Nathaniel, Emilie Kaufmann, and Rémi Munos. 2013. "Thompson Sampling for 1-Dimensional Exponential Family Bandits." *Advances in Neural Information Processing Systems*.


Lai, T. L., and H. Robbins. 1985. "Asymptotically Efficient Adaptive Allocation Rules." *Advances in Applied Mathematics* 6 (1): 4--22.


Lattimore, T., and C. Szepesvari. 2020. *Bandit Algorithms*. Cambridge University Press.


MacKay, David J. C. 2003. *Information Theory, Inference, and Learning Algorithms*. Cambridge University Press.


Wainwright, Martin J., and Michael I. Jordan. 2008. "Graphical Models, Exponential Families, and Variational Inference." *Foundations and Trends in Machine Learning* 1 (1--2): 1--305.
