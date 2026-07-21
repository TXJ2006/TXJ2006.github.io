---
title: "Recurrent Neural Networks and Sequential Learning Systems"
subtitle: "State, Memory, Gradient Flow, Gating, and PyTorch"
summary: "Beginning with the failure of stateless predictors on ordered data, this chapter develops recurrent neural networks as nonlinear state-space models, derives backpropagation through time, analyzes vanishing and exploding gradients through products of Jacobians, studies LSTM and GRU gates, reports controlled numerical experiments, and builds shape-safe PyTorch implementations for variable-length and streaming sequences."
description: "Beginning with the failure of stateless predictors on ordered data, this chapter develops recurrent neural networks as nonlinear state-space models, derives backpropagation through time, analyzes vanishing and exploding gradients through products of Jacobians, studies LSTM and GRU gates, reports controlled numerical experiments, and builds shape-safe PyTorch implementations for variable-length and streaming sequences."
date: 2026-07-21
lastmod: 2026-07-21
weight: 60
libraryFolder: "ai-foundations"
libraryFolderName: "人工智能基础"
libraryFolderColor: 0
tags: ["Machine Learning", "Recurrent Neural Networks", "Sequential Models", "Dynamical Systems", "PyTorch"]
draft: false
ShowToc: false
hideMeta: true
---

## Recurrent Neural Networks and Sequential Learning Systems

## Introduction

The linear and convolutional systems developed in the previous chapters are naturally described as maps from one completed object to another. A vector is presented, transformed, and classified. An image is presented, filtered through a hierarchy of spatial operators, and assigned a label. Two different examples do not ordinarily share a hidden state: once one output has been computed, the next example begins from the same architectural rule rather than from a memory of what happened before.

Sequential data invalidate this description. A word is interpreted partly through the words that preceded it. A speech frame is ambiguous without neighboring acoustic context. A sensor reading may be harmless in isolation and alarming as part of a trend. A controller must react not only to its current observation, but also to an internal estimate of a physical state that is not directly observed. In these problems, the relevant input is not merely a collection of vectors. It is an ordered process.

One could concatenate a sequence $(x_1,x_2,\ldots,x_T)$ into one enormous vector and apply a feedforward network. This can be reasonable when $T$ is small and fixed. It is unsatisfactory as a general solution. The number of parameters may depend on the chosen length, the model has no natural way to process a stream before it ends, and the same temporal relationship must be learned separately at different positions. A transition learned from time $3$ to time $4$ is not automatically reused from time $103$ to time $104$.

A recurrent neural network changes the object being learned. Instead of fitting one unrestricted map from an entire sequence to an output, it learns a state transition that is applied repeatedly:

<div class="display-equation">
$$
h_t=F_\theta(x_t,h_{t-1}),
\qquad
\widehat y_t=G_\theta(h_t).
$$
</div>

The hidden state $h_t$ is a compressed summary of the processed prefix. The same parameter vector $\theta$ is reused at every time step. This parameter sharing gives recurrence both its efficiency and its difficulty. It permits one rule to process sequences of many lengths, but it also forces learning signals to travel through repeated applications of the same nonlinear map. A recurrent network is therefore simultaneously a neural architecture, a nonlinear dynamical system, a shared-parameter computation graph, and a finite-dimensional memory mechanism.

The usual introduction moves quickly from the recurrence equation to the statement that gradients may vanish, then introduces the LSTM as a cure. That summary is directionally correct but mathematically incomplete. Vanishing and exploding gradients are properties of products of Jacobians, not merely one scalar recurrent weight. Spectral radius explains only part of the phenomenon; singular values, activation derivatives, non-normality, and the trajectory of the hidden state also matter. An LSTM does not make gradients immortal. It creates an additive state path whose retention factors are trainable, making useful time scales easier to represent and optimize. A GRU builds a related interpolation directly in hidden-state space. Both remain finite-state and sequential.

This chapter develops those distinctions from first principles. We begin with recurrence as a state-space model, derive the Elman network and backpropagation through time in vector form, analyze gradient transport through matrix products, and then study LSTM and GRU gates as learned time-scale controls. Controlled experiments make the optimization phenomena visible. The final sections treat variable lengths, truncated backpropagation, streaming state, encoder-decoder systems, and PyTorch implementation details.

## Sequences, State, and Causality

Let a sequence be

<div class="display-equation">
$$
x_{1:T}=(x_1,x_2,\ldots,x_T),
\qquad
x_t\in\mathbb R^D.
$$
</div>

The index $t$ may denote physical time, token position, spatial scan order, algorithmic depth, or any ordering that affects meaning. An ordinary RNN has no explicit knowledge of irregular time intervals unless elapsed time is included in the input or the transition itself is modified.

A causal sequence model produces its output at time $t$ from the prefix $x_{1:t}$:

<div class="display-equation">
$$
\widehat y_t=\Phi_\theta(x_1,\ldots,x_t).
$$
</div>

Recurrence factors this growing family of prefix maps through a fixed-dimensional state space:

<div class="display-equation">
$$
h_t=F_\theta(x_t,h_{t-1}),
\qquad
\widehat y_t=G_\theta(h_t),
\qquad
h_t\in\mathbb R^H.
$$
</div>

The state dimension $H$ does not grow with $t$. Every prefix, however long, must be represented by one vector in $\mathbb R^H$. This is the source of both the computational appeal and the information bottleneck. The recurrent transition does not literally store the complete past. It learns a task-dependent statistic of the past.

### State as a sufficient summary

For an ideal predictive task, one would like $h_t$ to be sufficient for the future target: once $h_t$ is known, the raw prefix should contain no additional useful information. Informally,

<div class="display-equation">
$$
Y_{t+1:}\perp\!\!\!\perp X_{1:t}\mid h_t.
$$
</div>

A trained neural state rarely satisfies exact statistical sufficiency. The equation is nevertheless a useful design objective. It explains why a hidden state should forget some details and preserve others. For sentiment classification, font style is irrelevant while a negation may matter for many later positions. For physical tracking, a filtered estimate of position and velocity may be more useful than the complete sensor history. The state is therefore not merely a cache; it is a learned compression map constrained by future usefulness.

### Four input-output organizations

The recurrence is compatible with several readout patterns.

- **One-to-many:** one initial input or condition produces a sequence, as in conditional generation.
- **Many-to-one:** a complete sequence produces one decision, as in sequence classification.
- **Synchronous many-to-many:** each input position has an aligned output, as in frame labeling or token tagging.
- **Asynchronous many-to-many:** an input sequence is encoded and an output sequence of a different length is decoded, as in classical neural machine translation.

These names describe surrounding topology, not fundamentally different recurrent cells. The same transition may emit at every step, only at the final step, or first encode and then decode.

<figure>
<img src="/images/notes/recurrent-neural-networks/rnn-unrolled.png" alt="An RNN cell unrolled across six time steps, with inputs below, outputs above, and hidden states passed horizontally." loading="lazy">
<figcaption><strong>Figure 1.</strong> Unrolling converts one recurrent rule into a depth-$T$ computation graph. The boxes are copies of the same function $F_\theta$; they do not have independent parameters.</figcaption>
</figure>

### Parameter sharing is the temporal inductive bias

A recurrence applies the same $F_\theta$ at all positions. This is the temporal analogue of convolutional weight sharing. It assumes that the rule for updating memory should not depend on an absolute time index. A phrase occurring near the beginning and the same phrase occurring near the end should be processed by the same transition mechanism.

The assumption is powerful but not universal. Seasonal processes, finite-horizon control, and positional language phenomena may depend explicitly on time. Such information can be supplied through a time coordinate, positional features, elapsed-time embeddings, or a controlled time-varying transition. Recurrence removes absolute position by default; it does not prove that absolute position is irrelevant.

A unidirectional RNN is causal when it is evaluated from past to future. A bidirectional RNN uses another recurrence from future to past and is therefore noncausal. The data pipeline can also violate causality: normalizing a time series with future statistics, imputing a missing value from later observations, or randomly splitting overlapping windows can leak future information even when the cell itself is unidirectional. Causality is a property of the complete modeling procedure.

## The Elman Recurrent Neural Network

Let the input width be $D$, hidden width $H$, and output width $K$. The Elman network computes

<div class="display-equation">
$$
\begin{aligned}
a_t&=W_xx_t+W_hh_{t-1}+b_h,\\
h_t&=\phi(a_t),\\
z_t&=W_yh_t+b_y,\\
\widehat y_t&=\psi(z_t),
\end{aligned}
$$
</div>

with

<div class="display-equation">
$$
W_x\in\mathbb R^{H\times D},
\qquad
W_h\in\mathbb R^{H\times H},
\qquad
W_y\in\mathbb R^{K\times H}.
$$
</div>

The hidden activation $\phi$ is commonly $\tanh$, although identity and ReLU recurrences have also been studied. The output map $\psi$ depends on the task: identity for regression, sigmoid for binary prediction, or a softmax interpreted through a cross-entropy loss.

### Why the recurrence carries history

Starting from $h_0$, expansion gives

<div class="display-equation">
$$
\begin{aligned}
h_1&=\phi(W_xx_1+W_hh_0+b_h),\\
h_2&=\phi\!\left(W_xx_2+W_h\phi(W_xx_1+W_hh_0+b_h)+b_h\right),\\
h_3&=\phi(W_xx_3+W_hh_2+b_h),
\end{aligned}
$$
</div>

and so forth. The state $h_t$ is a nested nonlinear function of every earlier input. In principle, the influence of $x_1$ can reach arbitrarily large $t$. In practice, the magnitude and usefulness of that influence depend on the recurrent dynamics and on whether optimization can assign credit through the intervening steps.

The statement that “an RNN can remember arbitrary history” is therefore only a graph-connectivity statement. A path exists from every earlier input to every later state. It says nothing about whether the path preserves distinguishable information, whether its derivative is numerically useful, or whether a finite-width state can avoid collisions between different histories.

### Parameter count

Under the convention of one hidden bias vector, the recurrent core contains

<div class="display-equation">
$$
HD+H^2+H=H(D+H+1)
$$
</div>

parameters. An output head adds $KH+K$. The count is independent of sequence length. Longer sequences require more computation and activation storage but introduce no new learned coefficients.

PyTorch stores separate input-hidden and hidden-hidden bias vectors. Its exact core count is therefore $H(D+H+2)$. The two biases could be algebraically merged in a simple cell, but they are stored separately for compatibility with fused kernels.

### Recurrence as a nonlinear state-space model

A discrete-time state-space system has the form

<div class="display-equation">
$$
s_t=f(s_{t-1},u_t),
\qquad
y_t=g(s_t).
$$
</div>

An RNN has exactly this structure, with a learned nonlinear transition and learned observation map. This connection provides tools for studying stability, fixed points, observability, and memory.

For constant input $x_t=x$, a fixed point satisfies

<div class="display-equation">
$$
h^*=\phi(W_xx+W_hh^*+b_h).
$$
</div>

Linearizing near $h^*$ gives

<div class="display-equation">
$$
\delta h_t\approx J^*\delta h_{t-1},
\qquad
J^*=\operatorname{Diag}(\phi'(a^*))W_h.
$$
</div>

The same local Jacobian controls the forward evolution of perturbations and the backward transport of gradients. Forward memory stability and backward gradient stability are two views of repeated linearization.

### Contractive recurrence and deliberate forgetting

Suppose $\phi$ is $L_\phi$-Lipschitz. For two states driven by the same input,

<div class="display-equation">
$$
\|h_t-h_t'\|_2
\leq
L_\phi\|W_h\|_2\|h_{t-1}-h_{t-1}'\|_2.
$$
</div>

If $L_\phi\|W_h\|_2<1$, then

<div class="display-equation">
$$
\|h_t-h_t'\|_2
\leq
(L_\phi\|W_h\|_2)^t\|h_0-h_0'\|_2.
$$
</div>

This is excellent for stability and poor for long memory: the influence of the initial condition decays exponentially. A strongly contractive model is robust and forgetful. A model near marginal stability can retain information longer but becomes harder to optimize and more sensitive to perturbations. Gating does not eliminate this trade-off; it makes the contraction factor input-dependent and learnable.

### Hidden-state initialization

The initial state $h_0$ is often zero, but this is a convention rather than a theorem. Alternatives include a learned global state, a state predicted from side information, the final state of a preceding stream chunk, or a physical estimator's state in a hybrid model. Zero initialization asserts that every sequence begins from the same absence of memory. That is reasonable for independent documents and wrong for a continuous stream arbitrarily cut into minibatches.

## Backpropagation Through Time

Training requires derivatives of a scalar loss with respect to parameters shared across time. The recurrence is unrolled into an ordinary acyclic computation graph, and reverse-mode differentiation is applied to that graph. This is backpropagation through time, or BPTT.

Let

<div class="display-equation">
$$
L=\sum_{t=1}^{T}\ell_t(h_t),
$$
</div>

where $\ell_t$ may include the output head and may be zero at unsupervised positions. Define the local hidden gradient

<div class="display-equation">
$$
g_t=\frac{\partial\ell_t}{\partial h_t},
$$
</div>

and the transition Jacobian

<div class="display-equation">
$$
J_t=\frac{\partial h_t}{\partial h_{t-1}}
=D_tW_h,
\qquad
D_t=\operatorname{Diag}(\phi'(a_t)).
$$
</div>

Let

<div class="display-equation">
$$
\delta_t=\frac{\partial L}{\partial h_t}
$$
</div>

be the total gradient arriving at state $h_t$. It contains both the local contribution and all future contributions. The adjoint recursion is

<div class="display-equation">
$$
\delta_T=g_T,
\qquad
\delta_t=g_t+J_{t+1}^{\top}\delta_{t+1}.
$$
</div>

This is BPTT in compact form. The forward state repeatedly applies $F_\theta$; the backward adjoint repeatedly applies transposed Jacobians and adds local error signals.

Define the preactivation gradient

<div class="display-equation">
$$
\lambda_t=\frac{\partial L}{\partial a_t}=D_t\delta_t.
$$
</div>

Then

<div class="display-equation">
$$
\begin{aligned}
\frac{\partial L}{\partial W_x}
&=\sum_{t=1}^{T}\lambda_t x_t^\top,\\
\frac{\partial L}{\partial W_h}
&=\sum_{t=1}^{T}\lambda_t h_{t-1}^\top,\\
\frac{\partial L}{\partial b_h}
&=\sum_{t=1}^{T}\lambda_t.
\end{aligned}
$$
</div>

The sums are the derivative signature of parameter sharing. One entry of $W_h$ participates at every time step, so its gradient accumulates every path through which it influenced the loss.

### The long-distance term

A loss at time $t$ affects an earlier state $h_k$ through

<div class="display-equation">
$$
\frac{\partial\ell_t}{\partial h_k}
=
\left(J_tJ_{t-1}\cdots J_{k+1}\right)^\top g_t,
\qquad k&lt;t.
$$
</div>

The product contains $t-k$ Jacobians. Long-range credit assignment is therefore controlled by a time-ordered product of matrices. The matrices need not commute, their singular directions change with the state, and replacing the product by $W_h^{t-k}$ is valid only for a linear recurrence or a special operating regime.

### Complexity and activation memory

A dense Elman cell requires approximately $O(HD+H^2)$ arithmetic per time step. A length-$T$ forward pass costs $O(T(HD+H^2))$ work; the backward pass has the same order. For batch size $B$, hidden-state storage alone is $O(TBH)$. Gated cells retain several additional preactivations and states per time step, so activation memory often determines the maximum trainable sequence length.

### Truncated BPTT

For long or unbounded streams, full BPTT may be too expensive. Truncated BPTT processes chunks of length $K$, carries the numerical hidden state into the next chunk, but detaches that state from the previous computation graph. In PyTorch this is `h = h.detach()` or, for an LSTM, detaching both `(h, c)`.

Truncation is not merely a memory optimization. It changes the gradient estimator. Dependencies longer than $K$ receive no direct credit through the detached boundary. A model may still exploit longer history if useful state is learned within earlier chunks, but the training signal for constructing that state is biased. Choosing $K$ is therefore a statistical and computational decision.

## Why Recurrent Gradients Vanish or Explode

The long-distance derivative is a product of matrices. Taking spectral norms gives the elementary bound

<div class="display-equation">
$$
\left\|
\frac{\partial h_t}{\partial h_k}
\right\|_2
=
\left\|J_tJ_{t-1}\cdots J_{k+1}\right\|_2
\leq
\prod_{j=k+1}^{t}\|J_j\|_2.
$$
</div>

For a $\tanh$ Elman network,

<div class="display-equation">
$$
\|J_j\|_2
\leq
\|D_j\|_2\|W_h\|_2
\leq
\|W_h\|_2,
$$
</div>

because $|\tanh'(a)|\leq1$. If the typical singular gain is below one, the product contracts exponentially. If it is persistently above one, it may grow exponentially. The word *typical* is important. One unusually small factor can destroy a gradient path, and one unusually large transient can dominate a finite sequence.

### The scalar recurrence exposes the basic mechanism

For

<div class="display-equation">
$$
h_t=\tanh(wx_t+uh_{t-1}+b),
$$
</div>

we have

<div class="display-equation">
$$
\frac{\partial h_t}{\partial h_k}
=
\prod_{j=k+1}^{t}
\left[u\bigl(1-h_j^2\bigr)\right].
$$
</div>

Even $|u|=1$ does not preserve gradients if the hidden units enter the saturated region $|h_j|\approx1$, because $1-h_j^2\approx0$. Conversely, choosing $|u|>1$ to compensate for saturation creates unstable regimes elsewhere. The problem is not simply “make the recurrent weight equal to one.” The activation derivative and the visited trajectory are inseparable from the recurrent matrix.

A numerical example makes the exponential scale visible. If every step contributes a factor $0.81$, then a signal transported across $d$ steps is multiplied by $0.81^d$. At $d=10$ this is about $0.12$; at $d=50$ it is about $2.7\times10^{-5}$; at $d=100$ it is below $10^{-9}$. A graph path may still exist, but for floating-point optimization it has become nearly irrelevant.

### Singular values are more informative than eigenvalues for finite-horizon gradients

The spectral radius $\rho(W_h)$ is often used as a memory diagnostic. For the linear autonomous recurrence $h_t=W_hh_{t-1}$, the asymptotic behavior of $W_h^t$ is indeed related to the eigenvalues. Gradient propagation, however, is governed by products of state-dependent Jacobians. At a finite horizon, the relevant amplification is measured by singular values:

<div class="display-equation">
$$
\sup_{\|v\|_2=1}
\left\|J_t\cdots J_{k+1}v\right\|_2
=
\sigma_{\max}(J_t\cdots J_{k+1}).
$$
</div>

Eigenvalues describe invariant directions of a square matrix. Singular values describe the largest and smallest Euclidean gains, whether or not those directions are invariant. For non-normal matrices, the two can be radically different.

A matrix $A$ is normal when $A^\top A=AA^\top$. Symmetric, orthogonal, and unitary matrices are normal. A non-normal matrix may have every eigenvalue strictly inside the unit circle and still exhibit large transient amplification before eventually decaying. For example,

<div class="display-equation">
$$
A=
\begin{pmatrix}
0.92 & \alpha\\
0 & 0.92
\end{pmatrix}
$$
</div>

has spectral radius $0.92$ for every $\alpha$, yet its powers contain an off-diagonal term proportional to $t\alpha(0.92)^{t-1}$. A finite sequence can therefore experience a large exploding-gradient episode even though the asymptotic eigenvalue test labels the linear system stable.

<figure>
<img src="/images/notes/recurrent-neural-networks/jacobian-product-gradient-flow.png" alt="Log-scale plots of products of recurrent Jacobian norms under contractive, near-neutral, expansive, and non-normal dynamics." loading="lazy">
<figcaption><strong>Figure 2.</strong> Products of recurrent gains are exponentially sensitive to their average logarithm. The non-normal example has spectral radius below one but undergoes severe finite-horizon amplification, showing why eigenvalues alone do not characterize BPTT stability.</figcaption>
</figure>

### A Lyapunov-exponent view

For a long trajectory, a useful summary is the average logarithmic growth rate of the Jacobian product. The largest finite-time Lyapunov exponent is

<div class="display-equation">
$$
\lambda_{k,t}
=
\frac{1}{t-k}
\log\sigma_{\max}
\left(J_tJ_{t-1}\cdots J_{k+1}\right).
$$
</div>

When this quantity remains negative, perturbations and gradients contract exponentially. When it is positive, some direction expands exponentially. Near zero, information can persist, but the system is close to the boundary between stable forgetting and instability. Different singular directions may have different exponents, so a state may preserve a small memory subspace while rapidly forgetting everything orthogonal to it.

This is a more faithful description of learned recurrence than one global scalar such as $\|W_h\|_2$. The activation pattern makes $J_t$ data-dependent; the relevant stability is along the trajectory induced by the data and parameters.

### Forward sensitivity and backward sensitivity are linked

A product that expands gradients also expands small perturbations in hidden state along corresponding directions. Exploding gradients are therefore not only an optimization pathology. They indicate that the state dynamics are locally sensitive. A small numerical error, input perturbation, or parameter change may be amplified through time.

The converse is equally important. Strongly contracting dynamics are robust, but they erase distinctions between histories. Long memory requires at least some directions that are close to neutral. The central design problem is therefore not to make every direction neutral. It is to create controllable subspaces with task-appropriate time scales while keeping the remaining dynamics stable.

### Gradient clipping controls the update, not the memory mechanism

Global norm clipping replaces a gradient $g$ by

<div class="display-equation">
$$
\widetilde g
=
\begin{cases}
g, & \|g\|_2\leq\tau,\\[3pt]
\displaystyle \tau\frac{g}{\|g\|_2}, & \|g\|_2>\tau,
\end{cases}
$$
</div>

where $\tau$ is a threshold. This preserves direction and limits update magnitude. Elementwise clipping instead truncates each coordinate separately and can alter direction severely.

Clipping is effective against rare explosions because one abnormal sequence cannot produce an arbitrarily large parameter step. It does not repair a vanished gradient. Once a long-distance signal has been multiplied down to $10^{-12}$, clipping has nothing to restore. Nor does clipping make the forward state stable; it changes the optimizer's response after the backward pass has already computed the unstable derivative.

### Initialization can delay, but not universally solve, the problem

Orthogonal initialization chooses $W_h^\top W_h=I$, so the linear recurrent map preserves Euclidean norm. This is valuable near the origin of a $\tanh$ network, where $D_t\approx I$. Once units saturate, $D_t$ contracts. An identity initialization creates a similar near-integrator and can work well with ReLU activations for some tasks, but it permits unbounded positive states and does not adapt its time scale to the input.

Unitary and orthogonal RNNs constrain the recurrent transformation to preserve norms more exactly. They improve gradient transport and expose a useful principle: memory can be protected by structured dynamics rather than only by gates. The restriction also changes the hypothesis class. A perfectly norm-preserving linear transition cannot directly forget; forgetting must arise from nonlinearities, input interactions, or additional mechanisms.

Leaky recurrence creates an explicit skip path:

<div class="display-equation">
$$
h_t=(1-\alpha)h_{t-1}+\alpha\phi(W_xx_t+W_hh_{t-1}+b),
\qquad 0&lt;\alpha\leq1.
$$
</div>

The coefficient $1-\alpha$ supplies a direct persistence term, much as a residual connection does across network depth. A fixed $\alpha$ selects one characteristic time scale. LSTM and GRU generalize the idea by making the interpolation coefficient data-dependent and learned.

## A Controlled Long-Range Recall Experiment

A useful experiment should isolate the mechanism under discussion rather than hide it inside a large benchmark. Consider a length-$50$ binary delayed-recall task. At the first time step, the model receives either $+1$ or $-1$ with equal probability. Every later input is zero. The target, revealed only at the final time step, is the class encoded by the first symbol:

<div class="display-equation">
$$
x_1\in\{-1,+1\},
\qquad
x_2=\cdots=x_{50}=0,
\qquad
y=\mathbf 1\{x_1=+1\}.
$$
</div>

There are no distractors and no statistical ambiguity. The only challenge is to carry one bit of information through forty-nine state transitions and train the transition from a loss at the end.

We trained scalar-state Elman, LSTM, and GRU models with Adam for $300$ updates, using minibatches of $256$, binary cross-entropy, a fixed random seed, and global gradient clipping. The experiment is intentionally small enough to reproduce in seconds. It is a diagnostic, not a universal ranking of architectures. Different initializations or hyperparameters can change individual curves, but the task exposes the structural advantage of an additive gated path.

<figure>
<img src="/images/notes/recurrent-neural-networks/delayed-recall-loss.png" alt="Training loss curves for scalar vanilla RNN, LSTM, and GRU models on a length-50 delayed recall task." loading="lazy">
<figcaption><strong>Figure 3.</strong> Training loss on the delayed-recall task. In this fixed-seed run, the scalar Elman network remains near chance-level cross-entropy, whereas the gated models rapidly find a persistent state.</figcaption>
</figure>

<figure>
<img src="/images/notes/recurrent-neural-networks/delayed-recall-accuracy.png" alt="Training accuracy curves for scalar vanilla RNN, LSTM, and GRU models on a length-50 delayed recall task." loading="lazy">
<figcaption><strong>Figure 4.</strong> Minibatch accuracy for the same run. The LSTM first exceeds $99\%$ at update $13$, the GRU at update $34$, and the Elman model never exceeds the threshold during the $300$ updates.</figcaption>
</figure>

The final measurements are:

| Model | Final loss | Final accuracy | First update with accuracy $\geq99\%$ |
|---|---:|---:|---:|
| Elman RNN | $0.693993$ | $0.5127$ | not reached |
| LSTM | $0.009774$ | $1.0000$ | $13$ |
| GRU | $0.004710$ | $1.0000$ | $34$ |

The result should not be interpreted as “an RNN cannot store one bit.” A hand-designed linear recurrence with weight exactly one stores it perfectly. The failure is a joint failure of representation and optimization: a generic saturating recurrence must discover a nearly neutral channel through a long product of derivatives, while the gated cells already contain an explicit interpolation path whose coefficient can approach one.

The experiment also illustrates why final benchmark accuracy alone is a weak diagnostic. All three model classes have enough formal capacity for the task. The difference lies in how easily gradient descent can reach the useful solution from a generic initialization.

## Long Short-Term Memory

The LSTM separates the recurrent state into a cell state $c_t$ and an exposed hidden state $h_t$. Under a common convention,

<div class="display-equation">
$$
\begin{aligned}
f_t&=\sigma(W_fx_t+U_fh_{t-1}+b_f),\\
i_t&=\sigma(W_ix_t+U_ih_{t-1}+b_i),\\
o_t&=\sigma(W_ox_t+U_oh_{t-1}+b_o),\\
g_t&=\tanh(W_gx_t+U_gh_{t-1}+b_g),\\
c_t&=f_t\odot c_{t-1}+i_t\odot g_t,\\
h_t&=o_t\odot\tanh(c_t).
\end{aligned}
$$
</div>

The forget gate $f_t$ controls retention, the input gate $i_t$ controls writing, the candidate $g_t$ proposes new content, and the output gate $o_t$ controls exposure. Each gate is a vector, so different coordinates can operate on different time scales.

### One affine map is sufficient in implementation

The four preactivations can be computed together:

<div class="display-equation">
$$
\begin{pmatrix}
a_t^{(f)}\\a_t^{(i)}\\a_t^{(o)}\\a_t^{(g)}
\end{pmatrix}
=
W_{ih}x_t+W_{hh}h_{t-1}+b,
$$
</div>

where the result has width $4H$. It is then split into four blocks and transformed by three sigmoids and one $\tanh$. Fused implementations exploit this layout so that an LSTM does not launch eight separate dense operations at every step.

With one combined bias, the core parameter count is

<div class="display-equation">
$$
4H(D+H+1).
$$
</div>

PyTorch stores two bias vectors and therefore uses $4H(D+H+2)$ parameters per layer and direction. A GRU uses three affine blocks, while an Elman cell uses one.

<figure>
<img src="/images/notes/recurrent-neural-networks/recurrent-parameter-counts.png" alt="Parameter count comparison among Elman RNN, GRU, and LSTM cells as hidden width increases." loading="lazy">
<figcaption><strong>Figure 5.</strong> Recurrent-core parameter counts for a fixed input width under the two-bias convention used by PyTorch. Gating improves optimization geometry at the cost of approximately three or four times the cell parameters and additional activation memory.</figcaption>
</figure>

### The additive cell path

If the gates were treated as fixed with respect to $c_{t-1}$, the direct derivative along the cell path would be

<div class="display-equation">
$$
\left.
\frac{\partial c_t}{\partial c_{t-1}}
\right|_{\text{direct}}
=
\operatorname{Diag}(f_t).
$$
</div>

Consequently, the direct path from $c_k$ to $c_t$ contains the product

<div class="display-equation">
$$
\prod_{j=k+1}^{t}\operatorname{Diag}(f_j).
$$
</div>

This is the central architectural advantage. A vanilla RNN repeatedly multiplies by a full recurrent Jacobian entangled with an activation derivative. The LSTM contains an additive state highway whose coordinatewise retention factors can be trained toward one.

The formula is a *partial-path* analysis, not the complete derivative. Gates depend on $h_{t-1}$, which depends on $c_{t-1}$, so the total derivative contains additional terms. Those terms can still vanish or explode. The LSTM creates a favorable path; it does not delete every unfavorable path or guarantee stable training for arbitrary parameters.

### Forget gates encode learned half-lives

Suppose one coordinate has approximately constant forget value $f\in(0,1)$ and receives no new input. Then

<div class="display-equation">
$$
c_{t+d}=f^dc_t.
$$
</div>

Its half-life $d_{1/2}$ satisfies $f^{d_{1/2}}=1/2$, hence

<div class="display-equation">
$$
d_{1/2}
=
\frac{\log(1/2)}{\log f}.
$$
</div>

A forget value $0.9$ has a half-life of about $6.6$ steps; $0.99$ gives about $69$ steps; $0.999$ gives about $693$ steps. Small changes near one create enormous changes in memory horizon. Initializing the forget-gate bias positively is therefore common: it places the initial model in a regime that does not erase state immediately, while training remains free to shorten the time scale where appropriate.

The same equation reveals a limitation. To preserve a signal across ten thousand steps through one coordinate, the average forget gate must be extraordinarily close to one. Any persistent deviation accumulates exponentially. LSTM improves long memory; it does not make finite precision and finite state dimension disappear.

### The cell state is not bounded by construction

The hidden state satisfies $|h_{t,j}|<1$ under the usual $\tanh$ output, but the cell state is an additive accumulator:

<div class="display-equation">
$$
c_t=f_t\odot c_{t-1}+i_t\odot g_t.
$$
</div>

Because additions can accumulate over many steps, $c_t$ need not lie in $[-1,1]$. This allows counting and integration-like behavior. It also permits drift or saturation of $\tanh(c_t)$ when the gates repeatedly write in one direction. Cell clipping, normalization, regularization, and careful initialization may be useful in difficult regimes.

### Reading a learned scalar LSTM

The scalar LSTM from the delayed-recall experiment converges to an integrator-like solution. Its forget gate remains high, the output gate exposes the accumulated cell, and the cell state separates the two classes throughout the zero-input tail.

<figure>
<img src="/images/notes/recurrent-neural-networks/lstm-gate-dynamics.png" alt="Time series of input, forget, input, output gates, cell state, and hidden state for a learned scalar LSTM on delayed recall." loading="lazy">
<figcaption><strong>Figure 6.</strong> Gate and state trajectories for the trained scalar LSTM. The network does not need to implement a human-designed “write once and freeze” algorithm. It learns an alternative persistent dynamical solution in which the cell state remains class-separating until the final readout.</figcaption>
</figure>

A gate plot is not a proof of semantic interpretation. A high forget gate means that one coordinate retains its current cell value; it does not tell us what abstract concept that coordinate represents. Hidden dimensions are coupled through the affine maps, and equivalent functions can be represented after rotations, permutations, or rescalings of internal coordinates. Gate inspection is a dynamical diagnostic, not a complete explanation.

### Variants of the LSTM

Several common variants alter how gates interact.

- **Peephole connections** feed $c_{t-1}$ or $c_t$ directly into gate preactivations, allowing gates to inspect the accumulator.
- **Coupled input-forget gates** impose $i_t=1-f_t$, turning the cell update into a convex interpolation between old and proposed content.
- **Projection LSTMs** compute a large cell and then project the hidden output to a smaller dimension, reducing recurrent and output cost.
- **Layer-normalized LSTMs** normalize gate preactivations or cell transformations within each example, which is compatible with variable sequence lengths and small batches.
- **Bidirectional LSTMs** combine a forward and backward recurrence. They improve offline contextual representation but cannot be used for strict streaming inference without future lookahead.

No variant dominates every task. The basic cell remains strong because it exposes a simple additive memory path while leaving the gates expressive enough to learn multiple time scales.

## Gated Recurrent Units

The GRU merges cell and hidden state and uses two main gates. One widely used convention is

<div class="display-equation">
$$
\begin{aligned}
z_t&=\sigma(W_zx_t+U_zh_{t-1}+b_z),\\
r_t&=\sigma(W_rx_t+U_rh_{t-1}+b_r),\\
\widetilde h_t&=\tanh\!
\left(W_hx_t+U_h(r_t\odot h_{t-1})+b_h\right),\\
h_t&=(1-z_t)\odot h_{t-1}+z_t\odot\widetilde h_t.
\end{aligned}
$$
</div>

Some libraries reverse the interpretation of $z_t$ and write $h_t=z_t\odot h_{t-1}+(1-z_t)\odot\widetilde h_t$. Both are valid. A derivation must state its convention before calling $z_t$ an “update” or “retention” gate.

The reset gate $r_t$ controls how much previous state enters the candidate. The update gate interpolates between the old state and the candidate. The direct derivative path includes

<div class="display-equation">
$$
\left.
\frac{\partial h_t}{\partial h_{t-1}}
\right|_{\text{skip}}
=
\operatorname{Diag}(1-z_t),
$$
</div>

under the convention above. As with the LSTM, the total derivative contains additional terms through the gates and candidate. The skip path nevertheless gives the optimizer a direct way to preserve selected coordinates.

### GRU versus LSTM

A GRU has core parameter count $3H(D+H+1)$ under one-bias notation, or $3H(D+H+2)$ in PyTorch. It usually uses less memory and computation than an LSTM of the same hidden width. The LSTM offers a separate unbounded cell state and an output gate, giving more independent control over storage and exposure. The GRU uses a single bounded hidden state and a simpler interpolation.

The correct comparison is not “which architecture is theoretically better?” Their parameter budgets, hidden widths, regularization, sequence lengths, and hardware kernels should be matched. On some datasets a smaller GRU is sufficient; on others the separate cell path of an LSTM is useful. Both are much closer to one another than either is to an ungated Elman cell in terms of gradient geometry.

### Minimal gated units

Further simplifications couple or remove gates. A minimal gated unit may use one gate to control both resetting and interpolation. Fewer gates reduce parameters and can be advantageous on small datasets or constrained hardware. They also remove degrees of freedom. Whether that helps depends on whether the eliminated control was unnecessary capacity or a useful time-scale mechanism.

## Stacking, Directionality, and Sequence Topologies

A single recurrent layer maps $x_{1:T}$ to hidden states $h_{1:T}^{(1)}$. A stacked RNN feeds the first layer's sequence to a second recurrence:

<div class="display-equation">
$$
h_t^{(\ell)}
=
F_{\theta_\ell}
\left(h_t^{(\ell-1)},h_{t-1}^{(\ell)}\right),
\qquad h_t^{(0)}=x_t.
$$
</div>

Depth now appears in two directions: across time and across layers. Even a modest network with $L$ layers and $T$ steps has computational paths of length on the order of $LT$. Residual connections between recurrent layers, normalization, dropout, and careful initialization become increasingly important.

### Dropout must respect temporal structure

Naively drawing an independent dropout mask at every time step injects rapidly changing noise into the state transition. Variational or locked dropout instead samples one mask per sequence and reuses it across time. This preserves a coherent subnetwork throughout the sequence while still regularizing feature usage. Recurrent-weight dropout requires additional care because perturbing the hidden-to-hidden map directly changes the memory dynamics.

PyTorch's built-in `dropout` argument for stacked `nn.RNN`, `nn.GRU`, and `nn.LSTM` is applied between recurrent layers, not along the recurrent connection of a single-layer module. It has no effect when `num_layers=1`.

### Bidirectional recurrence

A bidirectional layer computes

<div class="display-equation">
$$
\overrightarrow h_t
=F_{\rightarrow}(x_t,\overrightarrow h_{t-1}),
\qquad
\overleftarrow h_t
=F_{\leftarrow}(x_t,\overleftarrow h_{t+1}),
$$
</div>

and concatenates or combines the two states. Every output position can use both past and future context. This is appropriate for offline tagging, transcription with full utterances, and representation learning over completed sequences. It is inappropriate for a causal deployment unless latency permits a lookahead window.

Bidirectionality also doubles output width and usually doubles recurrent parameters. A shape-safe implementation must account for `num_directions=2` when constructing the downstream head.

## Encoder-Decoder Recurrence and the Fixed-Context Bottleneck

For asynchronous many-to-many problems, an encoder processes an input sequence and a decoder generates an output sequence. The earliest neural sequence-to-sequence systems summarized the entire source through the final encoder state:

<div class="display-equation">
$$
c=h_T^{\mathrm{enc}},
\qquad
s_m=F_{\mathrm{dec}}(y_{m-1},s_{m-1},c).
$$
</div>

This permits different input and output lengths, but it asks one fixed-dimensional vector to preserve all source information. Performance degrades as the source becomes longer because the decoder has no direct access to intermediate encoder states.

Attention removes this fixed-context bottleneck. At decoder step $m$, it constructs a data-dependent context

<div class="display-equation">
$$
\alpha_{m,t}
=
\frac{
\exp e(s_{m-1},h_t^{\mathrm{enc}})
}{
\sum_{j=1}^{T}
\exp e(s_{m-1},h_j^{\mathrm{enc}})
},
\qquad
c_m=
\sum_{t=1}^{T}\alpha_{m,t}h_t^{\mathrm{enc}}.
$$
</div>

The decoder can now retrieve different source positions at different output steps. This was historically a decisive improvement in neural machine translation and conceptually foreshadowed the Transformer: recurrence maintained the state, while attention created direct content-based paths to the full source sequence.

### Teacher forcing and exposure bias

During training, a decoder often receives the true previous token $y_{m-1}$; during inference, it receives its own sampled or selected prediction $\widehat y_{m-1}$. Teacher forcing makes optimization easier because errors do not immediately corrupt subsequent inputs, but it creates a train-test mismatch. At inference, one early mistake can move the decoder into a prefix distribution that was rare during training.

Scheduled sampling, sequence-level objectives, beam search, and minimum-risk training attempt to address parts of this problem. None changes the basic fact that autoregressive generation is a sequential decision process over model-generated prefixes. Token-level likelihood is tractable and effective, but it does not directly optimize every property of a completed sequence.

## Variable-Length Sequences, Padding, and Masks

Real minibatches contain sequences of different lengths. A common representation pads every sequence to the maximum length $T_{\max}$ in the batch. Let $\ell_b$ be the true length of example $b$. A mask is

<div class="display-equation">
$$
M_{b,t}=\mathbf 1\{t&lt;\ell_b\}.
$$
</div>

For a token-level loss, the padded positions must be excluded:

<div class="display-equation">
$$
L
=
\frac{
\sum_{b,t}M_{b,t}\,\ell_{b,t}
}{
\sum_{b,t}M_{b,t}
}.
$$
</div>

Merely setting padded input vectors to zero is not sufficient. A recurrent transition with bias can continue changing the hidden state after the true sequence ends. If the final padded state is used for classification, the result depends on how much padding happened to be added. One must either gather the state at the last valid position, use packed sequences, or explicitly preserve the state under the mask.

### Packed sequences

PyTorch's `pack_padded_sequence` removes padded time steps from recurrent computation. The lengths are conventionally supplied on the CPU, and the sequences should be sorted unless `enforce_sorted=False` is used. The recurrent output is a `PackedSequence`; `pad_packed_sequence` converts it back to a padded tensor when per-position outputs are needed.

For a unidirectional final-state classifier, the final hidden tensor returned by the recurrent module is often the cleanest summary. For a bidirectional multilayer network, its shape is

<div class="display-equation">
$$
(L\cdot D_{\mathrm{dir}},\ B,\ H),
$$
</div>

not $(B,H)$. It should be reshaped into `(num_layers, num_directions, batch, hidden)` before selecting the last layer and combining directions.

### Sorting, restoration, and label alignment

When a data loader sorts sequences by length, every aligned object—labels, sample weights, identifiers, and side information—must undergo the same permutation. If predictions must be returned in original order, store the inverse permutation. Sequence models are particularly vulnerable to silent alignment bugs because tensor shapes remain valid even when examples no longer match their targets.

### Bucketing reduces wasted computation

Padding overhead depends on the length spread within a minibatch. Grouping examples of similar length into buckets can substantially reduce memory and computation while preserving randomization between buckets. The gain is architectural rather than statistical: the recurrent model processes fewer artificial time steps.

## Stateful and Streaming Recurrence

A recurrent module is *stateful* when the hidden state from one call is intentionally reused in the next. This is appropriate for a continuous signal divided into chunks. It is dangerous when minibatches contain unrelated sequences.

Suppose a stream is partitioned into chunks $x_{1:K}$, $x_{K+1:2K}$, and so on. The final state from one chunk should initialize the next:

<div class="display-equation">
$$
h_{mK}
\longrightarrow
h_{mK}^{\text{initial for next chunk}}.
$$
</div>

After each optimizer step, the carried state must usually be detached from the old graph. Otherwise the next backward pass attempts to traverse every preceding chunk, causing memory growth and eventually a second-backward error after graph buffers have been freed.

State reuse also requires stable stream identity. If a batch slot contains user A in one chunk and user B in the next, blindly carrying the slot's hidden state leaks information between users. Production systems need an explicit state store keyed by stream identity, plus reset policies for session boundaries, missing chunks, model-version changes, and inactivity timeouts.

### Training and deployment state distributions can differ

A model trained only on independent sequences beginning from zero may be deployed on states produced by hours of its own recurrence. Conversely, a model trained with state carried across shuffled chunks can learn artificial cross-example dependencies. The distribution of initial states is part of the training data distribution and should match deployment as closely as possible.

## Shape-Safe PyTorch Implementations

PyTorch recurrent modules support two sequence layouts:

- with `batch_first=True`, inputs and outputs use `(batch, time, feature)`;
- hidden states always use `(layers × directions, batch, hidden)`.

The phrase *batch first* does not apply to hidden tensors. Many recurrent shape errors come from overlooking this exception.

### A manual Elman cell

Implementing one cell directly makes parameter sharing and state updates explicit.

```python
from __future__ import annotations

import torch
from torch import Tensor, nn


class ManualElmanRNN(nn.Module):
    """A minimal tanh RNN that returns every state and the final state."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0:
            raise ValueError("input_size and hidden_size must be positive")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.x_to_h = nn.Linear(input_size, hidden_size, bias=True)
        self.h_to_h = nn.Linear(hidden_size, hidden_size, bias=False)

        # Orthogonal initialization delays recurrent norm distortion near h=0.
        nn.init.orthogonal_(self.h_to_h.weight)

    def forward(self, x: Tensor, h0: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if x.ndim != 3:
            raise ValueError(f"expected (batch, time, feature), got {tuple(x.shape)}")
        batch, steps, features = x.shape
        if features != self.input_size:
            raise ValueError(
                f"expected feature width {self.input_size}, got {features}"
            )

        if h0 is None:
            h = x.new_zeros(batch, self.hidden_size)
        else:
            expected = (batch, self.hidden_size)
            if tuple(h0.shape) != expected:
                raise ValueError(f"expected h0 shape {expected}, got {tuple(h0.shape)}")
            h = h0

        states: list[Tensor] = []
        for t in range(steps):
            h = torch.tanh(self.x_to_h(x[:, t]) + self.h_to_h(h))
            states.append(h)

        if not states:
            empty = x.new_empty(batch, 0, self.hidden_size)
            return empty, h

        return torch.stack(states, dim=1), h
```

The module loops in Python and is therefore slower than a fused backend. Its value is conceptual and diagnostic. The same `x_to_h` and `h_to_h` parameters are applied at every iteration, while autograd records a different operation node for each use.

### A variable-length bidirectional GRU classifier

```python
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence


class PackedBiGRUClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        *,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least one")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.num_directions = 2

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=recurrent_dropout,
            bidirectional=True,
            batch_first=True,
        )
        self.head = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x: Tensor, lengths: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected x=(batch,time,feature), got {tuple(x.shape)}")
        if lengths.ndim != 1 or lengths.numel() != x.size(0):
            raise ValueError("lengths must have one entry per batch example")
        if torch.any(lengths <= 0) or torch.any(lengths > x.size(1)):
            raise ValueError("every length must be in [1, time]")

        packed = pack_padded_sequence(
            x,
            lengths.detach().to("cpu"),
            batch_first=True,
            enforce_sorted=False,
        )
        _, h_n = self.gru(packed)

        # (layers * directions, batch, hidden)
        h_n = h_n.view(
            self.num_layers,
            self.num_directions,
            x.size(0),
            self.hidden_size,
        )
        last_layer = h_n[-1]                 # (directions, batch, hidden)
        summary = torch.cat(
            (last_layer[0], last_layer[1]),
            dim=-1,
        )
        return self.head(summary)
```

This model is noncausal because it is bidirectional. For streaming classification, set `bidirectional=False` and adjust the head width to `hidden_size`.

### A stable training step

```python
from collections.abc import Iterable

import torch
from torch import Tensor, nn


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    x: Tensor,
    lengths: Tensor,
    targets: Tensor,
    *,
    max_grad_norm: float = 1.0,
) -> dict[str, float]:
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    logits = model(x, lengths)
    loss = criterion(logits, targets)
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss: {loss.detach().item()}")

    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    if not torch.isfinite(grad_norm):
        optimizer.zero_grad(set_to_none=True)
        raise FloatingPointError("non-finite gradient norm")

    optimizer.step()

    with torch.no_grad():
        accuracy = (logits.argmax(dim=-1) == targets).float().mean()

    return {
        "loss": float(loss.detach()),
        "accuracy": float(accuracy),
        "grad_norm_before_clip": float(grad_norm),
    }
```

The norm returned by `clip_grad_norm_` is the total norm before clipping. Logging it is useful: if it exceeds the threshold on nearly every update, clipping is no longer an occasional safety mechanism and may be masking a systematically unstable model or an excessive learning rate.

### Truncated BPTT over a continuous stream

```python
from __future__ import annotations

import torch
from torch import Tensor, nn


def detach_state(state: Tensor | tuple[Tensor, Tensor]) -> Tensor | tuple[Tensor, Tensor]:
    if isinstance(state, tuple):
        return tuple(component.detach() for component in state)  # type: ignore[return-value]
    return state.detach()


def train_stream_chunks(
    recurrent: nn.Module,
    head: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    chunks: list[tuple[Tensor, Tensor]],
) -> float:
    """Each chunk is (x_chunk, target_chunk); state is carried but graph is truncated."""
    state = None
    total_loss = 0.0

    recurrent.train()
    head.train()

    for x_chunk, target_chunk in chunks:
        optimizer.zero_grad(set_to_none=True)

        output, state = recurrent(x_chunk, state)  # works for nn.GRU/nn.LSTM signatures
        logits = head(output)
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            target_chunk.reshape(-1),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(recurrent.parameters()) + list(head.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        state = detach_state(state)
        total_loss += float(loss.detach())

    return total_loss / max(len(chunks), 1)
```

A production implementation should also reset selected batch entries when individual streams end. Detaching every entry is a graph operation; resetting is a semantic operation. They solve different problems.

## Practical Diagnostics

A recurrent model can fail without producing a Python exception. The following measurements are more informative than a single validation score.

### Gradient norms by time distance

Attach hooks to hidden states or use `torch.autograd.grad` to measure $\|\partial L/\partial h_t\|$. Plot the norm against distance from the supervised output. Exponential decay reveals vanishing credit assignment; sharp spikes reveal unstable transitions. The profile is task- and input-dependent, so averaging over examples should be accompanied by quantiles rather than only a mean.

### Gate distributions and saturation

For LSTM and GRU models, inspect histograms of gates over time and data. Values concentrated at zero or one imply nearly discrete routing and very small sigmoid derivatives. This may be exactly what a solved task requires, or it may indicate premature saturation that prevents further learning. Context matters: a forget gate near one can preserve memory, while an input gate stuck near zero can block all new evidence.

### State norms and drift

Track $\|h_t\|_2$ and, for an LSTM, $\|c_t\|_2$. Exploding state norms, rapidly increasing cell means, or a large fraction of $\tanh(c_t)$ values at $\pm1$ indicate drift or saturation. Hidden norms that collapse immediately toward one common value indicate excessive contraction.

### Effective memory tests

Do not infer memory length only from architecture. Perturb or erase an early input and measure how later outputs change. Train probing classifiers to recover past symbols from current states. Evaluate performance while increasing the delay between informative input and target. A task may have length $1000$ while requiring only a local dependency of length $5$; conversely, good average accuracy can hide complete failure on the rare examples that require long memory.

### Padding invariance

Append additional padding to a batch and verify that predictions remain unchanged. This simple test catches incorrect last-state selection, missing masks, and normalization leakage. For packed models, compare packed and carefully masked padded implementations on the same short examples.

### Causality tests

Shift or remove future inputs and verify that a purportedly causal model's earlier outputs do not change. Run preprocessing with only prefix information. In time-series evaluation, move the split date and confirm that feature statistics and data selection do not reach across it.

### Throughput and latency

Recurrent networks may have modest floating-point operation counts yet low hardware utilization because each time step waits for the preceding state. Measure wall-clock throughput, not only parameter count. For streaming systems, report per-step latency and state-memory cost. For offline systems, compare against temporal convolutions, attention, and state-space models under the same sequence length and hardware.

## Structural Limitations of Recurrence

Gating addresses a major optimization barrier, but it does not remove the architecture's fundamental constraints.

### Sequential depth

The state $h_t$ cannot be computed before $h_{t-1}$. Across time, the critical path has length $T$. Operations inside one step can be parallelized over batch and hidden dimensions, but time steps remain ordered. This limits training throughput on parallel accelerators.

Parallel prefix algorithms can accelerate special associative recurrences, and linear state-space models can exploit convolution or scan formulations. A general nonlinear LSTM transition is not associative in a form that removes the dependency without changing the computation.

### Fixed-dimensional compression

Every prefix is compressed into $H$ numbers. For tasks requiring exact access to an unbounded number of independent past facts, no fixed finite-dimensional continuous state can preserve everything robustly. Increasing $H$ postpones the bottleneck; it does not eliminate it. External memory, attention over stored states, retrieval, or structured state representations provide alternative access patterns.

### Long path length

Even when a gate preserves one cell coordinate, information used by a final nonlinear decision may still travel through many transformations. Attention creates a path of constant graph length between two positions within one layer. This does not automatically make attention statistically superior, but it changes both optimization and parallelism.

### Difficulty of selective retrieval

A recurrent state updates through every intervening input. To recover one specific event from far in the past, the model must have encoded it into a state coordinate or distributed pattern that survived all later updates. Attention instead retains a set of token representations and performs content-based retrieval at query time. The trade-off is memory and quadratic interaction cost for dense attention.

### Error accumulation in generation

An autoregressive recurrent decoder feeds its own outputs back into future steps. Approximation errors can alter the state distribution and compound. This is not unique to RNNs—autoregressive Transformers have the same feedback at the token level—but recurrence adds another hidden dynamical state whose errors also accumulate.

## Recurrence After the Transformer

The Transformer displaced RNNs in many large-scale language tasks because self-attention permits parallel processing of training tokens and direct interactions between distant positions. This historical change should not be mistaken for a theorem that recurrence is obsolete.

RNNs remain attractive when inference is naturally streaming, state memory must be constant in sequence length, latency is measured per arriving sample, or datasets are too small to justify a large attention model. Speech enhancement, sensor processing, control, anomaly detection, and embedded systems still contain many such regimes. Hybrid architectures combine recurrence with convolution or attention rather than treating them as exclusive choices.

Modern state-space sequence models return to the recurrence

<div class="display-equation">
$$
h_t=Ah_{t-1}+Bx_t,
\qquad
y_t=Ch_t+Dx_t,
$$
</div>

but impose structure that allows the same linear system to be evaluated as a convolution during training and as a recurrence during streaming inference. Nonlinear gating and input-dependent parameters can then be added around the structured transition. The enduring lesson is not one particular cell. It is the idea that sequence modeling is the design of state evolution, memory time scales, information routing, and trainable credit assignment.

## Summary

A recurrent neural network replaces a fixed-length stateless map with a shared state transition. The hidden state compresses a prefix into a task-dependent statistic, giving the model a natural streaming interface and a parameter count independent of sequence length.

Unrolling recurrence exposes a depth-$T$ computation graph. BPTT is ordinary reverse-mode differentiation on that graph, with gradients for shared parameters summed over time. Long-range credit assignment is governed by products of state-dependent Jacobians. Vanishing and exploding gradients depend on singular values, activation derivatives, non-normal transients, and the trajectory—not only on one recurrent eigenvalue.

LSTM and GRU cells introduce additive interpolation paths with learned gates. These paths make near-neutral memory directions easier to represent and optimize. They do not guarantee infinite memory, eliminate every unstable derivative, or remove the finite-state bottleneck. Their gates should be understood as learned dynamical controls rather than literal human-readable memory switches.

Correct implementation requires more than selecting `nn.LSTM`. Variable lengths must be packed or masked, final states must be indexed at true sequence ends, hidden tensor shapes must account for layers and directions, continuous streams need explicit state identity and reset rules, and truncated BPTT must detach graphs without confusing detachment with semantic reset.

The central question in sequential learning is not merely whether a network contains recurrence. It is which information the state preserves, how long each state direction remains useful, how credit reaches the transition that created it, and whether the resulting computation matches the causal and computational constraints of deployment.

## Exercises

1. For the linear recurrence $h_t=Wh_{t-1}+Ux_t$, derive $h_t$ explicitly as a sum of transformed inputs and an initial-state term. State conditions under which the initial-state contribution converges to zero.
2. Construct a $2\times2$ non-normal matrix with spectral radius below one whose power norm initially grows. Compute $A^t$ in closed form.
3. Derive all Elman RNN parameter gradients from the adjoint recursion, including the output head.
4. Show that a constant forget gate $f$ corresponds to an exponential moving average, and derive its effective time constant and half-life.
5. Under the GRU convention used in this chapter, identify the limiting behavior when $z_t\to0$, $z_t\to1$, $r_t\to0$, and $r_t\to1$.
6. Modify the delayed-recall experiment by inserting random distractors. Measure accuracy as a function of delay and hidden width.
7. Implement a masked Elman recurrence without packed sequences and verify numerically that appending padding leaves the final valid state unchanged.
8. Compare independent-time-step dropout with locked dropout on a synthetic memory task. Explain the difference using state-transition noise.
9. For a bidirectional two-layer LSTM, write the exact shapes of the output, $h_n$, and $c_n$ under both values of `batch_first`.
10. Design a causal evaluation protocol for a multivariate time series with overlapping windows, per-device normalization, and delayed labels. Identify every possible leakage path.

## References

1. Jeffrey L. Elman, [Finding Structure in Time](https://doi.org/10.1207/s15516709cog1402_1), *Cognitive Science*, 1990.
2. Paul J. Werbos, [Backpropagation Through Time: What It Does and How to Do It](https://doi.org/10.1109/5.58337), *Proceedings of the IEEE*, 1990.
3. Yoshua Bengio, Patrice Simard, and Paolo Frasconi, [Learning Long-Term Dependencies with Gradient Descent Is Difficult](https://doi.org/10.1109/72.279181), *IEEE Transactions on Neural Networks*, 1994.
4. Sepp Hochreiter and Jürgen Schmidhuber, [Long Short-Term Memory](https://doi.org/10.1162/neco.1997.9.8.1735), *Neural Computation*, 1997.
5. Felix A. Gers, Jürgen Schmidhuber, and Fred Cummins, [Learning to Forget: Continual Prediction with LSTM](https://doi.org/10.1162/089976600300015015), *Neural Computation*, 2000.
6. Razvan Pascanu, Tomas Mikolov, and Yoshua Bengio, [On the Difficulty of Training Recurrent Neural Networks](https://proceedings.mlr.press/v28/pascanu13.html), *ICML*, 2013.
7. Kyunghyun Cho et al., [Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078), 2014.
8. Ilya Sutskever, Oriol Vinyals, and Quoc V. Le, [Sequence to Sequence Learning with Neural Networks](https://proceedings.neurips.cc/paper/2014/hash/a14ac55a4f27472c5d894ec1c3c743d2-Abstract.html), *NeurIPS*, 2014.
9. Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio, [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473), 2014.
10. Martin Arjovsky, Amar Shah, and Yoshua Bengio, [Unitary Evolution Recurrent Neural Networks](https://proceedings.mlr.press/v48/arjovsky16.html), *ICML*, 2016.
11. Quoc V. Le, Navdeep Jaitly, and Geoffrey E. Hinton, [A Simple Way to Initialize Recurrent Networks of Rectified Linear Units](https://arxiv.org/abs/1504.00941), 2015.
12. Stephen Merity, Nitish Shirish Keskar, and Richard Socher, [Regularizing and Optimizing LSTM Language Models](https://openreview.net/forum?id=SyyGPP0TZ), *ICLR*, 2018.
13. Albert Gu, Karan Goel, and Christopher Ré, [Efficiently Modeling Long Sequences with Structured State Spaces](https://openreview.net/forum?id=uYLFoz1vlAC), *ICLR*, 2022.
14. PyTorch, [Recurrent Layers Documentation](https://pytorch.org/docs/stable/nn.html#recurrent-layers), accessed 2026.
