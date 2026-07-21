---
title: "Convolutional Layers and Spatial Learning Systems"
subtitle: "Equivariance, Locality, Multiscale Representation, Backpropagation, and PyTorch"
summary: "Beginning with the failure of flattened image models, this chapter develops convolution as the canonical linear translation-equivariant operator, derives its forward and backward maps, studies receptive fields, sampling, frequency behavior, architectural evolution, dense prediction, metric learning, and a shape-safe PyTorch implementation."
description: "Beginning with the failure of flattened image models, this chapter develops convolution as the canonical linear translation-equivariant operator, derives its forward and backward maps, studies receptive fields, sampling, frequency behavior, architectural evolution, dense prediction, metric learning, and a shape-safe PyTorch implementation."
date: 2026-07-20
lastmod: 2026-07-20
weight: 80
libraryFolder: "ai-foundations"
libraryFolderName: "人工智能基础"
libraryFolderColor: 0
tags: ["Machine Learning", "Convolutional Neural Networks", "Computer Vision", "Equivariance", "PyTorch"]
draft: false
ShowToc: false
hideMeta: true
---
## Convolutional Layers and Spatial Learning Systems

## Introduction

A fully connected network can classify a small handwritten digit after the image has been flattened into a vector. That success is useful precisely because it exposes the limitation of the representation. A $28\times28$ grayscale image occupies only $784$ coordinates, and a ten-class linear head therefore needs fewer than eight thousand weights. A color image of size $224\times224$, however, already contains $150{,}528$ scalar measurements. Connecting every input coordinate to a modest hidden layer of width $4096$ would require more than six hundred million weights before the network had performed a single nonlinear transformation. The difficulty is not merely that the number is large. Flattening tells the model that a red edge at the upper left and the same red edge one pixel to the right are unrelated events requiring unrelated parameters. It discards the very structure that makes images statistically learnable.

A convolutional layer repairs this mistake by changing the hypothesis class before optimization begins. It assumes that nearby pixels should first interact locally, that the same local pattern may matter at many spatial positions, and that a translation of the input should induce a corresponding translation of intermediate features. These assumptions are not universal laws of vision. They are inductive biases: restrictions that reduce the number of functions the learner may represent in exchange for better statistical and computational efficiency on spatial data. The success of convolutional networks follows less from the arithmetic of a sliding window than from the compatibility between those restrictions and the regularities of natural images.

The usual introduction describes convolution as a small matrix sliding across an image. That picture is correct but incomplete. A convolution is simultaneously a structured linear operator, a symmetry-respecting map, a sparse matrix with tied coefficients, a local message-passing rule, and, under periodic boundary conditions, a Fourier multiplier. Its backward pass is the adjoint of its forward map. A transposed convolution is therefore not an inverse operation but a particular adjoint operator. Pooling and stride are not harmless ways to obtain invariance; they are sampling operations that can destroy exact equivariance through aliasing. Receptive field size is not just a count of layers, because the theoretical set of reachable pixels can be much larger than the region that exerts appreciable influence. Once these distinctions are made explicit, many architectural choices that otherwise look like folklore become consequences of linear algebra, harmonic analysis, and optimization.

This chapter develops that view from first principles. The starting point is the translation group acting on a lattice. From it we derive convolution as the canonical linear equivariant operator, extend the formula to channels, stride, padding, dilation, and groups, and then derive gradients with respect to inputs, kernels, and biases. The same operator is examined in the spatial, matrix, and frequency domains. Pooling, anti-aliasing, upsampling, transposed convolution, residual blocks, depthwise separable kernels, detection, segmentation, and metric learning then appear as variations on a common problem: how should information move across position, scale, and channel while preserving the structure needed by the task?

## From a Dense Map to a Spatial Operator

Let an image be represented by a tensor $X\in\mathbb{R}^{C_{\mathrm{in}}\times H\times W}$. Flattening produces a vector $x\in\mathbb{R}^{C_{\mathrm{in}}HW}$, after which an affine layer computes $z=Ax+b$. If the output is itself a feature map $Y\in\mathbb{R}^{C_{\mathrm{out}}\times H'\times W'}$, then the unrestricted matrix $A$ contains

<div class="display-equation">
$$
C_{\mathrm{out}}H'W'\,C_{\mathrm{in}}HW
$$
</div>

weights. Every output site is allowed to depend independently on every input site. This is the most general linear map between the two tensor spaces, but generality is not free. It spends parameters on long-range interactions that may be unnecessary in early layers, and it learns a separate response for every absolute position even when the underlying visual pattern is the same.

Consider a vertical edge represented locally by dark pixels on the left and bright pixels on the right. If that edge is shifted by one column, a dense layer sees a different collection of coordinates. Nothing in the parameterization forces the two responses to agree. A learner could eventually discover approximately repeated weights from data, but it would have to infer the same rule independently at many positions. Convolution builds the repetition into the operator. The learned object is not a separate coefficient for every pair of global coordinates. It is a small rule indexed by relative displacement, and the same rule is reused wherever the local neighborhood is available.

The reduction is dramatic. A standard convolution with $C_{\mathrm{in}}$ input channels, $C_{\mathrm{out}}$ output channels, and a $k_h\times k_w$ kernel contains

<div class="display-equation">
$$
C_{\mathrm{out}}\bigl(C_{\mathrm{in}}k_hk_w+1\bigr)
$$
</div>

parameters when each output channel has a bias. The number is independent of the image height and width. A $3\times3$ layer mapping $64$ channels to $128$ channels has $128(64\cdot9+1)=73{,}856$ parameters whether it is applied to a $32\times32$ feature map or a $2048\times2048$ one. The computational cost still grows with spatial resolution because the rule must be evaluated at more sites, and the activation memory may dominate the parameter memory, but the statistical object being learned remains small.

The difference can be summarized as a distinction between an arbitrary linear map and a structured linear map. The dense matrix has no preferred geometry. The convolutional matrix is sparse because each output depends on a local neighborhood, and it has tied entries because the coefficient associated with a relative displacement is reused across positions. Locality and parameter sharing are often presented as two unrelated tricks. They are better understood as two restrictions on the same operator: locality limits which relative displacements are allowed, while translation symmetry determines how the allowed coefficients are shared.

## Images as Functions on a Lattice

It is useful to remove array notation for a moment and view an image as a function. Let

<div class="display-equation">
$$
G=\mathbb{Z}_H\times\mathbb{Z}_W
$$
</div>

be a finite two-dimensional cyclic lattice. An input is a function $X:G\to\mathbb{R}^{C_{\mathrm{in}}}$, so $X(u)$ is a channel vector attached to spatial site $u\in G$. The cyclic convention identifies positions modulo $H$ and $W$. Real images are not periodic, but this temporary choice gives translations an exact group structure and allows the central theorem to be stated without boundary exceptions.

For a displacement $a\in G$, define the translation operator $\tau_a$ by

<div class="display-equation">
$$
(\tau_aX)(u)=X(u-a).
$$
</div>

The sign convention means that the content originally at $u-a$ appears at $u$ after translation by $a$. The family $\{\tau_a:a\in G\}$ satisfies $\tau_a\tau_b=\tau_{a+b}$ and $\tau_0=I$.

### Equivariance and invariance

A map $F$ from images to feature maps is translation equivariant when translating the input and then applying $F$ gives the same result as applying $F$ and translating the output:

<div class="display-equation">
$$
F(\tau_aX)=\tau_aF(X)
\qquad\text{for every }a\in G.
$$
</div>

Equivariance does not discard position. It preserves the relation between positions. If an edge detector activates at one location and the input edge moves, the activation should move with it. An invariant map instead satisfies

<div class="display-equation">
$$
F(\tau_aX)=F(X).
$$
</div>

A classifier that assigns the same category after a translation is intended to be approximately invariant, while an intermediate feature extractor is usually intended to be equivariant. Confusing the two leads to vague statements such as “convolution is translation invariant.” A stride-one convolution is equivariant under ideal boundary conditions. Invariance usually appears only after spatial aggregation, such as global averaging, or after a decision rule deliberately removes location.

The distinction is compositional. If $E$ is equivariant and $A$ is invariant, then $A\circ E$ is invariant:

<div class="display-equation">
$$
(A\circ E)(\tau_aX)
=A(\tau_aE(X))
=A(E(X)).
$$
</div>

This is the architectural logic behind an equivariant convolutional backbone followed by global pooling and a linear classifier. The backbone moves evidence with the object; the pooling stage discards the final position while retaining the accumulated evidence.

### Why linear translation-equivariant maps are convolutions

The strongest justification for convolution is not that sliding windows seem natural. On a homogeneous lattice, convolution is forced by linearity and translation equivariance.

**Theorem.** Let $T:(\mathbb{R}^{C_{\mathrm{in}}})^G\to(\mathbb{R}^{C_{\mathrm{out}}})^G$ be linear. Then $T$ commutes with every translation,

<div class="display-equation">
$$
T\tau_a=\tau_aT
\qquad\text{for all }a\in G,
$$
</div>

if and only if there exists a matrix-valued kernel $K:G\to\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}}$ such that

<div class="display-equation">
$$
(TX)(u)
=
\sum_{v\in G}K(v)X(u-v).
$$
</div>

**Proof.** For each input channel $c$, let $e_c\in\mathbb{R}^{C_{\mathrm{in}}}$ be the $c$th standard basis vector, and define the impulse $\delta_{0,c}$ to equal $e_c$ at the origin and zero elsewhere. Every image can be decomposed into translated impulses:

<div class="display-equation">
$$
X
=
\sum_{v\in G}\sum_{c=1}^{C_{\mathrm{in}}}
X_c(v)\,\tau_v\delta_{0,c}.
$$
</div>

By linearity and equivariance,

<div class="display-equation">
$$
TX
=
\sum_{v,c}X_c(v)\,T\tau_v\delta_{0,c}
=
\sum_{v,c}X_c(v)\,\tau_vT\delta_{0,c}.
$$
</div>

The response $T\delta_{0,c}$ completely determines how the operator reacts to an impulse in channel $c$. Define the kernel entries by $K_{:,c}(r)=(T\delta_{0,c})(r)$. Evaluating the preceding expression at position $u$ gives

<div class="display-equation">
$$
(TX)(u)
=
\sum_{v,c}X_c(v)K_{:,c}(u-v)
=
\sum_{r\in G}K(r)X(u-r),
$$
</div>

where the last equality uses $r=u-v$. This is convolution. Conversely, substituting a translated input into the convolution formula and changing variables immediately shows $T\tau_a=\tau_aT$. $\square$

The theorem changes the interpretation of weight sharing. If every position is to be treated as the same kind of position, a linear operator cannot assign unrelated coefficients to different absolute coordinates. Its coefficients may depend only on relative displacement. Convolution is therefore the coordinate form of a symmetry constraint. A dense layer becomes necessary when the task genuinely depends on absolute position or when the spatial domain is not homogeneous; otherwise, it spends degrees of freedom violating a symmetry that the data often approximately possess.

### Locality, support, and boundary conditions

The theorem permits a kernel supported on the entire lattice. Local convolution adds the assumption that $K(v)=0$ outside a small set $S$, usually a rectangle centered near the origin. Then

<div class="display-equation">
$$
(TX)(u)=\sum_{v\in S}K(v)X(u-v).
$$
</div>

Locality is not implied by translation symmetry. It is an additional prior that short-range interactions should be composed before long-range interactions. Natural images make this plausible because neighboring pixels are strongly correlated and elementary structures such as edges, corners, and textures can be detected locally. Deep composition then expands the range over which information interacts. The early layers need not decide the identity of an entire object; they only need to construct a representation from which later layers can do so.

Real implementations are defined on a finite rectangle rather than a torus. Padding specifies how the image is extended beyond that rectangle. Zero padding imagines that the unobserved exterior is black. Reflect padding mirrors values at the edge, replicate padding repeats boundary values, and circular padding restores the cyclic model used in the theorem. These choices are not mathematically neutral. Zero padding gives the network an implicit coordinate signal because boundary neighborhoods contain artificial zeros that never occur in the interior. Exact translation equivariance therefore fails when a translation changes which receptive fields interact with the boundary. Away from the boundary, a stride-one convolution remains equivariant to translations that keep all relevant neighborhoods inside the observed region. Circular padding preserves exact equivariance on the discrete torus, but it may impose an unrealistic seam on ordinary photographs.

The boundary issue matters because “convolutional” does not automatically mean “equivariant under every transformation encountered in code.” Cropping, resizing, padding, stride, finite precision, and nonlinear preprocessing can all alter the symmetry. Equivariance is a property of the complete operator under a specified group action, not a label inherited from the presence of a `Conv2d` module.

## Convolution, Cross-Correlation, and the Operator Used in Deep Learning

For integrable functions on the real line, continuous convolution is

<div class="display-equation">
$$
(f*g)(t)
=
\int_{-\infty}^{\infty}f(\tau)g(t-\tau)\,d\tau.
$$
</div>

For discrete two-dimensional signals, the corresponding operation is

<div class="display-equation">
$$
(X*K)(i,j)
=
\sum_{a}\sum_b X(a,b)K(i-a,j-b).
$$
</div>

The negative signs reverse the kernel in both spatial coordinates. Deep-learning libraries usually implement cross-correlation instead:

<div class="display-equation">
$$
(X\star K)(i,j)
=
\sum_a\sum_b X(i+a,j+b)K(a,b).
$$
</div>

The distinction is a $180^\circ$ rotation of the kernel. When every kernel coefficient is learned freely, the two parameterizations represent the same family of maps: a network that would learn $K$ under one convention can learn the rotated kernel under the other. This is why the community calls the operation convolution even though the forward implementation in libraries such as PyTorch is cross-correlation. The distinction must nevertheless be retained when discussing fixed filters, Fourier phases, or adjoints, because in those settings the orientation is no longer absorbed by relearning the coefficients.

### The full multi-channel operator

Let the batched input have shape

<div class="display-equation">
$$
X\in\mathbb{R}^{N\times C_{\mathrm{in}}\times H_{\mathrm{in}}\times W_{\mathrm{in}}},
$$
</div>

and let a convolutional layer have $C_{\mathrm{out}}$ output channels. Write the spatial kernel size as $(k_h,k_w)$, stride as $(s_h,s_w)$, dilation as $(d_h,d_w)$, and symmetric padding as $(p_h,p_w)$. With groups $G$, the input and output channels are partitioned into $G$ blocks. An output channel $o$ interacts only with the input-channel block $\mathcal C(o)$ assigned to the same group. After extending the input by padding, the forward map is

<div class="display-equation">
$$
Y_{n,o,i,j}
=
b_o+
\sum_{c\in\mathcal C(o)}
\sum_{a=0}^{k_h-1}
\sum_{b=0}^{k_w-1}
W_{o,c,a,b}\,
X^{\mathrm{pad}}_{n,c,\,i s_h+a d_h,\,j s_w+b d_w}.
$$
</div>

This single expression contains the common variants. Stride determines the displacement between adjacent output windows. Dilation determines the displacement between adjacent samples inside a window. Padding determines the coordinate system in which the first window is placed. Groups determine which channels may communicate. The bias is shared over every spatial site of one output channel, just as the kernel is shared over sites.

The effective spatial extent of a dilated kernel is

<div class="display-equation">
$$
k_h^{\mathrm{eff}}=d_h(k_h-1)+1,
\qquad
k_w^{\mathrm{eff}}=d_w(k_w-1)+1.
$$
</div>

Consequently, the output dimensions are

<div class="display-equation">
$$
H_{\mathrm{out}}
=
\left\lfloor
\frac{H_{\mathrm{in}}+2p_h-k_h^{\mathrm{eff}}}{s_h}
\right\rfloor+1,
\qquad
W_{\mathrm{out}}
=
\left\lfloor
\frac{W_{\mathrm{in}}+2p_w-k_w^{\mathrm{eff}}}{s_w}
\right\rfloor+1.
$$
</div>

A formula for shape should be read geometrically rather than memorized. After padding, the input has length $H_{\mathrm{in}}+2p_h$. A window occupies $k_h^{\mathrm{eff}}$ positions, so the coordinate of its left edge may range from zero through $H_{\mathrm{in}}+2p_h-k_h^{\mathrm{eff}}$. Sampling those valid starting coordinates every $s_h$ positions gives the floor and the final $+1$.

**A complete shape and cost calculation.** Suppose

<div class="display-equation">
$$
X\in\mathbb{R}^{32\times64\times56\times56},
$$
</div>

and the layer uses $C_{\mathrm{out}}=128$, a $3\times3$ kernel, stride $2$, padding $2$, dilation $2$, and groups $G=4$. The effective kernel is $2(3-1)+1=5$, so

<div class="display-equation">
$$
H_{\mathrm{out}}=W_{\mathrm{out}}
=
\left\lfloor\frac{56+4-5}{2}\right\rfloor+1
=28.
$$
</div>

Each output channel receives $64/4=16$ input channels. The weight tensor therefore has $128\cdot16\cdot3\cdot3=18{,}432$ entries, and the biases add $128$, for a total of $18{,}560$ parameters. Ignoring the bias additions, one forward pass performs approximately

<div class="display-equation">
$$
32\cdot28\cdot28\cdot128\cdot16\cdot3\cdot3
\approx 4.62\times10^8
$$
</div>

multiply-accumulate operations. The example illustrates three different notions of size. Parameter count depends on channels, groups, and kernel size. Arithmetic also depends on batch size and output area. Activation memory depends on tensor shapes and can dominate both, especially at high resolution.

### Parameter count is not computational cost

For a grouped convolution, the number of trainable parameters is

<div class="display-equation">
$$
P
=
C_{\mathrm{out}}
\left(
\frac{C_{\mathrm{in}}}{G}k_hk_w+1
\right),
$$
</div>

while the leading multiply-accumulate count is

<div class="display-equation">
$$
\operatorname{MACs}
=
NH_{\mathrm{out}}W_{\mathrm{out}}C_{\mathrm{out}}
\frac{C_{\mathrm{in}}}{G}k_hk_w.
$$
</div>

Neither expression alone predicts wall-clock speed. Memory layout, cache behavior, kernel-launch overhead, hardware-specific fused operations, arithmetic intensity, and the availability of optimized implementations all matter. A theoretically cheaper depthwise convolution can be less efficient than a denser convolution on hardware that is optimized for large matrix multiplications. Architectural efficiency must therefore be evaluated on the deployment system rather than inferred entirely from parameter counts or nominal floating-point operations.

## Convolution as Structured Linear Algebra

A convolutional layer is linear in its input when the weights are fixed. It can therefore be written as a matrix multiplication. In one dimension, a valid cross-correlation with kernel $(w_0,w_1,w_2)$ applied to $x=(x_0,\ldots,x_4)^\top$ is

<div class="display-equation">
$$
\begin{pmatrix}
y_0\\y_1\\y_2
\end{pmatrix}
=
\begin{pmatrix}
w_0&w_1&w_2&0&0\\
0&w_0&w_1&w_2&0\\
0&0&w_0&w_1&w_2
\end{pmatrix}
\begin{pmatrix}
x_0\\x_1\\x_2\\x_3\\x_4
\end{pmatrix}.
$$
</div>

The matrix is sparse, and its nonzero diagonals repeat. In one dimension it is Toeplitz; in two dimensions the corresponding matrix is block Toeplitz with Toeplitz blocks. Under circular padding it becomes block circulant with circulant blocks. This matrix view explains all of the familiar properties at once. Locality is sparsity. Weight sharing is repeated matrix entries. The backward map to the input is multiplication by the transpose. A transposed convolution is the same structured transpose applied as a forward layer.

The matrix is almost never formed explicitly. Its dimensions would be enormous, and most entries would be zero. Implementations instead exploit structure. One classical method is `im2col`, called `unfold` in PyTorch. Each local input patch is flattened into a column. If

<div class="display-equation">
$$
\mathcal P(X)
\in
\mathbb{R}^{(C_{\mathrm{in}}k_hk_w)\times(H_{\mathrm{out}}W_{\mathrm{out}})}
$$
</div>

contains those patch columns and

<div class="display-equation">
$$
W_{\mathrm{flat}}
\in
\mathbb{R}^{C_{\mathrm{out}}\times(C_{\mathrm{in}}k_hk_w)},
$$
</div>

then an ungrouped convolution can be written as

<div class="display-equation">
$$
Y_{\mathrm{flat}}
=
W_{\mathrm{flat}}\mathcal P(X)
+b\mathbf 1^\top.
$$
</div>

The spatial operator has become a dense matrix multiplication, allowing highly optimized general matrix-multiplication kernels to perform the main arithmetic. The price is that overlapping input values are copied into several patch columns, which may require substantial temporary memory. Modern libraries therefore choose among explicit unfolding, implicit GEMM, direct kernels, Winograd transforms, FFT-based methods, and hardware-specific fused algorithms. The mathematical layer is fixed; the algorithm used to evaluate it is an implementation decision.

The unfolded representation also clarifies the backward pass. If $\Delta_{\mathrm{flat}}$ is the upstream gradient, then

<div class="display-equation">
$$
\frac{\partial L}{\partial W_{\mathrm{flat}}}
=
\Delta_{\mathrm{flat}}\mathcal P(X)^\top,
\qquad
\frac{\partial L}{\partial \mathcal P(X)}
=
W_{\mathrm{flat}}^\top\Delta_{\mathrm{flat}}.
$$
</div>

The second result must be folded back into image coordinates. When patches overlap, several columns contribute to the same input pixel, so folding sums the contributions. This is exactly the gradient accumulation required by the chain rule.

## Important Factorizations and Variants

A standard convolution couples spatial mixing and channel mixing in one tensor $W_{o,c,a,b}$. Many architectures gain efficiency or flexibility by factoring that tensor in different ways.

### Pointwise convolution

A $1\times1$ convolution has

<div class="display-equation">
$$
Y_{:,i,j}=AX_{:,i,j}+b,
$$
</div>

where the same matrix $A\in\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}}$ is applied independently at every spatial location. It has no spatial receptive field beyond one site, but it performs unrestricted channel mixing and remains translation equivariant. Calling it merely a dimension-reduction device understates its role. Placed between nonlinearities, pointwise convolutions learn new channel combinations at every site, implement bottlenecks, and make it possible to separate expensive spatial processing from channel communication.

Suppose a $5\times5$ convolution maps $C$ channels to $M$ channels. Its leading parameter cost is $25CM$. If a pointwise layer first reduces the channel count to $R$, followed by a $5\times5$ convolution from $R$ to $M$, the cost becomes

<div class="display-equation">
$$
CR+25RM.
$$
</div>

When $R\ll C$, this can be far smaller while preserving a nonlinear path if an activation is inserted between the two maps. The Inception and residual bottleneck families rely heavily on this arithmetic.

### Grouped and depthwise convolution

Grouped convolution partitions channels into $G$ independent blocks. In matrix language, the channel-mixing structure is block diagonal. At $G=1$, every output channel sees every input channel. At the opposite extreme $G=C_{\mathrm{in}}$ with one spatial filter per input channel, the operation is depthwise convolution. A depthwise layer performs spatial filtering independently in each channel; a following $1\times1$ pointwise layer then mixes channels.

For equal input and output spatial dimensions, a standard $k\times k$ convolution from $C_{\mathrm{in}}$ to $C_{\mathrm{out}}$ has leading cost

<div class="display-equation">
$$
k^2C_{\mathrm{in}}C_{\mathrm{out}}.
$$
</div>

A depthwise convolution followed by a pointwise convolution has cost

<div class="display-equation">
$$
k^2C_{\mathrm{in}}+C_{\mathrm{in}}C_{\mathrm{out}}.
$$
</div>

The ratio is

<div class="display-equation">
$$
\frac{k^2C_{\mathrm{in}}+C_{\mathrm{in}}C_{\mathrm{out}}}
{k^2C_{\mathrm{in}}C_{\mathrm{out}}}
=
\frac{1}{C_{\mathrm{out}}}+\frac{1}{k^2}.
$$
</div>

For $k=3$ and large $C_{\mathrm{out}}$, the spatial-channel factorization uses roughly one ninth of the multiply-accumulates of a standard convolution. The saving comes from a stronger restriction on the kernel tensor: spatial patterns are learned within channels first, and cross-channel combinations are learned separately. This restriction may reduce expressivity per layer, but additional depth and nonlinearities can recover substantial capacity at much lower cost.

### Spatial separability is a different factorization

Depthwise separability should not be confused with spatial separability. A two-dimensional single-channel kernel is spatially separable when it has rank one,

<div class="display-equation">
$$
K=uv^\top,
$$
</div>

so a $k\times k$ filter can be evaluated as a $k\times1$ filter followed by a $1\times k$ filter. A Gaussian kernel is separable; a general learned kernel is not. Depthwise separability instead separates spatial filtering from channel mixing. The two ideas factor different axes of the four-dimensional convolutional weight tensor and may be combined.

### Dilated and deformable sampling

Dilation inserts gaps between sampled kernel positions. A $3\times3$ kernel with dilation $2$ has an effective $5\times5$ footprint while retaining only nine coefficients. This expands the theoretical receptive field without reducing feature-map resolution, which is valuable in dense prediction. The trade-off is that a fixed dilation samples a sparse lattice. Repeated layers with the same dilation can create a gridding pattern in which neighboring output units depend on disjoint subsets of the input. Mixing dilation rates or inserting ordinary convolutions helps reconnect those subsets.

A deformable convolution goes further by learning offsets $\Delta p_{i,j,a,b}$ for the sampling positions:

<div class="display-equation">
$$
Y_{o,i,j}
=
\sum_{c,a,b}
W_{o,c,a,b}\,
X_c\bigl(p_{i,j}+p_{a,b}+\Delta p_{i,j,a,b}\bigr).
$$
</div>

The learned coordinates are usually nonintegral, so $X$ is evaluated by bilinear interpolation. Ordinary convolution assumes the same rigid stencil everywhere; deformable convolution lets the stencil bend around object geometry. The price is a weaker exact symmetry, additional parameters and interpolation cost, and a more complicated optimization problem. It is best understood not as a replacement for convolution but as a controlled relaxation of its fixed geometric prior.

## Receptive Fields and Hierarchical Composition

A single local layer cannot represent a relationship between distant pixels. Depth changes this. The receptive field of a unit is the set of input coordinates that can influence it through the computation graph. Two quantities are needed to track receptive fields correctly. Let $r_\ell$ be the receptive-field size at layer $\ell$, and let $j_\ell$ be the jump, sometimes called the effective stride: the distance in input coordinates between adjacent units at layer $\ell$. Starting with $r_0=j_0=1$, a layer with stride $s_\ell$ and effective kernel size $k_\ell^{\mathrm{eff}}=d_\ell(k_\ell-1)+1$ obeys

<div class="display-equation">
$$
j_\ell=j_{\ell-1}s_\ell,
\qquad
r_\ell=r_{\ell-1}+\bigl(k_\ell^{\mathrm{eff}}-1\bigr)j_{\ell-1}.
$$
</div>

The jump recurrence records how coarse the current coordinate system has become. The receptive-field recurrence then measures the new window in units of the original input. Omitting $j_{\ell-1}$ is a common error: a $3\times3$ convolution after a stride-two layer expands the receptive field by four input pixels, not two.

Consider the sequence

<div class="display-equation">
$$
\operatorname{Conv}(3,1)
\to
\operatorname{Conv}(3,1)
\to
\operatorname{Pool}(2,2)
\to
\operatorname{Conv}(3,1)
\to
\operatorname{Conv}(3,1).
$$
</div>

The first convolution gives $(r,j)=(3,1)$, the second gives $(5,1)$, pooling gives $(6,2)$, the next convolution gives $(10,2)$, and the final one gives $(14,2)$. An output unit after the final convolution can therefore depend on a $14\times14$ region of the original image, while adjacent output units are centered two input pixels apart. This bookkeeping becomes indispensable in detection and segmentation, where feature maps must be aligned with anchors, regions, or output pixels.

Several small kernels can replace one large kernel. Two stride-one $3\times3$ layers have a $5\times5$ theoretical receptive field; three have a $7\times7$ field. If all channel widths equal $C$, one $5\times5$ layer contains $25C^2$ spatial-channel weights, whereas two $3\times3$ layers contain $18C^2$. More importantly, the two-layer construction can place a nonlinearity and normalization between the convolutions. The resulting function class is not merely a cheaper parameterization of one linear $5\times5$ filter. It is a deeper nonlinear computation with the same nominal spatial reach. This observation motivated the regular stacks of small kernels in VGG-style networks and remains a useful design principle even when modern blocks use depthwise or bottleneck factorizations.

### The theoretical and effective receptive fields are different

The theoretical receptive field asks whether a path exists. It does not ask how much influence travels along that path. In a deep network, the derivative of one output with respect to input pixels often concentrates near the center of the theoretical field. A simple one-dimensional model explains why. Repeatedly convolving with a three-tap all-ones kernel produces coefficients of

<div class="display-equation">
$$
(z^{-1}+1+z)^L.
$$
</div>

The coefficient at a displacement counts the number of computational paths reaching that displacement. Central positions have many more paths than extreme positions. After normalization, the distribution approaches a Gaussian by the central limit mechanism, so the nominal interval grows linearly with depth while most of the mass remains concentrated in a smaller central region. Random learned weights, nonlinear gates, residual paths, and normalization alter the exact profile, but the qualitative conclusion remains: pixels near the boundary of the theoretical receptive field may have negligible practical influence.

This smaller region is called the effective receptive field. It can be measured by backpropagating from a selected unit to the input and inspecting gradient magnitudes. Increasing theoretical size with dilation or depth does not guarantee that the model will use distant context. Auxiliary losses, attention, multiscale fusion, larger kernels, architectural shortcuts, and task structure all affect whether long-range paths receive useful learning signals. A receptive-field calculation is therefore a necessary geometric check, not a complete account of information flow.

The same distinction explains why deeper layers often look increasingly semantic without becoming completely locationless. A unit may have a theoretical field covering nearly the entire image, yet its strongest evidence may still come from a compact texture or object part. Classification training rewards whatever predictive features reduce the loss, not a prescribed progression from edges to parts to whole objects. The familiar hierarchy is a frequent empirical pattern, not a theorem guaranteed by convolution alone.

## Backpropagation Through a Convolution

Let the forward map be the ungrouped, padded, strided, dilated cross-correlation

<div class="display-equation">
$$
Y_{n,o,i,j}
=
b_o+
\sum_{c=0}^{C_{\mathrm{in}}-1}
\sum_{a=0}^{k_h-1}
\sum_{b=0}^{k_w-1}
W_{o,c,a,b}
X^{\mathrm{pad}}_{n,c,\,i s_h+a d_h,\,j s_w+b d_w}.
$$
</div>

Let

<div class="display-equation">
$$
\Delta_{n,o,i,j}
=
\frac{\partial L}{\partial Y_{n,o,i,j}}
$$
</div>

be the upstream gradient of a scalar loss. Every derivative follows from asking where a particular parameter or input coordinate appears in the forward sum.

### The bias gradient

The bias $b_o$ is added once at every batch and spatial coordinate in output channel $o$. Its derivative is therefore the total upstream signal in that channel:

<div class="display-equation">
$$
\frac{\partial L}{\partial b_o}
=
\sum_{n=0}^{N-1}
\sum_{i=0}^{H_{\mathrm{out}}-1}
\sum_{j=0}^{W_{\mathrm{out}}-1}
\Delta_{n,o,i,j}.
$$
</div>

The sum is not an implementation accident. It is the consequence of parameter sharing. One scalar influenced many outputs, so its gradient accumulates the contributions from all of them.

### The kernel gradient

A coefficient $W_{o,c,a,b}$ multiplies the input sampled at offset $(a d_h,b d_w)$ in every output window. Differentiating gives

<div class="display-equation">
$$
\frac{\partial L}{\partial W_{o,c,a,b}}
=
\sum_{n,i,j}
\Delta_{n,o,i,j}
X^{\mathrm{pad}}_{n,c,\,i s_h+a d_h,\,j s_w+b d_w}.
$$
</div>

The expression is another correlation: align the upstream gradient with every input patch and sum over batch and location. In unfolded form, it is the matrix product $\Delta_{\mathrm{flat}}\mathcal P(X)^\top$. This is why convolutional training can reuse the same computational primitives as the forward pass.

### The input gradient

An input value may participate in many overlapping windows and may affect every output channel. Its gradient is the sum over all paths in which it appears. A compact formula uses indicator functions:

<div class="display-equation">
$$
\frac{\partial L}{\partial X^{\mathrm{pad}}_{n,c,r,t}}
=
\sum_{o,i,j,a,b}
\Delta_{n,o,i,j}W_{o,c,a,b}
\mathbf 1\{r=i s_h+a d_h\}
\mathbf 1\{t=j s_w+b d_w\}.
$$
</div>

Only index combinations whose forward sampling location equals $(r,t)$ contribute. For stride one and dilation one, the formula can be reindexed into a full convolution of $\Delta$ with spatially reversed kernels. For larger stride, zeros must effectively be inserted between adjacent upstream-gradient samples before the kernel is applied. Cropping then removes the gradient associated with artificial padded coordinates.

This expression reveals an important difference between the weight gradient and the input gradient. The weight gradient sums evidence over all spatial uses of one shared coefficient. The input gradient redistributes output error through every kernel placement that touched one input coordinate. The former learns a reusable local rule; the latter tells the previous layer how changing each activation would affect the loss.

### The adjoint operator

The cleanest description uses inner products. With the bias omitted, let $C_W$ denote the linear convolutional map from $X$ to $Y$. Its adjoint $C_W^*$ is defined by

<div class="display-equation">
$$
\langle C_WX,\Delta\rangle_F
=
\langle X,C_W^*\Delta\rangle_F
$$
</div>

for every $X$ and $\Delta$, where $\langle A,B\rangle_F=\sum A_{i}B_i$ is the Frobenius inner product. The chain rule for a scalar loss gives

<div class="display-equation">
$$
\nabla_XL=C_W^*\Delta.
$$
</div>

If $C_W$ were represented by the sparse Toeplitz matrix $C$, then $C_W^*$ would simply be $C^\top$. A transposed convolution applies this transpose-shaped operator in the forward direction. It is called transposed because of this matrix relation, not because it undoes convolution. Unless the original operator is square, one-to-one, and well conditioned, no inverse need exist. Stride in particular discards information, so many distinct inputs can produce the same output.

The adjoint identity is more than terminology. It provides a decisive implementation test. For randomly generated tensors of compatible shapes, a convolution and its proposed transpose should satisfy

<div class="display-equation">
$$
\bigl|\langle C_WX,Z\rangle_F-\langle X,C_W^*Z\rangle_F\bigr|
$$
</div>

up to floating-point error. This catches kernel flips, padding mistakes, and output-shape mismatches that can survive ordinary shape tests.

### A complete numerical backward pass

Consider a single-channel input and a $2\times2$ kernel:

<div class="display-equation">
$$
X=
\begin{pmatrix}
1&2&0\\
3&1&2\\
0&1&4
\end{pmatrix},
\qquad
W=
\begin{pmatrix}
1&-1\\
2&0
\end{pmatrix}.
$$
</div>

Use stride one, no padding, no bias, and the cross-correlation convention. The forward output is

<div class="display-equation">
$$
Y=
\begin{pmatrix}
1\cdot1+2(-1)+3\cdot2+1\cdot0
&
2\cdot1+0(-1)+1\cdot2+2\cdot0\\
3\cdot1+1(-1)+0\cdot2+1\cdot0
&
1\cdot1+2(-1)+1\cdot2+4\cdot0
\end{pmatrix}
=
\begin{pmatrix}
5&4\\
2&1
\end{pmatrix}.
$$
</div>

Let $L=\frac12\|Y\|_F^2$. Then the upstream gradient is $\Delta=Y$, and $L=23$. The bias gradient, had a shared bias been present, would equal $5+4+2+1=12$. For the kernel, each upstream value multiplies the input patch that produced it. Summing those contributions gives

<div class="display-equation">
$$
\nabla_WL
=
5\begin{pmatrix}1&2\\3&1\end{pmatrix}
+4\begin{pmatrix}2&0\\1&2\end{pmatrix}
+2\begin{pmatrix}3&1\\0&1\end{pmatrix}
+1\begin{pmatrix}1&2\\1&4\end{pmatrix}
=
\begin{pmatrix}
20&14\\
20&19
\end{pmatrix}.
$$
</div>

For the input gradient, place a scaled copy of $W$ at the location of every output window and add overlaps:

<div class="display-equation">
$$
\nabla_XL
=
\begin{pmatrix}
5&-5&0\\10&0&0\\0&0&0
\end{pmatrix}
+
\begin{pmatrix}
0&4&-4\\0&8&0\\0&0&0
\end{pmatrix}
+
\begin{pmatrix}
0&0&0\\2&-2&0\\4&0&0
\end{pmatrix}
+
\begin{pmatrix}
0&0&0\\0&1&-1\\0&2&0
\end{pmatrix}
=
\begin{pmatrix}
5&-1&-4\\
12&7&-1\\
4&2&0
\end{pmatrix}.
$$
</div>

Every entry has a traceable origin. The center input participates in all four output windows and therefore accumulates four gradient paths. Corner inputs participate in only one. This unequal overlap near boundaries is one reason padding conventions affect optimization as well as forward values.

### Gradient checking

Automatic differentiation should not remove the habit of verification. For a scalar-valued implementation $L(W)$, a directional finite-difference test compares

<div class="display-equation">
$$
\frac{L(W+\varepsilon V)-L(W-\varepsilon V)}{2\varepsilon}
$$
</div>

with

<div class="display-equation">
$$
\langle\nabla_WL,V\rangle_F.
$$
</div>

The directional test is often more efficient and numerically stable than checking every coordinate separately. In double precision and away from nondifferentiable points, the discrepancy should decrease with $\varepsilon$ until floating-point cancellation begins to dominate. A failed check may indicate an incorrect forward map, because the backward derivative can be internally consistent with the wrong computation. Shape assertions, a hand-calculated example, an adjoint test, and a finite-difference test examine different failure modes and should be treated as complementary.

## Initialization, Conditioning, and Gradient Flow

The kernel shape changes the scale at which signals and gradients propagate. For an ungrouped convolution, one output preactivation is approximately a sum of

<div class="display-equation">
$$
\operatorname{fan}_{\mathrm{in}}=C_{\mathrm{in}}k_hk_w
$$
</div>

terms. With $G$ groups, the relevant fan-in is $(C_{\mathrm{in}}/G)k_hk_w$. If the inputs and weights are centered and approximately independent, then

<div class="display-equation">
$$
\operatorname{Var}(Y_{o,i,j})
\approx
\operatorname{fan}_{\mathrm{in}}
\operatorname{Var}(W)
\operatorname{Var}(X).
$$
</div>

For a linear or symmetric activation, choosing $\operatorname{Var}(W)\approx1/\operatorname{fan}_{\mathrm{in}}$ preserves variance in this simplified model. For ReLU, roughly half of a centered symmetric signal is set to zero, motivating He initialization,

<div class="display-equation">
$$
\operatorname{Var}(W)
\approx
\frac{2}{\operatorname{fan}_{\mathrm{in}}}.
$$
</div>

The derivation is approximate. Overlapping patches are correlated, natural-image pixels are far from independent, normalization alters distributions, and residual addition changes the variance recurrence. The value of the calculation is not that it predicts every activation exactly, but that it prevents signal scale from changing exponentially before learning has begun.

The same concern appears in reverse. Each input activation receives gradient contributions from multiple output channels and kernel placements. A fan-out calculation controls backward variance. Xavier initialization compromises between fan-in and fan-out, while residual architectures often introduce additional scaling so that a sum of many branches does not grow without bound. In very deep systems, initialization, normalization, residual parameterization, and optimizer scale cannot be designed independently; together they determine the spectrum of the input-output Jacobian and therefore the conditioning of learning.

### Kernel norm and operator norm are not the same

A small Frobenius norm of the kernel does not directly imply a small Lipschitz constant for the convolutional layer. The operator reuses the kernel at overlapping positions, and different channels can reinforce one another. A crude form of Young's convolution inequality gives, in the single-channel case,

<div class="display-equation">
$$
\|K*X\|_2
\leq
\|K\|_1\|X\|_2,
$$
</div>

so $\|K\|_1$ is an upper bound on the induced $\ell_2$ gain, but it is not generally tight. Under periodic boundary conditions, the exact answer is obtained in the frequency domain. This distinction matters for spectral normalization, robustness, and stability analysis. Regularizing individual coefficients is not automatically equivalent to regularizing the spatial operator they define.

Residual blocks change the Jacobian from $J_F$ to $I+J_F$. If a block computes

<div class="display-equation">
$$
x_{\ell+1}=x_\ell+F_\ell(x_\ell),
$$
</div>

then its local derivative is

<div class="display-equation">
$$
\frac{\partial x_{\ell+1}}{\partial x_\ell}
=I+J_{F_\ell}(x_\ell).
$$
</div>

The identity term supplies a direct gradient path, but it does not guarantee perfect conditioning: products of $I+J_{F_\ell}$ can still grow or contract, and interactions with normalization and branch scale matter. Residual parameterization makes the identity map easy to represent and often keeps early optimization near a well-conditioned reference map. Its effect is stronger and more precise than the loose claim that skip connections simply “prevent vanishing gradients.”

## Convolution in the Frequency Domain

Spatial convolution becomes multiplication after a Fourier transform. On a cyclic grid, let $\widehat X(\omega)$ and $\widehat K(\omega)$ denote the discrete Fourier transforms. For a single-channel circular convolution,

<div class="display-equation">
$$
\widehat{K*X}(\omega)
=
\widehat K(\omega)\widehat X(\omega).
$$
</div>

The block-circulant convolution matrix is diagonalized by the discrete Fourier basis. Each frequency is an eigenvector, and $\widehat K(\omega)$ is the corresponding eigenvalue. A convolution therefore does not mix spatial frequencies in the linear single-channel setting; it rescales and phase-shifts each one.

With multiple channels, the transform at every frequency is a matrix:

<div class="display-equation">
$$
\widehat Y(\omega)
=
\widehat K(\omega)\widehat X(\omega),
\qquad
\widehat K(\omega)
\in
\mathbb{C}^{C_{\mathrm{out}}\times C_{\mathrm{in}}}.
$$
</div>

The layer can mix channels differently at different frequencies. Under circular boundary conditions, its exact Euclidean operator norm is

<div class="display-equation">
$$
\|C_K\|_2
=
\max_{\omega}\sigma_{\max}\bigl(\widehat K(\omega)\bigr).
$$
</div>

This formula connects spatial kernels, spectral amplification, and gradient stability. It also explains why FFT algorithms can accelerate sufficiently large convolutions: transform the signals, multiply frequency responses, and transform back. For the small $3\times3$ kernels common in CNNs, transform overhead often outweighs the saving, so direct or Winograd-style methods are preferable. The fastest algorithm depends on kernel size, resolution, channels, batch size, and hardware.

Nonlinearities break the simple diagonal frequency picture because multiplying or thresholding in the spatial domain creates interactions among frequencies. Stride also aliases frequencies, and finite padding destroys exact circulant structure. Even so, Fourier analysis remains a powerful local model of what a convolutional block can amplify or suppress.

### Fixed filters as interpretable kernels

Classical image processing chooses kernels analytically instead of learning them. A normalized box filter,

<div class="display-equation">
$$
K_{\mathrm{box}}
=
\frac{1}{9}
\begin{pmatrix}
1&1&1\\
1&1&1\\
1&1&1
\end{pmatrix},
$$
</div>

is a crude low-pass filter. A discrete Gaussian approximation,

<div class="display-equation">
$$
K_{\mathrm{Gauss}}
=
\frac{1}{16}
\begin{pmatrix}
1&2&1\\
2&4&2\\
1&2&1
\end{pmatrix},
$$
</div>

suppresses high frequencies more smoothly. Sobel kernels approximate first derivatives:

<div class="display-equation">
$$
G_x=
\begin{pmatrix}
-1&0&1\\
-2&0&2\\
-1&0&1
\end{pmatrix},
\qquad
G_y=
\begin{pmatrix}
-1&-2&-1\\
0&0&0\\
1&2&1
\end{pmatrix}.
$$
</div>

If $I_x=I\star G_x$ and $I_y=I\star G_y$, then

<div class="display-equation">
$$
M=\sqrt{I_x^2+I_y^2},
\qquad
\theta=\operatorname{atan2}(I_y,I_x)
$$
</div>

estimate local gradient magnitude and orientation. A discrete Laplacian,

<div class="display-equation">
$$
K_{\Delta}
=
\begin{pmatrix}
0&1&0\\
1&-4&1\\
0&1&0
\end{pmatrix},
$$
</div>

approximates a second derivative and emphasizes rapid intensity change. Sharpening can be written as $I_{\mathrm{sharp}}=I-\alpha(I\star K_\Delta)$ for a suitable sign convention and scale.

A learned convolutional layer generalizes this filter-bank idea. The first-layer kernels of networks trained on natural images often become oriented edge detectors, color-opponent filters, or localized frequency-selective patterns reminiscent of Gabor functions. This does not mean the network has rediscovered the only correct visual basis. It means that edge and color contrast are economical coordinates for many natural-image tasks. At later layers, kernels operate on learned feature channels rather than raw intensity, so their interpretation as ordinary image filters becomes less direct.

The transition from fixed to learned filters marks a central change in computer vision. Classical systems chose the representation and learned only a shallow decision rule. Convolutional networks learn both the local measurements and the downstream decision. The price is reduced interpretability and a much larger optimization problem; the gain is that the filters can adapt to the task rather than reflect only the designer's prior expectations.

## Pooling, Downsampling, and the Price of Compression

Convolution often preserves spatial resolution, but a visual system cannot carry full-resolution feature maps indefinitely. High-resolution activations consume memory, and a classifier eventually needs to aggregate evidence over a region rather than preserve every coordinate. Pooling and strided convolution reduce spatial sampling density. Their computational usefulness is obvious; their mathematical effect is subtler than the phrase “keep the important features” suggests.

For a pooling window $\mathcal W_{i,j}$ in channel $c$, average pooling computes

<div class="display-equation">
$$
Y_{c,i,j}
=
\frac{1}{|\mathcal W_{i,j}|}
\sum_{(r,t)\in\mathcal W_{i,j}}X_{c,r,t},
$$
</div>

while max pooling computes

<div class="display-equation">
$$
Y_{c,i,j}
=
\max_{(r,t)\in\mathcal W_{i,j}}X_{c,r,t}.
$$
</div>

Neither operation has learned weights, but both remain part of the computation graph. For average pooling, an upstream derivative $\Delta_{c,i,j}$ is distributed uniformly over the window:

<div class="display-equation">
$$
\frac{\partial L}{\partial X_{c,r,t}}
=
\sum_{(i,j):(r,t)\in\mathcal W_{i,j}}
\frac{\Delta_{c,i,j}}{|\mathcal W_{i,j}|}.
$$
</div>

If windows overlap, the contributions add. For max pooling with a unique maximizer $(r^*,t^*)$, the derivative is routed only to that coordinate:

<div class="display-equation">
$$
\frac{\partial Y_{c,i,j}}{\partial X_{c,r,t}}
=
\mathbf 1\{(r,t)=(r^*,t^*)\}.
$$
</div>

Ties make max pooling nondifferentiable. A valid subgradient may distribute mass among tied maxima, while a concrete software implementation usually records one selected index during the forward pass and sends the gradient there. The resulting map is piecewise linear: the gradient is simple inside a region where the argmax pattern is fixed, and changes discontinuously when maxima exchange order.

Max pooling is sometimes described as a learned-feature detector made locally invariant to small translations. The description is only approximate. If a strong activation moves inside the same pooling cell without changing the maximum value, the pooled output remains stable. If it crosses a cell boundary, the output can change abruptly. Average pooling is smoother but can dilute a small high-amplitude feature. Which operator is appropriate depends on what the channel encodes. A feature channel whose activation means “this pattern is present somewhere nearby” may benefit from a maximum. A channel whose values should accumulate as evidence may be better summarized by an average.

### Global average pooling and exact invariance under ideal translations

Global average pooling maps each channel to one scalar:

<div class="display-equation">
$$
z_c
=
\frac{1}{HW}
\sum_{i=0}^{H-1}
\sum_{j=0}^{W-1}X_{c,i,j}.
$$
</div>

On a cyclic grid, translation merely permutes the summands, so

<div class="display-equation">
$$
\operatorname{GAP}(\tau_aX)
=
\operatorname{GAP}(X).
$$
</div>

This is an exact invariance statement under the assumed boundary model. If the preceding network is equivariant, global averaging produces an invariant representation. It also removes the need for a large fully connected layer tied to a particular resolution. A tensor with $C$ channels becomes a vector in $\mathbb{R}^C$ regardless of $H$ and $W$, which makes variable-resolution inference and transfer easier.

The invariance is not free. Global averaging discards where evidence occurred and how many distinct objects produced it beyond what survives in channel amplitudes. It is appropriate for image-level classification but inadequate for localization unless spatial information is preserved elsewhere. Moreover, the cyclic proof does not apply literally to cropped natural images with zero padding. A translated object may be partly removed from the field of view, and the convolutional representation near the border changes. Practical translation robustness is therefore an empirical property of the full data and preprocessing pipeline, not a consequence of one algebraic identity.

### Downsampling is sampling, and sampling can alias

A stride-$s$ convolution can be written conceptually as

<div class="display-equation">
$$
Y=D_s(C_KX),
$$
</div>

where $C_K$ is a stride-one filtering operator and $D_s$ keeps every $s$th spatial sample. Average pooling with stride larger than one has the same form with a fixed low-pass-like kernel; max pooling replaces linear filtering with a nonlinear selection before sampling.

The sampling theorem warns that downsampling without removing frequencies above the new Nyquist limit causes aliasing. In one dimension, if $y[m]=x[sm]$, then the discrete-time Fourier transform of $y$ is a sum of shifted and compressed copies of the spectrum of $x$:

<div class="display-equation">
$$
\widehat y(\omega)
=
\frac{1}{s}
\sum_{q=0}^{s-1}
\widehat x\left(\frac{\omega+2\pi q}{s}\right).
$$
</div>

High-frequency components that were distinct before sampling can overlap after sampling. A one-pixel translation changes their phases and can therefore produce a large change in the sampled output. This is why stride breaks exact equivariance to arbitrary one-pixel translations. A stride-$s$ layer can at best be exactly equivariant to translations aligned with the sampling lattice, and even that statement depends on boundary conventions.

Anti-aliased downsampling inserts a low-pass filter before subsampling. A blur-pooling block, for example, may perform a stride-one maximum or convolution followed by a fixed binomial blur and then decimation. The goal is not to make the network mathematically invariant to every shift; it is to suppress high frequencies that would fold into unstable low-frequency artifacts. The trade-off is a possible loss of fine detail. In tasks where one-pixel boundaries matter, such as medical segmentation or keypoint localization, aggressive low-pass filtering can be harmful. Sampling design must reflect the task's tolerance for spatial precision.

Strided convolution often replaces pooling in modern networks because it lets the model learn the pre-sampling filter and channel expansion jointly. This does not automatically solve aliasing: the learned kernel is not forced to be low-pass. It does, however, give optimization the possibility of discovering a task-appropriate compromise between retaining discriminative high-frequency detail and reducing shift sensitivity.

## Upsampling and Learned Decoders

Classification compresses space until one decision remains. Segmentation, image generation, super-resolution, depth estimation, and autoencoding must reverse that trend and construct a high-resolution output from a lower-resolution representation. Upsampling is therefore not one operation but a family of ways to map a coarse lattice to a finer one.

### Nearest-neighbor and bilinear interpolation

Nearest-neighbor upsampling by an integer factor $s$ copies each coarse value into an $s\times s$ block. In one dimension,

<div class="display-equation">
$$
y[sm+r]=x[m],
\qquad
r=0,\ldots,s-1.
$$
</div>

The map is linear in the input despite its piecewise-constant geometric appearance. Its adjoint sums gradients from each copied block back into the source value. Nearest-neighbor interpolation preserves sharp changes but can create blocky outputs.

Linear interpolation forms a convex combination between neighboring samples. If a fine-grid coordinate lies at relative position $t\in[0,1]$ between $x_0$ and $x_1$, then

<div class="display-equation">
$$
y=(1-t)x_0+tx_1.
$$
</div>

Bilinear interpolation applies this rule along both axes. If a fine location corresponds to coarse coordinates $(u,v)$ with integer parts $(i,j)$ and fractional parts $(\alpha,\beta)$, then

<div class="display-equation">
$$
\begin{aligned}
y(u,v)
={}&(1-\alpha)(1-\beta)x_{i,j}
+\alpha(1-\beta)x_{i+1,j}\\
&+(1-\alpha)\beta x_{i,j+1}
+\alpha\beta x_{i+1,j+1}.
\end{aligned}
$$
</div>

The weights depend only on geometry, not on training data. The operation is differentiable with respect to values and, away from cell boundaries, with respect to sampling coordinates. Spatial transformer and deformable convolution modules exploit this second derivative to learn where to sample.

Interpolation followed by an ordinary convolution separates resolution change from feature refinement. The fixed interpolation establishes a smooth or piecewise-constant fine grid; the learned convolution then removes artifacts and synthesizes task-specific detail. This separation often gives more predictable behavior than asking one transposed-convolution kernel to determine both geometry and content.

### Transposed convolution as a learnable adjoint map

Let an ordinary convolutional map be represented by $y=Cx$. Its input gradient is $C^\top\delta$. A transposed-convolution layer uses a matrix with this $C^\top$ structure as its forward map:

<div class="display-equation">
$$
z=C^\top h.
$$
</div>

When the associated ordinary convolution has stride greater than one, $C$ maps a larger grid to a smaller one. Its transpose maps the smaller grid back to a larger one. Operationally, one can understand this by inserting $s-1$ zeros between neighboring input samples and then applying a stride-one spatial filter with the appropriate padding and kernel orientation. Each coarse activation “paints” a weighted kernel-shaped patch onto the fine grid, and overlapping patches add.

For one spatial dimension, the output length of a transposed convolution is

<div class="display-equation">
$$
L_{\mathrm{out}}
=
(L_{\mathrm{in}}-1)s-2p+d(k-1)+o+1,
$$
</div>

where $o$ is `output_padding`. The two-dimensional formula applies independently to height and width. `output_padding` resolves an output-shape ambiguity created by stride: several input lengths can map to the same downsampled length under an ordinary convolution. It changes the calculated size on one side; it does not insert a band of zeros into the computed output.

The transpose can restore shape without restoring information. If $C$ discarded dimensions, then $C^\top C$ is not generally the identity. Even when input and output sizes match, convolution may suppress frequencies or have a nontrivial null space. Calling transposed convolution “deconvolution” therefore invites a false inverse interpretation. True deconvolution is an inverse problem requiring assumptions about the kernel, noise, boundary conditions, and regularization. A neural transposed convolution is simply a learned linear upsampling operator with the sparsity and weight-tying pattern of an adjoint convolution.

### Uneven overlap and checkerboard artifacts

Suppose a stride-two transposed convolution uses a kernel of width three. Some output positions receive contributions from two coarse samples, while others receive only one. In two dimensions, the horizontal and vertical overlap patterns multiply, producing a checkerboard of unequal default gains. A learned network could in principle tune coefficients to cancel the imbalance, but doing so restricts the available filters and becomes harder across channels and layers.

Even overlap, obtained when the kernel size is divisible by the stride, removes the simplest counting imbalance but does not prevent a learned filter from generating periodic artifacts. Resize-convolution is often more robust: first upsample with nearest-neighbor or bilinear interpolation, then apply a stride-one convolution. Pixel shuffle offers another factorization. A convolution produces $s^2C$ channels on the coarse grid, and a deterministic reshape rearranges groups of $s^2$ channels into an $s\times s$ spatial neighborhood. This moves the learned degrees of freedom into channel space before a fixed reindexing.

The best decoder depends on the output. Transposed convolution is compact and learnable, interpolation-convolution is predictable and often artifact-resistant, pixel shuffle is efficient for super-resolution, and max-unpooling can preserve indices selected during a matching encoder. No method can recover detail that the latent representation does not contain. Upsampling changes the coordinate grid; synthesis quality still depends on whether the encoder preserved phase, boundaries, and fine-scale evidence through skip connections or other side paths.

## Architectural Evolution as Changes in Inductive Bias

The history of convolutional networks is often narrated as a sequence of model names. A more useful reading asks what structural constraint each architecture changed. The central operator remained convolution, but the organization of resolution, channel width, receptive field, normalization, and information paths changed the function class and the optimization geometry.

### LeNet and the first complete convolutional system

LeNet-5 established the enduring pattern of local filtering, spatial reduction, and a final classifier. Its early convolutions learned shared local features from handwritten characters, while subsampling progressively reduced resolution. The network was small enough for the data and hardware of its time, and its success demonstrated that feature extraction and classification could be trained jointly rather than assembled from unrelated hand-engineered stages.

The deeper lesson is not the exact sequence of $5\times5$ filters and average-pooling layers. It is that an image classifier should preserve spatial organization while constructing intermediate features and should remove location only when the task permits it. Later networks changed activations, normalization, pooling, scale, and optimization, but retained this separation between an equivariant feature extractor and an invariant decision mechanism.

### AlexNet and the importance of optimization at scale

AlexNet did not introduce convolution, but it demonstrated that a substantially deeper and wider convolutional model could exploit a million-scale labeled dataset when trained with GPUs. ReLU replaced saturating nonlinearities in the hidden layers, dropout regularized large fully connected components, data augmentation enlarged the effective training distribution, and overlapping max pooling altered the spatial reduction schedule. The model's 2012 ImageNet result changed the empirical standard of computer vision because it showed that learned hierarchical representations could outperform pipelines built around fixed descriptors.

Its architecture also exposed limitations that later designs addressed. Large early kernels and strides discarded resolution quickly. The fully connected tail contained most of the parameters. Local response normalization did not become a durable default. The two-GPU split reflected hardware constraints rather than a general design principle. AlexNet should therefore be read as proof that scale, data, and optimization had crossed a threshold, not as a final template for convolutional design.

### VGG and regular depth

VGG replaced heterogeneous kernel sizes with a nearly uniform stack of $3\times3$ convolutions. The regularity made depth the principal variable. Two $3\times3$ layers produce a $5\times5$ receptive field with fewer parameters than one dense $5\times5$ layer at equal width, and the intervening nonlinearity increases expressive power. Repeating the pattern created a clean hierarchy in which resolution was halved by pooling and channel count was increased as spatial area decreased.

The simplicity made VGG features easy to transfer and analyze, but the architecture remained computationally heavy. Its fully connected classifier dominated parameter count, and a plain sequence of many transformations offered no short path for information or gradients. Adding still more layers eventually produced optimization degradation. The model clarified what ordinary stacked convolution could achieve and, by doing so, made the need for improved information flow unmistakable.

### Inception and explicit multiscale factorization

An Inception module applies several transformations to the same input in parallel, such as $1\times1$, $3\times3$, and larger receptive-field branches, then concatenates their output channels. The module treats scale as a dimension over which the representation may specialize. A small texture and a larger object part can be processed at the same stage without choosing one kernel size globally.

The decisive engineering idea was the pointwise bottleneck. If a $5\times5$ branch directly maps $C$ input channels to $M$ outputs, it costs $25CM$ coefficients per spatial site. Reducing to $R$ channels first changes the leading cost to $CR+25RM$. This made multibranch processing affordable and established $1\times1$ convolution as a general channel-projection operator. Global average pooling also reduced dependence on enormous fully connected layers.

Parallel branches increase design complexity, and later architectures found that many benefits could be obtained with more regular residual or factorized blocks. Inception nevertheless made an enduring contribution: expensive spatial operations should often be preceded by cheap channel projections, and multiscale computation can be represented explicitly inside a layer rather than only through depth.

### Residual networks and identity-centered parameterization

A residual block computes

<div class="display-equation">
$$
y=x+F(x;\theta).
$$
</div>

Instead of asking several layers to learn a desired map $H(x)$ directly, the block parameterizes $H(x)=x+F(x)$. The identity is represented by $F=0$, so adding blocks need not force the network away from a useful shallow solution. The backward derivative contains an identity term,

<div class="display-equation">
$$
\frac{\partial y}{\partial x}=I+J_F(x),
$$
</div>

which creates short computational paths across depth. A network of residual blocks expands into a collection of paths of different effective lengths, and optimization can initially adjust small residual corrections rather than coordinate a long chain of unrelated transformations.

In Basic Blocks, two $3\times3$ convolutions operate at one width. Bottleneck blocks use a $1\times1$ reduction, a $3\times3$ spatial transform, and a $1\times1$ expansion. When shape changes, the shortcut may use a projection. These details are not cosmetic: the addition requires compatible tensor shapes, and the bottleneck controls where channel mixing and spatial computation occur.

Residual networks made hundred-layer CNNs trainable and became a general backbone for classification, detection, segmentation, and representation learning. Yet residual addition is not a universal guarantee. If branch Jacobians are poorly scaled, products of $I+J_F$ can still be ill conditioned. Pre-activation, normalization placement, initialization, and stochastic depth all influence the result. The residual idea succeeds because it changes both representational coordinates and gradient paths, not because an addition symbol mechanically solves every deep-learning problem.

A useful continuous-depth interpretation writes

<div class="display-equation">
$$
x_{\ell+1}-x_\ell=F_\ell(x_\ell),
$$
</div>

which resembles a forward Euler step for an ordinary differential equation. The analogy explains why small residual updates compose smoothly and motivates stability analyses, but an ordinary ResNet is still a finite, learned, nonautonomous discrete system. The ODE view is a lens, not an identity.

### Dense connectivity and feature reuse

DenseNet replaces addition with concatenation:

<div class="display-equation">
$$
x_\ell
=
H_\ell([x_0,x_1,\ldots,x_{\ell-1}]).
$$
</div>

Every layer receives all preceding feature maps, and its new channels are made available to all later layers. Concatenation preserves earlier representations explicitly instead of blending them through addition. A small growth rate can therefore suffice: each layer contributes a modest number of new channels while reusing the accumulated basis.

The short paths improve gradient access and encourage feature reuse, but the growing concatenated state consumes memory. Implementations use bottlenecks, transition layers, and checkpointing to manage this cost. Dense connectivity and residual connectivity solve related problems with different algebra. Addition keeps width fixed and superposes representations in one space. Concatenation expands the space and lets later layers choose how to combine old and new coordinates.

### Efficient convolutional families

MobileNet made depthwise separable convolution the central computational primitive. Its factorization reduces arithmetic by replacing one dense spatial-channel kernel with a depthwise spatial stage and a pointwise channel stage. Later inverted residual blocks expand channels with a pointwise layer, apply a depthwise convolution in the expanded space, and project back to a narrow representation. A shortcut connects narrow endpoints when shapes agree. The block spends computation where nonlinear channel combinations are rich while keeping memory-efficient interfaces between blocks.

EfficientNet studied network scaling rather than only block design. Increasing depth, width, and input resolution changes computation approximately as

<div class="display-equation">
$$
\operatorname{cost}
\propto
(\text{depth})
(\text{width})^2
(\text{resolution})^2.
$$
</div>

Compound scaling chooses

<div class="display-equation">
$$
d=\alpha^\phi,
\qquad
w=\beta^\phi,
\qquad
r=\gamma^\phi,
$$
</div>

with a constraint such as $\alpha\beta^2\gamma^2\approx2$ so one unit of $\phi$ roughly doubles compute. The exact coefficients are empirical, but the principle is general: scaling only one axis can create an imbalanced network. More pixels require sufficient depth and width to process them; more channels require sufficient resolution and depth to be useful.

Efficiency remains hardware dependent. Depthwise kernels minimize arithmetic but may have low arithmetic intensity. Squeeze-and-excitation blocks add global channel reweighting with little nominal cost but introduce synchronization and memory access. A model with fewer floating-point operations is not automatically faster, and a model with fewer parameters is not automatically smaller at runtime once activations and operator implementations are included.

### ConvNeXt and the persistence of the convolutional prior

Vision Transformers prompted a re-examination of which improvements were genuinely tied to attention and which came from modern training and block design. ConvNeXt retained a convolutional backbone while adopting large depthwise kernels, inverted bottlenecks, fewer activation and normalization sites, LayerNorm-style normalization, and stage configurations influenced by contemporary transformer practice. The result showed that a carefully modernized CNN could remain highly competitive without abandoning translation-structured local operators.

The broader conclusion is not that convolution defeats attention or that attention makes convolution obsolete. Convolution applies a content-independent local routing pattern with strong parameter sharing. Self-attention applies content-dependent aggregation, often over a wider domain, at a higher data-dependent computational cost. Hybrid systems use convolution for efficient local processing and attention for adaptive long-range interaction. Architecture should follow the symmetry, scale, and compute structure of the problem rather than a contest between names.

## From Image Classification to Structured Visual Prediction

A convolutional backbone produces a spatial tensor. What happens next depends on the semantics of the target. Image classification wants one label for the entire field of view, object detection wants a set of labeled boxes, semantic segmentation wants a label distribution at every pixel, and face recognition wants an embedding whose geometry remains useful for identities that were absent during training. These tasks can share the same early visual operator while demanding very different final representations and losses.

### Image classification as equivariant evidence followed by invariant aggregation

Let a backbone produce $F(X)\in\mathbb{R}^{C\times H'\times W'}$. A modern classifier often applies global average pooling and a linear head:

<div class="display-equation">
$$
h_c(X)
=
\frac{1}{H'W'}
\sum_{i,j}F_c(X)_{i,j},
\qquad
z=Wh+b.
$$
</div>

The feature map retains where local evidence appears; the average converts each channel into an image-level statistic. If channel $c$ responds to a particular texture or object part, $h_c$ measures its mean presence over the sampled field. The linear head combines these statistics into class logits. This interpretation is more faithful than saying that the network “recognizes an object in one neuron.” Evidence is usually distributed across channels and locations, and the decision depends on a learned combination.

Data augmentation complements the architectural prior. Random crops and translations expose the model to label-preserving changes, encouraging robustness beyond the exact equivariance supplied by the operator. Color perturbations, blur, erasing, mixup, and other transformations encode further assumptions about what should or should not change the label. An augmentation is not merely extra data; it defines an orbit of examples that the learner is encouraged to treat consistently. If the transformation can change the target, the augmentation injects label noise. Horizontal flips are reasonable for many natural-object categories but not for text, laterality-sensitive medical images, or tasks in which orientation is itself the label.

Classification accuracy can hide important spatial failures. A network may base its prediction on background texture, watermarks, borders, or acquisition artifacts that correlate with the training labels. Convolution does not distinguish causal object evidence from a stable shortcut. Saliency maps, occlusion tests, counterfactual backgrounds, group-wise evaluation, and distribution shifts are therefore needed to learn what the classifier actually uses. The spatial inductive bias reduces sample complexity; it does not guarantee semantic alignment.

### Object detection: predicting a set of localized hypotheses

A detector must answer two coupled questions: what is present, and where is it? A bounding box can be parameterized by corner coordinates $(x_1,y_1,x_2,y_2)$ or by center, width, and height $(c_x,c_y,w,h)$. Localization quality is commonly measured through intersection over union. For boxes $A$ and $B$,

<div class="display-equation">
$$
\operatorname{IoU}(A,B)
=
\frac{|A\cap B|}{|A\cup B|}.
$$
</div>

IoU equals one for identical boxes and zero for disjoint boxes. Unlike coordinatewise squared error, it reflects overlap directly and is invariant to a common scale in the coordinate system. It is nevertheless flat at zero overlap, which makes direct optimization difficult. Generalized IoU, distance IoU, and complete IoU add geometric terms that provide gradients when boxes do not overlap or differ in center and aspect ratio.

Many convolutional detectors historically used anchors. An anchor $a=(c_x^a,c_y^a,w^a,h^a)$ is a reference box attached to a feature-map location. A target box $b$ is encoded by relative offsets such as

<div class="display-equation">
$$
t_x=\frac{c_x-c_x^a}{w^a},
\qquad
t_y=\frac{c_y-c_y^a}{h^a},
\qquad
t_w=\log\frac{w}{w^a},
\qquad
t_h=\log\frac{h}{h^a}.
$$
</div>

The inverse map is

<div class="display-equation">
$$
c_x=c_x^a+w^a t_x,
\qquad
c_y=c_y^a+h^a t_y,
\qquad
w=w^ae^{t_w},
\qquad
h=h^ae^{t_h}.
$$
</div>

Relative coordinates improve conditioning because the regression target is normalized by the reference scale. A detector then combines a classification or objectness loss with a localization loss:

<div class="display-equation">
$$
L
=
L_{\mathrm{cls}}
+\lambda_{\mathrm{box}}L_{\mathrm{box}}
+\lambda_{\mathrm{aux}}L_{\mathrm{aux}}.
$$
</div>

The simple sum conceals a difficult assignment problem. Which prediction is responsible for which object? Anchored systems use overlap thresholds or matching rules. Anchor-free systems assign objects to centers, points, or regions. Set-prediction systems solve a bipartite matching problem. Detection quality depends as much on this assignment and sampling policy as on the convolutional backbone.

R-CNN introduced deep convolutional features into region-based detection by generating candidate regions outside the network, warping each region, and running a CNN on it. The representation was powerful but the computation was duplicated across thousands of overlapping crops. SPPNet and Fast R-CNN recognized that convolution should be performed once on the whole image. Region features could then be extracted from the shared feature map by spatial pyramid pooling or RoI pooling. This moved the expensive computation before the proposal-specific branch and allowed classification and box refinement to be optimized jointly.

Faster R-CNN learned the proposal mechanism itself. A region proposal network slides over the shared feature map and predicts objectness and box offsets for anchors at every location. The proposal network and the second-stage classifier therefore reuse the same convolutional representation. The architecture is called two-stage because it first selects a sparse set of candidate regions and then performs more expensive per-region refinement. This conditional allocation of computation is especially effective when high localization accuracy matters and the number of objects is much smaller than the number of possible windows.

RoI pooling divides a proposal into bins and applies max pooling, but quantization of proposal boundaries can misalign features and pixels. RoIAlign replaces coarse coordinate rounding with bilinear sampling at continuous locations. The difference is crucial for instance segmentation, where a one-pixel shift can deform a mask boundary. Mask R-CNN adds a per-region mask head in parallel with classification and box regression, showing that the same detected instance can support several aligned outputs.

One-stage detectors remove the explicit proposal-refinement split and predict classes and boxes densely in one pass. The original YOLO formulated detection as direct regression from the image to grid-associated boxes and class probabilities. SSD made predictions at several feature-map resolutions, associating shallow high-resolution maps with small objects and deeper coarse maps with large objects. Later one-stage systems improved assignment, focal reweighting, anchor-free parameterizations, and multiscale feature fusion, narrowing much of the accuracy gap while retaining high throughput.

The class imbalance in dense detection is severe: most spatial hypotheses are background. If ordinary cross-entropy treats every easy negative equally, their aggregate gradient can dominate rare positives. Focal loss modifies the binary cross-entropy with a factor that suppresses already-correct examples. For target $y\in\{0,1\}$ and model probability $p_t$ assigned to the true class,

<div class="display-equation">
$$
L_{\mathrm{focal}}
=-\alpha_t(1-p_t)^\gamma\log p_t.
$$
</div>

When $p_t$ is near one, the factor $(1-p_t)^\gamma$ is small; hard or misclassified examples retain substantial weight. This is not a generic replacement for cross-entropy. It is a response to a specific sampling geometry in which an enormous number of easy negatives would otherwise determine the update.

Scale is another central problem. A deep low-resolution feature map has strong semantics but weak spatial detail; a shallow high-resolution map has precise location but weak semantic abstraction. Feature Pyramid Networks construct a top-down pathway with lateral connections, combining deep semantic features with earlier high-resolution maps. The resulting pyramid offers semantically rich representations at multiple strides. This is an architectural solution to the same tension already seen in U-shaped segmentation networks: high-level context and low-level localization must be brought into the same prediction.

Non-maximum suppression historically reconciles multiple boxes that describe one object. Predictions are sorted by score, and lower-scoring boxes with large overlap with a selected box are removed. The procedure is effective but nondifferentiable and controlled by a threshold. DETR reframed detection as direct set prediction. A fixed collection of object queries produces a set of candidate objects, and Hungarian matching assigns predictions to ground-truth boxes under a global cost. The set loss encourages one prediction per object, eliminating anchor design and conventional non-maximum suppression. Convolution often remains in the backbone, while attention handles global interaction and set-level reasoning. The transition illustrates a recurring pattern: convolution constructs spatial evidence efficiently; another operator may organize that evidence according to task-specific combinatorics.

Detection metrics require careful interpretation. Average precision integrates precision across recall levels after ranking predictions by confidence, usually at one or several IoU thresholds. It mixes classification, localization, duplicate suppression, and calibration into one summary. A detector can fail by missing objects, assigning the wrong class, shifting boxes, duplicating one object, or overpredicting background. Per-size, per-class, and error-type analyses are therefore more diagnostic than one aggregate number.

### Semantic segmentation: preserving semantics and boundaries at every pixel

Semantic segmentation assigns a class to each spatial site. If the network outputs logits

<div class="display-equation">
$$
Z\in\mathbb{R}^{C\times H\times W},
$$
</div>

then pixelwise cross-entropy is

<div class="display-equation">
$$
L_{\mathrm{CE}}
=-\frac{1}{|\Omega|}
\sum_{u\in\Omega}
\sum_{c=1}^C
Y_c(u)\log P_c(u),
$$
</div>

where $P(u)=\operatorname{softmax}(Z(u))$ and $Y(u)$ is a one-hot target. The formula treats sites as conditionally separate in the loss, but the predictions are coupled through shared receptive fields and the network's spatial structure.

A classification CNN can be converted into a fully convolutional network by replacing dense layers with equivalent convolutions and preserving spatial outputs. The network can then accept variable image sizes and produce a coarse label map. Upsampling restores the original resolution. The challenge is that repeated stride and pooling erase boundary detail. A deep unit may know that a bicycle is present but no longer know the exact contour of its spokes.

Skip architectures combine coarse semantics with fine geometry. U-Net uses a contracting encoder and an expanding decoder with lateral connections between corresponding resolutions. At decoder stage $\ell$, a typical map has the form

<div class="display-equation">
$$
D_\ell
=
\Phi_\ell\bigl([\operatorname{Up}(D_{\ell+1}),E_\ell]\bigr),
$$
</div>

where $E_\ell$ is the encoder feature at the same scale, brackets denote channel concatenation, and $\Phi_\ell$ is a convolutional block. The decoder receives both a high-level hypothesis from the coarse path and local evidence from the encoder. The skip connection is not merely a gradient aid; it transfers information that would otherwise be irrecoverably compressed.

Dilated convolution offers another way to enlarge context without further downsampling. DeepLab-style systems use atrous spatial pyramid pooling, evaluating several dilation rates in parallel so the output can integrate context over multiple scales. Very large dilation may sample too sparsely, while very small dilation may miss global context. The parallel construction lets the network combine both. Boundary refinement may also use conditional random fields, explicit edge losses, high-resolution branches, or learned decoders.

Class imbalance in segmentation is spatial rather than proposal-based. A small tumor, vessel, or defect may occupy a tiny fraction of the image, so pixel accuracy can be high even when the target is completely missed. The Dice coefficient for a soft prediction $p_u\in[0,1]$ and binary target $y_u$ is

<div class="display-equation">
$$
\operatorname{Dice}(p,y)
=
\frac{2\sum_u p_uy_u+\varepsilon}
{\sum_u p_u+\sum_u y_u+\varepsilon}.
$$
</div>

A Dice loss is $1-\operatorname{Dice}$. It normalizes overlap by predicted and target mass, making small structures more influential than they would be under an unweighted pixel average. Its derivative couples every pixel through the numerator and denominator. This global coupling can improve overlap but may produce unstable behavior when both prediction and target are nearly empty; the smoothing constant and empty-class policy therefore matter.

Intersection over union has the soft analogue

<div class="display-equation">
$$
\operatorname{IoU}(p,y)
=
\frac{\sum_u p_uy_u+\varepsilon}
{\sum_u p_u+\sum_u y_u-\sum_u p_uy_u+\varepsilon}.
$$
</div>

Cross-entropy, Dice, and boundary losses emphasize different errors. Cross-entropy is a proper probabilistic scoring rule at each site. Dice directly emphasizes region overlap but does not by itself guarantee calibrated probabilities. Boundary losses focus on contour distance. Combining them may be useful, but each coefficient changes the implied decision problem. A loss should be chosen from the cost of errors, not from the expectation that adding more terms is automatically better.

Segmentation exposes a fundamental tension in CNNs. Classification benefits from invariance and aggressive spatial compression. Dense prediction requires equivariance and precise coordinate recovery. A backbone designed only for classification may discard exactly the information a decoder needs. Output stride, anti-aliasing, skip placement, padding alignment, and interpolation conventions therefore become first-order design choices.

### Face recognition and the geometry of learned embeddings

Closed-set classification assumes that every test category belongs to the training label set. Face recognition usually operates in an open-set regime. During deployment, the system must compare identities that were never represented by a fixed training output neuron. The network therefore learns an embedding

<div class="display-equation">
$$
f_\theta(x)\in\mathbb{R}^d
$$
</div>

whose distances or angles encode identity similarity. Face verification asks whether two embeddings belong to the same person; identification searches a gallery for the nearest compatible identity. The decision threshold depends on the acceptable false-match and false-nonmatch costs, not only on training accuracy.

FaceNet directly optimized relative distances with the triplet loss. For an anchor $x^a$, a positive example $x^p$ of the same identity, and a negative example $x^n$ of a different identity,

<div class="display-equation">
$$
L_{\mathrm{triplet}}
=
\max\left\{
0,
\|f(x^a)-f(x^p)\|_2^2
-
\|f(x^a)-f(x^n)\|_2^2
+\alpha
\right\}.
$$
</div>

The margin $\alpha$ demands that the negative be farther than the positive by a prescribed amount. The loss is geometrically direct but statistically inefficient under random sampling. Most negative faces are already far away and contribute zero gradient. Training therefore depends on finding semi-hard or hard triplets. Extremely hard negatives may be mislabeled or visually corrupted, so indiscriminate hard mining can destabilize learning.

Classification-based metric losses avoid explicit triplet enumeration. Let both the feature $x_i$ and every class weight $w_j$ be normalized to unit length. Then the logit is proportional to cosine similarity:

<div class="display-equation">
$$
w_j^\top x_i
=
\cos\theta_{i,j}.
$$
</div>

Because the cosine lies in $[-1,1]$, an additional scale $s$ is used so the softmax can produce sufficiently confident probabilities:

<div class="display-equation">
$$
z_{i,j}=s\cos\theta_{i,j}.
$$
</div>

Without $s$, even the best possible angular separation may yield logits too small for a low cross-entropy when the number of identities is large. The scale acts like an inverse temperature and controls gradient concentration.

ArcFace modifies the target-class angle by an additive margin:

<div class="display-equation">
$$
z_{i,j}
=
\begin{cases}
 s\cos(\theta_{i,y_i}+m),&j=y_i,\\
 s\cos\theta_{i,j},&j\ne y_i.
\end{cases}
$$
</div>

The sample must move closer in angle to its class prototype before receiving the same target logit. The margin has a direct interpretation as extra geodesic separation on the unit hypersphere. At inference, the class prototypes may be discarded and normalized embeddings compared by cosine similarity.

This framework separates training identities from deployment identities, but it does not eliminate operational risks. Recognition error depends on image quality, pose, illumination, demographic distribution, gallery size, threshold choice, and the prevalence of impostor comparisons. A tiny pairwise false-match rate can produce many false candidates in a gallery containing millions of identities. Accuracy on a balanced verification benchmark is therefore not sufficient evidence for a deployment decision. Evaluation must match the gallery scale, acquisition process, and harm model of the intended use.

## Convolution Beyond Two-Dimensional Photographs

The defining idea is not restricted to RGB images. One-dimensional convolution acts on sequences such as audio waveforms, sensor traces, genomic signals, or temporal feature streams. A kernel $K\in\mathbb{R}^{C_{\mathrm{out}}\times C_{\mathrm{in}}\times k}$ shares a local temporal rule across time. Causal convolution restricts the support so an output at time $t$ depends only on present and past inputs:

<div class="display-equation">
$$
y_t
=
\sum_{a=0}^{k-1}K_a x_{t-a}.
$$
</div>

Stacked dilated causal convolutions can produce exponentially growing receptive fields while retaining parallel training, which made temporal convolution competitive with recurrent models in several sequence tasks.

Three-dimensional convolution operates on volumes or spacetime. For medical imaging, the axes may be depth, height, and width. For video, they may be time, height, and width. A $k_t\times k_h\times k_w$ kernel captures local motion or volumetric structure but is expensive: both arithmetic and activation memory grow with the extra dimension. Factorizations such as a spatial $1\times k_h\times k_w$ convolution followed by a temporal $k_t\times1\times1$ convolution reduce cost and insert an additional nonlinearity. Anisotropic medical voxels may require kernels and strides that respect unequal physical spacing rather than treating array indices as equal distances.

The translation group itself can be generalized. Group-equivariant CNNs define feature maps over rotations, reflections, or other transformation groups and replace ordinary convolution with group convolution. If $G$ acts on the domain, the goal is

<div class="display-equation">
$$
F(\rho_{\mathrm{in}}(g)X)
=
\rho_{\mathrm{out}}(g)F(X)
\qquad\text{for }g\in G.
$$
</div>

Ordinary translation convolution is the simplest case. Rotation-equivariant designs reduce the need to learn separately rotated copies of a filter, but they increase implementation complexity and impose a stronger prior that may not match all data. The general lesson remains: choose the symmetry group from transformations that preserve the task, then build it into the operator when the gain in sample efficiency justifies the restriction.

Irregular domains require other notions of neighborhood and symmetry. Graph neural networks aggregate messages over edges, spherical CNNs respect rotations on the sphere, and point-cloud networks handle unordered samples in three-dimensional space. These methods are descendants of the same principle: replace an arbitrary dense map with a structured operator compatible with the geometry of the domain.

## What Convolutional Networks Learn, and What They Do Not Guarantee

A convolutional architecture makes some functions easy and others difficult. It does not decide which easy function optimization will select. Understanding this gap is essential when interpreting visual representations.

### Translation is only one transformation

Ordinary convolution is not inherently invariant or equivariant to rotation, scale, perspective, elastic deformation, or illumination. A rotated edge generally activates a different filter. Scale changes move patterns across receptive-field sizes and feature-pyramid levels. Data augmentation, multiscale architectures, spatial transformers, deformable sampling, steerable filters, or group-equivariant constructions can improve these behaviors, but each adds a different assumption. Saying that CNNs “recognize an object regardless of where or how it appears” confuses empirical robustness from data and architecture with a theorem about the operator.

Absolute position can still enter through boundaries, padding, cropping, and coordinate channels. Even without explicit coordinates, zero padding lets a network infer distance to an edge by detecting the artificial exterior. This may be useful when the task is position dependent, as in medical imaging with standardized anatomy, but it weakens the claim of translation symmetry. CoordConv deliberately supplies normalized coordinates when absolute location is difficult to infer but relevant to the target. The choice is not between pure and impure models; it is between priors appropriate or inappropriate to the problem.

### Texture, shape, and shortcuts

CNNs trained on standard image classification data often rely heavily on local texture. This follows naturally from the operator: local repeated patterns are easy to detect and remain predictive across position. Global shape requires integrating evidence over larger distances and may be statistically less reliable under ordinary training. Stylized data, stronger augmentation, larger receptive fields, or objectives emphasizing contours can alter the balance, but no architecture guarantees a human-like notion of shape.

A shortcut is any feature that predicts the training target without matching the intended causal rule. Hospital identity can predict disease prevalence, snow can predict wolves, a compression signature can predict data source, and a frame border can predict a class. Convolution may make some shortcuts especially accessible because repeated local artifacts produce strong pooled evidence. High accuracy under the same data collection process cannot distinguish a desired feature from a shortcut. Deliberate distribution shifts and intervention-like tests are required.

### Equivariance is not robustness

Exact equivariance states how outputs transform under one known transformation. Robustness asks whether task performance remains acceptable under a family of perturbations, including transformations not built into the model. A map can be perfectly equivariant and still amplify noise, fail under blur, or change class under a small adversarial perturbation. Conversely, a non-equivariant model may be empirically robust over a limited deployment distribution. Operator symmetry, Lipschitz behavior, calibration, and distributional robustness are separate properties.

The frequency response of strided layers provides one concrete failure mechanism. Aliasing can turn a small shift into a large representation change. High-frequency features can also support adversarial directions because many pixel-level perturbations accumulate through shared filters. Anti-aliasing and spectral control may improve stability, but robustness involves the full nonlinear network and data distribution rather than one layer-level remedy.

### Locality can become a limitation

Locality is statistically efficient when nearby structure dominates. It can be inefficient for relations between distant objects, global counting, or tasks in which a remote context changes the interpretation of a local pattern. Depth enlarges receptive fields, but the effective receptive field may remain concentrated, and information must pass through many transformations. Attention provides direct, content-dependent interaction between distant sites. Large-kernel convolution, global pooling inside blocks, Fourier operators, and recurrent spatial passes offer other solutions.

A sensible architecture may therefore combine operators. Local convolution efficiently extracts edges, textures, and short-range geometry. Attention or global mixing organizes long-range dependencies. Residual and multiscale pathways preserve information across depth and resolution. The question is not which operator is universally superior; it is which decomposition places the right computation at the right scale.

### Interpretability remains limited

A first-layer kernel can be visualized directly, but a deep feature channel is defined by the entire preceding computation and may respond to many heterogeneous patterns. Activation maximization, feature visualization, linear probes, concept activation vectors, and attribution maps reveal aspects of the representation, not complete causal explanations. Attention maps and saliency maps are especially easy to overinterpret because they are model-dependent summaries and can change under equivalent parameterizations.

The most reliable interpretation is often behavioral. Construct controlled input variations, remove candidate evidence, test invariances, examine counterexamples, and evaluate across groups and environments. Mathematical inspection of kernels explains the operator class; empirical intervention reveals which member of that class the training process selected.

## The Statistical Price and Benefit of Symmetry

The parameter-count argument suggests why convolution can learn from fewer examples than an unrestricted dense map, but parameter count alone is not a complete theory of generalization. Modern networks can interpolate training data while containing far more parameters than examples. Norms, margins, optimization bias, augmentation, data geometry, and architectural sharing all affect the effective hypothesis class. Even so, a simple calculation makes the statistical role of sharing concrete.

Let $p_u(X)\in\mathbb{R}^{Ck_hk_w}$ be the patch extracted at position $u$, and consider a shared linear detector followed by spatial averaging:

<div class="display-equation">
$$
f_w(X)
=
\frac{1}{M}
\sum_{u=1}^M
\langle w,p_u(X)\rangle
=
\langle w,\bar p(X)\rangle,
\qquad
\bar p(X)=\frac{1}{M}\sum_{u=1}^Mp_u(X).
$$
</div>

Restrict $\|w\|_2\leq B$ and suppose $\|\bar p(X_i)\|_2\leq R$ on a sample of $n$ images. The empirical Rademacher complexity is

<div class="display-equation">
$$
\widehat{\mathfrak R}_n
=
\mathbb E_\sigma
\sup_{\|w\|_2\leq B}
\frac{1}{n}
\sum_{i=1}^n
\sigma_i\langle w,\bar p(X_i)\rangle.
$$
</div>

By duality of the Euclidean norm,

<div class="display-equation">
$$
\widehat{\mathfrak R}_n
=
\frac{B}{n}
\mathbb E_\sigma
\left\|
\sum_{i=1}^n\sigma_i\bar p(X_i)
\right\|_2
\leq
\frac{B}{n}
\sqrt{
\sum_{i=1}^n\|\bar p(X_i)\|_2^2
}
\leq
\frac{BR}{\sqrt n}.
$$
</div>

The bound depends on one kernel norm, not on a separate norm for every spatial position. A locally connected layer without weight sharing would learn parameters $w_u$ for each location and would require a different norm control over the concatenated parameter vector. The exact comparison depends on the pooling and data norms, but the mechanism is clear: symmetry reduces the number of independently adjustable directions in function space.

It would be incorrect to conclude that an image containing $M$ patches supplies $M$ independent training examples for the kernel. Neighboring patches overlap and natural-image structure creates strong dependence. Weight sharing allows evidence from many positions and images to update the same coefficient, but the effective sample size depends on correlation. A thousand nearly identical patches do not contain the information of a thousand independent observations. This is one reason broad data diversity remains important even when convolution generates many local training signals per image.

The prior can also be wrong. If a lesion at one anatomical location has a different meaning from an identical-looking lesion elsewhere, strict translation sharing hides useful coordinates. If the imaging sensor has position-dependent noise, a homogeneous kernel is misspecified. A larger hypothesis class can reduce this approximation error, while the shared class reduces estimation error. Architecture selection is therefore a bias-variance decision expressed through symmetry. The best prior is not the strongest one that can be imposed; it is the strongest one that remains approximately valid for the target distribution.

Data augmentation and architectural equivariance solve related but distinct problems. Exact equivariance constrains every function represented by the layer. Augmentation leaves the function class unchanged but changes the empirical objective by averaging losses over transformed samples:

<div class="display-equation">
$$
\widehat R_{\mathrm{aug}}(\theta)
=
\frac{1}{n}
\sum_{i=1}^n
\mathbb E_{g\sim\mu}
\ell\bigl(f_\theta(\rho(g)x_i),y_i\bigr).
$$
</div>

The model is encouraged, not forced, to behave consistently under transformations drawn from $\mu$. This flexibility is useful when the symmetry is approximate. An exactly rotation-invariant classifier cannot use orientation even when orientation becomes relevant; an augmented classifier can learn a residual orientation dependence if the data consistently reward it.

The deep convolutional network combines both mechanisms. Local equivariant layers sharply restrict early computation, while nonlinear composition and pooling create a large global function class. Optimization then imposes another implicit bias by favoring solutions reachable from the initialization under stochastic gradient updates. Generalization emerges from this entire system rather than from parameter sharing in isolation.

## A Shape-Safe PyTorch Implementation

PyTorch represents image batches in `NCHW` order: batch, channel, height, width. `nn.Conv2d` stores weights in the shape

<div class="display-equation">
$$
(C_{\mathrm{out}},C_{\mathrm{in}}/G,k_h,k_w)
$$
</div>

and performs cross-correlation. The module parameters `stride`, `padding`, `dilation`, and `groups` correspond directly to the forward formula developed earlier. A robust implementation should make those contracts visible through shape checks and tests instead of relying on trial and error.

### Reconstructing `Conv2d` with `unfold`

The following function implements an ungrouped two-dimensional convolution by extracting patches and multiplying them by flattened kernels. It is not intended to outperform the library operator. Its purpose is to expose the exact matrix structure and provide a reference implementation for debugging.

```python
from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

Pair: TypeAlias = tuple[int, int]


def _pair(value: int | Sequence[int]) -> Pair:
    if isinstance(value, int):
        return value, value
    if len(value) != 2:
        raise ValueError(f"Expected an int or a pair, received {value!r}.")
    return int(value[0]), int(value[1])


def conv2d_via_unfold(
    x: Tensor,
    weight: Tensor,
    bias: Tensor | None = None,
    *,
    stride: int | Pair = 1,
    padding: int | Pair = 0,
    dilation: int | Pair = 1,
) -> Tensor:
    """Reference implementation of an ungrouped 2D cross-correlation."""
    if x.ndim != 4:
        raise ValueError(f"x must have shape (N, Cin, H, W), got {tuple(x.shape)}.")
    if weight.ndim != 4:
        raise ValueError(
            "weight must have shape (Cout, Cin, kh, kw), "
            f"got {tuple(weight.shape)}."
        )

    n, c_in, h_in, w_in = x.shape
    c_out, c_weight, k_h, k_w = weight.shape
    if c_weight != c_in:
        raise ValueError(
            f"Input has {c_in} channels but weight expects {c_weight}."
        )
    if bias is not None and bias.shape != (c_out,):
        raise ValueError(f"bias must have shape ({c_out},), got {tuple(bias.shape)}.")

    s_h, s_w = _pair(stride)
    p_h, p_w = _pair(padding)
    d_h, d_w = _pair(dilation)

    k_eff_h = d_h * (k_h - 1) + 1
    k_eff_w = d_w * (k_w - 1) + 1
    h_out = (h_in + 2 * p_h - k_eff_h) // s_h + 1
    w_out = (w_in + 2 * p_w - k_eff_w) // s_w + 1
    if h_out <= 0 or w_out <= 0:
        raise ValueError(
            "Kernel, dilation, and padding produce a nonpositive output size."
        )

    patches = F.unfold(
        x,
        kernel_size=(k_h, k_w),
        dilation=(d_h, d_w),
        padding=(p_h, p_w),
        stride=(s_h, s_w),
    )
    # patches: (N, Cin * kh * kw, Hout * Wout)
    weight_flat = weight.reshape(c_out, -1)
    output_flat = torch.einsum("oc,ncl->nol", weight_flat, patches)

    if bias is not None:
        output_flat = output_flat + bias.view(1, c_out, 1)

    return output_flat.reshape(n, c_out, h_out, w_out)
```

A direct comparison should use double precision and nontrivial stride, padding, and dilation:

```python
torch.manual_seed(7)

x = torch.randn(2, 3, 11, 13, dtype=torch.float64)
weight = torch.randn(5, 3, 3, 2, dtype=torch.float64)
bias = torch.randn(5, dtype=torch.float64)

expected = F.conv2d(
    x,
    weight,
    bias,
    stride=(2, 1),
    padding=(2, 1),
    dilation=(2, 1),
)
actual = conv2d_via_unfold(
    x,
    weight,
    bias,
    stride=(2, 1),
    padding=(2, 1),
    dilation=(2, 1),
)

torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
print(actual.shape)
```

The comparison tests the full operator, not merely its shape. A wrong kernel orientation, patch order, or dilation convention can produce the expected dimensions while returning incorrect values.

### Testing the adjoint relation

The next calculation verifies that `conv_transpose2d` acts as the adjoint of `conv2d` with respect to the input. The selected sizes avoid an `output_padding` ambiguity, so the transposed result has exactly the original input shape.

```python
torch.manual_seed(11)

x = torch.randn(2, 3, 17, 19, dtype=torch.float64)
weight = torch.randn(5, 3, 3, 3, dtype=torch.float64)

y = F.conv2d(x, weight, stride=2, padding=1)
z = torch.randn_like(y)

adjoint_z = F.conv_transpose2d(
    z,
    weight,
    stride=2,
    padding=1,
)

lhs = torch.sum(y * z)
rhs = torch.sum(x * adjoint_z)

torch.testing.assert_close(lhs, rhs, rtol=1e-10, atol=1e-10)
print(float((lhs - rhs).abs()))
```

The equality does not say that `adjoint_z` reconstructs `x`. The tensor `z` is arbitrary, and the test concerns inner products. Replacing `z` by `y` would produce $C^\top Cx$, not generally $x$.

### Verifying the hand-derived backward example

Automatic differentiation can also verify the numerical calculation from the previous section:

```python
x = torch.tensor(
    [[[[1.0, 2.0, 0.0],
       [3.0, 1.0, 2.0],
       [0.0, 1.0, 4.0]]]],
    dtype=torch.float64,
    requires_grad=True,
)
weight = torch.tensor(
    [[[[1.0, -1.0],
       [2.0,  0.0]]]],
    dtype=torch.float64,
    requires_grad=True,
)

output = F.conv2d(x, weight)
loss = 0.5 * output.square().sum()
loss.backward()

expected_output = torch.tensor(
    [[[[5.0, 4.0],
       [2.0, 1.0]]]],
    dtype=torch.float64,
)
expected_d_weight = torch.tensor(
    [[[[20.0, 14.0],
       [20.0, 19.0]]]],
    dtype=torch.float64,
)
expected_d_x = torch.tensor(
    [[[[5.0, -1.0, -4.0],
       [12.0, 7.0, -1.0],
       [4.0, 2.0, 0.0]]]],
    dtype=torch.float64,
)

torch.testing.assert_close(output, expected_output)
torch.testing.assert_close(weight.grad, expected_d_weight)
torch.testing.assert_close(x.grad, expected_d_x)
```

A small exact example is often more informative than inspecting a large trained network. It separates operator correctness from optimization, data loading, and numerical scale.

### A residual classifier without a hard-coded spatial size

The following model is designed for small color images such as CIFAR-10. It uses residual blocks, strided convolutions for downsampling, and adaptive global average pooling. The classifier therefore depends on channel count rather than a manually calculated flattened spatial dimension.

```python
class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("This block supports stride 1 or 2.")

        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )

        if stride == 1 and in_channels == out_channels:
            self.shortcut: nn.Module = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.main(x) + self.shortcut(x))


class SpatialClassifier(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2),
            ResidualBlock(128, 128),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2),
            ResidualBlock(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(256, num_classes)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)

    def forward_features(self, x: Tensor) -> Tensor:
        if x.ndim != 4 or x.size(1) != 3:
            raise ValueError(
                "Expected input shape (N, 3, H, W), "
                f"received {tuple(x.shape)}."
            )
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        return self.stage3(x)

    def forward(self, x: Tensor) -> Tensor:
        x = self.forward_features(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        return self.head(x)
```

The two stride-two blocks reduce a $32\times32$ image to an $8\times8$ feature map, but the head never assumes that size. A $40\times40$ image would produce a different intermediate grid and the same $256$-dimensional pooled vector. This is safer than setting `nn.Linear(256 * 8 * 8, 10)`, which silently binds the model to one resolution and introduces far more parameters.

A basic smoke test should check output shape, finiteness, gradient existence, and the ability to overfit a tiny sample:

```python
torch.manual_seed(23)
model = SpatialClassifier(num_classes=10)
inputs = torch.randn(8, 3, 32, 32)
labels = torch.randint(0, 10, (8,))

logits = model(inputs)
if logits.shape != (8, 10):
    raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")
if not torch.isfinite(logits).all():
    raise FloatingPointError("The forward pass produced nonfinite logits.")

loss = F.cross_entropy(logits, labels)
loss.backward()

missing = [
    name
    for name, parameter in model.named_parameters()
    if parameter.requires_grad and parameter.grad is None
]
if missing:
    raise RuntimeError(f"Parameters without gradients: {missing}")
```

### Inspecting shapes and receptive-field stages

Forward hooks can reveal tensor shapes without modifying the model. They should be removed after use so they do not accumulate across experiments.

```python
def report_shape(name: str):
    def hook(_module: nn.Module, _inputs: tuple[Tensor, ...], output: Tensor) -> None:
        print(f"{name:>8s}: {tuple(output.shape)}")
    return hook


handles = [
    model.stem.register_forward_hook(report_shape("stem")),
    model.stage1.register_forward_hook(report_shape("stage1")),
    model.stage2.register_forward_hook(report_shape("stage2")),
    model.stage3.register_forward_hook(report_shape("stage3")),
    model.pool.register_forward_hook(report_shape("pool")),
]

with torch.inference_mode():
    _ = model(torch.randn(2, 3, 32, 32))

for handle in handles:
    handle.remove()
```

The expected sequence is $(2,64,32,32)$, $(2,64,32,32)$, $(2,128,16,16)$, $(2,256,8,8)$, and $(2,256,1,1)$. Shape traces should be read together with the receptive-field recurrence. Equal tensor sizes do not imply equal spatial context; each stride-one block enlarges the receptive field even when height and width are unchanged.

### Training data and normalization

For CIFAR-10, a conventional training transform uses random padded crops and horizontal flips, followed by tensor conversion and channelwise normalization. The exact statistics should be estimated from the training data or taken from a documented preprocessing convention, never from validation or test examples.

```python
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

mean = (0.4914, 0.4822, 0.4465)
std = (0.2470, 0.2435, 0.2616)

train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)
eval_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

# Two views of the same training files allow training augmentation to remain
# disabled on validation examples.
train_source = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=train_transform,
)
validation_source = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=eval_transform,
)
test_set = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=eval_transform,
)

split_generator = torch.Generator().manual_seed(20260720)
permutation = torch.randperm(len(train_source), generator=split_generator).tolist()
validation_size = 5_000
validation_indices = permutation[:validation_size]
train_indices = permutation[validation_size:]

train_set = Subset(train_source, train_indices)
validation_set = Subset(validation_source, validation_indices)

loader_kwargs = {
    "num_workers": 4,
    "pin_memory": torch.cuda.is_available(),
    "persistent_workers": True,
}
train_loader = DataLoader(
    train_set,
    batch_size=128,
    shuffle=True,
    **loader_kwargs,
)
validation_loader = DataLoader(
    validation_set,
    batch_size=256,
    shuffle=False,
    **loader_kwargs,
)
test_loader = DataLoader(
    test_set,
    batch_size=256,
    shuffle=False,
    **loader_kwargs,
)
```

The official test set is used only for final evaluation. The training and validation subsets share the same underlying image files but not the same transform object: random cropping and flipping occur only for training. The split indices and seed are explicit. In a serious experiment, the transform version, dataset checksum, and software environment also belong in the experiment record. Reproducibility is not achieved by setting one random seed while leaving the partition and preprocessing pipeline unspecified.

### A complete training and evaluation pass

The generic optimization logic is the same as for the linear model in the preceding chapter, but convolution introduces mode-dependent BatchNorm behavior and substantially larger activation tensors. The following functions keep the train and evaluation states explicit and weight epoch losses by batch size.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float


def run_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochMetrics:
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss: {loss.item()}")

        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += batch_size

    return EpochMetrics(loss=loss_sum / count, accuracy=correct / count)


@torch.inference_mode()
def evaluate_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EpochMetrics:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += batch_size

    return EpochMetrics(loss=loss_sum / count, accuracy=correct / count)
```

An optimizer and schedule can then be attached in the same disciplined order used previously:

```python
import copy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SpatialClassifier(num_classes=10).to(device)

num_epochs = 100
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4,
    nesterov=True,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=num_epochs,
    eta_min=1e-5,
)

best_validation_loss = float("inf")
best_state: dict[str, Tensor] | None = None

# validation_loader must be constructed from a held-out part of the training set.
for epoch in range(1, num_epochs + 1):
    train_metrics = run_training_epoch(
        model,
        train_loader,
        optimizer,
        device,
    )
    validation_metrics = evaluate_classifier(
        model,
        validation_loader,
        device,
    )

    if validation_metrics.loss < best_validation_loss:
        best_validation_loss = validation_metrics.loss
        best_state = copy.deepcopy(model.state_dict())

    current_lr = optimizer.param_groups[0]["lr"]
    print(
        f"epoch={epoch:03d} "
        f"lr={current_lr:.3e} "
        f"train_loss={train_metrics.loss:.4f} "
        f"train_acc={train_metrics.accuracy:.3%} "
        f"val_loss={validation_metrics.loss:.4f} "
        f"val_acc={validation_metrics.accuracy:.3%}"
    )
    scheduler.step()

if best_state is None:
    raise RuntimeError("Training ended without a validation checkpoint.")

model.load_state_dict(best_state)
test_metrics = evaluate_classifier(model, test_loader, device)
print(
    f"test_loss={test_metrics.loss:.4f} "
    f"test_acc={test_metrics.accuracy:.3%}"
)
```

The learning-rate schedule is stepped after the epoch so the first epoch uses the declared initial rate. Other schedulers have different contracts, particularly those driven by validation metrics, so the call order must follow the documented semantics rather than a universal template.

### Convolution-specific debugging

A convolutional run that fails should be localized before the architecture is redesigned. First verify data layout: accidentally supplying `NHWC` tensors to an `NCHW` model can produce an immediate shape error or, in unfortunate dimensions, a silent semantic error. Check normalization by visualizing transformed samples after reversing the normalization. Confirm that the final logits have one coordinate per class and that no Softmax is applied before `CrossEntropyLoss`.

Next inspect spatial sizes and receptive fields. An unexpected floor in the output formula can remove a row or column and break skip alignment. Different interpolation conventions can shift a decoder output by half a pixel. A model that collapses resolution too early may fit image labels while failing localization. Hooks and explicit assertions should document every intended stage boundary.

Then inspect optimization. Record activation means and variances, gradient norms, parameter norms, and the fraction of zero ReLU outputs. Compare training and evaluation behavior because BatchNorm uses minibatch statistics in training and running estimates in evaluation. Very small batches can make those estimates noisy. Freezing BatchNorm, replacing it with GroupNorm, accumulating larger batches, or changing the architecture may be appropriate, but only after the actual mismatch has been measured.

Finally test a tiny subset. A sufficiently expressive classifier should drive the loss on a handful of examples near zero. Failure usually indicates a bug, incompatible augmentation, excessive regularization, poor learning-rate scale, or an objective mismatch. Success does not prove generalization; it proves that the data-to-loss path can express and optimize at least one simple fit. The tiny-set test remains one of the highest-value debugging tools in deep learning because it separates implementation failure from statistical difficulty.

## The Role of the Convolutional Layer

The central formula of this chapter remained linear:

<div class="display-equation">
$$
Y_{o,i,j}
=
b_o+
\sum_{c,a,b}
W_{o,c,a,b}X_{c,i+a,j+b}.
$$
</div>

What made the layer powerful was not a hidden nonlinearity inside that sum. It was the structure imposed on the linear map. Coefficients were indexed by relative displacement rather than absolute coordinate, which encoded translation symmetry. Kernel support restricted early interactions to local neighborhoods. Channel dimensions allowed a bank of learned measurements to be propagated across the image. Depth, nonlinear activation, normalization, and residual paths then composed these structured maps into a representation whose receptive field and semantics expanded with the computation.

The same formula acquired several interpretations. On the lattice, it was the general linear translation-equivariant operator. As a matrix, it was sparse Toeplitz structure with tied entries. Under Fourier transformation, it became a frequency-dependent channel matrix. During backpropagation, its adjoint became the input-gradient operator and the foundation of transposed convolution. With groups and factorization, it separated spatial and channel computation. With stride, it became a filter followed by sampling and therefore inherited the risks of aliasing. With global aggregation, an equivariant representation became approximately invariant for classification.

The larger lesson is methodological. A successful learning system does not begin by granting every input coordinate an unrelated parameter and asking optimization to discover all structure from scratch. It identifies transformations, neighborhoods, conservation laws, or exchangeabilities that the task approximately respects and encodes them in the hypothesis class. Convolution is one of the clearest examples of this strategy because its symmetry, algebra, statistics, and implementation all agree. The modern visual system may later add attention, deformable sampling, multiscale routing, or set prediction, but those additions do not make the convolutional idea obsolete. They reveal where its prior is strong, where it is incomplete, and how structured operators can be combined.

A convolutional network is therefore not simply a neural network that happens to contain sliding kernels. It is a spatial learning system whose behavior is determined jointly by symmetry, sampling, scale, boundary conditions, optimization, and the semantics of the output. Understanding those interactions is what turns a familiar layer into a principled model.

## References

1. Ian Goodfellow, Yoshua Bengio, and Aaron Courville, [*Deep Learning*, Chapter 9: Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html), MIT Press, 2016.

2. Stanford University, [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/), course notes.

3. Vincent Dumoulin and Francesco Visin, [A Guide to Convolution Arithmetic for Deep Learning](https://arxiv.org/abs/1603.07285), 2016.

4. Taco Cohen and Max Welling, [Group Equivariant Convolutional Networks](https://proceedings.mlr.press/v48/cohenc16.html), ICML, 2016.

5. Risi Kondor and Shubhendu Trivedi, [On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups](https://arxiv.org/abs/1802.03690), ICML, 2018.

6. Michael M. Bronstein, Joan Bruna, Taco Cohen, and Petar Veličković, [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478), 2021.

7. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html), ICCV, 2015.

8. Hanie Sedghi, Vineet Gupta, and Philip M. Long, [The Singular Values of Convolutional Layers](https://openreview.net/forum?id=HJlnC1rKPB), ICLR, 2019.

9. Wenjie Luo, Yujia Li, Raquel Urtasun, and Richard Zemel, [Understanding the Effective Receptive Field in Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2016/hash/c8067ad1937f728f51288b3eb986afaa-Abstract.html), NeurIPS, 2016.

10. Richard Zhang, [Making Convolutional Networks Shift-Invariant Again](https://proceedings.mlr.press/v97/zhang19a.html), ICML, 2019.

11. Augustus Odena, Vincent Dumoulin, and Chris Olah, [Deconvolution and Checkerboard Artifacts](https://distill.pub/2016/deconv-checkerboard/), *Distill*, 2016.

12. Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, and Yichen Wei, [Deformable Convolutional Networks](https://openaccess.thecvf.com/content_iccv_2017/html/Dai_Deformable_Convolutional_Networks_ICCV_2017_paper.html), ICCV, 2017.

13. Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner, [Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf), *Proceedings of the IEEE*, 1998.

14. Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton, [ImageNet Classification with Deep Convolutional Neural Networks](https://proceedings.neurips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html), NeurIPS, 2012.

15. Karen Simonyan and Andrew Zisserman, [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556), ICLR, 2015.

16. Christian Szegedy et al., [Going Deeper with Convolutions](https://openaccess.thecvf.com/content_cvpr_2015/html/Szegedy_Going_Deeper_With_2015_CVPR_paper.html), CVPR, 2015.

17. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, [Deep Residual Learning for Image Recognition](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html), CVPR, 2016.

18. Gao Huang, Zhuang Liu, Laurens van der Maaten, and Kilian Q. Weinberger, [Densely Connected Convolutional Networks](https://openaccess.thecvf.com/content_cvpr_2017/html/Huang_Densely_Connected_Convolutional_CVPR_2017_paper.html), CVPR, 2017.

19. Andrew G. Howard et al., [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861), 2017.

20. Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, and Liang-Chieh Chen, [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html), CVPR, 2018.

21. Mingxing Tan and Quoc V. Le, [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://proceedings.mlr.press/v97/tan19a.html), ICML, 2019.

22. Zhuang Liu et al., [A ConvNet for the 2020s](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.html), CVPR, 2022.

23. Jonathan Long, Evan Shelhamer, and Trevor Darrell, [Fully Convolutional Networks for Semantic Segmentation](https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html), CVPR, 2015.

24. Olaf Ronneberger, Philipp Fischer, and Thomas Brox, [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597), MICCAI, 2015.

25. Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam, [Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1802.02611), ECCV, 2018.

26. Ross Girshick, Jeff Donahue, Trevor Darrell, and Jitendra Malik, [Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation](https://openaccess.thecvf.com/content_cvpr_2014/html/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.html), CVPR, 2014.

27. Ross Girshick, [Fast R-CNN](https://openaccess.thecvf.com/content_iccv_2015/html/Girshick_Fast_R-CNN_ICCV_2015_paper.html), ICCV, 2015.

28. Shaoqing Ren, Kaiming He, Ross Girshick, and Jian Sun, [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497), NeurIPS, 2015.

29. Joseph Redmon, Santosh Divvala, Ross Girshick, and Ali Farhadi, [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640), CVPR, 2016.

30. Wei Liu et al., [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325), ECCV, 2016.

31. Tsung-Yi Lin, Piotr Dollár, Ross Girshick, Kaiming He, Bharath Hariharan, and Serge Belongie, [Feature Pyramid Networks for Object Detection](https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html), CVPR, 2017.

32. Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár, [Focal Loss for Dense Object Detection](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html), ICCV, 2017.

33. Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick, [Mask R-CNN](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html), ICCV, 2017.

34. Nicolas Carion et al., [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872), ECCV, 2020.

35. Florian Schroff, Dmitry Kalenichenko, and James Philbin, [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://openaccess.thecvf.com/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html), CVPR, 2015.

36. Weiyang Liu et al., [SphereFace: Deep Hypersphere Embedding for Face Recognition](https://openaccess.thecvf.com/content_cvpr_2017/html/Liu_SphereFace_Deep_Hypersphere_CVPR_2017_paper.html), CVPR, 2017.

37. Hao Wang et al., [CosFace: Large Margin Cosine Loss for Deep Face Recognition](https://openaccess.thecvf.com/content_cvpr_2018/html/Wang_CosFace_Large_Margin_CVPR_2018_paper.html), CVPR, 2018.

38. Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou, [ArcFace: Additive Angular Margin Loss for Deep Face Recognition](https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html), CVPR, 2019.

39. Robert Geirhos et al., [ImageNet-Trained CNNs Are Biased Towards Texture; Increasing Shape Bias Improves Accuracy and Robustness](https://openreview.net/forum?id=Bygh9j09KX), ICLR, 2019.

40. Rosanne Liu et al., [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://proceedings.neurips.cc/paper/2018/hash/60106888f8977b71e1f15db7bc9a88d1-Abstract.html), NeurIPS, 2018.

41. PyTorch, [`torch.nn.Conv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.conv.Conv2d.html), official documentation.

42. PyTorch, [`torch.nn.ConvTranspose2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html), official documentation.

43. PyTorch, [`torch.nn.Unfold`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Unfold.html), official documentation.

44. PyTorch, [`torch.nn.AdaptiveAvgPool2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html), official documentation.

45. PyTorch, [Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html), official documentation.

46. PyTorch, [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), official documentation.

47. Adam Paszke et al., [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://proceedings.neurips.cc/paper/2019/hash/bdbca288fee7f92f2bfa9f7012727740-Abstract.html), NeurIPS, 2019.
