---
title: "The Mathematical Foundations of Artificial Intelligence"
subtitle: "Structure, Change, and Uncertainty"
summary: "The Mathematical Foundations of Artificial Intelligence: Structure, Change, and Uncertainty"
description: "The Mathematical Foundations of Artificial Intelligence: Structure, Change, and Uncertainty"
date: 2026-07-19
lastmod: 2026-07-19
weight: 80
tags: ["Mathematics for AI", "Analysis", "Linear Algebra", "Probability", "Statistics"]
draft: false
ShowToc: false
hideMeta: true
---

## Introduction

Artificial intelligence is often presented as a collection of models. A neural network has layers, an optimizer changes parameters, and a probability distribution assigns uncertainty to predictions. This description is operationally useful, but it hides the deeper structure of the subject.

Every learning system makes mathematical commitments before it processes any data. It chooses a space in which objects are represented, a notion of distance or similarity, a family of transformations, a model of uncertainty, and a rule for deciding what counts as improvement. These choices are not implementation details. They determine which patterns a model can express, which distinctions it ignores, and which conclusions can be justified from finite observations.

The standard mathematical core of artificial intelligence consists of analysis, linear algebra, probability, statistics, and optimization. Tensors provide the coordinate language used by modern software, while information theory and numerical analysis explain many of the objectives and computational failures encountered in practice.

This chapter develops those subjects as a connected system. The aim is not to reproduce a catalogue of formulas. It is to identify the mathematical objects that machine learning manipulates, the assumptions that make those manipulations valid, and the points at which an intuitive calculation must be replaced by a precise statement.

## Analysis: Change as Local Structure

Calculus enters machine learning through a simple question: how does the output of a model change when its input or parameters change?

That question appears in gradient-based training, sensitivity analysis, continuous-time models, normalizing flows, differential equations, and statistical estimation. The important idea is not differentiation as a symbolic operation. It is differentiation as local linearization.

### Limits and continuity

Let $f : \mathbb{R} \to \mathbb{R}$. The statement

<div class="display-equation">
$$
\lim_{x\to x_0} f(x)=L
$$
</div>

means that the values of $f(x)$ can be made arbitrarily close to $L$ by taking $x$ sufficiently close to $x_0$, without requiring $x=x_0$. Formally, for every $\varepsilon>0$, there exists $\delta>0$ such that

<div class="display-equation">
$$
0&lt;|x-x_0|&lt;\delta
\quad\Longrightarrow\quad
|f(x)-L|&lt;\varepsilon.
$$
</div>

The definition separates the behavior of a function near a point from its value at the point. A function may have a limit at $x_0$ even when it is not defined there.

Continuity adds compatibility between the local behavior and the actual value:

<div class="display-equation">
$$
\lim_{x\to x_0}f(x)=f(x_0).
$$
</div>

For learning systems, continuity is a minimal stability property. Small perturbations of the input produce small perturbations of the output locally. It does not guarantee robustness in a quantitative sense, because a continuous function can still change extremely rapidly. Quantitative stability requires stronger controls, such as a Lipschitz bound

<div class="display-equation">
$$
\|f(x)-f(y)\|\leq L\|x-y\|.
$$
</div>

The constant $L$ measures how strongly perturbations may be amplified. This distinction matters in adversarial robustness and numerical stability. Continuity says that sufficiently small changes remain small. A Lipschitz estimate says how small.

#### One-sided limits and discontinuities

A two-sided limit exists only when the left and right limits both exist and agree:

<div class="display-equation">
$$
\lim_{x\to x_0}f(x)=L
\quad\Longleftrightarrow\quad
\lim_{x\to x_0^-}f(x)=
\lim_{x\to x_0^+}f(x)=L.
$$
</div>

This distinction is essential at boundaries and thresholds. For $f(x)=1/x$, the one-sided limits at zero diverge with opposite signs, so there is no two-sided limit. By contrast, $\sin x/x$ is undefined at zero but has limit one there. Defining its value at zero to be one removes the discontinuity.

Whenever the relevant limits exist and the denominator does not approach zero, limits obey the familiar algebraic laws:

<div class="display-equation">
$$
\lim(f+g)=\lim f+\lim g,
\qquad
\lim(fg)=(\lim f)(\lim g),
\qquad
\lim\frac{f}{g}=\frac{\lim f}{\lim g}.
$$
</div>

Discontinuities are usefully separated by mechanism.

- A **removable discontinuity** has a finite two-sided limit, but the function is missing or assigned the wrong value at the point.
- A **jump discontinuity** has finite left and right limits that are unequal.
- An **infinite discontinuity** occurs when at least one one-sided limit diverges.
- An **oscillatory discontinuity** occurs when values continue to oscillate without settling, as in $\sin(1/x)$ near zero.

This classification matters in learning systems. A hard decision threshold creates a jump. A reciprocal or logarithmic singularity may create an infinite discontinuity. Rapid oscillation can make a model locally unpredictable even when its output remains bounded.

### Derivatives and differentials

For a scalar function, the derivative is defined by

<div class="display-equation">
$$
f'(x_0)=\lim_{h\to 0}\frac{f(x_0+h)-f(x_0)}{h},
$$
</div>

when this limit exists. Differentiability implies continuity, but continuity does not imply differentiability. The function $f(x)=|x|$ is continuous at the origin and is not differentiable there.

The derivative is most useful when read as a local approximation:

<div class="display-equation">
$$
f(x_0+h)=f(x_0)+f'(x_0)h+o(|h|).
$$
</div>

The linear term $f'(x_0)h$ is the differential. It predicts the first-order response of the function to a small perturbation. The remainder $o(|h|)$ becomes negligible relative to $|h|$ as $h\to 0$.

This interpretation extends cleanly to many variables. For a map $f:\mathbb{R}^n\to\mathbb{R}^m$, differentiability at $x$ means that there is a linear map $Df(x)$ such that

<div class="display-equation">
$$
f(x+h)=f(x)+Df(x)h+o(\|h\|).
$$
</div>

In coordinates, $Df(x)$ is the Jacobian matrix. Its entries are

<div class="display-equation">
$$
[J_f(x)]_{ij}=\frac{\partial f_i}{\partial x_j}(x).
$$
</div>

This definition is stronger than the existence of all partial derivatives. Partial derivatives describe change along coordinate axes. Total differentiability requires one linear map to approximate the function uniformly in every direction. A function can possess every partial derivative at a point and still fail to be differentiable there.

For a scalar-valued function $f:\mathbb{R}^n\to\mathbb{R}$, the derivative $Df(x)$ is naturally a linear functional. Once the Euclidean inner product is chosen, that functional can be represented by the gradient:

<div class="display-equation">
$$
Df(x)h=\langle \nabla f(x),h\rangle.
$$
</div>

The gradient therefore depends on the geometry used to identify linear functionals with vectors. Under a different inner product or on a curved space, the coordinate vector called the gradient changes even when the underlying differential does not. This is the mathematical idea behind natural gradients and Riemannian optimization.

#### Differentiation rules and differential algebra

For differentiable scalar functions, the elementary rules are

<div class="display-equation">
$$
(f+g)'=f'+g',
\qquad
(fg)'=f'g+fg',
\qquad
\left(\frac{f}{g}\right)'=\frac{f'g-fg'}{g^2},
$$
</div>

together with the chain rule $(f\circ g)'(x)=f'(g(x))g'(x)$. Written in differential form, the same identities become

<div class="display-equation">
$$
d(u+v)=du+dv,
\qquad
d(uv)=u\,dv+v\,du,
\qquad
d\left(\frac{u}{v}\right)=\frac{v\,du-u\,dv}{v^2}.
$$
</div>

The differential is not merely another notation for a derivative. It is the linear part of an increment. If $y=f(x)$, then

<div class="display-equation">
$$
\Delta y=f(x+\Delta x)-f(x)
=f'(x)\Delta x+o(|\Delta x|),
$$
</div>

so $dy=f'(x)dx$ is the first-order approximation, while $\Delta y$ is the actual change. For the area $S=\pi r^2$ of a circle,

<div class="display-equation">
$$
\Delta S=2\pi r\Delta r+\pi(\Delta r)^2,
\qquad
dS=2\pi r\,dr.
$$
</div>

The quadratic remainder explains why the differential becomes accurate for small radius changes.

First differentials are form-invariant under substitution. If $y=f(u)$ and $u=g(x)$, then

<div class="display-equation">
$$
dy=f'(u)\,du=f'(g(x))g'(x)\,dx.
$$
</div>

This immediately yields the inverse-function and parametric-curve formulas, whenever the relevant denominators are nonzero:

<div class="display-equation">
$$
\frac{dx}{dy}=\frac{1}{dy/dx},
\qquad
\frac{dy}{dx}=\frac{dy/dt}{dx/dt}.
$$
</div>

Higher derivatives describe higher-order change, but higher differentials are not generally form-invariant under nonlinear substitutions. It is also useful to reserve **smooth** for functions with derivatives of all orders, or at least to state explicitly whether $C^1$, $C^2$, or $C^k$ regularity is intended. Merely being differentiable once is not normally called smooth in analysis.

### The chain rule and computational graphs

Suppose $g:\mathbb{R}^n\to\mathbb{R}^p$ and $f:\mathbb{R}^p\to\mathbb{R}^m$. The chain rule is a composition rule for local linear models:

<div class="display-equation">
$$
D(f\circ g)(x)=Df(g(x))Dg(x).
$$
</div>

Backpropagation is an efficient organization of this rule. A neural network is represented as a computational graph, and local derivatives are composed from the output back toward the parameters.

Modern automatic differentiation does not manipulate symbolic derivative expressions in the ordinary algebraic sense, nor does it approximate derivatives by finite differences. It applies the chain rule to elementary numerical operations recorded by a program. Forward mode efficiently computes Jacobian-vector products, while reverse mode efficiently computes vector-Jacobian products. When a model has many parameters and a scalar loss, reverse mode is usually the relevant choice because it obtains the full parameter gradient with a cost comparable to a small number of forward evaluations.

This explains why software can differentiate a large program without constructing its full Jacobian. The program propagates only the contractions needed by the final scalar objective.

### Second-order structure

When $f:\mathbb{R}^n\to\mathbb{R}$ is twice differentiable, its Hessian is

<div class="display-equation">
$$
H_f(x)=\left[\frac{\partial^2 f}{\partial x_i\partial x_j}(x)\right]_{i,j}.
$$
</div>

The second-order Taylor approximation is

<div class="display-equation">
$$
f(x+h)=f(x)+\nabla f(x)^\top h+\frac{1}{2}h^\top H_f(x)h+o(\|h\|^2).
$$
</div>

The gradient gives the local direction of first-order change. The Hessian describes local curvature. Positive curvature in every direction indicates a locally convex shape, negative curvature in every direction indicates a locally concave shape, and mixed signs indicate a saddle geometry.

For a twice differentiable function on a convex domain, convexity is equivalent to

<div class="display-equation">
$$
H_f(x)\succeq 0
$$
</div>

for every $x$ in the domain. Without twice differentiability, convexity must be defined through line segments:

<div class="display-equation">
$$
f(\lambda x+(1-\lambda)y)
\leq
\lambda f(x)+(1-\lambda)f(y),
\qquad \lambda\in[0,1].
$$
</div>

The distinction matters because many useful convex objectives, including absolute-value penalties, are not differentiable everywhere.

### Integration, expectation, and change of variables

Differentiation studies local change. Integration aggregates local quantities.

#### Antiderivatives and the two meanings of integration

An antiderivative of $f$ is a function $F$ satisfying $F'=f$. The indefinite integral denotes the full family

<div class="display-equation">
$$
\int f(x)\,dx=F(x)+C.
$$
</div>

The constant $C$ is necessary because differentiation erases additive constants. An indefinite integral is therefore an inverse operation to differentiation only up to a constant. A definite integral has a different origin: it is a limit of signed sums. The fundamental theorem of calculus is what connects these two constructions.

Two indispensable computational rules follow from the chain and product rules. Substitution gives

<div class="display-equation">
$$
\int f(g(x))g'(x)\,dx
=
\int f(u)\,du,
$$
</div>

while integration by parts gives

<div class="display-equation">
$$
\int u\,dv=uv-\int v\,du.
$$
</div>

For example,

<div class="display-equation">
$$
\int x^3\log x\,dx
=
\frac{x^4}{4}\log x-\frac{x^4}{16}+C.
$$
</div>

For a continuous function on a closed interval, the fundamental theorem of calculus connects derivatives and integrals:

<div class="display-equation">
$$
\int_a^b f(x)\,dx=F(b)-F(a),
\qquad F'(x)=f(x).
$$
</div>

The definite integral is a signed accumulation, not always an ordinary geometric area. Contributions below the horizontal axis enter with negative sign.

The Riemann integral is sufficient for many elementary calculations, but probability and modern analysis are usually formulated with the Lebesgue integral. A bounded function on a compact interval is Riemann integrable if and only if its set of discontinuities has Lebesgue measure zero. Boundedness is part of this criterion.

Improper integrals must be defined through limits. For example,

<div class="display-equation">
$$
\int_a^\infty f(x)\,dx
=
\lim_{b\to\infty}\int_a^b f(x)\,dx,
$$
</div>

provided that the limit exists and is finite. Bounded partial integrals alone do not guarantee convergence. The partial integrals of $\sin x$, for example, remain bounded but do not converge as the upper limit tends to infinity.

There are two basic kinds of improper integral. An integral over an unbounded domain is defined by sending a finite endpoint to infinity. An integral with a singularity is defined by approaching the singular point from within the domain. If $c\in(a,b)$ is singular, both one-sided integrals must converge separately:

<div class="display-equation">
$$
\int_a^b f(x)\,dx
=
\lim_{u\to c^-}\int_a^u f(x)\,dx
+
\lim_{v\to c^+}\int_v^b f(x)\,dx.
$$
</div>

The Cauchy convergence criterion gives the correct test at infinity: $\int_a^\infty f$ converges if and only if for every $\varepsilon>0$ there is $M$ such that

<div class="display-equation">
$$
\left|\int_u^v f(x)\,dx\right|<\varepsilon
\qquad\text{for all }v>u>M.
$$
</div>

A Cauchy principal value is a different object. Symmetric cancellation can make

<div class="display-equation">
$$
\operatorname{p.v.}\int_{-\infty}^{\infty}f(x)\,dx
=
\lim_{R\to\infty}\int_{-R}^{R}f(x)\,dx
$$
</div>

exist even when the ordinary improper integral diverges. Thus principal-value existence is weaker, not stronger, than separate convergence of the two tails.

#### Several variables: joint limits, partial derivatives, and repeated integrals

For $f:\mathbb{R}^2\to\mathbb{R}$, a joint limit requires the same value along every path approaching $(x_0,y_0)$. Iterated limits instead take one coordinate limit and then the other:

<div class="display-equation">
$$
\lim_{y\to y_0}\lim_{x\to x_0}f(x,y),
\qquad
\lim_{x\to x_0}\lim_{y\to y_0}f(x,y).
$$
</div>

Agreement of these two iterated limits alone does not guarantee the joint limit. A diagonal or curved path may behave differently. This is why total differentiability is stronger than the existence of coordinatewise partial derivatives.

For a rectangular domain $D=[a,b]\times[c,d]$, a repeated integral has the form

<div class="display-equation">
$$
\int_a^b\int_c^d f(x,y)\,dy\,dx.
$$
</div>

Under the hypotheses of Fubini or Tonelli, it represents the corresponding double integral and the order may be exchanged. A change to polar coordinates,

<div class="display-equation">
$$
x=r\cos\theta,
\qquad
y=r\sin\theta,
$$
</div>

introduces the Jacobian factor $r$:

<div class="display-equation">
$$
\iint_D f(x,y)\,dx\,dy
=
\int_{\theta_1}^{\theta_2}\int_{r_1(\theta)}^{r_2(\theta)}
f(r\cos\theta,r\sin\theta)\,r\,dr\,d\theta.
$$
</div>

In several variables, integration introduces an additional question: when may the order of integration be exchanged? Tonelli's theorem permits this for nonnegative measurable functions, even when the integral is infinite. Fubini's theorem permits it for integrable functions. These conditions are not technical decoration. They prevent invalid rearrangements of infinite positive and negative contributions.

Under a smooth invertible change of variables $y=T(x)$, volume is corrected by the absolute Jacobian determinant:

<div class="display-equation">
$$
\int_{T(U)} g(y)\,dy
=
\int_U g(T(x))\left|\det J_T(x)\right|\,dx.
$$
</div>

This formula is central to probability densities, normalizing flows, Bayesian transformations, and geometric integration. The determinant measures local volume distortion.

## Linear Algebra: Representation and Transformation

Machine learning represents observations and parameters with coordinates, but the underlying objects are vectors, subspaces, and transformations. Linear algebra makes those structures explicit.

### Vectors, matrices, and affine maps

A vector is an element of a vector space. A coordinate array is one representation of that vector relative to a chosen basis. A matrix similarly represents a linear map after bases have been selected in the input and output spaces.

If $A\in\mathbb{R}^{m\times n}$ and $x\in\mathbb{R}^n$, then

<div class="display-equation">
$$
y=Ax
$$
</div>

defines a linear transformation from $\mathbb{R}^n$ to $\mathbb{R}^m$. Linearity means

<div class="display-equation">
$$
A(\alpha x+\beta z)=\alpha Ax+\beta Az.
$$
</div>

Translations are not linear because they do not preserve the origin. A map of the form

<div class="display-equation">
$$
x\mapsto Ax+b
$$
</div>

is affine. Most neural-network layers are affine maps followed by nonlinear transformations.

Matrix multiplication represents composition. If $B$ acts first and $A$ acts second, the composite map is $AB$. The order matters because matrix multiplication is generally not commutative.

#### From linear systems to matrix form

A system of $m$ linear equations in $n$ unknowns can be compressed into

<div class="display-equation">
$$
Ax=b,
\qquad
A\in\mathbb{R}^{m\times n},
\quad
x\in\mathbb{R}^n,
\quad
b\in\mathbb{R}^m.
$$
</div>

The matrix $A$ stores the coefficients, and the augmented matrix $[A\mid b]$ stores both coefficients and targets. Componentwise,

<div class="display-equation">
$$
(Ax)_i=\sum_{j=1}^n a_{ij}x_j.
$$
</div>

More generally, if $A\in\mathbb{R}^{m\times n}$ and $B\in\mathbb{R}^{n\times k}$, then

<div class="display-equation">
$$
(AB)_{ij}=\sum_{\ell=1}^n a_{i\ell}b_{\ell j}.
$$
</div>

The inner dimensions must agree. This shape rule is the coordinate form of composition: the output space of the first map must match the input space of the second. Matrix multiplication is associative and distributive, while

<div class="display-equation">
$$
(AB)^\top=B^\top A^\top.
$$
</div>

The zero matrix maps every vector to zero, and the identity matrix $I$ leaves every vector unchanged. These are the additive and compositional neutral elements of matrix algebra.

#### Determinants and inverses

For a square matrix, the determinant measures oriented volume scaling. In two dimensions,

<div class="display-equation">
$$
\det\begin{pmatrix}a&b\\c&d\end{pmatrix}=ad-bc.
$$
</div>

A zero determinant means that some dimension has collapsed: the columns are linearly dependent and the transformation is not invertible. Elementary row operations expose the determinant efficiently:

- exchanging two rows changes its sign;
- multiplying one row by $c$ multiplies the determinant by $c$;
- adding a multiple of one row to another leaves it unchanged;
- a triangular matrix has determinant equal to the product of its diagonal entries.

For an $n\times n$ matrix, the following statements are equivalent:

<div class="display-equation">
$$
\det(A)\neq 0
\quad\Longleftrightarrow\quad
\operatorname{rank}(A)=n
\quad\Longleftrightarrow\quad
\ker(A)=\{0\}
\quad\Longleftrightarrow\quad
A^{-1}\text{ exists}.
$$
</div>

The inverse is defined by $AA^{-1}=A^{-1}A=I$. For a $2\times2$ matrix,

<div class="display-equation">
$$
\begin{pmatrix}a&b\\c&d\end{pmatrix}^{-1}
=
\frac{1}{ad-bc}
\begin{pmatrix}d&-b\\-c&a\end{pmatrix}.
$$
</div>

For larger matrices, Gaussian elimination transforms $[A\mid I]$ into $[I\mid A^{-1}]$ when the inverse exists. This construction is valuable theoretically. In numerical machine learning, however, solving a system or using a factorization is usually preferable to explicitly constructing the inverse.

### Rank, null spaces, and identifiability

The rank of a matrix is the dimension of its image:

<div class="display-equation">
$$
\operatorname{rank}(A)=\dim\{Ax:x\in\mathbb{R}^n\}.
$$
</div>

It equals both the dimension of the column space and the dimension of the row space. The null space

<div class="display-equation">
$$
\ker(A)=\{x:Ax=0\}
$$
</div>

contains directions that the transformation erases. The rank-nullity theorem states

<div class="display-equation">
$$
\operatorname{rank}(A)+\dim\ker(A)=n.
$$
</div>

These ideas describe identifiability. If two parameter vectors differ by an element of the null space, a linear model cannot distinguish them from its outputs alone.

For the system $Ax=b$, consistency is determined by whether $b$ lies in the column space of $A$. A square matrix is invertible exactly when it has full rank, a trivial null space, and a nonzero determinant. For rectangular or rank-deficient systems, the Moore-Penrose pseudoinverse provides a canonical least-squares solution:

<div class="display-equation">
$$
x^+=A^+b.
$$
</div>

When solutions are not unique, $A^+b$ is the solution with minimum Euclidean norm.

The augmented matrix gives a complete classification of an exact linear system:

<div class="display-equation">
$$
\begin{aligned}
\operatorname{rank}(A)&=\operatorname{rank}([A\mid b])=n
&&\Longrightarrow &&\text{one solution},\\
\operatorname{rank}(A)&=\operatorname{rank}([A\mid b])&lt;n
&&\Longrightarrow &&\text{infinitely many solutions},\\
\operatorname{rank}(A)&lt;\operatorname{rank}([A\mid b])
&&\Longrightarrow &&\text{no solution}.
\end{aligned}
$$
</div>

This is not merely a classroom classification. In regression, collinear features reduce rank and make parameters non-identifiable. Ridge regularization replaces $X^\top X$ with $X^\top X+\lambda I$, improving invertibility and conditioning, although the resulting estimate solves a modified problem rather than recovering information that the data never contained.

### Least squares without explicit inversion

Linear regression is often written as

<div class="display-equation">
$$
\widehat{w}=(X^\top X)^{-1}X^\top y.
$$
</div>

This formula is mathematically informative and often computationally unwise. Forming $X^\top X$ squares the spectral condition number when $X$ has full column rank:

<div class="display-equation">
$$
\kappa_2(X^\top X)=\kappa_2(X)^2.
$$
</div>

Explicit inversion also performs more work than solving the system that is actually needed. Numerical implementations usually solve least-squares problems with QR factorization or SVD. Iterative methods can be preferable for large sparse systems, and conjugate gradients apply directly only to symmetric positive-definite systems.

The condition number

<div class="display-equation">
$$
\kappa_2(A)=\|A\|_2\|A^{-1}\|_2
$$
</div>

measures sensitivity of an invertible linear system to perturbations. Ill-conditioning is a property of the problem. Numerical instability is a property of an algorithm. A stable algorithm cannot recover information that an ill-conditioned problem has effectively lost, but an unstable algorithm can lose accuracy even on a well-conditioned problem.

### Eigenvalues and the spectral theorem

An eigenvector is a nonzero vector whose direction is preserved by a square linear transformation:

<div class="display-equation">
$$
Av=\lambda v.
$$
</div>

Not every matrix has enough linearly independent eigenvectors to be diagonalized. If it does, then

<div class="display-equation">
$$
A=V\Lambda V^{-1}.
$$
</div>

Real symmetric matrices have a stronger property. The spectral theorem guarantees an orthonormal eigenbasis:

<div class="display-equation">
$$
A=Q\Lambda Q^\top.
$$
</div>

Their eigenvalues are real, and eigenvectors associated with distinct eigenvalues can be chosen orthogonal.

Eigenvalues are found from the characteristic equation

<div class="display-equation">
$$
\det(A-\lambda I)=0.
$$
</div>

For example, with

<div class="display-equation">
$$
A=\begin{pmatrix}4&1\\2&3\end{pmatrix},
$$
</div>

the characteristic polynomial is $\lambda^2-7\lambda+10$, so the eigenvalues are $5$ and $2$. Solving $(A-5I)v=0$ gives a direction proportional to $(1,1)^\top$, while solving $(A-2I)v=0$ gives one proportional to $(1,-2)^\top$. The trace and determinant provide useful checks:

<div class="display-equation">
$$
\operatorname{tr}(A)=\sum_i\lambda_i,
\qquad
\det(A)=\prod_i\lambda_i.
$$
</div>

If $A=V\Lambda V^{-1}$, then powers and analytic matrix functions reduce to scalar operations on eigenvalues:

<div class="display-equation">
$$
A^k=V\Lambda^kV^{-1}.
$$
</div>

This is central in Markov chains, linear dynamical systems, stability analysis, and repeated message propagation.

Positive-semidefinite matrices satisfy

<div class="display-equation">
$$
x^\top Ax\geq 0
$$
</div>

for every $x$. Covariance matrices, Gram matrices, and many Hessians have this structure. For a symmetric matrix, positive semidefiniteness is equivalent to all eigenvalues being nonnegative.

### Singular value decomposition

Eigendecomposition is restricted to square matrices and may fail to provide a basis. Singular value decomposition applies to every real matrix. In reduced form, if $A\in\mathbb{R}^{m\times n}$ has rank $r$, then

<div class="display-equation">
$$
A=U_r\Sigma_r V_r^\top,
$$
</div>

where the columns of $U_r$ and $V_r$ are orthonormal and

<div class="display-equation">
$$
\Sigma_r=\operatorname{diag}(\sigma_1,\ldots,\sigma_r),
\qquad
\sigma_1\geq\cdots\geq\sigma_r>0.
$$
</div>

The singular values measure the principal amounts of stretching performed by the map. The right singular vectors identify input directions, and the left singular vectors identify the corresponding output directions.

The decomposition is connected to two symmetric positive-semidefinite matrices:

<div class="display-equation">
$$
A^\top A=V\Sigma^2V^\top,
\qquad
AA^\top=U\Sigma^2U^\top.
$$
</div>

Thus the nonzero singular values are square roots of the nonzero eigenvalues of either product. For $\sigma_i>0$, the singular vectors satisfy

<div class="display-equation">
$$
Av_i=\sigma_i u_i,
\qquad
A^\top u_i=\sigma_i v_i.
$$
</div>

This construction also reveals the four fundamental subspaces. Right singular vectors with positive singular values span the row space, left singular vectors with positive singular values span the column space, and zero singular values identify null directions.

Truncating the decomposition after $k$ components gives

<div class="display-equation">
$$
A_k=U_k\Sigma_kV_k^\top.
$$
</div>

The Eckart-Young-Mirsky theorem states that $A_k$ is a best rank-$k$ approximation to $A$ under both the spectral norm and the Frobenius norm. The approximation errors are

<div class="display-equation">
$$
\|A-A_k\|_2=\sigma_{k+1}
$$
</div>

and

<div class="display-equation">
$$
\|A-A_k\|_F^2=\sum_{i>k}\sigma_i^2.
$$
</div>

This result explains compression and denoising when a matrix has rapidly decaying singular values. It does not by itself solve matrix completion with missing entries. Standard SVD assumes a fully observed matrix. Matrix completion requires an observation model, structural assumptions such as low rank or incoherence, and an optimization or probabilistic procedure for handling unobserved entries.

### Principal component analysis

Let $X\in\mathbb{R}^{n\times d}$ be a centered data matrix. PCA finds orthogonal directions that successively maximize projected sample variance. If

<div class="display-equation">
$$
X=U\Sigma V^\top,
$$
</div>

then the columns of $V$ are principal directions. Equivalently, they are eigenvectors of the sample covariance matrix

<div class="display-equation">
$$
C=\frac{1}{n-1}X^\top X.
$$
</div>

The variance explained by the $i$th component is proportional to $\sigma_i^2$. PCA is therefore both a spectral method and a low-rank approximation method.

PCA preserves directions of high variance, not necessarily directions that are useful for prediction or causally meaningful. Feature scaling, centering, outliers, and the choice between covariance and correlation matrices can substantially alter the result.

### Norms and duality

A norm measures size while satisfying positivity, absolute homogeneity, and the triangle inequality. For $p\geq 1$,

<div class="display-equation">
$$
\|x\|_p=\left(\sum_{i=1}^n |x_i|^p\right)^{1/p}.
$$
</div>

Important special cases are $\|x\|_1$, $\|x\|_2$, and $\|x\|_\infty$.

Different norms encode different geometries. They change nearest neighbors, regularization penalties, robustness regions, and the meaning of a small perturbation. In finite-dimensional spaces all norms are topologically equivalent, but their constants and high-dimensional geometry can differ greatly.

The dual norm is defined by

<div class="display-equation">
$$
\|y\|_*=\sup_{\|x\|\leq 1}x^\top y.
$$
</div>

The dual of the $p$-norm is the $q$-norm when $1/p+1/q=1$. This relationship underlies Holder's inequality and many optimization bounds.

For matrices, the Frobenius norm is

<div class="display-equation">
$$
\|A\|_F=\left(\sum_{i,j}a_{ij}^2\right)^{1/2}
=\left(\sum_i\sigma_i^2\right)^{1/2}.
$$
</div>

The spectral norm is

<div class="display-equation">
$$
\|A\|_2=\sup_{x\neq 0}\frac{\|Ax\|_2}{\|x\|_2}=\sigma_1.
$$
</div>

The Frobenius norm measures aggregate energy. The spectral norm measures the largest amplification of any unit input.

## Tensors and Multilinear Structure

Modern frameworks call scalars, vectors, matrices, and higher-dimensional arrays tensors. This computational convention is useful, but a mathematical tensor is more than an array. It is a multilinear object whose coordinates transform in a prescribed way when bases change. The array stores coordinates; the tensor is the basis-independent object those coordinates represent.

For many machine-learning calculations, the array viewpoint is sufficient. A color image may have axes for height, width, and channel. A video adds time. A batch adds a sample axis. The meaning of each axis matters, even when two tensors have the same shape.

Operations such as reshaping and transposing reorganize coordinates. A contraction sums over paired indices and generalizes matrix multiplication. If $X_{ijk}$ and $W_{k\ell}$ are contracted over $k$, then

<div class="display-equation">
$$
Y_{ij\ell}=\sum_k X_{ijk}W_{k\ell}.
$$
</div>

Broadcasting is a software convention for extending lower-order arrays across selected axes. It is not an independent algebraic law, so silent broadcasting errors can produce shape-correct but semantically incorrect computations.

Higher-order tensor rank is more subtle than matrix rank. Matrices have a canonical SVD with strong optimality properties. For tensors of order three or higher, several inequivalent rank notions and decompositions coexist. CP and Tucker decompositions are important examples, and best low-rank approximation can behave differently from the matrix case.

### Matrix and tensor derivatives

For a scalar function $f(X)$ of a matrix $X$, the gradient is commonly arranged with the same shape as $X$ under the Frobenius inner product:

<div class="display-equation">
$$
df=\langle \nabla_X f,dX\rangle_F
=\operatorname{tr}\left((\nabla_X f)^\top dX\right).
$$
</div>

This differential notation avoids many row-versus-column ambiguities. For example,

<div class="display-equation">
$$
f(x)=a^\top x
\quad\Longrightarrow\quad
df=a^\top dx,
\qquad
\nabla_x f=a.
$$
</div>

For a quadratic form,

<div class="display-equation">
$$
f(x)=x^\top Ax,
$$
</div>

we obtain

<div class="display-equation">
$$
df=x^\top A\,dx+x^\top A^\top dx,
$$
</div>

and therefore

<div class="display-equation">
$$
\nabla_x f=(A+A^\top)x.
$$
</div>

If $A$ is symmetric, this becomes $2Ax$.

Derivative shape follows from the input-output pair:

- a scalar with respect to a vector gives a gradient vector;
- a scalar with respect to a matrix gives a matrix of the same shape;
- a vector with respect to a vector gives a Jacobian matrix;
- a matrix with respect to a matrix gives a fourth-order coordinate derivative;
- in general, an order-$p$ output differentiated with respect to an order-$q$ input produces an order-$(p+q)$ coordinate object before contractions are applied.

For $f:\mathbb{R}^n\to\mathbb{R}^m$, the Jacobian convention used here is

<div class="display-equation">
$$
[J_f(x)]_{ij}=\frac{\partial f_i}{\partial x_j},
\qquad
J_f(x)\in\mathbb{R}^{m\times n}.
$$
</div>

Consider the affine map $f(W)=W^\top x+b$, where $W\in\mathbb{R}^{n\times m}$. Differentiating the vector output with respect to the matrix $W$ formally produces a third-order object. If a scalar loss $L$ follows the affine map, however, the chain rule contracts that object with $\nabla_f L$. The result is the familiar matrix

<div class="display-equation">
$$
\nabla_W L=x(\nabla_f L)^\top,
$$
</div>

which has the same shape as $W$. The apparent reduction in tensor order is a contraction, not a disappearance of mathematical structure.

For scalar-valued functions of a tensor, linearity and product rules retain their familiar form:

<div class="display-equation">
$$
\nabla_X(\alpha f+\beta g)
=
\alpha\nabla_X f+\beta\nabla_X g,
$$
</div>

<div class="display-equation">
$$
\nabla_X(fg)=f\nabla_X g+g\nabla_X f.
$$
</div>

The chain rule is likewise a composition of derivatives, but its coordinate implementation requires the correct index contraction. Writing it as ordinary multiplication without checking dimensions can conceal transposes or entire tensor axes.

When both the input and output are tensors, the full coordinate derivative may itself have high order. In practice, automatic differentiation rarely materializes it. It computes the Jacobian-vector or vector-Jacobian products required by the surrounding computation. The slogan that a gradient has the same shape as a parameter is correct for a scalar objective under a chosen inner product. It is not a general statement about derivatives of tensor-valued functions.

## Probability: A Language for Uncertainty

Learning from data requires reasoning about events that were not observed and samples that have not yet arrived. Probability provides the formal language for that reasoning.

### Probability spaces and random variables

A probability model consists of a sample space $\Omega$, a collection of measurable events $\mathcal{F}$, and a probability measure $\mathbb{P}$. A random variable is a measurable function

<div class="display-equation">
$$
X:\Omega\to\mathbb{R}.
$$
</div>

This definition distinguishes an uncertain outcome from the numerical value used to represent it.

For a discrete random variable, the probability mass function is

<div class="display-equation">
$$
p_X(x)=\mathbb{P}(X=x),
\qquad
\sum_x p_X(x)=1.
$$
</div>

For a continuous random variable with density $p_X$, probabilities are obtained by integration:

<div class="display-equation">
$$
\mathbb{P}(X\in A)=\int_A p_X(x)\,dx.
$$
</div>

A density value is not itself the probability of a point. For a continuous distribution, an individual point usually has probability zero even when the density there is positive.

The cumulative distribution function

<div class="display-equation">
$$
F_X(x)=\mathbb{P}(X\leq x)
$$
</div>

exists for every real-valued random variable. It is nondecreasing, right-continuous, and approaches zero and one at the two ends of the real line.

### Conditioning and Bayes' rule

For events $A$ and $B$ with $\mathbb{P}(B)>0$,

<div class="display-equation">
$$
\mathbb{P}(A\mid B)=\frac{\mathbb{P}(A\cap B)}{\mathbb{P}(B)}.
$$
</div>

Bayes' rule follows:

<div class="display-equation">
$$
\mathbb{P}(A\mid B)
=
\frac{\mathbb{P}(B\mid A)\mathbb{P}(A)}{\mathbb{P}(B)}.
$$
</div>

For a partition $A_1,\ldots,A_k$ of the sample space,

<div class="display-equation">
$$
\mathbb{P}(A_i\mid B)
=
\frac{\mathbb{P}(B\mid A_i)\mathbb{P}(A_i)}
{\sum_j\mathbb{P}(B\mid A_j)\mathbb{P}(A_j)}.
$$
</div>

The denominator is the marginal probability of the evidence. In statistical language, it is also called the evidence or marginal likelihood.

Diagnostic testing illustrates the base-rate effect. A highly sensitive and specific test can still have a modest positive predictive value when the condition is rare. Repeating the test does not justify multiplying likelihoods unless the test outcomes are conditionally independent given the true state. Shared laboratory conditions, systematic calibration errors, and correlated sampling can violate that assumption.

Suppose a test has sensitivity $0.95$, specificity $0.98$, and the prevalence of the condition is $0.01$. Let $D$ denote the condition and $+$ a positive result. Then

<div class="display-equation">
$$
\mathbb{P}(D\mid +)
=
\frac{0.95\times0.01}
{0.95\times0.01+0.02\times0.99}
\approx 0.324.
$$
</div>

The posterior probability is only about $32.4\%$ despite the apparently strong test, because false positives are applied to a much larger healthy population. If a second result is conditionally independent with the same operating characteristics, the first posterior becomes the second prior:

<div class="display-equation">
$$
\mathbb{P}(D\mid +,+)
=
\frac{0.95\times0.324}
{0.95\times0.324+0.02\times(1-0.324)}
\approx 0.958.
$$
</div>

The numerical jump is real only under the conditional-independence model. Bayes' rule does not create independence; that assumption must come from the data-generating process.

### Expectation, variance, and covariance

Expectation is integration with respect to a probability distribution:

<div class="display-equation">
$$
\mathbb{E}[g(X)]=\int g(x)\,dP_X(x).
$$
</div>

This notation covers discrete and continuous distributions in one expression. Expectation is linear whenever the relevant expectations exist:

<div class="display-equation">
$$
\mathbb{E}[aX+bY]=a\mathbb{E}[X]+b\mathbb{E}[Y].
$$
</div>

Variance measures quadratic dispersion:

<div class="display-equation">
$$
\operatorname{Var}(X)
=
\mathbb{E}\left[(X-\mathbb{E}X)^2\right]
=
\mathbb{E}[X^2]-(\mathbb{E}X)^2.
$$
</div>

Covariance measures linear co-variation:

<div class="display-equation">
$$
\operatorname{Cov}(X,Y)
=
\mathbb{E}\left[(X-\mathbb{E}X)(Y-\mathbb{E}Y)\right].
$$
</div>

Independence implies zero covariance when second moments exist. Zero covariance does not generally imply independence. The converse holds for jointly Gaussian variables, but not for arbitrary distributions.

For a random vector $X$, the covariance matrix is

<div class="display-equation">
$$
\Sigma=\mathbb{E}\left[(X-\mu)(X-\mu)^\top\right].
$$
</div>

It is symmetric and positive semidefinite because

<div class="display-equation">
$$
v^\top\Sigma v=\operatorname{Var}(v^\top X)\geq 0.
$$
</div>

### Independence, laws of large numbers, and concentration

The i.i.d. assumption means that observations share one distribution and are mutually independent. It enables much of classical statistical theory, but it is not a universal property of machine-learning data. Time series, reinforcement learning, bandit feedback, graph data, adaptive experiments, and distribution shift all introduce dependence or nonstationarity.

Under suitable assumptions, a law of large numbers states that a sample average approaches its expectation. A central limit theorem describes the scaled fluctuations of that average by an approximately Gaussian distribution. These are asymptotic statements. They do not directly say how accurate an estimate is at a given finite sample size.

Concentration inequalities provide nonasymptotic control. For example, if independent random variables are bounded, Hoeffding-type bounds show that deviations of the sample mean decay exponentially with the sample size. More refined bounds use variance, martingale structure, or distributional tails. This distinction is central in learning theory and sequential decision-making, where finite-time guarantees matter more than eventual convergence.

## Statistical Inference: From Data to Claims

Probability begins with a model and derives consequences. Statistics begins with observations and asks what can be learned about the model, its parameters, or future data.

### Likelihood is not probability over parameters

Suppose data $x=(x_1,\ldots,x_n)$ are modeled by a family $p(x\mid\theta)$. With the observed data held fixed, the likelihood is

<div class="display-equation">
$$
L(\theta;x)=p(x\mid\theta).
$$
</div>

The likelihood is a function of $\theta$, but it is not generally a normalized probability distribution over $\theta$. Multiplying it by a prior and normalizing produces a posterior distribution.

For conditionally independent observations,

<div class="display-equation">
$$
L(\theta;x)=\prod_{i=1}^n p(x_i\mid\theta),
$$
</div>

and the log-likelihood is

<div class="display-equation">
$$
\ell(\theta;x)=\sum_{i=1}^n\log p(x_i\mid\theta).
$$
</div>

Logarithms turn products into sums and improve numerical stability. They do not change the maximizer because the logarithm is strictly increasing.

### Maximum likelihood, MAP, and Bayesian inference

Maximum-likelihood estimation chooses

<div class="display-equation">
$$
\widehat{\theta}_{\mathrm{MLE}}
=
\arg\max_\theta p(x\mid\theta).
$$
</div>

Its behavior depends on identifiability, regularity conditions, model specification, and sample size. An MLE may fail to exist, may not be unique, or may lie at the boundary of the parameter space.

Given a prior $p(\theta)$, Bayes' rule gives

<div class="display-equation">
$$
p(\theta\mid x)
=
\frac{p(x\mid\theta)p(\theta)}{p(x)}.
$$
</div>

The maximum a posteriori estimate is

<div class="display-equation">
$$
\widehat{\theta}_{\mathrm{MAP}}
=
\arg\max_\theta p(\theta\mid x)
=
\arg\max_\theta\left[\log p(x\mid\theta)+\log p(\theta)\right].
$$
</div>

MAP is one summary of the posterior, not the whole of Bayesian inference. A full Bayesian analysis propagates posterior uncertainty into predictions:

<div class="display-equation">
$$
p(x_{\mathrm{new}}\mid x)
=
\int p(x_{\mathrm{new}}\mid\theta)p(\theta\mid x)\,d\theta.
$$
</div>

Priors need not be understood as claims that a physical parameter changes randomly. They can encode uncertainty about a fixed but unknown quantity. Their influence should be examined through sensitivity analysis, especially when data are limited.

#### A pooled-testing example: why MLE and MAP differ

Consider a pool of ten independent individuals. The infection prevalence is $0.1$, test sensitivity is $0.95$, and specificity is one. Let $X$ be the number of infected individuals and let $B$ denote a positive pooled result. Conditional on $X=x>0$, the test misses every infected individual with probability $0.05^x$, so

<div class="display-equation">
$$
\mathbb{P}(B\mid X=x)=1-0.05^x,
\qquad
\mathbb{P}(B\mid X=0)=0.
$$
</div>

As a function of $x$, this likelihood increases monotonically. If $x$ alone is treated as the parameter, maximum likelihood therefore chooses $\widehat{x}_{\mathrm{MLE}}=10$. This answer is mathematically consistent with the likelihood and practically implausible because the likelihood ignores how rare ten simultaneous infections are.

The prevalence supplies a binomial prior:

<div class="display-equation">
$$
\mathbb{P}(X=x)
=
\binom{10}{x}0.1^x0.9^{10-x}.
$$
</div>

The posterior mass is proportional to

<div class="display-equation">
$$
\mathbb{P}(X=x\mid B)
\propto
(1-0.05^x)\binom{10}{x}0.1^x0.9^{10-x},
$$
</div>

whose mode is $x=1$. The contrast is not evidence that MAP is universally better. It shows that MLE and MAP answer different questions and that a sparse-data result can be dominated by modeling choices.

#### A continuous example: exponential-scale estimation

Let $X_1,\ldots,X_n$ be independent with density

<div class="display-equation">
$$
p(x\mid\theta)
=
\frac{1}{\theta}e^{-x/\theta}\mathbf{1}\{x\geq0\},
\qquad \theta>0.
$$
</div>

Nonnegativity is immediate, and normalization follows from

<div class="display-equation">
$$
\int_0^\infty \frac{1}{\theta}e^{-x/\theta}\,dx=1.
$$
</div>

The likelihood and log-likelihood are

<div class="display-equation">
$$
L(\theta)
=
\theta^{-n}\exp\left(-\frac{1}{\theta}\sum_{i=1}^n x_i\right),
$$
</div>

<div class="display-equation">
$$
\ell(\theta)
=
-n\log\theta-\frac{1}{\theta}\sum_{i=1}^n x_i.
$$
</div>

Differentiating gives

<div class="display-equation">
$$
\ell'(\theta)
=
-\frac{n}{\theta}
+\frac{1}{\theta^2}\sum_{i=1}^n x_i.
$$
</div>

Setting this derivative to zero yields

<div class="display-equation">
$$
\widehat{\theta}_{\mathrm{MLE}}
=
\frac{1}{n}\sum_{i=1}^n x_i.
$$
</div>

Here the sample mean is not chosen by intuition; it emerges from the likelihood model. A different observation distribution would generally produce a different estimator.

### Point estimates and uncertainty intervals

A point estimate compresses uncertainty into one value. An interval communicates a range, but its interpretation depends on the inferential framework.

A frequentist confidence interval is produced by a procedure whose long-run coverage is controlled under repeated sampling. After an interval has been computed, the parameter is not random within the classical framework.

A Bayesian credible interval is a region containing a specified posterior probability under the chosen model and prior. The two intervals may be numerically similar in regular large-sample settings, but their meanings are different.

### Loss functions as modeling decisions

Many familiar losses are negative log-likelihoods. Squared error corresponds to a Gaussian observation model with fixed variance. Binary cross-entropy corresponds to a Bernoulli model. Multiclass cross-entropy corresponds to a categorical model.

This connection reveals what a loss assumes about noise and uncertainty. Choosing a loss is therefore not only an optimization decision. It is often an implicit probabilistic modeling decision.

Regularization can also admit a probabilistic interpretation. An $L_2$ penalty corresponds to a Gaussian prior in a MAP formulation, while an $L_1$ penalty corresponds to a Laplace prior. This equivalence concerns the optimizer of a particular posterior objective. It does not make regularized optimization identical to full Bayesian inference.

## Optimization: Learning as a Search Problem

The source chapter names optimization as a foundation but does not develop it separately. That omission leaves a gap, because a model class does not specify how its parameters are learned.

### Empirical risk

Given a model $f_\theta$, observations $(x_i,y_i)$, and loss $\ell$, empirical risk minimization takes the form

<div class="display-equation">
$$
\min_{\theta}\widehat{R}_n(\theta)
=
\min_{\theta}\frac{1}{n}\sum_{i=1}^n
\ell(f_\theta(x_i),y_i).
$$
</div>

A regularized objective adds a structural preference:

<div class="display-equation">
$$
\min_\theta
\frac{1}{n}\sum_{i=1}^n\ell(f_\theta(x_i),y_i)
+\lambda\Omega(\theta).
$$
</div>

Optimization error, estimation error, and approximation error are different. An optimizer may fail to minimize the training objective. A perfectly minimized training objective may fail to generalize. A model class may be unable to represent the target relationship even with unlimited data.

### Gradient methods

Gradient descent updates parameters by

<div class="display-equation">
$$
\theta_{t+1}=\theta_t-\eta_t\nabla f(\theta_t).
$$
</div>

The negative gradient is the direction of steepest local decrease under the Euclidean norm. Under other geometries, steepest descent takes a different form.

For a differentiable convex function, every local minimum is global. If the gradient is Lipschitz continuous, suitable step sizes provide quantitative convergence guarantees. Strong convexity gives stronger control and, for basic gradient descent, can yield linear convergence in function value.

Deep-network objectives are generally nonconvex. A zero gradient may indicate a local minimum, a local maximum, a saddle point, or a flat region. Convex guarantees cannot be transferred to such objectives without additional assumptions.

Stochastic gradient methods replace the full gradient with an estimate computed from a sample or minibatch:

<div class="display-equation">
$$
\theta_{t+1}=\theta_t-\eta_t g_t,
\qquad
\mathbb{E}[g_t\mid\theta_t]=\nabla f(\theta_t)
$$
</div>

in the ideal unbiased setting. The noise can reduce computation and may help exploration, but it also introduces variance. Momentum, adaptive preconditioning, variance reduction, and learning-rate schedules modify this basic process. Their behavior depends on both objective geometry and stochastic sampling assumptions.

### Constraints and nonsmooth objectives

Not every learning problem is unconstrained or smooth. Constraints may enforce probabilities, conservation laws, fairness requirements, stability, or physical feasibility. Projected methods, proximal methods, Lagrange multipliers, and duality provide systematic ways to handle such structure.

Nonsmooth penalties such as $L_1$ regularization require subgradients or proximal operators rather than ordinary derivatives at every point. Differentiability is useful, but it is not a prerequisite for meaningful optimization.

## Information Theory: Comparing Distributions

Information theory connects probability to learning objectives.

For a discrete distribution $p$, Shannon entropy is

<div class="display-equation">
$$
H(p)=-\sum_x p(x)\log p(x).
$$
</div>

Cross-entropy between $p$ and $q$ is

<div class="display-equation">
$$
H(p,q)=-\sum_x p(x)\log q(x).
$$
</div>

The Kullback-Leibler divergence is

<div class="display-equation">
$$
D_{\mathrm{KL}}(p\|q)
=
\sum_x p(x)\log\frac{p(x)}{q(x)}.
$$
</div>

They satisfy

<div class="display-equation">
$$
H(p,q)=H(p)+D_{\mathrm{KL}}(p\|q).
$$
</div>

Since $D_{\mathrm{KL}}(p\|q)\geq 0$, minimizing cross-entropy with respect to $q$ is equivalent to minimizing $D_{\mathrm{KL}}(p\|q)$ when $p$ is fixed.

KL divergence is not a metric. It is asymmetric and does not satisfy the triangle inequality. Its direction determines which mismatches are penalized strongly. This asymmetry matters in variational inference, distribution approximation, and generative modeling.

Negative log-likelihood is an empirical cross-entropy. This gives a common interpretation of classification losses, language-model training, density estimation, and probabilistic prediction.

## Numerical Analysis: Mathematics on Finite Machines

Mathematical identities are exact. Computer arithmetic is not.

Floating-point numbers represent only a finite subset of the real numbers. Rounding, overflow, underflow, cancellation, and limited precision can change the behavior of an otherwise correct formula.

Numerical analysis separates conditioning from stability. Conditioning asks how sensitive the exact solution is to perturbations in the input. Stability asks whether an algorithm introduces errors comparable to small perturbations of the original problem.

Several practical principles follow:

- Solve linear systems instead of explicitly forming inverses.
- Prefer QR or SVD to normal equations when least-squares conditioning is a concern.
- Compute products of probabilities in the log domain.
- Evaluate expressions such as log-sum-exp after subtracting the maximum input.
- Scale features and monitor gradient magnitudes when parameterization creates extreme curvature.
- Treat small residuals cautiously when the underlying problem is ill-conditioned.

For example,

<div class="display-equation">
$$
\log\sum_i e^{z_i}
=
m+\log\sum_i e^{z_i-m},
\qquad
m=\max_i z_i,
$$
</div>

is algebraically identical to the naive expression but much less likely to overflow.

The success of an AI system therefore depends on three different levels of correctness: the mathematical model must be coherent, the algorithm must solve the intended problem, and the numerical implementation must remain reliable in finite precision.

## The Foundation Is Broader Than Calculus

Calculus, linear algebra, probability, and optimization support much of contemporary machine learning. They are not the boundary of mathematical artificial intelligence.

Functional analysis studies operators and function spaces, providing language for kernels, inverse problems, approximation, and infinite-dimensional learning. Dynamical systems describe recurrent computation, control, stability, and continuous-depth models. Differential geometry studies optimization and probability on manifolds. Algebra exposes symmetry, equivariance, and compositional structure. Topology studies qualitative properties that survive deformation and supports topological data analysis. Logic, type theory, and formal methods address proof, specification, and machine-checked correctness.

These fields do not merely decorate an existing differentiable framework. They can change how a learning problem is represented. A symmetry can remove redundant degrees of freedom. A topological invariant can preserve global structure that local coordinates miss. A formal specification can distinguish empirical success from verified correctness.

The mathematical foundations of artificial intelligence should therefore be understood as an expanding language. Calculus explains local change, but intelligence also involves global structure, uncertainty, interaction, abstraction, and proof.

## Conclusion

The mathematics of artificial intelligence is not a prerequisite checklist placed before the real subject begins. It is the subject's internal architecture.

Analysis explains local approximation and sensitivity. Linear algebra explains representation, transformation, rank, and spectral structure. Tensor methods organize multilinear data and computation. Probability describes uncertainty, while statistics turns observations into inferential claims. Optimization converts objectives into parameter updates. Information theory compares distributions, and numerical analysis determines whether exact ideas survive contact with finite machines.

Each tool comes with conditions. Partial derivatives do not automatically imply differentiability. Bounded partial integrals do not automatically converge. A translation is affine rather than linear. A covariance of zero does not generally imply independence. Repeated evidence cannot be multiplied without a conditional-independence assumption. MAP is not the whole of Bayesian inference. An explicit inverse is rarely the best numerical route to a solution.

Rigor begins by making such conditions visible. It does not make machine learning less intuitive. It identifies exactly which intuition is justified, which approximation is being used, and which failure modes remain possible.

The deeper purpose of mathematics in artificial intelligence is not only to optimize existing models. It is to enlarge the space of models we are able to imagine.

## References

1. Marc Peter Deisenroth, A. Aldo Faisal, and Cheng Soon Ong, [Mathematics for Machine Learning](https://mml-book.github.io/), Cambridge University Press, 2020.

2. Ian Goodfellow, Yoshua Bengio, and Aaron Courville, [Deep Learning: Applied Math and Machine Learning Basics](https://www.deeplearningbook.org/contents/TOC.html), MIT Press, 2016.

3. Stephen Boyd and Lieven Vandenberghe, [Convex Optimization](https://web.stanford.edu/~boyd/cvxbook/), Cambridge University Press, 2004.

4. Atilim Gunes Baydin, Barak A. Pearlmutter, Alexey Andreyevich Radul, and Jeffrey Mark Siskind, [Automatic Differentiation in Machine Learning: A Survey](https://www.jmlr.org/papers/v18/17-468.html), *Journal of Machine Learning Research*, 2018.

5. Tamara G. Kolda and Brett W. Bader, [Tensor Decompositions and Applications](https://doi.org/10.1137/07070111X), *SIAM Review*, 2009.

6. Ian T. Jolliffe and Jorge Cadima, [Principal Component Analysis: A Review and Recent Developments](https://doi.org/10.1098/rsta.2015.0202), *Philosophical Transactions of the Royal Society A*, 2016.

7. Kevin P. Murphy, [Probabilistic Machine Learning: An Introduction](https://mitpress.mit.edu/9780262369305/probabilistic-machine-learning/), MIT Press, 2022.

8. John Tsitsiklis, [Probabilistic Systems Analysis and Applied Probability](https://ocw.mit.edu/courses/6-041sc-probabilistic-systems-analysis-and-applied-probability-fall-2013/pages/unit-i/), MIT OpenCourseWare, 2013.

9. Nicholas J. Higham, [Accuracy and Stability of Numerical Algorithms](https://doi.org/10.1137/1.9780898718027), second edition, SIAM, 2002.

10. Thomas M. Cover and Joy A. Thomas, [Elements of Information Theory](https://doi.org/10.1002/047174882X), second edition, Wiley, 2005.
