---
title: "Chapter 3: Tensor Algebra and Einstein Summation"
layout: "single"
url: "/book/chapters/chapter003/"
summary: "Multilinear relations, universal constructions, and the index grammar of information flow."
draft: false
ShowToc: true
TocOpen: true
hideMeta: true
math: true
---

<style>
  .post-content,
  .post-content h1, .post-content h2, .post-content h3, .post-content h4,
  .post-content p, .post-content li, .post-content blockquote,
  .post-content td, .post-content th {
    font-family: 'Times New Roman', 'Times', 'Noto Serif', Georgia, serif;
  }
  .post-content { font-size: 12pt; line-height: 1.72; }
  .def-box  { border-left:4px solid #4a148c; background:#faf5ff; padding:1em 1.2em; margin:1.5em 0; border-radius:4px; }
  .prop-box { border-left:4px solid #1565c0; background:#f0f6ff; padding:1em 1.2em; margin:1.5em 0; border-radius:4px; }
  .proof-box{ border-left:4px solid #999;    background:#fafafa; padding:.8em 1.2em; margin:1em 0 1.5em; border-radius:4px; }
  .ml-box   { border-left:4px solid #e65100; background:#fff8f0; padding:.8em 1.2em; margin:1em 0 1.5em; border-radius:4px; }
  .scholium-box { border:2px solid #6a1b9a; background:#fdf5ff; padding:1em 1.2em; margin:1.5em 0; border-radius:6px; }
</style>

<div style="text-align:center; margin:1.5em 0 2.5em 0;">

# Volume I &mdash; The Mathematical Principles of Machine Learning

## Chapter 3 &mdash; Tensor Algebra and Einstein Summation

*Xujiang Tang*

</div>

## Abstract

After establishing vector spaces as the minimal container for learnable states (Chapter 1) and spectral theory as the language of persistent modes of evolution (Chapter 2), we turn to the central phenomenon of intelligent systems: **coordinated multi-way interaction**. Tensors are not "multi-dimensional arrays" by definition; they are the canonical algebraic closure of multilinear relations. The tensor product turns multilinear maps into linear maps via a universal property, and Einstein summation is not merely notational convenience but a grammar of information flow: contraction is the algebraic signature of measurement, marginalization, and feature binding. We develop the tensor product from first principles (free vector spaces and quotient construction), prove its universal property without omitted steps, and then reinterpret key machine-learning primitives &mdash; bilinear pooling, quadratic features, and multi-head attention &mdash; as tensorial constructions with explicit index topology. We conclude by isolating the precise point where inner products and norms enter: they are additional structure on tensor spaces, not prerequisites.

## Notation

- $V,W,U$ are finite-dimensional vector spaces over a field $\mathbb{F}$ (usually $\mathbb{R}$).
- $\{e_i\}$ denotes a basis of $V$ and $\{\varepsilon^i\}$ its dual basis with $\varepsilon^i(e_j)=\delta^i_j$.
- $\mathrm{Bilin}(V,W;U)$ is the set of bilinear maps $V\times W\to U$.
- $\mathrm{Lin}(X,Y)$ is the set of linear maps $X\to Y$.
- $F(S)$ denotes the free vector space on a set $S$.
- Einstein convention: repeated upper-lower index pairs are summed unless stated otherwise.

---

## 3.1 The Tensor Product: From Linearity to Multilinearity

### 3.1.1 The misleading "array" narrative and the intrinsic definition

In engineering practice, a tensor is often introduced as a multidimensional array. While arrays provide a coordinate representation once bases and inner products are fixed, the concept relevant to learning theory is basis-independent: tensors are elements of spaces defined by universal properties. In particular, the tensor product $V\otimes W$ is characterized by *how it represents bilinear interactions*, not by how its elements are stored.

<div class="ml-box">

**Machine-learning remark.** Every deep-learning framework stores tensors as rectangular arrays in memory. This is a coordinate representation relative to the standard basis. The intrinsic definition matters because: (i) change of basis (e.g., PCA whitening, learned reparameterization) must preserve meaning, and (ii) architectural invariants (what interactions are representable) are basis-free statements about the tensor product, not about any particular array layout.

</div>

---

### 3.1.2 Bilinear maps as the primitive notion of interaction

Let $V,W,U$ be vector spaces over a field $\mathbb{F}$.

<div class="def-box">

**Definition 3.1 (Bilinear map).** A map $B:V\times W\to U$ is *bilinear* if, for all $v,v'\in V$, $w,w'\in W$, and $\alpha,\beta\in\mathbb{F}$,

$$B(\alpha v+\beta v',w)=\alpha B(v,w)+\beta B(v',w),$$

$$B(v,\alpha w+\beta w')=\alpha B(v,w)+\beta B(v,w').$$

That is, $B$ is linear in each argument separately when the other is held fixed.

</div>

<div class="ml-box">

**Machine-learning translation.** A bilinear layer is a mechanism that is linear in each argument separately &mdash; for example, certain forms of cross-attention scoring, bilinear pooling, and second-order feature interactions. The tensor product is precisely the object that makes such interactions *linear in a larger space*.

</div>

---

### 3.1.3 Universal property of the tensor product

<div class="def-box">

**Definition 3.2 (Tensor product via universal property).** A *tensor product* of $V$ and $W$ is a pair $(V\otimes W,\,\tau)$ where $V\otimes W$ is a vector space and $\tau:V\times W\to V\otimes W$ is bilinear, such that for every vector space $U$ and every bilinear map $B:V\times W\to U$, there exists a **unique** linear map $\widetilde{B}:V\otimes W\to U$ satisfying

$$B(v,w) = \widetilde{B}\big(\tau(v,w)\big)\quad \text{for all }(v,w)\in V\times W.$$

We write $\tau(v,w)=v\otimes w$, hence $B(v,w)=\widetilde{B}(v\otimes w)$.

</div>

This statement is not "philosophical": it is an explicit equivalence

$$\mathrm{Bilin}(V,W;U)\ \cong\ \mathrm{Lin}(V\otimes W,\,U),$$

natural in $U$. It is the formal reason that bilinear computation can be implemented as linear computation in a lifted space.

<div class="scholium-box">

**Scholium 3.3 (The universal property as a design principle).** Whenever an engineering system requires bilinear interaction between two representations, the universal property guarantees that one may equivalently: (a) design a bilinear mechanism directly, or (b) lift representations into the tensor product and apply a linear readout. The second form is often preferable because linear maps compose, spectralize, and regularize cleanly.

</div>

---

### 3.1.4 Explicit construction: free vector space and quotient by bilinearity relations

We now construct $V\otimes W$ and prove the universal property without skipping steps.

#### Step 1 &mdash; The free vector space on $V\times W$

Let $F(V\times W)$ denote the free vector space on the set $V\times W$. Concretely, $F(V\times W)$ consists of finite formal linear combinations

$$\sum_{k=1}^{N}\alpha_k\,[(v_k,w_k)],$$

where $\alpha_k\in\mathbb{F}$ and $[(v,w)]$ is a formal basis symbol indexed by $(v,w)\in V\times W$. Addition and scalar multiplication are defined coefficient-wise.

#### Step 2 &mdash; Impose bilinearity by factoring out a subspace

Let $R\subseteq F(V\times W)$ be the linear span of the following elements (intended to enforce bilinearity):

<div class="def-box">

**(R1)** Additivity in the first argument:

$$[(v+v',w)] - [(v,w)] - [(v',w)].$$

**(R2)** Homogeneity in the first argument:

$$[(\alpha v,w)] - \alpha\,[(v,w)].$$

**(R3)** Additivity in the second argument:

$$[(v,w+w')] - [(v,w)] - [(v,w')].$$

**(R4)** Homogeneity in the second argument:

$$[(v,\alpha w)] - \alpha\,[(v,w)].$$

</div>

#### Step 3 &mdash; Define the tensor product as the quotient space

Define

$$V\otimes W := F(V\times W)\,/\,R.$$

Define $\tau:V\times W\to V\otimes W$ by

$$\tau(v,w) := [(v,w)] + R.$$

We denote $\tau(v,w)$ by $v\otimes w$.

---

### 3.1.5 Verification that $\tau$ is bilinear

<div class="prop-box">

**Proposition 3.4.** The canonical map $\tau:V\times W\to V\otimes W$ defined above is bilinear.

</div>

<div class="proof-box">

**Proof.** We check each axiom explicitly.

**Additivity in the first argument:**

$$\tau(v+v',w) - \tau(v,w) - \tau(v',w) = \big([(v+v',w)]-[(v,w)]-[(v',w)]\big)+R = 0+R,$$

since the bracketed expression is a generator of type (R1), hence lies in $R$. Therefore $\tau(v+v',w)=\tau(v,w)+\tau(v',w)$.

**Homogeneity in the first argument:**

$$\tau(\alpha v,w)-\alpha\,\tau(v,w) = \big([(\alpha v,w)]-\alpha\,[(v,w)]\big)+R = 0+R,$$

since the bracketed expression is a generator of type (R2). Therefore $\tau(\alpha v,w)=\alpha\,\tau(v,w)$.

**Additivity in the second argument:**

$$\tau(v,w+w') - \tau(v,w) - \tau(v,w') = \big([(v,w+w')]-[(v,w)]-[(v,w')]\big)+R = 0+R,$$

by generator (R3). Therefore $\tau(v,w+w')=\tau(v,w)+\tau(v,w')$.

**Homogeneity in the second argument:**

$$\tau(v,\alpha w)-\alpha\,\tau(v,w) = \big([(v,\alpha w)]-\alpha\,[(v,w)]\big)+R = 0+R,$$

by generator (R4). Therefore $\tau(v,\alpha w)=\alpha\,\tau(v,w)$.

Hence $\tau$ is bilinear. $\blacksquare$

</div>

---

### 3.1.6 Proof of the universal property

<div class="prop-box">

**Theorem 3.5 (Universal property of the tensor product).** Let $U$ be any vector space and let $B:V\times W\to U$ be bilinear. Then there exists a unique linear map $\widetilde{B}:V\otimes W\to U$ such that $\widetilde{B}(v\otimes w)=B(v,w)$ for all $v\in V,\,w\in W$.

</div>

<div class="proof-box">

**Proof.**

**Existence.** Define a linear map $L:F(V\times W)\to U$ by linear extension of the assignment on formal basis vectors:

$$L\!\left(\sum_{k=1}^{N}\alpha_k\,[(v_k,w_k)]\right) := \sum_{k=1}^{N}\alpha_k\,B(v_k,w_k).$$

Linearity of $L$ is immediate from the definition.

We now verify that $R\subseteq \ker L$. It suffices to check that each generator of $R$ is mapped to zero.

**Generator (R1):**

$$L\big([(v+v',w)]-[(v,w)]-[(v',w)]\big) = B(v+v',w)-B(v,w)-B(v',w) = 0,$$

by additivity of $B$ in the first argument.

**Generator (R2):**

$$L\big([(\alpha v,w)]-\alpha\,[(v,w)]\big) = B(\alpha v,w)-\alpha\,B(v,w) = 0,$$

by homogeneity of $B$ in the first argument.

**Generator (R3):**

$$L\big([(v,w+w')]-[(v,w)]-[(v,w')]\big) = B(v,w+w')-B(v,w)-B(v,w') = 0,$$

by additivity of $B$ in the second argument.

**Generator (R4):**

$$L\big([(v,\alpha w)]-\alpha\,[(v,w)]\big) = B(v,\alpha w)-\alpha\,B(v,w) = 0,$$

by homogeneity of $B$ in the second argument.

Thus every generator lies in $\ker L$, hence $R\subseteq \ker L$.

By the universal property of quotient spaces, $L$ factors through the quotient: there exists a unique linear map $\widetilde{B}:F(V\times W)/R \to U$ such that

$$\widetilde{B}(\xi+R)=L(\xi)\quad \text{for all }\xi\in F(V\times W).$$

In particular, for $(v,w)\in V\times W$,

$$\widetilde{B}(v\otimes w) = \widetilde{B}\big([(v,w)]+R\big) = L\big([(v,w)]\big) = B(v,w).$$

This constructs the required $\widetilde{B}$.

**Uniqueness.** Suppose $\Phi:V\otimes W\to U$ is linear and satisfies $\Phi(v\otimes w)=B(v,w)$ for all $v,w$. We show $\Phi=\widetilde{B}$.

Every element of $V\otimes W$ is a finite linear combination of simple tensors, because $V\otimes W$ is generated by $\{v\otimes w : v\in V,\,w\in W\}$ as a vector space. Therefore any $z\in V\otimes W$ can be written as

$$z = \sum_{k=1}^{N}\alpha_k\,(v_k\otimes w_k).$$

By linearity,

$$\Phi(z) = \sum_{k=1}^{N}\alpha_k\,\Phi(v_k\otimes w_k) = \sum_{k=1}^{N}\alpha_k\,B(v_k,w_k).$$

But this is precisely the value of $\widetilde{B}(z)$ by the construction above. Hence $\Phi(z)=\widetilde{B}(z)$ for all $z$, so $\Phi=\widetilde{B}$. $\blacksquare$

</div>

---

### 3.1.7 Machine-learning consequences of the universal property

The universal property is a theorem about **representational adequacy**: any bilinear interaction $B(v,w)$ can be implemented as a linear functional on the tensor product.

<div class="ml-box">

**Example 3.6 (Bilinear scoring in representation learning).** Let $V=W=\mathbb{R}^d$. Consider a bilinear score

$$s(v,w) := v^\top M w, \quad M\in\mathbb{R}^{d\times d}.$$

Define $B(v,w)=v^\top M w\in\mathbb{R}$. This is bilinear.

By the universal property, there exists a unique linear functional $\widetilde{B}:V\otimes W\to\mathbb{R}$ such that $\widetilde{B}(v\otimes w)=v^\top M w$.

To make $\widetilde{B}$ explicit, choose the standard basis $\{e_i\}$ and write

$$v=\sum_i v^i e_i,\quad w=\sum_j w^j e_j.$$

Then

$$v\otimes w = \sum_{i,j} v^i w^j\,(e_i\otimes e_j).$$

Define a linear functional by specifying its values on the basis tensors:

$$\widetilde{B}(e_i\otimes e_j) := M_{ij}.$$

Then, by linearity,

$$\widetilde{B}(v\otimes w) = \sum_{i,j} v^i w^j\,\widetilde{B}(e_i\otimes e_j) = \sum_{i,j} v^i w^j\,M_{ij} = v^\top M w.$$

Thus a bilinear score is literally a **linear readout on the lifted space** $V\otimes W$. This observation is the algebraic core behind "second-order features" and polynomial kernels.

</div>

<div class="ml-box">

**Example 3.7 (Quadratic feature maps and polynomial kernels).** Let $V=\mathbb{R}^d$. The map $x\mapsto x\otimes x\in V\otimes V$ is not linear in $x$, but it embeds $x$ into a space where any quadratic form becomes linear:

$$x^\top A\,x = \widetilde{B}(x\otimes x),$$

for a suitable linear functional $\widetilde{B}$ determined by $A$. In kernel methods, this is precisely the mechanism by which nonlinear decision boundaries are represented by linear separators in a higher-order tensor feature space.

More concretely, the polynomial kernel $k(x,y)=(x^\top y)^2$ satisfies

$$k(x,y) = (x^\top y)^2 = \langle x\otimes x,\,y\otimes y\rangle_{V\otimes V},$$

where the inner product on $V\otimes V$ is induced from the standard inner product on $V$. The "kernel trick" is the observation that this inner product can be computed without explicitly forming the $d^2$-dimensional tensor product.

</div>

---

## 3.2 Einstein Summation: Index Grammar for Information Flow

### 3.2.1 Indices as variance bookkeeping, not decoration

Einstein summation is often introduced as a notational shortcut: repeated indices are summed. The substantive content is that **contraction is the coordinate representation of canonical pairings between primal and dual spaces**.

Let $V$ be finite-dimensional with basis $\{e_i\}$ and dual basis $\{\varepsilon^i\}$ where $\varepsilon^i(e_j)=\delta^i_j$.

<div class="def-box">

**Convention 3.8 (Einstein summation).** A vector $x\in V$ has coordinates $x^i$ via $x=x^i e_i$ (summation over $i$ implied). A covector $\phi\in V^*$ has coordinates $\phi_i$ via $\phi=\phi_i\,\varepsilon^i$ (summation over $i$ implied). Upper indices are called *contravariant*; lower indices are called *covariant*.

**Rule:** Any index that appears once as a superscript and once as a subscript in a single term is summed over.

</div>

The canonical pairing is then

$$\langle \phi,x\rangle = \phi(x) = (\phi_i\,\varepsilon^i)(x^j\,e_j) = \phi_i\,x^j\,\varepsilon^i(e_j) = \phi_i\,x^j\,\delta^i_j = \phi_i\,x^i.$$

Here the repeated index $i$ is summed. The disappearance of the index is not cosmetic: it signifies the **completion of a measurement** (a scalar output).

<div class="ml-box">

**Machine-learning translation.** Every time a network produces a scalar score from a representation, it performs a contraction between a covector (weights) and a vector (features). The fact that the index disappears is precisely the fact that degrees of freedom have been aggregated into a scalar decision statistic.

</div>

---

### 3.2.2 Contraction as elimination and creation of degrees of freedom

A tensor with indices $T^{i_1\cdots i_p}{}_{j_1\cdots j_q}$ can be viewed as a multilinear map with $p$ contravariant and $q$ covariant slots. Contracting an upper index with a lower index is the act of **summing out a latent channel**.

<div class="prop-box">

**Example 3.9 (Matrix-vector multiplication).** Let $A^i{}_j$ be a linear map $V\to V$. Then

$$y^i = A^i{}_j\,x^j.$$

The index $j$ disappears: one input dimension is consumed, producing an output dimension $i$. The contraction is the algebraic expression of "applying the operator."

</div>

<div class="prop-box">

**Example 3.10 (Composition of linear maps).** Let $A^i{}_j$ and $B^j{}_k$ be two linear maps. Then

$$(AB)^i{}_k = A^i{}_j\,B^j{}_k.$$

The contracted $j$ is the **internal wire** of a computation graph: an intermediate channel that is not observable at the output.

</div>

<div class="ml-box">

**Machine-learning translation.** Contraction indices are precisely the hidden wires of computational graphs. The topology of indices is a basis-independent representation of "who interacts with whom."

</div>

---

### 3.2.3 Dummy indices as latent channels: a structural view of model architecture

In Einstein notation, contracted indices are *dummy indices*: they do not appear in the result, but they define the internal connectivity. A complicated tensor expression is therefore a graph whose edges are dummy indices and whose open legs are external interfaces.

<div class="scholium-box">

**Scholium 3.11 (Index topology as architecture).** Architectural design can be described as **index topology design**. For instance:

- Changing from self-attention to cross-attention is literally changing which index sets are identified and contracted.
- Depth-wise vs. point-wise convolution corresponds to restricting which spatial and channel indices participate in the same contraction.
- Bottleneck layers correspond to contracting through a low-dimensional index set.

This viewpoint makes certain invariants visible: permutation symmetries, conservation of feature dimension, and the existence of bottlenecks are all readable from the index structure without inspecting numerical values.

</div>

---

### 3.2.4 Trace as the fundamental contraction

<div class="def-box">

**Definition 3.12 (Trace).** The trace of a linear map $A:V\to V$ is defined by

$$\mathrm{tr}(A) := A^i{}_i = \sum_{i=1}^{n} A^i{}_i.$$

This is the contraction of the single contravariant index with the single covariant index.

</div>

<div class="prop-box">

**Proposition 3.13 (Basis-independence of trace).** The trace is independent of the choice of basis.

</div>

<div class="proof-box">

**Proof.** Let $\{e_i\}$ and $\{f_j\}$ be two bases related by $f_j = P^i{}_j\,e_i$ with inverse $e_i = (P^{-1})^j{}_i\,f_j$. In the new basis, the matrix entries are

$$\bar{A}^j{}_k = (P^{-1})^j{}_i\,A^i{}_l\,P^l{}_k.$$

Then

$$\bar{A}^j{}_j = (P^{-1})^j{}_i\,A^i{}_l\,P^l{}_j = \delta^l_i\,A^i{}_l = A^i{}_i,$$

where we used $(P^{-1})^j{}_i\,P^l{}_j = \delta^l_i$. $\blacksquare$

</div>

<div class="ml-box">

**Machine-learning translation.** Trace appears throughout learning theory: the trace of the Hessian (Laplacian) controls flatness of loss surfaces; the trace of the Fisher information matrix governs natural gradient geometry; the nuclear norm (sum of singular values, i.e. trace of $\sqrt{A^\top A}$) is the convex surrogate for matrix rank in low-rank regularization.

</div>

---

## 3.3 Multi-Head Attention: A Tensorial Deconstruction

### 3.3.1 Setup and typing

Let $X\in\mathbb{R}^{n\times d}$ be a sequence of $n$ tokens with embedding dimension $d$. For one head, define projection matrices

$$W_Q\in\mathbb{R}^{d\times d_k},\quad W_K\in\mathbb{R}^{d\times d_k},\quad W_V\in\mathbb{R}^{d\times d_v}.$$

Define

$$Q = X\,W_Q \in\mathbb{R}^{n\times d_k},\quad K = X\,W_K \in\mathbb{R}^{n\times d_k},\quad V = X\,W_V \in\mathbb{R}^{n\times d_v}.$$

In Einstein indices (token index $a,b$; channel indices $i,j,r$):

$$Q_a^{\ i} = X_a^{\ j}\,(W_Q)_j^{\ i},\qquad K_b^{\ i} = X_b^{\ j}\,(W_K)_j^{\ i},\qquad V_b^{\ r} = X_b^{\ j}\,(W_V)_j^{\ r}.$$

<div class="prop-box">

**Observation 3.14 (Attention logits as contraction).** The attention logits (before scaling) are

$$S_{ab} = Q_a^{\ i}\,K_b^{\ i},$$

i.e., contraction over the key-channel index $i$. This is exactly a bilinear pairing. In a fully typed setting, one may treat $K_b$ as a covector acting on $Q_a$; the Euclidean dot product is one particular identification of $V\cong V^*$ (the Riesz map, to be developed in Chapter 4).

</div>

---

### 3.3.2 From outer interaction to normalized operator

Define the unnormalized interaction tensor

$$\widetilde{S}_{ab} := Q_a^{\ i}\,K_b^{\ i}.$$

After scaling and row-wise softmax,

$$A_{ab} := \mathrm{softmax}_b\!\left(\frac{\widetilde{S}_{ab}}{\sqrt{d_k}}\right), \quad \text{so that}\quad \sum_{b=1}^n A_{ab}=1,\ A_{ab}\ge 0.$$

Thus $A\in\mathbb{R}^{n\times n}$ is a **stochastic operator** acting on token space: for each query token $a$, it defines a probability distribution over keys $b$.

<div class="ml-box">

**Machine-learning translation.** Attention produces a data-dependent linear operator $A(X)$ on token space. It is not "just weighted averaging"; it is the construction of an adaptive operator whose spectrum and conditioning matter for stability and expressivity. The $1/\sqrt{d_k}$ scaling ensures that the variance of $\widetilde{S}_{ab}$ is $O(1)$, preventing softmax saturation (which would collapse the operator to a rank-1 projection).

</div>

---

### 3.3.3 Contraction as aggregation: output formation

The output is

$$O_a^{\ r} = A_{ab}\,V_b^{\ r},$$

contracting over the key/token index $b$. Here the index $b$ disappears: key-token degrees of freedom are marginalized into a new representation indexed by $a$ and $r$.

Expanding everything (no steps skipped):

$$O_a^{\ r} = A_{ab}\,V_b^{\ r} = A_{ab}\,X_b^{\ j}\,(W_V)_j^{\ r}.$$

Hence attention is a composite of linear maps and nonlinear normalization that dynamically selects which token channels are preserved and which are eliminated by contraction.

<div class="scholium-box">

**Scholium 3.15 (The anatomy of attention in index language).**

| Step | Tensorial operation | Indices consumed | Indices produced |
|------|-------------------|-----------------|-----------------|
| Projection | $Q_a^{\ i} = X_a^{\ j}(W_Q)_j^{\ i}$ | $j$ (input channel) | $i$ (key channel) |
| Scoring | $\widetilde{S}\_{ab} = Q_a^{\ i}K_b^{\ i}$ | $i$ (key channel) | $a,b$ (token pair) |
| Normalization | $A_{ab} = \mathrm{softmax}_b(\widetilde{S}\_{ab}/\sqrt{d_k})$ | &mdash; | $a,b$ (stochastic) |
| Aggregation | $O_a^{\ r} = A_{ab}V_b^{\ r}$ | $b$ (key token) | $a$ (query token), $r$ (value channel) |

Each contraction eliminates degrees of freedom; each free index is an interface to the next layer. The full attention block is therefore a typed tensor network with precisely specified information flow.

</div>

---

### 3.3.4 Multi-head structure: parallel tensor logics

For $h$ heads, $Q^{(m)},K^{(m)},V^{(m)}$ are computed with head-specific projections ($m=1,\ldots,h$). Each head produces

$$O^{(m)} = A^{(m)}\,V^{(m)}.$$

Then outputs are concatenated and mixed by $W_O\in\mathbb{R}^{hd_v\times d}$:

$$\mathrm{MultiHead}(X) = \mathrm{Concat}(O^{(1)},\ldots,O^{(h)})\,W_O.$$

<div class="prop-box">

**Proposition 3.16 (Multi-head attention as operator superposition).** Each head defines a distinct index topology (a distinct operator $A^{(m)}(X)$). Multi-head attention is a superposition of multiple adaptive operators, each specializing to different interaction channels. The final projection $W_O$ performs a learned linear combination of these parallel tensor logics.

</div>

<div class="ml-box">

**Machine-learning consequence.** The expressive power of multi-head attention comes not from having more parameters per se, but from maintaining **parallel index topologies** &mdash; multiple independent contraction patterns &mdash; and then mixing them. This is why pruning heads that learn near-identical attention patterns often does not degrade performance: redundant index topologies carry no additional relational information.

</div>

---

## 3.4 Tensor Rank, Low-Rank Structure, and the Limits of Relational Compression

### 3.4.1 Tensor rank as a complexity measure of interaction

<div class="def-box">

**Definition 3.17 (Matrix rank via outer products).** A matrix $A\in\mathbb{R}^{m\times n}$ has rank equal to the minimum number of rank-1 outer products needed to represent it:

$$A = \sum_{i=1}^k u_i\,v_i^\top,\quad k=\mathrm{rank}(A).$$

</div>

For higher-order tensors, multiple rank notions exist:

<div class="def-box">

**Definition 3.18 (CP rank).** The *canonical polyadic (CP) rank* of a tensor $T\in V_1\otimes V_2\otimes\cdots\otimes V_p$ is the minimum $r$ such that

$$T = \sum_{i=1}^{r} v_1^{(i)}\otimes v_2^{(i)}\otimes\cdots\otimes v_p^{(i)},$$

where each $v_k^{(i)}\in V_k$.

**Definition 3.19 (Tucker rank).** The *Tucker rank* (or multilinear rank) of $T$ is the tuple $(\mathrm{rank}(T_{(1)}),\,\mathrm{rank}(T_{(2)}),\,\ldots,\,\mathrm{rank}(T_{(p)}))$, where $T_{(k)}$ denotes the mode-$k$ matricization (unfolding) of $T$.

</div>

The conceptual commonality is: **rank measures how many separable interaction components are required to represent a relational structure.**

<div class="ml-box">

**Machine-learning translation.** A low-rank relational tensor corresponds to an interaction that decomposes into a small number of independent factors; high-rank residual structure indicates **irreducible entanglement** &mdash; interaction patterns that cannot be explained by a small set of separable templates.

This distinction is fundamental to model compression and adaptation:
- **LoRA** (Low-Rank Adaptation) constrains weight updates to rank-$r$ perturbations $\Delta W = BA$ with $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times d}$, $r\ll d$.
- **Tensor decomposition** of embedding tables (e.g., TT-Embedding) exploits low multilinear rank to compress massive lookup tables.
- The **effective rank** of attention matrices $A^{(m)}$ is a diagnostic for head utility: near-rank-1 heads attend to a single token and may be prunable.

</div>

---

### 3.4.2 Practical manifestation: factorized parameterizations

<div class="prop-box">

**Proposition 3.20 (Rank constraint as regularization).** Let $W\in\mathbb{R}^{m\times n}$ and consider the constrained set $\{W:\mathrm{rank}(W)\le r\}$. This is equivalent to parametrizing $W=UV^\top$ with $U\in\mathbb{R}^{m\times r}$, $V\in\mathbb{R}^{n\times r}$.

The number of parameters drops from $mn$ to $r(m+n)$, which is a dramatic reduction when $r\ll\min(m,n)$.

</div>

Many efficient architectures implement low-rank constraints implicitly:

| Architecture | Algebraic mechanism | Rank constraint |
|-------------|-------------------|-----------------|
| Factorized embeddings | $E = U V^\top$ | Embedding rank $\le r$ |
| LoRA | $W + \Delta W = W + BA$ | Update rank $\le r$ |
| Low-rank attention | $A \approx \Phi_Q \Phi_K^\top$ | Attention matrix rank $\le m$ |
| Mixture of Experts | $\sum_k g_k(x)\,W_k$ | Conditional rank selection |
| Tensorized layers | TT/Tucker decomposition | Multilinear rank constraint |

---

### 3.4.3 The tensor rank problem: computational hardness

<div class="scholium-box">

**Scholium 3.21 (Hardness of tensor rank).** Unlike matrix rank (computable in $O(\min(m,n)^2\max(m,n))$ via SVD), determining the CP rank of a tensor of order $\ge 3$ is NP-hard in general (Håstad, 1990). This has a practical consequence: there is no polynomial-time algorithm that certifies the optimal compression of a general multi-way interaction.

Machine-learning implication: heuristic decompositions (ALS, gradient descent on factored forms) are used in practice, and the gap between the achieved rank and the true CP rank is generally unknown. This is a fundamental limit on provably optimal tensor compression in neural architectures.

</div>

---

## 3.5 Higher-Order Tensor Products and the Algebraic Closure of Multilinear Maps

### 3.5.1 Iterated tensor products

The tensor product construction generalizes to multiple factors. For spaces $V_1,V_2,\ldots,V_p$, the tensor product $V_1\otimes V_2\otimes\cdots\otimes V_p$ is characterized by the universal property:

<div class="def-box">

**Definition 3.22 (Multilinear universal property).** A $p$-fold tensor product $(V_1\otimes\cdots\otimes V_p,\,\tau)$ satisfies: for every vector space $U$ and every $p$-linear map $M:V_1\times\cdots\times V_p\to U$, there exists a unique linear map $\widetilde{M}:V_1\otimes\cdots\otimes V_p\to U$ with

$$M(v_1,\ldots,v_p) = \widetilde{M}(v_1\otimes\cdots\otimes v_p).$$

Equivalently,

$$\mathrm{Multilin}(V_1,\ldots,V_p;\,U)\ \cong\ \mathrm{Lin}(V_1\otimes\cdots\otimes V_p,\,U).$$

</div>

<div class="prop-box">

**Proposition 3.23 (Associativity of tensor product).** The tensor product is associative up to canonical isomorphism:

$$(V_1\otimes V_2)\otimes V_3 \cong V_1\otimes(V_2\otimes V_3) \cong V_1\otimes V_2\otimes V_3.$$

</div>

<div class="proof-box">

**Proof (sketch).** Both $(V_1\otimes V_2)\otimes V_3$ and $V_1\otimes V_2\otimes V_3$ satisfy the universal property for trilinear maps $V_1\times V_2\times V_3\to U$. By uniqueness of objects satisfying universal properties (up to unique isomorphism), they are canonically isomorphic. $\blacksquare$

</div>

---

### 3.5.2 Dimension formula

<div class="prop-box">

**Proposition 3.24 (Dimension of tensor products).** If $\dim V_k = n_k$ for $k=1,\ldots,p$, then

$$\dim(V_1\otimes V_2\otimes\cdots\otimes V_p) = n_1\cdot n_2\cdots n_p = \prod_{k=1}^{p}n_k.$$

A basis is given by $\{e_{i_1}^{(1)}\otimes e_{i_2}^{(2)}\otimes\cdots\otimes e_{i_p}^{(p)}\}$ where each $i_k$ ranges over $1,\ldots,n_k$.

</div>

<div class="proof-box">

**Proof.** We prove the case $p=2$; the general case follows by induction using associativity.

Let $\{e_1,\ldots,e_m\}$ be a basis for $V$ and $\{f_1,\ldots,f_n\}$ a basis for $W$. We claim that $\{e_i\otimes f_j : 1\le i\le m,\,1\le j\le n\}$ is a basis for $V\otimes W$.

**Spanning:** Any simple tensor satisfies $v\otimes w = (\sum_i v^i e_i)\otimes(\sum_j w^j f_j) = \sum_{i,j}v^i w^j(e_i\otimes f_j)$ by bilinearity. Since $V\otimes W$ is spanned by simple tensors, the set $\{e_i\otimes f_j\}$ spans $V\otimes W$.

**Linear independence:** Suppose $\sum_{i,j}\alpha_{ij}(e_i\otimes f_j)=0$. For each pair $(a,b)$, define $B_{ab}:V\times W\to\mathbb{F}$ by $B_{ab}(v,w)=v^a w^b$ (the product of the $a$-th and $b$-th coordinates). This is bilinear. By the universal property, there exists linear $\widetilde{B}_{ab}:V\otimes W\to\mathbb{F}$ with $\widetilde{B}_{ab}(e_i\otimes f_j)=\delta_{ia}\delta_{jb}$. Applying $\widetilde{B}_{ab}$ to $\sum_{i,j}\alpha_{ij}(e_i\otimes f_j)=0$ gives $\alpha_{ab}=0$.

Hence $\dim(V\otimes W)=mn$. $\blacksquare$

</div>

<div class="ml-box">

**Machine-learning consequence.** The exponential growth $\dim(V^{\otimes p})=(\dim V)^p$ is the **curse of dimensionality** in tensor feature spaces. Kernel methods and tensor decompositions exist precisely to avoid materializing this exponential space while still accessing its representational power.

</div>

---

## 3.6 The Tensor Algebra and Graded Structure

### 3.6.1 Definition of the tensor algebra

<div class="def-box">

**Definition 3.25 (Tensor algebra).** The *tensor algebra* of a vector space $V$ is the direct sum

$$T(V) := \bigoplus_{p=0}^{\infty} V^{\otimes p} = \mathbb{F} \oplus V \oplus (V\otimes V) \oplus (V\otimes V\otimes V) \oplus \cdots,$$

where $V^{\otimes 0}:=\mathbb{F}$ and $V^{\otimes p}:=\underbrace{V\otimes\cdots\otimes V}_{p}$.

Multiplication is given by the tensor product: for $x\in V^{\otimes p}$ and $y\in V^{\otimes q}$,

$$x\cdot y := x\otimes y \in V^{\otimes(p+q)}.$$

This makes $T(V)$ a graded, associative, unital algebra.

</div>

<div class="ml-box">

**Machine-learning translation.** The tensor algebra is the universal container for polynomial features of all degrees. A representation $x\in V$ generates the feature hierarchy

$$1,\ x,\ x\otimes x,\ x\otimes x\otimes x,\ \ldots$$

which encodes interactions of all orders. Any polynomial function of $x$ is a linear functional on a finite truncation of $T(V)$. This is the algebraic foundation of polynomial kernels and of architectures that explicitly construct higher-order feature interactions.

</div>

---

## 3.7 Inner Products on Tensor Spaces: Additional Structure, Not Prerequisite

### 3.7.1 The key distinction

Chapters 1&ndash;3 have developed algebraic structure: vector spaces, dual spaces, operators, tensors, and contractions. **None of this requires an inner product.** The inner product is additional structure that:

1. Identifies $V$ with $V^*$ (the Riesz isomorphism, Chapter 4).
2. Induces norms, hence notions of distance, magnitude, and energy.
3. Provides a canonical way to lower/raise indices (the metric tensor).

<div class="scholium-box">

**Scholium 3.26 (Why the distinction matters).** In machine learning, the choice of inner product is a modeling decision, not a mathematical necessity:

- **Standard Euclidean inner product:** $\langle x,y\rangle = x^\top y$. This is the default in most frameworks. It treats all feature dimensions as equally important.
- **Mahalanobis inner product:** $\langle x,y\rangle_M = x^\top M y$ for positive definite $M$. This rescales and rotates the geometry, equivalent to whitening. Metric learning is literally the task of learning $M$.
- **Attention-induced inner product:** In self-attention, the effective geometry on token space is determined by $W_Q^\top W_K$, which defines a learned bilinear form on the embedding space.

The algebraic structure (tensor products, contractions, ranks) is independent of all these choices. The inner product introduces **quantitative geometry** on top of the algebra.

</div>

---

### 3.7.2 Induced inner product on tensor spaces

<div class="prop-box">

**Proposition 3.27 (Induced inner product on $V\otimes W$).** If $V$ and $W$ are equipped with inner products $\langle\cdot,\cdot\rangle_V$ and $\langle\cdot,\cdot\rangle_W$ respectively, then there is a unique inner product on $V\otimes W$ satisfying

$$\langle v_1\otimes w_1,\,v_2\otimes w_2\rangle_{V\otimes W} = \langle v_1,v_2\rangle_V\cdot\langle w_1,w_2\rangle_W,$$

extended by bilinearity to all of $V\otimes W$.

</div>

<div class="proof-box">

**Proof.** The map $(v_1\otimes w_1,\,v_2\otimes w_2)\mapsto \langle v_1,v_2\rangle_V\cdot\langle w_1,w_2\rangle_W$ is bilinear in each tensor argument. By the universal property of the tensor product (applied twice), it extends uniquely to a bilinear form on $V\otimes W$. Positive definiteness follows from choosing orthonormal bases: the induced inner product makes $\{e_i\otimes f_j\}$ orthonormal whenever $\{e_i\}$ and $\{f_j\}$ are orthonormal. $\blacksquare$

</div>

<div class="ml-box">

**Machine-learning consequence.** The Frobenius norm of a matrix $\lVert A\rVert_F = \sqrt{\sum_{ij}A_{ij}^2}$ is exactly the norm induced by this tensor inner product when we view $A\in\mathbb{R}^{m\times n}\cong\mathbb{R}^m\otimes\mathbb{R}^n$. Weight decay (L2 regularization) penalizes $\lVert W\rVert_F^2$, which is the energy of $W$ as an element of a tensor product space equipped with the standard induced inner product.

</div>

---

## 3.8 Transition to Chapter 4: From Algebra to Analysis

Chapters 1&ndash;3 have established the algebraic infrastructure of machine intelligence:

| Chapter | Object | Role |
|---------|--------|------|
| 1 | Vector spaces, dual spaces | States and measurements |
| 2 | Operators, eigenvalues, SVD | Evolution, persistence, capacity |
| 3 | Tensors, contraction, index topology | Multi-way interaction, information flow |

The missing ingredient is **quantitative geometry**: how to measure magnitude, energy, and distance in these spaces, and how such measurements induce stability, generalization bounds, and optimization geometry.

<div class="scholium-box">

**Scholium 3.28 (Preview of Chapter 4).** Once inner products and norms are introduced on the spaces constructed in Chapters 1&ndash;3, every tensorial construction inherits a notion of energy and conditioning. This allows one to speak sharply about:

- **Lipschitz control of networks:** $\lVert f(x)-f(y)\rVert \le L\lVert x-y\rVert$ requires bounding operator norms of Jacobians, which are tensors.
- **Stability of attention operators:** the conditioning number of $A(X)$ determines how sensitive the output is to perturbations of the input.
- **Spectral growth of Jacobians in deep compositions:** $\prod_{l=1}^{L}\lVert J_l\rVert$ controls signal propagation, requiring norm estimates on tensor-valued derivatives.
- **Norm-regularized generalization:** PAC-Bayes and Rademacher bounds involve $\lVert W\rVert_F$ and $\lVert W\rVert_2$, which are norms on tensor spaces.

Accordingly, **Chapter 4** develops Hilbert space structure and normed spaces as the quantitative layer that turns algebra into analysis.

</div>

---

<div style="text-align:center; margin-top:2em; padding:1em; border-top:2px solid #6a1b9a;">

**End of Chapter 3**

*Next: Chapter 4 &mdash; Hilbert Spaces, Norms, and the Geometry of Learning*

</div>
