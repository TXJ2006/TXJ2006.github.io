---
title: "Reading Hatcher, Chapter 0: Homotopy and Deformation Retractions"
date: 2026-10-13 10:00:00
series: Algebraic Topology Reading Notes
series_order: 1
subtitle: A close reading of Chapter 0, with one complete calculation
description: Reading notes on homotopy, retractions, deformation retractions, and homotopy equivalence
categories:
  - Algebraic Topology
tags:
  - Hatcher
  - Homotopy
  - Deformation Retraction
  - Reading Notes
mathjax: true
toc: true
toc_number: false
comments: true
---

This note is a reading record, not a replacement for Hatcher's exposition. The main source is Allen Hatcher, *Algebraic Topology*, Chapter 0, pp. 1--4 (PDF pp. 10--13). The Stanford MATH 215A notes are used only as a parallel explanation of the same first ideas. I will keep one question in view: **when may a complicated space be replaced by a simpler one without changing its homotopy-level information?**

<!--more-->

## 1. A homotopy is a map

Let $X$ and $Y$ be topological spaces, and write $I=[0,1]$ for the unit interval. A map is assumed continuous unless stated otherwise. Given maps $f,g:X\to Y$, a **homotopy from $f$ to $g$** is a continuous map

$$
H:X\times I\longrightarrow Y
$$

such that

$$
H(x,0)=f(x),\qquad H(x,1)=g(x)\qquad (x\in X).
$$

For each $t\in I$, the map $H_t:X\to Y$ defined by $H_t(x)=H(x,t)$ is one stage of the deformation. The parameter $t$ records time; it is not a second coordinate of the point $x$. We write $f\simeq g$ when such an $H$ exists.

The important point is that continuity belongs to the joint map $H(x,t)$. A collection of pictures, or a separate continuous path for each $x$, is not enough until these paths vary continuously with $x$.

<div class="lecture-example">
  <p><strong>A first calculation.</strong> For $f,g:\mathbb R\to\mathbb R$ given by $f(x)=x$ and $g(x)=x^2$, set $H(x,t)=(1-t)x+tx^2$. This is a polynomial in $(x,t)$, hence continuous, and it satisfies the two endpoint conditions. Therefore $f\simeq g$. The formula is more informative than the picture: it tells us exactly what every point does at every time.</p>
</div>

The relation $\simeq$ is an equivalence relation on maps with the same domain and codomain. Reflexivity uses the constant homotopy $H(x,t)=f(x)$. If $H$ runs from $f$ to $g$, then $H(x,1-t)$ runs from $g$ to $f$. Finally, if $H$ runs from $f$ to $g$ and $K$ runs from $g$ to $h$, concatenate them by

$$
L(x,t)=
\begin{cases}
H(x,2t),&0\leq t\leq\frac12,\\
K(x,2t-1),&\frac12\leq t\leq1.
\end{cases}
$$

The two formulas agree at $t=\frac12$ because $H(x,1)=g(x)=K(x,0)$, so the pasting lemma makes $L$ continuous. This elementary fact is what allows us to speak of a map's homotopy class.

## 2. Retractions and the fixed part of a deformation

Let $A\subseteq X$ be a subspace. A **retraction** of $X$ onto $A$ is a map $r:X\to A$ satisfying

$$
r(a)=a\qquad (a\in A).
$$

The map $r$ is allowed to move points outside $A$, but it must leave every point of $A$ fixed. Let $i:A\hookrightarrow X$ denote the inclusion map. We may also view the retraction as the map $i\circ r:X\to X$ whose image is contained in $A$.

A **deformation retraction** of $X$ onto $A$ is a homotopy $H:X\times I\to X$ satisfying

$$
H(x,0)=x,\qquad H(x,1)=i(r(x)),\qquad H(a,t)=a.
$$

The first condition starts at the identity map of $X$; the second ends at a retraction; the third says that the target subspace is fixed throughout the deformation. With this notation,

$$
i\circ r\simeq\operatorname{id}_X\quad\text{and}\quad r\circ i=\operatorname{id}_A.
$$

The last equality is strict, while the first is only a homotopy. This distinction is easy to lose in an informal drawing and is one reason to write the three conditions separately.

## 3. Our complete example: an annulus retracts to its core circle

Consider the closed annulus

$$
A=\{x\in\mathbb R^2:1\leq\lVert x\rVert\leq 2\}
$$

and its core circle

$$
C=S^1=\{x\in\mathbb R^2:\lVert x\rVert=1\}.
$$

Here $\lVert x\rVert$ is the Euclidean norm. Define

$$
H:A\times I\longrightarrow A,
\qquad
H(x,t)=\left((1-t)+\frac{t}{\lVert x\rVert}\right)x.
$$

Why does this formula work?

1. Since $1\leq\lVert x\rVert\leq2$, the denominator never vanishes, so $H$ is continuous.
2. At $t=0$, $H(x,0)=x$.
3. At $t=1$, $H(x,1)=x/\lVert x\rVert\in C$.
4. If $x\in C$, then $\lVert x\rVert=1$, so $H(x,t)=x$ for every $t$.
5. The norm of $H(x,t)$ is $(1-t)\lVert x\rVert+t$, which lies between $1$ and $2$; hence the image stays inside $A$.

Thus $H$ is a deformation retraction of the annulus onto its core circle. The geometric phrase “push every point radially inward” becomes a theorem only after these five checks.

## 4. What the source figure is showing

![A deformation retraction of a Möbius band onto its core circle, reproduced as a source excerpt from Hatcher, Chapter 0, p. 2.](/images/notes/assets/algebraic-topology/hatcher-ch0-deformation-retraction-excerpt.png)

<p class="figure-caption"><em>Source excerpt: Allen Hatcher, <span lang="en">Algebraic Topology</span>, Chapter 0, p. 2 (PDF p. 11).</em></p>

The figure comes from Hatcher's discussion of deformation retractions. It shows a Möbius band being collapsed onto its core circle through intermediate stages. The picture supplies the geometric idea; it does not replace the continuity condition in the definition. In the annulus calculation above, the formula plays the role that the sequence of drawings plays in the source.

## 5. Homotopy equivalence

Maps $f:X\to Y$ and $g:Y\to X$ are **homotopy inverses** if

$$
g\circ f\simeq\operatorname{id}_X,
\qquad
f\circ g\simeq\operatorname{id}_Y.
$$

In this case $X$ and $Y$ are **homotopy equivalent**, written $X\simeq Y$. A deformation retraction is therefore a particularly concrete homotopy equivalence: the inclusion $i:A\hookrightarrow X$ and the retraction $r:X\to A$ are homotopy inverses.

This relation is weaker than homeomorphism. A homeomorphism preserves the whole topological structure, while a homotopy equivalence preserves only the structure visible to homotopy-invariant constructions. The point of algebraic topology is that fundamental groups, homology groups, and cohomology groups will be examples of such constructions.

<div class="lecture-note">
  <p><strong>Reading rule.</strong> Whenever a source says that one space “shrinks to” another, ask three questions: What are the two spaces? What is the actual map $H(x,t)$? Which points are required to remain fixed? If one of these is missing, the sentence is intuition rather than a completed proof.</p>
</div>

## 6. Exercises for this reading

1. Verify directly that the annulus formula above is continuous and fixes $C$ pointwise.
2. For a point $a\in X$, show that the constant map $X\to\{a\}$ is a retraction onto the subspace $\{a\}$, but that the existence of this retraction alone does not give a deformation retraction.
3. Write an explicit homotopy between two constant maps $\mathbb R^m\to\mathbb R^n$.

The next note will read quotient spaces and attaching maps before turning to Hatcher's CW-complex construction. This will be our first place where a geometric picture and a precise topological construction must be kept together.

## Sources

- Allen Hatcher, *Algebraic Topology*, Chapter 0, pp. 1--4: [author's PDF](https://pi.math.cornell.edu/~hatcher/AT/AT.pdf).
- Andrew Lin, notes for Zhenkun Li's Stanford MATH 215A, pp. 1--3: [course notes](https://web.stanford.edu/~lindrew/math215A.pdf).
