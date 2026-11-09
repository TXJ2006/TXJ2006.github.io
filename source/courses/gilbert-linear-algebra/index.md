---
title: 吉尔伯特线性代数
layout: page
permalink: /courses/gilbert-linear-algebra/
comments: false
---

本页整理 Gilbert Strang 的线性代数教材，以及 MIT OpenCourseWare 18.06SC Linear Algebra（2011 年秋季）的课程 PDF。课程从线性方程组与矩阵消元出发，逐步讨论四个基本子空间、正交投影、最小二乘、行列式、特征值、正定矩阵和奇异值分解。

## 学习笔记

[读书与做题笔记](/courses/gilbert-linear-algebra/notes/)

## 教材

### Introduction to Linear Algebra

- **作者：** Gilbert Strang
- **版本：** Sixth Edition，Wellesley-Cambridge Press，2023；440 页；ISBN 978-1-7331466-7-8
- **完整 PDF：** [here](/files/math-resources/strang-introduction-linear-algebra-6e.pdf)。打开时需要输入密码。
- **内容概述：** 以线性方程组和矩阵分解为主线，发展向量空间、四个基本子空间、特征值与特征向量、正交性、最小二乘和奇异值分解；第六版增加 A = CR 分解及新的应用材料。
- **授权说明：** 完整 PDF 由用户在获得版权方明确授权后提供，文件内容未作修改；请勿在授权范围之外复制或再分发。

## MIT 18.06SC 课程资料

- **授课教师：** Gilbert Strang
- **学期：** Fall 2011
- **课程 PDF：** 本站整理了 35 份讲次摘要、31 份课堂问题及对应解答，以及三次考试、期末考试、解答和复习资料。PDF 来自 MIT OCW 官方课程包，未修改内容。
- **视频：** [MIT OCW 视频与课程资源索引](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/resource-index/)列出逐讲课程视频和问题讲解视频。
- **课程大纲：** [MIT OCW Syllabus](https://ocw.mit.edu/courses/18-06sc-linear-algebra-fall-2011/pages/syllabus/)
- **许可：** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。请按许可要求注明来源、限非商业用途并以相同许可分享改编内容。

### Unit I：Ax = b 与四个基本子空间

| 讲次 | 主题 | 讲义摘要 | 课堂问题 | 解答 |
| --- | --- | --- | --- | --- |
| 1.1 | The Geometry of Linear Equations | [PDF](/files/mit-18-06sc/unit-1-session-1-summary.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-1-problems.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-1-solutions.pdf) |
| 1.2 | Elimination with Matrices | [PDF](/files/mit-18-06sc/unit-1-session-2-summary.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-2-problems.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-2-solutions.pdf) |
| 1.3 | Multiplication and Inverse Matrices | [PDF](/files/mit-18-06sc/unit-1-session-3-summary.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-3-problems.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-3-solutions.pdf) |
| 1.4 | Factorization into A = LU | [PDF](/files/mit-18-06sc/unit-1-session-4-summary.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-4-problems.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-4-solutions.pdf) |
| 1.5 | Transposes, Permutations, Vector Spaces | [PDF](/files/mit-18-06sc/unit-1-session-5-summary.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-5-problems.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-5-solutions.pdf) |
| 1.6 | Column Space and Nullspace | [PDF](/files/mit-18-06sc/unit-1-session-6-summary.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-6-problems.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-6-solutions.pdf) |
| 1.7 | Solving Ax = 0: Pivot Variables, Special Solutions | [PDF](/files/mit-18-06sc/unit-1-session-7-summary.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-7-problems.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-7-solutions.pdf) |
| 1.8 | Solving Ax = b: Row Reduced Form R | [PDF](/files/mit-18-06sc/unit-1-session-8-summary.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-8-problems.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-8-solutions.pdf) |
| 1.9 | Independence, Basis and Dimension | [PDF](/files/mit-18-06sc/unit-1-session-9-summary.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-9-problems.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-9-solutions.pdf) |
| 1.10 | The Four Fundamental Subspaces | [PDF](/files/mit-18-06sc/unit-1-session-10-summary.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-10-problems.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-10-solutions.pdf) |
| 1.11 | Matrix Spaces; Rank 1; Small World Graphs | [PDF](/files/mit-18-06sc/unit-1-session-11-summary.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-11-problems.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-11-solutions.pdf) |
| 1.12 | Graphs, Networks, Incidence Matrices | [PDF](/files/mit-18-06sc/unit-1-session-12-summary.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-12-problems.pdf) | [PDF](/files/mit-18-06sc/unit-1-session-12-solutions.pdf) |
| 1.13 | An Overview of Key Ideas | [PDF](/files/mit-18-06sc/unit-1-session-13-summary.pdf) | - | - |
| 1.14 | Exam 1 Review | [PDF](/files/mit-18-06sc/unit-1-session-14-summary.pdf) | - | - |

### Unit II：最小二乘、行列式与特征值

| 讲次 | 主题 | 讲义摘要 | 课堂问题 | 解答 |
| --- | --- | --- | --- | --- |
| 2.1 | Orthogonal Vectors and Subspaces | [PDF](/files/mit-18-06sc/unit-2-session-1-summary.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-1-problems.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-1-solutions.pdf) |
| 2.2 | Projections onto Subspaces | [PDF](/files/mit-18-06sc/unit-2-session-2-summary.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-2-problems.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-2-solutions.pdf) |
| 2.3 | Projection Matrices and Least Squares | [PDF](/files/mit-18-06sc/unit-2-session-3-summary.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-3-problems.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-3-solutions.pdf) |
| 2.4 | Orthogonal Matrices and Gram-Schmidt | [PDF](/files/mit-18-06sc/unit-2-session-4-summary.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-4-problems.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-4-solutions.pdf) |
| 2.5 | Properties of Determinants | [PDF](/files/mit-18-06sc/unit-2-session-5-summary.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-5-problems.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-5-solutions.pdf) |
| 2.6 | Determinant Formulas and Cofactors | [PDF](/files/mit-18-06sc/unit-2-session-6-summary.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-6-problems.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-6-solutions.pdf) |
| 2.7 | Cramer's Rule, Inverse Matrix and Volume | [PDF](/files/mit-18-06sc/unit-2-session-7-summary.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-7-problems.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-7-solutions.pdf) |
| 2.8 | Eigenvalues and Eigenvectors | [PDF](/files/mit-18-06sc/unit-2-session-8-summary.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-8-problems.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-8-solutions.pdf) |
| 2.9 | Diagonalization and Powers of A | [PDF](/files/mit-18-06sc/unit-2-session-9-summary.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-9-problems.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-9-solutions.pdf) |
| 2.10 | Differential Equations and exp(At) | [PDF](/files/mit-18-06sc/unit-2-session-10-summary.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-10-problems.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-10-solutions.pdf) |
| 2.11 | Markov Matrices; Fourier Series | [PDF](/files/mit-18-06sc/unit-2-session-11-summary.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-11-problems.pdf) | [PDF](/files/mit-18-06sc/unit-2-session-11-solutions.pdf) |
| 2.12 | Exam 2 Review | [PDF](/files/mit-18-06sc/unit-2-session-12-summary.pdf) | - | - |

### Unit III：正定矩阵与应用

| 讲次 | 主题 | 讲义摘要 | 课堂问题 | 解答 |
| --- | --- | --- | --- | --- |
| 3.1 | Symmetric Matrices and Positive Definiteness | [PDF](/files/mit-18-06sc/unit-3-session-1-summary.pdf) | [PDF](/files/mit-18-06sc/unit-3-session-1-problems.pdf) | [PDF](/files/mit-18-06sc/unit-3-session-1-solutions.pdf) |
| 3.2 | Complex Matrices; Fast Fourier Transform (FFT) | [PDF](/files/mit-18-06sc/unit-3-session-2-summary.pdf) | [PDF](/files/mit-18-06sc/unit-3-session-2-problems.pdf) | [PDF](/files/mit-18-06sc/unit-3-session-2-solutions.pdf) |
| 3.3 | Positive Definite Matrices and Minima | [PDF](/files/mit-18-06sc/unit-3-session-3-summary.pdf) | [PDF](/files/mit-18-06sc/unit-3-session-3-problems.pdf) | [PDF](/files/mit-18-06sc/unit-3-session-3-solutions.pdf) |
| 3.4 | Similar Matrices and Jordan Form | [PDF](/files/mit-18-06sc/unit-3-session-4-summary.pdf) | [PDF](/files/mit-18-06sc/unit-3-session-4-problems.pdf) | [PDF](/files/mit-18-06sc/unit-3-session-4-solutions.pdf) |
| 3.5 | Singular Value Decomposition | [PDF](/files/mit-18-06sc/unit-3-session-5-summary.pdf) | [PDF](/files/mit-18-06sc/unit-3-session-5-problems.pdf) | [PDF](/files/mit-18-06sc/unit-3-session-5-solutions.pdf) |
| 3.6 | Linear Transformations and Their Matrices | [PDF](/files/mit-18-06sc/unit-3-session-6-summary.pdf) | [PDF](/files/mit-18-06sc/unit-3-session-6-problems.pdf) | [PDF](/files/mit-18-06sc/unit-3-session-6-solutions.pdf) |
| 3.7 | Change of Basis; Image Compression | [PDF](/files/mit-18-06sc/unit-3-session-7-summary.pdf) | [PDF](/files/mit-18-06sc/unit-3-session-7-problems.pdf) | [PDF](/files/mit-18-06sc/unit-3-session-7-solutions.pdf) |
| 3.8 | Left and Right Inverses; Pseudoinverse | [PDF](/files/mit-18-06sc/unit-3-session-8-summary.pdf) | [PDF](/files/mit-18-06sc/unit-3-session-8-problems.pdf) | [PDF](/files/mit-18-06sc/unit-3-session-8-solutions.pdf) |
| 3.9 | Exam 3 Review | [PDF](/files/mit-18-06sc/unit-3-session-9-summary.pdf) | - | - |

## 考试与补充讲义

- [第一次考试](/files/mit-18-06sc/exam-1.pdf) · [解答](/files/mit-18-06sc/exam-1-solutions.pdf)
- [第二次考试](/files/mit-18-06sc/exam-2.pdf) · [解答](/files/mit-18-06sc/exam-2-solutions.pdf)
- [第三次考试](/files/mit-18-06sc/exam-3.pdf) · [解答](/files/mit-18-06sc/exam-3-solutions.pdf)
- [期末考试](/files/mit-18-06sc/final-exam.pdf) · [解答](/files/mit-18-06sc/final-exam-solutions.pdf)
- [期末复习讲义](/files/mit-18-06sc/final-review.pdf)
- [Zoom 课堂笔记](/files/mit-18-06sc/zoom-notes.pdf)
