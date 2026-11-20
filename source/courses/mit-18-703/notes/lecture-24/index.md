---
title: 第 24 讲：全课程复习｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-24/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-23/">上一讲</a></p>
<div class="note-lead"><p>复习不是把定义重新抄一遍，而是把课程中的“结构—映射—商—分解”四条线重新接起来。遇到新对象时，先问它的运算是什么，再问哪些映射保留结构。</p></div>
<nav class="reading-links" aria-label="课程复习资料"><a href="/files/mit-18-703/practice-final-exam.pdf">MIT 练习期末考试 PDF</a></nav>

<h2 id="section-90">一、群论的统一框架</h2>
<p>群从对称和可逆动作开始。子群提供局部结构；陪集把子群平移成分割，并给出 Lagrange 定理；正规子群使陪集可以相乘，从而得到商群。群同态的核记录信息损失，第一同构定理把任意同态分解为“先商掉核，再无损到达像”。置换群和群作用则把抽象群嵌入具体的对称群，Sylow 定理用计数寻找正规子群。</p>
<p>做群论题时，优先寻找三个对象：作用的轨道、元素的稳定子、同态的核。它们分别负责计数、对称性和商结构。</p>

<h2 id="section-91">二、环论的统一框架</h2>
<p>环把加法群与乘法结合起来。理想是同态的核，也是构造商环的准确条件；素理想使商成为整环，极大理想使商成为域。分式域把整环嵌入最小的域中。整除理论再按结构强弱排列：Euclidean domain \(\Rightarrow\) PID \(\Rightarrow\) UFD \(\Rightarrow\) integral domain。</p>
<p>这条链中的每一步都增加了可计算性：欧几里得除法给出算法，PID 给出 Bézout 等式，UFD 给出唯一因式分解。多项式环通过 Gauss 引理把系数环的分解性质传递上去，Eisenstein 判别法则用模素数信息证明不可约。</p>

<h2 id="section-92">三、遇到证明题的检查顺序</h2>
<ol><li><strong>先核对结构：</strong>对象是否闭合，单位元和零元分别是什么，乘法是否交换，是否存在零因子。</li><li><strong>再找自然映射：</strong>优先考虑包含映射、评价映射、商映射、作用表示和共轭映射。</li><li><strong>最后使用正确的判别工具：</strong>群用核、指数和轨道；环用理想与商；因式分解用 gcd、ACC、Euclidean algorithm、Gauss lemma 或 Eisenstein criterion。</li></ol>
<p>一个常见错误是把结论的充分条件当成必要条件：例如整除 \(|G|\) 不保证存在对应阶子群，素理想不保证商是域，不可约元在任意整环中也不一定是素元。复习时应把每个定理的假设一起记住。</p>

<h2 id="section-93">四、后续学习的接口</h2>
<p>群作用将自然进入表示论、李群和几何对称；理想与商环将进入模、域扩张和 Galois 理论；UFD、PID 与多项式不可约性则是代数数论和代数几何的入口。课程到这里完成的是基础语言和基本证明工具，而不是抽象代数的终点。</p>
<p class="note-footer">本页依据 MIT OCW 18.703 全课程讲义整理；第 24 讲为课程复习页，原课程资料以练习期末考试和前 23 讲讲义为准。</p>
</section>
