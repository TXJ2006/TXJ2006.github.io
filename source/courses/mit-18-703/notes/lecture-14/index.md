---
title: 第 14 讲：环｜阅读笔记
layout: page
permalink: /courses/mit-18-703/notes/lecture-14/
mathjax: true
toc_number: false
comments: false
---

<section class="la-notes la-notes--algebra">
<p class="note-nav"><a href="/courses/mit-18-703/">← 课程目录</a> · <a href="/courses/mit-18-703/notes/lecture-13/">上一讲</a> · <a href="/courses/mit-18-703/notes/lecture-15/">下一讲 →</a></p>
<div class="note-lead"><p>群只有一个运算。环在同一个集合上同时保留加法与乘法，并要求乘法服从分配律；整数、多项式、矩阵和函数环都由这套结构统一描述。</p></div>
<nav class="reading-links" aria-label="原讲义"><a href="/files/mit-18-703/lecture-14.pdf">MIT 第 14 讲 PDF</a></nav>

<h2 id="section-50">一、环把两种运算放在一起</h2>
<p>本课程采用带单位元且 \(1\ne0\) 的环。环 \(R\) 首先是加法阿贝尔群；乘法满足结合律并有单位元 \(1\)；最后要求左右分配律：\(a(b+c)=ab+ac\)、\((a+b)c=ac+bc\)。乘法不必交换，因此矩阵环和四元数环是环，但通常不是交换环。</p>
<p>加法部分沿用群论中的记号。分配律把两种运算连接起来，正是它保证了诸如 \(a0=0\)、\(a(-b)=-(ab)\) 这类结论能够从环公理推出，而不是另行规定。</p>

<h2 id="section-51">二、子环与典型例子</h2>
<p>若子集 \(S\subseteq R\) 含有同一个单位元 \(1\)，并且对加法、加法逆元和乘法都闭合，那么 \(S\) 是子环。这里必须单独要求含 \(1\)：环没有乘法逆元，非空和闭合并不能自动产生单位元。</p>
<p>基本例子是 \(\mathbb Z\subset\mathbb Q\subset\mathbb R\subset\mathbb C\)。\(\mathbb Z[i]=\{a+bi:a,b\in\mathbb Z\}\) 是 \(\mathbb C\) 的子环；分母为奇数的有理数也构成子环。若 \(X\) 是集合、\(R\) 是环，则函数集 \(R^X\) 按逐点加法和乘法成环，连续函数集 \(C([0,1],\mathbb R)\) 是其中的子环。</p>
<p>矩阵集 \(M_n(R)\) 的加法逐项进行，乘法仍按“行乘列”定义，单位元是单位矩阵。乘法一般不交换，而且即使 \(R\) 没有零因子，矩阵环也可能有零因子。多项式环 \(R[x]\) 的元素是形式和 \(\sum a_ix^i\)，乘法由卷积给出；这里的 \(x\) 是形式不定元，不是必须代入某个数的函数变量。</p>

<h2 id="section-52">三、为什么 \(\mathbb Z/6\mathbb Z\) 有零因子？</h2>
<p>商类上的加法和乘法定义为 \([a]+[b]=[a+b]\)、\([a][b]=[ab]\)，良定义性来自同余关系。可是 \([2][3]=[6]=[0]\)，虽然 \([2]\)、\([3]\) 都不是零。这说明“环”只要求两种运算相容，并不要求非零元素相乘仍非零；后者是整环的额外性质。</p>
<p>讲义最后用四元数说明非交换乘法的来源：\(i^2=j^2=k^2=-1\)、\(ij=k\)、\(ji=-k\)，并由分配律和结合律扩张到所有 \(a+bi+cj+dk\)。因此环的统一语言同时容纳交换和非交换的算术。</p>
<h2 id="section-53">理解检查</h2>
<ol><li>为什么 \(2\mathbb Z\) 不是按本课程约定的子环？</li><li>在 \(M_2(\mathbb Z)\) 中找两个非零矩阵，其乘积为零。</li><li>说明多项式 \(x\) 与函数 \(x\mapsto x\) 在一般环中不能混为一谈。</li></ol>
<p class="note-footer">依据 MIT OCW 18.703 Lecture 14, <em>Rings</em> 整理。</p>
</section>
