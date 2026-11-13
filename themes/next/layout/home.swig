{% extends '_layout.swig' %}
{% block title %}Xujiang Tang 唐煦江{% endblock %}
{% block page_class %}page-home page-profile{% endblock %}
{% block content %}

<section class="profile" aria-labelledby="profile-name">
  <div class="profile-copy">
    <p class="eyebrow">Algebraic Topology · Efficient AI Reasoning · Bandits · Reinforcement Learning · Learning Theory · Algorithm Design · Formalized Mathematics</p>
    <h1 id="profile-name">Xujiang Tang <span lang="zh-CN">唐煦江</span></h1>
    <p class="profile-role">I believe mathematics can open broader frontiers for machine learning and artificial intelligence, extending their foundations beyond the calculus-centered framework that has shaped modern AI.</p>
    <p class="profile-affiliation">B.Sc. candidate in Mathematics and Applied Mathematics and Research Assistant at The Hong Kong University of Science and Technology (Guangzhou), working with Prof. Tianyuan Jin.</p>
    <div class="profile-actions">
      <a class="button-primary" href="/research/">Research</a>
      <a href="/publications/">Publications</a>
      <a href="/files/XujiangTang_CV_2026.pdf">Download CV</a>
    </div>
    <div class="profile-social"><a href="mailto:txj_262538@163.com">Email</a><a href="https://github.com/TXJ2006">GitHub</a><a href="https://orcid.org/0009-0008-1127-6420">ORCID</a></div>
  </div>
  <figure class="profile-portrait">
    <img src="/images/youyu-chitang-source.jpg" alt="You Yu and Chi Tang reading together" width="240" height="240">
  </figure>
</section>

<dl class="profile-summary" aria-label="Academic profile">
  <div><dt>Degree</dt><dd>B.Sc. Mathematics, 2027</dd></div>
  <div><dt>Current</dt><dd>Research Assistant, HKUST(GZ)</dd></div>
</dl>

<section class="home-block" aria-labelledby="research-heading">
  <div class="block-heading">
    <p class="eyebrow">Research</p>
    <h2 id="research-heading">Mathematics for broader frontiers in AI.</h2>
    <a href="/research/">Research overview</a>
  </div>
  <div class="research-grid">
    <article>
      <h3>Algebraic topology &amp; formalization</h3>
      <p>Homotopy theory, polyhedral products, quasi-Lie structures, and Lean-based formal mathematics.</p>
    </article>
    <article>
      <h3>Bandits &amp; reinforcement learning</h3>
      <p>Exploration, regret, best-arm identification, adaptive decisions, and sample-efficient learning.</p>
    </article>
    <article>
      <h3>Efficient AI reasoning</h3>
      <p>Resource-efficient reasoning, automated inference, and learning under limited data, feedback, and computation.</p>
    </article>
    <article>
      <h3>Learning theory &amp; algorithm design</h3>
      <p>Provable methods, optimization, implicit differentiation, and algorithms that transfer across AI model classes.</p>
    </article>
  </div>
</section>

<section class="home-block" aria-labelledby="work-heading">
  <div class="block-heading">
    <p class="eyebrow">Selected work</p>
    <h2 id="work-heading">Evidence of research preparation.</h2>
    <a href="/publications/">Complete list</a>
  </div>
  <ol class="work-list">
    <li><span class="status submitted">Submitted</span><div><strong>The Curvature Filter in Bilevel Optimization</strong><small>JMLR Theory and Methods · implicit differentiation without nondegeneracy</small></div></li>
    <li><span class="status published">Accepted</span><div><strong>Structure-preserving Koopman Predictive Control</strong><small>Biological Cybernetics</small></div></li>
    <li><span class="status published">Accepted</span><div><strong>Spectral Network Determinants of Seizure-like Synchronization</strong><small>Cognitive Neurodynamics · coupled Hindmarsh-Rose brain models</small></div></li>
  </ol>
</section>

<section class="home-block home-library" aria-labelledby="library-heading">
  <div class="block-heading"><h2 id="library-heading">Writing &amp; learning</h2></div>
  <div class="home-destinations">
    <a href="/blog/"><h3>博客</h3><p>数学讲义、阅读专题与研究随笔。</p><span class="text-link">浏览博客 ↗</span></a>
    <a href="/math-resources/"><h3>数学资料</h3><p>教材、课程讲义与学习参考资料。</p><span class="text-link">进入资料合集 ↗</span></a>
  </div>
</section>

{% endblock %}
