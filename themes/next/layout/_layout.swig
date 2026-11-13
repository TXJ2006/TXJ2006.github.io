<!DOCTYPE html>
{% set version = next_env('version') %}
{% set title = config.title %}
{% set subtitle = config.subtitle %}
{% set author = config.author %}
{% set description = config.description %}
<html class="theme-next editorial" lang="{{ config.language }}" data-theme="light">
<head>
  {{ partial('_partials/head/head.swig', {}, {cache: theme.cache.enable}) }}
  {% include '_partials/head/head-unique.swig' %}
  <title>{% block title %}{% endblock %}</title>
  <link rel="stylesheet" href="/css/editorial.css?v=1">
  <script>try{var t=localStorage.getItem('blog-theme');if(t==='dark'||(!t&&matchMedia('(prefers-color-scheme: dark)').matches))document.documentElement.dataset.theme='dark'}catch(e){}</script>
</head>
<body itemscope itemtype="https://schema.org/WebPage" class="editorial-site">
<a class="skip-link" href="#content">跳到正文</a>
<div class="container {% block page_class %}{% endblock %}">
  <header class="masthead">
    <a class="wordmark" href="/">Xujiang Tang</a>
    <button class="menu-toggle" aria-expanded="false" aria-controls="primary-navigation">菜单</button>
    <nav id="primary-navigation" class="primary-navigation" aria-label="主导航">
      <a href="/">首页</a><a href="/about/">关于</a><a href="/research/">Research</a><a href="/publications/">Publications</a><a href="/cv/">CV</a><a href="/blog/">博客</a>
      <details class="more-navigation"><summary>资料与课程</summary><div><a href="/math-resources/">数学资料</a><a href="/courses/gilbert-linear-algebra/">吉尔伯特线性代数</a><a href="/courses/mit-18-703/">现代代数</a><a href="/courses/mit-18-700/">MIT 18.700</a><a href="/math-qual-exams/">博资考资料</a><a href="/categories/">分类</a><a href="/tags/">标签</a><a href="/archives/">归档</a></div></details>
    </nav>
    <div class="header-actions"><button class="theme-toggle" aria-label="切换为深色模式">深色</button><button class="search-toggle">搜索</button></div>
  </header>
  <main id="main" class="main"><div class="main-inner">
    <div class="content-wrap"><div id="content" class="content" tabindex="-1">{% block content %}{% endblock %}</div></div>
    {% block sidebar %}{% endblock %}
  </div></main>
  <footer class="editorial-footer">
    <a class="footer-wordmark" href="/">Xujiang Tang</a>
    <div class="footer-links"><a href="/about/">关于</a><a href="/math-resources/">数学资料</a><a href="/math-qual-exams/">博资考资料</a><a href="/archives/">归档</a><a href="https://github.com/TXJ2006">GitHub</a><a href="/atom.xml">RSS</a></div>
    <p>数学、理论机器学习与人工智能。</p><p class="copyright">© {{ date(Date.now(), 'YYYY') }} Xujiang Tang</p>
    {% block footer %}{% endblock %}
  </footer>
</div>
<div class="site-search-overlay" hidden><section class="blog-search-panel" role="dialog" aria-modal="true" aria-labelledby="site-search-title">
  <div class="search-dialog-heading"><h2 id="site-search-title">搜索本站</h2><button class="search-close" aria-label="关闭搜索">关闭</button></div>
  <form class="site-search-form" role="search"><label for="site-search-input">文章、专题与数学资料</label><input id="site-search-input" type="search" placeholder="试试：Research、分析学、Hatcher" autocomplete="off"></form>
  <p class="site-search-status" role="status">输入关键词搜索本站内容。</p><div class="site-search-results"></div>
</section></div>
<script src="{{ url_for(theme.vendors._internal) }}/jquery/index.js?v=2.1.3"></script>
<script src="{{ url_for(theme.js) }}/utils.js?v={{ version }}"></script>
{% include '_third-party/math/index.swig' %}
<script src="/js/editorial.js?v=1" defer></script>
</body>
</html>
