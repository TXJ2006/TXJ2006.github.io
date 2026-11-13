{% extends '_layout.swig' %}
{% block title %}{{ title }} | 博客{% endblock %}
{% block page_class %}page-home{% endblock %}
{% block content %}
<header class="journal-intro"><h1>数学、阅读与研究。</h1></header>
{% set featured = editorial_featured() %}
<section class="featured-stories" aria-label="精选阅读">
  {% for item in featured %}
  <article class="feature-story {% if loop.first %}feature-story--lead{% endif %}">
    <a href="{{ item.url }}" class="story-image" aria-label="阅读：{{ item.title }}"><img src="/images/editorial/{{ item.image }}.webp" alt="" width="1440" height="960" {% if loop.first %}fetchpriority="high"{% else %}loading="lazy"{% endif %}></a>
    <h2><a href="{{ item.url }}">{{ item.title }}</a></h2>
    <p class="story-meta">{{ item.category }}{% if item.date %} · <time datetime="{{ date(item.date,'YYYY-MM-DD') }}">{{ date(item.date,'YYYY年M月D日') }}</time>{% endif %}</p>
    {% if loop.first %}<p class="story-summary">{{ item.summary }}</p><a class="text-link" href="{{ item.url }}">进入专题 <span aria-hidden="true">↗</span></a>{% endif %}
  </article>
  {% endfor %}
</section>
<section class="journal-library" aria-labelledby="library-title">
  <h2 id="library-title" class="section-title">文章与书单</h2>
  <div class="article-grid">
    {% set catalog = editorial_catalog() %}
    {% for item in catalog %}
    {% if item.category != '专题系列' %}
    <article class="journal-card">
      <a class="story-image" href="{{ item.url }}" tabindex="-1" aria-hidden="true"><img src="/images/editorial/{{ item.image }}.webp" alt="" width="1440" height="960" loading="lazy"></a>
      <h3><a href="{{ item.url }}">{{ item.title }}</a></h3>
      <p class="story-meta">{{ item.category }}{% if item.date %} · <time datetime="{{ date(item.date,'YYYY-MM-DD') }}">{{ date(item.date,'YYYY年M月D日') }}</time>{% endif %}</p>
    </article>
    {% endif %}
    {% endfor %}
  </div>
</section>
{% endblock %}
