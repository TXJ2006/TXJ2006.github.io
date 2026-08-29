{% extends '_layout.swig' %}
{% import '_macro/post.swig' as post_template %}
{% import '_macro/sidebar.swig' as sidebar_template %}

{% block title %}{{ title }}{% if theme.index_with_subtitle and subtitle %} – {{ subtitle }}{% endif %}{% endblock %}

{% block page_class %}
  {% if is_home() %}page-home{% endif -%}
{% endblock %}

{% block content %}
  <section id="posts" class="posts-expand">
    {% if page.posts and page.posts.length > 0 %}
      {% for post in page.posts %}
        {{ post_template.render(post, true) }}
      {% endfor %}
    {% else %}
      <div class="post-block page empty-blog">
        <header class="post-header">
          <h1 class="post-title">研究博客</h1>
        </header>
        <div class="post-body">
          <p>新的研究笔记正在整理中。这里将记录数学、理论机器学习与人工智能的学习和思考。</p>
          <p><a href="{{ url_for('/about/') }}">了解 Xujiang Tang</a> · <a href="{{ url_for('/archives/') }}">浏览归档</a></p>
        </div>
      </div>
    {% endif %}
  </section>

  {% include '_partials/pagination.swig' %}
{% endblock %}

{% block sidebar %}
  {{ sidebar_template.render(false) }}
{% endblock %}
