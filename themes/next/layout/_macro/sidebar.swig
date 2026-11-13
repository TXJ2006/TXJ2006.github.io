{% macro render(is_post) %}
{% if page.toc !== false and toc(page.content).length > 1 %}
<aside class="reading-toc" aria-label="文章目录"><details open><summary>本页目录</summary><nav>{{ toc(page.content, {list_number:false,max_depth:2}) }}</nav></details></aside>
{% endif %}
{% endmacro %}
