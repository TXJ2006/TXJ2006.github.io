'use strict';

const escapeHTML = require('hexo-util').escapeHTML;

// Protect TeX from Markdown escapes, smart punctuation, and HTML parsing.
hexo.extend.filter.register('before_post_render', function (data) {
  if (!data.mathjax) return data;

  const tokens = /(`{3,}|~{3,})[\s\S]*?\1|`+[^`\n]*`+|\$\$[\s\S]*?\$\$|(?<!\\)\$(?!\$)(?:\\.|[^$\\\n])+\$(?!\$)/g;
  data.content = data.content.replace(tokens, function (token) {
    if (token[0] !== '$') return token;
    return '<escape>' + escapeHTML(token) + '</escape>';
  });

  return data;
}, 5);
