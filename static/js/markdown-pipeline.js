(() => {
  const FRONT_MATTER = /^---\s*\n[\s\S]*?\n---\s*(?:\n|$)/;
  const FENCE = /^\s*(`{3,}|~{3,})/;
  const DISPLAY_DELIMITER = /^(?<prefix>\s*(?:>\s*)?)\$\$\s*$/;
  const INLINE_MATH = /(?<!\\)\$(?!\$)([^\n$]+?)(?<!\\)\$/g;
  const FOOTNOTE_DEFINITION = /^\[\^([^\]]+)\]:\s*(.*)$/;

  function escapeHtml(value) {
    return value
      .replaceAll('&', '&amp;')
      .replaceAll('<', '&lt;')
      .replaceAll('>', '&gt;')
      .replaceAll('"', '&quot;');
  }

  function stripFrontMatter(markdown) {
    return markdown.replace(FRONT_MATTER, '');
  }

  function extractFootnotes(markdown) {
    const definitions = new Map();
    const lines = markdown.split('\n');
    const body = [];

    for (let index = 0; index < lines.length; index += 1) {
      const match = FOOTNOTE_DEFINITION.exec(lines[index]);
      if (!match) {
        body.push(lines[index]);
        continue;
      }

      const content = [match[2]];
      while (index + 1 < lines.length && /^(?: {2,}|\t)\S/.test(lines[index + 1])) {
        content.push(lines[index + 1].trim());
        index += 1;
      }
      definitions.set(match[1], content.join(' '));
    }

    const order = [];
    const numbered = body.join('\n').replace(/\[\^([^\]]+)\]/g, (full, id) => {
      if (!definitions.has(id)) return full;
      if (!order.includes(id)) order.push(id);
      const number = order.indexOf(id) + 1;
      const safeId = escapeHtml(id);
      return `<sup class="footnote-ref"><a href="#fn-${safeId}" id="fnref-${safeId}">${number}</a></sup>`;
    });

    if (!order.length) return numbered;
    const items = order.map((id) => {
      const content = window.marked.parseInline(definitions.get(id), { gfm: true });
      const safeId = escapeHtml(id);
      return `<li id="fn-${safeId}">${content} <a class="footnote-backref" href="#fnref-${safeId}" aria-label="Back to reference">&#8617;</a></li>`;
    }).join('');
    return `${numbered}\n\n<section class="footnotes"><ol>${items}</ol></section>`;
  }

  function protectMath(markdown) {
    const math = [];
    const lines = markdown.split('\n');
    const output = [];
    let fence = '';

    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index];
      const fenceMatch = FENCE.exec(line);
      if (fenceMatch) {
        if (!fence) fence = fenceMatch[1][0];
        else if (fenceMatch[1][0] === fence) fence = '';
        output.push(line);
        continue;
      }
      if (fence) {
        output.push(line);
        continue;
      }

      const display = DISPLAY_DELIMITER.exec(line);
      if (display) {
        const prefix = display.groups.prefix;
        const content = [];
        let end = index + 1;
        for (; end < lines.length; end += 1) {
          const closing = DISPLAY_DELIMITER.exec(lines[end]);
          if (closing && closing.groups.prefix === prefix) break;
          content.push(prefix.trimStart().startsWith('>') ? lines[end].replace(/^\s*>\s?/, '') : lines[end]);
        }
        if (end < lines.length) {
          const token = `md-math-${math.length}`;
          math.push({ token, value: `$$\n${content.join('\n')}\n$$` });
          output.push(`${prefix}<span data-md-math="${token}"></span>`);
          index = end;
          continue;
        }
      }

      output.push(line.replace(INLINE_MATH, (_, formula) => {
        const token = `md-math-${math.length}`;
        math.push({ token, value: `$${formula}$` });
        return `<span data-md-math="${token}"></span>`;
      }));
    }
    return { markdown: output.join('\n'), math };
  }

  function restoreMath(html, math) {
    return math.reduce((result, item) => {
      const placeholder = `<span data-md-math="${item.token}"></span>`;
      return result.replaceAll(placeholder, item.value);
    }, html);
  }

  function sanitizeEquationHtml(markdown) {
    const lines = markdown.split('\n');
    let inContainer = false;
    let inMath = false;
    return lines.map((line) => {
      if (/<div class="(?:display-equation|numbered-equation)/.test(line)) inContainer = true;
      if (inContainer && DISPLAY_DELIMITER.test(line)) {
        inMath = !inMath;
        return line;
      }
      const result = inContainer && inMath ? line.replaceAll('<', '&lt;') : line;
      if (inContainer && line.trim() === '</div>') {
        inContainer = false;
        inMath = false;
      }
      return result;
    }).join('\n');
  }

  function prepareForPublish(markdown) {
    const lines = markdown.replaceAll('\r\n', '\n').split('\n');
    const output = [];
    let fence = '';
    let inEquationContainer = false;

    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index];
      const fenceMatch = FENCE.exec(line);
      if (fenceMatch) {
        if (!fence) fence = fenceMatch[1][0];
        else if (fenceMatch[1][0] === fence) fence = '';
        output.push(line);
        continue;
      }
      if (fence) {
        output.push(line);
        continue;
      }
      if (/<div class="(?:display-equation|numbered-equation)/.test(line)) inEquationContainer = true;
      if (inEquationContainer) {
        output.push(line);
        if (line.trim() === '</div>') inEquationContainer = false;
        continue;
      }

      const opening = DISPLAY_DELIMITER.exec(line);
      if (!opening) {
        output.push(line);
        continue;
      }
      const prefix = opening.groups.prefix;
      let end = index + 1;
      for (; end < lines.length; end += 1) {
        const closing = DISPLAY_DELIMITER.exec(lines[end]);
        if (closing && closing.groups.prefix === prefix) break;
      }
      if (end === lines.length) {
        output.push(line);
        continue;
      }
      output.push(`${prefix}<div class="display-equation">`, ...lines.slice(index, end + 1), `${prefix}</div>`);
      index = end;
    }
    return `${sanitizeEquationHtml(output.join('\n')).trimEnd()}\n`;
  }

  function prepareForEdit(markdown) {
    const lines = markdown.replaceAll('\r\n', '\n').split('\n');
    const output = [];
    let containerPrefix = null;
    let inMath = false;

    lines.forEach((line) => {
      const opening = /^(\s*(?:>\s*)?)<div class="display-equation">\s*$/.exec(line);
      if (opening && containerPrefix === null) {
        containerPrefix = opening[1];
        return;
      }
      if (containerPrefix !== null && line === `${containerPrefix}</div>`) {
        containerPrefix = null;
        inMath = false;
        return;
      }
      if (containerPrefix !== null && DISPLAY_DELIMITER.test(line)) {
        inMath = !inMath;
        output.push(line);
        return;
      }
      output.push(containerPrefix !== null && inMath ? line.replaceAll('&lt;', '<') : line);
    });
    return `${output.join('\n').trimEnd()}\n`;
  }

  function render(markdown) {
    const withoutFrontMatter = stripFrontMatter(markdown.replaceAll('\r\n', '\n'));
    const withFootnotes = extractFootnotes(withoutFrontMatter);
    const protectedSource = protectMath(withFootnotes);
    const html = window.marked.parse(protectedSource.markdown, {
      async: false,
      breaks: false,
      gfm: true,
    });
    return restoreMath(html, protectedSource.math);
  }

  function diagnose(markdown) {
    const source = markdown.replaceAll('\r\n', '\n');
    const lines = source.split('\n');
    const issues = [];
    const footnoteDefinitions = new Set();
    const footnoteReferences = [];
    let displayStart = 0;
    let fence = '';
    let fenceStart = 0;

    if (source.startsWith('---\n') && !FRONT_MATTER.test(source)) {
      issues.push({ stage: 'Front matter', line: 1, message: 'YAML front matter 没有闭合的 ---。' });
    }

    lines.forEach((line, index) => {
      const lineNumber = index + 1;
      const fenceMatch = FENCE.exec(line);
      if (fenceMatch) {
        if (!fence) {
          fence = fenceMatch[1][0];
          fenceStart = lineNumber;
        } else if (fenceMatch[1][0] === fence) {
          fence = '';
          fenceStart = 0;
        }
        return;
      }
      if (fence) return;

      if (DISPLAY_DELIMITER.test(line)) {
        displayStart = displayStart ? 0 : lineNumber;
      }
      const definition = FOOTNOTE_DEFINITION.exec(line);
      if (definition) footnoteDefinitions.add(definition[1]);
      for (const reference of line.matchAll(/\[\^([^\]]+)\]/g)) {
        if (!definition || reference[1] !== definition[1]) {
          footnoteReferences.push({ id: reference[1], line: lineNumber });
        }
      }
    });

    if (fence) issues.push({ stage: 'Markdown', line: fenceStart, message: '代码围栏没有闭合。' });
    if (displayStart) issues.push({ stage: 'Math', line: displayStart, message: '块级公式缺少结尾的 $$。' });
    footnoteReferences.forEach((reference) => {
      if (!footnoteDefinitions.has(reference.id)) {
        issues.push({ stage: 'Markdown', line: reference.line, message: `脚注 [^${reference.id}] 缺少定义。` });
      }
    });
    return issues;
  }

  window.MarkdownPipeline = { diagnose, prepareForEdit, prepareForPublish, render, stripFrontMatter };
})();
