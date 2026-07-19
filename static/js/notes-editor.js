(() => {
  const page = document.querySelector('.editor-page');
  if (!page) return;

  const repo = page.dataset.repo;
  const owner = page.dataset.owner;
  const branch = page.dataset.branch;
  const apiRoot = `https://api.github.com/repos/${repo}`;
  const tokenKey = 'txj_notes_editor_token';
  const draftsKey = 'txj_notes_editor_drafts';
  const params = new URLSearchParams(window.location.search);

  const mathMacros = {
    '\\E': '\\mathbb{E}',
    '\\Pbb': '\\mathbb{P}',
    '\\Pp': '\\mathbb{P}',
    '\\Prob': '\\mathbb{P}',
    '\\KL': '\\operatorname{KL}',
    '\\kl': '\\operatorname{kl}',
    '\\Ber': '\\operatorname{Bern}',
    '\\Beta': '\\operatorname{Beta}',
    '\\Poi': '\\operatorname{Poisson}',
    '\\Normal': '\\mathcal{N}',
    '\\Var': '\\operatorname{Var}',
    '\\Cov': '\\operatorname{Cov}',
    '\\Reg': '\\operatorname{Reg}',
    '\\TV': '\\operatorname{TV}',
    '\\Alt': '\\operatorname{Alt}',
    '\\argmax': '\\operatorname*{arg\\,max}',
    '\\argmin': '\\operatorname*{arg\\,min}',
    '\\dd': '\\mathrm{d}',
    '\\one': '\\mathbf{1}',
    '\\ind': '\\mathbf{1}',
    '\\R': '\\mathbb{R}',
    '\\F': '\\mathcal{F}',
    '\\G': '\\mathcal{G}',
    '\\GP': '\\mathcal{GP}',
    '\\given': '\\,\\middle|\\,',
    '\\st': '\\text{subject to}',
    '\\bm': '\\boldsymbol{#1}',
    '\\label': '\\phantom{#1}',
    '\\midrule': '\\hline',
    '\\calA': '\\mathcal{A}',
    '\\calD': '\\mathcal{D}',
    '\\calF': '\\mathcal{F}',
    '\\calH': '\\mathcal{H}',
    '\\calX': '\\mathcal{X}',
  };

  const formulaCatalog = {
    Structures: [
      ['a/b', 'Fraction', '\\frac{a}{b}'],
      ['x²', 'Superscript', '{{selection}}^{2}'],
      ['xᵢ', 'Subscript', '{{selection}}_{i}'],
      ['√x', 'Square root', '\\sqrt{{{selection}}}'],
      ['ⁿ√x', 'Nth root', '\\sqrt[n]{{{selection}}}'],
      ['|x|', 'Absolute value', '\\left|{{selection}}\\right|'],
      ['‖x‖', 'Norm', '\\left\\|{{selection}}\\right\\|'],
      ['(n k)', 'Binomial coefficient', '\\binom{n}{k}'],
      ['cases', 'Piecewise function', 'f(x)=\\begin{cases}x^2,&x\\ge 0,\\\\-x,&x<0.\\end{cases}'],
      ['align', 'Aligned equations', '\\begin{aligned}a&=b+c,\\\\d&=e+f.\\end{aligned}'],
    ],
    Greek: [
      ['α', 'alpha', '\\alpha'], ['β', 'beta', '\\beta'], ['γ', 'gamma', '\\gamma'],
      ['δ', 'delta', '\\delta'], ['ε', 'epsilon', '\\varepsilon'], ['ζ', 'zeta', '\\zeta'],
      ['η', 'eta', '\\eta'], ['θ', 'theta', '\\theta'], ['κ', 'kappa', '\\kappa'],
      ['λ', 'lambda', '\\lambda'], ['μ', 'mu', '\\mu'], ['ν', 'nu', '\\nu'],
      ['ξ', 'xi', '\\xi'], ['π', 'pi', '\\pi'], ['ρ', 'rho', '\\rho'],
      ['σ', 'sigma', '\\sigma'], ['τ', 'tau', '\\tau'], ['φ', 'phi', '\\phi'],
      ['χ', 'chi', '\\chi'], ['ψ', 'psi', '\\psi'], ['ω', 'omega', '\\omega'],
      ['Γ', 'Gamma', '\\Gamma'], ['Δ', 'Delta', '\\Delta'], ['Θ', 'Theta', '\\Theta'],
      ['Λ', 'Lambda', '\\Lambda'], ['Σ', 'Sigma', '\\Sigma'], ['Φ', 'Phi', '\\Phi'],
      ['Ψ', 'Psi', '\\Psi'], ['Ω', 'Omega', '\\Omega'],
    ],
    Operators: [
      ['Σ', 'Summation', '\\sum_{i=1}^{n} {{selection}}'],
      ['Π', 'Product', '\\prod_{i=1}^{n} {{selection}}'],
      ['∫', 'Integral', '\\int_a^b {{selection}}\\,\\dd x'],
      ['∬', 'Double integral', '\\iint_D {{selection}}\\,\\dd x\\,\\dd y'],
      ['∮', 'Contour integral', '\\oint_\\gamma {{selection}}\\,\\dd z'],
      ['lim', 'Limit', '\\lim_{n\\to\\infty} {{selection}}'],
      ['d/dx', 'Derivative', '\\frac{\\dd}{\\dd x}{{selection}}'],
      ['∂/∂x', 'Partial derivative', '\\frac{\\partial}{\\partial x}{{selection}}'],
      ['∇', 'Gradient', '\\nabla {{selection}}'],
      ['Δ', 'Laplacian', '\\Delta {{selection}}'],
      ['∞', 'Infinity', '\\infty'], ['max', 'Maximum', '\\max_{x\\in X} {{selection}}'],
      ['min', 'Minimum', '\\min_{x\\in X} {{selection}}'],
      ['arg max', 'Arg maximum', '\\argmax_{x\\in X} {{selection}}'],
      ['arg min', 'Arg minimum', '\\argmin_{x\\in X} {{selection}}'],
    ],
    Relations: [
      ['=', 'Equal', '='], ['≠', 'Not equal', '\\ne'], ['≈', 'Approximately', '\\approx'],
      ['∼', 'Similar', '\\sim'], ['≅', 'Congruent', '\\cong'], ['≤', 'Less than or equal', '\\le'],
      ['≥', 'Greater than or equal', '\\ge'], ['≪', 'Much less than', '\\ll'],
      ['≫', 'Much greater than', '\\gg'], ['∝', 'Proportional', '\\propto'],
      ['⊥', 'Perpendicular', '\\perp'], ['∥', 'Parallel', '\\parallel'],
      ['≡', 'Equivalent', '\\equiv'], [':=', 'Defined as', ':='],
    ],
    Sets: [
      ['∈', 'Element of', '\\in'], ['∉', 'Not an element of', '\\notin'],
      ['⊂', 'Subset', '\\subset'], ['⊆', 'Subset or equal', '\\subseteq'],
      ['⊃', 'Superset', '\\supset'], ['⊇', 'Superset or equal', '\\supseteq'],
      ['∪', 'Union', '\\cup'], ['∩', 'Intersection', '\\cap'],
      ['∅', 'Empty set', '\\varnothing'], ['∖', 'Set difference', '\\setminus'],
      ['∀', 'For all', '\\forall'], ['∃', 'There exists', '\\exists'],
      ['ℕ', 'Natural numbers', '\\mathbb{N}'], ['ℤ', 'Integers', '\\mathbb{Z}'],
      ['ℚ', 'Rationals', '\\mathbb{Q}'], ['ℝ', 'Real numbers', '\\mathbb{R}'],
      ['ℂ', 'Complex numbers', '\\mathbb{C}'],
    ],
    Arrows: [
      ['→', 'Right arrow', '\\to'], ['←', 'Left arrow', '\\leftarrow'],
      ['↔', 'Left-right arrow', '\\leftrightarrow'], ['⇒', 'Implies', '\\Rightarrow'],
      ['⇐', 'Implied by', '\\Leftarrow'], ['⇔', 'If and only if', '\\Longleftrightarrow'],
      ['↦', 'Maps to', '\\mapsto'], ['↑', 'Up arrow', '\\uparrow'],
      ['↓', 'Down arrow', '\\downarrow'], ['⟶', 'Long right arrow', '\\longrightarrow'],
      ['⇀', 'Weak convergence', '\\rightharpoonup'], ['↗', 'North-east arrow', '\\nearrow'],
    ],
    'Linear algebra': [
      ['v⃗', 'Vector', '\\vec{{{selection}}}'], ['𝐱', 'Bold vector', '\\bm{{{selection}}}'],
      ['Aᵀ', 'Transpose', 'A^{\\mathsf T}'], ['A⁻¹', 'Inverse', 'A^{-1}'],
      ['⟨x,y⟩', 'Inner product', '\\langle x,y\\rangle'],
      ['2×2', '2 by 2 matrix', '\\begin{pmatrix}a&b\\\\c&d\\end{pmatrix}'],
      ['3×3', '3 by 3 matrix', '\\begin{pmatrix}a&b&c\\\\d&e&f\\\\g&h&i\\end{pmatrix}'],
      ['det', 'Determinant', '\\det(A)'], ['tr', 'Trace', '\\operatorname{tr}(A)'],
      ['rank', 'Rank', '\\operatorname{rank}(A)'], ['ker', 'Kernel', '\\ker(A)'],
      ['span', 'Span', '\\operatorname{span}\\{v_1,\\ldots,v_n\\}'],
      ['⊗', 'Tensor product', '\\otimes'], ['⊕', 'Direct sum', '\\oplus'],
    ],
    Probability: [
      ['E', 'Expectation', '\\E[{{selection}}]'], ['P', 'Probability', '\\Pbb({{selection}})'],
      ['Var', 'Variance', '\\Var({{selection}})'], ['Cov', 'Covariance', '\\Cov(X,Y)'],
      ['1', 'Indicator', '\\one\\{A\\}'], ['|', 'Conditional bar', '\\given'],
      ['N', 'Normal distribution', '\\Normal(\\mu,\\sigma^2)'],
      ['Bern', 'Bernoulli distribution', '\\Ber(p)'], ['Beta', 'Beta distribution', '\\Beta(\\alpha,\\beta)'],
      ['Poi', 'Poisson distribution', '\\Poi(\\lambda)'],
      ['KL', 'KL divergence', '\\KL(P\\Vert Q)'], ['kl', 'Binary KL divergence', '\\kl(p,q)'],
      ['→p', 'Convergence in probability', '\\xrightarrow{p}'],
      ['→d', 'Convergence in distribution', '\\xrightarrow{d}'],
      ['a.s.', 'Almost surely', '\\text{a.s.}'],
    ],
    'Bandits and ML': [
      ['R(T)', 'Regret', '\\Reg(T)=\\sum_{t=1}^{T}(\\mu^*-\\mu_{A_t})'],
      ['Nₐ(t)', 'Pull count', 'N_a(t)=\\sum_{s=1}^{t}\\one\\{A_s=a\\}'],
      ['μ̂', 'Empirical mean', '\\widehat\\mu_a(t)'], ['UCB', 'UCB index', '\\widehat\\mu_a(t)+\\sqrt{\\frac{2\\log t}{N_a(t)}}'],
      ['Aₜ', 'Action at time t', 'A_t'], ['Hₜ', 'History', 'H_t=(A_1,X_1,\\ldots,A_t,X_t)'],
      ['𝓕ₜ', 'Filtration', '\\calF_t'], ['Δₐ', 'Suboptimality gap', '\\Delta_a=\\mu^*-\\mu_a'],
      ['θ̂', 'Estimated parameter', '\\widehat\\theta_n'], ['ℒ', 'Loss', '\\mathcal{L}(\\theta)'],
      ['∇L', 'Gradient of loss', '\\nabla_\\theta\\mathcal{L}(\\theta)'],
      ['softmax', 'Softmax', '\\operatorname{softmax}(z)_i=\\frac{e^{z_i}}{\\sum_j e^{z_j}}'],
    ],
  };

  const elements = {
    auth: document.querySelector('#editor-auth'),
    authStatus: document.querySelector('#editor-auth-status'),
    connect: document.querySelector('#editor-connect'),
    compileCopy: document.querySelector('#editor-compile-copy'),
    compileItems: document.querySelector('#editor-compile-items'),
    compileLog: document.querySelector('#editor-compile-log'),
    compileMeta: document.querySelector('#editor-compile-meta'),
    compileSummary: document.querySelector('#editor-compile-summary'),
    content: document.querySelector('#editor-content'),
    disconnect: document.querySelector('#editor-disconnect'),
    draftBanner: document.querySelector('#editor-draft-banner'),
    draftDiscard: document.querySelector('#editor-draft-discard'),
    draftMessage: document.querySelector('#editor-draft-message'),
    draftRestore: document.querySelector('#editor-draft-restore'),
    filename: document.querySelector('#editor-filename'),
    files: document.querySelector('#editor-files'),
    formulaCategory: document.querySelector('#formula-category'),
    formulaModes: [...document.querySelectorAll('[data-formula-mode]')],
    formulaSymbols: document.querySelector('#formula-symbols'),
    image: document.querySelector('#editor-image'),
    imageFile: document.querySelector('#editor-image-file'),
    identity: document.querySelector('#editor-identity'),
    message: document.querySelector('#editor-message'),
    markdownActions: [...document.querySelectorAll('[data-md-action]')],
    modes: [...document.querySelectorAll('.editor-modes [data-mode]')],
    newNote: document.querySelector('#editor-new'),
    panes: document.querySelector('.editor-panes'),
    preview: document.querySelector('#editor-preview'),
    previewStatus: document.querySelector('#editor-preview-status'),
    publish: document.querySelector('#editor-publish'),
    remember: document.querySelector('#editor-remember'),
    status: document.querySelector('#editor-status'),
    token: document.querySelector('#editor-token'),
    workspace: document.querySelector('#editor-workspace'),
  };

  let token = localStorage.getItem(tokenKey) || sessionStorage.getItem(tokenKey) || '';
  let currentPath = '';
  let currentSha = '';
  let formulaMode = 'inline';
  let compileLogText = '';
  let currentCompileIssues = [];
  let markdownIssues = [];
  let pendingDraft = null;
  let previewTimer;
  let saveTimer;

  function setStatus(message, kind = '') {
    elements.status.textContent = message;
    elements.status.dataset.kind = kind;
  }

  function setAuthStatus(message, kind = '') {
    elements.authStatus.textContent = message;
    elements.authStatus.dataset.kind = kind;
  }

  function clearStoredToken() {
    localStorage.removeItem(tokenKey);
    sessionStorage.removeItem(tokenKey);
  }

  function storeToken() {
    clearStoredToken();
    const storage = elements.remember.checked ? localStorage : sessionStorage;
    storage.setItem(tokenKey, token);
  }

  function updateCompileLog(issues) {
    const unique = issues.filter((issue, index, all) => (
      all.findIndex((item) => item.stage === issue.stage && item.line === issue.line && item.message === issue.message) === index
    ));
    currentCompileIssues = unique;
    const timestamp = new Date().toLocaleTimeString('zh-CN', { hour12: false });
    elements.compileItems.replaceChildren();
    unique.forEach((issue) => {
      const item = document.createElement('li');
      const location = issue.line ? ` · 第 ${issue.line} 行` : '';
      item.textContent = `[${issue.stage}]${location} · ${issue.message}`;
      elements.compileItems.append(item);
    });
    elements.compileSummary.textContent = unique.length ? `${unique.length} 个问题` : '通过 · 0 个问题';
    elements.compileSummary.dataset.kind = unique.length ? 'error' : 'success';
    elements.compileMeta.textContent = `最后编译 ${timestamp} · Markdown / GFM / KaTeX`;
    compileLogText = unique.length
      ? unique.map((issue) => `[${issue.stage}]${issue.line ? ` line ${issue.line}` : ''}: ${issue.message}`).join('\n')
      : `[${timestamp}] Compilation passed with 0 issues.`;
    if (unique.length) elements.compileLog.open = true;
  }

  function apiHeaders() {
    return {
      Accept: 'application/vnd.github+json',
      Authorization: `Bearer ${token}`,
      'X-GitHub-Api-Version': '2022-11-28',
    };
  }

  async function githubRequest(url, options = {}) {
    const response = await fetch(url, {
      ...options,
      headers: { ...apiHeaders(), ...(options.headers || {}) },
    });
    if (!response.ok) {
      let message = `${response.status} ${response.statusText}`;
      try {
        const payload = await response.json();
        if (payload.message) message = payload.message;
      } catch (_) {}
      throw new Error(message);
    }
    return response.status === 204 ? null : response.json();
  }

  function api(path, options = {}) {
    return githubRequest(`${apiRoot}${path}`, options);
  }

  function decodeBase64(value) {
    const binary = atob(value.replace(/\s/g, ''));
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    return new TextDecoder().decode(bytes);
  }

  function encodeBase64(value) {
    const bytes = new TextEncoder().encode(value);
    let binary = '';
    const chunk = 0x8000;
    for (let index = 0; index < bytes.length; index += chunk) {
      binary += String.fromCharCode(...bytes.subarray(index, index + chunk));
    }
    return btoa(binary);
  }

  function prettyName(path) {
    return path.split('/').pop().replace(/\.md$/, '').split('-')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  }

  function readDrafts() {
    try {
      return JSON.parse(localStorage.getItem(draftsKey) || '{}');
    } catch (_) {
      return {};
    }
  }

  function draftPath() {
    const filename = elements.filename.value.trim().toLowerCase() || 'new-note.md';
    return currentPath || `content/notes/${filename}`;
  }

  function saveDraft() {
    if (!elements.content.value.trim()) return;
    const drafts = readDrafts();
    const path = draftPath();
    drafts[path] = {
      content: elements.content.value,
      filename: elements.filename.value,
      path: currentPath,
      sha: currentSha,
      updatedAt: new Date().toISOString(),
    };
    localStorage.setItem(draftsKey, JSON.stringify(drafts));
    setStatus('草稿已保存在本机', 'success');
  }

  function removeDraft(path = draftPath()) {
    const drafts = readDrafts();
    delete drafts[path];
    localStorage.setItem(draftsKey, JSON.stringify(drafts));
  }

  function hideDraftOffer() {
    pendingDraft = null;
    elements.draftBanner.hidden = true;
  }

  function offerDraft(path, remoteContent) {
    const draft = readDrafts()[path];
    if (!draft || draft.content === remoteContent) {
      if (draft?.content === remoteContent) removeDraft(path);
      hideDraftOffer();
      return;
    }
    pendingDraft = { ...draft, key: path };
    const updated = new Date(draft.updatedAt).toLocaleString('zh-CN', { hour12: false });
    elements.draftMessage.textContent = `发现 ${updated} 保存的本地草稿`;
    elements.draftBanner.hidden = false;
  }

  function restoreDraft() {
    if (!pendingDraft) return;
    currentPath = pendingDraft.path;
    currentSha = pendingDraft.sha;
    elements.filename.value = pendingDraft.filename;
    elements.filename.disabled = Boolean(currentPath);
    elements.content.value = pendingDraft.content;
    hideDraftOffer();
    renderPreview();
    setStatus('已恢复本地草稿', 'success');
  }

  function discardDraft() {
    if (!pendingDraft) return;
    removeDraft(pendingDraft.key);
    hideDraftOffer();
    setStatus('已放弃本地草稿');
  }

  function selectFile(path) {
    elements.files.querySelectorAll('button').forEach((button) => {
      button.classList.toggle('active', button.dataset.path === path);
    });
  }

  async function loadFiles(selectedPath = '') {
    const files = await api(`/contents/content/notes?ref=${encodeURIComponent(branch)}`);
    const markdownFiles = files
      .filter((file) => file.type === 'file' && file.name.endsWith('.md') && file.name !== '_index.md')
      .sort((left, right) => left.name.localeCompare(right.name));

    elements.files.replaceChildren();
    markdownFiles.forEach((file) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.dataset.path = file.path;
      button.textContent = prettyName(file.path);
      button.addEventListener('click', () => loadFile(file.path));
      elements.files.append(button);
    });
    if (selectedPath) selectFile(selectedPath);
    return markdownFiles;
  }

  async function loadFile(path) {
    if (saveTimer) {
      clearTimeout(saveTimer);
      saveDraft();
    }
    setStatus('Loading...');
    try {
      const file = await api(`/contents/${path}?ref=${encodeURIComponent(branch)}`);
      currentPath = path;
      currentSha = file.sha;
      elements.filename.value = path.split('/').pop();
      elements.filename.disabled = true;
      const remoteContent = decodeBase64(file.content);
      const editableContent = window.MarkdownPipeline.prepareForEdit(remoteContent);
      elements.content.value = editableContent;
      selectFile(path);
      renderPreview();
      offerDraft(path, editableContent);
      setStatus('Ready', 'success');
    } catch (error) {
      setStatus(error.message, 'error');
    }
  }

  function newNote() {
    if (saveTimer) {
      clearTimeout(saveTimer);
      saveDraft();
    }
    const today = new Date().toISOString().slice(0, 10);
    currentPath = '';
    currentSha = '';
    elements.filename.disabled = false;
    elements.filename.value = 'new-note.md';
    elements.content.value = `---
title: "New Note"
subtitle: ""
summary: ""
date: ${today}
lastmod: ${today}
weight: 90
tags: []
draft: false
ShowToc: false
hideMeta: true
---

Write the note here.
`;
    selectFile('');
    renderPreview();
    offerDraft(draftPath(), elements.content.value);
    elements.filename.focus();
    elements.filename.select();
    setStatus('New document', 'success');
  }

  function renderPreview() {
    markdownIssues = window.MarkdownPipeline.diagnose(elements.content.value);
    let rendered = '';
    try {
      rendered = window.MarkdownPipeline.render(elements.content.value);
    } catch (error) {
      markdownIssues.push({ stage: 'Markdown', line: 0, message: error.message });
      rendered = '<p>Preview compilation failed.</p>';
    }
    updateCompileLog(markdownIssues);
    const katexBase = `${window.location.origin}/vendor/katex`;
    const contentCss = `${window.location.origin}/css/markdown-content.css`;
    elements.preview.onload = () => {
      const mathErrors = [];
      try {
        window.renderMathInElement(elements.preview.contentDocument.body, {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true },
          ],
          throwOnError: false,
          strict: false,
          errorCallback: (message) => mathErrors.push(String(message)),
          macros: mathMacros,
        });
      } catch (error) {
        mathErrors.push(error.message);
      }
      updateCompileLog([
        ...markdownIssues,
        ...mathErrors.map((message) => ({ stage: 'KaTeX', line: 0, message })),
      ]);
      elements.previewStatus.textContent = mathErrors.length ? `${mathErrors.length} 个公式需要检查` : 'Markdown · GFM · KaTeX';
      elements.previewStatus.dataset.kind = mathErrors.length ? 'error' : 'success';
    };
    elements.preview.srcdoc = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="${katexBase}/katex.min.css">
<link rel="stylesheet" href="${contentCss}">
<style>
*{box-sizing:border-box}body{margin:0;padding:24px}a{color:#176d73}.markdown-content h1{font-size:34px}.markdown-content h2{font-size:28px}.markdown-content h3{font-size:22px}
</style></head><body>${rendered}
</body></html>`.replace('<body>', '<body class="markdown-content">');
  }

  function insertAtCursor(value) {
    const start = elements.content.selectionStart;
    const end = elements.content.selectionEnd;
    elements.content.setRangeText(value, start, end, 'end');
    elements.content.focus();
    renderPreview();
  }

  function transformSelection(action) {
    const start = elements.content.selectionStart;
    const end = elements.content.selectionEnd;
    const selected = elements.content.value.slice(start, end);
    const linePrefix = (prefix) => (selected || 'Text').split('\n').map((line, index) => `${typeof prefix === 'function' ? prefix(index) : prefix}${line}`).join('\n');
    const actions = {
      h1: () => linePrefix('# '),
      h2: () => linePrefix('## '),
      h3: () => linePrefix('### '),
      h4: () => linePrefix('#### '),
      bold: () => `**${selected || 'bold text'}**`,
      italic: () => `_${selected || 'italic text'}_`,
      strike: () => `~~${selected || 'strikethrough'}~~`,
      quote: () => linePrefix('> '),
      link: () => `[${selected || 'link text'}](https://example.com)`,
      'inline-code': () => `\`${selected || 'code'}\``,
      'code-block': () => `\n\`\`\`text\n${selected || 'code'}\n\`\`\`\n`,
      'unordered-list': () => linePrefix('- '),
      'ordered-list': () => linePrefix((index) => `${index + 1}. `),
      'task-list': () => linePrefix('- [ ] '),
      table: () => `\n| Column 1 | Column 2 | Column 3 |\n| --- | --- | --- |\n| Value | Value | Value |\n`,
      rule: () => '\n---\n',
      footnote: () => `${selected || 'Statement'}[^1]\n\n[^1]: Footnote text.`,
    };
    const replacement = actions[action]?.();
    if (replacement === undefined) return;
    elements.content.setRangeText(replacement, start, end, 'end');
    elements.content.focus();
    renderPreview();
  }

  function insertFormula(template) {
    const selected = elements.content.value.slice(elements.content.selectionStart, elements.content.selectionEnd) || 'x';
    const formula = template.replaceAll('{{selection}}', selected);
    const wrapped = formulaMode === 'display' ? `\n$$\n${formula}\n$$\n` : `$${formula}$`;
    insertAtCursor(wrapped);
  }

  function renderFormulaPalette(category) {
    elements.formulaSymbols.replaceChildren();
    formulaCatalog[category].forEach(([label, title, template]) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = label;
      button.title = title;
      button.setAttribute('aria-label', title);
      button.addEventListener('click', () => insertFormula(template));
      elements.formulaSymbols.append(button);
    });
  }

  function initializeFormulaPalette() {
    Object.keys(formulaCatalog).forEach((category) => {
      const option = document.createElement('option');
      option.value = category;
      option.textContent = category;
      elements.formulaCategory.append(option);
    });
    renderFormulaPalette(Object.keys(formulaCatalog)[0]);
  }

  function fileToBase64(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result).split(',')[1]);
      reader.onerror = reject;
      reader.readAsDataURL(file);
    });
  }

  async function uploadImage(file, altText = 'Research figure') {
    if (!file.type.startsWith('image/')) throw new Error('Choose a PNG, JPEG, WebP, or GIF image.');
    if (file.size > 8 * 1024 * 1024) throw new Error('Image must be smaller than 8 MB.');
    const extension = (file.name.split('.').pop() || 'png').toLowerCase().replace(/[^a-z0-9]/g, '');
    const noteSlug = (elements.filename.value || 'note').replace(/\.md$/i, '').replace(/[^a-z0-9-]/gi, '-').toLowerCase();
    const assetName = `${noteSlug}-${Date.now()}.${extension}`;
    const assetPath = `static/images/notes/uploads/${assetName}`;
    setStatus('Uploading image...');
    const result = await api(`/contents/${assetPath}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: `Upload figure for ${noteSlug}`,
        content: await fileToBase64(file),
        branch,
      }),
    });
    insertAtCursor(`\n![${altText}](/images/notes/uploads/${assetName})\n`);
    setStatus(`Image uploaded | ${result.commit.sha.slice(0, 7)}`, 'success');
  }

  async function publish() {
    elements.publish.disabled = true;
    try {
      const filename = elements.filename.value.trim().toLowerCase();
      if (!/^[a-z0-9][a-z0-9-]*\.md$/.test(filename)) {
        throw new Error('文件名请使用小写字母、数字和连字符，例如 my-note.md。');
      }
      const path = currentPath || `content/notes/${filename}`;
      const today = new Date().toISOString().slice(0, 10);
      let editableContent = elements.content.value;
      if (/^lastmod:/m.test(editableContent)) editableContent = editableContent.replace(/^lastmod:.*$/m, `lastmod: ${today}`);
      const content = window.MarkdownPipeline.prepareForPublish(editableContent);
      const publishIssues = window.MarkdownPipeline.diagnose(content);
      const katexIssues = currentCompileIssues.filter((issue) => issue.stage === 'KaTeX');
      if (publishIssues.length || katexIssues.length) {
        updateCompileLog([...publishIssues, ...katexIssues]);
        throw new Error('请先处理编译日志中的问题。');
      }
      const payload = {
        message: elements.message.value.trim() || `Publish ${filename}`,
        content: encodeBase64(content),
        branch,
      };
      if (currentSha) payload.sha = currentSha;

      setStatus('Publishing...');
      const result = await api(`/contents/${path}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      currentPath = result.content.path;
      currentSha = result.content.sha;
      elements.filename.disabled = true;
      elements.content.value = editableContent;
      removeDraft(path);
      hideDraftOffer();
      await loadFiles(currentPath);
      setStatus(`Published | ${result.commit.sha.slice(0, 7)}`, 'success');
    } catch (error) {
      setStatus(error.message, 'error');
      if (!currentCompileIssues.length && error.message !== '请先处理编译日志中的问题。') {
        updateCompileLog([{ stage: 'Publish', line: 0, message: error.message }]);
      }
    } finally {
      elements.publish.disabled = false;
    }
  }

  async function connect() {
    token = elements.token.value.trim() || token;
    if (!token) {
      setAuthStatus('请输入 GitHub token。', 'error');
      return;
    }
    elements.connect.disabled = true;
    setAuthStatus('正在验证身份与仓库权限...');
    try {
      const [user, repository] = await Promise.all([
        githubRequest('https://api.github.com/user'),
        api(''),
      ]);
      if (user.login.toLowerCase() !== owner.toLowerCase()) {
        throw new Error(`当前账号 ${user.login} 没有编辑权限，仅 ${owner} 可以进入。`);
      }
      if (!repository.permissions?.push) {
        throw new Error('Token 缺少此仓库 Contents 的写入权限。');
      }
      storeToken();
      setAuthStatus(`已验证 ${user.login}`, 'success');
      elements.identity.textContent = `${user.login} · ${repository.full_name}`;
      elements.auth.hidden = true;
      elements.workspace.hidden = false;
      const requested = params.get('path');
      const normalizedRequested = requested ? requested.replaceAll('\\', '/').replace(/^content\//, '') : '';
      const requestedPath = normalizedRequested ? `content/${normalizedRequested}` : '';
      const files = await loadFiles(requestedPath);
      if (params.has('new')) newNote();
      else if (requestedPath) await loadFile(requestedPath);
      else if (files.length) await loadFile(files[0].path);
    } catch (error) {
      setAuthStatus(error.message, 'error');
      clearStoredToken();
      token = '';
    } finally {
      elements.connect.disabled = false;
    }
  }

  initializeFormulaPalette();
  elements.markdownActions.forEach((button) => button.addEventListener('click', () => transformSelection(button.dataset.mdAction)));
  elements.formulaCategory.addEventListener('change', () => renderFormulaPalette(elements.formulaCategory.value));
  elements.formulaModes.forEach((button) => button.addEventListener('click', () => {
    formulaMode = button.dataset.formulaMode;
    elements.formulaModes.forEach((item) => item.classList.toggle('active', item === button));
  }));
  elements.image.addEventListener('click', () => elements.imageFile.click());
  elements.imageFile.addEventListener('change', async () => {
    const file = elements.imageFile.files[0];
    if (!file) return;
    try { await uploadImage(file, file.name.replace(/\.[^.]+$/, '')); }
    catch (error) { setStatus(error.message, 'error'); }
    finally { elements.imageFile.value = ''; }
  });
  elements.connect.addEventListener('click', connect);
  elements.token.addEventListener('keydown', (event) => { if (event.key === 'Enter') connect(); });
  elements.disconnect.addEventListener('click', () => { clearStoredToken(); window.location.reload(); });
  elements.compileCopy.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(compileLogText);
      elements.compileCopy.textContent = '已复制';
      setTimeout(() => { elements.compileCopy.textContent = '复制日志'; }, 1200);
    } catch (_) {
      setStatus('浏览器未允许复制日志', 'error');
    }
  });
  elements.draftRestore.addEventListener('click', restoreDraft);
  elements.draftDiscard.addEventListener('click', discardDraft);
  elements.newNote.addEventListener('click', newNote);
  elements.publish.addEventListener('click', publish);
  elements.content.addEventListener('input', () => {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(renderPreview, 180);
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveDraft, 650);
  });
  elements.filename.addEventListener('input', () => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(saveDraft, 650);
  });
  elements.modes.forEach((button) => button.addEventListener('click', () => {
    elements.modes.forEach((item) => item.classList.toggle('active', item === button));
    elements.panes.dataset.mode = button.dataset.mode;
  }));

  if (token) {
    elements.remember.checked = Boolean(localStorage.getItem(tokenKey));
    elements.token.value = token;
    setAuthStatus('正在恢复此设备的编辑身份...');
    connect();
  }
})();
