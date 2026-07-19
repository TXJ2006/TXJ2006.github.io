(() => {
  const page = document.querySelector('.editor-page');
  if (!page) return;

  const repo = page.dataset.repo;
  const branch = page.dataset.branch;
  const apiRoot = `https://api.github.com/repos/${repo}`;
  const tokenKey = 'txj_notes_editor_token';
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
    content: document.querySelector('#editor-content'),
    disconnect: document.querySelector('#editor-disconnect'),
    filename: document.querySelector('#editor-filename'),
    files: document.querySelector('#editor-files'),
    formulaCategory: document.querySelector('#formula-category'),
    formulaModes: [...document.querySelectorAll('[data-formula-mode]')],
    formulaSymbols: document.querySelector('#formula-symbols'),
    image: document.querySelector('#editor-image'),
    imageFile: document.querySelector('#editor-image-file'),
    message: document.querySelector('#editor-message'),
    markdownActions: [...document.querySelectorAll('[data-md-action]')],
    modes: [...document.querySelectorAll('.editor-modes [data-mode]')],
    newNote: document.querySelector('#editor-new'),
    panes: document.querySelector('.editor-panes'),
    preview: document.querySelector('#editor-preview'),
    publish: document.querySelector('#editor-publish'),
    status: document.querySelector('#editor-status'),
    token: document.querySelector('#editor-token'),
    workspace: document.querySelector('#editor-workspace'),
  };

  let token = sessionStorage.getItem(tokenKey) || '';
  let currentPath = '';
  let currentSha = '';
  let formulaMode = 'inline';
  let previewTimer;

  function setStatus(message, kind = '') {
    elements.status.textContent = message;
    elements.status.dataset.kind = kind;
  }

  function setAuthStatus(message, kind = '') {
    elements.authStatus.textContent = message;
    elements.authStatus.dataset.kind = kind;
  }

  function apiHeaders() {
    return {
      Accept: 'application/vnd.github+json',
      Authorization: `Bearer ${token}`,
      'X-GitHub-Api-Version': '2022-11-28',
    };
  }

  async function api(path, options = {}) {
    const response = await fetch(`${apiRoot}${path}`, {
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
    setStatus('Loading...');
    try {
      const file = await api(`/contents/${path}?ref=${encodeURIComponent(branch)}`);
      currentPath = path;
      currentSha = file.sha;
      elements.filename.value = path.split('/').pop();
      elements.filename.disabled = true;
      elements.content.value = decodeBase64(file.content);
      selectFile(path);
      renderPreview();
      setStatus('Ready', 'success');
    } catch (error) {
      setStatus(error.message, 'error');
    }
  }

  function newNote() {
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
    elements.filename.focus();
    elements.filename.select();
    setStatus('New document', 'success');
  }

  function renderPreview() {
    const markdown = elements.content.value.replace(/^---\s*[\s\S]*?\s*---\s*/, '');
    const rendered = window.marked.parse(markdown, { gfm: true, breaks: false });
    const macros = JSON.stringify(mathMacros).replace(/</g, '\\u003c');
    const katexBase = `${window.location.origin}/vendor/katex`;
    elements.preview.srcdoc = `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<style>
:root{--ink:#1d2528;--muted:#5e696c;--line:#d9dfdd;--soft:#f5f7f6;--accent:#176d73}
*{box-sizing:border-box}body{margin:0;padding:24px;color:var(--ink);font:16px/1.75 Arial,sans-serif}
h1,h2,h3,h4{font-family:"Times New Roman",serif;line-height:1.15}h1{font-size:34px}h2{margin-top:38px;padding-top:12px;border-top:1px solid var(--line);font-size:28px}h3{font-size:22px}
a{color:var(--accent)}blockquote{margin:22px 0;padding:13px 18px;border-left:3px solid #a54e39;background:var(--soft)}
pre{overflow:auto;padding:16px;background:#182124;color:#eef4f3;border-radius:4px}code{font-family:Consolas,monospace}img{display:block;max-width:100%;margin:26px auto}table{display:block;overflow:auto;border-collapse:collapse}th,td{padding:8px 10px;border:1px solid var(--line)}.katex-display{overflow-x:auto;overflow-y:hidden;padding:4px 0}
</style></head><body>${rendered}
<script defer src="${katexBase}/katex.min.js"><\/script>
<script defer src="${katexBase}/contrib/auto-render.min.js"><\/script>
<script>window.addEventListener('load',()=>renderMathInElement(document.body,{delimiters:[{left:'$$',right:'$$',display:true},{left:'$',right:'$',display:false},{left:'\\\\(',right:'\\\\)',display:false},{left:'\\\\[',right:'\\\\]',display:true}],throwOnError:false,macros:${macros}}));<\/script>
</body></html>`;
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
    const filename = elements.filename.value.trim().toLowerCase();
    if (!/^[a-z0-9][a-z0-9-]*\.md$/.test(filename)) {
      setStatus('Use a lowercase filename such as my-note.md', 'error');
      return;
    }
    const path = currentPath || `content/notes/${filename}`;
    const today = new Date().toISOString().slice(0, 10);
    let content = elements.content.value;
    if (/^lastmod:/m.test(content)) content = content.replace(/^lastmod:.*$/m, `lastmod: ${today}`);
    const payload = {
      message: elements.message.value.trim() || `Publish ${filename}`,
      content: encodeBase64(content),
      branch,
    };
    if (currentSha) payload.sha = currentSha;

    elements.publish.disabled = true;
    setStatus('Publishing...');
    try {
      const result = await api(`/contents/${path}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      currentPath = result.content.path;
      currentSha = result.content.sha;
      elements.filename.disabled = true;
      elements.content.value = content;
      await loadFiles(currentPath);
      setStatus(`Published | ${result.commit.sha.slice(0, 7)}`, 'success');
    } catch (error) {
      setStatus(error.message, 'error');
    } finally {
      elements.publish.disabled = false;
    }
  }

  async function connect() {
    token = elements.token.value.trim() || token;
    if (!token) {
      setAuthStatus('Enter a token to connect.', 'error');
      return;
    }
    elements.connect.disabled = true;
    setAuthStatus('Connecting...');
    try {
      await api('');
      sessionStorage.setItem(tokenKey, token);
      setAuthStatus('Connected', 'success');
      elements.auth.hidden = true;
      elements.workspace.hidden = false;
      const requested = params.get('path');
      const requestedPath = requested ? `content/${requested.replace(/^content\//, '')}` : '';
      const files = await loadFiles(requestedPath);
      if (params.has('new')) newNote();
      else if (requestedPath) await loadFile(requestedPath);
      else if (files.length) await loadFile(files[0].path);
    } catch (error) {
      setAuthStatus(error.message, 'error');
      sessionStorage.removeItem(tokenKey);
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
  elements.disconnect.addEventListener('click', () => { sessionStorage.removeItem(tokenKey); window.location.reload(); });
  elements.newNote.addEventListener('click', newNote);
  elements.publish.addEventListener('click', publish);
  elements.content.addEventListener('input', () => {
    clearTimeout(previewTimer);
    previewTimer = setTimeout(renderPreview, 180);
  });
  elements.modes.forEach((button) => button.addEventListener('click', () => {
    elements.modes.forEach((item) => item.classList.toggle('active', item === button));
    elements.panes.dataset.mode = button.dataset.mode;
  }));

  if (token) {
    elements.token.value = token;
    connect();
  }
})();
