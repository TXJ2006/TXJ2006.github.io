(() => {
  const main = document.querySelector('.private-note-main');
  if (!main) return;

  const elements = {
    article: document.querySelector('#private-note-article'),
    auth: document.querySelector('#private-note-auth'),
    content: document.querySelector('#private-note-content'),
    date: document.querySelector('#private-note-date'),
    edit: document.querySelector('#private-note-edit'),
    gate: document.querySelector('#private-note-gate'),
    readingTime: document.querySelector('#private-note-reading-time'),
    remember: document.querySelector('#private-note-remember'),
    share: document.querySelector('#private-note-share'),
    status: document.querySelector('#private-note-status'),
    subtitle: document.querySelector('#private-note-subtitle'),
    tags: document.querySelector('#private-note-tags'),
    title: document.querySelector('#private-note-title'),
    token: document.querySelector('#private-note-token'),
    wordCount: document.querySelector('#private-note-word-count'),
  };
  const privateRepo = main.dataset.privateRepo;
  const owner = main.dataset.owner;
  const branch = main.dataset.branch || 'main';
  const siteTitle = main.dataset.siteTitle || '';
  const apiRoot = `https://api.github.com/repos/${privateRepo}`;
  const tokenKey = 'txj_notes_editor_token';
  const notePath = new URLSearchParams(location.search).get('path') || '';
  const objectUrls = new Set();

  function requestHeaders(token) {
    return {
      Accept: 'application/vnd.github+json',
      Authorization: `Bearer ${token}`,
      'X-GitHub-Api-Version': '2022-11-28',
    };
  }

  async function githubJson(url, token) {
    const response = await fetch(url, {
      headers: requestHeaders(token),
      cache: 'no-store',
    });
    if (!response.ok) {
      const error = new Error(`GitHub request failed (${response.status})`);
      error.status = response.status;
      throw error;
    }
    return response.json();
  }

  function decodeBase64(value) {
    const binary = atob(value.replace(/\s/g, ''));
    const bytes = Uint8Array.from(binary, (character) => character.charCodeAt(0));
    return new TextDecoder().decode(bytes);
  }

  function base64ToBytes(value) {
    const binary = atob(value.replace(/\s/g, ''));
    return Uint8Array.from(binary, (character) => character.charCodeAt(0));
  }

  function encodeRepositoryPath(path) {
    return path.split('/').map((part) => encodeURIComponent(part)).join('/');
  }

  function validNotePath(path) {
    return path.startsWith('notes/')
      && path.endsWith('.md')
      && !path.split('/').includes('..')
      && !/[\0\r\n]/.test(path);
  }

  function unquote(value) {
    const text = String(value || '').trim();
    if (text.length < 2) return text;
    if (text.startsWith("'") && text.endsWith("'")) return text.slice(1, -1).replaceAll("''", "'");
    if (text.startsWith('"') && text.endsWith('"')) {
      try {
        return JSON.parse(text);
      } catch (_) {
        return text.slice(1, -1);
      }
    }
    return text;
  }

  function parseArray(value) {
    const text = value.trim();
    if (!text.startsWith('[') || !text.endsWith(']')) return text ? [unquote(text)] : [];
    try {
      const parsed = JSON.parse(text);
      return Array.isArray(parsed) ? parsed.map(String) : [];
    } catch (_) {
      return text.slice(1, -1).split(',').map((item) => unquote(item)).filter(Boolean);
    }
  }

  function parseFrontMatter(markdown) {
    const match = /^---\s*\n([\s\S]*?)\n---(?:\s*\n|$)/.exec(markdown);
    if (!match) return {};
    const metadata = {};
    match[1].split('\n').forEach((line) => {
      const field = /^([A-Za-z][\w-]*):\s*(.*)$/.exec(line);
      if (!field) return;
      metadata[field[1]] = field[1] === 'tags' ? parseArray(field[2]) : unquote(field[2]);
    });
    return metadata;
  }

  function sanitizeMarkdownHtml(html) {
    const template = document.createElement('template');
    template.innerHTML = html;
    template.content.querySelectorAll('base, embed, form, iframe, link, meta, object, script, style')
      .forEach((element) => element.remove());
    template.content.querySelectorAll('*').forEach((element) => {
      [...element.attributes].forEach((attribute) => {
        const name = attribute.name.toLowerCase();
        const value = attribute.value.trim();
        if (name.startsWith('on')
            || ((name === 'href' || name === 'src' || name === 'xlink:href')
              && /^(?:javascript|vbscript|data:text\/html)/i.test(value))) {
          element.removeAttribute(attribute.name);
          return;
        }
        if (name === 'style') {
          const width = element.tagName === 'IMG'
            ? /(?:^|;)\s*width\s*:\s*(\d{1,3})%\s*(?:;|$)/i.exec(value)
            : null;
          if (width) {
            const normalized = Math.min(100, Math.max(10, Number.parseInt(width[1], 10)));
            element.setAttribute('style', `width: ${normalized}%`);
          } else {
            element.removeAttribute('style');
          }
        }
      });
      if (element.tagName === 'A') element.setAttribute('rel', 'noopener noreferrer');
    });
    return template.innerHTML;
  }

  function mimeTypeFromPath(path) {
    const extension = path.split('.').pop()?.toLowerCase();
    return ({
      gif: 'image/gif',
      jpeg: 'image/jpeg',
      jpg: 'image/jpeg',
      png: 'image/png',
      webp: 'image/webp',
    })[extension] || 'application/octet-stream';
  }

  async function hydratePrivateImages(token) {
    const images = [...elements.content.querySelectorAll('img[src^="private-image://"]')];
    await Promise.all(images.map(async (image) => {
      const source = image.getAttribute('src');
      const path = source.slice('private-image://'.length).replace(/^\/+/, '');
      if (!path || path.split('/').includes('..')) {
        image.removeAttribute('src');
        return;
      }
      try {
        const file = await githubJson(
          `${apiRoot}/contents/${encodeRepositoryPath(path)}?ref=${encodeURIComponent(branch)}`,
          token,
        );
        const blob = new Blob([base64ToBytes(file.content)], { type: mimeTypeFromPath(path) });
        const objectUrl = URL.createObjectURL(blob);
        objectUrls.add(objectUrl);
        image.src = objectUrl;
      } catch (_) {
        image.removeAttribute('src');
        image.alt = `${image.alt || 'Private image'}（读取失败）`;
      }
    }));
  }

  function wordCount(markdown) {
    const source = window.MarkdownPipeline.stripFrontMatter(markdown)
      .replace(/```[\s\S]*?```|~~~[\s\S]*?~~~/g, ' ')
      .replace(/<[^>]+>|[#>*_`~\[\](){}$|\\]/g, ' ');
    const latin = source.match(/[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*/g)?.length || 0;
    const cjk = source.match(/[\u3400-\u9fff]/g)?.length || 0;
    return latin + cjk;
  }

  function formattedDate(value) {
    const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(value || '');
    if (!match) return '';
    const date = new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
    return new Intl.DateTimeFormat('en-US', {
      day: 'numeric',
      month: 'long',
      year: 'numeric',
    }).format(date);
  }

  async function renderMath() {
    if (document.readyState === 'loading') {
      await new Promise((resolve) => document.addEventListener('DOMContentLoaded', resolve, { once: true }));
    }
    if (typeof window.renderMathInElement === 'function') {
      window.renderMathInElement(elements.article, window.siteMathOptions || {
        delimiters: [
          { left: '$$', right: '$$', display: true },
          { left: '$', right: '$', display: false },
          { left: '\\(', right: '\\)', display: false },
          { left: '\\[', right: '\\]', display: true },
        ],
        throwOnError: false,
      });
    }
  }

  async function shareForNote(token) {
    const filename = notePath.split('/').pop();
    try {
      const file = await githubJson(
        `${apiRoot}/contents/shares/${encodeURIComponent(filename)}.json?ref=${encodeURIComponent(branch)}`,
        token,
      );
      return JSON.parse(decodeBase64(file.content));
    } catch (error) {
      if (error.status === 404) return null;
      throw error;
    }
  }

  function populateHeader(metadata, markdown, share) {
    const title = metadata.title || notePath.split('/').pop().replace(/\.md$/i, '');
    const totalWords = wordCount(markdown);
    const date = formattedDate(metadata.date);
    elements.title.textContent = title;
    elements.subtitle.textContent = metadata.subtitle || '';
    elements.subtitle.hidden = !metadata.subtitle;
    elements.date.textContent = date ? `Published ${date}` : '';
    elements.date.hidden = !date;
    elements.readingTime.textContent = `${Math.max(1, Math.ceil(totalWords / 200))} min read`;
    elements.wordCount.textContent = `${totalWords} words`;
    elements.tags.replaceChildren();
    (metadata.tags || []).forEach((tag) => {
      const item = document.createElement('li');
      item.textContent = tag;
      elements.tags.append(item);
    });
    elements.tags.hidden = elements.tags.children.length === 0;
    elements.edit.href = `/admin/?repository=private&path=${encodeURIComponent(notePath)}`;
    if (share?.url) {
      elements.share.href = share.url;
      elements.share.hidden = false;
    } else {
      elements.share.hidden = true;
    }
    document.title = siteTitle ? `${title} | ${siteTitle}` : title;
  }

  function rememberToken(token) {
    localStorage.removeItem(tokenKey);
    sessionStorage.removeItem(tokenKey);
    (elements.remember.checked ? localStorage : sessionStorage).setItem(tokenKey, token);
  }

  function showGate(message, error = false) {
    elements.article.hidden = true;
    elements.gate.hidden = false;
    elements.status.textContent = message;
    elements.status.dataset.error = String(error);
  }

  async function openPrivateNote(token, persist = false) {
    if (!validNotePath(notePath)) {
      showGate('The requested private note is unavailable.', true);
      return;
    }
    showGate('Verifying owner access…');
    elements.auth.querySelector('button').disabled = true;
    try {
      const [user, repository] = await Promise.all([
        githubJson('https://api.github.com/user', token),
        githubJson(apiRoot, token),
      ]);
      if (user.login?.toLowerCase() !== owner.toLowerCase()
          || !repository.private
          || !repository.permissions?.push) {
        throw new Error('Owner verification failed.');
      }
      if (persist) rememberToken(token);
      const [documentFile, share] = await Promise.all([
        githubJson(
          `${apiRoot}/contents/${encodeRepositoryPath(notePath)}?ref=${encodeURIComponent(branch)}`,
          token,
        ),
        shareForNote(token),
      ]);
      const markdown = decodeBase64(documentFile.content);
      const metadata = parseFrontMatter(markdown);
      populateHeader(metadata, markdown, share);
      elements.content.innerHTML = sanitizeMarkdownHtml(window.MarkdownPipeline.render(markdown));
      await hydratePrivateImages(token);
      elements.gate.hidden = true;
      elements.article.hidden = false;
      await renderMath();
      window.scrollTo(0, 0);
    } catch (error) {
      if (error.status === 401 || error.status === 403) {
        localStorage.removeItem(tokenKey);
        sessionStorage.removeItem(tokenKey);
      }
      showGate('Owner verification failed. Enter a valid owner token to continue.', true);
      elements.token.focus();
    } finally {
      elements.auth.querySelector('button').disabled = false;
    }
  }

  elements.auth.addEventListener('submit', (event) => {
    event.preventDefault();
    const token = elements.token.value.trim();
    if (!token) {
      showGate('Enter the owner token to continue.', true);
      elements.token.focus();
      return;
    }
    openPrivateNote(token, true);
  });

  window.addEventListener('pagehide', () => {
    objectUrls.forEach((url) => URL.revokeObjectURL(url));
    objectUrls.clear();
    elements.content.replaceChildren();
  });

  const savedToken = localStorage.getItem(tokenKey) || sessionStorage.getItem(tokenKey);
  if (savedToken) openPrivateNote(savedToken);
  else showGate('Enter the owner token to continue.');
})();
