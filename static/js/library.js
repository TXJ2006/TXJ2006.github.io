(() => {
  const page = document.querySelector('.notes-page');
  const folderList = document.querySelector('.library-folders');
  const articleList = document.querySelector('.notes-list');
  const label = document.querySelector('#library-result-label');
  const count = document.querySelector('#library-result-count');
  const reader = {
    dialog: document.querySelector('#private-reader-dialog'),
    title: document.querySelector('#private-reader-title'),
    status: document.querySelector('#private-reader-status'),
    content: document.querySelector('#private-reader-content'),
    close: document.querySelector('#private-reader-close'),
  };
  if (!page || !folderList || !articleList || !label || !count) return;

  const privateRepo = page.dataset.privateRepo;
  const owner = page.dataset.owner;
  const branch = page.dataset.branch || 'main';
  const tokenKey = 'txj_notes_editor_token';
  let cards = [];
  let articles = [];
  let selectedFolderId = 'all';
  let ownerToken = '';
  let privateApiRoot = '';
  let privateState = {};
  let privateFilesByPath = new Map();
  let activePrivatePath = '';
  let readerRequest = 0;
  const readerObjectUrls = new Set();

  function refreshCollections() {
    cards = [...folderList.querySelectorAll('[data-library-filter]')];
    articles = [...articleList.querySelectorAll('[data-library-folder]')];
  }

  function bindFolderCard(card) {
    if (card.dataset.libraryBound === 'true') return;
    card.dataset.libraryBound = 'true';
    card.addEventListener('click', () => selectFolder(card.dataset.libraryFilter));
  }

  function selectFolder(folderId, updateHistory = true) {
    refreshCollections();
    const selected = cards.find((card) => card.dataset.libraryFilter === folderId) || cards[0];
    if (!selected) return;
    selectedFolderId = selected.dataset.libraryFilter;
    cards.forEach((card) => {
      const active = card === selected;
      card.classList.toggle('is-active', active);
      card.setAttribute('aria-pressed', String(active));
    });
    let visible = 0;
    articles.forEach((article) => {
      const show = selected.dataset.libraryFilter === 'all' || article.dataset.libraryFolder === selected.dataset.libraryFilter;
      article.hidden = !show;
      if (show) {
        visible += 1;
        const number = article.querySelector('.note-number');
        if (number) number.textContent = String(visible).padStart(2, '0');
      }
    });
    label.textContent = selected.querySelector('strong')?.textContent || '全部文章';
    count.textContent = String(visible);
    if (updateHistory) {
      const hash = selectedFolderId === 'all' ? '' : `#folder=${encodeURIComponent(selectedFolderId)}`;
      history.replaceState(null, '', `${location.pathname}${location.search}${hash}`);
    }
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

  function prettyName(path) {
    return path.split('/').pop().replace(/\.md$/i, '').split('-')
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');
  }

  function requestHeaders(token) {
    return {
      Accept: 'application/vnd.github+json',
      Authorization: `Bearer ${token}`,
      'X-GitHub-Api-Version': '2022-11-28',
    };
  }

  async function githubJson(url, token) {
    const response = await fetch(url, { headers: requestHeaders(token), cache: 'no-store' });
    if (!response.ok) throw new Error(`GitHub request failed (${response.status})`);
    return response.json();
  }

  function encodeRepositoryPath(path) {
    return path.split('/').map((part) => encodeURIComponent(part)).join('/');
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

  function markdownTitle(markdown, fallback) {
    const frontMatter = /^---\s*\n([\s\S]*?)\n---(?:\s*\n|$)/.exec(markdown);
    const match = frontMatter?.[1].match(/^title:\s*(.+?)\s*$/m);
    if (!match) return fallback;
    const value = match[1].trim();
    if ((value.startsWith('"') && value.endsWith('"'))
        || (value.startsWith("'") && value.endsWith("'"))) {
      return value.slice(1, -1);
    }
    return value;
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

  function privateReaderUrl(path = '') {
    const url = new URL(window.location.href);
    if (path) url.searchParams.set('private', path);
    else url.searchParams.delete('private');
    return url;
  }

  function releaseReaderContent() {
    readerRequest += 1;
    readerObjectUrls.forEach((url) => URL.revokeObjectURL(url));
    readerObjectUrls.clear();
    if (reader.content) {
      reader.content.replaceChildren();
      reader.content.removeAttribute('aria-busy');
    }
    if (reader.status) reader.status.textContent = '';
    activePrivatePath = '';
  }

  function dismissPrivateReader(updateHistory = true) {
    const previousPath = activePrivatePath;
    if (reader.dialog?.open) reader.dialog.close();
    releaseReaderContent();
    if (updateHistory && (previousPath || new URLSearchParams(location.search).has('private'))) {
      history.pushState(null, '', privateReaderUrl());
    }
  }

  async function hydratePrivateReaderImages(token, requestId) {
    const images = [...reader.content.querySelectorAll('img[src^="private-image://"]')];
    await Promise.all(images.map(async (image) => {
      const source = image.getAttribute('src');
      const path = source.slice('private-image://'.length).replace(/^\/+/, '');
      if (!path || path.split('/').includes('..')) {
        image.removeAttribute('src');
        return;
      }
      try {
        const file = await githubJson(
          `${privateApiRoot}/contents/${encodeRepositoryPath(path)}?ref=${encodeURIComponent(branch)}`,
          token,
        );
        if (requestId !== readerRequest) return;
        const blob = new Blob([base64ToBytes(file.content)], { type: mimeTypeFromPath(path) });
        const objectUrl = URL.createObjectURL(blob);
        readerObjectUrls.add(objectUrl);
        image.src = objectUrl;
      } catch (_) {
        image.removeAttribute('src');
        image.alt = `${image.alt || 'Private image'}（读取失败）`;
      }
    }));
  }

  async function openPrivateNote(file, state, updateHistory = true) {
    if (!reader.dialog || !reader.content || !ownerToken || !privateApiRoot) return;
    releaseReaderContent();
    const requestId = readerRequest;
    const key = `private:${file.path}`;
    const fallbackTitle = state.labels?.[key] || prettyName(file.path);
    activePrivatePath = file.path;
    reader.title.textContent = fallbackTitle;
    reader.status.textContent = '正在验证并读取私密笔记…';
    reader.content.setAttribute('aria-busy', 'true');
    if (!reader.dialog.open) reader.dialog.showModal();
    if (updateHistory && new URLSearchParams(location.search).get('private') !== file.path) {
      history.pushState(null, '', privateReaderUrl(file.path));
    }

    try {
      const documentFile = await githubJson(
        `${privateApiRoot}/contents/${encodeRepositoryPath(file.path)}?ref=${encodeURIComponent(branch)}`,
        ownerToken,
      );
      if (requestId !== readerRequest) return;
      const markdown = decodeBase64(documentFile.content);
      reader.title.textContent = markdownTitle(markdown, fallbackTitle);
      const rendered = window.MarkdownPipeline.render(markdown);
      reader.content.innerHTML = sanitizeMarkdownHtml(rendered);
      await hydratePrivateReaderImages(ownerToken, requestId);
      if (requestId !== readerRequest) return;
      if (typeof window.renderMathInElement === 'function') {
        window.renderMathInElement(reader.content, window.siteMathOptions || {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true },
          ],
          throwOnError: false,
        });
      }
      reader.content.removeAttribute('aria-busy');
      reader.status.textContent = '私密预览 · 正文未写入公开页面';
      reader.content.focus({ preventScroll: true });
    } catch (_) {
      if (requestId !== readerRequest) return;
      reader.content.removeAttribute('aria-busy');
      reader.content.innerHTML = '<p class="private-reader-error">暂时无法读取这篇私密笔记。请检查当前设备的所有者令牌后重试。</p>';
      reader.status.textContent = '读取失败';
    }
  }

  function createFolderCard(folder, privateCount) {
    const card = document.createElement('button');
    card.className = `library-folder-card folder-color-${folder.color ?? 0}`;
    card.type = 'button';
    card.dataset.libraryFilter = folder.id;
    card.setAttribute('aria-pressed', 'false');

    const glyph = document.createElement('span');
    glyph.className = 'folder-glyph';
    glyph.setAttribute('aria-hidden', 'true');

    const copy = document.createElement('span');
    copy.className = 'library-folder-copy';
    const title = document.createElement('strong');
    title.textContent = folder.name;
    const meta = document.createElement('small');
    meta.textContent = `${privateCount} 篇`;
    copy.append(title, meta);
    card.append(glyph, copy);
    bindFolderCard(card);
    return card;
  }

  function createPrivateArticle(file, state) {
    const key = `private:${file.path}`;
    const folderId = state.assignments?.[key] || 'unfiled';
    const title = state.labels?.[key] || prettyName(file.path);
    const privateUrl = privateReaderUrl(file.path);
    const openReader = (event) => {
      event.preventDefault();
      openPrivateNote(file, state);
    };

    const article = document.createElement('article');
    article.className = 'note-row note-row-private';
    article.dataset.libraryFolder = folderId;
    article.dataset.privateNote = 'true';
    article.dataset.privatePath = file.path;

    const number = document.createElement('p');
    number.className = 'note-number';

    const main = document.createElement('div');
    main.className = 'note-row-main';
    const heading = document.createElement('div');
    heading.className = 'note-row-heading';
    const h2 = document.createElement('h2');
    const titleLink = document.createElement('a');
    titleLink.href = privateUrl.href;
    titleLink.rel = 'nofollow';
    titleLink.textContent = title;
    titleLink.addEventListener('click', openReader);
    h2.append(titleLink);
    const privacy = document.createElement('span');
    privacy.className = 'note-privacy-badge';
    privacy.textContent = '私密 · 仅你可见';
    heading.append(h2, privacy);

    const summary = document.createElement('p');
    summary.className = 'note-summary';
    summary.textContent = '内容保存在私密仓库中，仅通过当前设备的所有者身份读取。';

    const footer = document.createElement('div');
    footer.className = 'note-row-footer';
    const ownerOnly = document.createElement('span');
    ownerOnly.textContent = 'Owner access';
    const open = document.createElement('a');
    open.href = privateUrl.href;
    open.rel = 'nofollow';
    open.textContent = '私密阅读';
    open.addEventListener('click', openReader);
    footer.append(ownerOnly, open);

    main.append(heading, summary, footer);
    article.append(number, main);
    return article;
  }

  function updateFolderCounts() {
    refreshCollections();
    cards.forEach((card) => {
      const folderId = card.dataset.libraryFilter;
      const total = folderId === 'all'
        ? articles.length
        : articles.filter((article) => article.dataset.libraryFolder === folderId).length;
      const meta = card.querySelector('.library-folder-copy small');
      if (meta) meta.textContent = `${total} 篇`;
    });
  }

  function addOwnerFolders(state, privateFiles) {
    const existing = new Set([...folderList.querySelectorAll('[data-library-filter]')]
      .map((card) => card.dataset.libraryFilter));
    const privateCounts = new Map();
    privateFiles.forEach((file) => {
      const folderId = state.assignments?.[`private:${file.path}`] || 'unfiled';
      privateCounts.set(folderId, (privateCounts.get(folderId) || 0) + 1);
    });

    (state.folders || []).forEach((folder) => {
      if (!existing.has(folder.id) && privateCounts.has(folder.id)) {
        folderList.append(createFolderCard(folder, privateCounts.get(folder.id)));
        existing.add(folder.id);
      }
    });

    if (privateCounts.has('unfiled') && !existing.has('unfiled')) {
      folderList.append(createFolderCard({
        id: 'unfiled',
        name: '未分类',
        color: 9,
      }, privateCounts.get('unfiled')));
    }
  }

  async function loadOwnerLibrary() {
    const token = localStorage.getItem(tokenKey) || sessionStorage.getItem(tokenKey);
    if (!token || !privateRepo || !owner) return;

    const apiRoot = `https://api.github.com/repos/${privateRepo}`;
    try {
      const [user, repository] = await Promise.all([
        githubJson('https://api.github.com/user', token),
        githubJson(apiRoot, token),
      ]);
      if (user.login?.toLowerCase() !== owner.toLowerCase()
          || !repository.private
          || !repository.permissions?.push) return;
      ownerToken = token;
      privateApiRoot = apiRoot;

      const [privateFiles, stateFile] = await Promise.all([
        githubJson(`${apiRoot}/contents/notes?ref=${encodeURIComponent(branch)}`, token),
        githubJson(`${apiRoot}/contents/library/folders.json?ref=${encodeURIComponent(branch)}`, token),
      ]);
      const files = privateFiles.filter((file) => file.type === 'file' && file.name.endsWith('.md'));
      const state = JSON.parse(decodeBase64(stateFile.content));
      privateState = state;
      privateFilesByPath = new Map(files.map((file) => [file.path, file]));
      addOwnerFolders(state, files);
      files.forEach((file) => articleList.append(createPrivateArticle(file, state)));
      page.dataset.ownerView = 'true';
      updateFolderCounts();
      selectFolder(selectedFolderId, false);
      const requestedPrivatePath = new URLSearchParams(location.search).get('private');
      if (requestedPrivatePath && privateFilesByPath.has(requestedPrivatePath)) {
        openPrivateNote(privateFilesByPath.get(requestedPrivatePath), privateState, false);
      }
    } catch (_) {
      // A missing or expired owner token leaves the public library unchanged.
    }
  }

  reader.close?.addEventListener('click', () => dismissPrivateReader());
  reader.dialog?.addEventListener('cancel', (event) => {
    event.preventDefault();
    dismissPrivateReader();
  });
  reader.dialog?.addEventListener('click', (event) => {
    if (event.target === reader.dialog) dismissPrivateReader();
  });
  window.addEventListener('popstate', () => {
    const path = new URLSearchParams(location.search).get('private');
    if (!path) {
      if (reader.dialog?.open) dismissPrivateReader(false);
      return;
    }
    const file = privateFilesByPath.get(path);
    if (file && path !== activePrivatePath) openPrivateNote(file, privateState, false);
  });

  refreshCollections();
  cards.forEach(bindFolderCard);
  const requested = location.hash.startsWith('#folder=') ? decodeURIComponent(location.hash.slice(8)) : 'all';
  selectedFolderId = requested;
  selectFolder(requested, false);
  loadOwnerLibrary();
})();
