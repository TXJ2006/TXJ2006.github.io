(() => {
  const page = document.querySelector('.notes-page');
  const folderList = document.querySelector('.library-folders');
  const articleList = document.querySelector('.notes-list');
  const label = document.querySelector('#library-result-label');
  const count = document.querySelector('#library-result-count');
  if (!page || !folderList || !articleList || !label || !count) return;

  const privateRepo = page.dataset.privateRepo;
  const owner = page.dataset.owner;
  const branch = page.dataset.branch || 'main';
  const adminUrl = page.dataset.adminUrl || '/admin/';
  const tokenKey = 'txj_notes_editor_token';
  let cards = [];
  let articles = [];
  let selectedFolderId = 'all';

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
    const privateUrl = new URL(adminUrl, window.location.href);
    privateUrl.searchParams.set('repository', 'private');
    privateUrl.searchParams.set('path', file.path);
    privateUrl.searchParams.set('mode', 'preview');

    const article = document.createElement('article');
    article.className = 'note-row note-row-private';
    article.dataset.libraryFolder = folderId;
    article.dataset.privateNote = 'true';

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

      const [privateFiles, stateFile] = await Promise.all([
        githubJson(`${apiRoot}/contents/notes?ref=${encodeURIComponent(branch)}`, token),
        githubJson(`${apiRoot}/contents/library/folders.json?ref=${encodeURIComponent(branch)}`, token),
      ]);
      const files = privateFiles.filter((file) => file.type === 'file' && file.name.endsWith('.md'));
      const state = JSON.parse(decodeBase64(stateFile.content));
      addOwnerFolders(state, files);
      files.forEach((file) => articleList.append(createPrivateArticle(file, state)));
      page.dataset.ownerView = 'true';
      updateFolderCounts();
      selectFolder(selectedFolderId, false);
    } catch (_) {
      // A missing or expired owner token leaves the public library unchanged.
    }
  }

  refreshCollections();
  cards.forEach(bindFolderCard);
  const requested = location.hash.startsWith('#folder=') ? decodeURIComponent(location.hash.slice(8)) : 'all';
  selectedFolderId = requested;
  selectFolder(requested, false);
  loadOwnerLibrary();
})();
