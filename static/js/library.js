(() => {
  const cards = [...document.querySelectorAll('[data-library-filter]')];
  const articles = [...document.querySelectorAll('[data-library-folder]')];
  const label = document.querySelector('#library-result-label');
  const count = document.querySelector('#library-result-count');
  if (!cards.length || !articles.length) return;

  function selectFolder(folderId, updateHistory = true) {
    const selected = cards.find((card) => card.dataset.libraryFilter === folderId) || cards[0];
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
      const hash = selected.dataset.libraryFilter === 'all' ? '' : `#folder=${encodeURIComponent(selected.dataset.libraryFilter)}`;
      history.replaceState(null, '', `${location.pathname}${location.search}${hash}`);
    }
  }

  cards.forEach((card) => card.addEventListener('click', () => selectFolder(card.dataset.libraryFilter)));
  const requested = location.hash.startsWith('#folder=') ? decodeURIComponent(location.hash.slice(8)) : 'all';
  selectFolder(requested, false);
})();
