(function () {
  'use strict';
  const root = document.documentElement;
  const themeButton = document.querySelector('.theme-toggle');
  function themeLabel() { themeButton.textContent = root.dataset.theme === 'dark' ? '浅色' : '深色'; themeButton.setAttribute('aria-label', root.dataset.theme === 'dark' ? '切换为浅色模式' : '切换为深色模式'); }
  themeLabel();
  themeButton.addEventListener('click', () => {root.dataset.theme = root.dataset.theme === 'dark' ? 'light' : 'dark'; try {localStorage.setItem('blog-theme',root.dataset.theme);} catch (_) {} themeLabel();});
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {try {if (localStorage.getItem('blog-theme')) return;} catch (_) {} root.dataset.theme = e.matches ? 'dark' : 'light'; themeLabel();});
  const menuButton = document.querySelector('.menu-toggle');
  const menu = document.querySelector('.primary-navigation');
  function closeMenu() {menu.classList.remove('is-open');menuButton.setAttribute('aria-expanded','false');}
  menuButton.addEventListener('click', () => {const open=menu.classList.toggle('is-open');menuButton.setAttribute('aria-expanded',String(open));});
  document.addEventListener('keydown',e=>{if(e.key==='Escape') {closeMenu();document.querySelectorAll('.more-navigation[open]').forEach(el=>el.open=false);}});
  menu.querySelectorAll('a').forEach(a=>{if(a.pathname===location.pathname) a.setAttribute('aria-current','page');});
  const dialog = document.querySelector('.site-search-overlay');
  const input = document.querySelector('#site-search-input');
  const status = document.querySelector('.site-search-status');
  const results = document.querySelector('.site-search-results');
  let indexPromise; let searchSequence=0;
  function loadIndex() {if(!indexPromise) indexPromise=fetch('/editorial-search.json').then(r=>{if(!r.ok)throw new Error('index');return r.json();}).catch(e=>{indexPromise=null;throw e;});return indexPromise;}
  function closeSearch(){dialog.hidden=true;document.body.style.overflow='';document.querySelector('.search-toggle').focus();}
  document.querySelector('.search-toggle').addEventListener('click',()=>{dialog.hidden=false;document.body.style.overflow='hidden';closeMenu();input.focus();});
  document.querySelector('.search-close').addEventListener('click',closeSearch);
  dialog.addEventListener('pointerdown',e=>{if(e.target===dialog)closeSearch();});
  dialog.addEventListener('keydown',e=>{if(e.key==='Escape'){e.preventDefault();closeSearch();}if(e.key==='Tab'){const focusable=[...dialog.querySelectorAll('button,input,a[href]')];const first=focusable[0],last=focusable[focusable.length-1];if(e.shiftKey&&document.activeElement===first){e.preventDefault();last.focus();}else if(!e.shiftKey&&document.activeElement===last){e.preventDefault();first.focus();}}});
  document.querySelector('.site-search-form').addEventListener('submit',e=>e.preventDefault());
  input.addEventListener('input',async()=>{
    const seq=++searchSequence;const query=input.value.trim().toLowerCase();results.replaceChildren();
    if(!query){status.textContent='输入关键词搜索本站内容。';return;}
    status.textContent='正在搜索…';
    try {const data=await loadIndex();if(seq!==searchSequence)return;
      const words=query.split(/\s+/);const matches=data.filter(p=>words.every(w=>(p.title+' '+p.content).toLowerCase().includes(w))).sort((a,b)=>Number(b.title.toLowerCase().includes(query))-Number(a.title.toLowerCase().includes(query)));
      status.textContent=matches.length ? '找到 '+matches.length+' 条结果。' : '没有找到匹配内容，请换一个关键词。';
      for(const p of matches.slice(0,30)){const item=document.createElement('article');const a=document.createElement('a');a.href=p.url;a.textContent=p.title;const meta=document.createElement('p');meta.textContent=p.category;item.append(a,meta);results.append(item);}
    } catch(_){if(seq===searchSequence) status.textContent='搜索暂时无法加载，请检查网络后重新输入关键词。';}
  });
  const tocDetails=document.querySelector('.reading-toc details');
  if(tocDetails && matchMedia('(max-width: 1000px)').matches) tocDetails.open=false;
  const links=[...document.querySelectorAll('.reading-toc a')];
  if(links.length){const map=new Map();links.forEach(a=>{let id;try{id=decodeURIComponent(a.hash.slice(1));}catch(_){return;}const h=document.getElementById(id);if(h)map.set(h,a);});const observer=new IntersectionObserver(entries=>{for(const e of entries)if(e.isIntersecting){links.forEach(a=>a.removeAttribute('aria-current'));map.get(e.target).setAttribute('aria-current','location');}},{rootMargin:'-5% 0px -70% 0px'});map.forEach((a,h)=>observer.observe(h));}
})();
