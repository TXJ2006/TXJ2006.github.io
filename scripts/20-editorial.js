'use strict';
const {stripHTML} = require('hexo-util');
function plain(value) { return stripHTML(String(value || '')).replace(/\s+/g, ' ').trim(); }
function pageUrl(path) { return '/' + String(path).replace(/^\/+/, '').replace(/index\.html$/, ''); }
const series = [
  {title:'基础数学系列',url:'/series/mathematics/',summary:'代数学、分析学、几何与拓扑：按学科与册次组织的系统讲义。',image:'geometry'},
  {title:'Reading Hatcher',url:'/series/hatcher/',summary:'从同伦、胞腔复形到基本群，沿着 Hatcher 的教材阅读与思考。',image:'topology'},
  {title:'分析学补充专题',url:'/series/analysis/',summary:'测度、Lebesgue 积分、Lp 空间、弱导数与 Sobolev 空间。',image:'analysis'},
  {title:'吉尔伯特线性代数 · 读书笔记',url:'/courses/gilbert-linear-algebra/notes/',summary:'课程阅读与习题讨论。各讲独立整理，在课程目录中阅读。',image:'linear'}
].map(p => Object.assign({category:'专题系列',content:p.summary},p));
function catalog(locals) {
  // Deliberately curate series entry points. Individual lessons and private notes
  // do not enter the blog feed or global search index.
  const posts = locals.posts.toArray().filter(p=>/推荐|入门书籍|^Writing Sample$/.test(p.title)).map(p=>({
    title:p.title,url:pageUrl(p.path),summary:plain(p.description||p.excerpt||p.content).slice(0,105),
    category:/推荐|入门书籍/.test(p.title)?'书籍推荐':'研究笔记',date:p.date,
    image:/图论/.test(p.title)?'topology':'algebra',content:plain(p.content).slice(0,9000)
  })).sort((a,b)=>new Date(b.date)-new Date(a.date));
  return series.concat(posts);
}
hexo.extend.helper.register('editorial_catalog',function(){return catalog(this.site);});
hexo.extend.helper.register('editorial_featured',function(){return series;});
hexo.extend.generator.register('editorial-search',function(locals){
  const items=catalog(locals);
  for(const p of locals.pages.toArray()){
    if(!/^(about|research|publications|cv|math-resources|math-qual-exams|courses\/(gilbert-linear-algebra|mit-18-700|mit-18-703))\/index\.html$/.test(p.path.replace(/^\/+/,'')))continue;
    items.push({title:p.title,url:pageUrl(p.path),category:'资料与课程',content:plain(p.content).slice(0,30000)});
  }
  return {path:'editorial-search.json',data:JSON.stringify(items.map(p=>({title:p.title,url:p.url,category:p.category,content:p.content})))};
});
// Preserve legacy paginated index URLs after moving the blog to /blog/.
hexo.extend.generator.register('legacy-blog-pagination',function(locals){
  const count=Math.ceil(locals.posts.length/hexo.config.index_generator.per_page);
  return Array.from({length:Math.max(0,count-1)},(_,i)=>({path:'page/'+(i+2)+'/index.html',data:'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=/blog/"><link rel="canonical" href="https://txj2006.github.io/blog/"><title>博客</title><a href="/blog/">进入博客</a></html>'}));
});
