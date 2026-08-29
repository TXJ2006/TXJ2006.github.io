'use strict';

var fs = require('fs');
var path = require('path');

var frontMatterPath = path.join(__dirname, '..', 'node_modules', 'hexo-front-matter', 'lib', 'front_matter.js');
var feedGeneratorPath = path.join(__dirname, '..', 'node_modules', 'hexo-generator-feed', 'lib', 'generator.js');
var generateConsolePath = path.join(__dirname, '..', 'node_modules', 'hexo', 'lib', 'plugins', 'console', 'generate.js');

if (fs.existsSync(frontMatterPath)) {
  var source = fs.readFileSync(frontMatterPath, 'utf8');
  var legacy = 'var isDate = util.isDate;';
  var compatible = "var isDate = util.isDate || function (value) { return Object.prototype.toString.call(value) === '[object Date]'; };";

  if (source.indexOf(legacy) !== -1 && source.indexOf(compatible) === -1) {
    fs.writeFileSync(frontMatterPath, source.replace(legacy, compatible));
  }
}

// hexo-generator-feed 1.x assumes at least one post when rendering Atom/RSS.
// Keep the generator installed, but emit a valid empty feed while the blog has no posts.
if (fs.existsSync(feedGeneratorPath)) {
  var feedSource = fs.readFileSync(feedGeneratorPath, 'utf8');
  var feedMarker = "  if (feedConfig.limit) posts = posts.limit(feedConfig.limit);\n";
  var feedPatch = feedMarker + "\n  if (!posts.length) {\n    var emptyFeed = '<?xml version=\"1.0\" encoding=\"utf-8\"?>\\n' +\n      '<feed xmlns=\"http://www.w3.org/2005/Atom\">\\n' +\n      '  <title>' + config.title + '</title>\\n' +\n      '  <link href=\"' + (config.root + feedConfig.path) + '\" rel=\"self\"/>\\n' +\n      '  <id>' + config.url + '</id>\\n' +\n      '</feed>\\n';\n\n    return {\n      path: feedConfig.path,\n      data: emptyFeed\n    };\n  }\n";

  if (feedSource.indexOf(feedMarker) !== -1 && feedSource.indexOf('if (!posts.length)') === -1) {
    fs.writeFileSync(feedGeneratorPath, feedSource.replace(feedMarker, feedPatch));
  }
}

// Hexo 3.8's generator overrides stream.destroy() to clear its cache. Modern
// Node calls destroy automatically when a stream ends, before the cache is written.
if (fs.existsSync(generateConsolePath)) {
  var generateSource = fs.readFileSync(generateConsolePath, 'utf8');
  var legacyCall = '      // Destroy cache\n      cacheStream.destroy();';
  var compatibleCall = '      // Clear the buffered data after the file has been written.\n      cacheStream.clearCache();';
  var legacyDestroy = 'CacheStream.prototype.destroy = function() {\n  this._cache.length = 0;\n};';
  var compatibleDestroy = 'CacheStream.prototype.clearCache = function() {\n  this._cache.length = 0;\n};';

  if (generateSource.indexOf(legacyCall) !== -1) {
    generateSource = generateSource.replace(legacyCall, compatibleCall);
  }

  if (generateSource.indexOf(legacyDestroy) !== -1) {
    generateSource = generateSource.replace(legacyDestroy, compatibleDestroy);
  }

  if (generateSource.indexOf('cacheStream.clearCache();') !== -1 && generateSource.indexOf('CacheStream.prototype.clearCache') !== -1) {
    fs.writeFileSync(generateConsolePath, generateSource);
  }
}
