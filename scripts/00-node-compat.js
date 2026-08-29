'use strict';

// Hexo 3.8 calls the Node API removed in recent Node releases.
var util = require('util');

if (typeof util.isDate !== 'function') {
  util.isDate = function (value) {
    return Object.prototype.toString.call(value) === '[object Date]';
  };
}
