// SPDX-License-Identifier: Apache-2.0
// Keep the downloadable HTML self-contained while retaining the Node model entry points.
'use strict';

const fs = require('node:fs');
const path = require('node:path');

const args = process.argv.slice(2);
if (args.length > 1 || (args.length === 1 && args[0] !== '--check')) {
    throw new Error('Usage: node bundle_calculator.js [--check]');
}
const htmlPath = path.join(__dirname, 'memory_calculator.html');
const html = fs.readFileSync(htmlPath, 'utf8');
const start = '  <!-- BEGIN INLINE MEMORY ASSETS -->';
const end = '  <!-- END INLINE MEMORY ASSETS -->';
const begin = html.indexOf(start);
const finish = html.indexOf(end);
if (begin < 0 || finish < begin || html.indexOf(start, begin + start.length) >= 0 ||
    html.indexOf(end, finish + end.length) >= 0) {
    throw new Error('Expected one complete inline asset section in memory_calculator.html');
}
const scripts = ['memory_buffers.js', 'memory_model.js'].map(name => {
    const source = fs.readFileSync(path.join(__dirname, name), 'utf8')
        .replace(/<\/script/gi, '<\\/script').trimEnd();
    return `  <script data-inline-source="${name}">\n${source}\n  </script>`;
});
const bundled = html.slice(0, begin) + [start, ...scripts, end].join('\n') +
    html.slice(finish + end.length);
if (args[0] === '--check') {
    if (bundled !== html) {
        throw new Error('Inline models are stale. Run node tools/proxy_memory_calculator/bundle_calculator.js');
    }
    console.log('Inline calculator models match their JavaScript sources.');
} else if (bundled !== html) {
    fs.writeFileSync(htmlPath, bundled);
    console.log('Updated the self-contained memory_calculator.html.');
}
