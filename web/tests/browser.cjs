// Run against the served production build using an existing Playwright install.
const assert = require('node:assert/strict');
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
(async () => {
  const browser = await chromium.launch({ executablePath: process.env.CHROME_BIN || '/usr/bin/chromium', headless: true, args: ['--no-sandbox'] });
  const errors = [];
  try {
    for (const width of [1440, 390, 320]) {
      const page = await browser.newPage({ viewport: { width, height: 1000 } });
      page.on('pageerror', error => errors.push(error.message));
      page.on('console', message => { if (message.type() === 'error') errors.push(message.text()); });
      const response = await page.goto(process.env.SITE_URL || 'http://localhost:4174');
      assert.equal(response.status(), 200);
      await page.waitForSelector('tbody tr');
      assert.equal(await page.locator('tbody tr').count(), 4);
      assert.equal(await page.locator('.bar').count(), 2);
      assert.deepEqual(await page.locator('tbody tr').evaluateAll(rows => rows.map(row => Array.from(row.cells, cell => cell.textContent))), [
        ['MPI', '10,000', '2 ranks (report p. 20)', '1.19×', '36'],
        ['MPI', '100,000', '2 ranks (report p. 20)', '6.45×', '36'],
        ['MPI + OpenMP', '100,000', '2 ranks, 8 threads per rank', '18.03×', '35'],
        ['OpenMP', '12,500', '2 threads', '39.35×', '21'],
      ]);
      const widths = await page.locator('.bar').evaluateAll(bars => bars.map(bar => parseFloat(bar.style.width)));
      assert.deepEqual(widths, [32.25, 90.15]);
      assert.equal(await page.locator('svg[role="img"] title').count(), 1);
      assert.equal(await page.locator('.table-wrap').getAttribute('tabindex'), '0');
      assert.equal(await page.locator('.table-wrap').getAttribute('aria-label'), 'Recorded observations, scroll horizontally for all columns');
      await page.keyboard.press('Tab');
      assert.equal(await page.locator(':focus').textContent(), 'Skip to content');
      await page.keyboard.press('Enter');
      assert.equal(new URL(page.url()).hash, '#main');
      assert.equal(await page.locator('h1').innerText(), 'When the graph changes,\nkeep the shortest path.');
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      await page.locator('nav a[href="#results"]').click();
      assert.equal(new URL(page.url()).hash, '#results');
      for (const href of await page.locator('a[href*="/assets/"]').evaluateAll(nodes => nodes.map(node => node.href))) {
        assert.equal((await page.request.get(href)).status(), 200);
      }
      await page.screenshot({ path: `/tmp/sssp-results-${width}.png`, fullPage: true });
      await page.close();
    }
    assert.deepEqual(errors, []);
    console.log('PASS: production build at desktop/mobile, chart/table content, navigation, linked evidence, no overflow or page errors');
  } finally { await browser.close(); }
})().catch(error => { console.error(error); process.exitCode = 1; });
