/* eslint-disable no-undef, @typescript-eslint/no-require-imports */
const { chromium } = require('playwright');

async function run() {
  const url = process.argv[2];
  if (!url) {
    console.error('No URL provided!');
    process.exit(1);
  }

  const browser = await chromium.launch({
    headless: false, // Must be false to render to Xvfb
    args: [
      '--disable-infobars',
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--autoplay-policy=no-user-gesture-required',
      '--window-position=0,0',
      '--window-size=1920,1080',
    ],
  });

  const context = await browser.newContext({
    viewport: null, // Use full window size, no fixed viewport
  });

  const page = await context.newPage();
  page.setDefaultTimeout(120000);

  // Log browser console messages to see what's happening
  page.on('console', (msg) => {
    const text = msg.text();
    if (text.includes('Hand #') || text.includes('Action:') || text.includes('DEBUG:')) {
      console.log(`  [BROWSER] ${text}`);
    }
  });

  await login(page);

  // Use CDP to set window to fullscreen
  const cdpSession = await context.newCDPSession(page);
  const { windowId } = await cdpSession.send('Browser.getWindowForTarget');
  await cdpSession.send('Browser.setWindowBounds', {
    windowId,
    bounds: { windowState: 'fullscreen' },
  });

  // Wait for "Press F11 to exit fullscreen" notification to disappear
  await page.waitForTimeout(5000);

  // --- REPLAY LOGIC ---
  // Change this selector to match whatever element appears
  // when your replay or animation is finished.
  console.log('Waiting for replay to finish...');

  await browser.close();
}

run();
