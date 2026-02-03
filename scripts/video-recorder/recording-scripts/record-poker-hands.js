/* eslint-disable no-undef, @typescript-eslint/no-require-imports, @typescript-eslint/no-unused-vars */
const { chromium } = require('playwright');
const { login } = require('../login.js');

const handsFile = process.argv[2];
if (!handsFile) {
  console.error('Usage: node record.js <hands-file.txt>');
  process.exit(1);
}

async function run() {
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

  // --- POKER REPLAY LOGIC ---
  await page.goto('https://www.kaggle.com/competitions/duplicate-poker/leaderboard?onlyStream=true', {
    waitUntil: 'domcontentloaded',
  });

  console.log('[Poker] Uploading file...');

  const fileChooserPromise = page.waitForEvent('filechooser');
  await page.getByRole('button', { name: 'Visualize Data' }).click();

  console.log('[Poker] Waiting for file chooser...');

  const fileChooser = await fileChooserPromise;
  await fileChooser.setFiles(handsFile);

  console.log('[Poker] Waiting for hands to load...');

  const handsList = page.getByTestId('episode-slices');
  await handsList.getByRole('button').first().waitFor({ state: 'visible', timeout: 30000 });
  const hands = await handsList.getByRole('button').all();
  console.log(`[Poker] Found ${hands.length} hands`);

  let handIndex = 0;
  for (const hand of hands) {
    console.log(`[Poker] Processing hand ${handIndex + 1} of ${hands.length}...`);

    // Extract the hand number from the hand button (e.g., "Hand #2" -> 2)
    const handText = await hand.textContent();
    const handNumberMatch = handText?.match(/Hand #(\d+)/i);
    const expectedHandNumber = handNumberMatch ? parseInt(handNumberMatch[1], 10) : null;
    console.log(`[Poker] Expected hand number: ${expectedHandNumber}`);

    await hand.click();

    // GATE 1: Wait for scroller to initialize with this hand before monitoring
    await page
      .waitForFunction(
        (num) => {
          const scroller = document.querySelector('[data-testid="virtuoso-scroller"]');
          if (!scroller) return false;
          // Use word boundary to avoid "Hand 100" matching "Hand 1000"
          const pattern = new RegExp(`Hand #?${num}\\b`);
          return pattern.test(scroller.innerText);
        },
        expectedHandNumber,
        { timeout: 15000 }
      )
      .catch(() => {});

    console.log(`[Poker] Monitoring playback for hand ${handIndex + 1}...`);

    // Poll until we see "wins", "split pot", or "collected" for the correct hand
    await page.waitForFunction(
      (num) => {
        const scroller = document.querySelector('[data-testid="virtuoso-scroller"]');
        if (!scroller) return false;

        // Auto-scroll to keep the virtualized list rendering new elements
        scroller.scrollTop = scroller.scrollHeight;

        // Get text from last few elements (matching working version)
        const actions = Array.from(scroller.querySelectorAll('div, button[role="button"]'));
        const lastFewText = actions
          .slice(-5)
          .map((a) => a.innerText.trim())
          .filter((t) => t.length > 0);

        // console.log(
        //  `DEBUG: Hand #${num} recent: ${lastFewText.slice(-2).join(" | ").substring(0, 60)}`,
        // );

        const fullRecentText = lastFewText.join(' ').toLowerCase();
        const isWin = /wins|split pot|collected/.test(fullRecentText);
        // const isCorrectHand =
        //  fullRecentText.includes(`hand ${num}`) ||
        //  fullRecentText.includes(`hand #${num}`);

        return isWin;
      },
      expectedHandNumber,
      { polling: 2000, timeout: 2400000 }
    );
    console.log(`[Poker] Hand ${handIndex + 1} complete (win/tie detected)`);
    await page.waitForTimeout(2000);
    handIndex++;
  }

  await browser.close();
}

run();
