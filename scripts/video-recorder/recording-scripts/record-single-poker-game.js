import { chromium } from 'playwright';
import { login } from '../login.js';

const TARGET_URL =
  'https://kaggle.com/competitions/repeated-poker/leaderboard?submissionId=47607376&episodeId=73734724&onlyStream=true';
const TOTAL_HANDS = 100;

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

  let handsCompleted = 0;

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

  console.log(`[Poker] Navigating to episode: ${TARGET_URL}`);
  await page.goto(TARGET_URL, {
    waitUntil: 'domcontentloaded',
  });

  console.log(`[Poker] Waiting for game to load...`);

  // Wait for the scroller to appear (indicates game is loaded)
  await page.waitForSelector('[data-testid="virtuoso-scroller"]', {
    timeout: 60000,
  });

  console.log(`[Poker] Game loaded. Recording ${TOTAL_HANDS} hands...`);

  // Monitor for hands completing - look for "wins", "split pot", or "collected"
  // Each time we see a new hand complete, increment our counter
  let lastHandNumber = 0;

  while (handsCompleted < TOTAL_HANDS) {
    // Poll until we detect a hand completion
    const currentHandNumber = await page.evaluate(() => {
      const scroller = document.querySelector('[data-testid="virtuoso-scroller"]');
      if (!scroller) return 0;

      // Auto-scroll to keep the virtualized list rendering new elements
      scroller.scrollTop = scroller.scrollHeight;

      const text = scroller.innerText.toLowerCase();

      // Find all "hand X" mentions to determine current hand
      const handMatches = text.match(/hand\s+#?(\d+)/gi) || [];
      const handNumbers = handMatches
        .map((m) => {
          const num = m.match(/\d+/);
          return num ? parseInt(num[0], 10) : 0;
        })
        .filter((n) => n > 0);

      const maxHand = handNumbers.length > 0 ? Math.max(...handNumbers) : 0;

      // Get the most recent visible content to check for completion
      const actions = Array.from(scroller.querySelectorAll('div, button[role="button"]'));
      const lastFewText = actions
        .slice(-10)
        .map((a) => a.innerText.trim())
        .filter((t) => t.length > 0)
        .join(' ')
        .toLowerCase();

      const recentHasWin = /wins|split pot|collected/.test(lastFewText);

      if (recentHasWin && maxHand > 0) {
        return maxHand;
      }

      return 0;
    });

    if (currentHandNumber > lastHandNumber) {
      handsCompleted = currentHandNumber;
      lastHandNumber = currentHandNumber;
      console.log(`[Poker] Hand ${handsCompleted} completed (${TOTAL_HANDS - handsCompleted} remaining)`);
    }

    // Small delay between polls
    await page.waitForTimeout(1000);
  }

  console.log(`[Poker] All ${TOTAL_HANDS} hands recorded. Exiting...`);

  // Brief pause before closing
  await page.waitForTimeout(2000);

  await browser.close();
}

run();
