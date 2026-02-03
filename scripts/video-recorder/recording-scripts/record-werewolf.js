/* eslint-disable no-undef, @typescript-eslint/no-require-imports */
const { chromium } = require('playwright');
const { login } = require('../login.js');

async function run() {
  const episodeId = process.argv[2];
  if (!episodeId) {
    console.error('No Episode ID provided!');
    process.exit(1);
  }

  let browser;
  try {
    browser = await chromium.launch({
      headless: false, // Must be false to render to Xvfb
      args: [
        '--disable-infobars',
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--autoplay-policy=no-user-gesture-required',
        '--window-position=0,0',
        '--window-size=1920,1080',
        // GPU and WebGL flags for 3D rendering
        '--enable-gpu',
        '--enable-webgl',
        '--enable-webgl2',
        '--ignore-gpu-blocklist',
        '--enable-accelerated-2d-canvas',
        '--enable-gpu-rasterization',
        '--disable-software-rasterizer',
        // Memory and performance
        '--disable-dev-shm-usage', // Avoid /dev/shm issues in Docker
        '--disable-background-timer-throttling',
        '--disable-backgrounding-occluded-windows',
        '--disable-renderer-backgrounding',
        // Shader compilation
        '--enable-unsafe-swiftshader',
      ],
    });

    const contextOptions = {
      viewport: null, // Use full window size, no fixed viewport
    };

    const context = await browser.newContext(contextOptions);

    const page = await context.newPage();

    // 4 hour timeout for asset loading and game completion = 4 * 60 * 60 * 1000
    const gameTimeoutMs = 4 * 60 * 60 * 1000;
    page.setDefaultTimeout(gameTimeoutMs);

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

    console.log(`Navigating to episode with ID ${episodeId}`);
    await page.goto(
      `https://www.kaggle.com/benchmarks/kaggle/werewolf?episodeId=${episodeId}`,
      { waitUntil: 'commit' } // Don't wait for DOM, just for navigation to start
    );
    console.log('[WEREWOLF] Page navigation started, waiting 3 seconds...');
    await page.waitForTimeout(3000);

    // --- WEREWOLF REPLAY LOGIC ---

    // Find the iframe containing the play-pause button
    let targetFrame;
    const frames = page.frames();
    for (const frame of frames) {
      const hasButton = await frame.evaluate(() => !!document.querySelector('button#play-pause'));
      if (hasButton) {
        targetFrame = frame;
        break;
      }
    }

    // Press 'e' to expand logs
    console.log("[WEREWOLF] Pressing 'e' to expand logs");
    await page.keyboard.press('e');
    console.log('[WEREWOLF] Logs expanded');

    // Wait for and click the play-pause button to start the game
    console.log('[WEREWOLF] Waiting for play-pause button');
    const playPauseButton = targetFrame.locator('button#play-pause');
    await playPauseButton.waitFor({ state: 'visible', timeout: gameTimeoutMs });
    await playPauseButton.click();
    console.log('[WEREWOLF] Clicked play-pause button, game started.');

    // Monitor the step counter until it reaches completion (X / Y where X === Y)
    // Use targetFrame since controls are inside the iframe
    console.log('[WEREWOLF] Monitoring step counter for game completion...');

    let lastLoggedStep = null;
    while (true) {
      const stepInfo = await targetFrame.evaluate(() => {
        const stepCounter = document.querySelector('.step-counter');
        if (!stepCounter) return null;

        const text = stepCounter.textContent || '';
        const match = text.match(/(\d+)\s*\/\s*(\d+)/);
        if (!match) return null;

        const current = parseInt(match[1], 10);
        const total = parseInt(match[2], 10);
        return { current, total };
      });

      if (stepInfo) {
        // Log to terminal (not browser console)
        if (lastLoggedStep !== stepInfo.current) {
          console.log(`[WEREWOLF] Step counter: ${stepInfo.current} / ${stepInfo.total}`);
          lastLoggedStep = stepInfo.current;
        }

        if (stepInfo.current === stepInfo.total) {
          break;
        }
      }

      await page.waitForTimeout(1000);
    }

    console.log('[WEREWOLF] Game complete! Waiting 5 seconds for final state...');
    await page.waitForTimeout(5000);

    console.log('Recording finished.');
  } catch (error) {
    console.error(`[ERROR] ${error.message}`);
    console.log('Waiting 5 seconds before closing to allow FFmpeg to finalize...');
    await new Promise((resolve) => setTimeout(resolve, 5000));
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

run();
