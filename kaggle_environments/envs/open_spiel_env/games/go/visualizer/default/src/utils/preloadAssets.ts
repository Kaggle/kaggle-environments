import stoneBlackPath from '../assets/stone-black.webp';
import stoneWhitePath from '../assets/stone-white.webp';
import scoreboardBlackPath from '../assets/scoreboard-player-black.webp';
import scoreboardWhitePath from '../assets/scoreboard-player-white.webp';
import potPath from '../assets/pot.webp';

function preloadImage(src: string): Promise<void> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = img.onerror = () => resolve();
    img.src = src;
  });
}

const criticalImages = [
  // Public directory images (CSS backgrounds)
  './images/paper.webp',
  './images/squiggle-solid.png',
  './images/squiggle-v.png',
  // Bundled assets (score panel)
  stoneBlackPath,
  stoneWhitePath,
  scoreboardBlackPath,
  scoreboardWhitePath,
  potPath,
];

export const assetsReady = Promise.all([...criticalImages.map(preloadImage), document.fonts.ready]);
