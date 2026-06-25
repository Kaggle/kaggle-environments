import playerWhitePath from '../assets/images/player-color-w.webp';
import playerBlackPath from '../assets/images/player-color-b.webp';

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
  './images/popover.webp',
  './images/squiggle-solid.png',
  './images/squiggle-v.png',
  // Bundled assets (player bar)
  playerWhitePath,
  playerBlackPath,
];

export const assetsReady = Promise.all([...criticalImages.map(preloadImage), document.fonts.ready]);
