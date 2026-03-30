import { useEffect } from 'react';

/**
 * Tracks the board canvas position relative to #go-playable-area and exposes
 * it as CSS custom properties (--board-top, --board-left, --board-width,
 * --board-height). This replaces CSS Anchor Positioning which lacks
 * cross-browser support.
 */
export default function useBoardRect(): void {
  useEffect(() => {
    const area = document.getElementById('go-playable-area');
    const canvas = document.getElementById('board');
    if (!area || !canvas) return;

    const sync = () => {
      const areaRect = area.getBoundingClientRect();
      const canvasRect = canvas.getBoundingClientRect();
      area.style.setProperty('--board-top', `${canvasRect.top - areaRect.top + area.scrollTop}px`);
      area.style.setProperty('--board-left', `${canvasRect.left - areaRect.left + area.scrollLeft}px`);
      area.style.setProperty('--board-width', `${canvasRect.width}px`);
      area.style.setProperty('--board-height', `${canvasRect.height}px`);
    };

    const ro = new ResizeObserver(sync);
    ro.observe(canvas);
    ro.observe(area);
    area.addEventListener('scroll', sync, { passive: true });
    sync();

    return () => {
      ro.disconnect();
      area.removeEventListener('scroll', sync);
    };
  }, []);
}
