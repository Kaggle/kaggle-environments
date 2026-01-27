export const ICON_PATHS = {
  play: 'M8 5v14l11-7z',
  pause: 'M6 19h4V5H6v14zm8-14v14h4V5h-4z',
  prev: 'M6 18V6h2v12H6zm3.5-6L18 6v12l-8.5-6z',
  next: 'M7 18l8.5-6L7 6v12zM15 6v12h2V6h-2z',
  restart: 'M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z',
} as const;

export type IconType = keyof typeof ICON_PATHS;
