import { postAnalyticsEvent } from '@kaggle-environments/core';

export const GAME_NAME = 'open_spiel_go';

export function trackEvent(event: string) {
  if (import.meta.env.DEV && import.meta.env.VITE_LOG_ANALYTICS) {
    console.log(`Track Event: ${event}`);
  }

  postAnalyticsEvent({ game: GAME_NAME, action: event });
}
