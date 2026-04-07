import { postAnalyticsEvent } from '../../../../../../../../../web/core/src/analytics';

export function trackEvent(event: string) {
  if (import.meta.env.DEV && import.meta.env.VITE_LOG_ANALYTICS) {
    console.log(`Track Event: ${event}`);
  }

  postAnalyticsEvent(event);
}
