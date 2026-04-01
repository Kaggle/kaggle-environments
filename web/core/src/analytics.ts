/**
 * Send an analytics event to the parent frame via postMessage.
 *
 * The parent listens for messages with an `analyticsEvent` key
 * and can forward them to its own analytics pipeline.
 *
 * @param event - String or object describing the event.
 *   String example: "play_clicked"
 *   Object example: { game: "chess", action: "step_changed", step: 5 }
 */
export function postAnalyticsEvent(event: string | Record<string, unknown>): void {
  if (typeof window === 'undefined' || window.parent === window) return;
  window.parent.postMessage({ analyticsEvent: event }, '*');
}
