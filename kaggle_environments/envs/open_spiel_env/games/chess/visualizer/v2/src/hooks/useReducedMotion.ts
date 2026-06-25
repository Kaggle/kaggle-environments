import type { Transition } from 'motion/react';
import usePreferences from '../stores/usePreferences';

const instant: Transition = { duration: 0 };

/** Returns `{ duration: 0 }` when reduced motion is on, otherwise the given transition. */
export function useTransition(transition: Transition): Transition {
  const reducedMotion = usePreferences((s) => s.reducedMotion);
  return reducedMotion ? instant : transition;
}
