import { useEffect } from 'react';
import useHeroAnimation from '../stores/useHeroAnimation';
import usePreferences from '../stores/usePreferences';
import { RivePopover } from './RivePopover';

export function ActivePopover() {
  const src = useHeroAnimation((s) => s.src);
  const close = useHeroAnimation((s) => s.close);
  const showAnimations = usePreferences((s) => s.showAnimations);

  useEffect(() => {
    if (src && !showAnimations) close();
  }, [src, showAnimations, close]);

  if (!src || !showAnimations) return null;

  return <RivePopover key={src} src={src} onClose={close} />;
}
