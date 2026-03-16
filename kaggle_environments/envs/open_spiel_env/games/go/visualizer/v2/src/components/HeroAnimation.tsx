import { useEffect } from 'react';
import useHeroAnimation from '../stores/useHeroAnimation';
import usePreferences from '../stores/usePreferences';
import { RivePopover } from './RivePopover';

export function HeroAnimation() {
  const src = useHeroAnimation((s) => s.src);
  const close = useHeroAnimation((s) => s.close);
  const showHeroAnimations = usePreferences((s) => s.showHeroAnimations);

  useEffect(() => {
    if (src && !showHeroAnimations) close();
  }, [src, showHeroAnimations, close]);

  if (!src || !showHeroAnimations) return null;

  return <RivePopover key={src} src={src} onClose={close} />;
}
