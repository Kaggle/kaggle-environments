import useHeroAnimation from '../stores/useHeroAnimation';
import { RivePopover } from './RivePopover';

export function ActivePopover() {
  const src = useHeroAnimation((s) => s.src);
  const close = useHeroAnimation((s) => s.close);

  if (!src) return null;

  return <RivePopover key={src} src={src} onClose={close} />;
}
