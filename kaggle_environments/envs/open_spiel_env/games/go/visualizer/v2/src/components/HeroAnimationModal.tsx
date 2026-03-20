import { memo, useEffect, useRef, useState } from 'react';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';
import { HeroTypes, detectHeroType } from '../utils/heroTypes.ts';
import { RivePopover } from './RivePopover.tsx';
import passRiv from '../assets/pass.riv?url';
import doublePassRiv from '../assets/double-pass.riv?url';

export default memo(function HeroAnimationModal() {
  const game = useGameStore((state) => state.game);
  const showHeroAnimations = usePreferences((s) => s.showHeroAnimations);
  const reducedMotion = usePreferences((s) => s.reducedMotion);

  const prevStepRef = useRef<number | null>(null);
  const [hero, setHero] = useState<{ src: string; text: string; color: string; step: number } | null>(null);

  useEffect(() => {
    const step = game.currentState().moveNumber;
    const prevStep = prevStepRef.current;
    prevStepRef.current = step;

    // Clear any existing hero on every step change
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setHero(null);

    // Only trigger on single-step navigation
    if (prevStep === null || Math.abs(step - prevStep) !== 1) return;
    if (!showHeroAnimations || reducedMotion) return;

    const heroType = detectHeroType(game);
    if (heroType === null) return;

    const color = game.currentState().color;
    const player = color.charAt(0).toUpperCase() + color.slice(1);
    const captures = game.currentState().capturedPositions?.length;

    const RIVE_MAP = {
      [HeroTypes.PASS]: { src: passRiv, text: `${player} passes the turn.` },
      [HeroTypes.DOUBLE_PASS]: { src: doublePassRiv, text: 'Double Pass: game over.' },
      [HeroTypes.FIRST_CAPTURE]: { src: passRiv, text: `${player} captures first.` },
      [HeroTypes.CRITICAL_HIT]: { src: passRiv, text: `${player} takes ${captures} pieces.` },
      [HeroTypes.DRAGON_LOSS]: { src: passRiv, text: 'Dragon was lost.' },
    };

    // Let the board play out before showing the Rive animation.
    const timeout = setTimeout(() => {
      setHero({ src: RIVE_MAP[heroType].src, text: RIVE_MAP[heroType].text, color, step });
    }, 600);

    return () => clearTimeout(timeout);
  }, [game, showHeroAnimations, reducedMotion]);

  if (!hero) return null;

  return (
    <RivePopover key={hero.step} src={hero.src} text={hero.text} color={hero.color} onClose={() => setHero(null)} />
  );
});
