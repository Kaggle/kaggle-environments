import { useEffect, useRef, useState } from 'react';
import passRiv from '../assets/pass.riv?url';
import doublePassRiv from '../assets/double-pass.riv?url';
import firstCaptureRiv from '../assets/first-capture.riv?url';
import criticalHitRiv from '../assets/critical-hit.riv?url';
import dragonLossRiv from '../assets/dragon-loss.riv?url';
import { HeroTypes, detectHeroType } from '../utils/heroTypes.ts';
import { RivePopover } from './RivePopover.tsx';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';

interface Hero { 
  src: string;
  text: string;
  color: string;
  step: number;
}

export default function HeroAnimation() {
  const game = useGameStore((state) => state.game);
  const showHeroAnimations = usePreferences((state) => state.showHeroAnimations);
  const reducedMotion = usePreferences((state) => state.reducedMotion);
  const prevStepRef = useRef<number | null>(null);
  const [hero, setHero] = useState<Hero | null>(null);

  useEffect(() => {
    const step = game.step ?? null;
    const prevStep = prevStepRef.current;
    prevStepRef.current = step;

    // Only trigger on single-step navigation
    if (!prevStep || !step || Math.abs(step - prevStep) > 1) return;
    if (!showHeroAnimations || reducedMotion) return;

    const heroType = detectHeroType(game);
    if (heroType === null) return;

    const state = game.currentState();
    const color = state.color;
    const player = color === 'black' ? 'Black' : 'White';
    const opponent = color === 'black' ? 'White' : 'Black';
    const captures = state.capturedPositions?.length;

    let src, text;
    switch (heroType) {
      case HeroTypes.PASS:
        src = passRiv;
        text = `${player} passes the turn.`;
        break;
      case HeroTypes.DOUBLE_PASS:
        src = doublePassRiv;
        text = 'Double Pass: game over.';
        break;
      case HeroTypes.FIRST_CAPTURE:
        src = firstCaptureRiv;
        text = `${player} captures first.`;
        break;
      case HeroTypes.CRITICAL_HIT:
        src = criticalHitRiv;
        text = `${player} takes ${captures} pieces.`;
        break;
      case HeroTypes.DRAGON_LOSS:
        src = dragonLossRiv;
        text = `${opponent} loses a dragon.`;
        break;
      default:
        return;
    }

    // Let the board play out before showing the Rive animation.
    const timeout = setTimeout(() => {
      setHero({ src, text, color, step });
    }, 600);

    return () => { 
      clearTimeout(timeout);
      setHero(null)
    };
  }, [game, showHeroAnimations, reducedMotion]);

  if (!hero) return null;
  if (game.gameOver) return null;

  return (
    <RivePopover key={hero.step} src={hero.src} text={hero.text} color={hero.color} onClose={() => setHero(null)} />
  );
}
