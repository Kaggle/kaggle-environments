import { useEffect, useRef, useState } from 'react';
import { AnimatePresence } from 'motion/react';
import checkmateRiv from '../assets/rives/checkmate.riv?url';
import queenLossRiv from '../assets/rives/queen-loss.riv?url';
import promotionRiv from '../assets/rives/promotion.riv?url';
import castlingRiv from '../assets/rives/castling.riv?url';
import firstCaptureRiv from '../assets/rives/first-capture.riv?url';
import { HeroTypes, detectHeroType } from '../utils/heroTypes';
import { RivePopover } from './RivePopover';
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
  const options = useGameStore((state) => state.options);
  const showHeroAnimations = usePreferences((state) => state.showHeroAnimations);
  const prevStepRef = useRef<number | null>(null);
  const [hero, setHero] = useState<Hero | null>(null);

  useEffect(() => {
    const step = game.moveNumber() ?? null;
    const prevStep = prevStepRef.current;
    prevStepRef.current = step;

    // Only trigger on single-step navigation
    if (!prevStep || !step || Math.abs(step - prevStep) > 1) return;
    if (!showHeroAnimations) return;

    const heroType = detectHeroType(game);
    if (heroType === null) return;

    const color = game.turn() === 'b' ? 'black' : 'white';
    const player = color === 'black' ? 'Black' : 'White';
    const opponent = color === 'black' ? 'White' : 'Black';

    let src, text;
    switch (heroType) {
      case HeroTypes.CHECKMATE:
        src = checkmateRiv;
        text = 'Checkmate';
        break;
      case HeroTypes.QUEEN_LOSS:
        src = queenLossRiv;
        text = `${opponent} Loses their queen`;
        break;
      case HeroTypes.PROMOTION:
        src = promotionRiv;
        text = `${player} Queens`;
        break;
      case HeroTypes.CASTLING:
        src = castlingRiv;
        text = `${player} Castles`;
        break;
      case HeroTypes.FIRST_CAPTURE:
        src = firstCaptureRiv;
        text = `${player} Captures First`;
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
      setHero(null);
    };
  }, [game, showHeroAnimations]);

  const isVisible = !!hero && !options.replay.steps.at(options.step)?.winner;

  return (
    <AnimatePresence>
      {isVisible && (
        <RivePopover key={hero.step} src={hero.src} text={hero.text} color={hero.color} onClose={() => setHero(null)} />
      )}
    </AnimatePresence>
  );
}
