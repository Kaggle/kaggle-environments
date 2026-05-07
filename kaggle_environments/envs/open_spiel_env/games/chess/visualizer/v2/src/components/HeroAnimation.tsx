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
import { trackEvent } from '../utils/analytics';

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

    let color = game.turn() === 'w' ? 'black' : 'white';
    const player = color === 'black' ? 'Black' : 'White';
    const opponent = color === 'black' ? 'White' : 'Black';

    let src, text, event;
    switch (heroType) {
      case HeroTypes.CHECKMATE:
        src = checkmateRiv;
        text = 'Checkmate';
        event = 'checkmate';
        break;
      case HeroTypes.QUEEN_LOSS:
        src = queenLossRiv;
        text = `${opponent} Loses their queen`;
        event = 'queen-loss';
        // Invert color when on queen loss.
        // TODO: This should be switched in the Rive file instead.
        color = color === 'white' ? 'black' : 'white';
        break;
      case HeroTypes.PROMOTION:
        src = promotionRiv;
        text = `${player} Queens`;
        event = 'promotion';
        break;
      case HeroTypes.CASTLING:
        src = castlingRiv;
        text = `${player} Castles`;
        event = 'castling';
        break;
      case HeroTypes.FIRST_CAPTURE:
        src = firstCaptureRiv;
        text = `${player} Captures First`;
        event = 'first-capture';
        break;
      default:
        return;
    }

    // Let the board play out before showing the Rive animation.
    const timeout = setTimeout(() => {
      setHero({ src, text, color, step });
      if (!options.replay.steps.at(options.step)?.winner) trackEvent(`pop-up-animation-${event}`);
    }, 600);

    return () => {
      clearTimeout(timeout);
      setHero(null);
    };
  }, [game, showHeroAnimations, options]);

  const isVisible = !!hero && !options.replay.steps.at(options.step)?.winner;

  return (
    <AnimatePresence>
      {isVisible && (
        <RivePopover key={hero.step} src={hero.src} text={hero.text} color={hero.color} onClose={() => setHero(null)} />
      )}
    </AnimatePresence>
  );
}
