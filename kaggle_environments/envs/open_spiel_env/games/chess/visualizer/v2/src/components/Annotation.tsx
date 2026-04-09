import { useCallback } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { useTransition } from '../hooks/useReducedMotion';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';
import styles from './Annotation.module.css';

type Notation = { term: string; title: string; text: string };

const LONG_THINKING_MINUTES = 5;

const THOUGHT_ANNOTATIONS: { term: string; label: string | null; description: string }[] = [
  { term: 'rethink', label: null, description: 'rethinks their decision.' },
  { term: "boden's mate", label: "Boden's mate", description: 'Checkmated by two criss-crossed bishops.' },
  { term: 'castling', label: 'Castling', description: 'Strategically moving a king and rook at the same time.' },
  {
    term: 'discovered attack',
    label: 'Discovered Attack',
    description: 'Attacking after a piece is moved out of the way.',
  },
  { term: 'gambit', label: 'Gambit', description: 'Sacrificing a piece for early game advantage.' },
  { term: 'interpose', label: 'Interpose', description: 'Moving a piece between an attacking piece and its target.' },
  { term: 'mating attack', label: 'Mating Attack', description: 'A move that aims to achieve checkmate.' },
  { term: 'royal fork', label: 'Royal Fork', description: 'A fork that threatens the king and queen.' },
  {
    term: 'stalemate',
    label: 'Stalemate',
    description: 'A draw when the active player has no legal moves and their king is not in check.',
  },
  { term: 'zugzwang', label: 'Zugzwang', description: 'Forced to move when any movement weakens your position.' },
];

function formatThoughtMatch(entry: (typeof THOUGHT_ANNOTATIONS)[number], agent: string): Notation {
  return {
    term: entry.term,
    title: entry.label ? `${agent} mentions ${entry.label}:` : '',
    text: entry.label ? entry.description : `${agent} ${entry.description}`,
  };
}

function getThinkingDurationMinutes(generateReturns: string[] | null | undefined): number | null {
  const json = generateReturns?.[0];
  if (!json) return null;
  try {
    return JSON.parse(json).duration_success_only_secs / 60;
  } catch {
    return null;
  }
}

// Resolves which annotation to show, in priority order:
// 1. Move-specific (goes first)
// 2. Thought keyword matches
// 3. Long thinking duration
function resolveAnnotation(
  player: { name: string; thoughts?: string; generateReturns?: string[] | null },
  moveNumber: number
): Notation | null {
  const agent = player.name;

  if (moveNumber === 1) {
    return { term: 'moves first', title: `${agent} moves first:`, text: 'In chess, white always goes first.' };
  }

  const thoughts = player.thoughts?.toLowerCase();
  if (thoughts) {
    const match = THOUGHT_ANNOTATIONS.find((a) => thoughts.includes(a.term));
    if (match) return formatThoughtMatch(match, agent);
  }

  const duration = getThinkingDurationMinutes(player.generateReturns);
  if (duration != null && duration > LONG_THINKING_MINUTES) {
    return { term: 'duration', title: '', text: `${agent} thought for over ${LONG_THINKING_MINUTES} minutes.` };
  }

  return null;
}

export default function Annotation() {
  const showAnnotations = usePreferences((state) => state.showAnnotations);
  const game = useGameStore((state) => state.game);
  const options = useGameStore((state) => state.options);
  const transition = useTransition({ duration: 0.35 });

  const gameOver = game.isGameOver();
  // React 18 doesn't support the `inert` HTML attribute as a prop, so we
  // set it imperatively via a ref callback. This can be replaced with a
  // regular `inert` prop once the project upgrades to React 19+.
  const inertRef = useCallback(
    (el: HTMLElement | null) => {
      if (!el) return;
      if (gameOver) {
        el.setAttribute('inert', '');
      } else {
        el.removeAttribute('inert');
      }
    },
    [gameOver]
  );

  const step = options?.replay.steps.at(options.step);
  const player = step?.players.find((player) => player?.isTurn);
  const moveNumber = game.moveNumber();
  const notation = showAnnotations && player ? resolveAnnotation(player, moveNumber) : null;

  return (
    <div className={styles.notationSlot} aria-live="polite" ref={inertRef}>
      <AnimatePresence mode="wait">
        {notation && (
          <motion.div
            key={notation.term}
            className={styles.notation}
            initial={{ opacity: 0, y: '20%' }}
            animate={{ opacity: 1, y: 0, rotate: -3 }}
            exit={{ opacity: 0, y: '-10%', rotate: -4 }}
            transition={transition}
          >
            {moveNumber > 1 && <span className={styles.checkLog}>*Check Log*</span>}
            {notation.title && (
              <h2>
                {notation.title.split(' ').map((word, i) => (
                  <span key={i}>{word} </span>
                ))}
              </h2>
            )}
            {notation.text && <p>{notation.text}</p>}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
