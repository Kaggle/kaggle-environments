import { useCallback } from 'react';
import { AnimatePresence, motion } from 'motion/react';
import { useTransition } from '../hooks/useReducedMotion.ts';
import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';
import styles from './Annotation.module.css';

type Notation = { term: string; title: string; text: string };

const LONG_THINKING_MINUTES = 5;

const THOUGHT_ANNOTATIONS: { term: string; label: string | null; description: string }[] = [
  { term: 'rethink', label: null, description: 'rethinks their decision.' },
  { term: 'monkey jump', label: 'Monkey Jump', description: `Reducing opposition territory at the boards edge.` },
  { term: 'ladder', label: 'Ladder', description: 'Capturing stones in a zigzag pattern.' },
  { term: "tiger's mouth", label: "Tiger's Mouth", description: 'A three-stone shape that creates a "trap".' },
  { term: "horse's head", label: "Horse's Head", description: 'An L shaped flexible attacking position.' },
  { term: 'wedge', label: 'Wedge', description: 'Playing between two opponent stones.' },
  { term: "crane's nest", label: "Crane's Nest", description: 'A group trap that resembles a "nest".' },
  { term: 'three crows', label: "Three Crow's", description: 'Guarding a corner with three stones.' },
  { term: 'false eye', label: 'False Eye', description: "A space that looks safe, but isn't." },
  { term: 'clamp', label: 'Clamp', description: "Playing both sides of an opponent's stone." },
  { term: 'iron pillar', label: 'Iron Pillar', description: 'A defensive, vertical two-stone tower.' },
  { term: 'snapback', label: 'Snapback', description: 'Sacrificing a stone to recapture several.' },
  { term: 'tortoise shell', label: 'Tortoise Shell', description: 'A powerful wall formed by two captures.' },
  {
    term: 'bamboo joint',
    label: 'Bamboo Joint',
    description: 'Two parallel pairs of stones in an unbreakable connection',
  },
  { term: 'flower', label: 'Flower', description: 'Diamond shape left after capturing one stone.' },
  { term: 'golden chicken', label: 'Golden Chicken', description: 'Neither player can move.' },
  { term: 'peep', label: 'Peep', description: 'A move that threatens to cut through enemy stones.' },
  { term: 'hane', label: 'Hane', description: '"Bending" a stone around an enemy stone.' },
  { term: 'nobi', label: 'Nobi', description: 'An extension move from your own stone.' },
  { term: 'kiri', label: 'Kiri', description: 'A move separating two enemy stones.' },
  { term: 'osae', label: 'Osae', description: 'Blocking the opponent from extending further.' },
  { term: 'seki', label: 'Seki', description: 'A non-capture stalemate.' },
  { term: 'shimari', label: 'Shimari', description: 'Two stones securing a corner area.' },
  { term: 'moyo', label: 'Moyo', description: 'A large, potential, unsecured territory.' },
  { term: 'dango', label: 'Dango', description: 'An inefficient group lacking the potential for safety.' },
  { term: 'akisankaku', label: 'Akisankaku', description: 'Three stones connected in an inefficient L-shape.' },
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
// 1. Move-specific (goes first / komi)
// 2. Thought keyword matches
// 3. Long thinking duration
function resolveAnnotation(
  player: { name: string; thoughts?: string; generateReturns?: string[] | null },
  moveNumber: number,
  komi: number
): Notation | null {
  const agent = player.name;

  if (moveNumber === 1) {
    return { term: 'goes first', title: `${agent} goes first:`, text: 'Unlike Chess, black plays first.' };
  }
  if (moveNumber === 2) {
    return { term: 'komi', title: `${agent} gets Komi:`, text: `A ${komi} point bonus for playing second.` };
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

  // React 18 doesn't support the `inert` HTML attribute as a prop, so we
  // set it imperatively via a ref callback. This can be replaced with a
  // regular `inert` prop once the project upgrades to React 19+.
  const inertRef = useCallback(
    (el: HTMLElement | null) => {
      if (!el) return;
      if (game.gameOver) {
        el.setAttribute('inert', '');
      } else {
        el.removeAttribute('inert');
      }
    },
    [game.gameOver]
  );

  const step = options?.replay.steps.at(options.step);
  const player = step?.players.find((player) => player?.isTurn);
  const moveNumber = game.moveNumber();
  const notation = showAnnotations && player ? resolveAnnotation(player, moveNumber, game._scorer._komi) : null;

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
            {moveNumber > 2 && <span className={styles.checkLog}>*Check Log*</span>}
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
