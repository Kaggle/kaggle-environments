import useGameStore from '../stores/useGameStore';
import usePreferences from '../stores/usePreferences';
import styles from './Annotation.module.css';

export default function Annotation() {
  console.log("Annotation");
  const showAnnotations = usePreferences((state) => state.showAnnotations);
  const game = useGameStore((state) => state.game);
  const options = useGameStore((state) => state.options);

  if (showAnnotations === false) return null;

  const step = options?.replay.steps.at(options.step);
  const player = step?.players.find((player) => player?.isTurn);

  if (!player) return null;

  const komi = game._scorer._komi;
  const agent = player.name;
  const searches: { term: string; priority: number; title: string; text: string }[] = [
    {
      term: 'rethink',
      priority: 1,
      title: '',
      text: `${agent} rethinks their decision.`,
    },
    {
      term: 'monkey jump',
      priority: 1,
      title: `${agent} mentions Monkey Jump:`,
      text: `Reducing opposition territory at the boards edge.`,
    },
    {
      term: 'ladder',
      priority: 1,
      title: `${agent} mentions Ladder:`,
      text: `Capturing stones in a zigzag pattern.`,
    },
    {
      term: "tiger's mouth",
      priority: 1,
      title: `${agent} mentions Tiger's Mouth:`,
      text: `A three-stone shape that creates a "trap".`,
    },
    {
      term: "horse's head",
      priority: 1,
      title: `${agent} mentions Horse's Head:`,
      text: `An L shaped flexible attacking position.`,
    },
    {
      term: 'wedge',
      priority: 1,
      title: `${agent} mentions Wedge:`,
      text: `Playing between two opponent stones.`,
    },
    {
      term: "crane's nest",
      priority: 1,
      title: `${agent} mentions Crane's Nest:`,
      text: `A group trap that resembles a "nest".`,
    },
    {
      term: 'three crows',
      priority: 1,
      title: `${agent} mentions Three Crow's:`,
      text: `Guarding a corner with three stones.`,
    },
    {
      term: 'false eye',
      priority: 1,
      title: `${agent} mentions False Eye:`,
      text: `A space that looks safe, but isn't.`,
    },
    {
      term: 'clamp',
      priority: 1,
      title: `${agent} mentions Clamp:`,
      text: `Playing both sides of an opponent's stone.`,
    },
    {
      term: 'iron pillar',
      title: `${agent} mentions Iron Pillar:`,
      priority: 1,
      text: `A defensive, vertical two-stone tower.`,
    },
    {
      term: 'snapback',
      title: `${agent} mentions Snapback:`,
      priority: 1,
      text: `Sacrificing a stone to recapture several.`,
    },
    {
      term: 'tortoise shell',
      title: `${agent} mentions Tortoise Shell:`,
      priority: 1,
      text: `A powerful wall formed by two captures.`,
    },
    {
      term: 'bamboo joint',
      title: `${agent} mentions Bamboo Joint:`,
      priority: 1,
      text: `Two parallel pairs of stones in an unbreakable connection`,
    },
    {
      term: 'flower',
      title: `${agent} mentions Flower:`,
      priority: 1,
      text: `Diamond shape left after capturing one stone.`,
    },
    {
      term: 'golden chicken',
      title: `${agent} mentions Golden Chicken:`,
      priority: 1,
      text: `Neither player can move.`,
    },
    {
      term: 'peep',
      title: `${agent} mentions Peep:`,
      priority: 1,
      text: `A move that threatens to cut through enemy stones.`,
    },
    {
      term: 'hane',
      title: `${agent} mentions Hane:`,
      priority: 1,
      text: `"Bending" a stone around an enemy stone.`,
    },
    {
      term: 'nobi',
      title: `${agent} mentions Nobi:`,
      priority: 1,
      text: `An extension move from your own stone.`,
    },
    {
      term: 'kiri',
      title: `${agent} mentions Kiri:`,
      priority: 1,
      text: `A move separating two enemy stones.`,
    },
    {
      term: 'osae',
      title: `${agent} mentions Osae:`,
      priority: 1,
      text: `Blocking the opponent from extending further.`,
    },
    {
      term: 'seki',
      title: `${agent} mentions Seki:`,
      priority: 1,
      text: `A non-capture stalemate.`,
    },
    {
      term: 'shimari',
      title: `${agent} mentions Shimari:`,
      priority: 1,
      text: `Two stones securing a corner area.`,
    },
    {
      term: 'moyo',
      title: `${agent} mentions Moyo:`,
      priority: 1,
      text: `A large, potential, unsecured territory.`,
    },
    {
      term: 'dango',
      title: `${agent} mentions Dango:`,
      priority: 1,
      text: `An inefficient group lacking the potential for safety.`,
    },
    {
      term: 'akisankaku',
      title: `${agent} mentions Akisankaku:`,
      priority: 1,
      text: `Three stones connected in an inefficient L-shape.`,
    },
  ];
  const matches = searches.filter((search) => player.thoughts?.toLowerCase().includes(search.term));
  const json = player.generateReturns?.[0];
  if (json) {
    const data = JSON.parse(json);
    const duration = 5;
    if (data.duration_success_only_secs > 60 * duration) {
      matches.push({
        term: 'duration',
        priority: 2,
        title: '',
        text: `${agent} thought for over ${duration} minutes.`,
      });
    }
  }
  if (game.moveNumber() === 1) {
    matches.push({
      term: 'goes first',
      priority: 0,
      title: `${agent} goes first:`,
      text: `Unlike Chess, black plays first.`,
    });
  }
  if (game.moveNumber() === 2) {
    matches.push({
      term: 'komi',
      priority: 0,
      title: `${agent} gets Komi:`,
      text: `A ${komi} point bonus for playing second.`,
    });
  }

  if (matches.length === 0) return null;

  const notation = matches.toSorted((a, b) => a.priority - b.priority)[0];

  const gameOver = options.replay.steps.at(options.step)?.winner;
  // React 18 doesn't support the `inert` HTML attribute as a prop, so we
  // set it imperatively via a ref callback. This can be replaced with a
  // regular `inert` prop once the project upgrades to React 19+.
  const inertRef = (el: HTMLElement | null) => {
    if (!el) return;
    if (gameOver) el.setAttribute('inert', '');
    else el.removeAttribute('inert');
  };

  return (
    <div className={styles.notationSlot} aria-live="polite" ref={inertRef}>
      <div className={styles.notation}>
        <span className={styles.checkLog}>*Check Log*</span>
        {notation.title && (
          <h2>
            {notation.title.split(' ').map((word, i) => (
              <span key={i}>{word} </span>
            ))}
          </h2>
        )}
        {notation.text && <p>{notation.text}</p>}
      </div>
    </div>
  );
}
