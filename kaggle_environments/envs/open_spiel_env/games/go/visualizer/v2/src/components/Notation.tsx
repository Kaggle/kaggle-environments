import { memo } from 'react';
import useGameStore from '../stores/useGameStore';
import styles from './Notation.module.css';

export default memo(function Notation() {
  const game = useGameStore((state) => state.game);
  const options = useGameStore((state) => state.options);

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

  return (
    <div className={styles.notation}>
      {notation.title && (
        <h2>
          {notation.title.split(' ').map((word, i) => (
            <span key={i}>{word} </span>
          ))}
        </h2>
      )}
      {notation.text && <p>{notation.text}</p>}
    </div>
  );
});
