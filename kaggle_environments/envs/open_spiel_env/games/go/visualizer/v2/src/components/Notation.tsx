import { memo } from 'react';
import useGameStore from '../stores/useGameStore';

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
      text: `${agent} rethinks their decision`,
    },
    {
      term: 'komi',
      priority: 2,
      title: `${agent} mentions Komi`,
      text: `${komi} bonus points for going second`,
    },
    {
      term: 'monkey jump',
      priority: 1,
      title: `${agent} mentions Monkey Jump`,
      text: `Reducing opposition territory at the boards edge`,
    },
    {
      term: 'ladder',
      priority: 1,
      title: `${agent} mentions Ladder`,
      text: `Capturing stones in a zigzag pattern`,
    },
    {
      term: "tiger's mouth",
      priority: 1,
      title: `${agent} mentions Tiger's Mouth`,
      text: `A three-stone shape that creates a "trap"`,
    },
    {
      term: "horse's head",
      priority: 1,
      title: `${agent} mentions Horse's Head`,
      text: `An L shaped flexible attacking position`,
    },
    {
      term: 'wedge',
      priority: 1,
      title: `${agent} mentions Wedge`,
      text: `Playing between two opponent stones`,
    },
    {
      term: "crane's nest",
      priority: 1,
      title: `${agent} mentions Crane's Nest`,
      text: `A group trap that resembles a "nest"`,
    },
    {
      term: 'three crows',
      priority: 1,
      title: `${agent} mentions Three Crow's`,
      text: `Guarding a corner with three stones`,
    },
    {
      term: 'false eye',
      priority: 1,
      title: `${agent} mentions False Eye`,
      text: `A space that looks safe, but isn't`,
    },
    {
      term: 'clamp',
      priority: 1,
      title: `${agent} mentions Clamp`,
      text: `Playing both sides of an opponent's stone`,
    },
    {
      term: 'iron pillar',
      title: `${agent} mentions Iron Pillar`,
      priority: 1,
      text: `A defensive, vertical two-stone tower`,
    },
  ];
  const matches = searches.filter((search) => player.thoughts?.match(new RegExp(search.term, 'i')) !== null);

  const duration = 2;
  const json = player.generateReturns![0];
  const data = JSON.parse(json);
  if (data.duration_success_only_secs > 60 * duration) {
    matches.push({
      term: 'duration',
      priority: 1,
      title: '',
      text: `${agent} took over ${duration} minutes to decide`,
    });
  }

  if (matches.length > 0) {
    const notation = matches.sort((a, b) => b.priority - a.priority)[0];
    console.log(notation.title, notation.text);
  }

  return null;
});
