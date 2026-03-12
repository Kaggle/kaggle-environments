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
  const searches: { term: string; priority: number; text: string }[] = [
    {
      term: 'rethink',
      priority: 1,
      text: `${agent} rethinks their decision`,
    },
    {
      term: 'komi',
      priority: 2,
      text: `${agent} mentions Komi: ${komi} bonus points for going second`,
    },
    {
      term: 'monkey jump',
      priority: 1,
      text: `${agent} mentions Monkey Jump: reducing opposition territory at the boards edge`,
    },
    {
      term: 'ladder',
      priority: 1,
      text: `${agent} mentions Ladder: capturing stones in a zigzag pattern`,
    },
    {
      term: "tiger's mouth",
      priority: 1,
      text: `${agent} mentions Tiger's Mouth: a three-stone shape that creates a "trap"`,
    },
    {
      term: "horse's head",
      priority: 1,
      text: `${agent} mentions Horse's Head: an L shaped flexible attacking position`,
    },
    {
      term: 'wedge',
      priority: 1,
      text: `${agent} mentions Wedge: playing between two opponent stones`,
    },
    {
      term: "crane's nest",
      priority: 1,
      text: `${agent} mentions Crane's Nest: a group trap that resembles a "nest"`,
    },
    {
      term: 'three crows',
      priority: 1,
      text: `${agent} mentions Three Crow's: guarding a corner with three stones`,
    },
    {
      term: 'false eye',
      priority: 1,
      text: `${agent} mentions False Eye: a space that looks safe, but isn't`,
    },
    {
      term: 'clamp',
      priority: 1,
      text: `${agent} mentions Clamp: playing both sides of an opponent's stone`,
    },
    {
      term: 'iron pillar',
      priority: 1,
      text: `${agent} mentions Iron Pillar: a defensive, vertical two-stone tower`,
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
      text: `${agent} took over ${duration} minutes to decide`,
    });
  }

  if (matches.length > 0) {
    matches.sort((a, b) => b.priority - a.priority);
    console.log(matches[0].text);
  }

  return null;
});
