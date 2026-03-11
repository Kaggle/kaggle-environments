import { memo } from 'react';
import useGameStore from '../stores/useGameStore';

export default memo(function Notation() {
  const game = useGameStore((state) => state.game);
  const options = useGameStore((state) => state.options);

  const step = options?.replay.steps.at(options.step);
  const player = step?.players.find((player) => player?.isTurn);

  if (!player || !player.generateReturns) return null;

  const komi = game._scorer._komi;
  const agent = player.name;
  const json = player.generateReturns[0];
  const data = JSON.parse(json);

  const thoughts = data.main_response;
  const searches: { term: string; priority: number; text: string }[] = [
    { term: 'rethink', priority: 1, text: `${agent} rethinks their decision` },
    { term: 'komi', priority: 2, text: `${agent} earns Komi: ${komi} bonus points for going second` },
  ];
  const matches = searches.filter((search) => thoughts.match(new RegExp(search.term, 'i')) !== null);

  const duration = 1;
  if (data.duration_success_only_secs > 60 * duration) {
    matches.push({ term: 'duration', priority: 1, text: `${agent} took over ${duration} minute to decide` });
  }

  if (matches.length > 0) {
    matches.sort((a, b) => b.priority - a.priority);
    console.log(matches[0].text);
  }

  return null;
});
