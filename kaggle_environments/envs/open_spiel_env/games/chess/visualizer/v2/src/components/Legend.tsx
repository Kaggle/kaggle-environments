import useGameStore from '../stores/useGameStore';

export default function Legend() {
  const game = useGameStore((state) => state.game);
  const move = game.history({ verbose: true }).at(0);
  const headers = game.getHeaders();

  return (
    <div id="legend">{move && `${headers[move.color]} (${move.color}): ${move.piece} ${move.from} → ${move.to}`}</div>
  );
}
