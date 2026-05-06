import useGameStore from '../stores/useGameStore';

export default function HiddenHeader() {
  const game = useGameStore((state) => state.game);

  return (
    <h1 className="visually-hidden">
      {game.blackName ?? 'Black'} vs. {game.whiteName ?? 'White'}
    </h1>
  );
}
