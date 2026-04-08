import useGameStore from '../stores/useGameStore';

export default function HiddenHeader() {
  const game = useGameStore((state) => state.game);
  const whiteName = game.getHeaders()['w'];
  const blackName = game.getHeaders()['b'];

  return (
    <h1 className="visually-hidden">
      {whiteName ?? 'White'} vs. {blackName ?? 'Black'}
    </h1>
  );
}
