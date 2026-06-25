import useGameStore from '../stores/useGameStore';

export default function HiddenHeader() {
  const game = useGameStore((state) => state.game);
  const headers = game.getHeaders();
  const whiteName = headers['w'];
  const blackName = headers['b'];

  return (
    <h1 className="visually-hidden">
      {whiteName ?? 'White'} vs. {blackName ?? 'Black'}
    </h1>
  );
}
