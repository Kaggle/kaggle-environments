import useChessStore from '../stores/useChessStore';

export default function Legend() {
  const chess = useChessStore((state) => state.chess);

  if (chess.isGameOver()) {
    console.log('Game Over');
  }

  return null;
}
