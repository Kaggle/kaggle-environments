import useChessStore from '../stores/useChessStore';

const Legend = () => {
  const chess = useChessStore((state) => state.chess);
  const move = chess.history({ verbose: true }).at(0);
  const headers = chess.getHeaders();

  return <div id="legend">{move && `${headers.name} (${move.color}): ${move.piece} ${move.from} → ${move.to}`}</div>;
};

export default Legend;
