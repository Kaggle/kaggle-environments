import { Chessboard } from 'react-chessboard';
import useGameStore from '../stores/useGameStore';

export default function GameBoard() {
  const game = useGameStore((state) => state.game);
  const position = game.fen();

  const style = { width: '100%', height: '100%' };
  const pieces = {
    wP: () => <img src="./react-chessboard/wP.png" style={style} />,
    wK: () => <img src="./react-chessboard/wK.png" style={style} />,
    wQ: () => <img src="./react-chessboard/wQ.png" style={style} />,
    wR: () => <img src="./react-chessboard/wR.png" style={style} />,
    wB: () => <img src="./react-chessboard/wB.png" style={style} />,
    wN: () => <img src="./react-chessboard/wN.png" style={style} />,
    bP: () => <img src="./react-chessboard/bP.png" style={style} />,
    bK: () => <img src="./react-chessboard/bK.png" style={style} />,
    bQ: () => <img src="./react-chessboard/bQ.png" style={style} />,
    bR: () => <img src="./react-chessboard/bR.png" style={style} />,
    bB: () => <img src="./react-chessboard/bB.png" style={style} />,
    bN: () => <img src="./react-chessboard/bN.png" style={style} />,
  };

  const lightSquareStyle = {
    // backgroundImage: 'url(./react-chessboard/wBg.png)',
    backgroundPosition: 'center',
    backgroundSize: 'cover',
    backgroundRepeat: 'no-repeat',
    backgroundColor: 'transparent',
  };

  const darkSquareStyle = {
    backgroundImage: 'url(./react-chessboard/bBg.png)',
    backgroundPosition: 'center',
    backgroundSize: 'cover',
    backgroundRepeat: 'no-repeat',
    backgroundColor: 'transparent',
  };

  return (
    <div id="board" style={{ padding: '2em' }}>
      <Chessboard
        position={position}
        customDarkSquareStyle={darkSquareStyle}
        customLightSquareStyle={lightSquareStyle}
        customPieces={pieces}
      />
    </div>
  );
}
