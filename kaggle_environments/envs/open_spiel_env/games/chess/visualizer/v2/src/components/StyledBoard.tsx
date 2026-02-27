import { Chessboard } from 'react-chessboard';
import useGameStore from '../stores/useGameStore';

export default function StyledBoard() {
  const game = useGameStore((state) => state.game);
  const position = game.fen();

  const style = { width: '100%', height: '100%' };
  const pieces = {
    wP: () => <img src="./images/wP.png" style={style} />,
    wK: () => <img src="./images/wK.png" style={style} />,
    wQ: () => <img src="./images/wQ.png" style={style} />,
    wR: () => <img src="./images/wR.png" style={style} />,
    wB: () => <img src="./images/wB.png" style={style} />,
    wN: () => <img src="./images/wN.png" style={style} />,
    bP: () => <img src="./images/bP.png" style={style} />,
    bK: () => <img src="./images/bK.png" style={style} />,
    bQ: () => <img src="./images/bQ.png" style={style} />,
    bR: () => <img src="./images/bR.png" style={style} />,
    bB: () => <img src="./images/bB.png" style={style} />,
    bN: () => <img src="./images/bN.png" style={style} />,
  };

  const lightSquareStyle = {
    backgroundImage: 'url(./images/wBg.png)',
    backgroundPosition: 'center',
    backgroundSize: 'cover',
    backgroundRepeat: 'no-repeat',
    backgroundColor: 'transparent',
  };

  const darkSquareStyle = {
    backgroundImage: 'url(./images/bBg.png)',
    backgroundPosition: 'center',
    backgroundSize: 'cover',
    backgroundRepeat: 'no-repeat',
    backgroundColor: 'transparent',
  };

  return (
    <Chessboard
      position={position}
      customDarkSquareStyle={darkSquareStyle}
      customLightSquareStyle={lightSquareStyle}
      customPieces={pieces}
    />
  );
}
