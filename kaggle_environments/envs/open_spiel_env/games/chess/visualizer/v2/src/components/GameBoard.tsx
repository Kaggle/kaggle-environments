import { Chessboard } from 'react-chessboard';
import bishopBlackPath from '../assets/images/bishop-b-small.webp';
import bishopWhitePath from '../assets/images/bishop-w-small.webp';
import kingBlackPath from '../assets/images/king-b-small.webp';
import kingWhitePath from '../assets/images/king-w-small.webp';
import knightBlackPath from '../assets/images/knight-b-small.webp';
import knightWhitePath from '../assets/images/knight-w-small.webp';
import pawnBlackPath from '../assets/images/pawn-b-small.webp';
import pawnWhitePath from '../assets/images/pawn-w-small.webp';
import queenBlackPath from '../assets/images/queen-b-small.webp';
import queenWhitePath from '../assets/images/queen-w-small.webp';
import rookBlackPath from '../assets/images/rook-b-small.webp';
import rookWhitePath from '../assets/images/rook-w-small.webp';
import useGameStore from '../stores/useGameStore';

export default function GameBoard() {
  const game = useGameStore((state) => state.game);
  const position = game.fen();

  const style = { width: '100%', height: '100%' };
  const pieces = {
    wP: () => <img src={pawnWhitePath} style={style} />,
    wK: () => <img src={kingWhitePath} style={style} />,
    wQ: () => <img src={queenWhitePath} style={style} />,
    wR: () => <img src={rookWhitePath} style={style} />,
    wB: () => <img src={bishopWhitePath} style={style} />,
    wN: () => <img src={knightWhitePath} style={style} />,
    bP: () => <img src={pawnBlackPath} style={style} />,
    bK: () => <img src={kingBlackPath} style={style} />,
    bQ: () => <img src={queenBlackPath} style={style} />,
    bR: () => <img src={rookBlackPath} style={style} />,
    bB: () => <img src={bishopBlackPath} style={style} />,
    bN: () => <img src={knightBlackPath} style={style} />,
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
    <div id="board" style={{ padding: '0.5em' }}>
      <Chessboard
        position={position}
        customDarkSquareStyle={darkSquareStyle}
        customLightSquareStyle={lightSquareStyle}
        customPieces={pieces}
      />
    </div>
  );
}
