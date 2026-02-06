import { useEffect, useRef } from 'react';
import { create } from 'zustand';
import { Chess } from 'chess.js';
import { Chessboard, PieceRenderObject, defaultPieces } from 'react-chessboard';
import {
  createReplayVisualizer,
  processEpisodeData,
  LegacyAdapter,
  LegacyRendererOptions,
  ReplayData,
  ChessStep,
  ChessPlayer,
} from '@kaggle-environments/core';
import './App.css';

interface ChessStore {
  chess: Chess;
  setState: (options: LegacyRendererOptions<ChessStep[]>) => void;
}

const useChessStore = create<ChessStore>((set) => ({
  chess: new Chess(),

  setState: (options: LegacyRendererOptions<ChessStep[]>) => {
    const step = options.steps.at(options.step);
    const player = step!.players.find((element: ChessPlayer) => element.isTurn);

    if (player) {
      const history = options.replay.info!.stateHistory;
      const chess = new Chess(history.at(options.step));
      const move = chess.move(player.actionDisplayText!);

      set({ chess });

      console.log(player.thoughts);
      console.log(`${player.name}: ${move.piece} ${move.from} -> ${move.to}`);
    }
  },
}));

function App() {
  const { chess, setState } = useChessStore();
  const controlsRef = useRef(null);

  useEffect(() => {
    const app = controlsRef.current!;
    const adapter = new LegacyAdapter<ChessStep[]>(setState);
    const transformer = (replay: ReplayData) => processEpisodeData(replay, 'open_spiel_chess');

    createReplayVisualizer(app, adapter, { transformer });
  }, [setState]);

  const move = chess.history({ verbose: true })[0];

  const style = { width: '100%', height: '100%' };

  const pieces: PieceRenderObject = {
    ...defaultPieces,
    wP: () => <img src="images/wP.png" style={style} />,
    wK: () => <img src="images/wK.png" style={style} />,
    wQ: () => <img src="images/wQ.png" style={style} />,
    wR: () => <img src="images/wR.png" style={style} />,
    wB: () => <img src="images/wB.png" style={style} />,
    wN: () => <img src="images/wN.png" style={style} />,
    bP: () => <img src="images/bP.png" style={style} />,
    bK: () => <img src="images/bK.png" style={style} />,
    bQ: () => <img src="images/bQ.png" style={style} />,
    bR: () => <img src="images/bR.png" style={style} />,
    bB: () => <img src="images/bB.png" style={style} />,
    bN: () => <img src="images/bN.png" style={style} />,
  };

  const lightSquareStyle = {
    backgroundImage: 'url(/images/wBg.png)',
    backgroundPosition: 'center',
    backgroundSize: 'cover',
    backgroundRepeat: 'no-repeat',
  };

  const darkSquareStyle = {
    backgroundImage: 'url(/images/bBg.png)',
    backgroundPosition: 'center',
    backgroundSize: 'cover',
    backgroundRepeat: 'no-repeat',
  };

  console.log(pieces);

  return (
    <div className="container">
      <Chessboard options={{ position: chess.fen(), pieces, lightSquareStyle, darkSquareStyle }} />
      <div id="controls" ref={controlsRef} />
      <div id="moves">{move ? `${move.piece} ${move.from} → ${move.to}` : `Loading...`}</div>
    </div>
  );
}

export default App;
