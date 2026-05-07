import { ReplayData } from '@kaggle-environments/core';
import { GoStep } from '../transformers/goReplayTypes';

export function replayToSgf(replayData: ReplayData<GoStep[]>): string {
  const blackPlayer = replayData.info?.TeamNames.at(0) || 'Black';
  const whitePlayer = replayData.info?.TeamNames.at(1) || 'White';
  const stateHistory = replayData.info?.stateHistory || [];

  let sgf = '(;';

  for (const [index, item] of stateHistory.entries()) {
    const state = JSON.parse(item);
    const boardSize = state.board_size;
    const komi = state.komi;

    if (index === 0) {
      sgf += 'GM[1]';
      sgf += 'FF[4]';
      sgf += 'CA[UTF-8]';
      sgf += `PB[${blackPlayer}]`;
      sgf += `PW[${whitePlayer}]`;
      sgf += `SZ[${boardSize}]`;
      sgf += `KM[${komi}]`;
    }

    const player = state.previous_move_a1?.split(' ').at(0);
    const move = state.previous_move_a1?.split(' ').at(1);

    if (player && move === 'PASS') {
      sgf += `;${player}[]`;
    } else if (player && move) {
      const y = boardSize - parseInt(move.slice(1));
      const x = 'abcdefghjklmnopqrst'.indexOf(move.charAt(0));
      const sgfLetters = 'abcdefghijklmnopqrs';
      const coordinates = sgfLetters[x] + sgfLetters[y];
      sgf += `;${player}[${coordinates}]`;
    }
  }

  sgf += ')';

  return sgf;
}

export function downloadSgf(replayData: ReplayData<GoStep[]>): void {
  const episode = replayData.info?.EpisodeId || '00000000';
  const blackPlayer = replayData.info?.TeamNames.at(0) || 'Black';
  const whitePlayer = replayData.info?.TeamNames.at(1) || 'White';
  const sgfContent = replayToSgf(replayData);
  const blob = new Blob([sgfContent], { type: 'application/x-go-sgf' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `${episode}-${blackPlayer}-${whitePlayer}.sgf`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}
