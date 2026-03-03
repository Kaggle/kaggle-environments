import { Game } from 'tenuki';

export function tenukiToString(game: Game) {
  const state = game.currentState();
  const size = game.boardSize;

  const chars: { [param: string]: string } = {
    black: '●',
    white: '○',
    tl: '┌',
    tm: '┬',
    tr: '┐',
    ml: '├',
    mm: '┼',
    mr: '┤',
    bl: '└',
    bm: '┴',
    br: '┘',
    hl: '─',
    nl: '\n',
  };

  let output = '  ';

  for (let x = 0; x < size; x++) {
    output += `${x}`.padEnd(2);
  }

  output += chars.nl;

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (x === 0) {
        output += `${y} `.padStart(2);
      }

      const color = state.intersectionAt(x, y).value!;

      if (color !== 'empty') {
        output += chars[color];
      }

      if (color === 'empty') {
        let row = 'm';
        if (y === 0) row = 't';
        if (y === size - 1) row = 'b';

        let col = 'm';
        if (x === 0) col = 'l';
        if (x === size - 1) col = 'r';

        output += chars[row + col];
      }

      if (x < size - 1) {
        output += chars.hl;
      }
    }

    output += chars.nl;
  }

  return output;
}
