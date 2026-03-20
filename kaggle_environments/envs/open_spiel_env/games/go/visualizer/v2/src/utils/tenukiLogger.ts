import { Game } from 'tenuki';

export function tenukiLogger(game: Game) {
  if (import.meta.env.DEV === false || !import.meta.env.VITE_TENUKI_LOGGER) return;

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

  const col = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't'];

  let output = '';

  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      if (x === 0) {
        output += `${size - y} `.padStart(3);
      }

      const color = state.intersectionAt(y, x).value!;

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

  output += '   ';
  for (let x = 0; x < size; x++) {
    output += col[x].padEnd(2);
  }

  console.log(output);
}
