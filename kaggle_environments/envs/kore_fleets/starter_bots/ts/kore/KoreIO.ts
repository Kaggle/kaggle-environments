import readline from "readline";
import { Board } from "./Board";
import { ShipyardAction } from "./ShipyardAction";

export class KoreIO {
  public getLine: () => Promise<string>;

  public _setup(): void {

    // Prepare to read input
    const rl = readline.createInterface({
      input: process.stdin,
      output: null,
    });

    const buffer = [];
    let currentResolve: () => void;
    let currentPromise;
    const makePromise = function () {
      return new Promise<void>((resolve) => {
        currentResolve = resolve;
      });
    };
    // on each line, push line to buffer
    rl.on('line', (line) => {
      buffer.push(line);
      currentResolve();
      currentPromise = makePromise();
    });
    // The current promise for retrieving the next line
    currentPromise = makePromise()


    // with await, we pause process until there is input
    this.getLine = async () => {
      return new Promise(async (resolve) => {
        while (buffer.length === 0) {
          // pause while buffer is empty, continue if new line read
          await currentPromise;
        }
        // once buffer is not empty, resolve the most recent line in stdin, and remove it
        resolve(buffer.shift());
      });
    };
  }

  /**
   * Constructor for a new agent
   * User should edit this according to the `Design` this agent will compete under
   */
  public constructor() {
    this._setup(); // DO NOT REMOVE
  }


  public async run(loop: (board: Board) => Board): Promise<void> {
    while (true) {
      const rawObservation = await this.getLine();
      const rawConfiguration = await this.getLine();
      const board = Board.fromRaw(rawObservation, rawConfiguration);
      try {
        const nextBoard = loop(board);
        let actions = []
        board.currentPlayer.nextActions.forEach((action: ShipyardAction, id: string) => actions.push(`${id}:${action.toString()}`));
        console.log(actions.join(","));
      } catch (err) {
        console.log(err);
      }
    }
  }
}