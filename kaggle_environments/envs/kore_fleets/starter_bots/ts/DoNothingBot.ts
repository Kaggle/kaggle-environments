import {Board} from "./kore/Board";
import { Direction } from "./kore/Direction";
import { ShipyardAction } from "./kore/ShipyardAction";
import { KoreIO } from "./kore/KoreIO";


export const tick = async (board: Board): Promise<Board> => {
    return board;
}

const io = new KoreIO();
io.run(tick);
