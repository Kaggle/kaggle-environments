import {Board} from "./Board";
import {Direction} from "./Direction";
import {Fleet} from "./Fleet";
import {Point} from "./Point";
import {Shipyard} from "./Shipyard";

export class Cell {

    public readonly position: Point;
    public kore: number;
    public shipyardId: string;
    public fleetId: string;
    public readonly board: Board;
    
    public constructor(position: Point, kore: number, shipyardId: string, fleetId: string, board: Board) {
        this.position = position;
        this.kore = kore;
        this.shipyardId = shipyardId;
        this.fleetId = fleetId;
        this.board = board;
    }

    public cloneToBoard(board: Board): Cell {
        return new Cell(this.position, this.kore, this.shipyardId, this.fleetId, board);
    }

    public get fleet(): Fleet | undefined {
        if (this.board.fleets.has(this.fleetId)) {
            return this.board.fleets.get(this.fleetId);
        }
        return undefined;
    }


    public get shipyard(): Shipyard | undefined {
        if (this.board.shipyards.has(this.shipyardId)) {
            return this.board.shipyards.get(this.shipyardId);
        }
        return undefined;
    }

    public neighbor(offset: Point): Cell {
        const next = this.position.translate(offset, this.board.size);
        return this.board.getCellAtPosition(next);
    }


    public north(): Cell {
        return this.neighbor(Direction.NORTH);
    }

    public south(): Cell {
        return this.neighbor((Direction.SOUTH));
    }

    public east(): Cell {
        return this.neighbor(Direction.EAST);
    }

    public west(): Cell {
        return this.neighbor(Direction.WEST);
    }
}
