
import {Board} from "./Board";
import {Cell} from "./Cell";
import {Player} from "./Player";
import {Point} from "./Point";
import {ShipyardAction} from "./ShipyardAction";

const SPAWN_VALUES = [];
const upgradeTimes: number[] = [];
for(let i = 1; i < 10; i++) {
    upgradeTimes[i-1] = Math.pow(i, 2) + 1;
}
let current = 0;
for(let i = 1; i < 10; i++) {
    current += upgradeTimes[i-1];
    SPAWN_VALUES[i-1] = current;
}

export class Shipyard {
    public readonly id: string; 
    public shipCount: number; 
    public position: Point; 
    public playerId: number; 
    public turnsControlled: number; 
    public readonly board: Board; 
    public nextAction: ShipyardAction | undefined;
    
    public constructor(shipyardId: string, shipCount: number, position: Point, playerId: number, turnsControlled: number, board: Board, nextAction: ShipyardAction | undefined) {
        this.id = shipyardId;
        this.shipCount = shipCount;
        this.position = position;
        this.playerId = playerId;
        this.turnsControlled = turnsControlled;
        this.board = board;
        this.nextAction = nextAction;
    }

    public cloneToBoard(board: Board): Shipyard {
        return new Shipyard(this.id, this.shipCount, this.position, this.playerId, this.turnsControlled, board, this.nextAction);
    }

    public setNextAction(action: ShipyardAction): void {
        this.nextAction = action;
    }

    public get maxSpawn(): number {
        for (let i = 0; i < SPAWN_VALUES.length; i++) {
            if (this.turnsControlled < SPAWN_VALUES[i]) {
                return i + 1;
            }
        }
        return SPAWN_VALUES.length + 1;
    }

    /**
     *  Returns the cell this shipyard is on.
     */
    public get cell(): Cell {
        return this.board.getCellAtPosition(this.position);
    }

    public get player(): Player {
        return this.board.players[this.playerId];
    }
   
    /**
     * Converts a shipyard back to the normalized observation subset that constructed it.
     */
    public observation(): number[] {
        return [this.position.toIndex(this.board.configuration.size), this.shipCount, this.turnsControlled];
    }
}