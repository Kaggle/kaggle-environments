import {Board} from "./Board";
import {Cell} from "./Cell";
import {Direction} from "./Direction";
import {Player} from "./Player";
import {Point} from "./Point";

export class Fleet {

    public readonly id: string;
    public shipCount: number;
    public direction: Direction;
    public position: Point;
    public flightPlan: string;
    public kore: number;
    public readonly playerId: number;
    public readonly board: Board;
    
    public constructor(fleetId: string, shipCount: number, direction: Direction, position: Point, kore: number, flightPlan: string, playerId: number, board: Board) {
        this.id = fleetId;
        this.shipCount = shipCount;
        this.direction = direction;
        this.position = position;
        this.flightPlan = flightPlan;
        this.kore = kore;
        this.playerId = playerId;
        this.board = board;
    }

    public cloneToBoard(board: Board): Fleet {
        return new Fleet(this.id, this.shipCount, this.direction, this.position, this.kore, this.flightPlan, this.playerId, board);
    }

    public get cell(): Cell {
        return this.board.getCellAtPosition(this.position);
    }

    public get player(): Player {
        return this.board.players[this.playerId];
    }

    public get collectionRate(): number {
        return Math.min(Math.log(this.shipCount) / 10, .99);
    }

    /**
     * Returns the length of the longest possible flight plan this fleet can be assigned
     * @return
     */
    public static maxFlightPlanLenForShipCount(shipCount: number): number {
        return (Math.floor(2 * Math.log(shipCount)) + 1);
    }

    /**
     * Converts a fleet back to the normalized observation subset that constructed it.
     */
    public observation(): string[] {
        return [
            this.position.toIndex(this.board.configuration.size).toString(), 
            this.kore.toString(),
            this.shipCount.toString(),
            this.direction.toIndex().toString(),
            this.flightPlan
        ];
    }

    public lessThanOtherAlliedFleet(other: Fleet): boolean {
        if (this.shipCount != other.shipCount) {
            return this.shipCount < other.shipCount;
        }
        if (this.kore != other.kore) {
            return this.kore < other.kore;
        }
        return this.direction.toIndex() > other.direction.toIndex();
}

    }
