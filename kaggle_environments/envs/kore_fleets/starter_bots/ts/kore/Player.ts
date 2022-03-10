import {Board} from "./Board";
import {Fleet} from "./Fleet";
import {Shipyard} from "./Shipyard";
import {ShipyardAction} from "./ShipyardAction";

export class Player {
    public readonly id: number; 
    public kore: number; 
    public readonly shipyardIds: string[]; 
    public readonly fleetIds: string[];
    public readonly board: Board;
    
    public constructor(playerId: number, kore: number, shipyardIds: string[], fleetIds: string[], board: Board) {
        this.id = playerId;
        this.kore = kore;
        this.shipyardIds = shipyardIds;
        this.fleetIds = fleetIds;
        this.board = board;
    }

    public cloneToBoard(board: Board): Player {
        return new Player(this.id, this.kore, this.shipyardIds.slice(), this.fleetIds.slice(), board);
    }

    /**
     * Returns all shipyards owned by this player.
     * @return
     */
    public get shipyards(): Shipyard[] {
        return Array.from(this.board.shipyards.values())
            .filter(shipyard => this.shipyardIds.some(sId => sId == shipyard.id));
    }

    /**
     * Returns all fleets owned by this player.
     */
    public get fleets(): Fleet[] {
        return Array.from(this.board.fleets.values())
            .filter(fleet => this.fleetIds.some(fId => fId == fleet.id));
    }

    /**
     * Returns whether this player is the current player (generally if this returns True, this player is you.
     */
    public isCurrentPlayer(): boolean {
        return this.id == this.board.currentPlayerId;
    }

    /**
     * Returns all queued fleet and shipyard actions for this player formatted for the kore interpreter to receive as an agent response.
     */
    public get nextActions(): Map<String, ShipyardAction> {
        const result = new Map<String, ShipyardAction>();
        this.shipyards.filter(shipyard => shipyard.nextAction).forEach(shipyard => result.set(shipyard.id, shipyard.nextAction as ShipyardAction));
        return result;
    }

    /**
     * Converts a player back to the normalized observation subset that constructed it.
     */
    public observation(): any[] {
        const shipyards = new Map<string, number[]>(); 
        this.shipyards.forEach(shipyard => shipyards.set(shipyard.id, shipyard.observation()));
        const fleets = new Map<string, string[]>();
        this.fleets.forEach(fleet => fleets.set(fleet.id, fleet.observation()));
        return [this.kore, shipyards, fleets];
    }
}
