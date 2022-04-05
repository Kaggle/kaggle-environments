import {Cell} from "./Cell";
import {Configuration} from "./Configuration";
import {Direction} from "./Direction";
import {Fleet} from "./Fleet";
import {Observation} from "./Observation";
import {Pair} from "./Pair";
import {Player} from "./Player";
import {Point} from "./Point";
import {Shipyard} from "./Shipyard";
import {ShipyardAction} from "./ShipyardAction";

export class Board {

    public readonly shipyards: Map<string, Shipyard>;
    public readonly fleets: Map<string, Fleet>;
    public readonly players: Player[];
    public readonly currentPlayerId: number;
    public readonly configuration: Configuration;
    public step: number;
    public readonly remainingOverageTime: number;
    public readonly cells: Cell[];
    public readonly size: number;

    private uidCounter: number;

    private constructor(shipyards: Map<string, Shipyard>, fleets: Map<string, Fleet>, players: Player[], currentPlayerId: number, configuration: Configuration, step: number, remainingOverageTime: number, cells: Cell[], size: number) {
        this.shipyards = new Map<string, Shipyard>();
        shipyards.forEach((shipyard, shipyardId) => this.shipyards.set(shipyardId, shipyard.cloneToBoard(this)));
        this.fleets = new Map<string, Fleet>();
        fleets.forEach((fleet, fleetId) => this.fleets.set(fleetId, fleet.cloneToBoard(this)));
        this.players = players.map(player => player.cloneToBoard(this));
        this.currentPlayerId = currentPlayerId;
        this.configuration = configuration;
        this.step = step;
        this.remainingOverageTime = remainingOverageTime;
        this.cells = cells.map(cell => cell.cloneToBoard(this));
        this.size = size;
    }

    public cloneBoard(): Board {
        return new Board(this.shipyards, this.fleets, this.players, this.currentPlayerId, this.configuration, this.step, this.remainingOverageTime, this.cells, this.size);
    }

    /**
     * Creates a board from the provided observation, configuration, and nextActions as specified by
     *  https://github.com/Kaggle/kaggle-environments/blob/master/kaggle_environments/envs/kore/kore.json
     *  Board tracks players (by id), fleets (by id), shipyards (by id), and cells (by position).
     *  Each entity contains both key values (e.g. fleet.player_id) as well as entity references (e.g. fleet.player).
     *  References are deep and chainable e.g.
     *      [fleet.kore for player in board.players for fleet in player.fleets]
     *      fleet.player.shipyards()[0].cell.north.east.fleet
     *  Consumers should not set or modify any attributes except and Shipyard.nextAction
     */
    public static fromRaw(rawObservation: string, rawConfiguration: string): Board {
        const observation = new Observation(rawObservation);

        const step = observation.step;
        const remainingOverageTime = observation.remainingOverageTime;
        const configuration = new Configuration(rawConfiguration);
        const currentPlayerId = observation.player;
        const players = new Array<Player>(observation.playerHlt.length);
        const fleets = new Map<string, Fleet>();
        const shipyards = new Map<string, Shipyard>();
        const cells = new Array<Cell>(observation.kore.length);
        const size = configuration.size;

        const board = new Board(shipyards, fleets, players, currentPlayerId, configuration, step, remainingOverageTime, cells, size)

        // Create a cell for every point in a size x size grid
        for (let x = 0; x < size; x++) {
            for (let y = 0; y < size; y++) {
                const position = new Point(x, y);
                const kore = observation.kore[position.toIndex(size)];
                // We'll populate the cell's fleets and shipyards in _add_fleet and addShipyard
                board.cells[position.toIndex(size)] = new Cell(position, kore, "", "", board);
            }

        }


        for (let playerId = 0; playerId < observation.playerHlt.length; playerId++) {
            let playerKore = observation.playerHlt[playerId];
            const playerShipyards = observation.playerShipyards[playerId];
            const playerFleets = observation.playerFleets[playerId];
            board.players[playerId] = new Player(playerId, playerKore, [], [], board);
            //player_actions = nextActions[player_id] or {}

            playerFleets.forEach((fleetStrs, fleetId) => {
                const fleetPosIdx = parseInt(fleetStrs[0]);
                const fleetKore = parseFloat(fleetStrs[1]);
                const shipCount = parseInt(fleetStrs[2]);
                const directionIdx = parseInt(fleetStrs[3]);
                const flightPlan = fleetStrs[4];

                const fleetPosition = Point.fromIndex(fleetPosIdx, size);
                const fleetDirection = Direction.fromIndex(directionIdx);
                board.addFleet(new Fleet(fleetId, shipCount, fleetDirection, fleetPosition, fleetKore, flightPlan, playerId, board));
            })

            playerShipyards.forEach((shipyardInts, shipyardId) => {
                const shipyardPosIdx = shipyardInts[0];
                const shipCount = shipyardInts[1];
                const turnsControlled = shipyardInts[2];
                const shipyardPosition = Point.fromIndex(shipyardPosIdx, size);
                board.addShipyard(new Shipyard(shipyardId, shipCount, shipyardPosition, playerId, turnsControlled, board, undefined));
            })
        }
        return board;
    }

    public getCellAtPosition(position: Point): Cell {
        return this.cells[position.toIndex(this.size)];
    }

    public addFleet(fleet: Fleet): void {
        fleet.player.fleetIds.push(fleet.id);
        fleet.cell.fleetId = fleet.id;
        this.fleets.set(fleet.id, fleet);
    }

    public addShipyard(shipyard: Shipyard): void {
        shipyard.player.shipyardIds.push(shipyard.id);
        shipyard.cell.shipyardId = shipyard.id;
        shipyard.cell.kore = 0;
        this.shipyards.set(shipyard.id, shipyard);
    }

    public deleteFleet(fleet: Fleet): void {
        const fleetIds = fleet.player.fleetIds
        fleetIds.splice(fleetIds.indexOf(fleet.id), 1);
        if (fleet.cell.fleetId == fleet.id) {
            fleet.cell.fleetId = "";
        }
        this.fleets.delete(fleet.id);
    }

    public deleteShipyard(shipyard: Shipyard): void {
        const shipyardsIds = shipyard.player.shipyardIds;
        shipyardsIds.splice(shipyardsIds.indexOf(shipyard.id), 1);
        if (shipyard.cell.shipyardId == shipyard.id) {
            shipyard.cell.shipyardId = "";
        }
        this.shipyards.delete(shipyard.id);
    }

    public getFleetAtPoint(position: Point): Fleet | undefined {
        const matches = Array.from(this.fleets.values()).filter(fleet => fleet.position.equals(position));
        return matches.length > 0 ? matches[0] : undefined;
    }

    public getShipyardAtPoint(position: Point): Shipyard | undefined {
        const matches = Array.from(this.shipyards.values()).filter(shipyard => shipyard.position.equals(position));
        return matches.length > 0 ? matches[0] : undefined;
    }

    /**
     * Returns the current player (generally this is you).
     * @return
     */
    public get currentPlayer(): Player {
        return this.players[this.currentPlayerId];
    }

    /**
     * Returns all players that aren't the current player.
     * You can get all opponent fleets with [fleet for fleet in player.fleets for player in board.opponents]
     */
    public get opponents(): Player[] {
        return this.players.filter(player => player.id != this.currentPlayerId);
    }

    private createUid(): string {
        this.uidCounter += 1;
        return `${this.step + 1}-${this.uidCounter - 1}`;
    }

    private isValidFlightPlan(flightPlan: string): boolean {
        const allowed = "NESWC0123456789";
        let matches = 0;
        for (let i = 0; i < flightPlan.length; i++) {
            const c = flightPlan.substring(i, i +1);
            if (allowed.indexOf(c) === -1) {
                return false;
            }
        }
        return true;
    }

    private findFirstNonDigit(candidateStr: string): number {
        if (candidateStr.length == 0) return 0;
        for (let i = 0; i < candidateStr.length; i++) {
            if (isNaN(Number(candidateStr.charAt(i)))) {
                return i;
            }
        }
        return candidateStr.length + 1;
    }

    private combineFleets(board: Board, fid1: string, fid2: string): string {
        let f1 = board.fleets.get(fid1);
        let f2 = board.fleets.get(fid2);
        if (f1.lessThanOtherAlliedFleet(f2)) {
            const temp = f1;
            f1 = f2;
            f2 = temp;
            const tempS = fid1;
            fid1 = fid2;
            fid2 = tempS;
        }
        f1.kore += f2.kore;
        f1.shipCount += f2.shipCount;
        board.deleteFleet(f2);
        return fid1;
    }

    /**
     * Accepts the list of fleets at a particular position (must not be empty).
     * Returns the fleet with the most ships or None in the case of a tie along with all other fleets.
     */
    public  resolveCollision(fleets: Fleet[]): Pair<(Fleet | undefined), Fleet[]> {
        if (fleets.length == 1) {
            return new Pair<Fleet | undefined, Fleet[]>(fleets[0], []);
        }
        const fleetsByShips = new Map<number, Fleet[]>(); 
        for (let fleet of fleets) {
            const ships = fleet.shipCount;
            if (!fleetsByShips.has(ships)) {
                fleetsByShips.set(ships, []);
            }
            fleetsByShips.get(ships).push(fleet);
        }
        let mostShips = Math.max(...Array.from(fleetsByShips.keys()));
        const largestFleets = fleetsByShips.get(mostShips);
        if (largestFleets.length == 1) {
            // There was a winner, return it
            const winner = largestFleets[0];
            return new Pair<(Fleet | undefined), Fleet[]>(winner, fleets.filter(f => !(f.id == winner.id)));
        }
        // There was a tie for most ships, all are deleted
        return new Pair<(Fleet | undefined), Fleet[]>(undefined, fleets);
    }


    /**
     * Returns a new board with the current board's next actions applied.
     * The current board is unmodified.
     * This can form a kore interpreter, e.g.
     *     next_observation = Board(current_observation, configuration, actions).next().observation
     */
    public next(): Board {
        // Create a copy of the board to modify so we don't affect the current board
        const board = this.cloneBoard();
        const configuration = board.configuration;
        const converstCost = configuration.convertCost;
        const spawnCost = configuration.spawnCost;
        this.uidCounter = 0;

        // Process actions and store the results in the fleets and shipyards lists for collision checking
        for (let player of board.players) {
            // shipyard actions
            for (let shipyard of player.shipyards) {
                if (!shipyard.nextAction) {
                    continue;
                }
                const nextAction: ShipyardAction = shipyard.nextAction;

                if  (nextAction.shipCount == 0) {
                    continue;
                }

                if (nextAction.actionType == ShipyardAction.SPAWN && player.kore >= spawnCost * nextAction.shipCount && nextAction.shipCount <= shipyard.maxSpawn) {
                    player.kore -= spawnCost * nextAction.shipCount;
                    shipyard.shipCount += nextAction.shipCount;
                } else if (nextAction.actionType == ShipyardAction.LAUNCH && shipyard.shipCount >= nextAction.shipCount) {
                    let flightPlan = nextAction.flightPlan;
                    if (flightPlan.length == 0 || !this.isValidFlightPlan(flightPlan)) {
                        continue;
                    }
                    shipyard.shipCount -= nextAction.shipCount;
                    const direction = Direction.fromChar(flightPlan.charAt(0));
                    const maxFlightPlanLen = Fleet.maxFlightPlanLenForShipCount(nextAction.shipCount);
                    if (flightPlan.length > maxFlightPlanLen) {
                        flightPlan = flightPlan.substring(0, maxFlightPlanLen);
                    }
                    board.addFleet(new Fleet(this.createUid(), nextAction.shipCount, direction, shipyard.position, 0, flightPlan, player.id, board));
                }
                
            }
            // clear next action and increase turns controlled
            for (let shipyard of player.shipyards) {
                shipyard.nextAction = undefined;
                shipyard.turnsControlled += 1;
            }

            
            // update fleets 
            for (let fleet of player.fleets) {
                // remove any errant 0s
                while (fleet.flightPlan.length > 0 && fleet.flightPlan.startsWith("0") ) {
                    fleet.flightPlan = fleet.flightPlan.substring(1);
                }
                if (fleet.flightPlan.length > 0 && fleet.flightPlan.startsWith("C") && fleet.shipCount >= converstCost && fleet.cell.shipyardId.length == 0) {
                    player.kore += fleet.kore;
                    fleet.cell.kore = 0;
                    board.addShipyard(new Shipyard(this.createUid(), fleet.shipCount - converstCost, fleet.position, player.id, 0, board, undefined));
                    board.deleteFleet(fleet);
                    continue;
                } 
                // remove all converts
                while (fleet.flightPlan.length > 0 && fleet.flightPlan.startsWith("C")) {
                    // couldn't build, remove the Convert and continue with flight plan
                    fleet.flightPlan = fleet.flightPlan.substring(1);
                }

                if (fleet.flightPlan.length > 0 && "NESW".indexOf(fleet.flightPlan.charAt(0)) > -1) {
                    fleet.direction = Direction.fromChar(fleet.flightPlan.charAt(0));
                    fleet.flightPlan = fleet.flightPlan.substring(1);
                } else if (fleet.flightPlan.length > 0) {
                    const idx = this.findFirstNonDigit(fleet.flightPlan);
                    let digits = parseInt(fleet.flightPlan.substring(0, idx));
                    const rest = fleet.flightPlan.substring(idx);
                    digits -= 1;
                    if (digits > 0) {
                        fleet.flightPlan = digits.toString() + rest;
                    } else {
                        fleet.flightPlan = rest;
                    }
                }

                // continue moving in the fleet's direction
                fleet.cell.fleetId = "";
                fleet.position = fleet.position.translate(fleet.direction, configuration.size);
                // We don't set the new cell's fleet_id here as it would be overwritten by another fleet in the case of collision.
            }

            const fleetsByLoc = new Map<number, Fleet[]>();
            for (let fleet of player.fleets) {
                const locIdx = fleet.position.toIndex(configuration.size);
                if (!fleetsByLoc.has(locIdx)) {
                    fleetsByLoc.set(locIdx, []);
                }
                fleetsByLoc.get(locIdx).push(fleet);
            }

            for (let fleets of Array.from(fleetsByLoc.values())) {
                fleets.sort((a, b) => {
                    if (a.shipCount != b.shipCount) {
                        return a.shipCount > b.shipCount ? -1 : 1;
                    }
                    if (a.kore != b.kore) {
                        return a.kore > b.kore ? -1 : 1;
                    }
                    return a.direction.toIndex() > a.direction.toIndex() ? 1 : -1;
                })
                let fid = fleets[0].id;
                for (let i = 1; i < fleets.length; i++) {
                    fid = this.combineFleets(board, fid, fleets[1].id);
                }

            }
        }

        
        // Check for fleet to fleet collisions
        const fleetCollisionGroups = new Map<number, Fleet[]>();
        board.fleets.forEach(fleet =>  {
            const posIdx = fleet.position.toIndex(board.size);
            if (!fleetCollisionGroups.has(posIdx)) {
                fleetCollisionGroups.set(posIdx, [fleet]);
            } else {
                fleetCollisionGroups.get(posIdx).push(fleet);
            }
        });
        fleetCollisionGroups.forEach((collidedFleets, positionIdx) => { 
            const position = Point.fromIndex(positionIdx, configuration.size);
            const pair = this.resolveCollision(collidedFleets);
            const winnerOptional = pair.first;
            const deleted = pair.second;
            const shipyardOpt = board.getShipyardAtPoint(position);
            if (winnerOptional) {
                const winner: Fleet = winnerOptional;
                winner.cell.fleetId = winner.id;
                const maxEnemySize = deleted.length > 0 ? deleted.map(f => f.shipCount).reduce((a, b) => a > b ? a : b, 0) : 0;
                winner.shipCount -= maxEnemySize;
            }
            for (let fleet of deleted) {
                board.deleteFleet(fleet);
                if (winnerOptional) {
                    // Winner takes deleted fleets' kore
                    (winnerOptional as Fleet).kore += fleet.kore;
                } else if (!winnerOptional && shipyardOpt) {
                    // Desposit the kore into the shipyard
                    (shipyardOpt as Shipyard).player.kore += fleet.kore;
                } else if  (!winnerOptional) {
                    // Desposit the kore on the square
                    board.getCellAtPosition(position).kore += fleet.kore;
                }
            }
        });

        // Check for fleet to shipyard collisions
        for (let shipyard of Array.from(board.shipyards.values())) {
            const optFleet = shipyard.cell.fleet;
            if (optFleet && (optFleet as Fleet).playerId != shipyard.playerId) {
                const fleet = optFleet as Fleet;
                if (fleet.shipCount > shipyard.shipCount) {
                    const count = fleet.shipCount - shipyard.shipCount;
                    board.deleteShipyard(shipyard);
                    board.addShipyard(new Shipyard(this.createUid(), count, shipyard.position, fleet.player.id, 1, board, undefined));
                    fleet.player.kore += fleet.kore;
                    board.deleteFleet(fleet);
                } else {
                    shipyard.shipCount -= fleet.shipCount;
                    shipyard.player.kore += fleet.kore;
                    board.deleteFleet(fleet);
                }
            }
        }

        // Deposit kore from fleets into shipyards
        for (let shipyard of Array.from(board.shipyards.values())) {
            const optFleet = shipyard.cell.fleet;
            if (optFleet && (optFleet as Fleet).playerId == shipyard.playerId) {
                const fleet = optFleet as Fleet;
                shipyard.player.kore += fleet.kore;
                shipyard.shipCount += fleet.shipCount;
                board.deleteFleet(fleet);
            }
        }

        // apply fleet to fleet damage on all orthagonally adjacent cells
        const incomingFleetDmg = new Map<string, Pair<string, number>[]>();
        for (const fleet of Array.from(board.fleets.values())) {
            for (const direction of Direction.listDirections()) {
                const currPos = fleet.position.translate(direction, board.configuration.size);
                const optFleet = board.getFleetAtPoint(currPos);
                if (optFleet && (optFleet as Fleet).playerId != fleet.playerId) {
                    const toAttack = optFleet as Fleet;
                    if (!incomingFleetDmg.has(toAttack.id)) {
                        incomingFleetDmg.set(toAttack.id, []);
                    }
                    incomingFleetDmg.get(toAttack.id).push(new Pair(fleet.id, fleet.shipCount));
                }
            }
        }

        // dump 1/2 kore to the cell of killed flets
        // mark the other 1/2 kore to go to attacking fleet proportionally
        const toDistrubute = new Map<string, Pair<number, number>[]>();
        incomingFleetDmg.forEach((attackers, fleetId) => {
            const totalDamage = attackers.map(pair => pair.second).reduce((a, b) => a + b, 0);
            const fleet = board.fleets.get(fleetId);
            if (totalDamage >= fleet.shipCount) {
                fleet.cell.kore += fleet.kore / 2;
                attackers.forEach(p => {
                    const attackerId = p.first;
                    const attackerDmg = p.second;
                    if (!toDistrubute.has(attackerId)) {
                        toDistrubute.set(attackerId, []);
                    }
                    const toGet = fleet.kore / 2 * attackerDmg / totalDamage;
                    toDistrubute.get(attackerId).push(new Pair(fleet.cell.position.toIndex(board.configuration.size), toGet));
                })
                board.deleteFleet(fleet);
            } else {
                fleet.shipCount -= totalDamage;
            }

        });

        // give kore claimed above to surviving fleets, otherwise add it back to the tile where the fleet died.
        toDistrubute.forEach((resourceFromLocs, fleetId) => {
            resourceFromLocs.forEach(p => {
                const cellIdx = p.first;
                const kore = p.second;
                if (!board.fleets.has(fleetId)) {
                    board.cells[cellIdx].kore += kore;
                } else {
                    const fleet = board.fleets.get(fleetId);
                    fleet.kore += kore;
                }
            });
        });

        // Collect kore from cells into fleets
        for (const fleet of Array.from(board.fleets.values())) {
            const cell = fleet.cell;
            const deltaKore = Board.roundToThreePlaces(cell.kore * Math.min(fleet.collectionRate, .99));
            if (deltaKore > 0) {
                fleet.kore += deltaKore;
                cell.kore -= deltaKore;
            }
        }

        // Regenerate kore in cells
        for (let cell of board.cells) {
            if (cell.fleetId === "" && cell.shipyardId === "") {
                if (cell.kore < configuration.maxRegenCellKore) {
                    const nextKore = Board.roundToThreePlaces(cell.kore * (1 + configuration.regenRate) * 1000.0) / 1000.0;
                    cell.kore = nextKore;
                }
            }
        }

        board.step += 1;

        return board;
    }

    private static roundToThreePlaces(num: number): number {
        return Math.round(num * 1000.0) / 1000.0;
    }

}
