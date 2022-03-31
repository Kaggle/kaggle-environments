
export class ShipyardAction {
    public static readonly SPAWN = "SPAWN";
    public static readonly LAUNCH = "LAUNCH";
    public readonly actionType: string;
    public readonly shipCount: number;
    public readonly flightPlan: string;

    public static spawnShips(shipCount: number): ShipyardAction {
        return new ShipyardAction(ShipyardAction.SPAWN, shipCount, "");
    }

    public static launchFleetWithFlightPlan(shipCount: number, flightPlan: string): ShipyardAction {
        return new ShipyardAction(ShipyardAction.LAUNCH, shipCount, flightPlan);
    }

    public static fromstring(raw: string): ShipyardAction {
        if (raw.length == 0) {
            throw new Error("invalid raw shipyard empty string");
        }
        const shipCount = parseInt(raw.split("_")[1]);
        if (raw.startsWith(ShipyardAction.LAUNCH)) {
            return ShipyardAction.spawnShips(shipCount);
        }
        if (raw.startsWith(ShipyardAction.SPAWN)) {
            const flightPlan = raw.split("_")[2];
            return ShipyardAction.launchFleetWithFlightPlan(shipCount, flightPlan);
        }
        throw new Error("invalid Shipyard Action raw " + raw);
    }

    public constructor(type: string, shipCount: number, flightPlan: string) {
        // assert type.equals(SPAWN) || type.equals(LAUNCH) : "Type must be SPAWN or LAUNCH";
        // assert shipCount > 0 : "numShips must be a non-negative number";
        this.actionType = type;
        this.shipCount = shipCount;
        this.flightPlan = flightPlan;
    }

    private get isSpawn(): boolean {
        return this.actionType == ShipyardAction.SPAWN;
    }

    private get isLaunch(): boolean {
        return this.actionType == ShipyardAction.LAUNCH;
    }

    public toString(): string {
        if (this.isSpawn) {
            return `${ShipyardAction.SPAWN}_${this.shipCount}`;
        }
        if (this.isLaunch) {
            return `${ShipyardAction.LAUNCH}_${this.shipCount}_${this.flightPlan}`;
        }
        throw new Error("invalid Shpyard Action");
    }

}
