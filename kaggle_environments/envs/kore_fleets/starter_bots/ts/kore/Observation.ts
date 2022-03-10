
export class Observation {
    public readonly kore: number[];
    public readonly playerHlt: number[]; 
    public readonly playerShipyards: Map<string, number[]>[];
    public readonly playerFleets: Map<string, string[]>[];
    public readonly player: number;
    public readonly step: number;
    public readonly remainingOverageTime: number;

    public constructor(rawObservation: string) {
        const json = JSON.parse(rawObservation);
        this.kore = json["kore"];
        this.player = json["player"];
        this.step = json["step"];
        this.remainingOverageTime = rawObservation["remainingOverageTime"];
        const playerParts = json["players"];
        this.playerHlt = [];
        this.playerShipyards = [];
        this.playerFleets = [];

        for (var i = 0; i < playerParts.length; i ++) {
            const playerPart = playerParts[i];
            this.playerHlt.push(parseInt(playerPart[0]));

            const shipyards = new Map<string, number[]>();
            Object.entries(playerPart[1]).forEach(entry => {
                const shipyardId = entry[0];
                const shipyardInts = entry[1];
                shipyards.set(shipyardId, shipyardInts as number[]);
            });
            this.playerShipyards.push(shipyards);

            const fleets = new Map<string, string[]>();
            Object.entries(playerPart[2]).forEach(entry => {
                const fleetId = entry[0];
                const fleetStrs = entry[1];
                fleets.set(fleetId, fleetStrs as string[])
            })
            this.playerFleets.push(fleets)
        }
    }

    
}
