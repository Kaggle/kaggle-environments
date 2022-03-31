export class Configuration {
    
    public readonly agentTimeout: number;
    public readonly startingKore: number;
    public readonly size: number;
    public readonly spawnCost: number;
    public readonly convertCost: number;
    public readonly regenRate: number;
    public readonly maxRegenCellKore: number;
    public readonly randomSeed: number;

    public constructor(rawConfiguration: string) {
        const config = JSON.parse(rawConfiguration);
        this.agentTimeout = config.agentTimeout;
        this.startingKore = config.startingKore;
        this.size = config.size;
        this.spawnCost = config.spawnCost;
        this.convertCost = config.convertCost;
        this.regenRate = config.regenRate;
        this.maxRegenCellKore = config.maxRegenCellKore;
        this.randomSeed = config.randomSeed;
    }
}