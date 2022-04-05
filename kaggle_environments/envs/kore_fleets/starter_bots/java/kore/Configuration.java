package kore;

public class Configuration {
    
    public final int agentTimeout;
    public final int startingKore;
    public final int size;
    public final double spawnCost;
    public final int convertCost;
    public final double regenRate;
    public final int maxRegenCellKore;
    public final int randomSeed;

    public Configuration(String rawConfiguration) {
        this.agentTimeout = KoreJson.getIntFromJson(rawConfiguration, "agentTimeout");
        this.startingKore = KoreJson.getIntFromJson(rawConfiguration, "startingKore");
        this.size = KoreJson.getIntFromJson(rawConfiguration, "size");
        this.spawnCost = KoreJson.getDoubleFromJson(rawConfiguration, "spawnCost");
        this.convertCost = KoreJson.getIntFromJson(rawConfiguration, "convertCost");
        this.regenRate = KoreJson.getDoubleFromJson(rawConfiguration, "regenRate");
        this.maxRegenCellKore = KoreJson.getIntFromJson(rawConfiguration, "maxRegenCellKore");
        this.randomSeed = KoreJson.getIntFromJson(rawConfiguration, "randomSeed");
    }
}
