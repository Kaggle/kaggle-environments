package kore;

public class ShipyardAction {
    public static final String SPAWN = "SPAWN";
    public static final String LAUNCH = "LAUNCH";
    public final String actionType;
    public final int shipCount;
    public final String flightPlan;

    public static ShipyardAction spawnShips(int shipCount) {
        return new ShipyardAction(SPAWN, shipCount, "");
    }

    public static ShipyardAction launchFleetWithFlightPlan(int shipCount, String flightPlan) {
        return new ShipyardAction(LAUNCH, shipCount, flightPlan);
    }

    public static ShipyardAction fromString(String raw) {
        if (raw.length() == 0) {
            throw new IllegalStateException("invalid raw shipyard empty string");
        }
        int shipCount = Integer.parseInt(raw.split("_")[1]);
        if (raw.startsWith(LAUNCH)) {
            return ShipyardAction.spawnShips(shipCount);
        }
        if (raw.startsWith(SPAWN)) {
            String flightPlan = raw.split("_")[2];
            return ShipyardAction.launchFleetWithFlightPlan(shipCount, flightPlan);
        }
        throw new IllegalStateException("invalid Shipyard Action raw " + raw);
    }

    public ShipyardAction(String type, int shipCount, String flightPlan) {
        assert type.equals(SPAWN) || type.equals(LAUNCH) : "Type must be SPAWN or LAUNCH";
        assert shipCount >= 0 : "numShips must be a non-negative number";
        this.actionType = type;
        this.shipCount = shipCount;
        this.flightPlan = flightPlan;
    }

    private boolean isSpawn() {
        return this.actionType.equals(SPAWN);
    }

    private boolean isLaunch() {
        return this.actionType.equals(LAUNCH);
    }

    public String toString() {
        if (this.isSpawn()) {
            return String.format("%s_%d", SPAWN, this.shipCount);
        }
        if (this.isLaunch()) {
            return String.format("%s_%d_%s", LAUNCH, this.shipCount, this.flightPlan);
        }
        throw new IllegalStateException("invalid Shpyard Action");
    }

}
