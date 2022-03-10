package kore;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class Observation {
    public final double[] kore;
    public final double[] playerHlt; 
    public final ArrayList<HashMap<String, int[]>> playerShipyards;
    public final ArrayList<HashMap<String, String[]>> playerFleets;
    public final int player;
    public final int step;
    public final double remainingOverageTime;

    private static String shortenFrontAndBack(String target, int n) {
        return target.substring(n, target.length() - n);
    }

    public Observation(String rawObservation) {
        // avoid importing json library? worth it?
        this.kore = KoreJson.getDoubleArrFromJson(rawObservation, "kore");
        this.player = KoreJson.getPlayerIdxFromJson(rawObservation);
        this.step = KoreJson.getIntFromJson(rawObservation, "step");
        this.remainingOverageTime = KoreJson.getDoubleFromJson(rawObservation, "remainingOverageTime");
        String[] playerParts = KoreJson.getPlayerPartsFromJson(rawObservation);
        playerHlt = new double[playerParts.length];
        playerShipyards = new ArrayList<HashMap<String, int[]>>();
        playerFleets = new ArrayList<HashMap<String, String[]>>();

        for (int i = 0; i < playerParts.length; i ++) {
            String playerPart = playerParts[i];
            playerHlt[i] = Double.parseDouble(playerPart.split(", ")[0]);

            int startShipyards = playerPart.indexOf("{");
            int endShipyards = playerPart.indexOf("}");
            String shipyardsStr = playerPart.substring(startShipyards + 1, endShipyards - 1);
            HashMap<String, int[]> shipyards = new HashMap<String, int[]>();
            Arrays.stream(shipyardsStr.split("], ")).forEach(shipyardStr -> {
                if (shipyardStr.length() == 0) {
                    return;
                }
                String[] kvparts = shipyardStr.split(": \\[");
                String shipyardId = shortenFrontAndBack(kvparts[0], 1);
                String[] shipyardStrs = kvparts[1].split(", ");
                int[] shipyard = new int[shipyardStrs.length];
                Integer[] shipyardInts = Arrays.stream(shipyardStrs).map(s -> Integer.parseInt(s)).toArray(Integer[]::new);
                for(int j = 0; j < shipyard.length; j++) {
                    shipyard[j] = shipyardInts[j];
                }
                shipyards.put(shipyardId, shipyard);
            });
            playerShipyards.add(shipyards);

            int startFleets = playerPart.indexOf("}, ");
            String fleetsStr = playerPart.substring(startFleets + 4, playerPart.length() - 1);
            HashMap<String, String[]> fleets = new HashMap<>();
            Arrays.stream(fleetsStr.split("], ")).forEach(fleetStr -> {
                if (fleetStr.length() == 0) {
                    return;
                }
                String[] kvparts = fleetStr.split(": ");
                String fleetId = shortenFrontAndBack(kvparts[0], 1);
                String[] fleet = shortenFrontAndBack(kvparts[1], 1).split(", ");
                fleets.put(fleetId, fleet);
            });
            playerFleets.add(fleets);
        }
    }

    
}
