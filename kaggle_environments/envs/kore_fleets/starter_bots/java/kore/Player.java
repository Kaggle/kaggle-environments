package kore;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.stream.Collectors;

public class Player {
    public final int id; 
    public double kore; 
    public final ArrayList<String> shipyardIds; 
    public final ArrayList<String> fleetIds;
    public final Board board;
    
    public Player(int playerId, double kore, ArrayList<String> shipyardIds, ArrayList<String> fleetIds, Board board) {
        this.id = playerId;
        this.kore = kore;
        this.shipyardIds = shipyardIds;
        this.fleetIds = fleetIds;
        this.board = board;
    }

    public Player cloneToBoard(Board board) {
        return new Player(this.id, this.kore, new ArrayList<String>(this.shipyardIds.stream().collect(Collectors.toList())), new ArrayList<String>(this.fleetIds.stream().collect(Collectors.toList())), board);
    }

    /**
     * Returns all shipyards owned by this player.
     * @return
     */
    public Shipyard[] shipyards() {
        return this.board.shipyards.values().stream().filter(shipyard -> this.shipyardIds.stream().anyMatch(sId -> sId == shipyard.id)).toArray(Shipyard[]::new);
    }

    /**
     * Returns all fleets owned by this player.
     */
    public Fleet[] fleets() {
        return this.board.fleets.values().stream().filter(fleet -> this.fleetIds.stream().anyMatch(fId -> fId == fleet.id)).toArray(Fleet[]::new);
    }

    /**
     * Returns whether this player is the current player (generally if this returns True, this player is you.
     */
    public boolean isCurrentPlayer() {
        return this.id == this.board.currentPlayerId;
    }

    /**
     * Returns all queued fleet and shipyard actions for this player formatted for the kore interpreter to receive as an agent response.
     */
    public HashMap<String, ShipyardAction> nextActions() {
        HashMap<String, ShipyardAction> result = new HashMap<>();
        Arrays.stream(this.shipyards()).filter(shipyard -> shipyard.nextAction.isPresent()).forEach(shipyard -> result.put(shipyard.id, shipyard.nextAction.get()));
        return result;
    }

    /**
     * Converts a player back to the normalized observation subset that constructed it.
     */
    public Object[] observation() {
        HashMap<String, int[]> shipyards = new HashMap<>(); 
        Arrays.stream(this.shipyards()).forEach(shipyard -> shipyards.put(shipyard.id, shipyard.observation()));
        HashMap<String, String[]> fleets = new HashMap<>();
        Arrays.stream(this.fleets()).forEach(fleet -> fleets.put(fleet.id, fleet.observation()));
        return new Object[]{this.kore, shipyards, fleets};
    }
}
