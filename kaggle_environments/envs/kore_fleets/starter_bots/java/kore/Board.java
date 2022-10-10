package kore;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Optional;
import java.util.Map.Entry;

public class Board {

    public final HashMap<String, Shipyard> shipyards;
    public final HashMap<String, Fleet> fleets;
    public final Player[] players;
    public final int currentPlayerId;
    public final Configuration configuration;
    public int step;
    public final double remainingOverageTime;
    public final Cell[] cells;
    public final int size;

    private int uidCounter;

    public Board(HashMap<String, Shipyard> shipyards, HashMap<String, Fleet> fleets, Player[] players, int currentPlayerId, Configuration configuration, int step, double remainingOverageTime, Cell[] cells, int size) {
        this.shipyards = new HashMap<>();
        shipyards.entrySet().stream().forEach(entry -> this.shipyards.put(entry.getKey(), entry.getValue().cloneToBoard(this)));
        this.fleets = new HashMap<>();
        fleets.entrySet().stream().forEach(entry -> this.fleets.put(entry.getKey(), entry.getValue().cloneToBoard(this)));
        this.players = Arrays.stream(players).map(player -> player.cloneToBoard(this)).toArray(Player[]::new);
        this.currentPlayerId = currentPlayerId;
        this.configuration = configuration;
        this.step = step;
        this.remainingOverageTime = remainingOverageTime;
        this.cells = Arrays.stream(cells).map(cell -> cell.cloneToBoard(this)).toArray(Cell[]::new);
        this.size = size;
    }

    public Board cloneBoard() {
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
    public Board(
        String rawObservation,
        String rawConfiguration
    ) {
        Observation observation = new Observation(rawObservation);

        this.step = observation.step;
        this.remainingOverageTime = observation.remainingOverageTime;
        this.configuration = new Configuration(rawConfiguration);
        this.currentPlayerId = observation.player;
        this.players = new Player[observation.playerHlt.length];
        this.fleets = new HashMap<String, Fleet>();
        this.shipyards = new HashMap<String, Shipyard>();
        this.cells =  new Cell[observation.kore.length];
        this.size = this.configuration.size;

        // Create a cell for every point in a size x size grid
        for (int x = 0; x < size; x++) {
            for (int y = 0; y < size; y++) {
                Point position = new Point(x, y);
                double kore = observation.kore[position.toIndex(size)];
                // We'll populate the cell's fleets and shipyards in _add_fleet and addShipyard
                this.cells[position.toIndex(size)] = new Cell(position, kore, "", "", this);
            }

        }

        for (int playerId = 0; playerId < observation.playerHlt.length; playerId++) {
            double playerKore = observation.playerHlt[playerId];
            HashMap<String, int[]> playerShipyards = observation.playerShipyards.get(playerId);
            HashMap<String, String[]> playerFleets = observation.playerFleets.get(playerId);
            this.players[playerId] = new Player(playerId, playerKore, new ArrayList<String>(), new ArrayList<String>(), this);
            //player_actions = nextActions[player_id] or {}

            for (Entry<String, String[]> entry : playerFleets.entrySet()) {
                String fleetId = entry.getKey();
                String[] fleetStrs = entry.getValue();

                int fleetPosIdx = Integer.parseInt(fleetStrs[0]);
                float fleetKore = Float.parseFloat(fleetStrs[1]);
                int shipCount = Integer.parseInt(fleetStrs[2]);
                int directionIdx = Integer.parseInt(fleetStrs[3]);
                String flightPlan = fleetStrs[4];

                Point fleetPosition = Point.fromIndex(fleetPosIdx, this.size);
                Direction fleetDirection = Direction.fromIndex(directionIdx);
                this.addFleet(new Fleet(fleetId, shipCount, fleetDirection, fleetPosition, fleetKore, flightPlan, playerId, this));
            }

            for (Entry<String, int[]> entry : playerShipyards.entrySet()) {
                String shipyardId = entry.getKey();
                int[] shipyardInts = entry.getValue();
                int shipyardPosIdx = shipyardInts[0];
                int shipCount = shipyardInts[1];
                int turnsControlled = shipyardInts[2];
                Point shipyardPosition = Point.fromIndex(shipyardPosIdx, this.size);
                Optional<ShipyardAction> action = Optional.empty();
                this.addShipyard(new Shipyard(shipyardId, shipCount, shipyardPosition, playerId, turnsControlled, this, action));
            }
        }
    }

    public Cell getCellAtPosition(Point position) {
        return this.cells[position.toIndex(this.size)];
    }

    public void addFleet(Fleet fleet) {
        fleet.player().fleetIds.add(fleet.id);
        fleet.cell().fleetId = fleet.id;
        this.fleets.put(fleet.id, fleet);
    }

    public void addShipyard(Shipyard shipyard) {
        shipyard.player().shipyardIds.add(shipyard.id);
        shipyard.cell().shipyardId = shipyard.id;
        shipyard.cell().kore = 0;
        this.shipyards.put(shipyard.id, shipyard);
    }

    public void deleteFleet(Fleet fleet) {
        fleet.player().fleetIds.remove(fleet.id);
        if (fleet.cell().fleetId == fleet.id) {
            fleet.cell().fleetId = "";
        }
        this.fleets.remove(fleet.id);
    }

    public void deleteShipyard(Shipyard shipyard) {
        shipyard.player().shipyardIds.remove(shipyard.id);
        if (shipyard.cell().shipyardId == shipyard.id) {
            shipyard.cell().shipyardId = "";
        }
        this.shipyards.remove(shipyard.id);
    }

    public Optional<Fleet> getFleetAtPoint(Point position) {
        return this.fleets.values().stream().filter(fleet -> fleet.position.equals(position)).findAny();
    }

    public Optional<Shipyard> getShipyardAtPoint(Point position) {
        return this.shipyards.values().stream().filter(shipyard -> shipyard.position.equals(position)).findAny();
    }

    /**
     * Returns the current player (generally this is you).
     * @return
     */
    public Player currentPlayer() {
        return this.players[this.currentPlayerId];
    }

    /**
     * Returns all players that aren't the current player.
     * You can get all opponent fleets with [fleet for fleet in player.fleets for player in board.opponents]
     */
    public Player[] opponents() {
        return Arrays.stream(this.players).filter(player -> player.id != this.currentPlayerId).toArray(Player[]::new);
    }

    private String createUid() {
        String uid = String.format("%d-%d", this.step + 1, this.uidCounter);
        this.uidCounter += 1;
        return uid;
    }

    private boolean isValidFlightPlan(String flightPlan) {
        String allowed = "NESWC0123456789";
        for (int i = 0; i < allowed.length(); i++) {
            String c = allowed.substring(i, i +1);
            if (allowed.indexOf(c) == -1) {
                return false;
            }
        }
        return true;
    }

    private int findFirstNonDigit(String candidateStr) {
        if (candidateStr.length() == 0) return 0;
        for (int i = 0; i < candidateStr.length(); i++) {
            if (!Character.isDigit(candidateStr.charAt(i))) {
                return i;
            }
        }
        return candidateStr.length() + 1;
    }

    private String combineFleets(Board board, String fid1, String fid2) {
        Fleet f1 = board.fleets.get(fid1);
        Fleet f2 = board.fleets.get(fid2);
        if (f1.lessThanOtherAlliedFleet(f2)) {
            Fleet temp = f1;
            f1 = f2;
            f2 = temp;
            String tempS = fid1;
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
    public Pair<Optional<Fleet>, Fleet[]> resolveCollision(List<Fleet> fleets) {
        if (fleets.size() == 1) {
            return new Pair<Optional<Fleet>, Fleet[]>(Optional.of(fleets.get(0)), new Fleet[0]);
        }
        HashMap<Integer, List<Fleet>> fleetsByShips = new HashMap<Integer, List<Fleet>>(); 
        for (Fleet fleet : fleets) {
            int ships = fleet.shipCount;
            if (!fleetsByShips.containsKey(ships)) {
                fleetsByShips.put(ships, new ArrayList<Fleet>());
            }
            fleetsByShips.get(ships).add(fleet);
        }
        int mostShips = fleetsByShips.keySet().stream().max((a, b) -> a > b ? 1 : -1).get();
        List<Fleet> largestFleets = fleetsByShips.get(mostShips);
        if (largestFleets.size() == 1) {
            // There was a winner, return it
            Fleet winner = largestFleets.get(0);
            return new Pair<Optional<Fleet>, Fleet[]>(Optional.of(winner), fleets.stream().filter(f -> !f.id.equals(winner.id)).toArray(Fleet[]::new));
        }
        // There was a tie for most ships, all are deleted
        return new Pair<Optional<Fleet>, Fleet[]>(Optional.empty(), fleets.stream().toArray(Fleet[]::new));
    }


    /**
     * Returns a new board with the current board's next actions applied.
     * The current board is unmodified.
     * This can form a kore interpreter, e.g.
     *     next_observation = Board(current_observation, configuration, actions).next().observation
     */
    public Board next() {
        // Create a copy of the board to modify so we don't affect the current board
        Board board = this.cloneBoard();
        Configuration configuration = board.configuration;
        int converstCost = configuration.convertCost;
        int spawnCost = configuration.spawnCost;
        this.uidCounter = 0;


        // Process actions and store the results in the fleets and shipyards lists for collision checking
        for (Player player : board.players) {
            for (Shipyard shipyard : player.shipyards()) {
                if (shipyard.nextAction.isEmpty()) {
                    continue;
                }
                ShipyardAction nextAction = shipyard.nextAction.get();

                if  (nextAction.shipCount == 0) {
                    continue;
                }

                if (nextAction.actionType.equals(ShipyardAction.SPAWN) && player.kore >= spawnCost * nextAction.shipCount && nextAction.shipCount <= shipyard.maxSpawn()) {
                    player.kore -= spawnCost * nextAction.shipCount;
                    shipyard.shipCount += nextAction.shipCount;
                } else if (nextAction.actionType.equals(ShipyardAction.LAUNCH) && shipyard.shipCount >= nextAction.shipCount) {
                    String flightPlan = nextAction.flightPlan;
                    if (flightPlan.length() == 0 || !isValidFlightPlan(flightPlan)) {
                        continue;
                    }
                    shipyard.shipCount -= nextAction.shipCount;
                    Direction direction = Direction.fromChar(flightPlan.charAt(0));
                    int maxFlightPlanLen = Fleet.maxFlightPlanLenForShipCount(nextAction.shipCount);
                    if (flightPlan.length() > maxFlightPlanLen) {
                        flightPlan = flightPlan.substring(0, maxFlightPlanLen);
                    }
                    board.addFleet(new Fleet(this.createUid(), nextAction.shipCount, direction, shipyard.position, 0, flightPlan, player.id, board));
                }
                
                shipyard.nextAction = Optional.empty();
            }
            for (Shipyard shipyard : player.shipyards()) {
                shipyard.nextAction = Optional.empty();
                shipyard.turnsControlled += 1;
            }

            
            for (Fleet fleet : player.fleets()) {
                // remove any errant 0s
                while (fleet.flightPlan.length() > 0 && fleet.flightPlan.startsWith("0") ) {
                    fleet.flightPlan = fleet.flightPlan.substring(1);
                }
                if (fleet.flightPlan.length() > 0 && fleet.flightPlan.startsWith("C") && fleet.shipCount >= converstCost && fleet.cell().shipyardId.length() == 0) {
                    player.kore += fleet.kore;
                    fleet.cell().kore = 0;
                    board.addShipyard(new Shipyard(this.createUid(), fleet.shipCount - converstCost, fleet.position, player.id, 0, board, Optional.empty()));
                    board.deleteFleet(fleet);
                    continue;
                } 
                while (fleet.flightPlan.length() > 0 && fleet.flightPlan.startsWith("C")) {
                    // couldn't build, remove the Convert and continue with flight plan
                    fleet.flightPlan = fleet.flightPlan.substring(1);
                }

                if (fleet.flightPlan.length() > 0 && Character.isLetter(fleet.flightPlan.charAt(0))) {
                    fleet.direction = Direction.fromChar(fleet.flightPlan.charAt(0));
                    fleet.flightPlan = fleet.flightPlan.substring(1);
                } else if (fleet.flightPlan.length() > 0) {
                    int idx = this.findFirstNonDigit(fleet.flightPlan);
                    int digits = Integer.parseInt(fleet.flightPlan.substring(0, idx));
                    String rest = fleet.flightPlan.substring(idx);
                    digits -= 1;
                    if (digits > 0) {
                        fleet.flightPlan = String.valueOf(digits) + rest;
                    } else {
                        fleet.flightPlan = rest;
                    }
                }

                // continue moving in the fleet's direction
                fleet.cell().fleetId = "";
                fleet.position = fleet.position.translate(fleet.direction, configuration.size);
                // We don't set the new cell's fleet_id here as it would be overwritten by another fleet in the case of collision.
            }

            HashMap<Integer, List<Fleet>> fleetsByLoc = new HashMap<Integer, List<Fleet>>();
            for (Fleet fleet : player.fleets()) {
                int locIdx = fleet.position.toIndex(configuration.size);
                if (!fleetsByLoc.containsKey(locIdx)) {
                    fleetsByLoc.put(locIdx, new ArrayList<Fleet>());
                }
                fleetsByLoc.get(locIdx).add(fleet);
            }

            for (List<Fleet> fleets : fleetsByLoc.values()) {
                fleets.sort((a, b) -> {
                    if (a.shipCount != b.shipCount) {
                        return a.shipCount > b.shipCount ? -1 : 1;
                    }
                    if (a.kore != b.kore) {
                        return a.kore > b.kore ? -1 : 1;
                    }
                    return a.direction.toIndex() > a.direction.toIndex() ? 1 : -1;
                });
                String fid = fleets.get(0).id;
                for (int i = 1; i < fleets.size(); i++) {
                    fid = this.combineFleets(board, fid, fleets.get(i).id);
                }

            }

            // Lets just check and make sure.
            assert player.kore >= 0 : "Player should have non-negative kore";
        }

        
        // Check for fleet to fleet collisions
        HashMap<Integer, List<Fleet>> fleetCollisionGroups = new HashMap<Integer, List<Fleet>>();
        for (Fleet fleet : board.fleets.values()) {
            int posIdx = fleet.position.toIndex(board.size);
            if (!fleetCollisionGroups.containsKey(posIdx)) {
                fleetCollisionGroups.put((Integer)posIdx, (List<Fleet>)new ArrayList<Fleet>());
            }
            fleetCollisionGroups.get(posIdx).add(fleet);
        }
        for (Entry<Integer, List<Fleet>> entry : fleetCollisionGroups.entrySet()) {
            Point position = Point.fromIndex(entry.getKey(), configuration.size);
            List<Fleet> collidedFleets = entry.getValue();
            Pair<Optional<Fleet>, Fleet[]> pair = this.resolveCollision(collidedFleets);
            Optional<Fleet> winnerOptional = pair.first;
            Fleet[] deleted = pair.second;
            Optional<Shipyard> shipyardOpt = board.getShipyardAtPoint(position);
            if (winnerOptional.isPresent()) {
                Fleet winner = winnerOptional.get();
                winner.cell().fleetId = winner.id;
                int maxEnemySize = deleted.length > 0 ? Arrays.stream(deleted).map(f -> f.shipCount).max((a, b) -> a > b ? 1: -1).get() : 0;
                winner.shipCount -= maxEnemySize;
            }
            for (Fleet fleet : deleted) {
                board.deleteFleet(fleet);
                if (winnerOptional.isPresent()) {
                    // Winner takes deleted fleets' kore
                    winnerOptional.get().kore += fleet.kore;
                } else if (winnerOptional.isEmpty() && shipyardOpt.isPresent()) {
                    // Desposit the kore into the shipyard
                    shipyardOpt.get().player().kore += fleet.kore;
                } else if  (winnerOptional.isEmpty()) {
                    // Desposit the kore on the square
                    board.getCellAtPosition(position).kore += fleet.kore;
                }
            }
        }


        // Check for fleet to shipyard collisions
        for (Shipyard shipyard : board.shipyards.values()) {
            Optional<Fleet> optFleet = shipyard.cell().fleet();
            if (!optFleet.isEmpty() && optFleet.get().playerId != shipyard.playerId) {
                Fleet fleet = optFleet.get();
                if (fleet.shipCount > shipyard.shipCount) {
                    int count = fleet.shipCount - shipyard.shipCount;
                    board.deleteShipyard(shipyard);
                    board.addShipyard(new Shipyard(this.createUid(), count, shipyard.position, fleet.player().id, 1, board, Optional.empty()));
                    fleet.player().kore += fleet.kore;
                    board.deleteFleet(fleet);
                } else {
                    shipyard.shipCount -= fleet.shipCount;
                    shipyard.player().kore += fleet.kore;
                    board.deleteFleet(fleet);
                }
            }
        }

        // Deposit kore from fleets into shipyards
        for (Shipyard shipyard : board.shipyards.values()) {
            Optional<Fleet> optFleet = shipyard.cell().fleet();
            if (!optFleet.isEmpty() && optFleet.get().playerId == shipyard.playerId) {
                Fleet fleet = optFleet.get();
                shipyard.player().kore += fleet.kore;
                shipyard.shipCount += fleet.shipCount;
                board.deleteFleet(fleet);
            }
        }

        // apply fleet to fleet damage on all orthagonally adjacent cells
        HashMap<String, ArrayList<Pair<String, Integer>>> incomingFleetDmg = new HashMap<String, ArrayList<Pair<String, Integer>>>();
        for (Fleet fleet : board.fleets.values()) {
            for (Direction direction : Direction.listDirections()) {
                Point currPos = fleet.position.translate(direction, board.configuration.size);
                Optional<Fleet> optFleet = board.getFleetAtPoint(currPos);
                if (optFleet.isPresent() && optFleet.get().playerId != fleet.playerId) {
                    Fleet f = optFleet.get();
                    if (!incomingFleetDmg.containsKey(f.id)) {
                        incomingFleetDmg.put(f.id, new ArrayList<Pair<String, Integer>>());
                    }
                    incomingFleetDmg.get(f.id).add(new Pair<String, Integer>(fleet.id, fleet.shipCount));
                }
            }
        }

        // dump 1/2 kore to the cell of killed fleets
        // mark the other 1/2 kore to go to surrounding fleets proportionally
        HashMap<String, ArrayList<Pair<Integer, Double>>> toDistribute = new HashMap<String, ArrayList<Pair<Integer, Double>>>();
        for(Entry<String, ArrayList<Pair<String, Integer>>> entry : incomingFleetDmg.entrySet()) {
            String fleetId = entry.getKey();
            ArrayList<Pair<String, Integer>> attackers = entry.getValue();
            int totalDamage = attackers.stream().map(p -> p.second).reduce(0, Integer::sum);
            Fleet fleet = board.fleets.get(fleetId);
            if (totalDamage >= fleet.shipCount) {
                fleet.cell().kore += fleet.kore / 2;
                attackers.forEach(p -> {
                    String attackerId = p.first;
                    int attackerDmg = p.second;
                    if (!toDistribute.containsKey(attackerId)) {
                        toDistribute.put(attackerId, new ArrayList<Pair<Integer, Double>>());
                    }
                    double toGet = fleet.kore / 2 * (double)attackerDmg / (double)totalDamage;
                    toDistribute.get(attackerId).add(new Pair<Integer, Double>(fleet.cell().position.toIndex(board.configuration.size), toGet));
                });
                board.deleteFleet(fleet);
            } else {
                fleet.shipCount -= totalDamage;
            }
        }

        // give kore claimed above to surviving fleets, otherwise add it to the kore of the tile where the fleet died
        for(Entry<String, ArrayList<Pair<Integer, Double>>> entry : toDistribute.entrySet()) {
            String fleetId = entry.getKey();
            ArrayList<Pair<Integer, Double>> resoureFromLocs = entry.getValue();
            resoureFromLocs.forEach(p -> {
                int cellIdx = p.first;
                double kore = p.second;
                if (!board.fleets.containsKey(fleetId)) {
                    board.cells[cellIdx].kore += kore;
                } else {
                    Fleet fleet = board.fleets.get(fleetId);
                    fleet.kore += kore;
                }
            });
        }

        // Collect kore from cells into fleets
        for (Fleet fleet : board.fleets.values() ) {
            Cell cell = fleet.cell();
            double deltaKore = roundToThreePlaces(cell.kore * Math.min(fleet.collectionRate(), .99));
            if (deltaKore > 0) {
                fleet.kore += deltaKore;
                cell.kore -= deltaKore;
            }
        }

        // Regenerate kore in cells
        for (Cell cell : board.cells) {
            if (cell.fleetId.equals("") && cell.shipyardId.equals("")) {
                if (cell.kore < configuration.maxRegenCellKore) {
                    double nextKore = roundToThreePlaces(cell.kore * (1 + configuration.regenRate));
                    cell.kore = nextKore;
                }
            }
        }

        board.step += 1;

        return board;
    }

    private static double roundToThreePlaces(double num) {
        return Math.round(num * 1000.0) / 1000.0;
    }

}
