import java.util.Scanner;
import java.util.Map.Entry;

import kore.*;

public class Bot {
    private final static Scanner scanner = new Scanner(System.in);
    public static void main(final String[] args) throws Exception {
        while (true) {
            /** Do not edit! **/
            String rawObservation = scanner.nextLine();
            String rawConfiguration = scanner.nextLine();
            Board board = new Board(rawObservation, rawConfiguration);
            /** end do not edit */

            Player me = board.currentPlayer();
            int turn = board.step;
            int spawnCost = board.configuration.spawnCost;
            double koreLeft = me.kore;

            for (Shipyard shipyard : me.shipyards()) {
                if (shipyard.shipCount > 10) {
                    Direction dir = Direction.fromIndex(turn % 4);
                    ShipyardAction action = ShipyardAction.launchFleetWithFlightPlan(2, dir.toChar());
                    shipyard.setNextAction(action);
                } else if (koreLeft > spawnCost * shipyard.maxSpawn()) {
                    ShipyardAction action = ShipyardAction.spawnShips(shipyard.maxSpawn());
                    shipyard.setNextAction(action);
                    koreLeft -= spawnCost * shipyard.maxSpawn();
                } else if (koreLeft > spawnCost) {
                    ShipyardAction action = ShipyardAction.spawnShips(1);
                    shipyard.setNextAction(action);
                    koreLeft -= spawnCost;
                }
            }

            /** AI Code Goes Above! **/

            /** Do not edit! **/
            StringBuilder commandBuilder = new StringBuilder("");
            boolean first = true;
            for (Entry<String, ShipyardAction> entry : board.currentPlayer().nextActions().entrySet()) {
                if (first) {
                    first = false;
                } else {
                    commandBuilder.append(",");
                }
                commandBuilder.append(String.format("%s:%s", entry.getKey(), entry.getValue().toString()));
            }
            System.out.println(commandBuilder.toString());
            System.out.flush();
        }
    }
}
