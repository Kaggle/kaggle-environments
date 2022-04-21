import {Board} from "./kore/Board";
import { Direction } from "./kore/Direction";
import { ShipyardAction } from "./kore/ShipyardAction";
import { KoreIO } from "./kore/KoreIO";

export const tick = async (board: Board): Promise<Board> => {
    const me = board.currentPlayer;
    const spawnCost = board.configuration.spawnCost;
    const convertCost = board.configuration.convertCost;
    let remainingKore = me.kore;

    // the following code is mostly auto-generated using GitHub co-pilot
    // using miner Python code and instruction "convert python into javascript" as comment prompts
    for (let shipyard of me.shipyards) {
        if(remainingKore > 1000 && shipyard.maxSpawn > 5) {
            if(shipyard.shipCount >= convertCost + 10) {
                const gap1 = getRandomInt(3, 9);
                const gap2 = getRandomInt(3, 9);
                const startDir = Math.floor(Math.random() * 4);
                let flightPlan = Direction.listDirections()[startDir].toChar() + gap1;
                const nextDir = (startDir + 1) % 4;
                flightPlan += Direction.listDirections()[nextDir].toChar() + gap2;
                const nextDir2 = (nextDir + 1) % 4;
                flightPlan += Direction.listDirections()[nextDir2].toChar();
                shipyard.setNextAction(ShipyardAction.launchFleetWithFlightPlan(Math.min(convertCost + 10, Math.floor(shipyard.shipCount / 2)), flightPlan));
            } else if(remainingKore >= spawnCost) {
                remainingKore -= spawnCost;
                shipyard.setNextAction(ShipyardAction.spawnShips(Math.min(shipyard.maxSpawn, Math.floor(remainingKore / spawnCost))));
            }
        } else if(shipyard.shipCount >= 21) {
            const gap1 = getRandomInt(3, 9);
            const gap2 = getRandomInt(3, 9);
            const startDir = Math.floor(Math.random() * 4);
            let flightPlan = Direction.listDirections()[startDir].toChar() + gap1;
            const nextDir = (startDir + 1) % 4;
            flightPlan += Direction.listDirections()[nextDir].toChar() + gap2;
            const nextDir2 = (nextDir + 1) % 4;
            flightPlan += Direction.listDirections()[nextDir2].toChar() + gap1;
            const nextDir3 = (nextDir2 + 1) % 4;
            flightPlan += Direction.listDirections()[nextDir3].toChar();
            shipyard.setNextAction(ShipyardAction.launchFleetWithFlightPlan(21, flightPlan));
        } else if(remainingKore > board.configuration.spawnCost * shipyard.maxSpawn) {
            remainingKore -= board.configuration.spawnCost;
            if(remainingKore >= spawnCost) {
                shipyard.setNextAction(ShipyardAction.spawnShips(Math.min(shipyard.maxSpawn, Math.floor(remainingKore / spawnCost))));
            }
        } else if(shipyard.shipCount >= 2) {
            const dirStr = Direction.randomDirection().toChar();
            shipyard.setNextAction(ShipyardAction.launchFleetWithFlightPlan(2, dirStr));
        }
    }

    // nextActions will be pulled off of your shipyards
    return board;
}

const io = new KoreIO();
io.run(tick);

function getRandomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
}