import { describe } from 'mocha';
import { expect } from 'chai';

import * as fs from 'fs';

import { Board } from "../kore/Board";
import { Direction } from '../kore/Direction';
import { Point } from '../kore/Point';
import { Fleet } from '../kore/Fleet';
import { Shipyard } from '../kore/Shipyard';
import { ShipyardAction } from '../kore/ShipyardAction';

const getStarterBoard = () => {
    const rawConfig = fs.readFileSync('./test/configuration.json','utf8');
    const rawObs = fs.readFileSync('./test/observation.json', 'utf8');
    
    return Board.fromRaw(rawObs, rawConfig);
}

describe('Board', () =>  {
    it('init works correctly', () => {
        const rawConfig = fs.readFileSync('./test/configuration.json','utf8');
        const rawObs = fs.readFileSync('./test/observation.json', 'utf8');
        
        const board = Board.fromRaw(rawObs, rawConfig);

        expect(board.step).to.equal(16);
    })

    describe('kore', () => {
        it('regenerates', () => {
            const board = getStarterBoard();

            const nextBoard = board.next();

            expect(board.cells[3].kore).to.be.lessThan(nextBoard.cells[3].kore);
        })
        it('is picked up by fleets', () => {
            const board = getStarterBoard();

            const p = new Point(10, 10);
            board.getCellAtPosition(p).kore = 100;
            board.getCellAtPosition(p.add(Direction.SOUTH)).kore = 100;

            const fleet = new Fleet("test-fleet", 100, Direction.SOUTH, p, 100.0, "8N", 0, board);

            board.addFleet(fleet);

            const nextBoard = board.next();

            const nextFleet = nextBoard.getFleetAtPoint(p.add(Direction.SOUTH));
            expect(nextFleet.kore).to.be.greaterThan(fleet.kore);

        })
    })


    describe('spawnShips', () => {
        it('spawnsShips', () => {
            const board = getStarterBoard();
            const shipyardId = board.players[0].shipyardIds[0];
            const shipyard = board.shipyards.get(shipyardId);

            shipyard.setNextAction(ShipyardAction.spawnShips(1));

            const nextBoard = board.next();
            const nextShipyard = nextBoard.shipyards.get(shipyardId);

            expect(shipyard.shipCount).to.equal(0);
            expect(nextShipyard.shipCount).to.equal(1);
        })

        it('can spawn 0 ships', () => {
            const board = getStarterBoard();
            const shipyardId = board.players[0].shipyardIds[0];
            const shipyard = board.shipyards.get(shipyardId);

            shipyard.setNextAction(ShipyardAction.spawnShips(0));

            const nextBoard = board.next();
            const nextShipyard = nextBoard.shipyards.get(shipyardId);

            expect(shipyard.shipCount).to.equal(0);
            expect(nextShipyard.shipCount).to.equal(0);
        })
    })

    describe('launchShips', () => {
        it('launches', () => {
            const board = getStarterBoard();
            const shipyardId = board.players[0].shipyardIds[0];
            const shipyard = board.shipyards.get(shipyardId);
            shipyard.shipCount = 100;

            shipyard.setNextAction(ShipyardAction.launchFleetWithFlightPlan(10, "N"));

            const nextBoard = board.next();
            const nextShipyard = nextBoard.shipyards.get(shipyardId);
            const launchedFleet = nextBoard.getFleetAtPoint(shipyard.position.add(Direction.NORTH));

            expect(shipyard.shipCount).to.equal(100);
            expect(nextShipyard.shipCount).to.equal(90);
            expect(!!launchedFleet).to.be.true;
            expect(launchedFleet.shipCount).to.equal(10);
        })

        it('can spawn 0 ships', () => {
            const board = getStarterBoard();
            const shipyardId = board.players[0].shipyardIds[0];
            const shipyard = board.shipyards.get(shipyardId);

            shipyard.setNextAction(ShipyardAction.spawnShips(0));

            const nextBoard = board.next();
            const nextShipyard = nextBoard.shipyards.get(shipyardId);

            expect(shipyard.shipCount).to.equal(0);
            expect(nextShipyard.shipCount).to.equal(0);
        })
    })


    describe('flight plan', () => {
        it('decrements', () => {
            const board = getStarterBoard();

            const p = new Point(10, 11);

            const f = new Fleet("test-fleet", 10, Direction.SOUTH, p, 100.0, "8N", 0, board);

            board.addFleet(f);

            const nextBoard = board.next();

            const nextFleet = nextBoard.getFleetAtPoint(new Point(10, 10));
            expect(nextFleet.direction.toChar()).to.equal(Direction.SOUTH.toChar());
            expect(nextFleet.flightPlan).to.equal("7N");
        })

        it('changed direction', () => {
            const board = getStarterBoard();

            const p = new Point(10, 11);

            const f = new Fleet("test-fleet", 10, Direction.NORTH, p, 100.0, "S", 0, board);

            board.addFleet(f);

            const nextBoard = board.next();

            const nextFleet = nextBoard.getFleetAtPoint(new Point(10, 10));
            expect(nextFleet.direction.toChar()).to.equal(Direction.SOUTH.toChar());
            expect(nextFleet.flightPlan).to.equal("");
        })

        it('converts to shipyard', () => {
            const board = getStarterBoard();

            const p = new Point(10, 11);

            const f = new Fleet("test-fleet", 10, Direction.SOUTH, p, 100.0, "C", 0, board);

            board.addFleet(f);

            const nextBoard = board.next();

            expect(!!nextBoard.getShipyardAtPoint(p)).to.be.false;
            const nextFleet = nextBoard.getFleetAtPoint(p.add(Direction.SOUTH));
            expect(nextFleet.playerId).to.equal(0);
            expect(nextFleet.shipCount).to.equal(10);
            expect(nextFleet.direction.toChar()).to.equal(Direction.SOUTH.toChar());
        })

        it('does not convert to shipyard if not enough ships', () => {
            const board = getStarterBoard();

            const p = new Point(10, 11);

            const f = new Fleet("test-fleet", 100, Direction.NORTH, p, 100.0, "C", 0, board);

            board.addFleet(f);

            const nextBoard = board.next();

            expect(!!nextBoard.getShipyardAtPoint(p)).to.be.true;
            const nextShipyard = nextBoard.getShipyardAtPoint(p);
            expect(nextShipyard.playerId).to.equal(0);
            expect(nextShipyard.shipCount).to.equal(50);
        })

        it('works with multiple converts ', () => {
            let board = getStarterBoard();

            const p = new Point(10, 11);

            const f = new Fleet("test-fleet", 100, Direction.NORTH, p, 100.0, "CCC", 0, board);

            board.addFleet(f);

            board = board.next();
            board = board.next();
            board = board.next();

            expect(true).to.be.true;
        })
    })

    describe('coalescence', () => {
        it('correctly joins allied fleets', () => {
            const board = getStarterBoard();

            const p1 = new Point(10, 11);
            const p2 = new Point(10, 9);

            const f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
            const f2 = new Fleet("f2", 11, Direction.NORTH, p2, 100.0, "", 0, board);

            board.addFleet(f1);
            board.addFleet(f2);

            const nextBoard = board.next();

            const combinedFleet = nextBoard.getFleetAtPoint(new Point(10, 10));
            expect(combinedFleet.direction.toChar()).to.equal(Direction.NORTH.toChar());
            expect(combinedFleet.shipCount).to.equal(21);
        })

        it('joins on first tie break', () => {
            const board = getStarterBoard();

            const p1 = new Point(10, 11);
            const p2 = new Point(10, 9);

            const f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
            const f2 = new Fleet("f2", 10, Direction.NORTH, p2, 101.0, "10S", 0, board);

            board.addFleet(f1);
            board.addFleet(f2);

            const nextBoard = board.next();

            const combinedFleet = nextBoard.getFleetAtPoint(new Point(10, 10));
            expect(combinedFleet.direction.toChar()).to.equal(Direction.NORTH.toChar());
            expect(combinedFleet.shipCount).to.equal(20);
            expect(combinedFleet.flightPlan).to.equal("9S");
        })

        it('joins on second tie break', () => {
            const board = getStarterBoard();

            const p1 = new Point(10, 11);
            const p2 = new Point(10, 9);

            const f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
            const f2 = new Fleet("f2", 10, Direction.NORTH, p2, 100.0, "", 0, board);

            board.addFleet(f1);
            board.addFleet(f2);

            const nextBoard = board.next();

            const combinedFleet = nextBoard.getFleetAtPoint(new Point(10, 10));
            expect(combinedFleet.direction.toChar()).to.equal(Direction.NORTH.toChar());
            expect(combinedFleet.shipCount).to.equal(20);
        })
    })

    describe('fleet battles', () => {
        it('resolve correctly with a winner', () => {
            const board = getStarterBoard();

            const p1 = new Point(9, 11);
            const p2 = new Point(10, 9);

            const f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
            const f2 = new Fleet("f2", 11, Direction.NORTH, p2, 100.0, "", 1, board);

            board.addFleet(f1);
            board.addFleet(f2);

            const nextBoard = board.next();

            expect(!!nextBoard.getFleetAtPoint(new Point(10, 10))).to.be.true;
            const survivingFleet = nextBoard.getFleetAtPoint(new Point(10, 10));
            expect(survivingFleet.direction.toChar()).to.equal(Direction.NORTH.toChar());
            expect(survivingFleet.shipCount).to.equal(1);
            expect(survivingFleet.playerId).to.equal(1);
        })

    
        it('resolve correctly with a tie', () => {
            const board = getStarterBoard();

            const p1 = new Point(10, 11);
            const p2 = new Point(9, 9);

            const f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
            const f2 = new Fleet("f2", 10, Direction.NORTH, p2, 100.0, "", 1, board);

            board.addFleet(f1);
            board.addFleet(f2);

            const nextBoard = board.next();

            const collision = new Point(10, 10);
            expect(!!nextBoard.getFleetAtPoint(collision)).to.be.false;
            expect(board.getCellAtPosition(collision).kore + 100).to.be.lessThan(nextBoard.getCellAtPosition(collision).kore);
        })
    })

    
    describe('fleet collisions', () => {
        it('resolve correctly with a winner', () => {
            const board = getStarterBoard();

            const p1 = new Point(10, 11);
            const p2 = new Point(10, 9);

            const f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
            const f2 = new Fleet("f2", 11, Direction.NORTH, p2, 100.0, "", 1, board);

            board.addFleet(f1);
            board.addFleet(f2);

            const nextBoard = board.next();

            expect(!!nextBoard.getFleetAtPoint(new Point(10, 10))).to.be.true;
            const survivingFleet = nextBoard.getFleetAtPoint(new Point(10, 10));
            expect(survivingFleet.direction.toChar()).to.equal(Direction.NORTH.toChar());
            expect(survivingFleet.shipCount).to.equal(1);
            expect(survivingFleet.playerId).to.equal(1);
        })

    
        it('resolve correctly with multiples from one player', () => {
            const board = getStarterBoard();

            const p1 = new Point(10, 11);
            const p2 = new Point(10, 9);
            const p3 = new Point(9, 10);

            const f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
            const f2 = new Fleet("f2", 11, Direction.NORTH, p2, 100.0, "", 1, board);
            const f3 = new Fleet("f3", 2, Direction.EAST, p3, 100.0, "", 0, board);

            board.addFleet(f1);
            board.addFleet(f2);
            board.addFleet(f3);

            const nextBoard = board.next();

            expect(!!nextBoard.getFleetAtPoint(new Point(10, 10))).to.be.true;
            const survivingFleet = nextBoard.getFleetAtPoint(new Point(10, 10));
            expect(survivingFleet.direction.toChar()).to.equal(Direction.SOUTH.toChar());
            expect(survivingFleet.shipCount).to.equal(1);
            expect(survivingFleet.playerId).to.equal(0);
        })

    
        it('resolve correctly with multiples players', () => {
            const board = getStarterBoard();

            const p1 = new Point(10, 11);
            const p2 = new Point(10, 9);
            const p3 = new Point(9, 10);

            const f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
            const f2 = new Fleet("f2", 11, Direction.NORTH, p2, 100.0, "", 1, board);
            const f3 = new Fleet("f3", 2, Direction.EAST, p3, 100.0, "", 2, board);

            board.addFleet(f1);
            board.addFleet(f2);
            board.addFleet(f3);

            const nextBoard = board.next();

            expect(!!nextBoard.getFleetAtPoint(new Point(10, 10))).to.be.true;
            const survivingFleet = nextBoard.getFleetAtPoint(new Point(10, 10));
            expect(survivingFleet.direction.toChar()).to.equal(Direction.NORTH.toChar());
            expect(survivingFleet.shipCount).to.equal(1);
            expect(survivingFleet.playerId).to.equal(1);
        })

    
        it('resolve correctly with a tie', () => {
            const board = getStarterBoard();

            const p1 = new Point(10, 11);
            const p2 = new Point(10, 9);

            const f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
            const f2 = new Fleet("f2", 10, Direction.NORTH, p2, 100.0, "", 1, board);

            board.addFleet(f1);
            board.addFleet(f2);

            const nextBoard = board.next();

            const collision = new Point(10, 10);
            expect(!!nextBoard.getFleetAtPoint(new Point(10, 10))).to.be.false;
            expect(board.getCellAtPosition(collision).kore + 100).to.be.lessThan(nextBoard.getCellAtPosition(collision).kore);
        })
    })

    
    describe('fleet/shipyard collisions', () => {
        it('works when they are allied', () => {
            const board = getStarterBoard();

            const p1 = new Point(10, 11);
            const p2 = new Point(10, 10);

            const f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
            const s1 = new Shipyard("s1", 0, p2, 0, 100, board, null);

            board.addFleet(f1);
            board.addShipyard(s1);

            const nextBoard = board.next();

            expect(!!nextBoard.getFleetAtPoint(p2)).to.be.false;
            expect(!!nextBoard.getShipyardAtPoint(p2)).to.be.true;
            const nextShipyard = nextBoard.getShipyardAtPoint(p2);
            expect(nextShipyard.shipCount).to.equal(10);
            expect(nextShipyard.playerId).to.equal(0);
            expect(s1.turnsControlled).to.equal(100);
            expect(nextShipyard.turnsControlled).to.equal(101);
            expect(nextBoard.players[0].kore).to.equal(board.players[0].kore + 100);
        })

    
        it('smaller fleet does not take over larger shipyard', () => {
            const board = getStarterBoard();

            const p1 = new Point(10, 11);
            const p2 = new Point(10, 10);

            const f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 1, board);
            const s1 = new Shipyard("s1", 100, p2, 0, 100, board, null);

            board.addFleet(f1);
            board.addShipyard(s1);

            const nextBoard = board.next();

            expect(!!nextBoard.getFleetAtPoint(p2)).to.be.false;
            expect(!!nextBoard.getShipyardAtPoint(p2)).to.be.true;
            const nextShipyard = nextBoard.getShipyardAtPoint(p2);
            expect(nextShipyard.shipCount).to.equal(90);
            expect(nextShipyard.playerId).to.equal(0);
            expect(s1.turnsControlled).to.equal(100);
            expect(nextShipyard.turnsControlled).to.equal(101);
            expect(nextBoard.players[0].kore).to.equal(board.players[0].kore + 100);
        })

    
        it('equal fleet does not take over larger shipyard', () => {
            const board = getStarterBoard();

            const p1 = new Point(10, 11);
            const p2 = new Point(10, 10);

            const f1 = new Fleet("f1", 100, Direction.SOUTH, p1, 100.0, "", 1, board);
            const s1 = new Shipyard("s1", 100, p2, 0, 100, board, null);

            board.addFleet(f1);
            board.addShipyard(s1);

            const nextBoard = board.next();

            expect(!!nextBoard.getFleetAtPoint(p2)).to.be.false;
            expect(!!nextBoard.getShipyardAtPoint(p2)).to.be.true;
            const nextShipyard = nextBoard.getShipyardAtPoint(p2);
            expect(nextShipyard.shipCount).to.equal(0);
            expect(nextShipyard.playerId).to.equal(0);
            expect(s1.turnsControlled).to.equal(100);
            expect(nextShipyard.turnsControlled).to.equal(101);
            expect(nextBoard.players[0].kore).to.equal(board.players[0].kore + 100);
        })

    
        it('larger fleet does take over smaller shipyard', () => {
            const board = getStarterBoard();

            const p1 = new Point(10, 11);
            const p2 = new Point(10, 10);

            const f1 = new Fleet("f1", 110, Direction.SOUTH, p1, 100.0, "", 1, board);
            const s1 = new Shipyard("s1", 100, p2, 0, 100, board, null);

            board.addFleet(f1);
            board.addShipyard(s1);

            const nextBoard = board.next();

            expect(!!nextBoard.getFleetAtPoint(p2)).to.be.false;
            expect(!!nextBoard.getShipyardAtPoint(p2)).to.be.true;
            const nextShipyard = nextBoard.getShipyardAtPoint(p2);
            expect(nextShipyard.shipCount).to.equal(10);
            expect(nextShipyard.playerId).to.equal(1);
            expect(s1.turnsControlled).to.equal(100);
            expect(nextShipyard.turnsControlled).to.equal(1);
            expect(nextBoard.players[1].kore).to.equal(board.players[1].kore + 100);
        })
    })

    describe('fleet adjenct damage', () => {
        it('resolves correctly when both die', () => {
            const board = getStarterBoard();

            const p1 = new Point(10, 11);
            const f1 = new Fleet("f1", 100, Direction.NORTH, p1, 100.0, "", 0, board);
            const p1Kore = board.getCellAtPosition(p1.add(Direction.NORTH)).kore;
            board.addFleet(f1);

            const p2 = p1.add(Direction.NORTH).add(Direction.NORTH).add(Direction.EAST);
            const f2 = new Fleet("f2", 100, Direction.SOUTH, p2, 100.0, "", 1, board);
            board.addFleet(f2);

            const nextBoard = board.next();

            const p1NextKore = nextBoard.getCellAtPosition(p1.add(Direction.NORTH)).kore;
            expect(!nextBoard.fleets.has("f1")).to.be.true;
            expect(p1Kore + 100 < p1NextKore).to.be.true;
        })

        it('resolves correctly when only one dies', () => {
            const board = getStarterBoard();

            const p1 = new Point(10, 11);
            const f1 = new Fleet("f1", 50, Direction.NORTH, p1, 100.0, "", 0, board);
            const p1Kore = board.getCellAtPosition(p1.add(Direction.NORTH)).kore;
            board.addFleet(f1);

            const p2 = p1.add(Direction.NORTH).add(Direction.NORTH).add(Direction.EAST);
            const f2 = new Fleet("f2", 100, Direction.SOUTH, p2, 100.0, "", 1, board);
            board.addFleet(f2);

            const nextBoard = board.next();

            const p1NextKore = nextBoard.getCellAtPosition(p1.add(Direction.NORTH)).kore;
            expect(!nextBoard.fleets.has("f1")).to.be.true;
            expect(p1Kore + 50 < p1NextKore && p1Kore + 55 > p1NextKore).to.be.true;

            const f2next = nextBoard.fleets.get("f2");
            expect(f2.kore + 50 <= f2next.kore && f2.kore + 55 > f2next.kore).to.be.true;
        });
    });


});