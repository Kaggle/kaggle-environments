package test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Optional;

import org.junit.Assert;
import org.junit.Test;

import kore.Board;
import kore.Direction;
import kore.Fleet;
import kore.Point;
import kore.Shipyard;
import kore.ShipyardAction;

public class BoardTest {

    @Test
    public void givenValidConfigAndObservation_createsSuccessful() throws IOException {
        Path configPath = Paths.get("bin", "test", "configuration.json");
        String rawConfig = Files.readString(configPath);        
        Path obsPath = Paths.get("bin", "test", "observation.json");
        String rawObs = Files.readString(obsPath);        
        
        Board board = new Board(rawObs, rawConfig);

        Assert.assertEquals(16, board.step);
    }

    @Test
    public void koreRegenerates() throws IOException {
        Board board = getStarterBoard();

        Board nextBoard = board.next();

        Assert.assertTrue("cells should regen", board.cells[3].kore < nextBoard.cells[3].kore);
    }

    @Test
    public void spawnShips_spawnsShips() throws IOException {
        Board board = getStarterBoard();

        String shipyardId = board.players[0].shipyardIds.get(0);
        Shipyard shipyard = board.shipyards.get(shipyardId);

        shipyard.setNextAction(ShipyardAction.spawnShips(1));

        Board nextBoard = board.next();
        Shipyard nextShipyard = nextBoard.shipyards.get(shipyardId);

        Assert.assertEquals(0, shipyard.shipCount);
        Assert.assertEquals(1, nextShipyard.shipCount);
    }

    @Test
    public void spawnShips_shawnZeroShipsDoesNothing() throws IOException {
        Board board = getStarterBoard();

        String shipyardId = board.players[0].shipyardIds.get(0);
        Shipyard shipyard = board.shipyards.get(shipyardId);

        shipyard.setNextAction(ShipyardAction.spawnShips(0));

        Board nextBoard = board.next();
        Shipyard nextShipyard = nextBoard.shipyards.get(shipyardId);

        Assert.assertEquals(0, shipyard.shipCount);
        Assert.assertEquals(0, nextShipyard.shipCount);
        Assert.assertEquals(board.players[0].kore, nextBoard.players[0].kore, .01);
    }

    @Test
    public void launchShips_launchesCorrectly() throws IOException {
        Board board = getStarterBoard();

        String shipyardId = board.players[0].shipyardIds.get(0);
        Shipyard shipyard = board.shipyards.get(shipyardId);
        shipyard.shipCount = 100;

        shipyard.setNextAction(ShipyardAction.launchFleetWithFlightPlan(10, "N"));

        Board nextBoard = board.next();
        Shipyard nextShipyard = nextBoard.shipyards.get(shipyardId);

        Assert.assertEquals(100, shipyard.shipCount);
        Assert.assertEquals(90, nextShipyard.shipCount);

        Assert.assertTrue("should have launched a fleet", nextBoard.getFleetAtPoint(shipyard.position.add(Direction.NORTH)).isPresent());
        Fleet launchedFleet = nextBoard.getFleetAtPoint(shipyard.position.add(Direction.NORTH)).get();
        Assert.assertEquals(10, launchedFleet.shipCount);
    }

    @Test
    public void fleetsCoalescence() throws IOException {
        Board board = getStarterBoard();

        Point p = new Point(10, 10);

        Fleet f1 = new Fleet("f1", 100, Direction.SOUTH, p.add(Direction.NORTH), 100.0, "", 0, board);
        board.addFleet(f1);
        Fleet f2 = new Fleet("f2", 60, Direction.NORTH, p.add(Direction.SOUTH), 100.0, "", 0, board);
        board.addFleet(f2);
        Fleet f3 = new Fleet("f3", 60, Direction.EAST, p.add(Direction.WEST), 100.0, "", 0, board);
        board.addFleet(f3);

        Board nextBoard = board.next();

        Fleet nextFleet = nextBoard.getFleetAtPoint(p).get();
        Assert.assertTrue(nextFleet.id.equals("f1"));
    }

    @Test
    public void fleetsPickUpKore() throws IOException {
        Board board = getStarterBoard();

        Point p = new Point(10, 10);
        board.getCellAtPosition(p).kore = 100;
        board.getCellAtPosition(p.add(Direction.SOUTH)).kore = 100;

        Fleet fleet = new Fleet("test-fleet", 100, Direction.SOUTH, p, 100.0, "8N", 0, board);

        board.addFleet(fleet);

        Board nextBoard = board.next();

        Fleet nextFleet = nextBoard.getFleetAtPoint(p.add(Direction.SOUTH)).get();
        Assert.assertTrue(nextFleet.kore > fleet.kore);
    }

    @Test
    public void updatesFlightPlan_decrements() throws IOException {
        Board board = getStarterBoard();

        Point p = new Point(10, 11);

        Fleet f = new Fleet("test-fleet", 10, Direction.SOUTH, p, 100.0, "8N", 0, board);

        board.addFleet(f);

        Board nextBoard = board.next();

        Fleet nextFleet = nextBoard.getFleetAtPoint(new Point(10, 10)).get();
        Assert.assertEquals(Direction.SOUTH.toChar(), nextFleet.direction.toChar());
        Assert.assertEquals("7N", nextFleet.flightPlan);
    }

    @Test
    public void updatesFlightPlan_changesDirection() throws IOException {
        Board board = getStarterBoard();

        Point p = new Point(10, 11);

        Fleet f = new Fleet("test-fleet", 10, Direction.NORTH, p, 100.0, "S", 0, board);

        board.addFleet(f);

        Board nextBoard = board.next();

        Fleet nextFleet = nextBoard.getFleetAtPoint(new Point(10, 10)).get();
        Assert.assertEquals(Direction.SOUTH.toChar(), nextFleet.direction.toChar());
        Assert.assertEquals("", nextFleet.flightPlan);
    }

    @Test
    public void updatesFlightPlan_convertsToShipyard() throws IOException {
        Board board = getStarterBoard();

        Point p = new Point(10, 11);

        Fleet f = new Fleet("test-fleet", 100, Direction.NORTH, p, 100.0, "C", 0, board);

        board.addFleet(f);

        Board nextBoard = board.next();

        Assert.assertTrue("should have made a shipyard", nextBoard.getShipyardAtPoint(p).isPresent());
        Shipyard nextShipyard = nextBoard.getShipyardAtPoint(p).get();
        Assert.assertEquals(0, nextShipyard.playerId);
        Assert.assertEquals(50, nextShipyard.shipCount);
    }

    @Test
    public void updatesFlightPlan_doesNotConvertIfNotEnoughShips() throws IOException {
        Board board = getStarterBoard();

        Point p = new Point(10, 11);

        Fleet f = new Fleet("test-fleet", 10, Direction.SOUTH, p, 100.0, "C", 0, board);

        board.addFleet(f);

        Board nextBoard = board.next();

        Assert.assertTrue("should not have made a shipyard", nextBoard.getShipyardAtPoint(p).isEmpty());
        Fleet nextFleet = nextBoard.getFleetAtPoint(new Point(10, 10)).get();
        Assert.assertEquals(Direction.SOUTH.toChar(), nextFleet.direction.toChar());
        Assert.assertEquals(10, nextFleet.shipCount);
        Assert.assertEquals("", nextFleet.flightPlan);
    }

    @Test
    public void updatesFlightPlan_worksWithMultipleConverts() throws IOException {
        Board board = getStarterBoard();

        Point p = new Point(10, 11);

        Fleet f = new Fleet("test-fleet", 10, Direction.SOUTH, p, 100.0, "CCC", 0, board);

        board.addFleet(f);

        board = board.next();
        board = board.next();
        board = board.next();

        Assert.assertTrue("should not have crashed", true);
    }

    @Test
    public void correctlyJoinAlliedFleet() throws IOException {
        Board board = getStarterBoard();

        Point p1 = new Point(10, 11);
        Point p2 = new Point(10, 9);

        Fleet f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
        Fleet f2 = new Fleet("f2", 11, Direction.NORTH, p2, 100.0, "", 0, board);

        board.addFleet(f1);
        board.addFleet(f2);

        Board nextBoard = board.next();

        Fleet combinedFleet = nextBoard.getFleetAtPoint(new Point(10, 10)).get();
        Assert.assertEquals(Direction.NORTH.toChar(), combinedFleet.direction.toChar());
        Assert.assertEquals(21, combinedFleet.shipCount);
    }

    @Test
    public void correctlyJoinAlliedFleet_onFirstTieBreak() throws IOException {
        Board board = getStarterBoard();

        Point p1 = new Point(10, 11);
        Point p2 = new Point(10, 9);

        Fleet f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
        Fleet f2 = new Fleet("f2", 10, Direction.NORTH, p2, 101.0, "10S", 0, board);

        board.addFleet(f1);
        board.addFleet(f2);

        Board nextBoard = board.next();

        Fleet combinedFleet = nextBoard.getFleetAtPoint(new Point(10, 10)).get();
        Assert.assertEquals(Direction.NORTH.toChar(), combinedFleet.direction.toChar());
        Assert.assertEquals(20, combinedFleet.shipCount);
        Assert.assertEquals("9S", combinedFleet.flightPlan);
    }

    @Test
    public void correctlyJoinAlliedFleet_onSecondTieBreak() throws IOException {
        Board board = getStarterBoard();

        Point p1 = new Point(10, 11);
        Point p2 = new Point(10, 9);

        Fleet f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
        Fleet f2 = new Fleet("f2", 10, Direction.NORTH, p2, 100.0, "", 0, board);

        board.addFleet(f1);
        board.addFleet(f2);

        Board nextBoard = board.next();

        Fleet combinedFleet = nextBoard.getFleetAtPoint(new Point(10, 10)).get();
        Assert.assertEquals(Direction.NORTH.toChar(), combinedFleet.direction.toChar());
        Assert.assertEquals(20, combinedFleet.shipCount);
    }

    @Test
    public void correctlyResolvesFleetBattles_whenThereIsAWinner() throws IOException {
        Board board = getStarterBoard();

        Point p1 = new Point(9, 11);
        Point p2 = new Point(10, 9);

        Fleet f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
        Fleet f2 = new Fleet("f2", 11, Direction.NORTH, p2, 100.0, "", 1, board);

        board.addFleet(f1);
        board.addFleet(f2);

        Board nextBoard = board.next();

        Assert.assertTrue("should have a surviving fleet", nextBoard.getFleetAtPoint(new Point(10, 10)).isPresent());
        Fleet survivingFleet = nextBoard.getFleetAtPoint(new Point(10, 10)).get();
        Assert.assertEquals(Direction.NORTH.toChar(), survivingFleet.direction.toChar());
        Assert.assertEquals(1, survivingFleet.shipCount);
        Assert.assertEquals(1, survivingFleet.playerId);
    }   

    @Test
    public void correctlyResolvesFleetBattles_whenThereIsATie() throws IOException {
        Board board = getStarterBoard();

        Point p1 = new Point(10, 11);
        Point p2 = new Point(10, 9);

        Fleet f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
        Fleet f2 = new Fleet("f2", 10, Direction.NORTH, p2, 100.0, "", 1, board);

        board.addFleet(f1);
        board.addFleet(f2);

        Board nextBoard = board.next();

        Point collision = new Point(10, 10);
        Assert.assertTrue("should not have a surviving fleet", nextBoard.getFleetAtPoint(collision).isEmpty());
        Assert.assertTrue("should have dropped halite", board.getCellAtPosition(collision).kore + 100 < nextBoard.getCellAtPosition(collision).kore);
    }

    @Test
    public void correctlyResolvesFleetCollisions_whenThereIsAWinner() throws IOException {
        Board board = getStarterBoard();

        Point p1 = new Point(10, 11);
        Point p2 = new Point(10, 9);

        Fleet f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
        Fleet f2 = new Fleet("f2", 11, Direction.NORTH, p2, 100.0, "", 1, board);

        board.addFleet(f1);
        board.addFleet(f2);

        Board nextBoard = board.next();

        Assert.assertTrue("should have a surviving fleet", nextBoard.getFleetAtPoint(new Point(10, 10)).isPresent());
        Fleet survivingFleet = nextBoard.getFleetAtPoint(new Point(10, 10)).get();
        Assert.assertEquals(Direction.NORTH.toChar(), survivingFleet.direction.toChar());
        Assert.assertEquals(1, survivingFleet.shipCount);
        Assert.assertEquals(1, survivingFleet.playerId);
    }   

    @Test
    public void correctlyResolvesFleetCollisions_whenThereAreMultipleFromOnePlayer() throws IOException {
        Board board = getStarterBoard();

        Point p1 = new Point(10, 11);
        Point p2 = new Point(10, 9);
        Point p3 = new Point(9, 10);

        Fleet f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
        Fleet f2 = new Fleet("f2", 11, Direction.NORTH, p2, 100.0, "", 1, board);
        Fleet f3 = new Fleet("f3", 2, Direction.EAST, p3, 100.0, "", 0, board);

        board.addFleet(f1);
        board.addFleet(f2);
        board.addFleet(f3);

        Board nextBoard = board.next();

        Assert.assertTrue("should have a surviving fleet", nextBoard.getFleetAtPoint(new Point(10, 10)).isPresent());
        Fleet survivingFleet = nextBoard.getFleetAtPoint(new Point(10, 10)).get();
        Assert.assertEquals(Direction.SOUTH.toChar(), survivingFleet.direction.toChar());
        Assert.assertEquals(1, survivingFleet.shipCount);
        Assert.assertEquals(0, survivingFleet.playerId);
    }

    @Test
    public void correctlyResolvesFleetCollisions_whenThereAreThreeFleets() throws IOException {
        Board board = getStarterBoard();

        Point p1 = new Point(10, 11);
        Point p2 = new Point(10, 9);
        Point p3 = new Point(9, 10);

        Fleet f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
        Fleet f2 = new Fleet("f2", 11, Direction.NORTH, p2, 100.0, "", 1, board);
        Fleet f3 = new Fleet("f3", 2, Direction.EAST, p3, 100.0, "", 2, board);

        board.addFleet(f1);
        board.addFleet(f2);
        board.addFleet(f3);

        Board nextBoard = board.next();

        Assert.assertTrue("should have a surviving fleet", nextBoard.getFleetAtPoint(new Point(10, 10)).isPresent());
        Fleet survivingFleet = nextBoard.getFleetAtPoint(new Point(10, 10)).get();
        Assert.assertEquals(Direction.NORTH.toChar(), survivingFleet.direction.toChar());
        Assert.assertEquals(1, survivingFleet.shipCount);
        Assert.assertEquals(1, survivingFleet.playerId);
    }   

    @Test
    public void correctlyResolvesFleetCollisions_whenThereIsATie() throws IOException {
        Board board = getStarterBoard();

        Point p1 = new Point(10, 11);
        Point p2 = new Point(10, 9);

        Fleet f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
        Fleet f2 = new Fleet("f2", 10, Direction.NORTH, p2, 100.0, "", 1, board);

        board.addFleet(f1);
        board.addFleet(f2);

        Board nextBoard = board.next();

        Point collision = new Point(10, 10);
        Assert.assertTrue("should not have a surviving fleet", nextBoard.getFleetAtPoint(collision).isEmpty());
        Assert.assertTrue("should have dropped halite", board.getCellAtPosition(collision).kore + 100 < nextBoard.getCellAtPosition(collision).kore);
    }

    @Test
    public void fleetShipyardCollision_worksWhenTheyAreAllied() throws IOException {
        Board board = getStarterBoard();

        Point p1 = new Point(10, 11);
        Point p2 = new Point(10, 10);

        Fleet f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 0, board);
        Shipyard s1 = new Shipyard("s1", 0, p2, 0, 100, board, Optional.empty());

        board.addFleet(f1);
        board.addShipyard(s1);

        Board nextBoard = board.next();

        Assert.assertTrue("should not have a fleet on the square", nextBoard.getFleetAtPoint(p2).isEmpty());
        Assert.assertTrue("should have a shipyard on the square", nextBoard.getShipyardAtPoint(p2).isPresent());
        Shipyard nextShipyard = nextBoard.getShipyardAtPoint(p2).get();
        Assert.assertEquals(10, nextShipyard.shipCount);
        Assert.assertEquals(0, nextShipyard.playerId);
        Assert.assertEquals(100, s1.turnsControlled);
        Assert.assertEquals(101, nextShipyard.turnsControlled);
        Assert.assertEquals(nextBoard.players[0].kore, board.players[0].kore + 100, .01);
    }

    @Test
    public void fleetShipyardCollision_smallerFleetDoesNotTakeOverLargerShipyard() throws IOException {
        Board board = getStarterBoard();

        Point p1 = new Point(10, 11);
        Point p2 = new Point(10, 10);

        Fleet f1 = new Fleet("f1", 10, Direction.SOUTH, p1, 100.0, "", 1, board);
        Shipyard s1 = new Shipyard("s1", 100, p2, 0, 100, board, Optional.empty());

        board.addFleet(f1);
        board.addShipyard(s1);

        Board nextBoard = board.next();

        Assert.assertTrue("should not have a fleet on the square", nextBoard.getFleetAtPoint(p2).isEmpty());
        Assert.assertTrue("should have a shipyard on the square", nextBoard.getShipyardAtPoint(p2).isPresent());
        Shipyard nextShipyard = nextBoard.getShipyardAtPoint(p2).get();
        Assert.assertEquals(90, nextShipyard.shipCount);
        Assert.assertEquals(0, nextShipyard.playerId);
        Assert.assertEquals(100, s1.turnsControlled);
        Assert.assertEquals(101, nextShipyard.turnsControlled);
        Assert.assertEquals(nextBoard.players[0].kore, board.players[0].kore + 100, .01);
    }

    @Test
    public void fleetShipyardCollision_equalFleetDoesNotTakeOverLargerShipyard() throws IOException {
        Board board = getStarterBoard();

        Point p1 = new Point(10, 11);
        Point p2 = new Point(10, 10);

        Fleet f1 = new Fleet("f1", 100, Direction.SOUTH, p1, 100.0, "", 1, board);
        Shipyard s1 = new Shipyard("s1", 100, p2, 0, 100, board, Optional.empty());

        board.addFleet(f1);
        board.addShipyard(s1);

        Board nextBoard = board.next();

        Assert.assertTrue("should not have a fleet on the square", nextBoard.getFleetAtPoint(p2).isEmpty());
        Assert.assertTrue("should have a shipyard on the square", nextBoard.getShipyardAtPoint(p2).isPresent());
        Shipyard nextShipyard = nextBoard.getShipyardAtPoint(p2).get();
        Assert.assertEquals(0, nextShipyard.shipCount);
        Assert.assertEquals(0, nextShipyard.playerId);
        Assert.assertEquals(100, s1.turnsControlled);
        Assert.assertEquals(101, nextShipyard.turnsControlled);
        Assert.assertEquals(nextBoard.players[0].kore, board.players[0].kore + 100, .01);
    }

    @Test
    public void fleetShipyardCollision_largerFleetDoesTakeOverSmallerShipyard() throws IOException {
        Board board = getStarterBoard();

        Point p1 = new Point(10, 11);
        Point p2 = new Point(10, 10);

        Fleet f1 = new Fleet("f1", 110, Direction.SOUTH, p1, 100.0, "", 1, board);
        Shipyard s1 = new Shipyard("s1", 100, p2, 0, 100, board, Optional.empty());

        board.addFleet(f1);
        board.addShipyard(s1);

        Board nextBoard = board.next();

        Assert.assertTrue("should not have a fleet on the square", nextBoard.getFleetAtPoint(p2).isEmpty());
        Assert.assertTrue("should have a shipyard on the square", nextBoard.getShipyardAtPoint(p2).isPresent());
        Shipyard nextShipyard = nextBoard.getShipyardAtPoint(p2).get();
        Assert.assertEquals(10, nextShipyard.shipCount);
        Assert.assertEquals(1, nextShipyard.playerId);
        Assert.assertEquals(100, s1.turnsControlled);
        Assert.assertEquals(1, nextShipyard.turnsControlled);
        Assert.assertEquals(nextBoard.players[1].kore, board.players[1].kore + 100, .01);
    }

    @Test
    public void fleetAdjacentBattle_dropsKoreCorrectlyWhenBothDie() throws IOException {
        Board board = getStarterBoard();

        Point p1 = new Point(10, 11);
        Fleet f1 = new Fleet("f1", 100, Direction.NORTH, p1, 100.0, "", 0, board);
        double p1Kore = board.getCellAtPosition(p1.add(Direction.NORTH)).kore;
        board.addFleet(f1);

        Point p2 = p1.add(Direction.NORTH).add(Direction.NORTH).add(Direction.EAST);
        Fleet f2 = new Fleet("f2", 100, Direction.SOUTH, p2, 100.0, "", 1, board);
        board.addFleet(f2);

        Board nextBoard = board.next();

        double p1NextKore = nextBoard.getCellAtPosition(p1.add(Direction.NORTH)).kore;
        Assert.assertTrue("should have been destroyed", !nextBoard.fleets.containsKey("f1"));
        Assert.assertTrue("should dump all kore", p1Kore + 100 < p1NextKore);
    }

    @Test
    public void fleetAdjacentBattle_dropsKoreCorrectlyWhenOneDies() throws IOException {
        Board board = getStarterBoard();

        Point p1 = new Point(10, 11);
        Fleet f1 = new Fleet("f1", 50, Direction.NORTH, p1, 100.0, "", 0, board);
        double p1Kore = board.getCellAtPosition(p1.add(Direction.NORTH)).kore;
        board.addFleet(f1);

        Point p2 = p1.add(Direction.NORTH).add(Direction.NORTH).add(Direction.EAST);
        Fleet f2 = new Fleet("f2", 100, Direction.SOUTH, p2, 100.0, "", 1, board);
        board.addFleet(f2);

        Board nextBoard = board.next();

        double p1NextKore = nextBoard.getCellAtPosition(p1.add(Direction.NORTH)).kore;
        Assert.assertTrue("should have been destroyed", !nextBoard.fleets.containsKey("f1"));
        Assert.assertTrue("should dump half kore", p1Kore + 50 < p1NextKore && p1Kore + 55 > p1NextKore);

        Fleet f2next = nextBoard.fleets.get("f2");
        Assert.assertTrue("Should have picked up half", f2.kore + 50 <= f2next.kore && f2.kore + 55 > f2next.kore);
    }

    private Board getStarterBoard() throws IOException {
        Path configPath = Paths.get("bin", "test", "configuration.json");
        String rawConfig = Files.readString(configPath);        
        Path obsPath = Paths.get("bin", "test", "observation.json");
        String rawObs = Files.readString(obsPath);        
        
        return new Board(rawObs, rawConfig);
    }
}
