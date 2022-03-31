package test;

import org.junit.Assert;
import org.junit.Test;

import kore.Point;
import kore.Shipyard;

public class ShipyardTest {

    @Test
    public void maxSpawn_worksCorrectly() {
        int[] turns = {0, 1, 2, 293, 294, 295};
        int[] expected = {1, 1, 2, 9, 10, 10};
        for (int i = 0; i < turns.length; i ++) {
            Shipyard shipyard = new Shipyard("A", 0, new Point(0, 0), 1, turns[i], null, null);

            Assert.assertEquals(shipyard.maxSpawn(), expected[i]);
        }
    }
    
}
