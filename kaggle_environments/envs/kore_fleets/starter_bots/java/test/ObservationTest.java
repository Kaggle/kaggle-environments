package test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Assert;
import org.junit.Test;

import kore.Observation;

public class ObservationTest {
    
    @Test
    public void givenValidObservation_createSuccessful() throws IOException {
        Path observation = Paths.get("bin", "test", "observation.json");
        String rawObservation = Files.readString(observation);        
        
        Observation ob = new Observation(rawObservation);

        Assert.assertEquals(0, ob.player);
        Assert.assertEquals(16, ob.step);
        Assert.assertEquals(4, ob.playerHlt.length);
        Assert.assertEquals(4, ob.playerFleets.size());
        Assert.assertEquals(0, ob.playerFleets.get(0).size());
        Assert.assertEquals(4, ob.playerShipyards.size());
        Assert.assertEquals(1, ob.playerShipyards.get(0).size());
    }

    @Test
    public void givenFullObservation_createSuccessful() throws IOException {
        Path observation = Paths.get("bin", "test", "fullob.json");
        String rawObservation = Files.readString(observation);        
        
        Observation ob = new Observation(rawObservation);

        Assert.assertEquals(0, ob.player);
        Assert.assertEquals(200, ob.step);
        Assert.assertEquals(2, ob.playerHlt.length);
        Assert.assertEquals(2, ob.playerFleets.size());
        Assert.assertEquals(1, ob.playerFleets.get(0).size());
        Assert.assertEquals(2, ob.playerShipyards.size());
        Assert.assertEquals(6, ob.playerShipyards.get(0).size());
    }
}