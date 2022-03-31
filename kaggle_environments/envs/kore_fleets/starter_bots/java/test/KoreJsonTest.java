package test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Assert;
import org.junit.Test;

import kore.KoreJson;

public class KoreJsonTest {
    @Test
    public void containsKey() throws IOException {
        Path observation = Paths.get("bin", "test", "observation.json");
        String raw = Files.readString(observation);        
        
        Assert.assertTrue(KoreJson.containsKey(raw, "kore"));
        Assert.assertFalse(KoreJson.containsKey(raw, "notThere"));
    }

    @Test
    public void getIntFromJson() throws IOException {
        Path observation = Paths.get("bin", "test", "observation.json");
        String raw = Files.readString(observation);        
        
        Assert.assertEquals(KoreJson.getIntFromJson(raw, "step"), 16);
    }

    @Test
    public void getStrFromJson() {
        Assert.assertTrue(KoreJson.getStrFromJson("{'test': 'foo'}", "test").equals("foo"));
    }

    @Test
    public void getFloatArrFromJson() throws IOException {
        Path observation = Paths.get("bin", "test", "observation.json");
        String raw = Files.readString(observation);        

        double[] kore = KoreJson.getDoubleArrFromJson(raw, "kore");
        Assert.assertEquals(kore[3], 1.372, 0.0001);
    }

    @Test
    public void getPlayerPartsFromJson() throws IOException {
        Path observation = Paths.get("bin", "test", "observation.json");
        String raw = Files.readString(observation);        

        String[] players = KoreJson.getPlayerPartsFromJson(raw);
        Assert.assertEquals(players.length, 4);
        Assert.assertEquals(players[0].substring(0, 3), "500");
    }

    @Test
    public void getPlayerIdxFromJson() throws IOException {
        Path observation = Paths.get("bin", "test", "observation.json");
        String raw = Files.readString(observation);        

        Assert.assertEquals(KoreJson.getPlayerIdxFromJson(raw), 0);
    }
}
