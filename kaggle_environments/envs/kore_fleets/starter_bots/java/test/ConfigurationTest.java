package test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.junit.Assert;
import org.junit.Test;

import kore.Configuration;

public class ConfigurationTest {

    @Test
    public void givenValidConfiguration_createSuccessful() throws IOException {
        Path configPath = Paths.get("bin", "test", "configuration.json");
        String rawConfig = Files.readString(configPath);        
        
        Configuration config = new Configuration(rawConfig);

        Assert.assertEquals(0.02, config.regenRate, .001);
    }
    
}
