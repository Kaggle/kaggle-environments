package test;

import org.junit.Assert;
import org.junit.Test;

import kore.Point;

public class PointTest {

    @Test
    public void fromIndexToIndex_isIdentity() {
        int idx = 254;
        int size = 31;

        Point point = Point.fromIndex(idx, size);
        int mirroredIdx = point.toIndex(size);

        Assert.assertEquals(idx, mirroredIdx);
    }
    
}
