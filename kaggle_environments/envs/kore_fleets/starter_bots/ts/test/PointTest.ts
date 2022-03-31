import { describe } from 'mocha';
import { expect } from 'chai';

import { Point } from "../kore/Point";

describe('Point', () =>  {
    it('fontIndex toIndex isIdetity', () => {
        const idx = 254;
        const size = 31;

        const point = Point.fromIndex(idx, size);
        const mirroredIdx = point.toIndex(size);

        expect(mirroredIdx).to.equal(idx);
    })

});
