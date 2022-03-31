import { describe } from 'mocha';
import { expect } from 'chai';

import { Shipyard } from "../kore/Shipyard";
import { Point } from '../kore/Point';

describe('Shipyard', () =>  {
    const turns = [0, 1, 2, 293, 294, 295];
    const expected = [1, 1, 2, 9, 10, 10]
    for (let i = 0; i < turns.length; i ++) {
        it(`max spawn is correct at ${turns[i]} turns controlled`, () => {
            const shipyard = new Shipyard("A", 0, new Point(0, 0), 1, turns[i], null, null);

            expect(shipyard.maxSpawn).to.equal(expected[i]);
        })
    }

});
