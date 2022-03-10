import { describe } from 'mocha';
import { expect } from 'chai';

import * as fs from 'fs';

import { Observation } from "../kore/Observation";
    
describe('Observation', () =>  {
    it('valid observation works', () => {
        const rawObs = fs.readFileSync('./test/observation.json', 'utf8');
        const ob = new Observation(rawObs);

        expect(ob.player).to.equal(0);
        expect(ob.step).to.equal(16);
        expect(ob.playerFleets.length).to.equal(4);
        expect(ob.playerFleets[0].size).to.equal(0);
        expect(ob.playerShipyards.length).to.equal(4);
        expect(ob.playerShipyards[0].size).to.equal(1);
    })

    it('full observation works', () => {
        const rawObs = fs.readFileSync('./test/fullob.json', 'utf8');
        const ob = new Observation(rawObs);

        expect(ob.player).to.equal(0);
        expect(ob.step).to.equal(200);
        expect(ob.playerHlt.length).to.equal(2);
        expect(ob.playerFleets.length).to.equal(2);
        expect(ob.playerFleets[0].size).to.equal(1);
        expect(ob.playerShipyards.length).to.equal(2);
        expect(ob.playerShipyards[0].size).to.equal(6);
    })
});