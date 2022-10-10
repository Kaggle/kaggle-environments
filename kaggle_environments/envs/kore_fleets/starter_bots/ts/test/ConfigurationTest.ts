import { describe } from 'mocha';
import { expect } from 'chai';

import * as fs from 'fs';

import { Configuration } from "../kore/Configuration";

describe('Configuration', () =>  {
    it('init works correctly', () => {
        const rawConfig = fs.readFileSync('./test/configuration.json','utf8');
        
        const config = new Configuration(rawConfig);

        expect(config.regenRate).to.equal(.02);
    })
});