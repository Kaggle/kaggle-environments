# ts-trueskill [![npm](https://img.shields.io/npm/v/ts-trueskill.svg?maxAge=3600)](https://www.npmjs.com/package/ts-trueskill) [![build status](https://circleci.com/gh/scttcper/ts-trueskill.svg?style=svg)](https://circleci.com/gh/scttcper/ts-trueskill) [![coverage status](https://codecov.io/gh/scttcper/ts-trueskill/branch/master/graph/badge.svg)](https://codecov.io/gh/scttcper/ts-trueskill)
> TypeScript port of the [python TrueSkill package](https://github.com/sublee/trueskill) by Heungsub Lee.  



### What's TrueSkill™
[TrueSkill](http://research.microsoft.com/en-us/projects/trueskill) is a rating system for players of a game. It was developed, patented, and trademarked by Microsoft Research and has been used on Xbox LIVE for ranking and matchmaking service. This system quantifies players’ TRUE skill points by the Bayesian inference algorithm. It also works well with any type of match rule including N:N team game or free-for-all.
Read about [how the trueskill model works](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/)

### Install
Built into es5 and published with typings. Available on [npm](https://www.npmjs.com/package/ts-trueskill):
```bash
npm install ts-trueskill
```

### Use  
2 vs 2 example:
```ts
import { rate, Rating, quality } from 'ts-trueskill';
const team1 = [new Rating(), new Rating()];
const team2 = [new Rating(), new Rating()];

// q is quality of the match with the players at their current rating
const q = quality([team1, team2]);

// Assumes the first team was the winner by default
const [rated1, rated2] = rate([team1, team2]); // rate also takes weights of winners or draw
// rated1 and rated2 are now arrays with updated scores from result of match

console.log(rated1.toString()) // team 1 went up in rating
// >> Rating(mu=28.108, sigma=7.774),Rating(mu=28.108, sigma=7.774)
console.log(rated2.toString()) // team 2 went down in rating
// >> Rating(mu=21.892, sigma=7.774),Rating(mu=21.892, sigma=7.774)
```

1 vs 1 example:  
using shortcut functions for 1vs1 matches
```ts
import { Rating, quality_1vs1, rate_1vs1 } from 'ts-trueskill';
const p1 = new Rating(40, 4); // 1P's skill
const p2 = new Rating(10, 4); // 2P's skill
const q = quality_1vs1(p1, p2); // quality will be low from large difference in scores
const [newP1, newP2] = rate_1vs1(p1, p2); // get new ratings after p1 wins
```

### API
https://trueskill.netlify.com/  

### Differences from python version
- Does not support multiple backends

### License
This package is Licensed under MIT, but is a port of the [BSD](http://en.wikipedia.org/wiki/BSD_licenses) licensed python TrueSkill package by Heungsub Lee. The _TrueSkill™_ brand is not very permissive. Microsoft permits only Xbox Live games or non-commercial projects to use TrueSkill™.
