# ts-gaussian [![npm](https://img.shields.io/npm/v/ts-gaussian.svg?maxAge=3600)](https://www.npmjs.com/package/ts-gaussian) [![CircleCI](https://circleci.com/gh/scttcper/ts-gaussian.svg?style=svg)](https://circleci.com/gh/scttcper/ts-gaussian) [![coverage status](https://codecov.io/gh/scttcper/ts-gaussian/branch/master/graph/badge.svg)](https://codecov.io/gh/scttcper/ts-gaussian)

> A JavaScript model of the [Normal](http://en.wikipedia.org/wiki/Normal_distribution)
(or Gaussian) distribution.

__API Docs__: https://ts-gaussian.netlify.com  

## Creating a Distribution

```ts
import { Gaussian } from 'ts-gaussian';
const distribution = new Gaussian(0, 1);
// Take a random sample using inverse transform sampling method.
const sample = distribution.ppf(Math.random());
// 0.5071973169873031 or something similar
```

## Properties

* `mean`: the mean (μ) of the distribution
* `variance`: the variance (σ^2) of the distribution
* `standardDeviation`: the standard deviation (σ) of the distribution

## Probability Functions

* `pdf(x)`: the probability density function, which describes the probability
  of a random variable taking on the value _x_
* `cdf(x)`: the cumulative distribution function, which describes the probability of a random variable falling in the interval (−∞, _x_]
* `ppf(x)`: the percent point function, the inverse of _cdf_

## Combination Functions

* `mul(d)`: returns the product distribution of this and the given distribution; equivalent to `scale(d)` when d is a constant
* `div(d)`: returns the quotient distribution of this and the given distribution; equivalent to `scale(1/d)` when d is a constant
* `add(d)`: returns the result of adding this and the given distribution's means and variances
* `sub(d)`: returns the result of subtracting this and the given distribution's means and variances
* `scale(c)`: returns the result of scaling this distribution by the given constant

## See Also

__ts-trueskill__: https://github.com/scttcper/ts-trueskill

### Forked From

__Source__: https://github.com/errcw/gaussian  
__ES5 Fork__: https://github.com/tomgp/gaussian
