# Notes on dimensions framework integration with kaggle


Issues: 
1. Need post-install scripts to install modules for the design (`node_modules`) and build the typescript code into javascript code
 (probably a part of the `make` function. TBD)


Speed


RPS Lux
python kaggle_environments/envs/rps/scratch.py  1.88s user 0.31s system 122% cpu 1.785 total

RPS Python
python kaggle_environments/envs/rps/scratch.py  0.35s user 0.08s system 98% cpu 0.440 total

RPS Lux (1k steps)
python kaggle_environments/envs/rps/scratch.py  3.48s user 0.58s system 124% cpu 3.250 total

RPS Python (1k steps)
python kaggle_environments/envs/rps/scratch.py  1.68s user 0.09s system 90% cpu 1.970 total

Cost of 1k rounds for lux is 3.48s - 1.88s = 1.6s

Cost of 1k rounds for python is 1.68s - 0.35s = 1.33s

(1.6 - 1.33) / 1000 = 2.7e-4 (.27 ms) and this is using RPS, which is heavy on data transfer as opposed to actual computations (but data transferred is smaller)

this is also using the optimization that we start only one background worker to run the matches