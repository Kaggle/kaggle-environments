# Notes on dimensions framework integration with kaggle


Issues: 
1. Need post-install scripts to install modules for the design (`node_modules`) and build the typescript code into javascript code
 (probably a part of the `make` function. TBD)

2. How to exclude all folders named `node_modules` via manifest.in?

3. Error logs? Where is kaggle envs handling them? (match engine probably output to stderr and the python script will redirect that somewhere) Print to either stdout or stderr (pipe from engine)

4. env.render() function, how will this work with our visualizer hosted at https://2021vis.lux-ai.org/. Data url? how is this sent


- using replay json generated by kaggle

- see main.py to look at how logs and replays are generated


test html:

kaggle-environments run --debug True --environment lux_ai_2021 --agents rock rock --render '{"mode":"html"}' > out.html

5. how should kaggle envs initialize js agents

6. for some reason local replays are generated 3 times (lux issue). 

Quite doable, just detached the whole backend and engine part of dimensions and use just the design.


TESTING Replay viewer via kaggle envs

use --display html 

html_renderer() function returns a string

copy the entire distribution over, keep in mind it might not load assets nicely?

Pros:
- Retain typescript/javascript design
- Retain typing with typescript, making it easier to develop a new design in general


Cons:
- Startup speed is slow, this is because nodejs is importing the entire dimensions library, maybe tree-shaking can be implemented (webpack like thing) - not a problem on the scale of a few seconds


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