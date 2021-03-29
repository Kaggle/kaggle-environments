# Notes on dimensions framework integration with kaggle


Issues: 
1. Need post-install scripts to install modules for the design (`node_modules`) and build the typescript code into javascript code

2. How to exclude all folders named `node_modules` via manifest.in?

3. Error logs? Where is kaggle envs handling them?

4. env.render() function, how will this work with our visualizer hosted at https://2021vis.lux-ai.org/. Data url? how is this sent

5. how should kaggle envs initialize js agents

6. for some reason local replays are generated 3 times (lux issue). 

7. is there a way to get the current episode # or just step #

Quite doable, just detached the whole backend and engine part of dimensions and use just the design.

Pros:
- Retain typescript/javascript design
- Retain typing with typescript, making it easier to develop a new design in general


Cons:
- Startup speed is slow, this is because nodejs is importing the entire dimensions library, maybe tree-shaking can be implemented (webpack like thing)
