# Testing

Some tests can be automated, however since the actual design is programmed in a seperate repository and uses a different engine for testing, some things needs to be manually checked at the moment

Whenever the design is changed, the following tests should be done:

- Engine Consistency tests: Run the same agent against itself using both kaggle-environments and dimensions engine using the same starting seed. The resulting replay should be the exact same visually on the visualizer (check final turn and see if stats and map look the same)

- 