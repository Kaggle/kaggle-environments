# Kore Fleets Java Bot

## Running locally

1. compile your bot with `javac Bot.java`
2. run a match with `kaggle-environments run --environment kore_fleets --agents ./main.py ./main.py`

* where `./main.py` is the relative path from your current working directory to your bot's `main.py` file
* 1, 2, or 4 agents are supported
## Watching a replay locally
1. compile your bot with `javac Bot.java`
2. run a match with `kaggle-environments run --environment kore_fleets --agents ./main.py ./main.py --log out.log --out replay.html --render '{"mode": "html"}'`
3. open `replay.html` with your browser of choice!


## Creating a submission

1. compile your both with `javac Bot.java`
2. create a tar.gz with `tar --exclude='test' --exclude='jars' -czvf submission.tar.gz  *`

note you must do this in the same directory as `main.py`!

## Running tests

1. Add the jars/ to visual studio as described [here](https://stackoverflow.com/questions/50232557/visual-studio-code-java-extension-howto-add-jar-to-classpath)
2. Run tests!
