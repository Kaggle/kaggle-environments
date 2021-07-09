Unfortunately at this moment, there isn't a progammatic way to test if the Kaggle Engine is the exact same as the engine competitors use when local testing. This is part of a TODO to make kaggle replays match the local replays players generate when they develop locally using https://github.com/Lux-AI-Challenge/Lux-Design-2021

The following steps work for now:

First download the simple kit from https://github.com/Lux-AI-Challenge/Lux-Design-2021/tree/master/kits/python (or another language) or use your own bot

First run a game in Kaggle like so:

```
kaggle-environments run --environment lux_ai_2021 --agents path/to/bot/main.py path/to/bot/main.py --render '{"mode": "json"}' --out out.json --debug=True
```

Then upload the out.json to https://2021vis.lux-ai.org/ and go to the final turn.

Then install the local engine `npm i -g install @lux-ai/2021-challenge` if you haven't done so already (you can remove `-g` and do `npx lux-ai-2021` as opposed to `lux-ai-2021`)

Then run

```
lux-ai-2021 path/to/bot/main.py path/to/bot/main.py --out=replay.json
```

upload `replay.json` to the replay viewer again in a new tab and compare the final state with the other kaggle produced match. If the statistics and layout of units on the map match, then the engine is working correctly. (Given the complexity of the game and that `lux-ai-2021` generates action based replays, should a single thing be wrong, the differences in the final turn would be fairly massive so this is generally a sufficient test)