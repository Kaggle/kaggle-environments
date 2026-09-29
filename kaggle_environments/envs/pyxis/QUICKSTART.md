# Pyxis Quick Start

Full guide: [AGENTS.md](AGENTS.md). Rules: [README.md](README.md).

## 1. Write `main.py`

```python
def agent(observation, configuration):
    masks = observation["actionMask"]
    # mask[slot][1]: starting the next trial phase is legal and affordable.
    return {"investments": [1 if slot[1] else 0 for slot in masks["investments"]]}
```

Omitted action heads default to no-op. Picking a choice the mask forbids forfeits the match.

## 2. Test locally

```bash
pip install -U kaggle-environments
python -c "
from kaggle_environments import make
env = make('pyxis', debug=True)
env.run(['main.py', 'knapsack'])
print([(s.reward, s.status) for s in env.steps[-1]])
"
```

## 3. Submit

1. Join the competition: https://www.kaggle.com/competitions/gsk-simulation
2. Save your API token (https://www.kaggle.com/settings/api) to `~/.kaggle/access_token`
3. Submit:

```bash
pip install kaggle
kaggle competitions submit gsk-simulation -f main.py -m "Baseline"
kaggle competitions submissions gsk-simulation
```
