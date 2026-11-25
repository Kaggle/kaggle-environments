# BattleCast Arena Adapter

This environment is a thin adapter, not a copy of the rules engine. Its host
must install a release of `battlecast-engine` that implements:

```text
battlecast-engine arena kaggle-step
```

The command reads one JSON request from standard input and writes one JSON
response to standard output. The competition image pins an exact npm release
and runs this command directly; it never fetches a Git dependency at match time.

Agents submit `{ "party": ... }` during setup. During combat, the active team
submits an action id or an action object from its `legalActions` observation.
The engine owns combat rules, legal-action generation, seeded randomness, and
serialized combat state. This adapter owns only Kaggle state/status/reward
translation.

The adapter stores the authoritative encounter in a Kaggle hidden observation
field. Live agents receive only their own detailed state and a redacted view of
the opponent. The seed and RNG state are host-private. Completed replays retain
the hidden state for auditing.
