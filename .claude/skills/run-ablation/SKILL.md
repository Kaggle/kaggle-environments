---
name: run-ablation
description: Run a prompt-ablation study on a kaggle-environments game's LLM harness. Use when the user mentions "ablation", "prompt sensitivity", "test prompt variants", "compare prompts", "ablate the prompt", "prompt rewrite study", or asks whether their prompt wording is doing real work. Bootstraps a prompt_variants.py if missing, proposes new variants interactively, then runs a paired-seat tournament and reports the leaderboards.
---

Follow the instructions in [.agents/skills/run-ablation/SKILL.md](../../../.agents/skills/run-ablation/SKILL.md).

Use `$ARGUMENTS` as the game name if provided.
