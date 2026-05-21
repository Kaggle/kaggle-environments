# Kaggriculture Playable Visualizer — Plan

A standalone interactive (human vs AI) version of Kaggriculture, modeled on
`kaggle_environments/envs/orbit_wars/visualizer/playable/`. The engine and
the bundled AI agents are re-implemented in TypeScript so the whole thing
runs in-browser.

Applies to the **full Kaggriculture competition only**, not `kaggriculture_beginner`.

## Stages

Each stage ends in something runnable and reviewable.

### Stage 1 — Scaffold
Create `visualizer/playable/` with `package.json`, `vite.config.ts`,
`tsconfig.json`, React entry. Auto-registered via the existing
`pnpm-workspace.yaml` glob `kaggle_environments/envs/*/visualizer/*`.

**Done when:** `pnpm dev` boots, `pnpm build` is green, root renders.

### Stage 2 — Engine port: data model
Port constants (crops, animals, prices, fib hire cost, town shops, weed /
decay / lifespan thresholds) and types
(`Tile` / `Plant` / `Weed` / `Coop` / `Pasture` / `Farm` / `Market` /
`Town` / `GameState` / `Action`). Port `initial_state`.

**Done when:** a typed initial `GameState` builds for any seed.

### Stage 3 — Engine port: turn processing
Port the per-turn pipeline phase-for-phase from `kaggriculture.py`:
1. Action processing (farmer + hired hands)
2. Market orders (capped per turn)
3. Town center + unlocked shop consumption
4. End-of-day (watering decay, weed spawn, feed/water refresh, plant /
   animal expiry, shed cap enforcement)
5. Terminal check + scores

Add vitest fixtures comparing TS rollouts against Python-generated
reference replays for a couple of fixed seeds. This is the heaviest stage.

**Done when:** TS `step()` output matches Python over a full 720-turn
season for the reference seeds.

### Stage 4 — AI agents
Port the three bundled agents (`pass`, `random`, `starter`).

**Done when:** TS-vs-TS rollouts complete a season without errors.

### Stage 5 — Worker + state hook
`WorkerClient` + `gameWorker` (`INIT` / `STEP` / `RESET`) + a `useGameWorker`
React hook, same shape as orbit_wars.

**Done when:** the React app can run a full AI-vs-AI game off the main
thread.

### Stage 6 — Read-only farm view
Render the `GameState` (both farms, market, town) on canvas. Likely reuse
`renderFarm.ts` from the sibling `default/` visualizer rather than
re-porting.

**Done when:** an AI-vs-AI game is watchable inside the playable app.

### Stage 7 — Action input UI
Build the action picker:
- Click farmer or hired hand → pick a per-unit op (move / plant / water /
  harvest / fertilize / build / feed / care / dig / pickup / place / pass)
- Market order panel (buy seed / product / animal, sell, hire, buy land)
  with pending queue and per-turn cap
- "End Turn" button submits the queued action dict for the human player
- Pending-action tray shows what's queued and lets you remove items

Second-heaviest stage.

**Done when:** a human can play a full game against AI opponents.

### Stage 8 — Setup + HUD + game over
- `SetupScreen` — pick opponent AI, seed, episode length
- `HUD` — money, day/hour, both farms' scores, pending action count
- Game-over modal — final scores, "Replay (same seed)" / "New Game"

**Done when:** polished end-to-end UX from setup → match → restart.

## Notes
- Engine lives entirely in TS (no Python round-trip); drift from
  `kaggriculture.py` is a known risk and is mitigated by the Stage 3
  reference-replay tests.
- Only the three bundled agents are ported; advanced sample agents are out
  of scope.
- Per project memory: avoid CSS `container-type` (it breaks inline playback
  controls in this codebase); use `@media` queries.
