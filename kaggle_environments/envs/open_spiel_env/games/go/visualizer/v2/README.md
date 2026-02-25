# Go v2 Visualizer

A test version of a visualizer for go built with `react` as the frontend
framework and `zustand` for state management.

_Note: Potentially useful libraries: JGoBoard, Tenuki, WGo.js_

_Note: Currently using JGoBoard for both the chess gamestate and board rendering
and although it's released under a CC licence, it's the NC (Non Commercial)
version which isn't a fit for Kaggle. The renderer will be a custom React
component, but the gamestate element will need replacing._

`pnpm install`

Make a copy of `.env` to use a local replay file, instead of `dev-with-replay`.

`pnpm dev`

A temporary change has been made to `web/core/src/player.ts` to use a local replay
file any time it's defined in `.env` for sharing work-in-progress.

`pnpm build`

<!-- Download a random batch of additional examples of OpenSpiel chess replay files
from Kaggle Game Arena.

`pnpm download-replays` -->
