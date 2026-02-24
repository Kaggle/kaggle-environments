# Go v2 Visualizer

A test version of a visualizer for go built with `react` as the frontend
framework and `zustand` for state management.

_Note: Potentially useful libraries: JGoBoard, Tenuki, WGo.js_

`pnpm install`

Make a copy of `.env` to use a local replay file, instead of `dev-with-replay`.

`pnpm dev`

A temporary change has been made to `web/core/src/player.ts` to use a local replay
file any time it's defined in `.env` for sharing work-in-progress.

`pnpm build`

<!-- Download a random batch of additional examples of OpenSpiel chess replay files
from Kaggle Game Arena.

`pnpm download-replays` -->
