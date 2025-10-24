# Kaggle Environments - Core Player Library

This package, `@kaggle-environments/core`, provides the foundational components for building and rendering simulation replays.

The goal of this library is to standardize the replay experience and provide shared toolkit that separates the generic player UI from the game-specific rendering logic.

## Core Components

### `Player`

The `Player` class is a self-contained UI component that provides the standard controls for navigating a game replay. This includes:

- A play/pause button.
- Previous/next step buttons.
- A timeline slider for scrubbing through steps.
- A step counter display.

In the near future we will allow users to either enable or disable these controls through config to support them existing in the external frame, such as on Kaggle.

It handles all the logic for playback, timing, and state management. It loads replay data passed to it via `window.postMessage`, making it easy to embed in an iframe.

### `GameAdapter` Interface

To connect a specific game's visualizer to the generic `Player`, you must provide a `GameAdapter`. This is a simple interface that tells the player how to:

- `mount()`: Initialize the rendering surface (e.g., a canvas, a DOM tree).
- `render()`: Draw a specific step of the replay.
- `unmount()`: Clean up any resources when the player is destroyed.

### `PreactAdapter`

For convenience, a `PreactAdapter` is included. This adapter allows you to use a Preact functional component as your renderer, abstracting away the `mount`, `render`, and `unmount` lifecycle methods. You simply provide a component, and the adapter handles the rest.

### Shared Types

This library also defines the standardized TypeScript interfaces for replay data (`ReplayData`, `ReplayStep`, etc.). This creates a clear and type-safe data contract that all visualizers must adhere to, ensuring compatibility between the Kaggle platform and the individual game renderers.
