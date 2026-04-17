# Visualizer Style Guide

All Kaggle environment visualizers should match a **paper-and-ink** aesthetic. The goal is a warm, tactile, stationery-like look -- as if the game were drawn on paper with hand-sketched borders.

## Aesthetic principles

1. **Warm paper-like background.** Use a warm parchment background (`#f5f1e2`) instead of dark or saturated colors. The canvas should have a transparent background so the page color shows through from the DOM layer beneath.

2. **Light color scheme.** Use near-black text (`#050001`) on the paper background. Avoid dark backgrounds, white-on-dark text, and neon/diffused glows.

3. **Sketched borders.** Use dashed borders (`1px dashed #3c3b37`) on containers instead of solid CSS borders or diffused `box-shadow`. This gives a hand-drawn, woodblock-print quality.

4. **High-resolution text.** Prefer **DOM elements** for all text, labels, and status displays rather than canvas text. Canvas `fillText` cannot use web fonts reliably. Use canvas only for the game board/grid itself. Wrap the canvas in a flex container alongside DOM-based status elements.

5. **Two typefaces.** Use **Inter** (sans-serif) for all UI text -- player names, scores, labels, controls. Use **Mynerve** (cursive) as an optional accent font for annotations, commentary, and decorative text. Load Inter via CSS `@import` in `style.css` and Mynerve via `<link>` in `index.html`.

6. **Hard offset shadows.** For modals and popover panels, use hard black offset shadows (e.g., `box-shadow: -0.75rem 0.75rem`) rather than soft diffused drop-shadows. This matches the woodblock/stamp aesthetic.

7. **Responsive sizing.** Use CSS container queries (`@container (max-width: 680px)`) for responsive layout adjustments. Set `container-type: inline-size` on the main wrapper. The **680px** breakpoint is the mobile threshold. Use `rem`-based font sizes (`0.8rem`, `1rem`, `1.1rem`).

## Color palette

| Element | Color / Treatment | Notes |
|---------|------------------|-------|
| Page background | `#f5f1e2` | Warm parchment, never dark or saturated |
| Primary text | `#050001` | Near-black, used on all body text |
| Secondary text | `#444343` | Softer dark for table values and metadata |
| Container background | `white` | Player cards, score tables, panels |
| Active player highlight | `#bdeeff` | Light blue background on the active player card |
| Borders | `1px dashed #3c3b37` | Sketched look on containers |
| Buttons / controls bg | `#f1f1f1` | Light gray for interactive elements |
| Button shadow | `box-shadow: -0.125rem 0.125rem 0 #000` | Hard black offset, not diffused |
| Canvas background | Transparent | Page background shows through from DOM layer |
| Board grid lines | `1px dashed #3c3b37` or `1px solid #3c3b37` | Sketched look for grid lines on canvas |
| Board labels | `#000000` (Inter font) | Column/row labels around the board |

## Rendering approach

Use a **hybrid DOM + canvas** architecture:

- **Canvas**: game board grid, pieces, move highlights, board decorations. Keep the canvas background transparent so the page background shows through.
- **DOM**: player names, score tables, turn indicators, game-over modals, annotations. Use `border: 1px dashed #3c3b37` on containers.

Cap the canvas at a maximum width (e.g., `max-width: 512px`) and use `aspect-ratio: 1` for square boards.

```
+------------------------------------------+
|  [DOM] Header: player cards with         |
|  dashed borders                          |
+------------------------------------------+
|                                          |
|  [Canvas] Game board (transparent bg)    |
|  on warm parchment background            |
|                                          |
+------------------------------------------+
|  [DOM] Status / score with dashed        |
|  borders, annotations in Mynerve font   |
+------------------------------------------+
```

## Sketched border container pattern

Use white containers with a dashed border for a hand-drawn look:

```typescript
const statusContainer = document.createElement('div');
Object.assign(statusContainer.style, {
  padding: '5px 12px',
  backgroundColor: 'white',
  border: '1px dashed #3c3b37',
  textAlign: 'center',
  minWidth: '200px',
  marginTop: '10px',
  fontFamily: "'Inter', sans-serif",
});
```

## Active player indication

Use background color change and scale transform on player containers:

```css
.player {
  background-color: white;
  transition: scale 300ms;
}

.player.active {
  background-color: #bdeeff;
  scale: 1.1;
}
```

## Game-over presentation

Use a modal overlay with staggered reveal animations:

```css
.game-over-modal {
  background-color: #f5f1e2;
  color: #050001;
}
```

Display results in a table with dashed borders. Use CSS `@starting-style` and `transition` for staggered element reveals.

## Standard CSS (`style.css`)

```css
@import url('https://fonts.googleapis.com/css2?family=Inter:ital,opsz,wght@0,14..32,100..900;1,14..32,100..900&display=swap');

html, body, #app {
  width: 100%;
  height: 100%;
  margin: 0;
  padding: 0;
  overflow: hidden;
}

.renderer-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
  height: 100%;
  min-height: 0;
  background-color: #f5f1e2;
  overflow: hidden;
  font-family: 'Inter', sans-serif;
  box-sizing: border-box;
  padding: 12px;
  color: #050001;
  container-type: inline-size;
}

.renderer-container canvas {
  position: relative;
  flex-grow: 1;
  width: 100%;
  max-width: 512px;
  min-height: 0;
}

.sketched-border {
  border: 1px dashed #3c3b37;
}

.header {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  padding: 8px 0;
  font-size: 1.1rem;
  font-weight: 600;
  flex-shrink: 0;
  gap: 16px;
}

.status-container {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 5px 16px;
  background-color: white;
  font-size: 0.9rem;
  font-weight: 600;
  min-height: 18px;
  min-width: 200px;
  margin-top: 8px;
  flex-shrink: 0;
}
```

## Standard `index.html` font links

```html
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Mynerve&display=swap" rel="stylesheet" />
```
