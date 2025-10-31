import JSONFormatter from 'json-formatter-js';

// We'll add our new class styles here.
const css = `
  .json-renderer-container {
    height: 100%; /* Make the container fill its parent's height */
    overflow-y: auto; /* Add a vertical scrollbar ONLY when needed */
    box-sizing: border-box; /* Ensures padding doesn't add to the height */
  }
`;

// Helper to inject styles into the document head once.
function _injectStyles() {
  if (typeof document === 'undefined' || (window as any).__json_renderer_styles_injected) {
    return;
  }
  const style = document.createElement('style');
  style.textContent = css;
  document.head.appendChild(style);
  (window as any).__json_renderer_styles_injected = true;
}

export function renderer(options: any) {
  const { parent } = options;
  const { environment } = options;
  const transformedResults = environment;

  if (!parent) {
    console.error('Renderer: Parent element not provided.');
    return;
  }

  // Ensure our scrolling styles are on the page.
  _injectStyles();

  // Prune data and limit open levels for performance (from previous step)
  const openLevels = 3;

  const formatter = new JSONFormatter(transformedResults.steps[options.step], openLevels, { theme: 'dark' });

  // Clear the parent and append the rendered tree.
  parent.innerHTML = '';

  // *** THE FIX IS HERE ***
  // Add our special class to make the parent container scrollable.
  parent.classList.add('json-renderer-container');

  parent.appendChild(formatter.render());
}
