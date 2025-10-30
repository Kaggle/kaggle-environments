import JSONFormatter from 'json-formatter-js';

export function renderer(options: any) {
  const { parent } = options;
  if (!parent) {
    console.error('Renderer: Parent element not provided.');
    return;
  }

  const openLevels = 1;

  const formatter = new JSONFormatter(options, openLevels, { theme: 'dark' });

  parent.innerHTML = '';
  parent.appendChild(formatter.render());
}
