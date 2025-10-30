export function renderer(options: any) {
  const { parent } = options;
  if (!parent) {
    console.error('Renderer: Parent element not provided.');
    return;
  }

  console.log('options is', options);

  // Clear the parent and append the new element
  parent.innerHTML = '<div>hello</div>';
  parent.classList.add('json-renderer-host');
}
