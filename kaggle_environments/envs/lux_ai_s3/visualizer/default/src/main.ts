window.addEventListener(
  'message',
  (event) => {
    const replayData = event.data?.environment;

    if (replayData) {
      (window as any).REPLAY = replayData;

      const loader = document.getElementById('loader');
      const root = document.getElementById('root');

      if (loader) {
        loader.style.display = 'none';
      }
      if (root) {
        root.style.display = 'block';
      }
    }
  },
  false
);
