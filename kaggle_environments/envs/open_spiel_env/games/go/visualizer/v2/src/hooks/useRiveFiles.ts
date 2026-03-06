import { useEffect, useState } from 'react';

interface RiveEntry {
  name: string;
  file: string;
  buffer: ArrayBuffer | null;
}

const rivePaths = Object.keys(import.meta.glob('/public/rive/*.riv')).map((path) => {
  const file = path.replace('/public', '');
  const name = file.replace('/rive/', '').replace('.riv', '');
  return { name, file };
});

export function useRiveFiles(): RiveEntry[] {
  const [entries, setEntries] = useState<RiveEntry[]>(() =>
    rivePaths.map(({ name, file }) => ({ name, file, buffer: null }))
  );

  useEffect(() => {
    const load = async () => {
      const buffers = await Promise.all(
        rivePaths.map(async ({ file }) => ({
          file,
          buffer: await fetch(file).then((res) => res.arrayBuffer()),
        }))
      );
      setEntries((prev) =>
        prev.map((e) => {
          const loaded = buffers.find((b) => b.file === e.file);
          return loaded ? { ...e, buffer: loaded.buffer } : e;
        })
      );
    };
    load();
  }, []);

  return entries;
}
