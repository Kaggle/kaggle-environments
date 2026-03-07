import { useEffect, useState } from 'react';

interface RiveEntry {
  name: string;
  file: string;
  buffer: ArrayBuffer | null;
}

const RIVE_FILES = [
  { name: 'kaggle_knight', file: '/rive/kaggle_knight.riv' },
  { name: 'kaggle_queen', file: '/rive/kaggle_queen.riv' },
];

export function useRiveFiles(): RiveEntry[] {
  const [entries, setEntries] = useState<RiveEntry[]>(() =>
    RIVE_FILES.map(({ name, file }) => ({ name, file, buffer: null }))
  );

  useEffect(() => {
    const load = async () => {
      const buffers = await Promise.all(
        RIVE_FILES.map(async ({ file }) => ({
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
