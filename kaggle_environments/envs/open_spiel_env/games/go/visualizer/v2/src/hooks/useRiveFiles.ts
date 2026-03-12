import { useEffect, useState } from 'react';

export interface RiveEntry {
  name: string;
  file: string;
  buffer: ArrayBuffer | null;
}

const RIVE_FILES = [
  { name: 'kaggle_knight', file: '/rive/kaggle_knight.riv' },
  { name: 'kaggle_queen', file: '/rive/kaggle_queen.riv' },
];

// Module-scoped cache — persists for the lifetime of the page
const bufferCache = new Map<string, ArrayBuffer>();
let loadPromise: Promise<void> | null = null;

function loadAll(): Promise<void> {
  if (!loadPromise) {
    loadPromise = Promise.all(
      RIVE_FILES.map(async ({ file }) => {
        if (bufferCache.has(file)) return;
        const buffer = await fetch(file).then((res) => res.arrayBuffer());
        bufferCache.set(file, buffer);
      })
    ).then(() => {});
  }
  return loadPromise;
}

function entriesFromCache(): RiveEntry[] {
  return RIVE_FILES.map(({ name, file }) => ({
    name,
    file,
    buffer: bufferCache.get(file) ?? null,
  }));
}

export function useRiveFiles(): RiveEntry[] {
  const [entries, setEntries] = useState<RiveEntry[]>(entriesFromCache);

  useEffect(() => {
    loadAll().then(() => setEntries(entriesFromCache()));
  }, []);

  return entries;
}
