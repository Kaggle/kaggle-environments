import { useEffect, useRef, useState } from 'react';
import type { ActionsByPlayer, GameState } from '../engine/types';
import { WorkerClient } from '../worker/workerClient';
import type { SetupResult } from './SetupScreen';

/** Owns the WorkerClient lifecycle and tracks the latest GameState from it. */
export function useGameWorker(setup: SetupResult) {
  const clientRef = useRef<WorkerClient | null>(null);
  const [state, setState] = useState<GameState | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const client = new WorkerClient();
    clientRef.current = client;
    setBusy(true);
    client
      .init(setup.config, setup.numAgents, setup.slots)
      .then((s) => {
        setState(s);
        setBusy(false);
      })
      .catch((e) => {
        setError(String(e));
        setBusy(false);
      });
    return () => {
      client.terminate();
      clientRef.current = null;
    };
  }, [setup]);

  const stepGame = async (humanActions: ActionsByPlayer) => {
    const client = clientRef.current;
    if (!client) return;
    setBusy(true);
    try {
      const next = await client.step(humanActions);
      setState(next);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const reset = async () => {
    const client = clientRef.current;
    if (!client) return;
    setBusy(true);
    try {
      const next = await client.reset();
      setState(next);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return { state, busy, error, stepGame, reset };
}
