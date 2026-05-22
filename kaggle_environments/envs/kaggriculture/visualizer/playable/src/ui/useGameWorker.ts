/**
 * Hook that owns a WorkerClient and tracks the latest GameState it produces.
 * Re-mounts (new worker, fresh init) whenever the `setup` reference changes.
 */

import { useEffect, useRef, useState } from 'react';
import type { Config, GameState } from '../engine/types';
import { WorkerClient } from '../worker/workerClient';
import type { HumanActions, SlotConfig } from '../worker/protocol';

export interface SetupResult {
  config: Config;
  numAgents: number;
  slots: SlotConfig[];
}

export function useGameWorker(setup: SetupResult) {
  const clientRef = useRef<WorkerClient | null>(null);
  const [state, setState] = useState<GameState | null>(null);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const client = new WorkerClient();
    clientRef.current = client;
    setBusy(true);
    setError(null);
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

  const stepGame = async (humanActions: HumanActions): Promise<void> => {
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

  const reset = async (): Promise<void> => {
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
