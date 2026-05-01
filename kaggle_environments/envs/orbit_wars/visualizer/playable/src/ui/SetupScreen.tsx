import { useState } from 'react';
import { DEFAULT_COMET_SPEED, DEFAULT_EPISODE_STEPS, DEFAULT_SHIP_SPEED } from '../engine/constants';
import type { Config } from '../engine/types';
import type { SlotConfig } from '../worker/protocol';
import { AGENTS, DEFAULT_AGENT_ID } from '../ai';

export interface SetupResult {
  config: Config;
  numAgents: 2 | 4;
  slots: SlotConfig[];
}

interface Props {
  onStart: (result: SetupResult) => void;
}

export function SetupScreen({ onStart }: Props) {
  const [numAgents, setNumAgents] = useState<2 | 4>(2);
  const [seedText, setSeedText] = useState('');
  const [aiSlots, setAiSlots] = useState<string[]>(() => new Array(3).fill(DEFAULT_AGENT_ID));

  const handleStart = () => {
    const seed = seedText.trim() === '' ? Math.floor(Math.random() * 0x7fffffff) : Number(seedText);
    const config: Config = {
      seed: Number.isFinite(seed) ? Math.floor(seed) : Math.floor(Math.random() * 0x7fffffff),
      episodeSteps: DEFAULT_EPISODE_STEPS,
      shipSpeed: DEFAULT_SHIP_SPEED,
      cometSpeed: DEFAULT_COMET_SPEED,
    };
    const slots: SlotConfig[] = [{ kind: 'human' }];
    for (let i = 1; i < numAgents; i++) {
      slots.push({ kind: 'ai', agentId: aiSlots[i - 1] });
    }
    onStart({ config, numAgents, slots });
  };

  const setAiSlot = (idx: number, agentId: string) => {
    setAiSlots((prev) => {
      const next = [...prev];
      next[idx] = agentId;
      return next;
    });
  };

  return (
    <div className="setup">
      <h1>Orbit Wars</h1>

      <div className="row">
        <label>Players</label>
        <select value={numAgents} onChange={(e) => setNumAgents(Number(e.target.value) as 2 | 4)}>
          <option value={2}>2 (Human + 1 AI)</option>
          <option value={4}>4 (Human + 3 AI)</option>
        </select>
      </div>

      <div className="row">
        <label>You</label>
        <span style={{ color: '#0072B2' }}>Player 1 (Human)</span>
      </div>

      {Array.from({ length: numAgents - 1 }).map((_, i) => (
        <div key={i} className="row">
          <label>AI {i + 1}</label>
          <select value={aiSlots[i]} onChange={(e) => setAiSlot(i, e.target.value)}>
            {Object.entries(AGENTS).map(([id, info]) => (
              <option key={id} value={id}>
                {info.label}
              </option>
            ))}
          </select>
        </div>
      ))}

      <div className="row">
        <label>Seed</label>
        <input
          type="text"
          placeholder="random"
          value={seedText}
          onChange={(e) => setSeedText(e.target.value)}
          style={{ width: 120 }}
        />
      </div>

      <div className="row" style={{ marginTop: 16, justifyContent: 'flex-end' }}>
        <button onClick={handleStart}>Start Game</button>
      </div>
    </div>
  );
}
