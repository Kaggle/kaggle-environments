import { useId, useState } from 'react';
import { AGENTS, DEFAULT_AGENT_ID } from '../ai';
import { resolveConfig } from '../engine/state';
import type { SlotConfig } from '../worker/protocol';
import type { SetupResult } from './useGameWorker';
import strawberryUrl from '../../../default/src/assets/sprites/market_strawberry.png';
import woodBgUrl from '../../../default/src/assets/sprites/wood_bg.svg';

type SlotPick = { kind: 'human' } | { kind: 'ai'; agentId: string };

const DEFAULT_SLOTS: SlotPick[] = [{ kind: 'human' }, { kind: 'ai', agentId: DEFAULT_AGENT_ID }];

interface Props {
  onStart(result: SetupResult): void;
}

const DAY_OPTIONS = [1, 3, 5, 10, 30] as const;
const DEFAULT_DAYS = 30;

export function SetupScreen({ onStart }: Props) {
  const [slots, setSlots] = useState<SlotPick[]>(DEFAULT_SLOTS);
  const [seedText, setSeedText] = useState('');
  const [days, setDays] = useState<number>(DEFAULT_DAYS);
  const idPrefix = useId();
  const daysId = `${idPrefix}-days`;
  const seedId = `${idPrefix}-seed`;

  const setSlot = (idx: number, slot: SlotPick) => {
    setSlots((prev) => {
      const next = [...prev];
      next[idx] = slot;
      return next;
    });
  };

  const handleStart = () => {
    const seedNum = seedText.trim() === '' ? Math.floor(Math.random() * 0x7fffffff) : Number(seedText);
    const seed = Number.isFinite(seedNum) ? Math.floor(seedNum) : Math.floor(Math.random() * 0x7fffffff);
    const base = resolveConfig({ seed });
    const config = { ...base, episodeSteps: days * base.turnsPerDay };
    const finalSlots: SlotConfig[] = slots.map((s) =>
      s.kind === 'human' ? { kind: 'human' } : { kind: 'ai', agentId: s.agentId }
    );
    onStart({ config, numAgents: finalSlots.length, slots: finalSlots });
  };

  return (
    <div className="setup sketched-border" style={{ backgroundImage: `url(${woodBgUrl})` }}>
      <h1>
        <img className="setup-strawberry" src={strawberryUrl} alt="" />
        Kaggriculture
      </h1>
      <p className="setup-sub">Pick who controls each farm, then start.</p>

      {slots.map((slot, i) => {
        const slotId = `${idPrefix}-slot-${i}`;
        return (
          <div key={i} className="setup-row">
            <label htmlFor={slotId}>Player {i + 1}</label>
            <select
              id={slotId}
              value={slot.kind === 'human' ? '__human' : slot.agentId}
              onChange={(e) => {
                const v = e.target.value;
                if (v === '__human') setSlot(i, { kind: 'human' });
                else setSlot(i, { kind: 'ai', agentId: v });
              }}
            >
              <option value="__human">Human</option>
              {Object.entries(AGENTS).map(([id, info]) => (
                <option key={id} value={id}>
                  AI — {info.label}
                </option>
              ))}
            </select>
          </div>
        );
      })}

      <div className="setup-row">
        <label htmlFor={daysId}>Days</label>
        <select id={daysId} value={days} onChange={(e) => setDays(Number(e.target.value))}>
          {DAY_OPTIONS.map((d) => (
            <option key={d} value={d}>
              {d} {d === 1 ? 'day' : 'days'}
            </option>
          ))}
        </select>
      </div>

      <div className="setup-row">
        <label htmlFor={seedId}>Seed</label>
        <input
          id={seedId}
          type="text"
          placeholder="random"
          value={seedText}
          onChange={(e) => setSeedText(e.target.value)}
          style={{ width: 120 }}
        />
      </div>

      <div className="setup-row setup-actions">
        <button className="setup-start" onClick={handleStart}>
          Start Game
        </button>
      </div>
    </div>
  );
}
