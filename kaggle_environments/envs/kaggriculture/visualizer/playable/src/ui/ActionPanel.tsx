/**
 * Composes the human player's PlayerAction: farmer op + one op per hand +
 * a list of market orders. Submit bundles them into a PlayerAction and the
 * panel resets to PASS / no orders.
 */

import { useEffect, useId, useState } from 'react';
import { ANIMALS, CROPS, PRODUCTS } from '../engine/constants';
import type { AnimalId, CropId, GameState, MarketOrder, PlayerAction, ShedItemId, UnitAction } from '../engine/types';

const MOVE_OPS = ['NORTH', 'SOUTH', 'EAST', 'WEST'] as const;
const SIMPLE_UNIT_OPS = [
  'PASS',
  'NORTH',
  'SOUTH',
  'EAST',
  'WEST',
  'WATER',
  'HARVEST',
  'FERTILIZE',
  'DIG',
  'BUILD_COOP',
  'BUILD_PASTURE',
  'FEED',
  'COLLECT_FERTILIZER',
  'CARE',
] as const;

type UnitOp = (typeof SIMPLE_UNIT_OPS)[number] | 'PLANT' | 'PICKUP' | 'PLACE';

const UNIT_OPS: UnitOp[] = [...SIMPLE_UNIT_OPS, 'PLANT', 'PICKUP', 'PLACE'];

const CROP_IDS = Object.keys(CROPS) as CropId[];
const ANIMAL_IDS = Object.keys(ANIMALS) as AnimalId[];
const SHED_ITEMS: ShedItemId[] = [...PRODUCTS, ...ANIMAL_IDS];

const MARKET_KINDS = ['HIRE', 'BUY_LAND', 'BUY_SEED', 'BUY_PRODUCT', 'BUY_ANIMAL', 'SELL'] as const;
type MarketKind = (typeof MARKET_KINDS)[number];

interface UnitDraft {
  op: UnitOp;
  crop: CropId;
  item: ShedItemId;
  qty: number;
}

interface MarketDraft {
  kind: MarketKind;
  crop: CropId;
  animal: AnimalId;
  product: ShedItemId;
  qty: number;
}

const defaultUnit: UnitDraft = { op: 'PASS', crop: 'WHEAT', item: 'WHEAT', qty: 1 };
const defaultMarket: MarketDraft = {
  kind: 'BUY_SEED',
  crop: 'WHEAT',
  animal: 'GOOSE',
  product: 'WHEAT',
  qty: 1,
};

function toUnitAction(d: UnitDraft): UnitAction {
  switch (d.op) {
    case 'PLANT':
      return ['PLANT', d.crop];
    case 'PICKUP':
      return ['PICKUP', d.item, Math.max(1, Math.floor(d.qty))];
    case 'PLACE':
      return ['PLACE', d.item, Math.max(1, Math.floor(d.qty))];
    default:
      return [d.op as (typeof SIMPLE_UNIT_OPS)[number]];
  }
}

function toMarketOrder(d: MarketDraft): MarketOrder | null {
  const qty = Math.max(1, Math.floor(d.qty));
  switch (d.kind) {
    case 'HIRE':
      return ['HIRE'];
    case 'BUY_LAND':
      return ['BUY_LAND'];
    case 'BUY_SEED':
      return ['BUY_SEED', d.crop, qty];
    case 'BUY_PRODUCT':
      if (d.product !== 'WHEAT' && d.product !== 'FERTILIZER') return null;
      return ['BUY_PRODUCT', d.product, qty];
    case 'BUY_ANIMAL':
      return ['BUY_ANIMAL', d.animal, qty];
    case 'SELL':
      if (d.product === 'GOOSE' || d.product === 'COW' || d.product === 'SHEEP') return null;
      return ['SELL', d.product, qty];
    default:
      return null;
  }
}

interface Props {
  state: GameState;
  player: number;
  busy: boolean;
  onSubmit(action: PlayerAction): void;
}

export function ActionPanel({ state, player, busy, onSubmit }: Props) {
  const farm = state.farms[player];
  const numHands = farm?.hands.length ?? 0;

  const [farmer, setFarmer] = useState<UnitDraft>(defaultUnit);
  const [hands, setHands] = useState<UnitDraft[]>([]);
  const [orders, setOrders] = useState<MarketDraft[]>([]);
  const idPrefix = useId();

  // Keep the hands array length in sync with the live farm. New hands default
  // to PASS; extra entries are trimmed if a hand was lost.
  useEffect(() => {
    setHands((prev) => {
      if (prev.length === numHands) return prev;
      const next = prev.slice(0, numHands);
      while (next.length < numHands) next.push({ ...defaultUnit });
      return next;
    });
  }, [numHands]);

  const submit = () => {
    const action: PlayerAction = {
      farmer: toUnitAction(farmer),
      hands: hands.map(toUnitAction),
      market: orders.map(toMarketOrder).filter((o): o is MarketOrder => o !== null),
    };
    onSubmit(action);
    setFarmer(defaultUnit);
    setHands(hands.map(() => ({ ...defaultUnit })));
    setOrders([]);
  };

  const renderUnit = (label: string, draft: UnitDraft, onChange: (d: UnitDraft) => void) => {
    const opId = `${idPrefix}-${label}-op`;
    return (
      <div className="action-row" key={label}>
        <label className="action-label" htmlFor={opId}>
          {label}
        </label>
        <select id={opId} value={draft.op} onChange={(e) => onChange({ ...draft, op: e.target.value as UnitOp })}>
          {UNIT_OPS.map((op) => (
            <option key={op} value={op}>
              {op}
            </option>
          ))}
        </select>
        {draft.op === 'PLANT' && (
          <select value={draft.crop} onChange={(e) => onChange({ ...draft, crop: e.target.value as CropId })}>
            {CROP_IDS.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        )}
        {(draft.op === 'PICKUP' || draft.op === 'PLACE') && (
          <>
            <select value={draft.item} onChange={(e) => onChange({ ...draft, item: e.target.value as ShedItemId })}>
              {SHED_ITEMS.map((it) => (
                <option key={it} value={it}>
                  {it}
                </option>
              ))}
            </select>
            <input
              type="number"
              min={1}
              value={draft.qty}
              onChange={(e) => onChange({ ...draft, qty: Number(e.target.value) || 1 })}
              style={{ width: 60 }}
            />
          </>
        )}
      </div>
    );
  };

  const renderOrder = (draft: MarketDraft, idx: number) => (
    <div className="action-row" key={idx}>
      <select
        value={draft.kind}
        onChange={(e) => {
          const next = [...orders];
          next[idx] = { ...draft, kind: e.target.value as MarketKind };
          setOrders(next);
        }}
      >
        {MARKET_KINDS.map((k) => (
          <option key={k} value={k}>
            {k}
          </option>
        ))}
      </select>
      {draft.kind === 'BUY_SEED' && (
        <select
          value={draft.crop}
          onChange={(e) => {
            const next = [...orders];
            next[idx] = { ...draft, crop: e.target.value as CropId };
            setOrders(next);
          }}
        >
          {CROP_IDS.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      )}
      {draft.kind === 'BUY_PRODUCT' && (
        <select
          value={draft.product}
          onChange={(e) => {
            const next = [...orders];
            next[idx] = { ...draft, product: e.target.value as ShedItemId };
            setOrders(next);
          }}
        >
          {(['WHEAT', 'FERTILIZER'] as ShedItemId[]).map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      )}
      {draft.kind === 'BUY_ANIMAL' && (
        <select
          value={draft.animal}
          onChange={(e) => {
            const next = [...orders];
            next[idx] = { ...draft, animal: e.target.value as AnimalId };
            setOrders(next);
          }}
        >
          {ANIMAL_IDS.map((a) => (
            <option key={a} value={a}>
              {a}
            </option>
          ))}
        </select>
      )}
      {draft.kind === 'SELL' && (
        <select
          value={draft.product}
          onChange={(e) => {
            const next = [...orders];
            next[idx] = { ...draft, product: e.target.value as ShedItemId };
            setOrders(next);
          }}
        >
          {PRODUCTS.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
      )}
      {draft.kind !== 'HIRE' && draft.kind !== 'BUY_LAND' && (
        <input
          type="number"
          min={1}
          value={draft.qty}
          onChange={(e) => {
            const next = [...orders];
            next[idx] = { ...draft, qty: Number(e.target.value) || 1 };
            setOrders(next);
          }}
          style={{ width: 60 }}
        />
      )}
      <button type="button" onClick={() => setOrders(orders.filter((_, i) => i !== idx))}>
        ×
      </button>
    </div>
  );

  return (
    <div className="action-panel">
      <h3>Your turn — Player {player + 1}</h3>
      {renderUnit('Farmer', farmer, setFarmer)}
      {hands.map((h, i) =>
        renderUnit(`Hand ${i + 1}`, h, (d) => {
          const next = [...hands];
          next[i] = d;
          setHands(next);
        })
      )}
      <div className="action-section">
        <div className="action-section-header">
          <span>Market orders ({orders.length})</span>
          <button type="button" onClick={() => setOrders([...orders, { ...defaultMarket }])}>
            + Add
          </button>
        </div>
        {orders.map(renderOrder)}
      </div>
      <button type="button" className="submit-turn" onClick={submit} disabled={busy || state.done}>
        Submit Turn
      </button>
      <div className="action-hints">
        Moves available: {MOVE_OPS.join(', ')}. Tile ops apply to the unit's current cell.
      </div>
    </div>
  );
}
