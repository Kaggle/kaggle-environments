import { AGENTS } from '../ai';
import type { GameState } from '../engine/types';
import type { SlotConfig } from '../worker/protocol';
import coinUrl from '../../../default/src/assets/sprites/coin.png';
import woodBgUrl from '../../../default/src/assets/sprites/wood_bg.svg';

interface Props {
  state: GameState;
  slots: SlotConfig[];
  humanPlayerId: number | null;
  onReplay(): void;
  onExit(): void;
}

function slotLabel(slot: SlotConfig, idx: number): string {
  if (slot.kind === 'human') return `Player ${idx + 1} (You)`;
  return `Player ${idx + 1} — ${AGENTS[slot.agentId]?.label ?? slot.agentId}`;
}

export function GameOverModal({ state, slots, humanPlayerId, onReplay, onExit }: Props) {
  const scores = state.scores ?? [];
  const maxScore = Math.max(...scores);
  const winners: number[] = [];
  for (let i = 0; i < scores.length; i++) {
    if (scores[i] === maxScore) winners.push(i);
  }
  const humanWon = humanPlayerId !== null && winners.includes(humanPlayerId);

  let headline: string;
  if (winners.length === 1) {
    if (humanPlayerId === null) headline = `${slotLabel(slots[winners[0]], winners[0])} wins!`;
    else headline = humanWon ? 'You win!' : `${slotLabel(slots[winners[0]], winners[0])} wins`;
  } else {
    headline = `Tie — players ${winners.map((w) => w + 1).join(', ')}`;
  }

  return (
    <div className="modal-bg">
      <div className="modal sketched-border" style={{ backgroundImage: `url(${woodBgUrl})` }}>
        <h2>Game Over</h2>
        <p className="modal-headline">{headline}</p>
        <ul className="modal-scores">
          {slots.map((slot, i) => (
            <li key={i} className={winners.includes(i) ? 'is-winner' : ''}>
              <span>{slotLabel(slot, i)}</span>
              <span className="modal-score-value">
                <img className="modal-coin" src={coinUrl} alt="coins" />
                <strong>{Math.floor(scores[i] ?? 0)}</strong>
              </span>
            </li>
          ))}
        </ul>
        <div className="modal-buttons">
          <button type="button" onClick={onReplay}>
            Replay (same setup)
          </button>
          <button type="button" onClick={onExit}>
            New Game
          </button>
        </div>
      </div>
    </div>
  );
}
