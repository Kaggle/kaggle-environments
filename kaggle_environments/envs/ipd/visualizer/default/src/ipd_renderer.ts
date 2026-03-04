import { RendererOptions } from '@kaggle-environments/core';
import { ACTION_NAMES, PAYOFF_MATRIX } from './consts';

// Vite asset imports — produces hashed URLs in production builds
import goose0Idle from './assets/goose_0_idle.jpg';
import goose0Joyful from './assets/goose_0_joyful.jpg';
import goose0Talking from './assets/goose_0_talking.jpg';
import goose0Upset from './assets/goose_0_upset.jpg';
import goose1Idle from './assets/goose_1_idle.jpg';
import goose1Joyful from './assets/goose_1_joyful.jpg';
import goose1Talking from './assets/goose_1_talking.jpg';
import goose1Upset from './assets/goose_1_upset.jpg';

const GOOSE_IMAGES: Record<number, Record<string, string>> = {
  0: { idle: goose0Idle, joyful: goose0Joyful, talking: goose0Talking, upset: goose0Upset },
  1: { idle: goose1Idle, joyful: goose1Joyful, talking: goose1Talking, upset: goose1Upset },
};

// SVG icons (inline to avoid external deps)
const EYE_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>`;
const FINGERPRINT_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 12C2 6.5 6.5 2 12 2a10 10 0 0 1 8 4"/><path d="M5 19.5C5.5 18 6 15 6 12c0-.7.12-1.37.34-2"/><path d="M17.29 21.02c.12-.6.43-2.3.5-3.02"/><path d="M12 10a2 2 0 0 0-2 2c0 1.02-.1 2.51-.26 4"/><path d="M8.65 22c.21-.66.45-1.32.57-2"/><path d="M14 13.12c0 2.38 0 6.38-1 8.88"/><path d="M2 16h.01"/><path d="M21.8 16c.2-2 .131-5.354 0-6"/><path d="M9 6.8a6 6 0 0 1 9 5.2c0 .47 0 1.17-.02 2"/></svg>`;
const LOCK_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="18" height="11" x="3" y="11" rx="2" ry="2"/><path d="M7 11V7a5 5 0 0 1 10 0v4"/></svg>`;
const AWARD_SVG = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="6"/><path d="M15.477 12.89 17 22l-5-3-5 3 1.523-9.11"/></svg>`;

// ── Animation state (module-level) ──
let pendingTimers: ReturnType<typeof setTimeout>[] = [];
let lastRenderedStep = -1;

function clearAnimations() {
  pendingTimers.forEach((t) => clearTimeout(t));
  pendingTimers = [];
}

// ── Helpers ──

function getGooseMood(playerAction: number, opponentAction: number): string {
  if (playerAction === opponentAction) return 'idle';
  if (playerAction === 0) return 'upset';
  return 'joyful';
}

function getOutcomeText(p0Name: string, p1Name: string, p0Action: number, p1Action: number): string {
  if (p0Action === 0 && p1Action === 0) return 'Mutual cooperation achieved. Both subjects receive a base reward of 5.';
  if (p0Action === 1 && p1Action === 1)
    return 'Mutual defection detected. Both subjects receive a penalty reward of 1.';
  if (p0Action === 1 && p1Action === 0) return `${p0Name} betrayed ${p1Name}. Maximum reward granted to ${p0Name}.`;
  return `${p1Name} betrayed ${p0Name}. Maximum reward granted to ${p1Name}.`;
}

/** Extract numeric action from various replay formats */
function extractAction(actionData: any): number {
  if (typeof actionData === 'number') return actionData;
  if (typeof actionData?.submission === 'number' && actionData.submission >= 0) return actionData.submission;
  return 0;
}

/** Get both players' actions from a step's data array */
function getStepActions(stepData: any[]): [number, number] {
  return [extractAction(stepData[0]?.action), extractAction(stepData[1]?.action)];
}

/** Compute cumulative scores up to a given step index */
function computeScores(steps: any[], upToStepIndex: number): [number, number] {
  if (upToStepIndex > 0 && upToStepIndex < steps.length) {
    const state = steps[upToStepIndex];
    const r0 = state?.[0]?.reward;
    const r1 = state?.[1]?.reward;
    if (typeof r0 === 'number' && typeof r1 === 'number') {
      return [r0, r1];
    }
  }
  let s0 = 0,
    s1 = 0;
  for (let i = 1; i <= upToStepIndex && i < steps.length; i++) {
    const [a0, a1] = getStepActions(steps[i]);
    s0 += PAYOFF_MATRIX[a0]?.[a1] ?? 0;
    s1 += PAYOFF_MATRIX[a1]?.[a0] ?? 0;
  }
  return [s0, s1];
}

function buildSkeleton(parent: HTMLElement): void {
  parent.innerHTML = '';
  parent.classList.add('ipd-root');

  const container = document.createElement('div');
  container.className = 'ipd-container';

  container.innerHTML = `
    <header class="ipd-header">
      <div>
        <h1 class="ipd-title">The Dilemma</h1>
        <div class="ipd-subtitle">
          ${EYE_SVG}
          Observation Mode Active
        </div>
      </div>
      <div class="ipd-phase">
        <div class="ipd-phase-label">Phase</div>
        <div class="ipd-phase-number">
          <span class="phase-current">00</span> <span class="ipd-phase-total">/ 10</span>
        </div>
      </div>
    </header>

    <div class="ipd-cells">
      <!-- Suspect 0 (left) -->
      <div class="ipd-cell ipd-cell-0" data-phase="waiting">
        <div class="bars"></div>
        <div class="action-glow"></div>

        <div class="suspect-label suspect-label-left">
          <span class="suspect-label-icon">${FINGERPRINT_SVG}</span>
          <div>
            <div class="suspect-name">Suspect 0</div>
            <div class="suspect-id">ID: Inmate 1</div>
          </div>
        </div>

        <div class="goose-frame">
          <img class="goose-image" src="${GOOSE_IMAGES[0].idle}" alt="Suspect 0" />
        </div>

        <div class="gain-overlay gain-overlay-left">
          <div class="gain-label">Gain</div>
          <div class="gain-value gain-value-0">+0</div>
        </div>

        <div class="action-text-container">
          <div>
            <div class="action-text action-text-0"></div>
            <div class="action-divider" style="display:none"></div>
          </div>
        </div>

        <div class="lock-icon lock-icon-0">${LOCK_SVG}</div>

        <div class="reward-container reward-left">
          <div class="reward-label">Accumulated Reward</div>
          <div class="reward-value-row">
            <span class="reward-icon reward-icon-default reward-icon-0">${AWARD_SVG}</span>
            <span class="reward-value reward-value-0">0</span>
          </div>
        </div>
      </div>

      <!-- Suspect 1 (right) -->
      <div class="ipd-cell ipd-cell-1" data-phase="waiting">
        <div class="bars"></div>
        <div class="action-glow"></div>

        <div class="suspect-label suspect-label-right">
          <span class="suspect-label-icon">${FINGERPRINT_SVG}</span>
          <div>
            <div class="suspect-name">Suspect 1</div>
            <div class="suspect-id">ID: Inmate 2</div>
          </div>
        </div>

        <div class="goose-frame">
          <img class="goose-image" src="${GOOSE_IMAGES[1].idle}" alt="Suspect 1" />
        </div>

        <div class="gain-overlay gain-overlay-right">
          <div class="gain-label">Gain</div>
          <div class="gain-value gain-value-1">+0</div>
        </div>

        <div class="action-text-container">
          <div>
            <div class="action-text action-text-1"></div>
            <div class="action-divider" style="display:none"></div>
          </div>
        </div>

        <div class="lock-icon lock-icon-1">${LOCK_SVG}</div>

        <div class="reward-container reward-right">
          <div class="reward-label">Accumulated Reward</div>
          <div class="reward-value-row">
            <span class="reward-icon reward-icon-default reward-icon-1">${AWARD_SVG}</span>
            <span class="reward-value reward-value-1">0</span>
          </div>
        </div>
      </div>
    </div>

    <div class="ipd-footer footer-waiting">
      <div class="observation-log">
        <div class="observation-label">Observation Log</div>
        <div class="observation-text">Awaiting deliberation\u2026</div>
      </div>
      <div class="security-badge">
        <div class="security-label">Security</div>
        <div class="security-tag">Restricted</div>
      </div>
    </div>

    <div class="ipd-spotlight"></div>
  `;

  parent.appendChild(container);
}

function q<T extends HTMLElement>(parent: HTMLElement, sel: string): T | null {
  return parent.querySelector<T>(sel);
}

export function renderer(context: RendererOptions) {
  const { replay, parent, step } = context;
  const steps = replay.steps as any[];

  if (!parent.querySelector('.ipd-container')) {
    buildSkeleton(parent);
    lastRenderedStep = -1;
  }

  const totalRounds = steps.length - 1;
  const isNewStep = step !== lastRenderedStep;

  if (isNewStep) {
    clearAnimations();
    lastRenderedStep = step;
  }

  // ── DOM elements ──
  const phaseCurrent = q(parent, '.phase-current');
  const phaseTotal = q(parent, '.ipd-phase-total');
  const cell0 = q(parent, '.ipd-cell-0');
  const cell1 = q(parent, '.ipd-cell-1');
  const img0 = q<HTMLImageElement>(parent, '.ipd-cell-0 .goose-image');
  const img1 = q<HTMLImageElement>(parent, '.ipd-cell-1 .goose-image');
  const actionText0 = q(parent, '.action-text-0');
  const actionText1 = q(parent, '.action-text-1');
  const divider0 = q(parent, '.ipd-cell-0 .action-divider');
  const divider1 = q(parent, '.ipd-cell-1 .action-divider');
  const lock0 = q(parent, '.lock-icon-0');
  const lock1 = q(parent, '.lock-icon-1');
  const gainOverlay0 = q(parent, '.ipd-cell-0 .gain-overlay');
  const gainOverlay1 = q(parent, '.ipd-cell-1 .gain-overlay');
  const gainValue0 = q(parent, '.gain-value-0');
  const gainValue1 = q(parent, '.gain-value-1');
  const rewardValue0 = q(parent, '.reward-value-0');
  const rewardValue1 = q(parent, '.reward-value-1');
  const rewardIcon0 = q(parent, '.reward-icon-0');
  const rewardIcon1 = q(parent, '.reward-icon-1');
  const footer = q(parent, '.ipd-footer');
  const obsText = q(parent, '.observation-text');

  if (!phaseCurrent || !cell0 || !cell1 || !footer || !obsText) return;

  // ── Player names ──
  const info = replay.info;
  const p0Name = info?.TeamNames?.[0] || info?.Agents?.[0]?.Name || 'Suspect 0';
  const p1Name = info?.TeamNames?.[1] || info?.Agents?.[1]?.Name || 'Suspect 1';
  const nameEls = parent.querySelectorAll<HTMLElement>('.suspect-name');
  if (nameEls[0]) nameEls[0].textContent = p0Name;
  if (nameEls[1]) nameEls[1].textContent = p1Name;

  if (phaseTotal) phaseTotal.textContent = `/ ${totalRounds}`;

  // ── Helpers ──
  const updateScores = (s0: number, s1: number) => {
    if (rewardValue0) {
      rewardValue0.textContent = String(s0);
      rewardValue0.classList.toggle('reward-leading', s0 > s1);
    }
    if (rewardIcon0) {
      rewardIcon0.classList.toggle('reward-icon-leading', s0 > s1);
      rewardIcon0.classList.toggle('reward-icon-default', s0 <= s1);
    }
    if (rewardValue1) {
      rewardValue1.textContent = String(s1);
      rewardValue1.classList.toggle('reward-leading', s1 > s0);
    }
    if (rewardIcon1) {
      rewardIcon1.classList.toggle('reward-icon-leading', s1 > s0);
      rewardIcon1.classList.toggle('reward-icon-default', s1 <= s0);
    }
  };

  const setObsText = (text: string, waiting: boolean) => {
    footer.classList.toggle('footer-waiting', waiting);
    obsText.textContent = text;
  };

  /** Reset both cells to waiting state */
  const resetCells = () => {
    cell0.setAttribute('data-phase', 'waiting');
    cell1.setAttribute('data-phase', 'waiting');
    if (img0) img0.src = GOOSE_IMAGES[0].idle;
    if (img1) img1.src = GOOSE_IMAGES[1].idle;
    if (actionText0) {
      actionText0.textContent = '';
      actionText0.classList.remove('action-visible');
    }
    if (actionText1) {
      actionText1.textContent = '';
      actionText1.classList.remove('action-visible');
    }
    if (divider0) divider0.style.display = 'none';
    if (divider1) divider1.style.display = 'none';
    if (lock0) lock0.style.display = '';
    if (lock1) lock1.style.display = '';
    gainOverlay0?.classList.remove('gain-visible');
    gainOverlay1?.classList.remove('gain-visible');
    setObsText('Awaiting deliberation\u2026', true);
  };

  /** Show full result state without animation */
  const showFullResult = (a0: number, a1: number, roundS0: number, roundS1: number, cumS0: number, cumS1: number) => {
    // Result phase: grayscale with reaction images
    cell0.setAttribute('data-phase', 'result');
    cell1.setAttribute('data-phase', 'result');
    if (img0) img0.src = GOOSE_IMAGES[0][getGooseMood(a0, a1)];
    if (img1) img1.src = GOOSE_IMAGES[1][getGooseMood(a1, a0)];
    if (actionText0) {
      actionText0.textContent = ACTION_NAMES[a0];
      actionText0.classList.add('action-visible');
    }
    if (divider0) divider0.style.display = 'block';
    if (lock0) lock0.style.display = 'none';
    if (actionText1) {
      actionText1.textContent = ACTION_NAMES[a1];
      actionText1.classList.add('action-visible');
    }
    if (divider1) divider1.style.display = 'block';
    if (lock1) lock1.style.display = 'none';
    if (gainValue0) gainValue0.textContent = `+${roundS0}`;
    if (gainValue1) gainValue1.textContent = `+${roundS1}`;
    gainOverlay0?.classList.add('gain-visible');
    gainOverlay1?.classList.add('gain-visible');
    updateScores(cumS0, cumS1);
    setObsText(getOutcomeText(p0Name, p1Name, a0, a1), false);
  };

  // ══════════════════════════════════════════════════
  // Final state (last slider position)
  // ══════════════════════════════════════════════════
  if (step >= steps.length - 1) {
    phaseCurrent.textContent = String(totalRounds).padStart(2, '0');

    const lastStepData = steps[steps.length - 1] as any;
    const [lastA0, lastA1] = getStepActions(lastStepData);
    const [finalS0, finalS1] = computeScores(steps, steps.length - 1);

    cell0.setAttribute('data-phase', 'idle');
    cell1.setAttribute('data-phase', 'idle');
    if (img0) img0.src = GOOSE_IMAGES[0].idle;
    if (img1) img1.src = GOOSE_IMAGES[1].idle;
    gainOverlay0?.classList.remove('gain-visible');
    gainOverlay1?.classList.remove('gain-visible');

    if (actionText0) {
      actionText0.textContent = ACTION_NAMES[lastA0];
      actionText0.classList.add('action-visible');
    }
    if (divider0) divider0.style.display = 'block';
    if (lock0) lock0.style.display = 'none';
    if (actionText1) {
      actionText1.textContent = ACTION_NAMES[lastA1];
      actionText1.classList.add('action-visible');
    }
    if (divider1) divider1.style.display = 'block';
    if (lock1) lock1.style.display = 'none';

    updateScores(finalS0, finalS1);
    setObsText(getOutcomeText(p0Name, p1Name, lastA0, lastA1), false);
    return;
  }

  // ══════════════════════════════════════════════════
  // Normal step
  // ══════════════════════════════════════════════════
  const state = steps[step + 1] as any;
  if (!state) return;

  phaseCurrent.textContent = String(step + 1).padStart(2, '0');

  const [p0Action, p1Action] = getStepActions(state);
  const p0RoundScore = PAYOFF_MATRIX[p0Action]?.[p1Action] ?? 0;
  const p1RoundScore = PAYOFF_MATRIX[p1Action]?.[p0Action] ?? 0;
  const [prevS0, prevS1] = computeScores(steps, step);
  const cumS0 = prevS0 + p0RoundScore;
  const cumS1 = prevS1 + p1RoundScore;

  if (isNewStep) {
    // ── Animated sequence (slowed down) ───────────

    // Phase 0: Reset to waiting
    resetCells();
    updateScores(prevS0, prevS1);

    // Phase 1: P1 speaks (200ms)
    pendingTimers.push(
      setTimeout(() => {
        cell0.setAttribute('data-phase', 'speaking');
        if (img0) img0.src = GOOSE_IMAGES[0].talking;
        if (actionText0) {
          actionText0.textContent = ACTION_NAMES[p0Action];
          actionText0.classList.add('action-visible');
        }
        if (divider0) divider0.style.display = 'block';
        if (lock0) lock0.style.display = 'none';
      }, 200)
    );

    // Phase 2: P1 returns to idle, P2 speaks (1100ms)
    pendingTimers.push(
      setTimeout(() => {
        // P1 goes back to idle
        cell0.setAttribute('data-phase', 'idle');
        if (img0) img0.src = GOOSE_IMAGES[0].idle;

        // P2 starts speaking
        cell1.setAttribute('data-phase', 'speaking');
        if (img1) img1.src = GOOSE_IMAGES[1].talking;
        if (actionText1) {
          actionText1.textContent = ACTION_NAMES[p1Action];
          actionText1.classList.add('action-visible');
        }
        if (divider1) divider1.style.display = 'block';
        if (lock1) lock1.style.display = 'none';
      }, 1100)
    );

    // Phase 3: Result reveal (2000ms)
    pendingTimers.push(
      setTimeout(() => {
        // Both return to grayscale with reaction images
        cell0.setAttribute('data-phase', 'result');
        cell1.setAttribute('data-phase', 'result');
        if (img0) img0.src = GOOSE_IMAGES[0][getGooseMood(p0Action, p1Action)];
        if (img1) img1.src = GOOSE_IMAGES[1][getGooseMood(p1Action, p0Action)];

        if (gainValue0) gainValue0.textContent = `+${p0RoundScore}`;
        if (gainValue1) gainValue1.textContent = `+${p1RoundScore}`;
        gainOverlay0?.classList.add('gain-visible');
        gainOverlay1?.classList.add('gain-visible');

        updateScores(cumS0, cumS1);
        setObsText(getOutcomeText(p0Name, p1Name, p0Action, p1Action), false);
      }, 2000)
    );
  } else {
    // Same step re-render: show final state instantly
    showFullResult(p0Action, p1Action, p0RoundScore, p1RoundScore, cumS0, cumS1);
  }
}
