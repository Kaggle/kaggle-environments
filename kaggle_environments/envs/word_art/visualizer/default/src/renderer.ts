import type { RendererOptions } from '@kaggle-environments/core';

interface HistoryEntry {
  word: string;
  blue_art: string;
  blue_art_disqualified?: boolean;
  blue_guesses: string[];
  blue_points: number;
  yellow_art: string;
  yellow_art_disqualified?: boolean;
  yellow_guesses: string[];
  yellow_points: number;
}

const TEAM_LABEL: Record<string, string> = { blue: 'Team Blue', yellow: 'Team Yellow' };

function escape(s: string): string {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function getStepObservation(stepArr: any[], idx: number): any {
  return stepArr?.[idx]?.observation ?? {};
}

function getCurrentArt(stepArr: any[], team: 'blue' | 'yellow', round: number): string {
  // teammate_art lives on the guesser's observation during the guess phase.
  const guesser = team === 'blue' ? 1 - (round % 2) : 2 + (1 - (round % 2));
  return getStepObservation(stepArr, guesser).teammate_art ?? '';
}

function getCurrentWord(stepArr: any[], history: HistoryEntry[], round: number): string {
  for (const p of stepArr ?? []) {
    const tw = p?.observation?.target_word;
    if (tw) return tw;
  }
  if (round < history.length) return history[round].word;
  return stepArr?.[0]?.observation?._words?.[round] ?? '';
}

function roleAt(agentIdx: number, round: number): 'artist' | 'guesser' {
  const teamBase = agentIdx < 2 ? 0 : 2;
  const artist = teamBase + (round % 2);
  return agentIdx === artist ? 'artist' : 'guesser';
}

function playerName(idx: number, replay: any): string {
  const teamNames = replay?.info?.TeamNames;
  if (teamNames && teamNames[idx]) return teamNames[idx];
  return `Agent ${idx}`;
}

// Pulls per-team in-progress guesses for the current round. During the guess
// phase the guesser's observation carries previous_guesses (wrong attempts so
// far). On the sub-step where the team locks in a correct guess, the guesser
// becomes INACTIVE — so we may need to splice the winning attempt back in
// from the most recent active state. Easiest is: take from the agent's
// previous_guesses obs field if non-empty, else from the most recent history
// entry if the round has just closed.
function getLiveGuesses(stepArr: any[], team: 'blue' | 'yellow', round: number): string[] {
  const guesser = team === 'blue' ? 1 - (round % 2) : 2 + (1 - (round % 2));
  const list = getStepObservation(stepArr, guesser).previous_guesses;
  return Array.isArray(list) ? list : [];
}

function pointsLabel(points: number): string {
  if (points >= 2) return `${points} pts (first try!)`;
  if (points === 1) return '1 pt';
  return '0 pts';
}

export function renderer(options: RendererOptions) {
  const { step, replay, parent } = options;
  const steps = (replay?.steps as any[]) ?? [];
  if (!steps.length) return;

  const stepIdx = Math.min(Math.max(step ?? 0, 0), steps.length - 1);
  const currentStep = steps[stepIdx] as any[];
  const obs0 = getStepObservation(currentStep, 0);
  const phase: string = obs0.phase ?? '';
  const currentRound: number = obs0.current_round ?? 0;
  const numRounds: number = obs0.num_rounds ?? 0;
  const maxAttempts: number = obs0.max_attempts ?? 3;
  const history: HistoryEntry[] = (obs0.history ?? []) as HistoryEntry[];
  const blueScore: number = obs0.blue_score ?? 0;
  const yellowScore: number = obs0.yellow_score ?? 0;
  const blueAttemptsUsed: number = obs0.blue_attempts_used ?? 0;
  const yellowAttemptsUsed: number = obs0.yellow_attempts_used ?? 0;

  const isDone = currentStep?.every?.((p: any) => p?.status === 'DONE') ?? false;

  // Anchor the display to the last completed round once the game ends.
  const displayRound = isDone ? Math.max(0, numRounds - 1) : currentRound;
  const isHistoricalView = displayRound < history.length;

  const word = getCurrentWord(currentStep, history, displayRound);

  let blueArt = getCurrentArt(currentStep, 'blue', displayRound);
  let yellowArt = getCurrentArt(currentStep, 'yellow', displayRound);
  let blueGuesses: string[];
  let yellowGuesses: string[];
  let bluePoints: number | null = null;
  let yellowPoints: number | null = null;
  let blueAttemptsRemaining = Math.max(0, maxAttempts - blueAttemptsUsed);
  let yellowAttemptsRemaining = Math.max(0, maxAttempts - yellowAttemptsUsed);
  // Disqualification flags. For historical rounds these live on the history
  // entry. For the current round we infer from the placeholder string that
  // the env writes into teammate_art (see DISQUALIFIED_ART_PLACEHOLDER).
  let blueDisqualified = false;
  let yellowDisqualified = false;

  if (isHistoricalView) {
    const h = history[displayRound];
    if (!blueArt) blueArt = h.blue_art;
    if (!yellowArt) yellowArt = h.yellow_art;
    blueGuesses = h.blue_guesses ?? [];
    yellowGuesses = h.yellow_guesses ?? [];
    bluePoints = h.blue_points ?? 0;
    yellowPoints = h.yellow_points ?? 0;
    blueAttemptsRemaining = 0;
    yellowAttemptsRemaining = 0;
    blueDisqualified = h.blue_art_disqualified === true;
    yellowDisqualified = h.yellow_art_disqualified === true;
  } else {
    blueGuesses = getLiveGuesses(currentStep, 'blue', displayRound);
    yellowGuesses = getLiveGuesses(currentStep, 'yellow', displayRound);
    blueDisqualified = typeof blueArt === 'string' && blueArt.includes('disqualified');
    yellowDisqualified = typeof yellowArt === 'string' && yellowArt.includes('disqualified');
  }

  const activeSet = new Set<number>();
  currentStep?.forEach?.((p: any, i: number) => {
    if (p?.status === 'ACTIVE') activeSet.add(i);
  });

  const phaseLabel = isDone
    ? 'Final'
    : phase === 'art'
      ? `Round ${displayRound + 1} / ${numRounds} — Artists drawing`
      : `Round ${displayRound + 1} / ${numRounds} — Guessers guessing`;

  parent.innerHTML = '';
  const container = document.createElement('div');
  container.className = 'renderer-container';
  parent.appendChild(container);

  // --- Header ---
  const header = document.createElement('div');
  header.className = 'wa-header';
  header.innerHTML = `
    <span class="wa-round-tag">${escape(phaseLabel)}</span>
    <div class="wa-scores">
      <span class="wa-score-pill blue sketched-border">Blue: ${blueScore}</span>
      <span class="wa-score-pill yellow sketched-border">Yellow: ${yellowScore}</span>
    </div>
  `;
  container.appendChild(header);

  // --- Teams ---
  const teamsRow = document.createElement('div');
  teamsRow.className = 'wa-teams';
  for (const team of ['blue', 'yellow'] as const) {
    const teamDiv = document.createElement('div');
    teamDiv.className = 'wa-team sketched-border';
    const teamHeader = document.createElement('div');
    teamHeader.className = `wa-team-header ${team}`;
    teamHeader.textContent = TEAM_LABEL[team];
    teamDiv.appendChild(teamHeader);

    const playersDiv = document.createElement('div');
    playersDiv.className = 'wa-players';
    const indices = team === 'blue' ? [0, 1] : [2, 3];
    for (const i of indices) {
      const role = roleAt(i, displayRound);
      const pDiv = document.createElement('div');
      pDiv.className = `wa-player sketched-border${activeSet.has(i) ? ' active' : ''}`;
      pDiv.innerHTML = `
        <div>${escape(playerName(i, replay))}</div>
        <div class="wa-role">${role}</div>
      `;
      playersDiv.appendChild(pDiv);
    }
    teamDiv.appendChild(playersDiv);
    teamsRow.appendChild(teamDiv);
  }
  container.appendChild(teamsRow);

  // --- Stage ---
  const stage = document.createElement('div');
  stage.className = 'wa-stage sketched-border';

  const wordTitle = document.createElement('div');
  wordTitle.className = 'wa-stage-title';
  wordTitle.textContent = word ? `Target word: ${word}` : 'Target word: (hidden)';
  stage.appendChild(wordTitle);

  const artRow = document.createElement('div');
  artRow.className = 'wa-art-row';
  for (const team of ['blue', 'yellow'] as const) {
    const cell = document.createElement('div');
    cell.className = 'wa-art-cell';

    const label = document.createElement('div');
    label.className = `wa-art-label ${team}`;
    const disqualified = team === 'blue' ? blueDisqualified : yellowDisqualified;
    label.textContent = disqualified ? `${TEAM_LABEL[team]}'s drawing — DISQUALIFIED` : `${TEAM_LABEL[team]}'s drawing`;
    if (disqualified) label.classList.add('disqualified');
    cell.appendChild(label);

    const box = document.createElement('div');
    box.className = 'wa-art-box sketched-border';
    if (disqualified) box.classList.add('disqualified');
    const art = team === 'blue' ? blueArt : yellowArt;
    if (disqualified && isHistoricalView) {
      // History view: show the raw art the artist tried to submit, plus an
      // explanatory header so the reader sees the cheat AND the consequence.
      const note = document.createElement('div');
      note.className = 'wa-art-disqualified-note';
      note.textContent =
        '⚠ Engine disqualified this art: it contained the target word. Teammate saw a placeholder instead.';
      box.appendChild(note);
      const pre = document.createElement('div');
      pre.className = 'wa-art-raw';
      pre.textContent = art || '(empty)';
      box.appendChild(pre);
    } else if (disqualified) {
      const note = document.createElement('div');
      note.className = 'wa-art-disqualified-note';
      note.textContent = '⚠ Disqualified: art contained the target word. Guesser sees placeholder.';
      box.appendChild(note);
    } else if (art && art.length > 0) {
      box.textContent = art;
    } else {
      const empty = document.createElement('span');
      empty.className = 'wa-art-empty';
      empty.textContent = '(waiting for the artist…)';
      box.appendChild(empty);
    }
    cell.appendChild(box);
    artRow.appendChild(cell);
  }
  stage.appendChild(artRow);

  // --- Guess strips: one row per team, slots for each attempt ---
  const guessGrid = document.createElement('div');
  guessGrid.className = 'wa-guess-grid';
  for (const team of ['blue', 'yellow'] as const) {
    const guesses = team === 'blue' ? blueGuesses : yellowGuesses;
    const points = team === 'blue' ? bluePoints : yellowPoints;
    const used = team === 'blue' ? blueAttemptsUsed : yellowAttemptsUsed;
    const remaining = team === 'blue' ? blueAttemptsRemaining : yellowAttemptsRemaining;
    const liveUsed = isHistoricalView ? guesses.length : used;

    const row = document.createElement('div');
    row.className = `wa-guess-team-row ${team}`;

    const teamTag = document.createElement('div');
    teamTag.className = `wa-guess-team-tag ${team}`;
    teamTag.textContent = `${team === 'blue' ? 'Blue' : 'Yellow'} guesses`;
    row.appendChild(teamTag);

    const slots = document.createElement('div');
    slots.className = 'wa-guess-slots';
    for (let i = 0; i < maxAttempts; i++) {
      const slot = document.createElement('div');
      let cls = 'wa-guess-slot sketched-border';
      const g = guesses[i];
      const isCorrectSlot = points !== null && points > 0 && i === guesses.length - 1;
      if (g !== undefined) {
        if (isCorrectSlot) cls += ' correct';
        else cls += ' wrong';
        slot.textContent = `${i + 1}. ${g || '(empty)'}`;
      } else if (!isHistoricalView && i < liveUsed) {
        cls += ' wrong';
        slot.textContent = `${i + 1}. ?`;
      } else {
        cls += ' empty';
        slot.textContent = `${i + 1}.`;
      }
      slot.className = cls;
      slots.appendChild(slot);
    }
    row.appendChild(slots);

    const tally = document.createElement('div');
    tally.className = 'wa-guess-tally';
    if (isHistoricalView || isDone) {
      tally.textContent = pointsLabel(points ?? 0);
    } else if (phase === 'guess') {
      tally.textContent = `${remaining} left`;
    } else {
      tally.textContent = '';
    }
    row.appendChild(tally);

    guessGrid.appendChild(row);
  }
  stage.appendChild(guessGrid);
  container.appendChild(stage);

  // --- Status bar ---
  const statusBar = document.createElement('div');
  statusBar.className = 'wa-status-bar sketched-border';
  if (isDone) {
    let outcome: string;
    if (blueScore > yellowScore) outcome = `Blue wins ${blueScore}–${yellowScore}!`;
    else if (yellowScore > blueScore) outcome = `Yellow wins ${yellowScore}–${blueScore}!`;
    else outcome = `Tie ${blueScore}–${yellowScore}`;
    const final = document.createElement('span');
    final.className = 'wa-final';
    final.textContent = outcome;
    statusBar.appendChild(final);
  } else if (phase === 'art') {
    statusBar.textContent = 'Artists are drawing — guessers wait.';
  } else {
    statusBar.textContent = `Guessers see their teammate's art (up to ${maxAttempts} attempts each).`;
  }
  container.appendChild(statusBar);
}
