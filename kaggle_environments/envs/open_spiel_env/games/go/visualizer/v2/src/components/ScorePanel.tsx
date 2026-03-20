import { memo } from 'react';
import useGameStore from '../stores/useGameStore';
import { getAgentLogo } from '../utils/agentLogos';
import styles from './ScorePanel.module.css';

interface ScoreRow {
  label: string;
  black: number | string;
  white: number | string;
}

function ThinkingLabel() {
  return (
    <span className={styles.thinkingLabel} aria-live="polite">
      Thinking
      <span className={styles.dot} aria-hidden="true">
        .
      </span>
      <span className={styles.dot} aria-hidden="true">
        .
      </span>
      <span className={styles.dot} aria-hidden="true">
        .
      </span>
    </span>
  );
}

export default memo(function ScorePanel() {
  const game = useGameStore((state) => state.game);
  const options = useGameStore((state) => state.options);

  const state = game.currentState();
  const isOver = game.isOver();
  const scorer = game._scorer.territory(game);
  const komi = game._scorer._komi;
  const activeColor = state.nextColor();
  const moveNum = game.moveNumber();

  const blackPassed = state.pass && state.color === 'black';
  const whitePassed = state.pass && state.color === 'white';
  const blackName = options?.replay.info?.TeamNames[0] ?? 'Black';
  const whiteName = options?.replay.info?.TeamNames[1] ?? 'White';
  const blackLogo = getAgentLogo(blackName, 'black');
  const whiteLogo = getAgentLogo(whiteName, 'white');
  const blackThinking = !isOver && activeColor === 'black';
  const whiteThinking = !isOver && activeColor === 'white';
  const lastPlayedColor = activeColor === 'black' ? 'white' : 'black';

  const rows: ScoreRow[] = [
    {
      label: 'Territory',
      black: game.moveNumber() === 1 ? 0 : scorer.black.length,
      white: game.moveNumber() === 1 ? 0 : scorer.white.length,
    },
    { label: 'Prisoners', black: state.whiteStonesCaptured, white: state.blackStonesCaptured },
    { label: 'Komi', black: 0, white: komi },
  ];

  return (
    <section className={styles.panel} aria-label="Score">
      <div
        className={`${styles.playerBlack} ${activeColor === 'black' ? styles.active : ''} ${lastPlayedColor === 'black' ? styles.lastPlayed : ''}`}
      >
        <span className={`${styles.logo} grid-pile`} aria-hidden="true">
          <img src={blackLogo.src} alt="" />
          {blackLogo.isUnknown && <span className={styles.logoInitial}>{blackName[0]}</span>}
        </span>
        <span className={styles.playerNameWrapper}>
          {blackPassed && (
            <span key={moveNum} className={styles.passIndicator} aria-hidden="true">
              Pass
            </span>
          )}
          <span className={styles.playerName}>{blackName}</span>
          {blackThinking && <ThinkingLabel />}
        </span>
      </div>
      <div
        className={`${styles.playerWhite} ${activeColor === 'white' ? styles.active : ''} ${lastPlayedColor === 'white' ? styles.lastPlayed : ''}`}
      >
        <span className={styles.playerNameWrapper}>
          {whitePassed && (
            <span key={moveNum} className={styles.passIndicator} aria-hidden="true">
              Pass
            </span>
          )}
          <span className={styles.playerName}>{whiteName}</span>
          {whiteThinking && <ThinkingLabel />}
        </span>
        <span className={`${styles.logo} grid-pile`} aria-hidden="true">
          <img src={whiteLogo.src} alt="" />
          {whiteLogo.isUnknown && <span className={styles.logoInitial}>{whiteName[0]}</span>}
        </span>
      </div>
      <table className={styles.table}>
        <thead className="visually-hidden">
          <tr>
            <th scope="col">{blackName}</th>
            <th scope="col">Category</th>
            <th scope="col">{whiteName}</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <td className={styles.cellBlack}>{row.black}</td>
              <td className={styles.cellLabel}>{row.label}</td>
              <td className={styles.cellWhite}>{row.white}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
});
