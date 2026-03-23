import { memo, useRef } from 'react';
import potImg from '../assets/pot.webp';
import blackImg from '../assets/stone-black.webp';
import whiteImg from '../assets/stone-white.webp';
import useGameStore from '../stores/useGameStore';
import { getAgentLogo } from '../utils/agentLogos';
import styles from './ScorePanel.module.css';

interface ScoreRow {
  label: string;
  black: number | string;
  white: number | string;
}

interface PlayerProps {
  isActive: boolean;
  isLastPlayed: boolean;
  isPassed: boolean;
  label: string;
  className?: string;
  icon: string;
}

interface PotProps {
  className?: string;
  count: number;
  stoneImg: string;
  label: string;
}

const SCATTER_RADIUS = 60;

interface ScatterPos {
  x: number;
  y: number;
}

function generateScatterPos(): ScatterPos {
  const angle = Math.random() * Math.PI * 2;
  const r = SCATTER_RADIUS * Math.sqrt(Math.random());
  return { x: Math.cos(angle) * r, y: Math.sin(angle) * r };
}

function Pot({ count, stoneImg, label, className }: PotProps) {
  const positionsRef = useRef<Map<number, ScatterPos>>(new Map());

  return (
    <div className={`grid-pile ${styles.potArea} ${className}`} role="img" aria-label={label}>
      <img src={potImg} className={styles.potImage} alt="" aria-hidden="true" draggable="false" />
      {/* eslint-disable-next-line react-hooks/refs */}
      {Array.from({ length: count }, (_, i) => {
        if (!positionsRef.current.has(i)) {
          positionsRef.current.set(i, generateScatterPos());
        }
        const pos = positionsRef.current.get(i)!;

        return (
          <img
            key={i}
            src={stoneImg}
            className={styles.prisoner}
            style={{ translate: `${pos.x}% ${pos.y}%` }}
            draggable={false}
            alt=""
          />
        );
      })}
    </div>
  );
}

function Player({ isActive, isLastPlayed, isPassed, icon, label, className }: PlayerProps) {
  return (
    <div
      className={`${className} ${styles.player} ${isActive && styles.active} ${isLastPlayed && styles.lastPlayed} squiggle-border`}
    >
      <span className={`${styles.playerLogo} grid-pile`} aria-hidden="true">
        {icon ? <img src={icon} alt="" role="presentation" /> : <span className={styles.logoInitial}>{label[0]}</span>}
      </span>
      <span className={styles.playerNameWrapper}>
        {isPassed && (
          <span className="squiggle-border" aria-hidden="true">
            Pass
          </span>
        )}
        <span className={styles.playerName}>{label}</span>
      </span>
    </div>
  );
}

export default memo(function ScorePanel() {
  const game = useGameStore((state) => state.game);
  const options = useGameStore((state) => state.options);

  const state = game.currentState();
  const scorer = game._scorer.territory(game);
  const komi = game._scorer._komi;
  const activeColor = state.nextColor();

  const blackPassed = state.pass && state.color === 'black';
  const whitePassed = state.pass && state.color === 'white';
  const blackName = options?.replay.info?.TeamNames[0] ?? 'Black';
  const whiteName = options?.replay.info?.TeamNames[1] ?? 'White';
  const blackLogo = getAgentLogo(blackName, 'black');
  const whiteLogo = getAgentLogo(whiteName, 'white');
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
      <h2 className="visually-hidden">Score</h2>

      {/* Players. */}
      <Player
        className={styles.playerBlack}
        isActive={activeColor === 'black'}
        isLastPlayed={lastPlayedColor === 'black'}
        isPassed={blackPassed}
        label={blackName}
        icon={blackLogo.src}
      />

      <Player
        className={styles.playerWhite}
        isActive={activeColor === 'white'}
        isLastPlayed={lastPlayedColor === 'white'}
        isPassed={whitePassed}
        label={whiteName}
        icon={whiteLogo.src}
      />

      {/* Score table. */}
      <div className={`squiggle-border ${styles.tableWrapper}`}>
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
      </div>

      <Pot
        className={styles.potBlack}
        count={state.whiteStonesCaptured}
        stoneImg={whiteImg}
        label={`White stones captured: ${state.whiteStonesCaptured}`}
      />
      <Pot
        className={styles.potWhite}
        count={state.blackStonesCaptured}
        stoneImg={blackImg}
        label={`Black stones captured: ${state.blackStonesCaptured}`}
      />
    </section>
  );
});
