import blackImg from '../assets/stone-black.webp';
import whiteImg from '../assets/stone-white.webp';
import ScorePlayer from './ScorePlayer';
import ScorePot from './ScorePot';
import { getAgentLogo } from '../utils/agentLogos';
import useGameStore from '../stores/useGameStore';
import styles from './ScorePanel.module.css';

interface ScoreRow {
  label: string;
  black: number | string;
  white: number | string;
}

export default function ScorePanel() {
  console.log("ScorePanel")
  const game = useGameStore((state) => state.game);

  const state = game.currentState();
  const scorer = game._scorer.territory(game);
  const komi = game._scorer._komi;
  const activeColor = state.nextColor();

  const blackPassed = state.pass && state.color === 'black';
  const whitePassed = state.pass && state.color === 'white';
  const blackName = game.blackName ?? 'Black';
  const whiteName = game.whiteName ?? 'White';
  const blackLogo = getAgentLogo(blackName, 'black');
  const whiteLogo = getAgentLogo(whiteName, 'white');
  const lastPlayedColor = state.color;

  const rows: ScoreRow[] = [
    {
      label: 'Territory',
      black: game.moveNumber() === 1 ? 0 : scorer.black.length,
      white: game.moveNumber() === 1 ? 0 : scorer.white.length,
    },
    {
      label: 'Prisoners',
      black: state.whiteStonesCaptured,
      white: state.blackStonesCaptured,
    },
    {
      label: 'Komi',
      black: 0,
      white: komi,
    },
  ];

  // React 18 doesn't support the `inert` HTML attribute as a prop, so we
  // set it imperatively via a ref callback. This can be replaced with a
  // regular `inert` prop once the project upgrades to React 19+.
  const inertRef = (el: HTMLElement | null) => {
    if (!el) return;
    if (game.gameOver) el.setAttribute('inert', '');
    else el.removeAttribute('inert');
  };

  return (
    <section className={styles.panel} aria-label="Score" ref={inertRef}>
      <h2 className="visually-hidden">Score</h2>

      {/* Players. */}
      <ScorePlayer
        className={styles.playerBlack}
        isActive={activeColor === 'black'}
        isLastPlayed={lastPlayedColor === 'black'}
        isPassed={blackPassed}
        label={blackName}
        icon={blackLogo.src}
      />

      <ScorePlayer
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

      <ScorePot
        className={styles.potBlack}
        count={state.whiteStonesCaptured}
        stoneImg={whiteImg}
        label={`White stones captured: ${state.whiteStonesCaptured}`}
      />
      <ScorePot
        className={styles.potWhite}
        count={state.blackStonesCaptured}
        stoneImg={blackImg}
        label={`Black stones captured: ${state.blackStonesCaptured}`}
      />
    </section>
  );
}
