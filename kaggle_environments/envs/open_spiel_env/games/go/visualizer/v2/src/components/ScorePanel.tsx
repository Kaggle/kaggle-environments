import { memo } from 'react';
import useGameStore from '../stores/useGameStore';
import styles from './ScorePanel.module.css';

interface ScoreRow {
  label: string;
  black: number | string;
  white: number | string;
}

export default memo(function ScorePanel() {
  const game = useGameStore((state) => state.game);
  const options = useGameStore((state) => state.options);

  const state = game.currentState();
  const scorer = game._scorer.territory(game);
  const komi = game._scorer._komi;
  const activeColor = state.nextColor();

  const agents = options?.agents ?? [];
  const blackName = agents[0]?.Name ?? 'Black';
  const whiteName = agents[1]?.Name ?? 'White';

  const rows: ScoreRow[] = [
    {
      label: 'Territory',
      black: game.moveNumber() === 1 ? 0 : scorer.black.length,
      white: game.moveNumber() === 1 ? 0 : scorer.white.length,
    },
    { label: 'Prisoners', black: state.blackStonesCaptured, white: state.whiteStonesCaptured },
    { label: 'Komi', black: 0, white: komi },
  ];

  return (
    <div className={styles.panel}>
      <div className={`${styles.playerBlack} ${activeColor === 'black' ? styles.active : ''}`}>
        <span className={styles.stoneBlack} />
        <span className={styles.playerName}>{blackName}</span>
      </div>
      <table className={styles.table}>
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
      <div className={`${styles.playerWhite} ${activeColor === 'white' ? styles.active : ''}`}>
        <span className={styles.stoneWhite} />
        <span className={styles.playerName}>{whiteName}</span>
      </div>
    </div>
  );
});
