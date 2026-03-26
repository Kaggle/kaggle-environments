import styles from './ScorePanel.module.css';

interface ScorePlayerProps {
  isActive: boolean;
  isLastPlayed: boolean;
  isPassed: boolean;
  label: string;
  className?: string;
  icon: string;
}

export default function ScorePlayer({ isActive, isLastPlayed, isPassed, icon, label, className }: ScorePlayerProps) {
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
