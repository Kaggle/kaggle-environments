import { Ribbon } from './Ribbon.tsx';
import useGameStore from '../stores/useGameStore.ts';
import styles from './VersusBanner.module.css';

export default function VersusBanner() {
  console.log("VersusBanner")
  const game = useGameStore((state) => state.game);

  if (!game.gameStart) return null;

  return (
    <div className={styles.versusBanner} aria-hidden="true">
      <Ribbon>
        {game.blackName ?? 'Black'} vs. {game.whiteName ?? 'White'}
      </Ribbon>
    </div>
  );
}
