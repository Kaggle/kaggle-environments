import { GameRendererProps, GoStep } from '@kaggle-environments/core';
import { memo } from 'react';
import { Ribbon } from './Ribbon.tsx';
import styles from './VersusBanner.module.css';

interface Props {
  options: GameRendererProps<GoStep[]>;
}

export default memo(function VersusBanner({ options }: Props) {
  const blackName = options?.replay.info?.TeamNames[0] ?? 'Black';
  const whiteName = options?.replay.info?.TeamNames[1] ?? 'White';

  return (
    <div className={styles.versusBanner} aria-hidden="true">
      <Ribbon>
        <span>{blackName}</span> vs. <span>{whiteName}</span>
      </Ribbon>
    </div>
  );
});
