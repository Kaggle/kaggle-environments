import { memo } from 'react';
import useGameStore from '../stores/useGameStore';

export default memo(function VersusBanner() {
  const options = useGameStore((state) => state.options);

  if (options?.step === 0) {
    const blackName = options?.replay.info?.TeamNames[0] ?? 'Black';
    const whiteName = options?.replay.info?.TeamNames[1] ?? 'White';

    console.log(`${blackName} vs ${whiteName}`);
  }

  return null;
});
