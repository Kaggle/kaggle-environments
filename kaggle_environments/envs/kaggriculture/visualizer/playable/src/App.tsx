import { useState } from 'react';
import { GameScreen } from './ui/GameScreen';
import { SetupScreen } from './ui/SetupScreen';
import type { SetupResult } from './ui/useGameWorker';

export function App() {
  const [setup, setSetup] = useState<SetupResult | null>(null);

  return (
    <div className="app">
      {setup === null ? <SetupScreen onStart={setSetup} /> : <GameScreen setup={setup} onExit={() => setSetup(null)} />}
    </div>
  );
}
