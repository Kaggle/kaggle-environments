import { useState } from 'react';
import { SetupScreen, type SetupResult } from './ui/SetupScreen';
import { GameScreen } from './ui/GameScreen';

export function App() {
  const [setup, setSetup] = useState<SetupResult | null>(null);

  return (
    <div className="app">
      {setup === null ? <SetupScreen onStart={setSetup} /> : <GameScreen setup={setup} onExit={() => setSetup(null)} />}
    </div>
  );
}
