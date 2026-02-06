import * as React from 'react';
import { createRoot } from 'react-dom/client';
import { ThemeProvider, CssBaseline, createTheme } from '@mui/material';
import { EpisodePlayer, GameRendererProps } from '../components/EpisodePlayer';
import { ReplayData } from '../types';
import { themeBreakpoints } from '../theme';

// Create a dark theme for the dev environment
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
  },
  breakpoints: themeBreakpoints,
});

// Simple placeholder game renderer
const PlaceholderRenderer: React.FC<GameRendererProps> = ({ replay, step, agents }) => {
  const currentStep = replay.steps[step];

  return (
    <div
      style={{
        width: '100%',
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: '#1a1a1a',
        color: '#fff',
        padding: '20px',
        boxSizing: 'border-box',
      }}
    >
      <h2 style={{ margin: '0 0 20px 0' }}>{replay.name} Visualizer</h2>
      <div style={{ marginBottom: '10px' }}>
        Step {step + 1} of {replay.steps.length}
      </div>
      {currentStep?.players && currentStep.players.length > 0 && (
        <div style={{ marginBottom: '10px' }}>
          Active Player: {currentStep.players.find((p) => p.isTurn)?.name ?? 'N/A'}
        </div>
      )}
      {agents.length > 0 && (
        <div style={{ marginTop: '20px' }}>
          <strong>Agents:</strong>
          <ul style={{ listStyle: 'none', padding: 0, margin: '10px 0' }}>
            {agents.map((agent, i) => (
              <li key={i} style={{ padding: '4px 0' }}>
                {agent.name || `Agent ${i + 1}`}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

interface DevAppState {
  replay: ReplayData | null;
  agents: any[];
  gameName: string;
  error: string | null;
  loading: boolean;
}

const DevApp: React.FC = () => {
  const [state, setState] = React.useState<DevAppState>({
    replay: null,
    agents: [],
    gameName: 'unknown',
    error: null,
    loading: true,
  });

  React.useEffect(() => {
    const loadReplayData = async () => {
      const replayFile = import.meta.env.VITE_REPLAY_FILE;

      if (replayFile) {
        try {
          const response = await fetch(replayFile);
          const data = await response.json();
          setState({
            replay: data,
            agents: data.info?.Agents || [],
            gameName: data.name || 'unknown',
            error: null,
            loading: false,
          });
        } catch (err) {
          setState((prev) => ({
            ...prev,
            error: `Failed to load replay file: ${replayFile}`,
            loading: false,
          }));
        }
      } else {
        setState((prev) => ({
          ...prev,
          error: 'No VITE_REPLAY_FILE environment variable set. Set it to a JSON replay file path.',
          loading: false,
        }));
      }
    };

    loadReplayData();

    // Also listen for postMessage data from parent
    const handleMessage = (event: MessageEvent) => {
      const data = event.data;
      if (!data) return;

      if (data.replay) {
        setState((prev) => ({
          ...prev,
          replay: data.replay,
          gameName: data.replay.name || prev.gameName,
          loading: false,
          error: null,
        }));
      }

      if (data.agents) {
        setState((prev) => ({ ...prev, agents: data.agents }));
      }

      if (data.environment) {
        setState((prev) => {
          const newReplay = prev.replay
            ? { ...prev.replay, ...data.environment }
            : {
                name: data.environment.name || 'unknown',
                version: data.environment.version || 'unknown',
                steps: data.environment.steps || [],
                configuration: data.environment.configuration || {},
                info: data.environment.info || {},
              };
          return {
            ...prev,
            replay: newReplay,
            gameName: newReplay.name,
            loading: false,
            error: null,
          };
        });
      }
    };

    window.addEventListener('message', handleMessage);
    return () => window.removeEventListener('message', handleMessage);
  }, []);

  if (state.loading) {
    return (
      <div
        style={{
          width: '100vw',
          height: '100vh',
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          backgroundColor: '#121212',
          color: '#fff',
        }}
      >
        Loading replay data...
      </div>
    );
  }

  if (state.error || !state.replay) {
    return (
      <div
        style={{
          width: '100vw',
          height: '100vh',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'center',
          alignItems: 'center',
          backgroundColor: '#121212',
          color: '#fff',
          padding: '20px',
          textAlign: 'center',
        }}
      >
        <h2>Kaggle Environments Dev Server</h2>
        <p style={{ color: '#f44336', marginTop: '20px' }}>{state.error || 'No replay data available'}</p>
        <p style={{ marginTop: '20px', color: '#888' }}>To load a replay file, start the dev server with:</p>
        <pre
          style={{
            backgroundColor: '#2a2a2a',
            padding: '10px 20px',
            borderRadius: '4px',
            marginTop: '10px',
          }}
        >
          VITE_REPLAY_FILE=/path/to/replay.json pnpm dev
        </pre>
        <p style={{ marginTop: '20px', color: '#888' }}>Or send replay data via postMessage from a parent frame.</p>
      </div>
    );
  }

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <div style={{ width: '100vw', height: '100vh' }}>
        <EpisodePlayer
          replay={state.replay}
          agents={state.agents}
          gameName={state.gameName}
          GameRenderer={PlaceholderRenderer}
          ui="side-panel"
          initialPlaying={false}
          initialSpeed={1}
          initialReplayMode="condensed"
        />
      </div>
    </ThemeProvider>
  );
};

// HMR state preservation
interface HMRState {
  step?: number;
  playing?: boolean;
  speed?: number;
  replay?: ReplayData;
  agents?: any[];
}

declare global {
  interface Window {
    __hmrState?: HMRState;
  }
}

// Initialize the app
const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(<DevApp />);
}

// HMR handling
if (import.meta.hot) {
  import.meta.hot.accept();
}
