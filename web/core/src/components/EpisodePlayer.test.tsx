// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { EpisodePlayer, EpisodePlayerProps, GameRendererProps } from './EpisodePlayer';
import { makeStep, makeReplay } from '../test-utils';
import { ReplayMode } from '../types';
import { theme } from '../theme';

const mockActions = {
  setStep: vi.fn(),
  setStepOnly: vi.fn(),
  stepForward: vi.fn(),
  stepBackward: vi.fn(),
  play: vi.fn(),
  pause: vi.fn(),
  toggle: vi.fn(),
  restart: vi.fn(),
  setSpeed: vi.fn(),
  setReplayMode: vi.fn(),
  setPlayingState: vi.fn(),
};

const mockState = {
  step: 0,
  playing: false,
  speed: 1,
  replayMode: 'condensed' as ReplayMode,
};

const mockParentData = { parentHandlesUi: false };

vi.mock('../hooks/usePlayerController/usePlayerController', () => ({
  usePlayerController: vi.fn(() => [mockState, mockActions, mockParentData]),
}));

const GAME_NAME = 'test-game';
const TOTAL_STEPS = 5;
const testSteps = Array.from({ length: TOTAL_STEPS }, (_, i) => makeStep({ step: i }));
const testReplay = makeReplay({ steps: testSteps, name: GAME_NAME });

function MockGameRenderer(props: GameRendererProps) {
  return (
    <div data-testid="game-renderer">
      <span data-testid="renderer-step">{props.step}</span>
      <button
        data-testid="register-handlers"
        onClick={() => props.onRegisterPlaybackHandlers?.({ onPlay: () => {}, onPause: () => {} })}
      />
      <button data-testid="announce" onClick={() => props.onAnnounce?.('test announcement')} />
    </div>
  );
}

const baseProps: EpisodePlayerProps = {
  replay: testReplay,
  gameName: GAME_NAME,
  GameRenderer: MockGameRenderer,
};

const renderPlayer = (overrides: Partial<EpisodePlayerProps> = {}) =>
  render(
    <ThemeProvider theme={theme}>
      <EpisodePlayer {...baseProps} {...overrides} />
    </ThemeProvider>
  );

afterEach(cleanup);

beforeEach(() => {
  Object.values(mockActions).forEach((fn) => fn.mockClear());
  mockState.step = 0;
  mockState.playing = false;
  mockState.speed = 1;
  mockState.replayMode = 'condensed';
  mockParentData.parentHandlesUi = false;
});

describe('EpisodePlayer', () => {
  describe('loading state', () => {
    it('shows Loading when no replay is provided', () => {
      renderPlayer({ replay: undefined });
      expect(screen.getByText('Loading...')).toBeDefined();
    });

    it('does not show Loading when replay exists', () => {
      renderPlayer();
      expect(screen.queryByText('Loading...')).toBeNull();
    });
  });

  describe('game renderer', () => {
    it('renders the GameRenderer component', () => {
      renderPlayer();
      expect(screen.getByTestId('game-renderer')).toBeDefined();
    });

    it('passes current step to GameRenderer', () => {
      mockState.step = 3;
      renderPlayer();
      expect(screen.getByTestId('renderer-step').textContent).toBe('3');
    });
  });

  describe('UI modes', () => {
    it('shows ReasoningLogs in side-panel mode (default)', () => {
      renderPlayer({ ui: 'side-panel' });
      expect(screen.getByText('Game Log')).toBeDefined();
    });

    it('shows PlaybackControls in inline mode', () => {
      renderPlayer({ ui: 'inline' });
      expect(screen.getByLabelText('Step slider')).toBeDefined();
    });

    it('shows no controls in none mode', () => {
      renderPlayer({ ui: 'none' });
      expect(screen.queryByText('Game Log')).toBeNull();
      expect(screen.queryByLabelText('Step slider')).toBeNull();
    });
  });

  describe('panel toggle (side-panel mode)', () => {
    it('hides logs and shows Game Log button when panel is closed', () => {
      renderPlayer({ ui: 'side-panel' });
      fireEvent.click(screen.getByLabelText('Collapse Episodes'));
      expect(screen.queryByLabelText('Collapse Episodes')).toBeNull();
      expect(screen.getByText('Game Log')).toBeDefined();
    });

    it('reopens logs panel when Game Log button is clicked', () => {
      renderPlayer({ ui: 'side-panel' });
      fireEvent.click(screen.getByLabelText('Collapse Episodes'));
      fireEvent.click(screen.getByText('Game Log'));
      expect(screen.getByLabelText('Collapse Episodes')).toBeDefined();
    });
  });

  describe('parent handles UI', () => {
    it('hides inline controls when parent handles UI', () => {
      mockParentData.parentHandlesUi = true;
      renderPlayer({ ui: 'inline' });
      expect(screen.queryByLabelText('Step slider')).toBeNull();
    });

    it('hides side-panel when parent handles UI', () => {
      mockParentData.parentHandlesUi = true;
      renderPlayer({ ui: 'side-panel' });
      expect(screen.queryByText('Game Log')).toBeNull();
    });
  });

  describe('keyboard shortcuts', () => {
    it('ArrowLeft calls stepBackward', () => {
      renderPlayer();
      fireEvent.keyDown(window, { key: 'ArrowLeft' });
      expect(mockActions.stepBackward).toHaveBeenCalled();
    });

    it('ArrowRight calls stepForward', () => {
      renderPlayer();
      fireEvent.keyDown(window, { key: 'ArrowRight' });
      expect(mockActions.stepForward).toHaveBeenCalled();
    });

    it('Space calls toggle', () => {
      renderPlayer();
      fireEvent.keyDown(window, { key: ' ' });
      expect(mockActions.toggle).toHaveBeenCalled();
    });

    it('Enter calls toggle', () => {
      renderPlayer();
      fireEvent.keyDown(window, { key: 'Enter' });
      expect(mockActions.toggle).toHaveBeenCalled();
    });

    it('ignores shortcuts when target is an input', () => {
      renderPlayer();
      const input = document.createElement('input');
      document.body.appendChild(input);
      fireEvent.keyDown(input, { key: 'ArrowLeft' });
      expect(mockActions.stepBackward).not.toHaveBeenCalled();
      document.body.removeChild(input);
    });

    it('ignores shortcuts when target is a button', () => {
      renderPlayer();
      const button = document.createElement('button');
      document.body.appendChild(button);
      fireEvent.keyDown(button, { key: ' ' });
      expect(mockActions.toggle).not.toHaveBeenCalled();
      document.body.removeChild(button);
    });
  });

  describe('accessibility', () => {
    it('has an aria-live region for announcements', () => {
      renderPlayer();
      const liveRegion = screen.getByRole('status');
      expect(liveRegion).toBeDefined();
      expect(liveRegion.getAttribute('aria-live')).toBe('polite');
    });

    it('updates announcement when renderer calls onAnnounce', () => {
      renderPlayer();
      fireEvent.click(screen.getByTestId('announce'));
      expect(screen.getByRole('status').textContent).toBe('test announcement');
    });
  });

  describe('skipTransform', () => {
    it('renders game when skipTransform is true', () => {
      renderPlayer({ skipTransform: true });
      expect(screen.getByTestId('game-renderer')).toBeDefined();
    });
  });
});
