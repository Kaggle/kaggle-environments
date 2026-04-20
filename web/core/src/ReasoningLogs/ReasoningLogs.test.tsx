// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import * as React from 'react';
import { ThemeProvider } from '@mui/material/styles';
import { ReasoningLogs, ReasoningLogsProps } from './ReasoningLogs';
import { makeStep } from '../test-utils';
import { BaseGameStep, ReplayMode } from '../types';
import { theme } from '../theme';

vi.mock('@mui/material/useMediaQuery', () => ({
  default: () => false,
}));

vi.mock('react-virtuoso', () => ({
  Virtuoso: React.forwardRef(({ data, itemContent }: any, ref: any) => {
    React.useImperativeHandle(ref, () => ({ scrollToIndex: vi.fn() }));
    return (
      <div data-testid="virtuoso">
        {data?.map((item: any, index: number) => (
          <div key={index}>{itemContent(index, item)}</div>
        ))}
      </div>
    );
  }),
}));

const GAME_NAME = 'test-game';
const TOTAL_STEPS = 5;
const steps: BaseGameStep[] = Array.from({ length: TOTAL_STEPS }, (_, i) => makeStep({ step: i }));

const baseProps: ReasoningLogsProps = {
  closePanel: vi.fn(),
  onPlayChange: vi.fn(),
  onSpeedChange: vi.fn(),
  onStepChange: vi.fn(),
  playing: false,
  replayMode: 'condensed' as ReplayMode,
  setReplayMode: vi.fn(),
  speedModifier: 1,
  totalSteps: TOTAL_STEPS,
  steps,
  currentStep: 0,
  gameName: GAME_NAME,
};

const renderLogs = (overrides: Partial<ReasoningLogsProps> = {}) =>
  render(
    <ThemeProvider theme={theme}>
      <ReasoningLogs {...baseProps} {...overrides} />
    </ThemeProvider>
  );

afterEach(cleanup);

beforeEach(() => {
  vi.mocked(baseProps.closePanel).mockClear();
  vi.mocked(baseProps.onPlayChange).mockClear();
  vi.mocked(baseProps.onSpeedChange).mockClear();
  vi.mocked(baseProps.onStepChange).mockClear();
  vi.mocked(baseProps.setReplayMode).mockClear();
});

describe('ReasoningLogs', () => {
  describe('header', () => {
    it('shows Game Log heading', () => {
      renderLogs();
      expect(screen.getByText('Game Log')).toBeDefined();
    });

    it('calls closePanel when collapse button is clicked', () => {
      const closePanel = vi.fn();
      renderLogs({ closePanel });
      fireEvent.click(screen.getByLabelText('Collapse Episodes'));
      expect(closePanel).toHaveBeenCalled();
    });
  });

  describe('player controls', () => {
    it('restart sets step to 0 and starts playing', () => {
      const onPlayChange = vi.fn();
      const onStepChange = vi.fn();
      renderLogs({ currentStep: 3, onPlayChange, onStepChange });
      fireEvent.click(screen.getByLabelText('Restart'));
      expect(onPlayChange).toHaveBeenCalledWith(true);
      expect(onStepChange).toHaveBeenCalledWith(0);
    });

    it('previous step pauses and decrements', () => {
      const onPlayChange = vi.fn();
      const onStepChange = vi.fn();
      renderLogs({ currentStep: 3, playing: true, onPlayChange, onStepChange });
      fireEvent.click(screen.getByLabelText('Previous Step'));
      expect(onPlayChange).toHaveBeenCalledWith(false);
      expect(onStepChange).toHaveBeenCalledWith(2);
    });

    it('previous step does nothing at step 0', () => {
      const onStepChange = vi.fn();
      renderLogs({ currentStep: 0, onStepChange });
      fireEvent.click(screen.getByLabelText('Previous Step'));
      expect(onStepChange).not.toHaveBeenCalled();
    });

    it('shows Play label when paused', () => {
      renderLogs({ playing: false });
      expect(screen.getByLabelText('Play')).toBeDefined();
    });

    it('shows Pause label when playing', () => {
      renderLogs({ playing: true });
      expect(screen.getByLabelText('Pause')).toBeDefined();
    });

    it('toggles play/pause on click', () => {
      const onPlayChange = vi.fn();
      renderLogs({ playing: false, onPlayChange });
      fireEvent.click(screen.getByLabelText('Play'));
      expect(onPlayChange).toHaveBeenCalled();
    });

    it('next step pauses and increments', () => {
      const onPlayChange = vi.fn();
      const onStepChange = vi.fn();
      renderLogs({ currentStep: 2, onPlayChange, onStepChange });
      fireEvent.click(screen.getByLabelText('Next Step'));
      expect(onPlayChange).toHaveBeenCalledWith(false);
      expect(onStepChange).toHaveBeenCalledWith(3);
    });

    it('next step does nothing at last step', () => {
      const onStepChange = vi.fn();
      renderLogs({ currentStep: TOTAL_STEPS - 1, onStepChange });
      fireEvent.click(screen.getByLabelText('Next Step'));
      expect(onStepChange).not.toHaveBeenCalled();
    });
  });

  describe('step counter', () => {
    it('shows 1-indexed step / total', () => {
      renderLogs({ currentStep: 2 });
      expect(screen.getByTestId('step-counter').textContent).toBe('3/5');
    });

    it('hides counter when totalSteps is 0', () => {
      renderLogs({ totalSteps: 0 });
      expect(screen.queryByTestId('step-counter')).toBeNull();
    });
  });

  describe('replay mode toggle', () => {
    it('shows Log View button in zen mode', () => {
      renderLogs({ replayMode: 'zen' });
      expect(screen.getByText('Log View')).toBeDefined();
    });

    it('shows Streaming View button in condensed mode', () => {
      renderLogs({ replayMode: 'condensed' });
      expect(screen.getByText('Streaming View')).toBeDefined();
    });

    it('switches from zen to condensed on click', () => {
      const setReplayMode = vi.fn();
      renderLogs({ replayMode: 'zen', setReplayMode });
      fireEvent.click(screen.getByText('Log View'));
      expect(setReplayMode).toHaveBeenCalledWith('condensed');
    });

    it('switches from condensed to zen on click', () => {
      const setReplayMode = vi.fn();
      renderLogs({ replayMode: 'condensed', setReplayMode });
      fireEvent.click(screen.getByText('Streaming View'));
      expect(setReplayMode).toHaveBeenCalledWith('zen');
    });
  });

  describe('expand/collapse all', () => {
    it('shows Expand All button in condensed mode', () => {
      renderLogs({ replayMode: 'condensed' });
      expect(screen.getByText('Expand All')).toBeDefined();
    });

    it('does not show Expand All button in zen mode', () => {
      renderLogs({ replayMode: 'zen' });
      expect(screen.queryByText('Expand All')).toBeNull();
    });

    it('toggles between Expand and Collapse on click', () => {
      renderLogs({ replayMode: 'condensed' });
      fireEvent.click(screen.getByText('Expand All'));
      expect(screen.getByText('Collapse All')).toBeDefined();
    });
  });

  describe('controls visibility', () => {
    it('hides controls in only-stream mode', () => {
      renderLogs({ replayMode: 'only-stream' });
      expect(screen.queryByLabelText('Play')).toBeNull();
      expect(screen.queryByLabelText('Pause')).toBeNull();
    });
  });

  describe('loading state', () => {
    it('shows loading spinner when steps array is empty', () => {
      renderLogs({ steps: [], totalSteps: 0 });
      expect(screen.getByRole('progressbar')).toBeDefined();
    });

    it('does not show spinner when steps exist', () => {
      renderLogs();
      expect(screen.queryByRole('progressbar')).toBeNull();
    });
  });

  describe('share button', () => {
    it('sends postMessage in iframe context', () => {
      const postMessageSpy = vi.fn();
      const originalParent = window.parent;
      Object.defineProperty(window, 'parent', {
        value: { postMessage: postMessageSpy },
        writable: true,
        configurable: true,
      });

      renderLogs({ currentStep: 3 });
      fireEvent.click(screen.getByLabelText('Share Episode'));
      expect(postMessageSpy).toHaveBeenCalledWith({ shareEpisode: { step: 3 } }, '*');

      Object.defineProperty(window, 'parent', {
        value: originalParent,
        writable: true,
        configurable: true,
      });
    });
  });
});
