// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act, cleanup } from '@testing-library/react';
import { ThemeProvider } from '@mui/material/styles';
import { ReasoningStep, ReasoningStepProps } from './ReasoningStep';
import { makeStep } from '../test-utils';
import { ReplayMode } from '../types';
import { theme } from '../theme';

const STEP_WITH_THOUGHTS = makeStep({
  players: [
    {
      id: 0,
      name: 'Alice',
      thumbnail: '',
      isTurn: true,
      actionDisplayText: 'moved pawn',
      thoughts: 'I think this is a good move',
    },
    { id: 1, name: 'Bob', thumbnail: '', isTurn: false },
  ],
});

const STEP_LABEL_ONLY = makeStep({
  players: [
    { id: 0, name: 'Alice', thumbnail: '', isTurn: true, actionDisplayText: 'moved pawn' },
    { id: 1, name: 'Bob', thumbnail: '', isTurn: false },
  ],
});

const STEP_THOUGHTS_ONLY = makeStep({
  players: [
    { id: 0, name: 'Alice', thumbnail: '', isTurn: true, thoughts: 'thinking hard' },
    { id: 1, name: 'Bob', thumbnail: '', isTurn: false },
  ],
});

const STEP_EMPTY = makeStep({
  players: [
    { id: 0, name: 'Alice', thumbnail: '', isTurn: true },
    { id: 1, name: 'Bob', thumbnail: '', isTurn: false },
  ],
});

const baseProps: ReasoningStepProps = {
  expandByDefault: true,
  isCurrentStep: false,
  showExpandButton: true,
  step: STEP_WITH_THOUGHTS,
  stepNumber: 1,
  replayMode: 'condensed' as ReplayMode,
  scrollLogs: vi.fn(),
  playing: false,
  gameName: 'test-game',
  onStepChange: vi.fn(),
};

const renderStep = (overrides: Partial<ReasoningStepProps> = {}) =>
  render(
    <ThemeProvider theme={theme}>
      <ReasoningStep {...baseProps} {...overrides} />
    </ThemeProvider>
  );

afterEach(cleanup);

beforeEach(() => {
  vi.mocked(baseProps.scrollLogs).mockClear();
  vi.mocked(baseProps.onStepChange).mockClear();
});

describe('ReasoningStep', () => {
  describe('rendering states', () => {
    it('returns null when both label and description are empty', () => {
      const { container } = renderStep({ step: STEP_EMPTY });
      expect(container.innerHTML).toBe('');
    });

    it('renders simplified card with description when label is empty', () => {
      renderStep({ step: STEP_THOUGHTS_ONLY });
      expect(screen.getByText('thinking hard')).toBeDefined();
      expect(screen.getByText('Alice')).toBeDefined();
    });

    it('renders simplified card with label when description is empty', () => {
      renderStep({ step: STEP_LABEL_ONLY });
      expect(screen.getByText('moved pawn')).toBeDefined();
    });

    it('renders full card with both label and description', () => {
      renderStep({ step: STEP_WITH_THOUGHTS, expandByDefault: true, replayMode: 'condensed' });
      expect(screen.getByText('moved pawn')).toBeDefined();
      expect(screen.getByText('I think this is a good move')).toBeDefined();
    });
  });

  describe('card interaction', () => {
    it('calls onStepChange with 0-indexed step on click', () => {
      const onStepChange = vi.fn();
      const stepNumber = 3;
      renderStep({ onStepChange, stepNumber });
      const cards = screen.getAllByRole('button');
      fireEvent.click(cards[0]);
      expect(onStepChange).toHaveBeenCalledWith(stepNumber - 1);
    });

    it('calls onStepChange on simplified card click', () => {
      const onStepChange = vi.fn();
      const stepNumber = 5;
      renderStep({ step: STEP_LABEL_ONLY, onStepChange, stepNumber });
      fireEvent.click(screen.getByRole('button'));
      expect(onStepChange).toHaveBeenCalledWith(stepNumber - 1);
    });
  });

  describe('step number display', () => {
    it('shows the step number', () => {
      renderStep({ stepNumber: 7 });
      expect(screen.getByText('7')).toBeDefined();
    });
  });

  describe('player display', () => {
    it('shows player name when a player has isTurn', () => {
      renderStep({ step: STEP_WITH_THOUGHTS });
      expect(screen.getByText('Alice')).toBeDefined();
    });

    it('shows System when no player has isTurn', () => {
      const systemStep = makeStep({
        players: [
          { id: 0, name: 'Alice', thumbnail: '', isTurn: false },
          { id: 1, name: 'Bob', thumbnail: '', isTurn: false },
        ],
      });
      renderStep({
        step: systemStep,
        getStepLabel: () => 'system action',
      });
      expect(screen.getByText('System')).toBeDefined();
    });
  });

  describe('expand/collapse', () => {
    it('shows expand button when showExpandButton is true and description exists', () => {
      renderStep({ showExpandButton: true, expandByDefault: false });
      expect(screen.getByText('Show thinking')).toBeDefined();
    });

    it('hides expand button when showExpandButton is false', () => {
      renderStep({ showExpandButton: false });
      expect(screen.queryByText('Show thinking')).toBeNull();
      expect(screen.queryByText('Hide thinking')).toBeNull();
    });

    it('toggles between Show/Hide on button click', () => {
      renderStep({ expandByDefault: false, showExpandButton: true });
      expect(screen.getByText('Show thinking')).toBeDefined();
      fireEvent.click(screen.getByText('Show thinking'));
      expect(screen.getByText('Hide thinking')).toBeDefined();
    });

    it('shows description content when expanded', () => {
      renderStep({ expandByDefault: true });
      expect(screen.getByText('I think this is a good move')).toBeDefined();
    });

    it('hides description content when collapsed', () => {
      renderStep({ expandByDefault: false });
      expect(screen.queryByText('I think this is a good move')).toBeNull();
    });

    it('syncs with expandByDefault prop changes', () => {
      const { rerender } = render(
        <ThemeProvider theme={theme}>
          <ReasoningStep {...baseProps} expandByDefault={false} />
        </ThemeProvider>
      );
      expect(screen.queryByText('I think this is a good move')).toBeNull();

      rerender(
        <ThemeProvider theme={theme}>
          <ReasoningStep {...baseProps} expandByDefault={true} />
        </ThemeProvider>
      );
      expect(screen.getByText('I think this is a good move')).toBeDefined();
    });
  });

  describe('custom getters', () => {
    it('uses custom getStepLabel when provided', () => {
      const customLabel = 'custom label';
      renderStep({
        getStepLabel: () => customLabel,
        getStepDescription: () => '',
      });
      expect(screen.getByText(customLabel)).toBeDefined();
    });

    it('uses custom getStepDescription when provided', () => {
      const customDescription = 'custom description';
      renderStep({
        getStepLabel: () => 'a label',
        getStepDescription: () => customDescription,
        expandByDefault: true,
      });
      expect(screen.getByText(customDescription)).toBeDefined();
    });
  });

  describe('token streaming', () => {
    beforeEach(() => vi.useFakeTimers());
    afterEach(() => vi.useRealTimers());

    it('shows full text immediately in condensed mode', () => {
      renderStep({
        isCurrentStep: true,
        playing: true,
        replayMode: 'condensed',
        expandByDefault: true,
      });
      expect(screen.getByText('I think this is a good move')).toBeDefined();
    });
  });

  describe('scroll behavior', () => {
    it('calls scrollLogs on expand when isCurrentStep', () => {
      vi.useFakeTimers();
      const scrollLogs = vi.fn();
      renderStep({
        expandByDefault: false,
        showExpandButton: true,
        isCurrentStep: true,
        scrollLogs,
      });

      fireEvent.click(screen.getByText('Show thinking'));
      act(() => {
        vi.advanceTimersByTime(300);
      });

      expect(scrollLogs).toHaveBeenCalledWith(true);
      vi.useRealTimers();
    });
  });
});
