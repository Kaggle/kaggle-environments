// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, act, cleanup } from '@testing-library/react';
import * as React from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { ReasoningStep, ReasoningStepProps } from './ReasoningStep';
import { makeStep } from '../test-utils';
import { BaseGameStep, ReplayMode } from '../types';

vi.mock('@mui/material/useMediaQuery', () => ({
  default: () => false,
}));

vi.mock('../components/UserContent', () => ({
  UserContent: React.forwardRef(({ markdown, ...rest }: any, ref: any) => (
    <div ref={ref} data-testid="user-content" {...rest}>
      {markdown}
    </div>
  )),
}));

const testTheme = createTheme({
  breakpoints: {
    values: {
      xs: 0,
      sm: 600,
      md: 960,
      lg: 1280,
      xl: 1920,
      tablet: 840,
      desktop: 1280,
      phone: 480,
      xs1: 360,
      xs2: 400,
      xs3: 480,
      sm1: 600,
      sm2: 720,
      sm3: 840,
      md1: 960,
      md2: 1024,
      lg1: 1280,
      lg2: 1440,
      lg3: 1600,
      xl1: 1920,
    } as any,
  },
});

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

const renderStep = (overrides: Partial<ReasoningStepProps> = {}) => {
  const props = { ...baseProps, ...overrides };
  return render(
    <ThemeProvider theme={testTheme}>
      <ReasoningStep {...props} />
    </ThemeProvider>
  );
};

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

    it('calls onStepChange on simplified card click too', () => {
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
        // Need at least label or description to render
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
        <ThemeProvider theme={testTheme}>
          <ReasoningStep {...baseProps} expandByDefault={false} />
        </ThemeProvider>
      );
      expect(screen.queryByText('I think this is a good move')).toBeNull();

      rerender(
        <ThemeProvider theme={testTheme}>
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

    it('streams tokens progressively when playing in streaming mode on current step', () => {
      const description = 'one two three';
      renderStep({
        isCurrentStep: true,
        playing: true,
        replayMode: 'zen',
        expandByDefault: true,
        getStepLabel: () => 'action',
        getStepDescription: () => description,
      });

      act(() => {
        vi.advanceTimersByTime(500);
      });

      const contents = screen.getAllByTestId('user-content');
      const descriptionContent = contents.find((el) => el.textContent !== 'action');
      expect(descriptionContent).toBeDefined();
    });

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

  describe('avatar rendering', () => {
    it('renders avatar image when player has thumbnail', () => {
      const thumbnailUrl = 'https://example.com/alice.png';
      const stepWithThumb = makeStep({
        players: [
          { id: 0, name: 'Alice', thumbnail: thumbnailUrl, isTurn: true, actionDisplayText: 'move' },
          { id: 1, name: 'Bob', thumbnail: '', isTurn: false },
        ],
      });
      const { container } = renderStep({ step: stepWithThumb });
      const img = container.querySelector('img');
      expect(img).not.toBeNull();
      expect(img!.getAttribute('src')).toBe(thumbnailUrl);
    });
  });
});
