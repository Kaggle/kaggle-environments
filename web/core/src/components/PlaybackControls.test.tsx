// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, fireEvent, cleanup } from '@testing-library/react';
import { PlaybackControls, PlaybackControlsProps } from './PlaybackControls';

const baseProps: PlaybackControlsProps = {
  playing: false,
  currentStep: 0,
  totalSteps: 10,
  speedModifier: 1,
  onPlayChange: vi.fn(),
  onStepChange: vi.fn(),
};

const renderControls = (overrides: Partial<PlaybackControlsProps> = {}) => {
  const props = { ...baseProps, ...overrides };
  return render(<PlaybackControls {...props} />);
};

afterEach(cleanup);

beforeEach(() => {
  vi.mocked(baseProps.onPlayChange).mockClear();
  vi.mocked(baseProps.onStepChange).mockClear();
});

describe('PlaybackControls', () => {
  describe('play/pause', () => {
    it('shows Play label when paused', () => {
      renderControls({ playing: false });
      expect(screen.getByLabelText('Play')).toBeDefined();
    });

    it('shows Pause label when playing', () => {
      renderControls({ playing: true });
      expect(screen.getByLabelText('Pause')).toBeDefined();
    });

    it('calls onPlayChange(true) when clicking Play', () => {
      const onPlayChange = vi.fn();
      renderControls({ playing: false, onPlayChange });
      fireEvent.click(screen.getByLabelText('Play'));
      expect(onPlayChange).toHaveBeenCalledWith(true);
    });

    it('calls onPlayChange(false) when clicking Pause', () => {
      const onPlayChange = vi.fn();
      renderControls({ playing: true, onPlayChange });
      fireEvent.click(screen.getByLabelText('Pause'));
      expect(onPlayChange).toHaveBeenCalledWith(false);
    });
  });

  describe('previous step', () => {
    it('pauses and decrements step', () => {
      const onPlayChange = vi.fn();
      const onStepChange = vi.fn();
      renderControls({ currentStep: 5, playing: true, onPlayChange, onStepChange });
      fireEvent.click(screen.getByLabelText('Previous step'));
      expect(onPlayChange).toHaveBeenCalledWith(false);
      expect(onStepChange).toHaveBeenCalledWith(4);
    });

    it('does not fire when already at step 0 (button disabled)', () => {
      const onStepChange = vi.fn();
      renderControls({ currentStep: 0, onStepChange });
      fireEvent.click(screen.getByLabelText('Previous step'));
      expect(onStepChange).not.toHaveBeenCalled();
    });

    it('is disabled at step 0', () => {
      renderControls({ currentStep: 0 });
      expect(screen.getByLabelText('Previous step').hasAttribute('disabled')).toBe(true);
    });

    it('is not disabled at step > 0', () => {
      renderControls({ currentStep: 3 });
      expect(screen.getByLabelText('Previous step').hasAttribute('disabled')).toBe(false);
    });
  });

  describe('next step', () => {
    it('pauses and increments step', () => {
      const onPlayChange = vi.fn();
      const onStepChange = vi.fn();
      renderControls({ currentStep: 3, totalSteps: 10, playing: true, onPlayChange, onStepChange });
      fireEvent.click(screen.getByLabelText('Next step'));
      expect(onPlayChange).toHaveBeenCalledWith(false);
      expect(onStepChange).toHaveBeenCalledWith(4);
    });

    it('does not fire when already at last step (button disabled)', () => {
      const onStepChange = vi.fn();
      renderControls({ currentStep: 9, totalSteps: 10, onStepChange });
      fireEvent.click(screen.getByLabelText('Next step'));
      expect(onStepChange).not.toHaveBeenCalled();
    });

    it('is disabled at last step', () => {
      renderControls({ currentStep: 9, totalSteps: 10 });
      expect(screen.getByLabelText('Next step').hasAttribute('disabled')).toBe(true);
    });

    it('is not disabled before last step', () => {
      renderControls({ currentStep: 5, totalSteps: 10 });
      expect(screen.getByLabelText('Next step').hasAttribute('disabled')).toBe(false);
    });
  });

  describe('restart', () => {
    it('sets step to 0 and starts playing', () => {
      const onPlayChange = vi.fn();
      const onStepChange = vi.fn();
      renderControls({ currentStep: 7, onPlayChange, onStepChange });
      fireEvent.click(screen.getByLabelText('Restart'));
      expect(onStepChange).toHaveBeenCalledWith(0);
      expect(onPlayChange).toHaveBeenCalledWith(true);
    });
  });

  describe('step slider', () => {
    it('pauses and sets step on change', () => {
      const onPlayChange = vi.fn();
      const onStepChange = vi.fn();
      renderControls({ playing: true, totalSteps: 10, onPlayChange, onStepChange });
      const slider = screen.getByLabelText('Step slider');
      fireEvent.change(slider, { target: { value: '5' } });
      expect(onPlayChange).toHaveBeenCalledWith(false);
      expect(onStepChange).toHaveBeenCalledWith(5);
    });

    it('has correct min/max bounds', () => {
      renderControls({ totalSteps: 10 });
      const slider = screen.getByLabelText('Step slider');
      expect(slider.getAttribute('min')).toBe('0');
      expect(slider.getAttribute('max')).toBe('9');
    });

    it('reflects current step value', () => {
      renderControls({ currentStep: 4, totalSteps: 10 });
      const slider = screen.getByLabelText('Step slider') as HTMLInputElement;
      expect(slider.value).toBe('4');
    });
  });

  describe('step counter', () => {
    it('displays 1-indexed step out of total', () => {
      renderControls({ currentStep: 0, totalSteps: 10 });
      expect(screen.getByText('1 / 10')).toBeDefined();
    });

    it('updates when step changes', () => {
      renderControls({ currentStep: 7, totalSteps: 10 });
      expect(screen.getByText('8 / 10')).toBeDefined();
    });
  });

  describe('speed selector', () => {
    it('is not rendered when onSpeedChange is not provided', () => {
      renderControls({ onSpeedChange: undefined });
      expect(screen.queryByLabelText('Playback speed')).toBeNull();
    });

    it('is rendered when onSpeedChange is provided', () => {
      renderControls({ onSpeedChange: vi.fn() });
      expect(screen.getByLabelText('Playback speed')).toBeDefined();
    });

    it('calls onSpeedChange with the selected value', () => {
      const onSpeedChange = vi.fn();
      renderControls({ onSpeedChange, speedModifier: 1 });
      fireEvent.change(screen.getByLabelText('Playback speed'), { target: { value: '2' } });
      expect(onSpeedChange).toHaveBeenCalledWith(2);
    });

    it('shows the current speed as selected', () => {
      renderControls({ onSpeedChange: vi.fn(), speedModifier: 0.5 });
      const select = screen.getByLabelText('Playback speed') as HTMLSelectElement;
      expect(select.value).toBe('0.5');
    });

    it('renders all speed options', () => {
      const expectedSpeeds = [0.25, 0.5, 1, 1.5, 2, 4];
      renderControls({ onSpeedChange: vi.fn() });
      const select = screen.getByLabelText('Playback speed') as HTMLSelectElement;
      const optionValues = Array.from(select.options).map((o) => parseFloat(o.value));
      expect(optionValues).toEqual(expectedSpeeds);
    });
  });

  describe('edge cases', () => {
    it('handles totalSteps of 0 without crashing', () => {
      renderControls({ totalSteps: 0, currentStep: 0 });
      expect(screen.getByText('1 / 0')).toBeDefined();
    });

    it('handles totalSteps of 1', () => {
      renderControls({ totalSteps: 1, currentStep: 0 });
      expect(screen.getByLabelText('Previous step').hasAttribute('disabled')).toBe(true);
      expect(screen.getByLabelText('Next step').hasAttribute('disabled')).toBe(true);
    });
  });
});
