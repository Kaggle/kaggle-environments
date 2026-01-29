import { h, FunctionComponent } from 'preact';
import { ICON_PATHS, IconType } from './icons';
import { SpeedSelector } from './SpeedSelector';
import { StepSlider } from './StepSlider';
import { InterestingEvent, ReplayMode } from '../types';

export interface PlayerControlsProps {
  // State
  playing: boolean;
  step: number;
  totalSteps: number;
  speedModifier: number;
  replayMode?: ReplayMode;

  // Features
  interestingEvents?: InterestingEvent[];
  showReplayModeToggle?: boolean;

  // Callbacks
  onPlay: () => void;
  onPause: () => void;
  onStepChange: (step: number) => void;
  onSpeedChange: (speed: number) => void;
  onReplayModeChange?: (mode: ReplayMode) => void;

  // Config
  disabled?: boolean;
  speedOptions?: number[];
}

interface IconButtonProps {
  icon: IconType;
  onClick: () => void;
  disabled?: boolean;
  id?: string;
}

const IconButton: FunctionComponent<IconButtonProps> = ({ icon, onClick, disabled = false, id }) => {
  return h(
    'button',
    { id, onClick, disabled },
    h(
      'svg',
      {
        xmlns: 'http://www.w3.org/2000/svg',
        width: '24px',
        height: '24px',
        viewBox: '0 0 24 24',
        fill: '#FFFFFF',
      },
      h('path', { d: ICON_PATHS[icon] })
    )
  );
};

export const PlayerControls: FunctionComponent<PlayerControlsProps> = ({
  playing,
  step,
  totalSteps,
  speedModifier,
  replayMode,
  interestingEvents,
  showReplayModeToggle = false,
  onPlay,
  onPause,
  onStepChange,
  onSpeedChange,
  onReplayModeChange,
  disabled = false,
  speedOptions,
}) => {
  const maxStep = totalSteps > 0 ? totalSteps - 1 : 0;

  const handlePlayPauseClick = () => {
    if (playing) {
      onPause();
    } else {
      onPlay();
    }
  };

  const handlePrevClick = () => {
    onStepChange(step - 1);
  };

  const handleNextClick = () => {
    onStepChange(step + 1);
  };

  const handleSliderInput = (newStep: number) => {
    onPause();
    onStepChange(newStep);
  };

  const handleSliderChange = (newStep: number) => {
    onStepChange(newStep);
  };

  const handleReplayModeToggle = () => {
    if (!onReplayModeChange) return;
    const nextMode: ReplayMode = replayMode === 'condensed' ? 'zen' : 'condensed';
    onReplayModeChange(nextMode);
  };

  const stepDisplay = `${step + 1} / ${totalSteps}`;

  const children: any[] = [
    h(IconButton, {
      id: 'play-pause',
      icon: playing ? 'pause' : 'play',
      onClick: handlePlayPauseClick,
      disabled,
    }),
    h(IconButton, {
      id: 'prev',
      icon: 'prev',
      onClick: handlePrevClick,
      disabled: disabled || step === 0,
    }),
    h(StepSlider, {
      value: step,
      max: maxStep,
      disabled,
      interestingEvents,
      onInput: handleSliderInput,
      onChange: handleSliderChange,
    }),
    h(IconButton, {
      id: 'next',
      icon: 'next',
      onClick: handleNextClick,
      disabled: disabled || step === maxStep,
    }),
    h('span', { class: 'step-counter' }, stepDisplay),
    h(SpeedSelector, {
      value: speedModifier,
      options: speedOptions,
      onChange: onSpeedChange,
      disabled,
    }),
  ];

  if (showReplayModeToggle && onReplayModeChange) {
    children.push(
      h(
        'button',
        {
          class: 'replay-mode-toggle',
          onClick: handleReplayModeToggle,
          title: `Mode: ${replayMode ?? 'condensed'}`,
        },
        replayMode === 'zen' ? 'Zen' : 'Fast'
      )
    );
  }

  return h('div', { class: 'controls' }, children);
};
