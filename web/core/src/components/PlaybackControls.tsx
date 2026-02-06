import * as React from 'react';
import { InterestingEvent, ReplayMode } from '../types';

export interface PlaybackControlsProps {
  // State
  playing: boolean;
  currentStep: number;
  totalSteps: number;
  speedModifier: number;
  replayMode: ReplayMode;

  // Callbacks
  onPlayChange: (playing?: boolean) => void;
  onStepChange: (step: number) => void;
  onSpeedChange: (speed: number) => void;
  onReplayModeChange?: (mode: ReplayMode) => void;

  // Optional features
  interestingEvents?: InterestingEvent[];
  showReplayModeToggle?: boolean;
  speedOptions?: number[];

  // Styling
  className?: string;
  style?: React.CSSProperties;
}

const DEFAULT_SPEED_OPTIONS = [0.5, 0.75, 1, 1.5, 2];

const controlsStyles: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: '8px',
  padding: '8px 16px',
  backgroundColor: '#1a1a1a',
  borderRadius: '8px',
};

const buttonStyles: React.CSSProperties = {
  background: 'none',
  border: 'none',
  cursor: 'pointer',
  padding: '8px',
  borderRadius: '4px',
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  color: '#fff',
};

const stepCounterStyles: React.CSSProperties = {
  color: '#fff',
  fontSize: '14px',
  minWidth: '80px',
  textAlign: 'center',
};

const selectStyles: React.CSSProperties = {
  padding: '4px 8px',
  borderRadius: '4px',
  border: '1px solid #444',
  backgroundColor: '#2a2a2a',
  color: '#fff',
  cursor: 'pointer',
};

export const PlaybackControls: React.FC<PlaybackControlsProps> = ({
  playing,
  currentStep,
  totalSteps,
  speedModifier,
  replayMode,
  onPlayChange,
  onStepChange,
  onSpeedChange,
  onReplayModeChange,
  interestingEvents,
  showReplayModeToggle = false,
  speedOptions = DEFAULT_SPEED_OPTIONS,
  className,
  style,
}) => {
  const maxStep = totalSteps > 0 ? totalSteps - 1 : 0;

  const handlePlayPauseClick = () => {
    onPlayChange(!playing);
  };

  const handlePrevClick = () => {
    onPlayChange(false);
    onStepChange(Math.max(0, currentStep - 1));
  };

  const handleNextClick = () => {
    onPlayChange(false);
    onStepChange(Math.min(maxStep, currentStep + 1));
  };

  const handleRestartClick = () => {
    onStepChange(0);
    onPlayChange(true);
  };

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onPlayChange(false);
    onStepChange(parseInt(e.target.value, 10));
  };

  const handleSpeedChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onSpeedChange(parseFloat(e.target.value));
  };

  const handleReplayModeToggle = () => {
    if (!onReplayModeChange) return;
    const nextMode: ReplayMode = replayMode === 'condensed' ? 'zen' : 'condensed';
    onReplayModeChange(nextMode);
  };

  const stepDisplay = `${currentStep + 1} / ${totalSteps}`;

  return (
    <div className={className} style={{ ...controlsStyles, ...style }}>
      <button style={buttonStyles} onClick={handleRestartClick} title="Restart" aria-label="Restart">
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z" />
        </svg>
      </button>

      <button
        style={buttonStyles}
        onClick={handlePrevClick}
        disabled={currentStep === 0}
        title="Previous step"
        aria-label="Previous step"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M6 18V6h2v12H6zm3.5-6L18 6v12l-8.5-6z" />
        </svg>
      </button>

      <button
        style={buttonStyles}
        onClick={handlePlayPauseClick}
        title={playing ? 'Pause' : 'Play'}
        aria-label={playing ? 'Pause' : 'Play'}
      >
        {playing ? (
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
          </svg>
        ) : (
          <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="currentColor">
            <path d="M8 5v14l11-7z" />
          </svg>
        )}
      </button>

      <button
        style={buttonStyles}
        onClick={handleNextClick}
        disabled={currentStep === maxStep}
        title="Next step"
        aria-label="Next step"
      >
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
          <path d="M7 18l8.5-6L7 6v12zM15 6v12h2V6h-2z" />
        </svg>
      </button>

      <div style={{ flex: 1, position: 'relative', margin: '0 16px' }}>
        <input
          type="range"
          min="0"
          max={maxStep}
          value={currentStep}
          onChange={handleSliderChange}
          aria-label="Step slider"
          style={{
            width: '100%',
            height: '6px',
            cursor: 'pointer',
            accentColor: '#1ebeff',
          }}
        />
        {interestingEvents && interestingEvents.length > 0 && (
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: 0,
              right: 0,
              pointerEvents: 'none',
            }}
          >
            {interestingEvents.map((event, index) => {
              const position = maxStep > 0 ? (event.step / maxStep) * 100 : 0;
              return (
                <div
                  key={`${event.step}-${index}`}
                  style={{
                    position: 'absolute',
                    left: `${position}%`,
                    transform: 'translate(-50%, -50%)',
                    width: '10px',
                    height: '10px',
                    borderRadius: '50%',
                    backgroundColor: '#1ebeff',
                    border: '2px solid #fff',
                    pointerEvents: 'auto',
                    cursor: 'pointer',
                    zIndex: 10,
                  }}
                  title={event.description}
                  onClick={() => {
                    onPlayChange(false);
                    onStepChange(event.step);
                  }}
                />
              );
            })}
          </div>
        )}
      </div>

      <span style={stepCounterStyles}>{stepDisplay}</span>

      <select value={speedModifier} onChange={handleSpeedChange} style={selectStyles} aria-label="Playback speed">
        {speedOptions.map((speed) => (
          <option key={speed} value={speed}>
            {speed}x
          </option>
        ))}
      </select>

      {showReplayModeToggle && onReplayModeChange && (
        <button
          style={{ ...buttonStyles, fontSize: '12px', padding: '4px 8px' }}
          onClick={handleReplayModeToggle}
          title={`Mode: ${replayMode}`}
        >
          {replayMode === 'zen' ? 'Zen' : 'Fast'}
        </button>
      )}
    </div>
  );
};
