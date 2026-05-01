import * as React from 'react';
export interface PlaybackControlsProps {
  // State
  playing: boolean;
  currentStep: number;
  totalSteps: number;
  speedModifier: number;

  // Callbacks
  onPlayChange: (playing?: boolean) => void;
  onStepChange: (step: number) => void;
  onSpeedChange?: (speed: number) => void;

  // Optional episode seed; when provided, shows a click-to-copy badge.
  seed?: number | string;

  // Styling
  className?: string;
  style?: React.CSSProperties;
}

const SPEED_OPTIONS = [0.25, 0.5, 1, 1.5, 2, 4];

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

const speedSelectStyles: React.CSSProperties = {
  background: '#2a2a2a',
  color: '#fff',
  border: '1px solid #444',
  borderRadius: '4px',
  padding: '4px 6px',
  fontSize: '13px',
  cursor: 'pointer',
};

const seedBadgeStyles: React.CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: '6px',
  background: '#2a2a2a',
  color: '#fff',
  border: '1px solid #444',
  borderRadius: '4px',
  padding: '4px 8px',
  fontSize: '13px',
  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
  cursor: 'pointer',
};

export const PlaybackControls: React.FC<PlaybackControlsProps> = ({
  playing,
  currentStep,
  totalSteps,
  speedModifier,
  onPlayChange,
  onStepChange,
  onSpeedChange,
  seed,
  className,
  style,
}) => {
  const maxStep = totalSteps > 0 ? totalSteps - 1 : 0;
  const [copied, setCopied] = React.useState(false);

  const handleCopySeed = async () => {
    if (seed === undefined || seed === null) return;
    const text = String(seed);
    try {
      if (navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
      } else {
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
      }
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      // ignore
    }
  };

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

  const stepDisplay = `${currentStep + 1} / ${totalSteps}`;

  return (
    <div className={className} style={{ ...controlsStyles, ...style }}>
      {seed !== undefined && seed !== null && (
        <button
          type="button"
          style={seedBadgeStyles}
          onClick={handleCopySeed}
          title={copied ? 'Copied!' : 'Click to copy seed'}
          aria-label={`Episode seed ${seed}. Click to copy.`}
        >
          <span>Seed: {String(seed)}</span>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="14"
            height="14"
            viewBox="0 0 24 24"
            fill="currentColor"
            aria-hidden="true"
          >
            {copied ? (
              <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41L9 16.17z" />
            ) : (
              <path d="M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z" />
            )}
          </svg>
        </button>
      )}
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
      </div>
      <span style={stepCounterStyles}>{stepDisplay}</span>

      {onSpeedChange && (
        <select
          style={speedSelectStyles}
          value={speedModifier}
          onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
          aria-label="Playback speed"
          title="Playback speed"
        >
          {SPEED_OPTIONS.map((s) => (
            <option key={s} value={s}>
              {s}x
            </option>
          ))}
        </select>
      )}
    </div>
  );
};
