import { h, FunctionComponent } from 'preact';

export interface SpeedSelectorProps {
  value: number;
  options?: number[];
  onChange: (speed: number) => void;
  disabled?: boolean;
}

const DEFAULT_SPEED_OPTIONS = [0.5, 0.75, 1, 1.5, 2];

export const SpeedSelector: FunctionComponent<SpeedSelectorProps> = ({
  value,
  options = DEFAULT_SPEED_OPTIONS,
  onChange,
  disabled = false,
}) => {
  const handleChange = (e: Event) => {
    const target = e.target as HTMLSelectElement;
    onChange(parseFloat(target.value));
  };

  return h(
    'select',
    {
      class: 'speed-selector',
      value: value.toString(),
      onChange: handleChange,
      disabled,
    },
    options.map((speed) =>
      h('option', { key: speed, value: speed.toString() }, `${speed}x`)
    )
  );
};
