import { h, FunctionComponent } from 'preact';
import { useState, useRef } from 'preact/hooks';
import { InterestingEvent } from '../types';

export interface StepSliderProps {
  value: number;
  max: number;
  disabled?: boolean;
  interestingEvents?: InterestingEvent[];
  onInput: (step: number) => void;
  onChange: (step: number) => void;
}

export const StepSlider: FunctionComponent<StepSliderProps> = ({
  value,
  max,
  disabled = false,
  interestingEvents,
  onInput,
  onChange,
}) => {
  const [hoveredEvent, setHoveredEvent] = useState<InterestingEvent | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState<{ left: number; top: number } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleInput = (e: Event) => {
    const target = e.target as HTMLInputElement;
    onInput(parseInt(target.value, 10));
  };

  const handleChange = (e: Event) => {
    const target = e.target as HTMLInputElement;
    onChange(parseInt(target.value, 10));
  };

  const handleMarkerClick = (event: InterestingEvent) => {
    onChange(event.step);
  };

  const handleMarkerHover = (event: InterestingEvent, markerElement: HTMLElement) => {
    if (!containerRef.current) return;

    const containerRect = containerRef.current.getBoundingClientRect();
    const markerRect = markerElement.getBoundingClientRect();

    setHoveredEvent(event);
    setTooltipPosition({
      left: markerRect.left - containerRect.left + markerRect.width / 2,
      top: -8,
    });
  };

  const handleMarkerLeave = () => {
    setHoveredEvent(null);
    setTooltipPosition(null);
  };

  const getMarkerPosition = (step: number): number => {
    if (max === 0) return 0;
    return (step / max) * 100;
  };

  const hasEvents = interestingEvents && interestingEvents.length > 0;

  const children: any[] = [
    h('input', {
      type: 'range',
      min: '0',
      max: max >= 0 ? max : 0,
      value,
      disabled,
      onInput: handleInput,
      onChange: handleChange,
    }),
  ];

  if (hasEvents) {
    children.push(
      h(
        'div',
        { class: 'event-markers' },
        interestingEvents.map((event, index) =>
          h('div', {
            key: `${event.step}-${index}`,
            class: 'event-marker',
            style: { left: `${getMarkerPosition(event.step)}%` },
            onClick: () => handleMarkerClick(event),
            onMouseEnter: (e: MouseEvent) => handleMarkerHover(event, e.currentTarget as HTMLElement),
            onMouseLeave: handleMarkerLeave,
            title: event.description,
          })
        )
      )
    );
  }

  if (hoveredEvent && tooltipPosition) {
    children.push(
      h(
        'div',
        {
          class: 'event-tooltip',
          style: {
            left: `${tooltipPosition.left}px`,
            top: `${tooltipPosition.top}px`,
          },
        },
        hoveredEvent.description
      )
    );
  }

  return h('div', { class: 'slider-container', ref: containerRef }, children);
};
