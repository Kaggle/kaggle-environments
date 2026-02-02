import { ReasoningStep } from "./ReasoningStep";
import {
  BaseGameStep,
  InterestingEvent,
  ReplayMode,
} from "@kaggle-environments/core";
import useMediaQuery from "@mui/material/useMediaQuery";
import * as React from "react";
import { useSearchParams } from "react-router";
import { Virtuoso, VirtuosoHandle, Components } from "react-virtuoso";
import { styled } from "@mui/material/styles";
import { Button, CircularProgress, Icon, IconButton, List, MenuItem, Select, Slider, Tooltip, Typography } from "@mui/material";

interface SelectOption<T> {
  label: string;
  value: T;
  icon?: string;
  valueLabel?: React.ReactNode;
}

const SPEED_OPTIONS: SelectOption<number>[] = [
  {
    label: "Speed 0.5",
    value: 0.5,
    icon: "speed_0_5",
    valueLabel: <></>,
  },
  {
    label: "Speed 0.75",
    value: 0.75,
    icon: "speed_0_75",
    valueLabel: <></>,
  },
  {
    label: "Speed 1",
    value: 1,
    icon: "1x_mobiledata",
    valueLabel: <></>,
  },
  {
    label: "Speed 1.5",
    value: 1.5,
    icon: "speed_1_5",
    valueLabel: <></>,
  },
  {
    label: "Speed 2",
    value: 2,
    icon: "speed_2x",
    valueLabel: <></>,
  },
];

const IMMERSIVE_SLIDER_PANEL_HEADER_HEIGHT = "74px";

const LogsContainer = styled('div')`
  border-left: 1px solid ${p => p.theme.palette.divider};
  display: flex;
  flex-direction: column;
  min-width: 300px;

  height: calc(100% - ${IMMERSIVE_SLIDER_PANEL_HEADER_HEIGHT});
  min-height: 0;

  ${({ theme }) => theme.breakpoints.down('tablet')} {
    border-left: none;
    height: 100%;
  }
`;

const SidebarHeader = styled('div')`
  align-items: center;
  background-color: ${p => p.theme.palette.background.default};
  border-bottom: 1px solid ${p => p.theme.palette.divider};
  display: flex;
  flex-direction: column;
  gap: 8px;
  padding: 16px;
  flex-shrink: 0;

  ${({ theme }) => theme.breakpoints.down('tablet')} {
    border-top: 1px solid ${p => p.theme.palette.divider};
    padding: 8px 16px;
  }
`;

const PlayerControlsSection = styled('div')`
  align-items: center;
  display: flex;
  gap: 8px;
  justify-content: space-between;
  width: 100%;
`;

const GameLogHeader = styled(PlayerControlsSection)`
  height: 28px;
`;

const PlayerButtons = styled('div')`
  display: flex;
  gap: 8px;
`;

const SpacedListItem = styled('li')`
  /* Reset default li styles just in case */
  margin: 0;
  padding: 0;
  list-style: none;

  /* Apply the gap here.
    Virtuoso measures this element, so it will account for the extra space. 
    WARNING: if this is margin instead of padding it will cause Jitter
    */
  padding-top: 20px;
  box-sizing: border-box;

  ${({ theme }) => theme.breakpoints.down('tablet')} {
    padding-top: 8px;
  }
`;

const StepList = styled(List)`
  && {
    margin: 0;
    padding: 16px 24px;
  }
`;

/* Hide all scrollbars- they get distracting when content is constantly scrolling */
const LogsScroller = styled('div')`
  scrollbar-width: none;
  -ms-overflow-style: none;
  overflow-anchor: none;
  scrollbar-gutter: stable;

  &::-webkit-scrollbar {
    display: none;
    width: 0;
    background: transparent;
  }
`;

/* Invert the colors we usually do for Sliders to make this look more like a playback (think Youtube) */
const PlaybackSlider = styled(Slider)`
  margin-right: 4px;

  && .MuiSlider-rail {
    color: ${({ theme }) => theme.palette.text.secondary};
  }

  && .MuiSlider-track {
    color: ${({ theme }) => theme.palette.text.primary};
    opacity: 1;
  }

  && .MuiSlider-mark {
    display: none;
  }

  && .MuiSlider-markLabel {
    top: 50%;
    transform: translate(-50%, -50%);
    z-index: 2;
  }

  /* MUI slider adds a default margin-bottom of 20px when marks are used to accomodate them.
  We use marks but move them up onto the slider line itself, so we don't need this margin.*/
  &&& {
    margin-bottom: 0;
  }
`;

const MarkDot = styled('div')<{ $active?: boolean }>`
  background-color: ${({ theme }) => theme.palette.primary.main};
  height: 10px;
  width: 10px;
  border-radius: 50%;
  cursor: pointer;
  border: 2px solid
    ${({ theme, $active }) =>
    $active
      ? theme.palette.mode === "dark"
        ? theme.palette.common.white
        : theme.palette.common.black
      : theme.palette.background.default};
  &:hover {
    border: 2px solid
      ${({ theme }) =>
    theme.palette.mode === "dark"
      ? theme.palette.common.white
      : theme.palette.common.black};
  }
`;

const NoStepsContainer = styled('div')`
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
`;

const VirtuosoScrollerList = React.forwardRef(
  (props, ref: React.ForwardedRef<any>) => {
    return <StepList {...props} ref={ref} />;
  }
);

const VirtuosoListItem = React.forwardRef(
  (props, ref: React.ForwardedRef<any>) => {
    return <SpacedListItem {...props} ref={ref} />;
  }
);

const Scroller = React.forwardRef(
  ({ ...props }, ref: React.ForwardedRef<any>) => (
    <LogsScroller ref={ref} {...props} />
  )
);

export interface ReasoningLogs {
  closePanel: () => void;
  onPlayChange: (playing?: boolean) => void;
  onSpeedChange: (modifier: number) => void;
  onStepChange: (currentStep: number) => void;
  playing: boolean;
  replayMode: ReplayMode;
  setReplayMode: (streaming: ReplayMode) => void;
  speedModifier: number;
  totalSteps: number;
  steps: BaseGameStep[];
  currentStep: number;
  gameName: string;
  interestingEvents?: InterestingEvent[];
}

export const ReasoningLogs: React.FC<ReasoningLogs> = ({
  closePanel,
  onPlayChange,
  onSpeedChange,
  onStepChange,
  playing,
  replayMode,
  setReplayMode,
  speedModifier,
  totalSteps,
  steps,
  currentStep,
  gameName,
  interestingEvents,
}) => {
  // TODO(b/462451568) - add werewolf transformer to handle this instead
  const [expandAll, setExpandAll] = React.useState(false);

  const virtuosoRef = React.useRef<VirtuosoHandle>(null);
  const scrollerRef = React.useRef<HTMLElement | null>(null);
  const isAtBottomRef = React.useRef(true);

  /* Users can enter a new step in a text field or with a slider- 
  make sure we don't overwrite their input as the episode keeps streaming */
  const [isChangingStep, setIsChangingStep] = React.useState(false);
  const [userInputStep, setUserInputStep] = React.useState(currentStep);

  const [searchParams] = useSearchParams();
  const [hasCopied, setHasCopied] = React.useState(false);

  const isTablet = useMediaQuery(theme => theme.breakpoints.down('tablet'));
  const showControls = replayMode !== "only-stream";

  const visibleSteps = React.useMemo(() => {
    return steps.slice(0, currentStep + 1);
  }, [steps, currentStep]);

  const virtuosoComponents = React.useMemo<Components<BaseGameStep>>(
    () => ({
      List: VirtuosoScrollerList,
      Item: VirtuosoListItem as React.ComponentType<any>,
      // eslint-disable-next-line @typescript-eslint/naming-convention
      Footer: () => (
        <div
          // This buffer is basically our bottom padding so that the list
          // doesn't hug the bottom of the screen.
          style={{
            height:
              replayMode === "only-stream"
                ? "80px"
                : isTablet
                  ? "16px"
                  : "64px",
          }}
        />
      ),
      Scroller,
    }),
    [replayMode, isTablet]
  );

  const scrollLogs = React.useCallback(
    (forceScroll?: boolean) => {
      if ((isAtBottomRef.current || forceScroll) && virtuosoRef.current) {
        // We defer slightly to ensure the new data prop has propagated
        setTimeout(() => {
          virtuosoRef.current?.scrollToIndex({
            index: visibleSteps.length - 1,
            // 'start' puts the top of the step at the top of the container.
            // This is crucial for long steps so the user starts reading from the top.
            // If you prefer terminal-style "bottom locking", change this to 'end'.
            align: "start",
            behavior: "auto",
          });
        }, 50);
      }
    },
    [visibleSteps.length]
  );

  React.useEffect(() => {
    if (isAtBottomRef.current && scrollerRef.current) {
      // Use requestAnimationFrame to ensure the new DOM element is rendered
      requestAnimationFrame(() => {
        if (scrollerRef.current) {
          scrollerRef.current.scrollTo({
            top: scrollerRef.current.scrollHeight,
            behavior: "smooth", // Smooth is fine here for the big "New Card" jump
          });
        }
      });
    }
  }, [visibleSteps.length]);

  React.useEffect(() => {
    if (replayMode === "condensed") {
      setExpandAll(false);
    }
    // Virtuoso's `followOutput` prop handles most auto-scrolling,
    // but we still force one on mode change just in case.
    scrollLogs(true);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [replayMode]);

  React.useEffect(() => {
    /* Pause the episode if the user is currently changing the step */
    if (isChangingStep && playing === true) {
      onPlayChange(false);
    } else {
      scrollLogs(true);
      onPlayChange(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isChangingStep]);

  /* Force scrolling if the episode ID or step param change */
  React.useEffect(() => {
    scrollLogs(true);
  }, [scrollLogs, searchParams]);

  /* Only update the value we show in the slider and text field
  if the user is not currently changing it */
  React.useEffect(() => {
    if (!isChangingStep) {
      setUserInputStep(currentStep + 1);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStep]);

  return (
    <LogsContainer>
      <SidebarHeader>
        <GameLogHeader>
          <Typography variant="h6" component="h2">Game Log</Typography>
          <div>
            {replayMode !== "only-stream" && (
              <IconButton
                size={isTablet ? "small" : "medium"}
                onClick={closePanel}
                aria-label="Collapse Episodes"
              >
                <Icon>{isTablet ? "bottom_panel_close" : "right_panel_close"}</Icon>
              </IconButton>
            )}
            {replayMode !== "only-stream" && !isTablet && (
              <IconButton
                size="medium"
                onClick={() => {
                  searchParams.set("step", currentStep.toString());
                  const urlToCopy = `${window.location.origin}${window.location.pathname}?${searchParams.toString()}`;

                  navigator.clipboard.writeText(urlToCopy);
                  setHasCopied(true);
                  setTimeout(() => {
                    setHasCopied(false);
                  }, 3000);
                }}
                aria-label="Share Episode"
              >
                <Icon>{hasCopied ? "done" : "ios_share"}</Icon>
              </IconButton>
            )}
          </div>
        </GameLogHeader>
        {showControls && (
          <>
            <PlayerControlsSection>
              <PlayerButtons>
                <IconButton
                  size="large"
                  aria-label="Restart"
                  onClick={() => {
                    onPlayChange(/* playing= */ true);
                    onStepChange(0);
                  }}
                >
                  <Icon>refresh</Icon>
                </IconButton>
                <IconButton
                  size="large"
                  aria-label="Previous Step"
                  onClick={() => {
                    if (currentStep > 0) {
                      onPlayChange(/* playing= */ false);
                      onStepChange(currentStep - 1);
                      scrollLogs(true);
                    }
                  }}
                >
                  <Icon>skip_previous</Icon>
                </IconButton>
                <IconButton
                  size="large"
                  aria-label={playing ? "Pause" : "Play"}
                  onClick={() => onPlayChange()}
                >
                  <Icon>{playing ? "pause" : "play_arrow"}</Icon>
                </IconButton>
                <IconButton
                  size="large"
                  aria-label="Next Step"
                  onClick={() => {
                    if (currentStep < totalSteps - 1) {
                      onPlayChange(/* playing= */ false);
                      onStepChange(currentStep + 1);
                      scrollLogs(true);
                    }
                  }}
                >
                  <Icon>skip_next</Icon>
                </IconButton>
              </PlayerButtons>
              {totalSteps > 0 && currentStep > -1 && (
                <Typography variant="body1" style={{ marginRight: "8px" }}>
                  {currentStep + 1}/{totalSteps}
                </Typography>
              )}
            </PlayerControlsSection>
            <PlayerControlsSection>
              <PlaybackSlider
                min={1}
                max={totalSteps > 0 ? totalSteps : 0}
                onChangeCommitted={(_: Event | React.SyntheticEvent, value: number | number[]) => {
                  if (typeof value === "number") {
                    onStepChange(value - 1);
                    setIsChangingStep(false);
                  }
                }}
                name="Change Step"
                value={userInputStep}
                onChange={(_: Event, value: number | number[]) => {
                  if (typeof value === "number") {
                    setUserInputStep(value);

                    if (!isChangingStep) {
                      setIsChangingStep(true);
                    }
                  }
                }}
                valueLabelDisplay="auto"
                marks={interestingEvents?.map(event => ({
                  value: event.step + 1, // The slider is 1-indexed.
                  label: (
                    <Tooltip title={event.description} placement="top">
                      <MarkDot
                        $active={currentStep === event.step} // We want the dot to be active when the current step on the dot, regardless of whether it was clicked or not.
                        onMouseDown={(e: React.MouseEvent) => e.stopPropagation()}
                        onClick={(e: React.MouseEvent) => {
                          e.stopPropagation();
                          onPlayChange(false);
                          onStepChange(event.step);
                        }}
                      />
                    </Tooltip>
                  ),
                }))}
              />
              <Select
                value={speedModifier}
                onChange={(e) => onSpeedChange(e.target.value as number)}
                aria-label="Change Playback Speed"
                size="small"
              >
                {SPEED_OPTIONS.map((option) => (
                  <MenuItem key={option.value} value={option.value}>
                    {option.label}
                  </MenuItem>
                ))}
              </Select>
            </PlayerControlsSection>
            <PlayerControlsSection
              style={{ justifyContent: "flex-start", marginTop: "8px" }}
            >
              <Button
                variant="outlined"
                size="small"
                startIcon={
                  <Icon>{replayMode === "zen" ? "view_object_track" : "sensors"}</Icon>
                }
                onClick={() => {
                  if (replayMode === "zen") {
                    setReplayMode("condensed");
                  } else {
                    setReplayMode("zen");
                  }
                }}
              >
                {replayMode === "zen" ? "Log View" : "Streaming View"}
              </Button>
              {replayMode === "condensed" && (
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<Icon>expand_all</Icon>}
                  onClick={() => setExpandAll(!expandAll)}
                >
                  {expandAll ? "Collapse" : "Expand"} All
                </Button>
              )}
            </PlayerControlsSection>
          </>
        )}
      </SidebarHeader>
      {steps.length > 0 ? (
        <Virtuoso
          style={{ flex: 1 }}
          ref={virtuosoRef}
          data={visibleSteps}
          components={virtuosoComponents}
          scrollerRef={ref => {
            scrollerRef.current = ref as HTMLElement;
          }}
          atBottomThreshold={50}
          atBottomStateChange={isAtBottom => {
            isAtBottomRef.current = isAtBottom;
          }}
          // Automatically scrolls to bottom when new items are added if user hasn't scrolled up
          // Scroll to the first step in the case where one was included as a URL param
          initialTopMostItemIndex={currentStep > 0 ? currentStep : 0}
          itemContent={(index, turn) => {
            return (
              <ReasoningStep
                key={index}
                expandByDefault={replayMode === "condensed" ? expandAll : true}
                showExpandButton={replayMode === "condensed"}
                step={turn}
                stepNumber={index + 1}
                scrollLogs={scrollLogs}
                replayMode={replayMode}
                playing={playing}
                isCurrentStep={index === currentStep}
                gameName={gameName}
                onStepChange={onStepChange}
              />
            );
          }}
        />
      ) : (
        <NoStepsContainer>
          <CircularProgress />
        </NoStepsContainer>
      )}
    </LogsContainer>
  );
}
