import { Button, Icon, Typography } from "@mui/material";
import { getPlayer } from "../utils";
import {
  BaseGameStep,
  getGameStepLabel,
  getGameStepDescription,
  ReplayMode,
  getTokenRenderDistribution,
} from "@kaggle-environments/core";
import useMediaQuery from "@mui/material/useMediaQuery";
import * as React from "react";
import { styled, css, keyframes } from "@mui/material/styles";
import { UserContent } from "../UserContent";

const MOBILE_CARD_HEIGHT = 100;
const MAX_STREAMING_ONLY_CARD_HEIGHT = 550;
const MAX_CARD_HEIGHT = 450;

const fadeIn = keyframes`
  0% {
    opacity: 0;
  }
  100% {
    opacity: 1;
  }
`;

const StepCard = styled('div')`
  border-radius: 16px;
  display: flex;
  flex-direction: column;
  gap: 10px;
  padding: 16px;
  animation: ${fadeIn} 0.3s ease-out forwards;
  min-width: 200px;
  width: 100%;
  overflow: hidden;
  box-sizing: border-box;
  border: 1px solid ${({ theme }) => theme.palette.divider};

  p {
    margin: 0;
  }

  ${({ theme }) => theme.breakpoints.down('lg2')} {
    padding-top: 8px;
    padding-bottom: 8px;
  }
`;

const StepMeta = styled('div')`
  align-items: center;
  display: flex;
  gap: 8px;
`;

const PlayerStep = styled('div')`
  align-items: center;
  display: flex;
  justify-content: space-between;
  width: 100%;
`;

const ReasoningContent = styled('div')<{ $replayMode: ReplayMode }>`
  border-top: 1px solid ${p => p.theme.palette.divider};
  max-height: ${p =>
    p.$replayMode === "only-stream"
      ? MAX_STREAMING_ONLY_CARD_HEIGHT
      : MAX_CARD_HEIGHT}px;
  overflow-y: scroll;
  scrollbar-width: none;
  -ms-overflow-style: none;
  overflow-anchor: none;
  scrollbar-gutter: stable;

  &::-webkit-scrollbar {
    display: none;
    width: 0;
    background: transparent;
  }

  ${({ theme }) => theme.breakpoints.down('tablet')} {
    max-height: ${MOBILE_CARD_HEIGHT}px;
  }
`;

const DescriptionAndLabelMarkdown = styled(UserContent) <{
  $useLargeFonts: boolean;
}>`
  background-color: ${p => p.theme.palette.background.paper};
  color: ${p => p.theme.palette.text.primary};
  max-width: 550px;

  ${p =>
    p.$useLargeFonts &&
    css`
      ${p.theme.breakpoints.up('tablet')} {
        margin-top: 16px;

        h1,
        h2,
        h3,
        h4,
        h5,
        h6 {
          font-size: 22px;
          line-height: 26px;
          margin: 32px 0 24px;
        }

        p,
        li,
        pre,
        code {
          font-size: 20px;
          line-height: 24px;
          margin: 0 0 24px;
        }

        li {
          list-style: none;
          margin: 16px 0;
          ol,
          ul {
            margin: 0 0 0 16px;
          }
          &:before {
            content: "• ";
          }
        }
      }
    `}
`;

const Avatar = styled('img')<{ $size: "small" | "medium" }>`
   position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: ${p => (p.$size === "small" ? 16 : 18)}px;
  height: ${p => (p.$size === "small" ? 16 : 18)}px;
  border-radius: 50%;
`

export interface ReasoningStep {
  expandByDefault: boolean;
  isCurrentStep: boolean;
  showExpandButton?: boolean;
  step: BaseGameStep;
  stepNumber: number;
  replayMode: ReplayMode;
  scrollLogs: (forceScroll?: boolean) => void;
  playing: boolean;
  gameName: string;
  onStepChange: (step: number) => void;
}

export const ReasoningStep: React.FC<ReasoningStep> = ({
  expandByDefault,
  isCurrentStep,
  showExpandButton,
  step,
  stepNumber,
  replayMode,
  scrollLogs,
  playing,
  gameName,
  onStepChange,
}) => {
  const [expanded, setExpanded] = React.useState(expandByDefault);
  const [text, setText] = React.useState("");
  const [currentTokenIndex, setCurrentTokenIndex] = React.useState(0);

  const internalScrollStickRef = React.useRef(true);
  const scrollRef = React.useRef<HTMLDivElement>(null);
  const isLargerThanTablet = useMediaQuery(theme => theme.breakpoints.up('tablet'));

  const streaming = replayMode !== "condensed";
  const useStreamingStyles = streaming && isLargerThanTablet;

  const player = getPlayer(step);
  const label = getGameStepLabel(step, gameName);
  const description = getGameStepDescription(step, gameName);
  const chunks = React.useMemo(
    () => (description.length > 0 ? description.split(" ") : []),
    [description]
  );
  const delays = getTokenRenderDistribution(chunks.length, gameName);

  const handleInternalScroll = React.useCallback(() => {
    const el = scrollRef.current;
    if (el) {
      const { scrollTop, scrollHeight, clientHeight } = el;
      // If within 10px of bottom, they are "sticky"
      const isAtBottom = scrollHeight - scrollTop - clientHeight <= 10;
      internalScrollStickRef.current = isAtBottom;
    }
  }, []);

  React.useEffect(() => {
    if (!playing && streaming && isCurrentStep) {
      return;
    }

    const shouldStreamTokens =
      currentTokenIndex < chunks.length &&
      isCurrentStep &&
      expanded &&
      streaming;

    if (shouldStreamTokens) {
      /* React 18 batches state updates together, but we want 
      each change in the text to render separately to get the
      token streaming effect. So, we use setTimeout to ensure
      each chunk is rendered individually. */
      const timeoutId = setTimeout(() => {
        setText(prev => prev + " " + chunks[currentTokenIndex]);
        setCurrentTokenIndex(prev => prev + 1);
      }, delays[currentTokenIndex]);
      return () => clearTimeout(timeoutId);
    } else if (text !== description) {
      setText(description);
    }
    return;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    chunks,
    currentTokenIndex,
    delays,
    isCurrentStep,
    description,
    expanded,
    playing,
  ]);

  React.useLayoutEffect(() => {
    const el = scrollRef.current;
    if (!el) return;

    const isInternallyScrollable = el.scrollHeight > el.clientHeight + 1;

    if (isInternallyScrollable) {
      if (internalScrollStickRef.current) {
        const distanceFromBottom =
          el.scrollHeight - el.scrollTop - el.clientHeight;

        if (distanceFromBottom > 1) {
          el.scrollTop = el.scrollHeight;
        }
      }
    } else {
      if (scrollLogs) {
        scrollLogs();
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  React.useEffect(() => {
    const el = scrollRef.current;
    if (el) {
      el.addEventListener("scroll", handleInternalScroll);
      return () => el.removeEventListener("scroll", handleInternalScroll);
    }
    return;
  }, [expanded, handleInternalScroll]);

  React.useEffect(() => {
    setExpanded(expandByDefault);
  }, [expandByDefault]);

  // If we switch to streaming mode, reset the text for the current step
  // so that we can start streaming tokens from the beginning
  React.useEffect(() => {
    if (isCurrentStep && streaming) {
      setText("");
    }
  }, [isCurrentStep, streaming]);

  if (label.length === 0 && description.length === 0) {
    return null;
  }

  if (label.length === 0 || description.length === 0) {
    return (
      <StepCard
        role="button"
        onClick={() => onStepChange(stepNumber - 1)}
      >
        <StepMeta>
          {player?.thumbnail ? (
            <Avatar
              $size={useStreamingStyles ? "medium" : "small"}
              src={player.thumbnail}
              alt=""
            />
          ) : (
            <Icon
              component="span"
              fontSize={useStreamingStyles ? "medium" : "small"}
            >
              person
            </Icon>
          )}
          {useStreamingStyles ? (
            <PlayerStep>
              {player && <Typography variant="h4" component="p">{player.name}</Typography>}
              <Typography variant="h5" component="p">{stepNumber}</Typography>
            </PlayerStep>
          ) : (
            <PlayerStep>
              <Typography variant="subtitle1">{player?.name ?? "System"}</Typography>
              <Typography variant="body1">{stepNumber}</Typography>
            </PlayerStep>
          )}
        </StepMeta>
        <DescriptionAndLabelMarkdown
          markdown={description.length > 0 ? description : label}
          youtubePreviewOnly
          $useLargeFonts={useStreamingStyles}
        />
      </StepCard>
    );
  }

  return (
    <StepCard
      role="button"
      onClick={() => onStepChange(stepNumber - 1)}
    >
      <StepMeta>
        {player?.thumbnail && (
          <Avatar
            $size={useStreamingStyles ? "medium" : "small"}
            src={player.thumbnail}
            alt=""
          />
        )}
        {useStreamingStyles ? (
          <PlayerStep>
            {player && <Typography variant="h4" component="p">{player.name}</Typography>}
            <Typography variant="h5" component="p">{stepNumber}</Typography>
          </PlayerStep>
        ) : (
          <PlayerStep>
            <Typography variant="subtitle1">{player?.name ?? "System"}</Typography>
            <Typography variant="body1">{stepNumber}</Typography>
          </PlayerStep>
        )}
      </StepMeta>
      <DescriptionAndLabelMarkdown
        markdown={label}
        youtubePreviewOnly
        $useLargeFonts={useStreamingStyles}
      />
      {showExpandButton && description && (
        <Button
          startIcon={<Icon>{expanded ? "keyboard_arrow_up" : "keyboard_arrow_down"}</Icon>}
          onClick={() => {
            setExpanded(!expanded);
            // Give a little time for the expanded card to render, then make sure all new content is visible
            if (isCurrentStep) {
              setTimeout(() => scrollLogs(true), 250);
            }
          }}
          variant="text"
        >
          {expanded ? "Hide" : "Show"} thinking
        </Button>
      )}
      {expanded && description.length > 0 && (
        <ReasoningContent ref={scrollRef} $replayMode={replayMode}>
          <DescriptionAndLabelMarkdown
            style={{ marginTop: "16px" }}
            markdown={text}
            youtubePreviewOnly
            $useLargeFonts={useStreamingStyles}
          />
        </ReasoningContent>
      )}
    </StepCard>
  );
};
