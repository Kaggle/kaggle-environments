import React, { useState, useEffect } from "react";
import type { GameRendererProps } from "@kaggle-environments/core";
// We import emotion styled components since @kaggle-environments/core provides it
import styled from "@emotion/styled";

// Define the styled components for our modern sleek visualizer
const AppContainer = styled.div`
  display: flex;
  flex-direction: row;
  width: 100%;
  height: calc(100vh - 60px);
  background-color: #121212;
  color: #ffffff;
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  overflow: hidden;
`;

const BoardPane = styled.div`
  flex: 3;
  padding: 24px;
  border-right: 1px solid #333;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  align-items: center;
`;

const LogPane = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  background-color: #1a1a1a;
  box-shadow: -4px 0 15px rgba(0, 0, 0, 0.2);
`;

const TopBar = styled.div`
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
`;

const Title = styled.h1`
  font-size: 24px;
  font-weight: 700;
  margin: 0;
  background: linear-gradient(90deg, #ff4081 0%, #2196f3 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`;

const ToggleButton = styled.button<{ active: boolean }>`
  background-color: ${(props: { active: boolean }) => (props.active ? "#4caf50" : "#333")};
  color: white;
  border: none;
  border-radius: 20px;
  padding: 8px 16px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background-color: ${(props: { active: boolean }) => (props.active ? "#45a049" : "#444")};
    transform: scale(1.05);
  }
`;

const ScoreBoard = styled.div`
  display: flex;
  gap: 16px;
`;

const TeamScore = styled.div<{ team: "red" | "blue" }>`
  background-color: ${(props: { team: "red" | "blue" }) => (props.team === "red" ? "rgba(244, 67, 54, 0.2)" : "rgba(33, 150, 243, 0.2)")};
  color: ${(props: { team: "red" | "blue" }) => (props.team === "red" ? "#ff8a80" : "#82b1ff")};
  border: 1px solid ${(props: { team: "red" | "blue" }) => (props.team === "red" ? "#f44336" : "#2196f3")};
  padding: 8px 16px;
  border-radius: 8px;
  font-weight: bold;
  font-size: 16px;
`;

const Grid = styled.div`
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  grid-template-rows: repeat(5, 1fr);
  gap: 12px;
  width: min(100%, 70vh);
  height: min(100%, 70vh);
  margin: auto;
`;



const getCardBackgroundColor = (
  role: string,
  revealed: boolean,
  spymasterView: boolean
) => {
  if (revealed) {
    switch (role) {
      case "red":
        return "#f44336"; // Material Red 500
      case "blue":
        return "#2196f3"; // Material Blue 500
      case "assassin":
        return "#212121"; // Dark Grey
      case "neutral":
      default:
        return "#e0e0e0"; // Light Grey
    }
  }

  if (spymasterView) {
    switch (role) {
      case "red":
        return "rgba(244, 67, 54, 0.3)";
      case "blue":
        return "rgba(33, 150, 243, 0.3)";
      case "assassin":
        return "rgba(33, 33, 33, 0.5)";
      case "neutral":
      default:
        return "rgba(224, 224, 224, 0.3)";
    }
  }

  return "#424242"; // Default unrevealed dark grey
};

const getCardColor = (role: string, revealed: boolean) => {
  if (revealed) {
    if (role === "neutral") return "#000000";
    return "#ffffff";
  }
  return "#ffffff";
};

const CardContainer = styled.div<{ revealed: boolean, spymasterView: boolean, role: string }>`
  perspective: 1000px;
  width: 100%;
  height: 100%;
  cursor: default;

  /* Subtle pulsing border if spymaster views an unrevealed card */
  border-radius: 12px;
  border: ${(props) =>
    !props.revealed && props.spymasterView
      ? `2px dashed ${getCardBackgroundColor(props.role, true, false)}`
      : "2px solid transparent"};
`;

const CardInner = styled.div<{ revealed: boolean }>`
  position: relative;
  width: 100%;
  height: 100%;
  text-align: center;
  transition: transform 0.6s cubic-bezier(0.4, 0.0, 0.2, 1);
  transform-style: preserve-3d;
  transform: ${(props) => (props.revealed ? "rotateY(180deg) scale(0.95)" : "rotateY(0deg) scale(1)")};
`;

const CardFace = styled.div`
  position: absolute;
  width: 100%;
  height: 100%;
  backface-visibility: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 12px;
  padding: 0.5vw;
  font-size: clamp(10px, 1.5vmin, 20px);
  font-weight: 700;
  text-transform: uppercase;
  word-break: break-word;
  box-shadow: 0 4px 6px rgba(0,0,0,0.3);
`;

const CardFront = styled(CardFace) <{ role: string, spymasterView: boolean }>`
  background-color: ${(props) => getCardBackgroundColor(props.role, false, props.spymasterView)};
  color: #ffffff;
  
  /* Subtle shine effect */
  overflow: hidden;
  &::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 50%;
    height: 100%;
    background: linear-gradient(
      to right,
      rgba(255, 255, 255, 0) 0%,
      rgba(255, 255, 255, 0.1) 50%,
      rgba(255, 255, 255, 0) 100%
    );
    transform: skewX(-20deg);
    transition: all 0.5s;
  }
  &:hover::before {
    left: 200%;
  }
`;

const CardBack = styled(CardFace) <{ role: string, revealed: boolean, spymasterView: boolean }>`
  background-color: ${(props) => getCardBackgroundColor(props.role, true, props.spymasterView)};
  color: ${(props) => getCardColor(props.role, true)};
  transform: rotateY(180deg);
  box-shadow: inset 0 0 10px rgba(0,0,0,0.5);
`;

const LogHeader = styled.div`
  padding: 20px;
  border-bottom: 1px solid #333;
  font-size: 18px;
  font-weight: 600;
  background-color: #222;
`;

const LogContent = styled.div`
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  
  /* Custom scrollbar */
  &::-webkit-scrollbar {
    width: 8px;
  }
  &::-webkit-scrollbar-track {
    background: #1a1a1a;
  }
  &::-webkit-scrollbar-thumb {
    background: #444;
    border-radius: 4px;
  }
  &::-webkit-scrollbar-thumb:hover {
    background: #666;
  }
`;

interface ChatBubbleProps { team: "red" | "blue" | "system", isSpymaster: boolean }
const ChatBubble = styled.div<ChatBubbleProps>`
  max-width: 85%;
  align-self: ${(props: ChatBubbleProps) =>
    props.team === "system" ? "center" :
      props.team === "red" ? "flex-start" : "flex-end"};
  background-color: ${(props: ChatBubbleProps) => {
    if (props.team === "system") return "#333";
    if (props.team === "red") return props.isSpymaster ? "#b71c1c" : "#f44336";
    return props.isSpymaster ? "#0d47a1" : "#2196f3";
  }};
  color: white;
  padding: 12px 16px;
  border-radius: 16px;
  border-bottom-left-radius: ${(props: ChatBubbleProps) => props.team === "red" ? "4px" : "16px"};
  border-bottom-right-radius: ${(props: ChatBubbleProps) => props.team === "blue" ? "4px" : "16px"};
  font-size: 14px;
  line-height: 1.4;
  box-shadow: 0 2px 5px rgba(0,0,0,0.2);
`;

const ActorName = styled.div`
  font-size: 11px;
  opacity: 0.8;
  margin-bottom: 4px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

// Types and Helper parsing
interface GameState {
  words: string[];
  roles: string[];
  revealed: boolean[];
  current_turn: number;
  clue?: string;
  guesses_remaining?: number;
}

export const GameRenderer: React.FC<GameRendererProps> = (options: GameRendererProps) => {
  const { replay, step } = options;
  const [spymasterView, setSpymasterView] = useState(false);

  const currentStepData = replay.steps[step] as any;
  const currentEnvStep = Array.isArray(currentStepData) ? currentStepData : currentStepData.rawAgents;

  // Access global environment state via the first agent's observation
  const state: GameState = currentEnvStep[0].observation;

  if (!state || !state.words) {
    console.log("State or words missing, initializing...");
    return <div style={{ color: "white", padding: 20 }}>Initializing...</div>;
  }

  // Calculate remaining unrevealed words for each team for the scoreboard
  const redRemaining = state.roles.filter((r, i) => r === "red" && !state.revealed[i]).length;
  const blueRemaining = state.roles.filter((r, i) => r === "blue" && !state.revealed[i]).length;

  // Build the game log dynamically up to the current step
  const logEntries: React.ReactNode[] = [];

  // We iterate through all history up to current step
  for (let i = 1; i <= step; i++) {
    const pastStepData = replay.steps[i] as any;
    const pastStep = Array.isArray(pastStepData) ? pastStepData : pastStepData.rawAgents;

    // Find who moved in this step. The easiest way is to look at the actions.
    const agent0Act = pastStep[0].action;
    const agent1ActRaw = pastStep[1].action;
    const agent2Act = pastStep[2].action;
    const agent3ActRaw = pastStep[3].action;

    // Handle either dicts {"guess": 1} or directly number (1)
    const agent1Act = (typeof agent1ActRaw === "object" && agent1ActRaw !== null && 'guess' in agent1ActRaw) ? agent1ActRaw.guess : agent1ActRaw;
    const agent3Act = (typeof agent3ActRaw === "object" && agent3ActRaw !== null && 'guess' in agent3ActRaw) ? agent3ActRaw.guess : agent3ActRaw;

    // Check whose turn it wasn't. The action at replay.steps[i] was produced given the observation at replay.steps[i-1].
    const prevTurnData = i > 0 ? (replay.steps[i - 1] as any) : null;
    const prevTurn = prevTurnData ? (Array.isArray(prevTurnData) ? prevTurnData : prevTurnData.rawAgents) : null;
    const currentTurnVal = prevTurn ? prevTurn[0].observation.current_turn : -1;
    const agent0IsActive = currentTurnVal === 0;
    const agent1IsActive = currentTurnVal === 1;
    const agent2IsActive = currentTurnVal === 2;
    const agent3IsActive = currentTurnVal === 3;

    if (agent0IsActive && agent0Act !== null && typeof agent0Act === "object" && 'clue' in agent0Act) {
      logEntries.push(
        <ChatBubble key={`s${i}-0`} team="red" isSpymaster={true}>
          <ActorName>Red Spymaster</ActorName>
          Clue: <strong>{agent0Act.clue}</strong> for {agent0Act.number} words.
        </ChatBubble>
      );
    }
    if (agent2IsActive && agent2Act !== null && typeof agent2Act === "object" && 'clue' in agent2Act) {
      logEntries.push(
        <ChatBubble key={`s${i}-2`} team="blue" isSpymaster={true}>
          <ActorName>Blue Spymaster</ActorName>
          Clue: <strong>{agent2Act.clue}</strong> for {agent2Act.number} words.
        </ChatBubble>
      );
    }
    if (agent1IsActive && agent1Act !== null && typeof agent1Act === "number") {
      if (agent1Act === -1) {
        logEntries.push(
          <ChatBubble key={`s${i}-1-pass`} team="red" isSpymaster={false}>
            <ActorName>Red Guesser</ActorName>
            Ended turn.
          </ChatBubble>
        );
      } else {
        // Find the role in the preceding step's state before it was revealed
        const word = prevTurn[0].observation.words[agent1Act];
        const actualRole = prevTurn[0].observation.roles[agent1Act];
        logEntries.push(
          <ChatBubble key={`s${i}-1`} team="red" isSpymaster={false}>
            <ActorName>Red Guesser</ActorName>
            Guessed: <strong>{word}</strong> ({actualRole})
          </ChatBubble>
        );
      }
    }
    if (agent3IsActive && agent3Act !== null && typeof agent3Act === "number") {
      if (agent3Act === -1) {
        logEntries.push(
          <ChatBubble key={`s${i}-3-pass`} team="blue" isSpymaster={false}>
            <ActorName>Blue Guesser</ActorName>
            Ended turn.
          </ChatBubble>
        );
      } else {
        const word = prevTurn[0].observation.words[agent3Act];
        const actualRole = prevTurn[0].observation.roles[agent3Act];
        logEntries.push(
          <ChatBubble key={`s${i}-3`} team="blue" isSpymaster={false}>
            <ActorName>Blue Guesser</ActorName>
            Guessed: <strong>{word}</strong> ({actualRole})
          </ChatBubble>
        );
      }
    }
  }

  // Scroll to bottom effect (Hack because we don't have a direct ref easily accessible across rerenders here)
  useEffect(() => {
    const logContainer = document.getElementById("log-content");
    if (logContainer) {
      logContainer.scrollTop = logContainer.scrollHeight;
    }
  }, [step]);

  return (
    <AppContainer>
      <BoardPane>
        <TopBar>
          <Title>CODENAMES</Title>
          <ScoreBoard>
            <TeamScore team="red">Red: {redRemaining}</TeamScore>
            <TeamScore team="blue">Blue: {blueRemaining}</TeamScore>
          </ScoreBoard>
          <ToggleButton
            active={spymasterView}
            onClick={() => setSpymasterView(!spymasterView)}
          >
            {spymasterView ? "👁 Spymaster View" : "👓 Guesser View"}
          </ToggleButton>
        </TopBar>

        <Grid>
          {state.words.map((word, index) => (
            <CardContainer
              key={index}
              role={state.roles[index]}
              revealed={state.revealed[index]}
              spymasterView={spymasterView}
            >
              <CardInner revealed={state.revealed[index]}>
                <CardFront role={state.roles[index]} spymasterView={spymasterView}>
                  {word}
                </CardFront>
                <CardBack role={state.roles[index]} revealed={true} spymasterView={spymasterView}>
                  {word}
                </CardBack>
              </CardInner>
            </CardContainer>
          ))}
        </Grid>
      </BoardPane>

      <LogPane>
        <LogHeader>
          Mission Log {state.current_turn === 0 ? "🟥 Spymaster" : state.current_turn === 1 ? "🟥 Guesser" : state.current_turn === 2 ? "🟦 Spymaster" : "🟦 Guesser"}
        </LogHeader>
        <LogContent id="log-content">
          {logEntries.length === 0 ? (
            <div style={{ opacity: 0.5, textAlign: 'center', marginTop: 20 }}>
              {step === 0 ? "Game starting. Press Play or the right arrow key to advance..." : "Waiting for communications..."}
            </div>
          ) : (
            logEntries
          )}
        </LogContent>
      </LogPane>
    </AppContainer>
  );
};
