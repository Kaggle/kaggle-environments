import React, { useState, useEffect } from 'react';
import type { GameRendererProps } from '@kaggle-environments/core';
import styled from '@emotion/styled';

const AppContainer = styled.div`
  display: flex;
  flex-direction: row;
  width: 100%;
  height: calc(100vh - 60px);
  background-color: #121212;
  color: #ffffff;
  font-family:
    'Inter',
    system-ui,
    -apple-system,
    sans-serif;
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
  background: linear-gradient(90deg, #20beff 0%, #e5cf4a 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
`;

const ToggleButton = styled.button<{ active: boolean }>`
  background-color: ${(props: { active: boolean }) => (props.active ? '#4caf50' : '#333')};
  color: white;
  border: none;
  border-radius: 20px;
  padding: 8px 16px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background-color: ${(props: { active: boolean }) => (props.active ? '#45a049' : '#444')};
    transform: scale(1.05);
  }
`;

const ScoreBoard = styled.div`
  display: flex;
  gap: 16px;
`;

const TeamScore = styled.div<{ team: 'blue' | 'yellow' }>`
  background-color: ${(props: { team: 'blue' | 'yellow' }) =>
    props.team === 'blue' ? 'rgba(32, 190, 255, 0.2)' : 'rgba(229, 207, 74, 0.2)'};
  color: ${(props: { team: 'blue' | 'yellow' }) => (props.team === 'blue' ? '#20BEFF' : '#E5CF4A')};
  border: 1px solid ${(props: { team: 'blue' | 'yellow' }) => (props.team === 'blue' ? '#20BEFF' : '#E5CF4A')};
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

const getCardBackgroundColor = (role: string, revealed: boolean, cluemasterView: boolean) => {
  if (revealed) {
    switch (role) {
      case 'blue':
        return '#20BEFF';
      case 'yellow':
        return '#E5CF4A';
      case 'trap':
        return '#212121';
      case 'neutral':
      default:
        return '#e0e0e0';
    }
  }

  if (cluemasterView) {
    switch (role) {
      case 'blue':
        return 'rgba(32, 190, 255, 0.3)';
      case 'yellow':
        return 'rgba(229, 207, 74, 0.3)';
      case 'trap':
        return 'rgba(33, 33, 33, 0.5)';
      case 'neutral':
      default:
        return 'rgba(224, 224, 224, 0.3)';
    }
  }

  return '#424242';
};

const getCardColor = (role: string, revealed: boolean) => {
  if (revealed) {
    if (role === 'neutral' || role === 'yellow') return '#000000';
    return '#ffffff';
  }
  return '#ffffff';
};

const CardContainer = styled.div<{ revealed: boolean; cluemasterView: boolean; role: string }>`
  perspective: 1000px;
  width: 100%;
  height: 100%;
  cursor: default;
  border-radius: 12px;
  border: ${(props) =>
    !props.revealed && props.cluemasterView
      ? `2px dashed ${getCardBackgroundColor(props.role, true, false)}`
      : '2px solid transparent'};
`;

const CardInner = styled.div<{ revealed: boolean }>`
  position: relative;
  width: 100%;
  height: 100%;
  text-align: center;
  transition: transform 0.6s cubic-bezier(0.4, 0, 0.2, 1);
  transform-style: preserve-3d;
  transform: ${(props) => (props.revealed ? 'rotateY(180deg) scale(0.95)' : 'rotateY(0deg) scale(1)')};
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
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
`;

const CardFront = styled(CardFace)<{ role: string; cluemasterView: boolean }>`
  background-color: ${(props) => getCardBackgroundColor(props.role, false, props.cluemasterView)};
  color: #ffffff;

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

const CardBack = styled(CardFace)<{ role: string; revealed: boolean; cluemasterView: boolean }>`
  background-color: ${(props) => getCardBackgroundColor(props.role, true, props.cluemasterView)};
  color: ${(props) => getCardColor(props.role, true)};
  transform: rotateY(180deg);
  box-shadow: inset 0 0 10px rgba(0, 0, 0, 0.5);
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

interface ChatBubbleProps {
  team: 'blue' | 'yellow' | 'system';
  isCluemaster: boolean;
}

const ChatBubble = styled.div<ChatBubbleProps>`
  max-width: 85%;
  align-self: ${(props: ChatBubbleProps) =>
    props.team === 'system' ? 'center' : props.team === 'blue' ? 'flex-start' : 'flex-end'};
  background-color: ${(props: ChatBubbleProps) => {
    if (props.team === 'system') return '#333';
    if (props.team === 'blue') return props.isCluemaster ? '#0088cc' : '#20BEFF';
    return props.isCluemaster ? '#b29c1f' : '#E5CF4A';
  }};
  color: ${(props: ChatBubbleProps) => (props.team === 'yellow' ? '#121212' : '#ffffff')};
  padding: 12px 16px;
  border-radius: 16px;
  border-bottom-left-radius: ${(props: ChatBubbleProps) => (props.team === 'blue' ? '4px' : '16px')};
  border-bottom-right-radius: ${(props: ChatBubbleProps) => (props.team === 'yellow' ? '4px' : '16px')};
  font-size: 14px;
  line-height: 1.4;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
`;

const ActorName = styled.div`
  font-size: 11px;
  opacity: 0.8;
  margin-bottom: 4px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
`;

interface GameState {
  words: string[];
  roles: string[];
  revealed: boolean[];
  current_turn: number;
  clue?: string;
  guesses_remaining?: number;
  reward?: number;
}

export const GameRenderer: React.FC<GameRendererProps> = (options: GameRendererProps) => {
  const { replay, step } = options;
  const [cluemasterView, setCluemasterView] = useState(false);

  const currentStepData = replay.steps[step] as any;
  const currentEnvStep = Array.isArray(currentStepData) ? currentStepData : currentStepData.rawAgents;

  const state: GameState = currentEnvStep[0].observation;

  if (!state || !state.words) {
    return <div style={{ color: 'white', padding: 20 }}>Initializing...</div>;
  }

  const blueRemaining = state.roles.filter((r, i) => r === 'blue' && !state.revealed[i]).length;
  const yellowRemaining = state.roles.filter((r, i) => r === 'yellow' && !state.revealed[i]).length;

  const logEntries: React.ReactNode[] = [];

  for (let i = 1; i <= step; i++) {
    const pastStepData = replay.steps[i] as any;
    const pastStep = Array.isArray(pastStepData) ? pastStepData : pastStepData.rawAgents;

    const agent0Act = pastStep[0].action;
    const agent1ActRaw = pastStep[1].action;
    const agent2Act = pastStep[2].action;
    const agent3ActRaw = pastStep[3].action;

    const agent1Act =
      typeof agent1ActRaw === 'object' && agent1ActRaw !== null && 'guess' in agent1ActRaw
        ? agent1ActRaw.guess
        : agent1ActRaw;
    const agent3Act =
      typeof agent3ActRaw === 'object' && agent3ActRaw !== null && 'guess' in agent3ActRaw
        ? agent3ActRaw.guess
        : agent3ActRaw;

    const prevTurnData = i > 0 ? (replay.steps[i - 1] as any) : null;
    const prevTurn = prevTurnData ? (Array.isArray(prevTurnData) ? prevTurnData : prevTurnData.rawAgents) : null;
    const currentTurnVal = prevTurn ? prevTurn[0].observation.current_turn : -1;
    const agent0IsActive = currentTurnVal === 0;
    const agent1IsActive = currentTurnVal === 1;
    const agent2IsActive = currentTurnVal === 2;
    const agent3IsActive = currentTurnVal === 3;

    if (agent0IsActive && agent0Act !== null && typeof agent0Act === 'object' && 'clue' in agent0Act) {
      logEntries.push(
        <ChatBubble key={`s${i}-0`} team="blue" isCluemaster={true}>
          <ActorName>Blue Cluemaster</ActorName>
          Clue: <strong>{agent0Act.clue}</strong> for {agent0Act.number} words.
        </ChatBubble>
      );
    }
    if (agent2IsActive && agent2Act !== null && typeof agent2Act === 'object' && 'clue' in agent2Act) {
      logEntries.push(
        <ChatBubble key={`s${i}-2`} team="yellow" isCluemaster={true}>
          <ActorName>Yellow Cluemaster</ActorName>
          Clue: <strong>{agent2Act.clue}</strong> for {agent2Act.number} words.
        </ChatBubble>
      );
    }
    if (agent1IsActive && agent1Act !== null && typeof agent1Act === 'number') {
      if (agent1Act === -1) {
        logEntries.push(
          <ChatBubble key={`s${i}-1-pass`} team="blue" isCluemaster={false}>
            <ActorName>Blue Guesser</ActorName>
            Ended turn.
          </ChatBubble>
        );
      } else {
        const word = prevTurn[0].observation.words[agent1Act];
        const actualRole = prevTurn[0].observation.roles[agent1Act];
        logEntries.push(
          <ChatBubble key={`s${i}-1`} team="blue" isCluemaster={false}>
            <ActorName>Blue Guesser</ActorName>
            Guessed: <strong>{word}</strong> ({actualRole})
          </ChatBubble>
        );
      }
    }
    if (agent3IsActive && agent3Act !== null && typeof agent3Act === 'number') {
      if (agent3Act === -1) {
        logEntries.push(
          <ChatBubble key={`s${i}-3-pass`} team="yellow" isCluemaster={false}>
            <ActorName>Yellow Guesser</ActorName>
            Ended turn.
          </ChatBubble>
        );
      } else {
        const word = prevTurn[0].observation.words[agent3Act];
        const actualRole = prevTurn[0].observation.roles[agent3Act];
        logEntries.push(
          <ChatBubble key={`s${i}-3`} team="yellow" isCluemaster={false}>
            <ActorName>Yellow Guesser</ActorName>
            Guessed: <strong>{word}</strong> ({actualRole})
          </ChatBubble>
        );
      }
    }
  }

  useEffect(() => {
    const logContainer = document.getElementById('log-content');
    if (logContainer) {
      logContainer.scrollTop = logContainer.scrollHeight;
    }
  }, [step]);

  const isGameOver = currentEnvStep[0].status === 'DONE';
  let winnerText: React.ReactNode = null;

  if (isGameOver) {
    const trapIndex = state.roles.findIndex((role) => role === 'trap');
    const trapRevealed = trapIndex !== -1 && state.revealed[trapIndex];

    if (trapRevealed) {
      if (state.reward === 1 || currentEnvStep[0].reward === 1) {
        winnerText = (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', lineHeight: '1.2' }}>
            <span style={{ color: '#20BEFF' }}>🔷 BLUE WINS! 🔷</span>
            <span style={{ fontSize: '14px', color: '#aaaaaa', marginTop: '4px', fontWeight: 'normal' }}>
              (Yellow picked the Trap)
            </span>
          </div>
        );
      } else if (currentEnvStep[2].reward === 1) {
        winnerText = (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', lineHeight: '1.2' }}>
            <span style={{ color: '#E5CF4A' }}>⬜ YELLOW WINS! ⬜</span>
            <span style={{ fontSize: '14px', color: '#aaaaaa', marginTop: '4px', fontWeight: 'normal' }}>
              (Blue picked the Trap)
            </span>
          </div>
        );
      }
    } else {
      if (state.reward === 1 || currentEnvStep[0].reward === 1) {
        winnerText = <span style={{ color: '#20BEFF' }}>🔷 BLUE WINS! 🔷</span>;
      } else if (currentEnvStep[2].reward === 1) {
        winnerText = <span style={{ color: '#E5CF4A' }}>⬜ YELLOW WINS! ⬜</span>;
      } else if (blueRemaining === 0) {
        winnerText = <span style={{ color: '#20BEFF' }}>🔷 BLUE WINS! 🔷</span>;
      } else if (yellowRemaining === 0) {
        winnerText = <span style={{ color: '#E5CF4A' }}>⬜ YELLOW WINS! ⬜</span>;
      } else {
        winnerText = 'GAME OVER';
      }
    }
  }

  const formatMissionLog = () => {
    if (isGameOver) return 'Results';
    const playerTurns = ['🔷 Blue Cluemaster', '🔷 Blue Guesser', '⬜ Yellow Cluemaster', '⬜ Yellow Guesser'];
    return `Mission Log - ${playerTurns[state.current_turn] || ''}`;
  };

  return (
    <AppContainer>
      <BoardPane>
        <TopBar>
          <Title>{isGameOver ? winnerText : 'WORD ASSOCIATION'}</Title>
          <ScoreBoard>
            <TeamScore team="blue">Blue: {blueRemaining}</TeamScore>
            <TeamScore team="yellow">Yellow: {yellowRemaining}</TeamScore>
          </ScoreBoard>
          <ToggleButton active={cluemasterView} onClick={() => setCluemasterView(!cluemasterView)}>
            {cluemasterView ? '👁 Cluemaster View' : '👓 Guesser View'}
          </ToggleButton>
        </TopBar>

        <Grid>
          {state.words.map((word, index) => (
            <CardContainer
              key={index}
              role={state.roles[index]}
              revealed={state.revealed[index]}
              cluemasterView={cluemasterView || isGameOver}
            >
              <CardInner revealed={state.revealed[index] || isGameOver}>
                <CardFront role={state.roles[index]} cluemasterView={cluemasterView || isGameOver}>
                  {word}
                </CardFront>
                <CardBack role={state.roles[index]} revealed={true} cluemasterView={cluemasterView || isGameOver}>
                  {word}
                </CardBack>
              </CardInner>
            </CardContainer>
          ))}
        </Grid>
      </BoardPane>

      <LogPane>
        <LogHeader>{formatMissionLog()}</LogHeader>
        <LogContent id="log-content">
          {logEntries.length === 0 ? (
            <div style={{ opacity: 0.5, textAlign: 'center', marginTop: 20 }}>
              {step === 0
                ? 'Game starting. Press Play or the right arrow key to advance...'
                : 'Waiting for communications...'}
            </div>
          ) : (
            logEntries
          )}
          {isGameOver && (
            <div
              style={{
                textAlign: 'center',
                marginTop: 20,
                padding: 20,
                background: '#333',
                borderRadius: 8,
                fontWeight: 'bold',
              }}
            >
              {winnerText}
            </div>
          )}
        </LogContent>
      </LogPane>
    </AppContainer>
  );
};
