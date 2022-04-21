import { Board } from './kore/Board';
import { tick as DoNothingTick } from './DoNothingBot';
import { tick as MinerTick } from './MinerBot';
import { tick as BotTick } from './Bot';

const { spawn } = require('child_process');
const fs = require('fs');

interface StepPlayerInfoRaw {
  action: Record<string, string>;
  observation?: object;
  reward: number;
  status: string;
}

interface Agent {
  name: string;
  status: 'ACTIVE' | 'DONE';
  reward: number;
  tickFunc: (board:Board) => void; 
}

const tickFuncMapping = {
  'miner': MinerTick,
  'do_nothing': DoNothingTick,
  'bot': BotTick,
}

const MODE_STEP = 'step' as const;
const MODE_RUN = 'run' as const;
type MODE = typeof MODE_STEP | typeof MODE_RUN;

const example = 'node --require ts-node/register interpreter.ts step 2 ./main.py miner [out.log] [replay.json]';

const DEFAULT_LOG_FILE_NAME = 'out.log';
const DEFAULT_RESULT_FILE_NAME = 'replay.json';

const MAX_CONCURRENT = 5;

// agent under test
const MAIN_AGENT_INDEX = 0;
// opponent agent
const OPPONENT_AGENT_INDEX = 1;
const myArgs = process.argv.slice(2);

// TODO: better handling of arguments, named args.
if (myArgs.length < 3) {
  console.log('Please follow the format in the example below:');
  console.log(example);
  process.exit(1);
}

const mode = myArgs[0];
if(mode !== MODE_STEP && mode !== MODE_RUN) {
  console.log('Mode must be either step or run. Example:');
  console.log(example);
  process.exit(1);
}

const episodes = parseInt(myArgs[1], 10);
if (!episodes) {
  console.log('Please provide number of episodes to run. Example:');
  console.log(example);
  process.exit(1);
}

const agentNames = myArgs.slice(2, 4);
// TODO: support other number of agents
if (agentNames.length !== 2) {
  console.log('Please provide exactly 2 agents. Example:');
  console.log(example);
  process.exit(1);
}

// validate agent names for step mode
if(mode === MODE_STEP) {
  for (let i = 0; i < agentNames.length; i++) {
    const agentName = agentNames[i];
    if(!tickFuncMapping[agentName]) {
      console.log(`Agent ${agentName} tick function mapping does not exit. Define the mapping in interpreter.js -> tickFuncMapping.`);
      console.log(example);
      process.exit(1);
    }
  }
}

const userLogfilename = myArgs[4] || DEFAULT_LOG_FILE_NAME;

const userResultfilename = myArgs[5] || DEFAULT_RESULT_FILE_NAME;

console.log(`Running ${episodes} episodes with agents: ${agentNames.join(' ')}`);

if(mode === MODE_RUN) {
  runAgent(episodes, agentNames, () => {
    // post processing
    console.log('run done');
  });
} else {
  stepAgent(episodes, agentNames, () => {
    // post processing
    console.log('step done');
  });
}

function initEnv(agents: Agent[]): Promise<Board> {
  // get init observation by playing a game between attacker agent and do_nothing agent
  const resultFilename = `${userResultfilename}_init`;

  const pyArguments = [
    'run',
    '--environment',
    'kore_fleets',
    '--agents',
    'attacker',
    'do_nothing',
    '--out',
    resultFilename,
  ];

  return new Promise<Board>((resolve, reject) => {
    const kaggle = spawn('kaggle-environments', pyArguments, { cwd: __dirname });

    kaggle.stderr.on('data', (data) => {
      console.error(`stderr: ${data}`);
      reject(data);
    });
  
    kaggle.on('close', (code) => {
      console.log(`init complete`);
      const result = processResult(resultFilename);
      const { configuration, steps } = result;
      const observation = steps[0][0].observation;
      const board = Board.fromRaw(JSON.stringify(observation), JSON.stringify(configuration));
      resolve(board);
    });
  });
}

// mimic the interpreter function in kore_fleets.py
function boardTick(board:Board, agents: Agent[]) {
  const players = board.players;

  // Remove players with invalid status or insufficient potential.
  for (let i = 0; i < players.length; i++) {
    const agent = agents[i];
    const player = players[i];
    const playerKore = player.kore;
    const shipyards = player.shipyards;
    const fleets = player.fleets;
    const canSpawn = shipyards.length > 0 && playerKore >= board.configuration.spawnCost;

    if(agent.status === 'ACTIVE' && shipyards.length === 0 && fleets.length === 0) {
      agent.status = 'DONE';
      agent.reward = board.step - board.configuration.episodeSteps - 1;
    }
    if(agent.status === 'ACTIVE' && playerKore === 0 && fleets.length === 0 && !canSpawn) {
      agent.status = 'DONE';
      agent.reward = board.step - board.configuration.episodeSteps - 1;
    }
    if(agent.status !== 'ACTIVE' && agent.status !== 'DONE') {
      // TODO: handle this
    }
  }

  // Check if done (< 2 players and num_agents > 1)
  const activeAgents = agents.filter(agent => agent.status === 'ACTIVE').length;
  if(agents.length > 1 && activeAgents < 2) {
    for (let i = 0; i < agents.length; i++) {
      const agent = agents[i];
      if(agent.status === 'ACTIVE') {
        agent.status = 'DONE';
      }
    }
  }

  // Update Rewards
  for (let i = 0; i < agents.length; i++) {
    const agent = agents[i];
    if(agent.status === 'ACTIVE') {
      agent.reward = players[i].kore;
    } else if(agent.status !== 'DONE') {
      agent.reward = 0;
    }
  }

  let end = true;
  if(board.step >= board.configuration.episodeSteps - 1) {
    return true;
  }
  for (let i = 0; i < agents.length; i++) {
    const agent = agents[i];
    if(agent.status === 'ACTIVE') {
      end = false;
    }
  }
  return end;
}

async function stepAgent(episodes: number, agentsNames: string[], callback: Function = () => {}) {
  let completed = 0;
  let wins = 0;

  for (let index = 0; index < episodes; index++) {
    let agentNamesMutable = agentsNames.slice();
    let episodeMainAgentIndex = MAIN_AGENT_INDEX;
    let episodeOpponentAgentIndex = OPPONENT_AGENT_INDEX;
    // randomize starting position
    if (Math.random() < 0.5) {
      const temp = agentNamesMutable[0];
      agentNamesMutable[0] = agentNamesMutable[1];
      agentNamesMutable[1] = temp;
      episodeMainAgentIndex = OPPONENT_AGENT_INDEX;
      episodeOpponentAgentIndex = MAIN_AGENT_INDEX;
    }

    const agents: Agent[] = agentNamesMutable.map((name) => {
      return {
        name: name,
        status: 'ACTIVE',
        reward: 0,
        tickFunc: tickFuncMapping[name],
      };
    });

    let gameBoard = await initEnv(agents);
    while(!boardTick(gameBoard, agents)) {
      // console.log(gameBoard.step);
      for (let i = 0; i < agents.length; i++) {
        const agent = agents[i];
        // rotate the board to the agent's perspective
        // and assign agent action to game board
        gameBoard.currentPlayerId = i;
        agent.tickFunc(gameBoard);

        gameBoard.currentPlayer.shipyards.forEach(shipyard => {
          // console.log(gameBoard.currentPlayerId, shipyard.position.toString(), shipyard.nextAction);
        })
      }
      gameBoard.currentPlayerId = episodeMainAgentIndex;
      if (gameBoard.step % 100 === 0) {
        console.log(
          `[epi ${index}][step ${pad2(gameBoard.step)}] current player kore:${gameBoard.currentPlayer.kore} action 0: ${
            gameBoard.currentPlayer.shipyards.length ? gameBoard.currentPlayer.shipyards[0].nextAction : 'none'
          } reward: ${agents[episodeMainAgentIndex].reward.toFixed(0)} status: ${agents[episodeMainAgentIndex].status}`
        );
      }
      gameBoard = gameBoard.next();
    }
    console.log(
      `[epi ${index}][step ${pad2(gameBoard.step)}] current player kore:${gameBoard.currentPlayer.kore} action 0: ${
        gameBoard.currentPlayer.shipyards.length ? gameBoard.currentPlayer.shipyards[0].nextAction : 'none'
      } reward: ${agents[episodeMainAgentIndex].reward.toFixed(0)} status: ${agents[episodeMainAgentIndex].status}`
    );
    console.log(agents[episodeMainAgentIndex].name, agents[episodeMainAgentIndex].reward);
    console.log(agents[episodeOpponentAgentIndex].name, agents[episodeOpponentAgentIndex].reward);
    if(agents[episodeMainAgentIndex].reward > agents[episodeOpponentAgentIndex].reward) {
      wins++;
    }

    completed++;
    console.log(`${completed}/${episodes}`);
    if (completed === episodes) {
      const mainAgentName = agentsNames[MAIN_AGENT_INDEX];
      console.log(`agent ${mainAgentName} wins: ${pad(wins)}/${pad(episodes)}`);
      callback();
    }
  }
}

async function runAgent(episodes: number, agents: string[], callback: Function = () => {}) {
  let running = 0;
  let completed = 0;
  const results = new Array(episodes);

  const agent = agents[MAIN_AGENT_INDEX];
  const cleanAgentName = agent.replace(/\.py$/g, '').replace(/\W/g, '');

  for (let index = 0; index < episodes; index++) {
    const resultFilename = `${userResultfilename}_${index}`;
    const logFilename = `${userLogfilename}_${index}`;
    const pyArguments = [
      'run',
      '--environment',
      'kore_fleets',
      '--agents',
      ...agents,
      '--log',
      logFilename,
      '--out',
      resultFilename,
    ];

    while (running >= MAX_CONCURRENT) {
      await sleep(1000);
    }

    const kaggle = spawn('kaggle-environments', pyArguments, { cwd: __dirname });
    running++;

    kaggle.stdout.on('data', (data) => {
      console.log(`stdout: ${data}`);
    });

    kaggle.stderr.on('data', (data) => {
      console.error(`stderr: ${data}`);
    });

    kaggle.on('close', (code) => {
      completed++;
      console.log(`${completed}/${episodes}`);
      running--;
      const result = processResult(resultFilename);
      processSteps(index, result.configuration, result.steps);
      // ensure consistent result order
      results[index] = result;
      if (completed === episodes) {
        compileRunResults(cleanAgentName, results);
        callback();
      }
    });
  }
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function compileRunResults(agent: string, results: any[]) {
  try {
    let games = 0;
    let wins = 0;
    for (let i = 0; i < results.length; i++) {
      games++;
      const steps = results[i].steps;
      const lastStep = steps[steps.length - 1];
      if (lastStep.length !== 2) {
        throw new Error(`${agent} ${i} has invalid result`);
      }
      const rewards = lastStep.map((r) => r.reward);
      console.log(`game ${pad(i)} rewards: ${rewards.map((r) => r.toFixed(0)).join(', ')}`);
      const [p0, p1] = rewards;
      if (p0 > p1) {
        wins++;
      }
    }
    console.log(`agent ${agent} wins: ${pad(wins)}/${pad(games)}`);
  } catch (e) {
    console.error(e);
  }
}

// an example of getting the obervation, action, reward and status
function processSteps(episode: number, configuration: object, steps: StepPlayerInfoRaw[][]) {
  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    // only the first agent has complete Observation, based analysis of replay.json
    const stepPlayerInfoRaw = step[MAIN_AGENT_INDEX];
    const obervation = stepPlayerInfoRaw.observation;
    // TODO: optimize this conversion thing
    const board = Board.fromRaw(JSON.stringify(obervation), JSON.stringify(configuration));

    const { action, reward, status } = stepPlayerInfoRaw;

    let lastTurn = false;
    if (status === 'DONE') {
      lastTurn = true;
    }

    if (i % 100 === 0 || lastTurn) {
      console.log(
        `[epi ${episode}][step ${pad2(i)}] current player kore:${board.currentPlayer.kore} action 0: ${
          action[Object.keys(action)[0]]
        } reward: ${reward.toFixed(0)} status: ${status}`
      );
    }
  }
}

function processResult(filename: string) {
  try {
    const results = JSON.parse(fs.readFileSync(filename, 'utf8'));
    return results;
  } catch (e) {
    console.error(e);
  }
}

function pad(number) {
  if (number < 10) {
    return '0' + number;
  }
  return number;
}

function pad2(number) {
  if (number < 10) {
    return '00' + number;
  } else if (number < 100) {
    return '0' + number;
  }
  return number;
}
