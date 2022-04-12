import { Board } from './kore/Board';
import { Observation } from './kore/Observation';

const { spawn } = require('child_process');
const fs = require('fs');

interface StepPlayerInfoRaw {
  action: Record<string, string>;
  observation?: object;
  reward: number;
  status: string;
}

interface StepPlayerInfo {
  action: Record<string, string>;
  observation: Observation;
  reward: number;
  status: string;
}

const example = 'node --require ts-node/register interpreter.ts 2 ./main.py simple [out.log] [replay.json]';

const DEFAULT_LOG_FILE_NAME = 'out.log';
const DEFAULT_RESULT_FILE_NAME = 'replay.json';

const MAX_CONCURRENT = 5;

// agent under test
const MAIN_AGENT_INDEX = 0;
const myArgs = process.argv.slice(2);

// TODO: better handling of arguments, named args.
if (myArgs.length < 3) {
  console.log('Please follow the format in the example below:');
  console.log(example);
  process.exit(1);
}

const episodes = parseInt(myArgs[0], 10);
if (!episodes) {
  console.log('Please provide number of episodes to run. Example:');
  console.log(example);
  process.exit(1);
}

const agentNames = myArgs.slice(1, 3);
// TODO: support other number of agents
if (agentNames.length !== 2) {
  console.log('Please provide exactly 2 agents. Example:');
  console.log(example);
  process.exit(1);
}

const userLogfilename = myArgs[3] || DEFAULT_LOG_FILE_NAME;

const userResultfilename = myArgs[4] || DEFAULT_RESULT_FILE_NAME;

console.log(`Running ${episodes} episodes with agents: ${agentNames.join(' ')}`);

runAgent(episodes, agentNames, () => {
  // post processing
  console.log('done');
});

async function runAgent(iterations: number, agents: string[], callback: Function = () => {}) {
  let running = 0;
  let completed = 0;
  const results = new Array(iterations);

  const agent = agents[MAIN_AGENT_INDEX];
  const cleanAgentName = agent.replace(/\.py$/g, '').replace(/\W/g, '');

  for (let index = 0; index < iterations; index++) {
    const resultFilename = `${userResultfilename}_${index}`;
    const logFilename = `${userLogfilename}_${index}`;
    const pyArguments = [
      'run',
      // TODO: support evaluate mode
      // 'evaluate',
      // '--episodes',
      // '1',
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
      console.log(`${completed}/${iterations}`);
      running--;
      const result = processResult(resultFilename);
      processSteps(index, result.configuration, result.steps);
      // ensure consistent result order
      results[index] = result;
      if (completed === iterations) {
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
