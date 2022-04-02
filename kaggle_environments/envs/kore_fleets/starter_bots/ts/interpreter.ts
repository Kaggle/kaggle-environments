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

const example =
  'node --require ts-node/register interpreter.ts 2 ./main.py simple simple attacker [out.log] [replay.json]';

const DEFAULT_LOG_FILE_NAME = 'out.log';
const DEFAULT_RESULT_FILE_NAME = 'replay.json';

const MAX_CONCURRENT = 5;

// agent under test
const MAIN_AGENT_INDEX = 0;
const myArgs = process.argv.slice(2);

// TODO: better handling of arguments, named args.
if (myArgs.length < 5) {
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

const agentNames = myArgs.slice(1, 5);
// TODO: support other number of agents
if (agentNames.length !== 4) {
  console.log('Please provide exactly 4 agents. Example:');
  console.log(example);
  process.exit(1);
}

const userLogfilename = myArgs[5] || DEFAULT_LOG_FILE_NAME;

const userResultfilename = myArgs[6] || DEFAULT_RESULT_FILE_NAME;

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
      processSteps(index, result.steps);
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
    let seconds = 0;
    for (let i = 0; i < results.length; i++) {
      games++;
      const steps = results[i].steps;
      const lastStep = steps[steps.length - 1];
      if (lastStep.length !== 4) {
        throw new Error(`${agent} ${i} has invalid result`);
      }
      const rewards = lastStep.map((r) => r.reward);
      console.log(`game ${pad(i)} rewards: ${rewards.map((r) => r.toFixed(0)).join(', ')}`);
      const [p0, p1, p2, p3] = rewards;
      if (p0 > p1 && p0 > p2 && p0 > p3) {
        wins++;
      } else if ((p0 > p1 && p0 > p2) || (p0 > p1 && p0 > p3) || (p0 > p2 && p0 > p3)) {
        seconds++;
      }
    }
    console.log(`agent ${agent} wins: ${pad(wins)}/${pad(games)}, top-twos: ${pad(wins + seconds)}/${pad(games)}`);
  } catch (e) {
    console.error(e);
  }
}

// an example of getting the obervation, action, reward and status
function processSteps(episode: number, steps: StepPlayerInfoRaw[][]) {
  for (let i = 0; i < steps.length; i++) {
    const step = steps[i];
    // only the first agent has complete Observation, based analysis of replay.json
    const stepPlayerInfoRaw = step[MAIN_AGENT_INDEX];
    const stepPlayerInfo: StepPlayerInfo = {
      action: stepPlayerInfoRaw.action,
      // no other way?
      observation: new Observation(JSON.stringify(stepPlayerInfoRaw.observation)),
      reward: stepPlayerInfoRaw.reward,
      status: stepPlayerInfoRaw.status,
    };
    const { action, observation, reward, status } = stepPlayerInfo;
    if (i % 100 === 0) {
      console.log(
        `[epi ${episode}][step ${pad2(i)}] kore 0:${observation.kore[0].toFixed(0)} action 0: ${
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
