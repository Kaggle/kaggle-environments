const { spawn } = require('child_process');
const fs = require('fs');
const resolve = require('path').resolve;

const example = 'node --require ts-node/register interpreter.ts 2 ./main.py simple simple attacker out.log replay.json';

const MAX_CONCURRENT = 5;

// agent under test
const MAIN_AGENT_INDEX = 0;
const myArgs = process.argv.slice(2);

// TODO: better handling of arguments, named args.
if (myArgs.length !== 7) {
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

const userLogfilename = myArgs[5];
if (!userLogfilename) {
  console.log('Please provide a filename for the log. Example:');
  console.log(example);
  process.exit(1);
}

const userResultfilename = myArgs[6];
if (!userResultfilename) {
  console.log('Please provide a filename for the result. Example:');
  console.log(example);
  process.exit(1);
}

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

// will be used in the future for evaluate mode
function compileEvaluateResults(agent: string, results: number[][]) {
  try {
    let games = 0;
    let wins = 0;
    let seconds = 0;
    for (let i = 0; i < results.length; i++) {
      games++;
      const [p0, p1, p2, p3] = results[i];
      if (p0 > p1 && p0 > p2 && p0 > p3) {
        wins++;
      } else if ((p0 > p1 && p0 > p2) || (p0 > p1 && p0 > p3) || (p0 > p2 && p0 > p3)) {
        seconds++;
      }
    }
    console.log(`${agent} win: ${pad(wins)}/${pad(games)} top two: ${pad(wins + seconds)}/${pad(games)}`);
  } catch (e) {
    console.error(e);
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
