import {
  writeFileSync
} from 'fs';

const Dimension = require('dimensions-ai');
const Match = Dimension.Match;

/**
 * This rock paper scissors game lets 2 agents play a best of n rock paper scissors 
 */
export class RockPaperScissorsDesign extends Dimension.Design {
  async initialize(match) {
    // This is the initialization step of the design, where you decide what to tell all the agents before they start
    // competing
    // You are given the match itself, and the configuration you passed in when running a match

    // let's create a state that persists through the entire match and can be updated in the update function
    let state = {
      // we will store the max rounds of rock paper scissors this game will run
      maxRounds: match.configs.bestOf ? match.configs.bestOf : 3,
      results: [], // we will also store the winner of each of those rounds by each agent's ID
      rounds: 0, // rounds passed so far
      failedAgent: null, // the id of the agent that failed to play correctly
      ties: 0
    }
    match.state = state; // this stores the state defined above into the match for re-use

    // Each match has a list of agents in match.agents, we can retrieve each of their IDs using agent.id
    // each agent ID is numbered from 0,... n-1 in a game of n agents. In this game, theres only two agents
    // agent0 and agent1
    for (let i = 0; i < 2; i++) {
      let agent = match.agents[i];

      // Here, we are sending each agent their own ID. This is a good practice for any AI competition design
      // The first arg is the message, the second arg is the id of the agent you want to send the message to
      // We use await to ensure that these ID messages are sent out first
      await match.send(`${agent.id}`, agent.id);
    }

    // we also want to send every agent the max number of rounds
    await match.sendAll(match.state.maxRounds);

    // This is initialization done!
  }
  async update(match, commands) {
    // This is the update step of the design, where all the run-time game logic goes
    // You are given the match itself, all the commands retrieved from the last round / time step from all agents, and
    // the original configuration you passed in when running a match.

    let winningAgent;

    // check which agents are still alive, if one timed out, the other wins. If both time out, it's a tie
    if (match.agents[0].isTerminated() && match.agents[1].isTerminated()) {
      match.state.terminated = {
        0: 'terminated',
        1: 'terminated'
      }
      match.state.terminatedResult = 'Tie'
      return Match.Status.FINISHED;
    } else if (match.agents[0].isTerminated()) {
      match.state.terminated = {
        0: 'terminated'
      }
      match.state.terminatedResult = match.agents[1].name
      return Match.Status.FINISHED;
    } else if (match.agents[1].isTerminated()) {
      match.state.terminated = {
        1: 'terminated'
      }
      match.state.terminatedResult = match.agents[0].name
      return Match.Status.FINISHED;
    }

    // if no commands, just return and skip update
    if (!commands.length) return;

    // each command in commands is an object with an agentID field and a command field, containing the string the agent sent
    let agent0Command = null;
    let agent1Command = null;


    // there isn't a gurantee in the command order, so we need to loop over the commands and assign them correctly
    for (let i = 0; i < commands.length; i++) {
      if (commands[i].agentID === 0) {
        agent0Command = commands[i].command;
        continue;
      } else if (commands[i].agentID === 1) {
        agent1Command = commands[i].command;
        continue;
      }

      // we also want to verify we received one command each from both players. If not, terminate the players at fault
      // We throw a MatchError through the match, indicating the agent at fault. 
      // This doesn't stop the whole process but logs the error or warning
      if (agent0Command != null && commands[i].agentID === 0) {
        // agent 0 already had a command sent, and tried to send another, so we store that agent0 is at fault 
        // and end the match
        match.throw(0, new Dimension.MatchError('attempted to send an additional command'));
        match.state.failedAgent = 0;
        return Match.Status.FINISHED;
      }
      if (agent0Command != null && commands[i].agentID === 0) {
        // agent 1 already had a command sent, and tried to send another, so we store that agent 1 is at fault 
        // and end the match
        match.throw(0, new Dimension.MatchError('attempted to send an additional command'));
        match.state.failedAgent = 1;
        return Match.Status.FINISHED;
      }
    }

    // We have that each agent will give us a command that is one of 'R', 'P', or 'S' indicating Rock, Paper, Scissors
    // if it isn't one of them, which doesn't stop the match but prints to console the error
    // we will end the match ourselves and set which agent failed
    let validChoices = new Set(['R', 'P', 'S']);
    if (!validChoices.has(agent0Command)) {
      match.throw(0, new Dimension.MatchError(agent0Command + ' is not a valid command!'));
      match.state.failedAgent = 0;
      return Match.Status.FINISHED;
    }
    if (!validChoices.has(agent1Command)) {
      match.throw(0, new Dimension.MatchError(agent1Command + ' is not a valid command!'));
      match.state.failedAgent = 1;
      return Match.Status.FINISHED;
    }

    // now we determine the winner, agent0 or agent1? or is it a tie?
    if (agent0Command === agent1Command) {
      // it's a tie if they are the same, so we set winningAgent = -1 as no one won!
      winningAgent = -1
    } else if (agent0Command === 'R') {
      if (agent1Command === 'P') {
        winningAgent = 1; // paper beats rock
      } else {
        winningAgent = 0;
      }
    } else if (agent0Command === 'P') {
      if (agent1Command === 'S') {
        winningAgent = 1; // scissors beats paper
      } else {
        winningAgent = 0;
      }
    } else if (agent0Command === 'S') {
      if (agent1Command === 'R') {
        winningAgent = 1; // rock beats scissors
      } else {
        winningAgent = 0;
      }
    }

    // update the match state
    match.state.results.push(winningAgent);
    // log the winner at the info level
    if (winningAgent != -1) {
      match.log.detail(`Round: ${match.state.rounds} - Agent ${winningAgent} won`);
    } else {
      match.log.detail(`Tie`);
    }
    // we increment the round if it wasn't a tie
    if (winningAgent != -1) {
      match.state.rounds++;
    } else {
      match.state.ties++;
    }

    // if way too many ties occured, stop the match
    if (match.state.ties === match.configs.bestOf) {

      return Match.Status.FINISHED;
    }

    // we send the status of this round to all agents
    match.sendAll(winningAgent);
    // we also want to tell the opposing agents what their opponents used last round
    match.send(agent1Command, 0);
    match.send(agent0Command, 1);

    // we now check the match status
    // if rounds reaches maxrounds, we return Match.Status.FINISHED
    if (match.state.rounds === match.state.maxRounds) {
      return Match.Status.FINISHED;
    }

    // not returning anything makes the engine assume the match is still running
  }
  async getResults(match) {
    // This is the final, result collection step occuring once the match ends
    let results = {
      scores: {
        0: 0,
        1: 0,
      },
      ties: 0,
      winner: '',
      loser: '',
      winnerID: -1,
      terminated: {

      }
    }

    // we now go over the round results and evaluate them
    match.state.results.forEach((res) => {
      if (res !== -1) {
        // if it wasn't a tie result, update the score
        results.scores[res] += 1;
      } else {
        // otherwise add to ties count
        results.ties += 1;
      }
    });

    // we store what agents were terminated by timeout and get results depending on termination
    // and stop result evaluation
    if (match.state.terminated) {
      results.terminated = match.state.terminated;
      results.winner = match.state.terminatedResult;
      if (results.winner != 'Tie') {
        if (match.state.terminated[0]) {
          results.winnerID = 1;
          results.loser = match.agents[0].name;
        } else {
          results.winnerID = 0;
          results.loser = match.agents[1].name;
        }
      } else {
        results.loser = 'Tie';
      }
      return results;
    }

    // determine the winner and store it
    if (results.scores[0] > results.scores[1]) {
      results.winner = match.agents[0].name;
      results.winnerID = 0;
      results.loser = match.agents[1].name;
    } else if (results.scores[0] < results.scores[1]) {
      results.winner = match.agents[1].name;
      results.winnerID = 1;
      results.loser = match.agents[0].name;
    } else {
      results.winner = 'Tie';
      results.loser = 'Tie';
    }

    // if there was an agent that failed, then they lose. The winner is the other agent
    if (match.state.failedAgent != null) {
      let winningAgent = (match.state.failedAgent + 1) % 2;
      results.winnerID = winningAgent;
      results.winner = match.agents[winningAgent].name;
      results.loser = match.agents[match.state.failedAgent].name;
    }

    if (match.configs.testReplays) {
      writeFileSync(`${match.id}.replay`, JSON.stringify(results.scores));
      results.replayFile = `${match.id}.replay`;
    }

    // we have to now return the results 
    return results;
  }
  static resultHandler(results) {
    let ranks = [];
    if (results.winner === 'Tie') {
      ranks = [{
        rank: 1,
        agentID: 0
      }, {
        rank: 1,
        agentID: 1
      }]
    } else {
      let loserID = (results.winnerID + 1) % 2;
      ranks = [{
        rank: 1,
        agentID: results.winnerID
      }, {
        rank: 2,
        agentID: loserID
      }]
    }
    return {
      ranks: ranks
    }
  }
}