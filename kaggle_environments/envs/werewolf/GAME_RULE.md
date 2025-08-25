# Werewolf: Game Rules

Welcome to Werewolf, a game of social deduction, team collaboration, deception, and survival. Players are secretly assigned roles on one of two teams: the Village or the Werewolves.

## Roles

Each player is assigned one of the following roles:

### Village Team

The goal of the Village team is to exile all the werewolves.

* **Villager:** You have no special abilities other than your power of observation and your voice. Use the discussion phase to identify suspicious behavior and vote to exile suspected werewolves.
* **Seer:** Each night, you may choose one player to investigate. You will learn if that player is a Werewolf or not. Your goal is to share this information strategically to help the village without revealing your identity too soon.
* **Doctor:** Each night, you may choose one player to protect. The player you protect cannot be eliminated by the werewolves that night.

### Werewolf Team

* **Werewolf:** Your goal is to eliminate villagers until the number of werewolves equals the number of remaining village members. Each night, you and your fellow werewolves will secretly agree on one player to eliminate.

## Game Phases

The game alternates between a Night phase and a Day phase.

### Night Phase 🐺

During the night, all players close their eyes. The moderator will ask players with special roles to wake up and perform their actions in this order:

1.  **Doctor:** Chooses one player to protect.
2.  **Seer:** Chooses one player to investigate their alignment.
3.  **Werewolves:** Silently vote on one player to eliminate.

### Day Phase ☀️

1.  **Announcement:** The moderator announces which player, if any, was eliminated during the night. That player is removed from the game and may not speak or participate further.
2.  **Discussion:** The surviving players discuss who they think the werewolves are.
3.  **Exile Vote:** Players vote on who to exile from the village. The player who receives the most votes is exiled, removed from the game, and their role is revealed.

The game continues with another Night phase until a winning condition is met.

## Customizable Rules

Before the game begins, the following options must be decided.

### 1. Doctor's Self-Save

* **Option A (Self-Save Allowed):** The Doctor is allowed to choose themselves as the target of their protection.
* **Option B (No Self-Save):** The Doctor must choose another player to protect.

### 2. Discussion Protocol

* **Option A (Parallel Discussion):** All players may speak simultaneously for a number of rounds.
* **Option B (Round Robin):** Each player speak one after another following a predefined order for a number of rounds.

### 3. Voting Protocol
The night wolf target election and the day exile election are both configurable. All voting protocols follow a random 
tie breaking mechanism, where a random draw is used when there multiple candidates with the same votes.

* **Option A (Sequential Voting):** Voters cast their votes one after another, where each voter has visibility to all earlier vote.
* **Option B (Parallel Voting):** All voters cast their votes simultaneously.

## Winning the Game

A team wins as soon as their winning condition is met.

* **The Village Team wins** when all werewolves have been successfully exiled.
* **The Werewolf Team wins** when the number of werewolves is equal to the number of remaining Village team members.

### Rewards

All members of the winning team will receive **1 reward**. This includes players who were eliminated before the end of the game.

### Tie Game (Forfeit)

If any back-end inference fails during the game, the match will immediately end. The game will be declared a **tie**, and no players will receive a reward.