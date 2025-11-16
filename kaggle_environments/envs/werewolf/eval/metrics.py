from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from kaggle_environments.envs.werewolf.eval.loaders import get_games, GameResult


def _mean_std(values: List[float]) -> Tuple[float, float]:
    """Helper to calculate mean and standard deviation, handling empty lists."""
    if not values:
        return 0.0, 0.0
    return np.mean(values), np.std(values)


class AgentMetrics:


    """Stores and calculates performance metrics for a single agent."""


    def __init__(self, agent_name: str):


        self.agent_name = agent_name


        self.wins: List[int] = []


        self.wins_by_role: Dict[str, List[int]] = defaultdict(list)


        self.irp_scores: List[int] = []


        self.vss_scores: List[int] = []


        self.survival_scores: List[int] = []


        self.survival_by_role: Dict[str, List[int]] = defaultdict(list)





    def get_win_rate(self) -> Tuple[float, float]:


        return _mean_std(self.wins)





    def get_win_rate_for_role(self, role: str) -> Tuple[float, float]:


        return _mean_std(self.wins_by_role.get(role, []))





    def get_irp(self) -> Tuple[float, float]:


        return _mean_std(self.irp_scores)





    def get_vss(self) -> Tuple[float, float]:


        return _mean_std(self.vss_scores)





    def get_ksr(self) -> Tuple[float, float]:


        return _mean_std(self.survival_scores)





    def get_ksr_for_role(self, role: str) -> Tuple[float, float]:


        return _mean_std(self.survival_by_role.get(role, []))








class GameSetEvaluator:


    """Evaluates a set of game replays and calculates metrics for each agent."""


    def __init__(self, input_dir: str):


        self.games = [GameResult(g) for g in get_games(input_dir)]


        self.metrics: Dict[str, AgentMetrics] = defaultdict(lambda: AgentMetrics(agent_name=None))





    def evaluate(self):


        """Processes all games and aggregates the metrics."""


        for game in self.games:


            # --- Win Rate & Survival Metrics ---


            for player in game.players:


                agent_name = player.agent.display_name


                if self.metrics[agent_name].agent_name is None:


                    self.metrics[agent_name].agent_name = agent_name


                


                won = 1 if player.role.team == game.winner_team else 0


                survived = 1 if player.alive else 0


                


                self.metrics[agent_name].wins.append(won)


                self.metrics[agent_name].wins_by_role[player.role.name].append(won)


                self.metrics[agent_name].survival_scores.append(survived)


                self.metrics[agent_name].survival_by_role[player.role.name].append(survived)





            # --- Voting Accuracy Metrics ---


            irp_results, vss_results = game.iterate_voting_mini_game()





            # IRP Calculation


            for agent_name, score in irp_results:


                self.metrics[agent_name].irp_scores.append(score)





            # VSS Calculation


            for agent_name, score in vss_results:


                self.metrics[agent_name].vss_scores.append(score)





    def print_results(self):


        """Prints a formatted summary of the evaluation results."""


        sorted_metrics = sorted(self.metrics.values(), key=lambda m: m.agent_name)





        for stats in sorted_metrics:


            print(f"Agent: {stats.agent_name}")


            


            win_rate, win_std = stats.get_win_rate()


            print(f"  Overall Win Rate: {win_rate:.2f} ± {win_std:.2f} ({len(stats.wins)} games)")





            ksr, ksr_std = stats.get_ksr()


            print(f"  Overall Survival Rate: {ksr:.2f} ± {ksr_std:.2f}")





            print("  Role-Specific Win Rates:")


            for role in sorted(stats.wins_by_role.keys()):


                role_rate, role_std = stats.get_win_rate_for_role(role)


                game_count = len(stats.wins_by_role[role])


                print(f"    {role:<10}: {role_rate:.2f} ± {role_std:.2f} ({game_count} games)")





            print("  Role-Specific Survival Rates (KSR):")


            for role in sorted(stats.survival_by_role.keys()):


                role_ksr, role_ksr_std = stats.get_ksr_for_role(role)


                game_count = len(stats.survival_by_role[role])


                print(f"    {role:<10}: {role_ksr:.2f} ± {role_ksr_std:.2f} ({game_count} games)")





            irp, irp_std = stats.get_irp()


            vss, vss_std = stats.get_vss()


            print("  Voting Accuracy (Villager Team):")


            print(f"    IRP (Identification Precision): {irp:.2f} ± {irp_std:.2f} ({len(stats.irp_scores)} votes)")


            print(f"    VSS (Voting Success Score):     {vss:.2f} ± {vss_std:.2f} ({len(stats.vss_scores)} votes)")


            


            print("-" * 30)






if __name__ == '__main__':
    # Example usage:
    evaluator = GameSetEvaluator("kaggle_environments/envs/werewolf/eval/test/data/w_replace")
    evaluator.evaluate()
    evaluator.print_results()
