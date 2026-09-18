import logging
import os
from typing import Optional

import numpy as np
import upath
from stable_baselines3.common.callbacks import BaseCallback

from pyxis_portfolio_challenge.agents import PyxieAgent
from pyxis_portfolio_challenge.game.asset_generators import JSONAssetGenerator
from pyxis_portfolio_challenge.game.constants import LEVELS
from pyxis_portfolio_challenge.game.game_state import GameState

logger = logging.getLogger(__name__)


class VecNormSyncCallback(BaseCallback):
    """
    Custom callback to synchronize vector normalization statistics.

    This copies over the observation normalization statistics from the training
    environment to the evaluation environment.
    """

    def __init__(self, train_env, eval_env):
        """Initialize the VecNormSyncCallback."""
        super().__init__()
        self.train_env = train_env
        self.eval_env = eval_env

    def _on_step(self) -> bool:
        self.eval_env.obs_rms = self.train_env.obs_rms
        return True


class PlayLevelsCallback(BaseCallback):
    """Custom callback to play levels during eval."""

    def __init__(
        self,
        experiment_dir: str,
        max_num_assets: int,
        assets_dir: upath.UPath,
        num_levels: int = 3,
    ):
        """Initialize the PlayLevelsCallback."""
        super().__init__()
        self.experiment_path = experiment_dir
        self.max_num_assets = max_num_assets
        self.assets_dir = assets_dir
        self.levels_info = LEVELS[:num_levels]

        self.temp_model_path = os.path.join(experiment_dir, "temp_model_for_levels.zip")
        self.temp_vecnorm_path = os.path.join(
            experiment_dir, "temp_vecnorm_for_levels.pkl"
        )

        self.best_path = os.path.join(experiment_dir, "levels_best_model")

        self.log_path = os.path.join(experiment_dir, "logs", "level_evaluations.npz")

        self.best_av_enpv = [-np.inf for _ in range(num_levels)]
        self.level_metrics = {
            "av_enpv": [[] for _ in range(num_levels)],
            "final_enpv": [[] for _ in range(num_levels)],
            "total_investments": [[] for _ in range(num_levels)],
        }

    def _on_step(self) -> bool:
        self.model.save(self.temp_model_path)
        self.training_env.save(self.temp_vecnorm_path)

        agent = PyxieAgent(
            algorithm=self.model.__class__,
            model_path=upath.UPath(self.temp_model_path),
            vecnorm_path=upath.UPath(self.temp_vecnorm_path),
        )

        for level_idx, level_info in enumerate(self.levels_info):
            game_state = GameState.initialise_new_game(
                asset_generator_cls=JSONAssetGenerator,
                num_assets=level_info["num_assets"],
                max_num_assets=self.max_num_assets,
                cash=level_info["starting_cash"],
                horizon=level_info["horizon"],
                seed=level_info["global_seed"],
                **{
                    "assets_dir": self.assets_dir,
                    "indication_spread": 4.0,
                    "indication_drift_speed": 1.0,
                    "trial_cost_multiplier": 1.0,
                },
            )
            result = agent.playthrough(
                game_state, level_idx, agent_name="Pyxie", verbose=True
            )
            game_metrics = result["game_metrics"]
            investments_per_step = result["investments_per_step"]

            av_enpv = game_metrics.av_enpv
            final_enpv = game_metrics.final_enpv
            total_investments = np.sum(investments_per_step)

            self.logger.record(f"levels_eval/av_enpv_{level_idx}", av_enpv)
            self.logger.record(f"levels_eval/final_enpv_{level_idx}", final_enpv)
            self.logger.record(
                f"levels_eval/total_investments_{level_idx}", total_investments
            )

            self.level_metrics["av_enpv"][level_idx].append(av_enpv)
            self.level_metrics["final_enpv"][level_idx].append(final_enpv)
            self.level_metrics["total_investments"][level_idx].append(total_investments)

            if av_enpv > self.best_av_enpv[level_idx]:
                print(f"New best average eNPV for level {level_idx}!")
                self.best_av_enpv[level_idx] = av_enpv
                best_level_path = os.path.join(self.best_path, str(level_idx))
                self.model.save(os.path.join(best_level_path, "best_model.zip"))
                self.training_env.save(
                    os.path.join(best_level_path, "vecnormalize.pkl")
                )
                with open(os.path.join(best_level_path, "num_timesteps.txt"), "w") as f:
                    f.write(str(self.num_timesteps))

        np.savez(self.log_path, **self.level_metrics)

        return True


# From https://github.com/DLR-RM/rl-baselines3-zoo/blob/f6b3ff70b13d2c2156b3e0faf9994c107c649c82/utils/callbacks.py#L55 # noqa: E501
# No unit tests needed
class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps.

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved,
    as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``,
    if None (default) only one file will be kept.
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: Optional[str] = None,
        verbose: int = 0,
    ):
        """Initialise SaveVecNormalizeCallback."""
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(
                    self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl"
                )
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True


class BenchmarkEvalCallback(BaseCallback):
    """
    Evaluation callback that uses the same evaluate code as benchmark.

    This creates a temporary PyxieAgent and runs the evaluate function
    with fixed seeds, matching exactly what the benchmark does.
    """

    def __init__(
        self,
        eval_freq: int = 50000,
        n_eval_episodes: int = 100,
        log_path: str = None,
        temp_model_dir: str = None,
        verbose: int = 1,
    ):
        """Initialize BenchmarkEvalCallback."""
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.log_path = log_path
        self.temp_model_dir = temp_model_dir

        self.evaluations_timesteps = []
        self.evaluations_results = []
        self.last_eval_timestep = 0

    def _init_callback(self) -> None:
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if self.temp_model_dir is not None:
            os.makedirs(self.temp_model_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Check if we should evaluate based on timesteps, not n_calls
        if self.num_timesteps - self.last_eval_timestep < self.eval_freq:
            return True
        self.last_eval_timestep = self.num_timesteps

        # Save current model and vecnorm temporarily
        temp_model_path = os.path.join(self.temp_model_dir, "temp_model.zip")
        temp_vecnorm_path = os.path.join(self.temp_model_dir, "temp_vecnorm.pkl")

        self.model.save(temp_model_path)
        if self.model.get_vec_normalize_env() is not None:
            self.model.get_vec_normalize_env().save(temp_vecnorm_path)

        # Create PyxieAgent
        agent = PyxieAgent(
            algorithm=self.model.__class__,
            model_path=upath.UPath(temp_model_path),
            vecnorm_path=upath.UPath(temp_vecnorm_path),
            deterministic=True,
        )

        # Create a simple eval env (no metrics, no verbose logging)
        from pyxis_portfolio_challenge.config import config, instantiate_from_config
        from pyxis_portfolio_challenge.environment.training_gym import (
            InvestmentGameEnv,
        )

        env = InvestmentGameEnv(
            equilibrium_num_assets=config.equilibrium_num_assets,
            max_num_assets=config.max_num_assets,
            asset_arrival_sensitivity_below=config.asset_arrival_sensitivity_below,
            asset_arrival_sensitivity_above=config.asset_arrival_sensitivity_above,
            starting_cash=config.starting_cash,
            horizon=config.horizon,
            reward_fn=instantiate_from_config(config.reward_fn),
            assets_dir=config.evaluation_data_dir,
            reinvestment_percentage=config.reinvestment_percentage,
            shuffle_order=False,
            flatten_obs=True,
            mask_first_order_assets=True,
            mask_negative_enpv_assets=False,
            uncertain_ptrs_config=config.uncertain_ptrs,
            investment_levels_config=config.investment_levels,
            interim_trial_observations_config=config.interim_trial_observations,
            distributional_ptrs_config=config.distributional_ptrs,
            ta_experience_config=config.ta_experience,
            rd_capacity_config=config.rd_capacity,
            drop_action_config=config.drop_action,
            marketing_config=config.marketing,
            clinical_sites_config=config.clinical_sites,
            ptrs_readings_config=config.ptrs_readings,
            approval_phase_config=config.approval_phase,
            metrics=[],
            initial_game_state=None,
        )
        agent.set_env(env)

        # Run episodes with fixed seeds (same as benchmark)
        episode_rewards = []
        eval_initial_seed = 891024889

        for ep_idx in range(self.n_eval_episodes):
            seed = eval_initial_seed + ep_idx
            obs, _ = env.reset(seed=seed)
            episode_reward = 0.0
            done = False

            while not done:
                action = agent(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                done = terminated or truncated

            episode_rewards.append(episode_reward)

        env.close()

        if episode_rewards:
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)

            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards)

            if self.log_path is not None:
                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                )

            if self.verbose > 0:
                print(
                    f"Benchmark eval @ {self.num_timesteps} steps: "
                    f"{mean_reward / 1e9:.2f}B ± {std_reward / 1e9:.2f}B"
                )

            self.logger.record("benchmark_eval/mean_reward", mean_reward)
            self.logger.record("benchmark_eval/std_reward", std_reward)

        return True


class KnapsackBaselineCallback(BaseCallback):
    """
    Evaluates the PPO agent against a knapsack baseline on the same eval episodes.

    Both agents run on identical episodes (same seeds). Logs PPO reward,
    knapsack reward, and PPO win rate to TensorBoard.
    """

    def __init__(
        self,
        eval_freq: int = 50000,
        n_eval_episodes: int = 20,
        eval_seed: int = 891024889,
        verbose: int = 1,
    ):
        """Initialize."""
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed = eval_seed
        self._env = None
        self._knapsack = None
        self._last_eval_timestep = 0

    def _lazy_init(self):
        """Create eval env and knapsack agent on first use."""
        if self._env is not None:
            return

        from pyxis_portfolio_challenge.agents.knapsack import KnapsackAgent
        from pyxis_portfolio_challenge.config import config, instantiate_from_config
        from pyxis_portfolio_challenge.environment.training_gym import (
            InvestmentGameEnv,
        )

        cfg = config
        self._env = InvestmentGameEnv(
            equilibrium_num_assets=cfg.equilibrium_num_assets,
            max_num_assets=cfg.max_num_assets,
            asset_arrival_sensitivity_below=cfg.asset_arrival_sensitivity_below,
            asset_arrival_sensitivity_above=cfg.asset_arrival_sensitivity_above,
            starting_cash=cfg.starting_cash,
            horizon=cfg.horizon,
            reward_fn=instantiate_from_config(cfg.reward_fn),
            assets_dir=cfg.evaluation_data_dir,
            reinvestment_percentage=cfg.reinvestment_percentage,
            shuffle_order=False,
            flatten_obs=True,
            mask_first_order_assets=cfg.mask_first_order_assets,
            mask_negative_enpv_assets=cfg.mask_negative_enpv_assets,
            uncertain_ptrs_config=cfg.uncertain_ptrs,
            investment_levels_config=cfg.investment_levels,
            interim_trial_observations_config=cfg.interim_trial_observations,
            distributional_ptrs_config=cfg.distributional_ptrs,
            ta_experience_config=cfg.ta_experience,
            rd_capacity_config=cfg.rd_capacity,
            drop_action_config=cfg.drop_action,
            marketing_config=cfg.marketing,
            clinical_sites_config=cfg.clinical_sites,
            ptrs_readings_config=cfg.ptrs_readings,
            approval_phase_config=cfg.approval_phase,
            metrics=[],
            initial_game_state=None,
        )
        self._knapsack = KnapsackAgent()
        self._knapsack.set_env(self._env)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_timestep < self.eval_freq:
            return True
        self._last_eval_timestep = self.num_timesteps

        self._lazy_init()

        vec_env = self.model.get_env()
        obs_rms_mean = vec_env.obs_rms.mean.copy()
        obs_rms_var = vec_env.obs_rms.var.copy()

        ppo_rewards = []
        knapsack_rewards = []
        wins = 0

        for ep in range(self.n_eval_episodes):
            seed = self.eval_seed + ep + self.num_timesteps

            # --- Run knapsack ---
            obs, _ = self._env.reset(seed=seed)
            total_knapsack = 0.0
            done = False
            while not done:
                action = self._knapsack(obs)
                obs, reward, terminated, truncated, _ = self._env.step(action)
                total_knapsack += reward
                done = terminated or truncated
            knapsack_rewards.append(total_knapsack)

            # --- Run PPO on the same seed ---
            obs, _ = self._env.reset(seed=seed)
            total_ppo = 0.0
            done = False
            while not done:
                norm_obs = (obs - obs_rms_mean) / np.sqrt(obs_rms_var + 1e-8)
                norm_obs = np.clip(norm_obs, -10.0, 10.0)

                action_masks = self._env.action_masks()
                action, _ = self.model.predict(
                    norm_obs,
                    deterministic=True,
                    action_masks=action_masks,
                )
                obs, reward, terminated, truncated, _ = self._env.step(action)
                total_ppo += reward
                done = terminated or truncated
            ppo_rewards.append(total_ppo)

            if total_ppo > total_knapsack:
                wins += 1

        mean_ppo = np.mean(ppo_rewards)
        mean_knapsack = np.mean(knapsack_rewards)
        win_rate = wins / self.n_eval_episodes

        self.logger.record("eval_vs_knapsack/ppo_reward", mean_ppo)
        self.logger.record("eval_vs_knapsack/knapsack_reward", mean_knapsack)
        self.logger.record("eval_vs_knapsack/ppo_win_rate", win_rate)

        logger.info(
            f"Eval vs Knapsack at {self.num_timesteps}: "
            f"PPO={mean_ppo:,.0f} vs Knapsack={mean_knapsack:,.0f} "
            f"| Win rate: {win_rate:.0%}"
        )

        return True
