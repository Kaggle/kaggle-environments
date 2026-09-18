import importlib.util
import inspect
import logging
from functools import partial
from typing import Callable

import click
import upath
import yaml
from pydantic import BaseModel

from pyxis_portfolio_challenge.config import config, instantiate_from_config
from pyxis_portfolio_challenge.environment.training_gym import InvestmentGameEnv
from pyxis_portfolio_challenge.logging_utils import setup_logging


def takes_kwargs(func: Callable) -> bool:
    """
    Return True if func accepts **kwargs.

    Args:
        func: Function to inspect.

    Returns:
        bool: True if func accepts **kwargs, False otherwise.

    """
    sig = inspect.signature(func)
    return any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in sig.parameters.values()
    )


def load_agent(agent: str, **kwargs) -> Callable:
    """
    Retrieve an investment agent by name or from a custom file.

    Args:
        agent (str): Name of pre-registered agent (see AGENTS.keys()) or a path
         to custom agent file.
        kwargs: Additional keyword arguments to pass to the agent.

    Returns:
        Callable: The agent's main function.

    """
    # Load custom agent from file
    agent_path = upath.UPath(agent)
    if not agent_path.exists():
        raise FileNotFoundError(f"Custom agent file not found: {agent_path}")

    spec = importlib.util.spec_from_file_location("custom_agent", str(agent_path))
    custom_agent_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(custom_agent_module)

    if not hasattr(custom_agent_module, "main"):
        raise AttributeError("Custom agent file must define a 'main' function.")

    if not takes_kwargs(custom_agent_module.main):
        if kwargs:
            raise AttributeError("Custom agent 'main' function does not accept kwargs.")
        return custom_agent_module.main
    return partial(custom_agent_module.main, **kwargs)


def resolve_cfg_args(cfg: BaseModel, **kwargs) -> BaseModel:
    """
    Resolve configuration arguments, prioritizing command-line inputs.

    Args:
        cfg: (BaseModel) Configuration object.
        kwargs: Command-line arguments to override configuration.

    Returns:
        Updated configuration object.

    """
    cfg_dict = cfg.model_dump()
    for key, value in kwargs.items():
        if value is not None:
            cfg_dict[key] = value
    cfg = cfg.__class__(**cfg_dict)
    return cfg


@click.command()
@click.argument("agent", type=str, required=True)
@click.option("--cfg_file", type=str, default=None)
@click.option("--seed", type=int, default=None)
@click.option(
    "--agent-kwargs",
    type=str,
    default=None,
    help="Additional kwargs for agent as a YAML string.",
)
@click.option("--log-level", type=str, default="INFO", help="Logging level.")
def main(agent, cfg_file, seed, agent_kwargs, log_level) -> None:
    """
    Run a playthrough of the investment game with the specified agent.

    Args:
        agent (str): Path to custom agent file.

            Note: Custom agent file must define main function that takes an observation, and optional kwargs as input, and returns an action. kwargs can be passed through the `agent_kwargs` optional argument.

        cfg_file (str): Path to configuration YAML file. Note: If not provided, default config will be used.
        seed (int): Random seed for reproducibility. Note will override any seed set in the config file.
        agent_kwargs (str): Additional keyword arguments for the agent as a YAML string.
        log_level (str): Logging level. One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".

    """  # noqa: E501
    # set logging level
    setup_logging(level=log_level)
    logger = logging.getLogger(__name__)
    logger.info(
        f"Loading configuration from "
        f"{cfg_file if cfg_file is not None else 'default config'}"
    )

    # load configuration
    if cfg_file is not None:
        cfg = config.from_yaml(cfg_file)
    else:
        cfg = config

    # resolve any command-line args that override config
    cfg = resolve_cfg_args(cfg)
    logger.debug(
        f"\n-------CONFIG-------\n {yaml.dump(cfg.model_dump())}-------CONFIG-------"
    )
    # kwargs that pass into agent function
    agent_kwargs_dict = {}
    if agent_kwargs is not None:
        agent_kwargs_dict = yaml.safe_load(agent_kwargs)

    logger.debug(f"Agent kwargs: {agent_kwargs_dict}")

    # load agent
    loaded_agent = load_agent(agent, **agent_kwargs_dict)

    # prepare environment and run agent
    env = InvestmentGameEnv(
        equilibrium_num_assets=cfg.equilibrium_num_assets,
        max_num_assets=cfg.max_num_assets,
        asset_arrival_sensitivity_below=cfg.asset_arrival_sensitivity_below,
        asset_arrival_sensitivity_above=cfg.asset_arrival_sensitivity_above,
        starting_cash=cfg.starting_cash,
        horizon=cfg.horizon,
        reward_fn=instantiate_from_config(cfg.reward_fn),
        assets_dir=cfg.training_data_dir,
        reinvestment_percentage=cfg.reinvestment_percentage,
        shuffle_order=cfg.shuffle_order,
        flatten_obs=cfg.flatten_obs,
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

    logger.info("Starting agent playthrough...")
    terminated = False
    obs, info = env.reset(seed=seed)
    reward = 0.0
    while not terminated:
        action = loaded_agent(obs)
        obs, reward, terminated, _, info = env.step(action)

    logger.info(f"Game over. Final reward: {reward}")


if __name__ == "__main__":
    main()
