import numpy as np
import os
import argparse
import json
import pickle

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
from gymnasium import spaces

# Import the custom environment
from environment.jaipur import env as jaipur_pettingzoo_env
from environment.jaipur_engine import JaipurEngine

torch, nn = try_import_torch()


class MaskedPPOModel(TorchModelV2, nn.Module):
    """
    A custom model that handles Dict observations containing an action mask.
    It passes the observation vector to a standard FC network and then uses
    the action mask to filter the action logits. Currently kind of buggy so we ignore the mask.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        model_config = model_config or {
            "fcnet_hiddens": [256, 1024],  # Two hidden layers
            "fcnet_activation": "relu",  # Specify activation function (optional, but good practice)
        }

        print("Initializing with model config", model_config)

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, spaces.Dict)
            and "action_mask" in orig_space.spaces
            and "observation" in orig_space.spaces
        )
        self._feature_space = orig_space["observation"]

        # Create a standard fully connected network for feature extraction
        self.internal_model = TorchFC(
            self._feature_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )
        self.mask_inf = -1e10  # A large negative value to mask invalid actions
        self._last_obs = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        assert isinstance(obs, dict)
        assert "action_mask" in obs
        action_mask = obs["action_mask"]

        # Get the unmasked logits and the value function output
        logits, value_out = self.internal_model(
            {"obs": obs["observation"]}, state, seq_lens
        )
        inf_mask = torch.log(
            torch.maximum(
                action_mask,
                torch.tensor(1e-10, dtype=torch.float32, device=logits.device),
            )
        )
        masked_logits = logits + inf_mask
        return masked_logits, state

    def value_function(self):
        # The value function output is stored in the internal model
        return self.internal_model.value_function()


def setup_training(env_name, model_name):
    # Register the environment creator function
    register_env(env_name, lambda config: PettingZooEnv(jaipur_pettingzoo_env()))

    # Register the custom action masking model
    ModelCatalog.register_custom_model(model_name, MaskedPPOModel)

    # RLLib PPO Configuration
    config = (
        PPOConfig()
        .environment(env=env_name, disable_env_checking=True)
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .resources(num_gpus=1)
        .multi_agent(
            policies={
                # Define a single policy for self-play
                "player_policy": PolicySpec(
                    config={
                        "model": {
                            "custom_model": model_name,
                            "_disable_action_flattening": True,
                            "fcnet_hiddens": [256, 512, 1024],
                            "fcnet_activation": "relu",
                        }
                    }
                ),
            },
            # map all agents to use the same policy
            policy_mapping_fn=(
                lambda agent_id, episode, worker, **kwargs: "player_policy"
            ),
        )
        .env_runners(
            num_env_runners=4,
            batch_mode="complete_episodes",
            rollout_fragment_length="auto",
        ).training(lr=0.0001)
    )

    config["sgd_minibatch_size"] = 256
    config["minibatch_size"] = 256

    config["num_sgd_iter"] = 10
    config["train_batch_size"] = 1600
    config["gamma"] = 0.995
    config["lambda"] = 1.0

    print("Starting Ray training...")

    out_dir = "/home/ubuntu/cs230/checkpoints/20251113_lower_reward_scale_lr1e-4/"
    ppo = config.build()
    for i in range(500):
        result = ppo.train()
        print("iteration: {}".format(i))
        print(result["info"])
        os.makedirs(f"{out_dir}/step_{i}", exist_ok=True)
        with open(f"{out_dir}/step_{i}/result_info.pkl", "wb") as f:
            pickle.dump(result["info"], f)
        with open(f"{out_dir}/step_{i}/result_env_runners.pkl", "wb") as f:
            pickle.dump(result["env_runners"], f)
        if i % 5 == 0 and i != 0:
            out_path = f"{out_dir}/step_{i}"
            checkpoint = ppo.save(out_path)


if __name__ == "__main__":
    # Initialize Ray
    ray.init()

    ENV_NAME = "JaipurAECEnv"
    MODEL_NAME = "MaskedPPOModel"

    setup_training(ENV_NAME, MODEL_NAME)
