import argparse
import os
import pickle

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.policy.policy import PolicySpec
import wandb

# Import the custom environment
from environment.jaipur import env as jaipur_pettingzoo_env
from environment.jaipur_engine import JaipurEngine

from model import MaskedPPOModel

parser = argparse.ArgumentParser(description="Example argparse script")

parser.add_argument("--run_name", type=str, help="Run name for logging")
parser.add_argument(
    "--fcnet_hiddens", type=int, nargs="+", help="List of hidden layer sizes"
)
parser.add_argument("--lr", type=float, help="LR")
parser.add_argument("--minibatch_size", type=int, help="minibatch size")
parser.add_argument("--num_sgd_iter", type=int, help="num SGD iterations")
parser.add_argument("--train_batch_size", type=int, help="train_batch_size")
parser.add_argument(
    "--include_intermediate_rewards",
    action="store_true",
    help="whether to include intermediate rewards",
)
parser.add_argument(
    "--num_iterations",
    type=int,
    help="Number of iterations to train",
)
parser.add_argument(
    "--checkpoint_every", type=int, help="Checkpoint every n iterations"
)

parser.add_argument("--clip_param", type=float, help="Clip param", default=0.2)
parser.add_argument("--vf_clip_param", type=float, help="VF clip param", default=10.0)

parser.add_argument("--entropy_coeff", type=float, help="Entropy coeff", default=0.0)
parser.add_argument("--vf_loss_coeff", type=float, help="VF loss coeff", default=1.0)

args = parser.parse_args()


def train(env_name, model_name, wandb_run):
    wandb_run_id = wandb_run.id
    # Register the environment creator function
    register_env(
        env_name,
        lambda config: PettingZooEnv(
            jaipur_pettingzoo_env(
                include_intermediate_rewards=args.include_intermediate_rewards,
            ),
        ),
    )

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
                            "fcnet_hiddens": args.fcnet_hiddens,
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
            num_env_runners=2,
            num_cpus_per_env_runner=2,
            batch_mode="complete_episodes",
            rollout_fragment_length="auto",
            sample_timeout_s=600,
        )
        .training(
            lr=args.lr,
            minibatch_size=args.minibatch_size,
            train_batch_size=args.train_batch_size,
            num_sgd_iter=args.num_sgd_iter,
            vf_clip_param=args.vf_clip_param,
        )
    )
    ppo = config.build()

    out_dir = f"/home/ubuntu/cs230/checkpoints/"
    out_dir += args.run_name.replace(" ", "-")
    out_dir += f"_lr{args.lr}_mbs{args.minibatch_size}"
    out_dir += f"_sgditer{args.num_sgd_iter}_tbs{args.train_batch_size}"
    out_dir += f"_hiddens"
    for n in args.fcnet_hiddens:
        out_dir += f"{n}-"
    out_dir += f"_intermediaterewards{args.include_intermediate_rewards}/"

    for i in range(args.num_iterations):
        result = ppo.train()

        print("iteration: {}".format(i))
        print(result["info"])
        os.makedirs(f"{out_dir}/step_{i}", exist_ok=True)
        with open(f"{out_dir}/step_{i}/result_info.pkl", "wb") as f:
            pickle.dump(result["info"], f)
        with open(f"{out_dir}/step_{i}/result_env_runners.pkl", "wb") as f:
            pickle.dump(result["env_runners"], f)

        # Log some things and increment the step counter
        wandb_run.log(
            {
                "policy_loss": result["info"]["learner"]["player_policy"][
                    "learner_stats"
                ]["policy_loss"],
                "vf_loss": result["info"]["learner"]["player_policy"]["learner_stats"][
                    "vf_loss"
                ],
                "total_loss": result["info"]["learner"]["player_policy"][
                    "learner_stats"
                ]["total_loss"],
                "vf_explained_var": result["info"]["learner"]["player_policy"][
                    "learner_stats"
                ]["vf_explained_var"],
                "kl": result["info"]["learner"]["player_policy"]["learner_stats"]["kl"],
                "entropy": result["info"]["learner"]["player_policy"]["learner_stats"][
                    "entropy"
                ],
            },
        )

        if i % args.checkpoint_every == 0 and i != 0:
            out_path = f"{out_dir}/step_{i}"
            ppo.save(out_path)


if __name__ == "__main__":
    # Initialize Ray
    ray.init()

    trainer_wandb_run = wandb.init(
        project="cs230-project",
        config=vars(args),
    )

    ENV_NAME = "JaipurAECEnv"
    MODEL_NAME = "MaskedPPOModel"

    train(ENV_NAME, MODEL_NAME, trainer_wandb_run)
