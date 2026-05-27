# Save trained actor from notebook after training:
#   torch.save(actor.state_dict(), "ppo_actor.pt")
#
# Run:
#   /home/elliot/.virtualenvs/ml/bin/python3 ppo_render.py --checkpoint ppo_actor.pt
#   /home/elliot/.virtualenvs/ml/bin/python3 ppo_render.py --smoke-test

import argparse
import sys

import matplotlib.pyplot as plt
import torch
import torch.distributions as d
import torch.nn as nn
from tensordict.nn import TensorDictModule
from torchrl.envs import Compose, DoubleToFloat, ObservationNorm, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import NormalParamExtractor, ProbabilisticActor

# ObservationNorm stats from training (init_stats on InvertedDoublePendulum-v4).
OBS_NORM_LOC = torch.tensor(
    [
        4.1208e-03,
        3.9874e-02,
        -3.7545e-02,
        -1.2142e01,
        -6.8242e00,
        1.0713e-02,
        1.8446e-02,
        -2.4675e-02,
        -0.0000e00,
        -0.0000e00,
        -0.0000e00,
    ]
)
OBS_NORM_SCALE = torch.tensor(
    [
        6.0560e00,
        3.1774e00,
        2.4446e00,
        1.2836e01,
        7.5586e00,
        6.0529e-01,
        2.9962e-01,
        2.5451e-01,
        1.0000e06,
        1.0000e06,
        1.0000e06,
    ]
)

HIDDEN_FEATURES = 256
DEFAULT_MAX_STEPS = 1000


def make_env(device: torch.device, render_mode: str = "rgb_array"):
    base_env = GymEnv(
        "InvertedDoublePendulum-v4", device=device, render_mode=render_mode
    )
    return TransformedEnv(
        base_env,
        Compose(
            ObservationNorm(
                in_keys=["observation"],
                loc=OBS_NORM_LOC.to(device),
                scale=OBS_NORM_SCALE.to(device),
            ),
            DoubleToFloat(),
        ),
    )


def build_actor(env, device: torch.device) -> ProbabilisticActor:
    """Mirror PPO-GAE.ipynb / PPO-TD0.ipynb actor architecture."""
    action_dim = env.action_spec.shape[-1]
    policy_net_raw = nn.Sequential(
        nn.LazyLinear(HIDDEN_FEATURES),
        nn.Tanh(),
        nn.LazyLinear(HIDDEN_FEATURES),
        nn.Tanh(),
        nn.LazyLinear(2 * action_dim),
        NormalParamExtractor(),
    ).to(device)

    policy_module = TensorDictModule(
        policy_net_raw,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        module=policy_module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=d.Normal,
        return_log_prob=True,
        spec=env.action_spec,
    ).to(device)
    return actor


def _materialize_lazy(actor: ProbabilisticActor, env) -> None:
    with torch.no_grad():
        td = env.reset()
        actor(td)


def load_actor(env, device: torch.device, checkpoint_path: str | None) -> ProbabilisticActor:
    actor = build_actor(env, device)
    if checkpoint_path is None:
        print("No --checkpoint: using randomly initialized policy.", file=sys.stderr)
        _materialize_lazy(actor, env)
        return actor

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    _materialize_lazy(actor, env)
    actor.load_state_dict(state)
    print(f"Loaded checkpoint: {checkpoint_path}")
    return actor


def rollout_episode(env, actor, max_steps: int, show: bool) -> float:
    actor.eval()
    total_reward = 0.0
    frames = []

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
        td = env.reset()
        for _ in range(max_steps):
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            td = actor(td)
            td = env.step(td)
            next_td = td["next"]
            total_reward += next_td["reward"].sum().item()

            if next_td["done"].any().item():
                break
            if next_td.get("terminated", default=False).any().item():
                break
            if next_td.get("truncated", default=False).any().item():
                break
            td = next_td

    if show and frames:
        for frame in frames:
            plt.imshow(frame)
            plt.axis("off")
            plt.pause(0.02)
            plt.clf()
        plt.close()

    return total_reward


def smoke_test(device: torch.device) -> None:
    env = make_env(device, render_mode="rgb_array")
    actor = load_actor(env, device, checkpoint_path=None)
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.inference_mode():
        td = env.reset()
        td = actor(td)
        td = env.step(td)
    print("smoke_test ok: reset -> policy -> one step")
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Render PPO policy on InvertedDoublePendulum-v4")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to actor.state_dict() .pt file")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--no-display", action="store_true", help="Skip matplotlib frame display")
    parser.add_argument("--smoke-test", action="store_true", help="reset + one step without checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if args.smoke_test:
        smoke_test(device)
        return

    env = make_env(device, render_mode="rgb_array")
    actor = load_actor(env, device, args.checkpoint)

    for ep in range(args.episodes):
        ret = rollout_episode(env, actor, args.max_steps, show=not args.no_display)
        print(f"episode {ep + 1}/{args.episodes} return={ret:.4f}")

    env.close()


if __name__ == "__main__":
    main()
