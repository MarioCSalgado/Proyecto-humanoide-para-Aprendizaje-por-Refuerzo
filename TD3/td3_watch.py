import torch
import gymnasium as gym
import argparse
from pathlib import Path
from td3_train import Actor, make_env_fn, TD3Config

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--target_steps", type=int, required=True)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--env_id", type=str, default="Humanoid-v5")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    ckpts = sorted((run_dir / "checkpoints").glob("td3_step_*.pt"))

    def step_from_name(p):
        return int(p.stem.replace("td3_step_", ""))

    ckpt = min(ckpts, key=lambda p: abs(step_from_name(p) - args.target_steps))
    print(f"[Info] usando {ckpt.name}")

    cfg = TD3Config(env_id=args.env_id)
    print("[Watch] stage =", cfg.stage, "| ignore_env_reward =", cfg.ignore_env_reward)
    env_fn = make_env_fn(cfg)

    env = env_fn(render_mode="human")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor = Actor(obs_dim, act_dim, act_limit)
    data = torch.load(ckpt, map_location="cpu")
    actor.load_state_dict(data["actor"])
    actor.eval()

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0

        while not done:
            with torch.no_grad():
                act = actor(torch.tensor(obs).float().unsqueeze(0)).numpy()[0]
            obs, r, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            ep_ret += r

        print(f"Episode {ep+1}: return={ep_ret:.2f}")

    env.close()

if __name__ == "__main__":
    main()
