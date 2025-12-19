import os
import time
import argparse
from dataclasses import dataclass
from pathlib import Path
from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def format_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}s"
    return f"{m:d}m {s:02d}s"


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act()]
    return nn.Sequential(*layers)



# -----------------------------
# Replay Buffer
# -----------------------------
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((size, 1), dtype=np.float32)
        self.done_buf = np.zeros((size, 1), dtype=np.float32)
        self.max_size = size
        self.ptr = 0
        self.size = 0
        self.device = device

    def store(self, obs, act, rew, obs2, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = obs2
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=torch.as_tensor(self.obs_buf[idxs], device=self.device),
            act=torch.as_tensor(self.act_buf[idxs], device=self.device),
            rew=torch.as_tensor(self.rew_buf[idxs], device=self.device),
            obs2=torch.as_tensor(self.obs2_buf[idxs], device=self.device),
            done=torch.as_tensor(self.done_buf[idxs], device=self.device),
        )


# -----------------------------
# Networks
# -----------------------------
class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit):
        super().__init__()
        self.net = mlp([obs_dim, 256, 256, act_dim], activation=nn.ReLU, output_activation=nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        return self.act_limit * self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.q = mlp([obs_dim + act_dim, 256, 256, 1], activation=nn.ReLU, output_activation=nn.Identity)

    def forward(self, obs, act):
        return self.q(torch.cat([obs, act], dim=-1))


# -----------------------------
# TD3 Config
# -----------------------------
@dataclass
class TD3Config:
    env_id: str = "Humanoid-v5"
    seed: int = 42
    total_steps: int = 5_000_000

    # Exploration / collection
    start_steps: int = 25_000
    replay_size: int = 1_000_000

    # Updates (95% load)
    update_after: int = 25_000
    update_every: int = 75          # antes 50
    updates_per_block: int = 64     
    batch_size: int = 256          

    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2

    # Behavior noise (exploración)
    act_noise: float = 0.05

    # Target policy smoothing
    target_noise: float = 0.2
    noise_clip: float = 0.5

    # Optim
    pi_lr: float = 1e-3
    q_lr: float = 1e-3

    # Logging / saving
    run_name: str = "TD3_run"
    log_root: str = "runs_td3"
    save_every: int = 200_000
    eval_every: int = 100_000
    eval_episodes: int = 5
    device: str = "auto"
    render_eval: bool = False

    # Console progress
    print_every: int = 10_000
    window_episodes: int = 20
    print_episode_every: int = 200

    # Stage (balance shaping)
    stage: str = "balance"          # balance | walk
    ignore_env_reward: bool = True

    # Balance shaping parameters
    h_ref: float = 0.95
    h_max: float = 1.30
    upright_min: float = 0.75
    z_terminate: float = 0.90
    pitch_fall: float = 0.35         # antes 20 grados
    survival_bonus: float = 1.0
    fall_penalty: float = 10.0
    fall_grace_steps: int = 360 #antes 25

    w_h: float = 1.0
    w_up: float = 2.0
    w_lat: float = 0.15
    w_ang: float = 0.5
    ang_tol: float = 0.25
    w_pitch: float = 1.4
    pitch_tol: float = 0.03

    # ===== Stage 2: Walk =====
    target_vx: float = 0.35  #antes 1 0.7      # velocidad deseada hacia adelante (m/s)
    vx_tol: float = 0.25          # tolerancia
    w_vx: float = 4.5    #antes 2          # recompensa por caminar

    w_alive: float = 0.2 #antes 1   0.2       # bonus por seguir vivo
    w_ctrl: float = 0.002         # penalizar acciones grandes
    w_smooth: float = 0.01     #antes 0.5  # penalizar cambios bruscos

    w_yaw: float = 0.05           # penalizar giros
    z_terminate_walk: float = 0.50 #antes 0.80 0.7
    upright_min_walk: float = 0.18 # antes 0.65 # antes 0.55 , 035    
    progress_coef: float = 3.0

# -----------------------------
# Balance shaping wrapper
# -----------------------------
def quat_to_up_dot(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    r22 = 1 - 2 * (x * x + y * y)
    return np.clip(r22, -1.0, 1.0)


def quat_to_pitch(q):
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    num = 2.0 * (w * x + y * z)
    den = 1.0 - 2.0 * (x * x + y * y)
    return np.arctan2(num, den)


class BalanceShapingWrapper(gym.Wrapper):
    """
    Stage 1: recompensa de equilibrio (opcional, ignora reward original si ignore_env_reward=True).
    Termina si cae por z, por up_dot bajo, o si pitch excesivo.
    """
    def __init__(self, env, cfg: TD3Config):
        super().__init__(env)
        self.cfg = cfg
        self.prev_x = None

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)

        try: 
            qpos = self.unwrapped.data.qpos
            qvel = self.unwrapped.data.qvel
            z = float(qpos[2])
            q = np.array([qpos[3], qpos[4], qpos[5], qpos[6]])
            up_dot = quat_to_up_dot(q)
            pitch = quat_to_pitch(q)
            vx, vy = float(qvel[0]), float(qvel[1])
            v_lat = float((vx * vx + vy * vy) ** 0.5)
            wx, wy = float(qvel[3]), float(qvel[4])
            ang_rate = abs(wx) + abs(wy)
        except Exception:
            z, up_dot, pitch, v_lat, ang_rate = 0.0, 0.0, 0.0, 0.0, 0.0

        base_r = 0.0 if self.cfg.ignore_env_reward else float(r)

        h_bonus = self.cfg.w_h * np.clip(
            (z - self.cfg.h_ref) / max(self.cfg.h_max - self.cfg.h_ref, 1e-6), 0.0, 1.0
        )
        up_bonus = self.cfg.w_up * np.clip(up_dot, 0.0, 1.0)

        lat_pen = self.cfg.w_lat * np.clip(v_lat - 0.2, 0.0, 1.0)
        ang_pen = self.cfg.w_ang * max(ang_rate - self.cfg.ang_tol, 0.0)
        pitch_pen = self.cfg.w_pitch * max(abs(pitch) - self.cfg.pitch_tol, 0.0)

        shaped_r = base_r + h_bonus + up_bonus - lat_pen - ang_pen - pitch_pen

        alive = (z > self.cfg.h_ref) and (up_dot > 0.5)
        if alive:
            shaped_r += self.cfg.survival_bonus

        pitch_fall = abs(pitch) > self.cfg.pitch_fall
        early_fall = (z < self.cfg.z_terminate) or (up_dot < self.cfg.upright_min) or pitch_fall

        if early_fall and not terminated:
            terminated = True
            shaped_r -= self.cfg.fall_penalty

        info = dict(info)
        info.update({
            "shaping/z": z,
            "shaping/up_dot": up_dot,
            "shaping/pitch": pitch,
            "shaping/pitch_fall": pitch_fall,
            "early_terminate": early_fall,
        })
        return obs, float(shaped_r), terminated, truncated, info


  
class WalkShapingWrapper(gym.Wrapper):
    """
    Stage 2: recompensa por caminar.
    """
    def __init__(self, env, cfg: TD3Config):
        super().__init__(env)
        self.cfg = cfg
        self.prev_act = None
        self.prev_x = None
        self.bad_steps = 0
        self.step_timer = 0
        self.left_foot_geoms = []
        self.right_foot_geoms = []
        self.floor_geoms = []
        self.same_support_steps = 0
        self.max_same_support = 180    # duracion
        self.no_switch_pen = 0.5       # castigo fuerte al atasco         

        # pesos
        self.w_contact = 2.0  #antes 1.0    # reward por apoyo
        self.w_switch  = 8.0    #antes 2.5  # bonus por alternar apoyo

        self.last_support = 0  # 0 none, 1 left, 2 right


    def _init_contact_geoms(self):
        m = self.unwrapped.model

        try:
            names = list(m.geom_names)
        except Exception:
            import mujoco
            names = []
            for i in range(m.ngeom):
                n = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i)
                names.append(n if n is not None else "")

        # suelo
        floor_keys = ["floor", "ground", "plane"]
        self.floor_geoms = [i for i, n in enumerate(names) if any(k in n.lower() for k in floor_keys)]

        # pies
        left_keys  = ["left_foot", "l_foot", "foot_left", "leftankle", "l_ankle", "lefttoe", "l_toe"]
        right_keys = ["right_foot","r_foot","foot_right","rightankle","r_ankle","righttoe","r_toe"]

        self.left_foot_geoms  = [i for i, n in enumerate(names) if any(k in n.lower() for k in left_keys)]
        self.right_foot_geoms = [i for i, n in enumerate(names) if any(k in n.lower() for k in right_keys)]

        if not self.floor_geoms:
            print("[WARN] No encontré geoms de suelo por nombre. Ejemplos:", names[:50])
        if not self.left_foot_geoms or not self.right_foot_geoms:
            print("[WARN] No encontré geoms de pies por nombre. Candidatos:", [n for n in names if "foot" in n.lower() or "ankle" in n.lower() or "toe" in n.lower()])
        
        print(f"[ContactGeoms] floor={len(self.floor_geoms)} left={len(self.left_foot_geoms)} right={len(self.right_foot_geoms)}")

    def reset(self, **kwargs):
        self.prev_act = None
        self.bad_steps = 0
        self.step_timer = 0
        obs, info = self.env.reset(**kwargs)

        if not hasattr(self, "_contact_ready"):
            self._init_contact_geoms()
            self._contact_ready = True

        self.last_support = 0
        self.same_support_steps = 0

        try:
            self.prev_x = float(self.unwrapped.data.qpos[0])
        except Exception:
            self.prev_x = 0.0
        return obs, info

    def step(self, action):
        obs, r_env, terminated, truncated, info = self.env.step(action)
        r_env = 0.0 if self.cfg.ignore_env_reward else float(r_env)

        try:
            qpos = self.unwrapped.data.qpos
            qvel = self.unwrapped.data.qvel
            vz = float(qvel[2])

            data = self.unwrapped.data
            left_contact = False
            right_contact = False

            if self.floor_geoms and (self.left_foot_geoms or self.right_foot_geoms):
                for i in range(data.ncon):
                    c = data.contact[i]
                    g1, g2 = c.geom1, c.geom2

                    is_floor = (g1 in self.floor_geoms) or (g2 in self.floor_geoms)
                    if not is_floor:
                        continue

                    # pie izq
                    if (g1 in self.left_foot_geoms) or (g2 in self.left_foot_geoms):
                        left_contact = True

                    # pie der
                    if (g1 in self.right_foot_geoms) or (g2 in self.right_foot_geoms):
                        right_contact = True

            x = float(qpos[0]) 
            z = float(qpos[2])

            hop_pen = 0.0
            if z > 1.02:
                hop_pen = 0.5 * (z - 1.02)
            hop_pen = float(np.clip(hop_pen, 0.0, 1.0))

            q = np.array([qpos[3], qpos[4], qpos[5], qpos[6]])
            up_dot = quat_to_up_dot(q)
            pitch = quat_to_pitch(q)

            vx, vy = float(qvel[0]), float(qvel[1])

            wx, wy, wz = float(qvel[3]), float(qvel[4]), float(qvel[5])

            v_lat = (vx * vx + vy * vy) ** 0.5
            ang_rate = abs(wx) + abs(wy)
            yaw_rate = abs(wz)

        except Exception:
            x = z = up_dot = pitch = vx = vy = v_lat = ang_rate = yaw_rate = 0.0
            vz = 0.0
            hop_pen = 0.0
            left_contact = False
            right_contact = False
        

        # --- velocidad objetivo ---
        dv = abs(vx - self.cfg.target_vx)
        vx_r = self.cfg.w_vx * max(0.0, 1.0 - dv / max(self.cfg.vx_tol, 1e-6))
        back_pen = 2.5 * max(0.0, -vx)  

        # --- estabilidad ---
        h_bonus = self.cfg.w_h * np.clip(
            (z - self.cfg.h_ref) / max(self.cfg.h_max - self.cfg.h_ref, 1e-6), 0.0, 1.0
        )
        up_bonus = self.cfg.w_up * np.clip(up_dot, 0.0, 1.0)

        lat_pen = self.cfg.w_lat * max(v_lat - 0.3, 0.0)
        ang_pen = self.cfg.w_ang * max(ang_rate - self.cfg.ang_tol, 0.0)
        yaw_pen = self.cfg.w_yaw * max(yaw_rate - 0.8, 0.0)
        pitch_pen = self.cfg.w_pitch * max(abs(pitch) - self.cfg.pitch_tol, 0.0)

        # --- control ---
        ctrl_pen = self.cfg.w_ctrl * float(np.sum(np.square(action)))

        smooth_pen = 0.0
        if self.prev_act is not None:
            smooth_pen = self.cfg.w_smooth * float(np.mean((action - self.prev_act) ** 2))
        self.prev_act = np.array(action, copy=True)

        dx = np.clip(x - self.prev_x, -0.05, 0.10)
        self.prev_x = x
        progress_r = self.cfg.progress_coef * dx

        contact_r = 0.0
        switch_r = 0.0
        support = 0  # 0 none, 1 left, 2 right, 3 both

        if left_contact and right_contact:
            support = 3
        elif left_contact:
            support = 1
        elif right_contact:
            support = 2

        # contacto simple (primer apoyo)
        contact_simple = (support in (1, 2, 3))

        # contacto estable (apoyo ya controlado)
        stable_contact = (
            support in (1, 2, 3)
            and z > 0.85
            and up_dot > 0.5
        )       

        if contact_simple:
            contact_r = 1.5

        if stable_contact:
            contact_r = 3.0


        # --- bonus por alternar---
        current_support = 0
        if support in (1, 2):
            current_support = support

        switch_r = 0.0
        if current_support in (1, 2):
            if self.last_support in (1, 2) and current_support != self.last_support:
                switch_r = self.w_switch
                self.same_support_steps = 0
            else:
                self.same_support_steps += 1

            self.last_support = current_support
        else:
            # en double support / sin soporte: no contamos atasco
            self.same_support_steps = 0

        no_switch_penalty = 0.0
        if self.same_support_steps > self.max_same_support:
            no_switch_penalty = self.no_switch_pen



        double_support_pen = 0.0
        if support == 3:
            double_support_pen = 0.05

        reward = (
            r_env
            + vx_r
            + h_bonus + up_bonus
            + self.cfg.w_alive
            - lat_pen - ang_pen - yaw_pen - pitch_pen
            - ctrl_pen - smooth_pen
            - back_pen
            + progress_r
            - hop_pen
            + contact_r
            + switch_r
            - double_support_pen
            - no_switch_penalty
        )

        step_phase = (z > 0.95) or (abs(vz) > 0.25)   # umbrales suaves

        if step_phase and not contact_simple:
            self.step_timer += 1
        elif contact_simple:
            self.step_timer = 0
        else:
            self.step_timer = max(0, self.step_timer - 1)

        early_fall = (z < self.cfg.z_terminate_walk) or (up_dot < self.cfg.upright_min_walk)

        # durante fase de paso o contacto, NO contamos caida
        if early_fall and not (step_phase or contact_simple):
            self.bad_steps += 1
        else:
            self.bad_steps = 0
            
        if self.bad_steps >= self.cfg.fall_grace_steps and not terminated:
            terminated = True
            reward -= self.cfg.fall_penalty

        if self.step_timer > 450 and not contact_simple:
            reward -= 1.0
       
        info = dict(info)
        info.update({
            "dbg/support": support,
            "dbg/left_contact": left_contact,
            "dbg/right_contact": right_contact,
            "dbg/same_support_steps": self.same_support_steps,
            "dbg/bad_steps": self.bad_steps,
        })

        return obs, float(reward), terminated, truncated, info

# -----------------------------
# TD3 core ops
# -----------------------------
def polyak_update(net, net_targ, tau):
    with torch.no_grad():
        for p, p_targ in zip(net.parameters(), net_targ.parameters()):
            p_targ.data.mul_(1 - tau)
            p_targ.data.add_(tau * p.data)


@torch.no_grad()
def eval_policy(env_fn, actor, device, episodes=5, render=False):
    returns, lengths = [], []
    for _ in range(episodes):
        env = env_fn(render_mode="human" if render else None)
        obs, _ = env.reset()
        done = False
        ep_ret, ep_len = 0.0, 0
        while not done:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            act = actor(obs_t).cpu().numpy()[0]
            obs, r, terminated, truncated, _ = env.step(act)
            done = terminated or truncated
            ep_ret += float(r)
            ep_len += 1
            if render:
                env.render()
        env.close()
        returns.append(ep_ret)
        lengths.append(ep_len)
    return float(np.mean(returns)), float(np.mean(lengths))


def make_env_fn(cfg: TD3Config):
    def _make(render_mode=None):
        try:
            env = gym.make(cfg.env_id, render_mode=render_mode, terminate_when_unhealthy=False)
        except TypeError:
            env = gym.make(cfg.env_id, render_mode=render_mode)

        if cfg.stage == "balance":
            env = BalanceShapingWrapper(env, cfg)

        elif cfg.stage == "walk":
            env = WalkShapingWrapper(env, cfg)

        return env
    return _make

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", type=str, default="TD3_stage1_balance_v1")
    ap.add_argument("--env_id", type=str, default="Humanoid-v5")
    ap.add_argument("--total_steps", type=int, default=5_000_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--stage", type=str, default="walk", choices=["balance", "walk"])
    ap.add_argument("--render_eval", action="store_true")
    ap.add_argument("--eval_every", type=int, default=100_000)
    ap.add_argument("--save_every", type=int, default=200_000)
    ap.add_argument("--print_every", type=int, default=10_000)
    ap.add_argument("--resume", type=str, default=None, help="Ruta a checkpoint .pt para reanudar")
    args = ap.parse_args()

    # ---- threads (mas suave para laptop) ----
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    # MuJoCo on Windows
    if os.name == "nt":
        os.environ.setdefault("MUJOCO_GL", "glfw")

    cfg = TD3Config(
    run_name=args.run_name,
    env_id=args.env_id,
    total_steps=args.total_steps,
    seed=args.seed,
    stage="walk",                  # alternar segun stage antes balance
    ignore_env_reward=False,        
    render_eval=bool(args.render_eval),
    device=args.device,
    eval_every=int(args.eval_every),
    save_every=int(args.save_every),
    print_every=int(args.print_every),
    )

    device = (
        torch.device("cuda") if (cfg.device == "auto" and torch.cuda.is_available())
        else torch.device(cfg.device if cfg.device != "auto" else "cpu")
    )

    set_seed(cfg.seed)

    run_dir = Path(cfg.log_root) / cfg.run_name
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(str(run_dir / "tb")) if SummaryWriter is not None else None

    def log_scalar(tag, val, step):
        if writer is not None:
            writer.add_scalar(tag, val, step)

    env_fn = make_env_fn(cfg)
    env = env_fn(render_mode=None)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor = Actor(obs_dim, act_dim, act_limit).to(device)
    actor_targ = Actor(obs_dim, act_dim, act_limit).to(device)
    actor_targ.load_state_dict(actor.state_dict())

    critic1 = Critic(obs_dim, act_dim).to(device)
    critic2 = Critic(obs_dim, act_dim).to(device)
    critic1_targ = Critic(obs_dim, act_dim).to(device)
    critic2_targ = Critic(obs_dim, act_dim).to(device)
    critic1_targ.load_state_dict(critic1.state_dict())
    critic2_targ.load_state_dict(critic2.state_dict())

    pi_opt = optim.Adam(actor.parameters(), lr=cfg.pi_lr)
    q1_opt = optim.Adam(critic1.parameters(), lr=cfg.q_lr)
    q2_opt = optim.Adam(critic2.parameters(), lr=cfg.q_lr)

    rb = ReplayBuffer(obs_dim, act_dim, cfg.replay_size, device=device)

    # ---- Resume support ----
    start_step = 0
    if args.resume is not None:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"No existe checkpoint: {ckpt_path}")

        ckpt = torch.load(str(ckpt_path), map_location=device)

        actor.load_state_dict(ckpt["actor"])
        critic1.load_state_dict(ckpt["critic1"])
        critic2.load_state_dict(ckpt["critic2"])
        actor_targ.load_state_dict(ckpt["actor_targ"])
        critic1_targ.load_state_dict(ckpt["critic1_targ"])
        critic2_targ.load_state_dict(ckpt["critic2_targ"])

        start_step = int(ckpt.get("step", 0))
        print(f"[Resume] Cargado checkpoint {ckpt_path} (step={start_step:,})")

        # al reanudar: sin fase random ni espera de update_after
        cfg.start_steps = 0
        cfg.update_after = 0

    obs, _ = env.reset(seed=cfg.seed)

    ep_ret = 0.0
    ep_len = 0
    episode = 0

    # Progress trackers
    t0 = time.time()
    last_print_t = time.time()
    last_print_step = start_step
    ep_len_window = deque(maxlen=cfg.window_episodes)
    ep_ret_window = deque(maxlen=cfg.window_episodes)

    def print_progress(step: int, last_len: int, last_ret: float):
        nonlocal last_print_t, last_print_step
        now = time.time()
        dt = now - last_print_t
        ds = step - last_print_step

        sps_inst = (ds / dt) if dt > 0 else 0.0
        elapsed = now - t0
        
        steps_done = step - start_step
        sps_avg = (steps_done / elapsed) if elapsed > 0 else 0.0

        remaining = cfg.total_steps - step
        eta = remaining / sps_avg if sps_avg > 0 else 0.0

        pct = 100.0 * step / cfg.total_steps

        mean_len = float(np.mean(ep_len_window)) if len(ep_len_window) > 0 else 0.0
        mean_ret = float(np.mean(ep_ret_window)) if len(ep_ret_window) > 0 else 0.0

        bar_len = 24
        filled = int(bar_len * pct / 100.0)
        bar = "█" * filled + "░" * (bar_len - filled)

        msg = (
            f"[{bar}] {pct:6.2f}% | "
            f"step={step:,}/{cfg.total_steps:,} | "
            f"SPS={sps_avg:6.1f} (inst {sps_inst:6.1f}) | "
            f"elapsed {format_time(elapsed)} | ETA {format_time(eta)} | "
            f"episodes={episode} | "
            f"last_len={last_len:4d} | "
            f"avg_len({len(ep_len_window):02d})={mean_len:6.1f} | "
            f"avg_ret({len(ep_ret_window):02d})={mean_ret:7.1f}"
        )
        print("\r" + msg, end="", flush=True)

        last_print_t = now
        last_print_step = step

    def save_checkpoint(step: int):
        ckpt = ckpt_dir / f"td3_step_{step}.pt"
        torch.save({
            "actor": actor.state_dict(),
            "critic1": critic1.state_dict(),
            "critic2": critic2.state_dict(),
            "actor_targ": actor_targ.state_dict(),
            "critic1_targ": critic1_targ.state_dict(),
            "critic2_targ": critic2_targ.state_dict(),
            "cfg": cfg.__dict__,
            "step": step,
        }, ckpt)
        print(f"\n[Save] {ckpt}")

    try:
        for t in range(start_step + 1, cfg.total_steps + 1):
            # --- action selection ---
            if t < cfg.start_steps:
                act = env.action_space.sample()
            else:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
                    act = actor(obs_t).cpu().numpy()[0]
                act += cfg.act_noise * act_limit * np.random.randn(act_dim)
                act = np.clip(act, -act_limit, act_limit)

            obs2, rew, terminated, truncated, info = env.step(act)
            done = terminated or truncated

            rb.store(obs, act, rew, obs2, float(done))

            obs = obs2
            ep_ret += float(rew)
            ep_len += 1

            # --- end episode ---
            if done:
                episode += 1
                ep_len_window.append(ep_len)
                ep_ret_window.append(ep_ret)

                log_scalar("train/ep_return", ep_ret, t)
                log_scalar("train/ep_length", ep_len, t)

                if (episode % cfg.print_episode_every) == 0:
                    print(f"\n[Episode {episode}] step={t:,} | return={ep_ret:.2f} | length={ep_len}")

                ep_ret = 0.0
                ep_len = 0
                obs, _ = env.reset()

            # --- updates ---
            if t >= cfg.update_after and (t % cfg.update_every == 0) and rb.size >= cfg.batch_size:
                q1_loss = None
                q2_loss = None
                pi_loss = None

                for j in range(cfg.updates_per_block):
                    batch = rb.sample_batch(cfg.batch_size)
                    o, a, r, o2, d = batch["obs"], batch["act"], batch["rew"], batch["obs2"], batch["done"]

                    with torch.no_grad():
                        noise = torch.randn_like(a) * (cfg.target_noise * act_limit)
                        noise = torch.clamp(noise, -cfg.noise_clip * act_limit, cfg.noise_clip * act_limit)
                        a2 = actor_targ(o2)
                        a2 = torch.clamp(a2 + noise, -act_limit, act_limit)

                        q1_t = critic1_targ(o2, a2)
                        q2_t = critic2_targ(o2, a2)
                        q_targ = torch.min(q1_t, q2_t)
                        backup = r + cfg.gamma * (1 - d) * q_targ

                    q1 = critic1(o, a)
                    q2 = critic2(o, a)
                    q1_loss = ((q1 - backup) ** 2).mean()
                    q2_loss = ((q2 - backup) ** 2).mean()

                    q1_opt.zero_grad()
                    q1_loss.backward()
                    q1_opt.step()

                    q2_opt.zero_grad()
                    q2_loss.backward()
                    q2_opt.step()

                    # delayed policy update
                    if (j % cfg.policy_delay) == 0:
                        pi_opt.zero_grad()
                        pi = actor(o)
                        pi_loss = -critic1(o, pi).mean()
                        pi_loss.backward()
                        pi_opt.step()

                        polyak_update(actor, actor_targ, cfg.tau)
                        polyak_update(critic1, critic1_targ, cfg.tau)
                        polyak_update(critic2, critic2_targ, cfg.tau)

                if q1_loss is not None:
                    log_scalar("loss/q1", float(q1_loss.item()), t)
                if q2_loss is not None:
                    log_scalar("loss/q2", float(q2_loss.item()), t)
                if pi_loss is not None:
                    log_scalar("loss/pi", float(pi_loss.item()), t)

            # --- eval ---
            if cfg.eval_every > 0 and (t % cfg.eval_every == 0):
                mean_ret, mean_len = eval_policy(
                    env_fn, actor, device, episodes=cfg.eval_episodes, render=cfg.render_eval
                )
                log_scalar("eval/mean_return", mean_ret, t)
                log_scalar("eval/mean_length", mean_len, t)
                print(f"\n[Eval] step={t:,} | mean_return={mean_ret:.2f} | mean_length={mean_len:.1f}")

            # --- save ---
            if cfg.save_every > 0 and (t % cfg.save_every == 0):
                save_checkpoint(t)

            # --- progress bar ---
            if (t % cfg.print_every) == 0:
                print_progress(t, ep_len, ep_ret)

        print()  # newline after progress bar

    finally:
        env.close()
        if writer is not None:
            writer.close()

    # Final save
    final_path = run_dir / "td3_final.pt"
    torch.save({
        "actor": actor.state_dict(),
        "critic1": critic1.state_dict(),
        "critic2": critic2.state_dict(),
        "actor_targ": actor_targ.state_dict(),
        "critic1_targ": critic1_targ.state_dict(),
        "critic2_targ": critic2_targ.state_dict(),
        "cfg": cfg.__dict__,
        "step": cfg.total_steps,
    }, final_path)
    print(f"[Done] Saved final to {final_path}")


if __name__ == "__main__":
    main()
