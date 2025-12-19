import math
import os
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from gymnasium import Wrapper
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
from dataclasses import asdict 

#Utils
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, gain=std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

#WRAPPERS PARA AMBIENTES MODIFICADOS
class WindForceWrapper(Wrapper):
    def __init__(self, env, wind_strength=2.0, wind_direction=90, 
                 variable_wind=False, change_frequency=500):
        super().__init__(env)
        self.wind_strength = wind_strength
        self.base_direction = wind_direction
        self.variable_wind = variable_wind
        self.change_frequency = change_frequency
        self.step_count = 0
        self.current_wind = self._compute_wind_force(wind_direction)
        
    def _compute_wind_force(self, angle_deg):
        """Convierte ángulo a vector de fuerza en el plano XY"""
        angle_rad = np.deg2rad(angle_deg)
        force_x = self.wind_strength * np.cos(angle_rad)
        force_y = self.wind_strength * np.sin(angle_rad)
        return np.array([force_x, force_y, 0.0])  # Z=0 (horizontal)
    
    def reset(self, **kwargs):
        self.step_count = 0
        self.current_wind = self._compute_wind_force(self.base_direction)
        return self.env.reset(**kwargs)
    
    def step(self, action):
        #Actualizar dirección del viento si es variable
        if self.variable_wind and self.step_count % self.change_frequency == 0:
            #Viento aleatorio ±45° de la dirección base
            angle = self.base_direction + np.random.uniform(-45, 45)
            self.current_wind = self._compute_wind_force(angle)
        
        #Aplicar fuerza antes del step
        if hasattr(self.env.unwrapped, 'data'):
            #Aplicar fuerza al torso (body 1 en Humanoid-v5)
            torso_id = 1
            self.env.unwrapped.data.xfrc_applied[torso_id, :3] = self.current_wind
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.step_count += 1
        
        return obs, reward, terminated, truncated, info


class SlipperyFloorWrapper(Wrapper):
    def __init__(self, env, friction_coef=0.3):
        super().__init__(env)
        self.friction_coef = max(0.01, min(1.0, friction_coef))
        self.original_friction = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        #Modificar fricción del suelo (geom 0 = floor)
        if hasattr(self.env.unwrapped, 'model'):
            model = self.env.unwrapped.model
            #Guardar fricción original la primera vez
            if self.original_friction is None:
                self.original_friction = model.geom_friction[0].copy()
            
            #Aplicar nueva fricción
            #geom_friction tiene 3 componentes: [sliding, torsional, rolling]
            model.geom_friction[0] = [
                self.friction_coef,     
                self.friction_coef * 0.5, 
                self.friction_coef * 0.1 
            ]
        
        return obs, info
    
    def close(self):
        # Restaurar fricción original al cerrar
        if self.original_friction is not None and hasattr(self.env.unwrapped, 'model'):
            self.env.unwrapped.model.geom_friction[0] = self.original_friction
        super().close()


class SlopedFloorWrapper(Wrapper):
    def __init__(self, env, slope_angle=10.0, slope_axis='y'):
        super().__init__(env)
        self.slope_angle = slope_angle
        self.slope_axis = slope_axis.lower()
        self.original_quat = None
        self.gravity_modified = False
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        if hasattr(self.env.unwrapped, 'model'):
            model = self.env.unwrapped.model
            
            # MÉTODO 1: Rotar el suelo (geom del floor)
            # Guardar orientación original del suelo
            if self.original_quat is None:
                # El suelo es el body 0 en Humanoid-v5
                self.original_quat = model.body_quat[0].copy()
            
            # Crear rotación para la pendiente
            angle_rad = np.deg2rad(self.slope_angle)
            
            if self.slope_axis == 'x':
                # Pendiente lateral (roll)
                quat = self._euler_to_quat(angle_rad, 0, 0)
            else:  # 'y'
                # Pendiente frontal (pitch) - SUBIDA hacia adelante
                quat = self._euler_to_quat(0, -angle_rad, 0)  # Negativo para inclinar correctamente
            
            # Aplicar rotación al suelo
            model.body_quat[0] = quat
            
            # MÉTODO 2 (ALTERNATIVO): Modificar gravedad para simular pendiente
            # Esto hace que "sienta" como si hubiera una pendiente
            if not self.gravity_modified:
                original_gravity = model.opt.gravity.copy()
                
                # Modificar componente de gravedad según pendiente
                if self.slope_axis == 'y':
                    # Pendiente hacia adelante: agregar componente Y a la gravedad
                    model.opt.gravity[1] = original_gravity[1] + 9.81 * np.sin(angle_rad)
                    model.opt.gravity[2] = original_gravity[2] * np.cos(angle_rad)
                else:  # 'x'
                    #Pendiente lateral
                    model.opt.gravity[0] = original_gravity[0] + 9.81 * np.sin(angle_rad)
                    model.opt.gravity[2] = original_gravity[2] * np.cos(angle_rad)
                
                self.gravity_modified = True
        
        return obs, info
    
    def _euler_to_quat(self, roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    def close(self):
        #Restaurar orientación y gravedad original
        if self.original_quat is not None and hasattr(self.env.unwrapped, 'model'):
            model = self.env.unwrapped.model
            model.body_quat[0] = self.original_quat
            #Restaurar gravedad por defecto
            if self.gravity_modified:
                model.opt.gravity[:] = [0, 0, -9.81]
        super().close()


def make_challenging_humanoid(difficulty='medium', wind=True, slippery=True, slope=False, **kwargs):
    # Configuraciones predefinidas
    configs = {
        'easy': {
            'wind_strength': 3.0,
            'friction_coef': 0.6,     
            'slope_angle': 0.0,        
            'wind_direction': 270,    
        },
        'medium': {
            'wind_strength': 10.0,
            'friction_coef': 0.6,    
            'slope_angle': 0.0,      
            'wind_direction': 270,    
        },
        'hard': {
            'wind_strength': 4.0,
            'friction_coef': 0.1,   
            'slope_angle': 15.0,     
            'wind_direction': 270,  
        },
        'extreme': {
            'wind_strength': 6.0,
            'friction_coef': 0.05,
            'slope_angle': 20.0,
            'wind_direction': 270,    
            'variable_wind': True,
        }
    }
    
    config = configs.get(difficulty, configs['medium'])
    
    #Crear ambiente base
    env = gym.make('Humanoid-v5', **kwargs)
    
    if slippery:
        env = SlipperyFloorWrapper(env, friction_coef=config['friction_coef'])
        print(f"Suelo resbaladizo")
    
    if slope:
        env = SlopedFloorWrapper(env, slope_angle=config['slope_angle'], slope_axis='y')
        print(f"Pendiente")
    
    if wind:
        env = WindForceWrapper(
            env,
            wind_strength=config['wind_strength'],
            wind_direction=config['wind_direction'],
            variable_wind=config.get('variable_wind', False)
        )
        wind_type = "variable" if config.get('variable_wind') else "constante"
        direction_str = {
            0: "derecha",
            90: "adelante", 
            180: "izquierda",
            270: "ATRAS"
        }.get(config['wind_direction'], f"{config['wind_direction']}°")
        
        print(f" Viento {wind_type}: {config['wind_strength']:.1f}N {direction_str}")
    
    return env

#Actor-Critic para acción continua con Tanh-Gaussian
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, log_std_init=-0.5, action_low=None, action_high=None):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),  # mu
        )
        #std como parámetro independiente por dimensión
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

        #Escalas/offset del espacio de accion (para mapear tanh∈[-1,1] a [low,high])
        self.register_buffer("action_low", torch.tensor(action_low, dtype=torch.float32))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))
        self.register_buffer("action_scale", (self.action_high - self.action_low) / 2.0)
        self.register_buffer("action_bias", (self.action_high + self.action_low) / 2.0)

    def forward(self, obs):
        raise NotImplementedError

    def critic_value(self, obs):
        return self.critic(obs)

    def _normal_dist(self, obs):
        mu = self.actor(obs)
        std = self.log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def sample_action_and_logprob(self, obs):
        dist = self._normal_dist(obs)
        u = dist.rsample()
        a_tanh = torch.tanh(u)
        action = a_tanh * self.action_scale + self.action_bias

        logprob_u = dist.log_prob(u).sum(axis=-1)
        correction = torch.log(1 - a_tanh.pow(2) + 1e-6).sum(axis=-1)
        logprob = logprob_u - correction

        return action, logprob, dist.mean, dist.stddev

    def logprob_of_action(self, obs, action):
        a_tanh = (action - self.action_bias) / (self.action_scale + 1e-8)
        a_tanh = torch.clamp(a_tanh, -0.999999, 0.999999)
        u = 0.5 * (torch.log1p(a_tanh) - torch.log1p(-a_tanh))

        dist = self._normal_dist(obs)
        logprob_u = dist.log_prob(u).sum(axis=-1)
        correction = torch.log(1 - a_tanh.pow(2) + 1e-6).sum(axis=-1)
        logprob = logprob_u - correction
        return logprob

#Rollout Buffer vectorizado con GAE(λ)
class VecRolloutBuffer:
    def __init__(self, T, N, obs_dim, act_dim, device):
        self.T = T
        self.N = N
        self.device = device
        self.obs = torch.zeros((T, N, obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((T, N, act_dim), dtype=torch.float32, device=device)
        self.logprobs = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.dones = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.values = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.returns = torch.zeros((T, N), dtype=torch.float32, device=device)
        self.ptr = 0

    def add_batch(self, obs, actions, logprobs, rewards, dones, values):
        t = self.ptr
        if t >= self.T:
            raise IndexError(f"VecRolloutBuffer overflow: t={t} T={self.T}")
        self.obs[t].copy_(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        self.actions[t].copy_(torch.as_tensor(actions, dtype=torch.float32, device=self.device))
        self.logprobs[t].copy_(torch.as_tensor(logprobs, dtype=torch.float32, device=self.device))
        self.rewards[t].copy_(torch.as_tensor(rewards, dtype=torch.float32, device=self.device))
        self.dones[t].copy_(torch.as_tensor(dones, dtype=torch.float32, device=self.device))
        self.values[t].copy_(torch.as_tensor(values, dtype=torch.float32, device=self.device))
        self.ptr += 1

    def compute_gae(self, last_values, gamma=0.995, gae_lambda=0.95):
        adv = torch.zeros(self.N, dtype=torch.float32, device=self.device)
        for t in reversed(range(self.T)):
            next_value = last_values if t == self.T - 1 else self.values[t + 1]
            next_nonterminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_nonterminal - self.values[t]
            adv = delta + gamma * gae_lambda * next_nonterminal * adv
            self.advantages[t] = adv
        self.returns = self.advantages + self.values

    def get(self, batch_size, shuffle=True):
        T, N = self.T, self.N
        flat_idx = torch.arange(T * N, device=self.device)
        if shuffle:
            flat_idx = flat_idx[torch.randperm(T * N, device=self.device)]
        for start in range(0, T * N, batch_size):
            mb = flat_idx[start:start + batch_size]
            t = (mb // N)
            n = (mb % N)
            yield (
                self.obs[t, n],
                self.actions[t, n],
                self.logprobs[t, n],
                self.advantages[t, n],
                self.returns[t, n],
                self.values[t, n],
            )


class PPOAgent:
    @staticmethod
    def make_env(env_id, seed, idx, reward_scaling, use_modified_env=False, difficulty='medium'):
        def thunk():
            if use_modified_env:
                # Ambiente modificado (viento + suelo resbaladizo)
                env = make_challenging_humanoid(
                    difficulty=difficulty,
                    wind=True,
                    slippery=True,
                    slope=False
                )
            else:
                # Ambiente normal
                env = gym.make(env_id)
            
            # Wrapper para escalar la recompensa
            def scale_reward(r):
                return r * reward_scaling
            env = gym.wrappers.TransformReward(env, scale_reward)
            env.action_space.seed(seed + idx)
            env.observation_space.seed(seed + idx)
            return env
        return thunk

    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        # Crear ambientes vectorizados
        self.envs = gym.vector.AsyncVectorEnv([
            self.make_env(
                cfg.env_id, 
                cfg.seed, 
                i, 
                getattr(cfg, 'reward_scaling', 1.0),
                use_modified_env=getattr(cfg, 'use_modified_env', False),
                difficulty=getattr(cfg, 'difficulty', 'medium')
            )
            for i in range(cfg.n_envs)
        ])
        assert isinstance(self.envs.single_action_space, gym.spaces.Box), "Este ejemplo asume acción continua."
        obs_dim = self.envs.single_observation_space.shape[0]
        act_dim = self.envs.single_action_space.shape[0]

        self.model = ActorCritic(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden=cfg.hidden,
            action_low=self.envs.single_action_space.low,
            action_high=self.envs.single_action_space.high,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)

        if getattr(cfg, 'use_lr_scheduler', False):
            total_updates = cfg.total_timesteps // (cfg.rollout_steps * cfg.n_envs)
            warmup_updates = getattr(cfg, 'lr_warmup_updates', 10)
            
            def lr_lambda(update):
                if update < warmup_updates:
                    return float(update) / float(max(1, warmup_updates))
                else:
                    progress = (update - warmup_updates) / max(1, total_updates - warmup_updates)
                    return 0.5 * (1 + np.cos(np.pi * progress))
            
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
            self.use_scheduler = True
            print(f" LR Scheduler activado (warmup: {warmup_updates} updates)")
        else:
            self.scheduler = None
            self.use_scheduler = False

        self.buffer = VecRolloutBuffer(
            T=cfg.rollout_steps,
            N=cfg.n_envs,
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=self.device,
        )

        self.global_steps = 0
        run_name = f"{cfg.env_id}"
        if getattr(cfg, 'use_modified_env', False):
            run_name += f"_modified_{cfg.difficulty}"
        run_name += f"_{int(time.time())}"
        self.writer = SummaryWriter(f"runs/{run_name}")
        print(f" TensorBoard logs: runs/{run_name}")

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    def collect_rollout(self):
        cfg = self.cfg
        device = self.device
        ep_logs = []

        obs, infos = self.envs.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)

        ep_len = torch.zeros(cfg.n_envs, dtype=torch.int32, device=device)
        ep_ret = torch.zeros(cfg.n_envs, dtype=torch.float32, device=device)

        self.buffer.ptr = 0

        steps = 0
        while steps < cfg.rollout_steps:
            with torch.no_grad():
                actions, logprobs, _, _ = self.model.sample_action_and_logprob(obs)
                value = self.model.critic_value(obs).squeeze(-1)

            actions_np = actions.cpu().numpy()
            next_obs, rewards, terminated, truncated, infos = self.envs.step(actions_np)
            dones = np.logical_or(terminated, truncated).astype(np.float32)

            if hasattr(self, 'global_steps'):
                for i in range(cfg.n_envs):
                    ep_len[i] += 1
                    ep_ret[i] += rewards[i]
                    if dones[i] > 0.5:
                        ep_logs.append((int(self.global_steps + steps*cfg.n_envs + i), float(ep_ret[i].item()), int(ep_len[i].item())))
                        ep_len[i] = 0
                        ep_ret[i] = 0.0

            self.buffer.add_batch(
                obs=obs,
                actions=actions,
                logprobs=logprobs,
                rewards=torch.as_tensor(rewards, dtype=torch.float32, device=device),
                dones=torch.as_tensor(dones, dtype=torch.float32, device=device),
                values=value,
            )

            obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            steps += 1

        with torch.no_grad():
            last_values = self.model.critic_value(obs).squeeze(-1)
        self.buffer.compute_gae(last_values, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
        if hasattr(self, 'writer'):
            self.writer.flush()
        return ep_logs

    def update(self, global_step):
        cfg = self.cfg
        approx_kl_all = []

        adv = self.buffer.advantages
        self.buffer.advantages = (adv - adv.mean()) / (adv.std() + 1e-8)

        for epoch in range(cfg.update_epochs):
            for obs_b, act_b, logp_old_b, adv_b, ret_b, val_old_b in self.buffer.get(cfg.minibatch_size):
                new_logp_b = self.model.logprob_of_action(obs_b, act_b)
                value_b = self.model.critic_value(obs_b).squeeze(-1)

                ratio = (new_logp_b - logp_old_b).exp()

                surr1 = ratio * adv_b
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef) * adv_b
                policy_loss = -torch.min(surr1, surr2).mean()

                v_pred_clipped = val_old_b + torch.clamp(value_b - val_old_b, -cfg.clip_vloss, cfg.clip_vloss)
                v_loss1 = (value_b - ret_b).pow(2)
                v_loss2 = (v_pred_clipped - ret_b).pow(2)
                value_loss = 0.5 * torch.min(v_loss1, v_loss2).mean()

                with torch.no_grad():
                    dist = self.model._normal_dist(obs_b)
                    entropy = dist.entropy().sum(axis=-1).mean()

                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                approx_kl = (logp_old_b - new_logp_b).mean().item()
                approx_kl_all.append(approx_kl)

            if np.mean(approx_kl_all[-(len(self.buffer.obs) // cfg.minibatch_size + 1):]) > cfg.target_kl:
                break
        
        self.writer.add_scalar('losses/policy_loss', policy_loss.item(), global_step)
        self.writer.add_scalar('losses/value_loss', value_loss.item(), global_step)
        self.writer.add_scalar('losses/entropy', entropy.item(), global_step)
        self.writer.add_scalar('losses/approx_kl', np.mean(approx_kl_all), global_step)
        self.writer.flush()  # Forzar escritura
        if global_step % cfg.log_interval == 0:
            print(f"[{global_step}] policy_loss: {policy_loss.item():.4f}  "
                  f"value_loss: {value_loss.item():.4f}  "
                  f"entropy: {entropy.item():.3f}  "
                  f"approx_kl: {np.mean(approx_kl_all):.5f}")

    def save_model(self, path):
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        torch.save({
            'model_state': self.model.state_dict(),
            'config': asdict(self.cfg)
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
    
    @torch.no_grad()
    def evaluate(self, episodes=5, render=True, use_modified_env=False, difficulty='medium'):
        render_mode = "human" if render else None
        
        if use_modified_env:
            print(f"\n  Evaluando en ambiente MODIFICADO (dificultad: {difficulty})")
            eval_env = make_challenging_humanoid(
                difficulty=difficulty,
                wind=True,
                slippery=True,
                slope=False,
                render_mode=render_mode,
                width=1024,
                height=800
            )
        else:
            print(f"\n Evaluando en ambiente NORMAL")
            eval_env = gym.make(self.cfg.env_id, render_mode=render_mode, width=1024, height=800)
        
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(episodes):
            obs, _ = eval_env.reset()
            done = False
            total_reward = 0
            steps = 0
            
            while not done:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    action = self.model.actor(obs_tensor.unsqueeze(0))
                    action = torch.tanh(action) * self.model.action_scale + self.model.action_bias
                    action = action.squeeze(0).cpu().numpy()
                
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            status = "CAIDA" if terminated else "TIEMPO"
            print(f"Episodio {ep+1}: Reward = {total_reward:7.2f} | Steps = {steps:4d} {status}")
        
        eval_env.close()
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        print(f"\n Reward promedio: {mean_reward:.2f} ± {std_reward:.2f}")
        return mean_reward, std_reward

    def save_checkpoint(self, path, global_steps, additional_info=None):
        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'global_steps': global_steps,
            'config': asdict(self.cfg),
            'rng_states': {
                'torch': torch.get_rng_state(),
                'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                'numpy': np.random.get_state(),
            }
        }
        
        if additional_info:
            checkpoint['additional_info'] = additional_info
        
        torch.save(checkpoint, path)
        print(f" Checkpoint guardado: {path} (pasos: {global_steps})")
    
    def load_checkpoint(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró checkpoint en: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if 'rng_states' in checkpoint:
            torch.set_rng_state(checkpoint['rng_states']['torch'])
            if checkpoint['rng_states']['torch_cuda'] and torch.cuda.is_available():
                torch.cuda.set_rng_state(checkpoint['rng_states']['torch_cuda'])
            np.random.set_state(checkpoint['rng_states']['numpy'])
        
        global_steps = checkpoint.get('global_steps', 0)
        
        print(f" Checkpoint cargado: {path}")
        print(f"  Pasos globales: {global_steps}")
        
        if 'additional_info' in checkpoint:
            print(f"  Info adicional: {checkpoint['additional_info']}")
        
        return global_steps
    
    def train(self, resume_from=None):
        cfg = self.cfg
        
        if resume_from and os.path.exists(resume_from):
            global_steps = self.load_checkpoint(resume_from)
            print(f" Continuando entrenamiento desde paso {global_steps}")
        else:
            global_steps = 0
            print(f" Iniciando entrenamiento desde cero")
        
        start = time.time()
        steps_per_rollout = cfg.rollout_steps * cfg.n_envs
        
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        save_interval = cfg.total_timesteps // 10
        next_save = global_steps + save_interval
        
        update_count = 0

        while global_steps < cfg.total_timesteps:
            logs = self.collect_rollout()
            
            self.global_steps = global_steps
            
            if logs:
                for step_x, ep_ret, ep_len in logs:
                    self.writer.add_scalar('charts/episode_return', ep_ret, step_x)
                    self.writer.add_scalar('charts/episode_length', ep_len, step_x)
                
                last_step, last_ret, last_len = logs[-1]
                elapsed = time.time() - start
                steps_per_sec = global_steps / elapsed if elapsed > 0 else 0
                
                print(f"[steps {last_step}] return={last_ret:.2f} len={last_len} "
                      f"({steps_per_sec:.0f} steps/s)")
            
            self.update(global_steps)
            global_steps += steps_per_rollout
            
            update_count += 1
            if self.use_scheduler:
                self.scheduler.step()
                current_lr = self.scheduler.get_last_lr()[0]
                self.writer.add_scalar('charts/learning_rate', current_lr, global_steps)

            if global_steps >= next_save:
                checkpoint_path = os.path.join(
                    checkpoint_dir, 
                    f"checkpoint_{global_steps}.pt"
                )
                self.save_checkpoint(
                    checkpoint_path, 
                    global_steps,
                    additional_info={'elapsed_time': time.time() - start}
                )
                next_save += save_interval
        
        final_checkpoint = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        self.save_checkpoint(final_checkpoint, global_steps)
        
        print(f" Entrenamiento terminado en {time.time()-start:.1f}s")
        self.envs.close()
        self.writer.close()


#Utilidades para checkpoints

def list_checkpoints(checkpoint_dir="checkpoints"):
    if not os.path.exists(checkpoint_dir):
        print(f"No existe el directorio: {checkpoint_dir}")
        return []
    
    checkpoints = []
    for fname in sorted(os.listdir(checkpoint_dir)):
        if fname.endswith('.pt'):
            path = os.path.join(checkpoint_dir, fname)
            try:
                ckpt = torch.load(path, map_location='cpu')
                checkpoints.append({
                    'path': path,
                    'filename': fname,
                    'global_steps': ckpt.get('global_steps', 'unknown'),
                    'size_mb': os.path.getsize(path) / (1024**2)
                })
            except Exception as e:
                print(f"Error leyendo {fname}: {e}")
    
    return checkpoints

def print_checkpoint_info(checkpoint_dir="checkpoints"):
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        print("No se encontraron checkpoints.")
        return
    
    print(f"\n{'='*70}")
    print(f"CHECKPOINTS DISPONIBLES ({len(checkpoints)})")
    print(f"{'='*70}")
    print(f"{'Archivo':<30} {'Pasos':<15} {'Tamaño (MB)':<15}")
    print(f"{'-'*70}")
    
    for ckpt in checkpoints:
        print(f"{ckpt['filename']:<30} {str(ckpt['global_steps']):<15} {ckpt['size_mb']:.2f}")
    
    print(f"{'='*70}\n")

def get_latest_checkpoint(checkpoint_dir="checkpoints"):
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        return None
    
    valid_ckpts = [c for c in checkpoints if isinstance(c['global_steps'], int)]
    if not valid_ckpts:
        return None
    
    latest = max(valid_ckpts, key=lambda x: x['global_steps'])
    return latest['path']

# Main
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Entrenar o evaluar agente PPO para Humanoid-v5')
    parser.add_argument('--mode', type=str, default='auto', 
                       choices=['train', 'resume', 'eval', 'eval_modified', 'compare', 'auto'],
                       help='''
                       train: entrenar desde cero
                       resume: continuar desde último checkpoint
                       eval: evaluar modelo en ambiente normal
                       eval_modified: evaluar en ambiente modificado
                       compare: comparar normal vs modificado
                       auto: detecta automáticamente (default)
                       ''')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Ruta específica del checkpoint/modelo')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Número de episodios para evaluación')
    parser.add_argument('--difficulty', type=str, default='medium',
                       choices=['easy', 'medium', 'hard', 'extreme'],
                       help='Dificultad del ambiente modificado')
    parser.add_argument('--no-render', action='store_true',
                       help='Desactivar visualización')
    args = parser.parse_args()
    
    @dataclass
    class PPOConfig:
        env_id: str = "Humanoid-v5"
        total_timesteps: int = 10_000_000
        rollout_steps: int = 2048
        update_epochs: int = 10
        minibatch_size: int = 256
        ent_coef: float = 0.01
        clip_coef: float = 0.2
        clip_vloss: float = 10.0
        vf_coef: float = 0.5
        target_kl: float = 0.01
        lr: float = 1e-4
        hidden: int = 512
        max_grad_norm: float = 1.0
        n_envs: int = 1
        seed: int = 42
        reward_scaling: float = 0.01
        gamma: float = 0.995
        gae_lambda: float = 0.95
        log_interval: int = 1
        use_lr_scheduler: bool = True
        lr_warmup_updates: int = 10
        use_modified_env: bool = True
        difficulty: str = 'medium'
        device: str = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = PPOConfig()
    agent = PPOAgent(cfg)
    
    #MODO: EVALUAR EN AMBIENTE MODIFICADO
    if args.mode == 'eval_modified':
        print("\n" + "="*70)
        print("MODO: EVALUACIÓN EN AMBIENTE MODIFICADO")
        print("="*70)
        
        if args.checkpoint:
            model_path = args.checkpoint
        else:
            if os.path.exists("modelo_humanoid_modified_medium.pt"):
                model_path = "modelo_humanoid_modified_medium.pt"
            else:
                model_path = get_latest_checkpoint()
                if not model_path:
                    print(" ERROR: No se encontró ningún modelo")
                    exit(1)
        
        if not os.path.exists(model_path):
            print(f" ERROR: No existe el modelo: {model_path}")
            exit(1)
        
        print(f" Modelo: {model_path}")
        print(f"  Dificultad: {args.difficulty}")
        print(f" Episodios: {args.episodes}\n")
        
        if 'checkpoint' in model_path:
            agent.load_checkpoint(model_path)
        else:
            agent.load_model(model_path)
        
        agent.evaluate(
            episodes=args.episodes, 
            render=not args.no_render,
            use_modified_env=True,
            difficulty=args.difficulty
        )
    #MODO AUTO
    elif args.mode == 'auto':
        print("\n" + "="*70)
        print("MODO AUTO: Detectando estado...")
        print("="*70)
        
        latest_checkpoint = get_latest_checkpoint()
        final_model = "modelo_humanoid_171225.pt"
        
        if os.path.exists(final_model):
            print(f" Se encontró modelo final: {final_model}")
            print("  Iniciando modo EVALUACIÓN\n")
            agent.load_model(final_model)
            agent.evaluate(episodes=args.episodes, render=not args.no_render)
            
        elif latest_checkpoint:
            print(f" Se encontró checkpoint: {latest_checkpoint}")
            ckpt_info = torch.load(latest_checkpoint, map_location='cpu')
            steps = ckpt_info.get('global_steps', 0)
            print(f"  Pasos: {steps:,} / {cfg.total_timesteps:,}")
            
            if steps >= cfg.total_timesteps:
                print("  Iniciando modo EVALUACIÓN\n")
                agent.load_checkpoint(latest_checkpoint)
                agent.evaluate(episodes=args.episodes, render=not args.no_render)
            else:
                print("  Iniciando modo RESUME\n")
                agent.train(resume_from=latest_checkpoint)
                agent.save_model(final_model)
        else:
            print("  Iniciando modo TRAIN\n")
            agent.train(resume_from=None)
            agent.save_model(final_model)

    #MODO TRAIN
    elif args.mode == 'train':
        print("\n" + "="*70)
        print("MODO TRAIN: Entrenamiento desde cero")
        print("="*70)
        
        agent.train(resume_from=None)
        agent.save_model("modelo_humanoid_test.pt")
        print("\n Modelo guardado: modelo_humanoid_test.pt")
    
    #MODO RESUME
    elif args.mode == 'resume':
        print("\n" + "="*70)
        print("MODO RESUME: Continuar entrenamiento")
        print("="*70)
        
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            checkpoint_path = get_latest_checkpoint()
            if not checkpoint_path and os.path.exists("modelo_humanoid_modified_easy.pt") and cfg.use_modified_env:
                print(" No se encontró checkpoint, pero existe modelo entrenado")
                print(f"   Dificultad: {cfg.difficulty}")
                print(f"   Cargando: modelo_humanoid_test.pt\n")
                
                agent.load_model("modelo_humanoid_test.pt")
                agent.train(resume_from=None) 
                agent.save_model(f"modelo_humanoid_modified_{cfg.difficulty}.pt")
                print(f"\n Modelo fine-tuned guardado: modelo_humanoid_modified_{cfg.difficulty}.pt")
                exit(0)
            
            if not checkpoint_path:
                print(" ERROR: No se encontraron checkpoints")
                exit(1)
        
        if not os.path.exists(checkpoint_path):
            print(f" ERROR: No existe: {checkpoint_path}")
            exit(1)
        
        ckpt_info = torch.load(checkpoint_path, map_location='cpu')
        steps = ckpt_info.get('global_steps', 0)
        print(f" Checkpoint: {checkpoint_path}")
        print(f" Pasos completados: {steps:,} / {cfg.total_timesteps:,}")
        print(f" Progreso: {steps/cfg.total_timesteps*100:.1f}%")
        
        if cfg.use_modified_env:
            print(f"  Entrenando en ambiente MODIFICADO (dificultad: {cfg.difficulty})")
        
        print()
        
        agent.train(resume_from=checkpoint_path)
        
        # Guardar con nombre apropiado
        if cfg.use_modified_env:
            model_name = f"modelo_humanoid_modified_{cfg.difficulty}.pt"
        else:
            model_name = "modelo_humanoid_181225.pt"
        
        agent.save_model(model_name)
        print(f"\n Modelo guardado: {model_name}")
    
    #MODO EVAL
    elif args.mode == 'eval':
        print("\n" + "="*70)
        print("MODO EVAL: Evaluación en ambiente normal")
        print("="*70)
        
        if args.checkpoint:
            model_path = args.checkpoint
        else:
            if os.path.exists("modelo_humanoid_test.pt"):
                model_path = "modelo_humanoid_test.pt"
            else:
                model_path = get_latest_checkpoint()
                if not model_path:
                    print(" ERROR: No se encontró ningún modelo")
                    exit(1)
        
        if not os.path.exists(model_path):
            print(f" ERROR: No existe: {model_path}")
            exit(1)
        
        if 'checkpoint' in model_path:
            agent.load_checkpoint(model_path)
        else:
            agent.load_model(model_path)
        
        agent.evaluate(episodes=args.episodes, render=not args.no_render)


"""
 EVALUACIÓN EN AMBIENTE MODIFICADO:
--------------------------------------
# Con visualización (recomendado para ver el efecto)
python ppo_funcional_vectorized.py --mode eval_modified --difficulty medium

EVALUACIÓN NORMAL:
---------------------
python ppo_funcional_vectorized.py --mode eval

ENTRENAMIENTO:
------------------
# Desde cero
python ppo_funcional_vectorized.py --mode train

# Continuar desde checkpoint
python ppo_funcional_vectorized.py --mode resume
"""