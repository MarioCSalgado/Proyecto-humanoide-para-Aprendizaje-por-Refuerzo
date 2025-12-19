import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import random
import glob
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm import tqdm

# ==============================================================================
# 1. CONFIGURACIÓN
# ==============================================================================
CONFIG = {
    "env_name": "Humanoid-v4",
    "total_steps": 1000000,
    "start_steps": 10_000,
    "batch_size": 256,
    "buffer_size": 1_000_000,
    "lr": 3e-4,
    "gamma": 0.99,
    "tau": 0.005,
    "hidden_dim": [400, 300],
    "save_freq": 100_000,
    "log_dir": "./logs_sac_propio",
    "model_dir": "./models_sac_propio"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Aseguramos que las carpetas existan ANTES de empezar
os.makedirs(CONFIG["model_dir"], exist_ok=True)
os.makedirs(CONFIG["log_dir"], exist_ok=True)

# ==============================================================================
# 2. REDES NEURONALES 
# ==============================================================================
def weights_init_ortho(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
        torch.nn.init.constant_(m.bias, 0)

class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions):
        super(SoftQNetwork, self).__init__()
        def create_q():
            return nn.Sequential(
                nn.Linear(num_inputs + num_actions, CONFIG["hidden_dim"][0]), nn.ReLU(),
                nn.Linear(CONFIG["hidden_dim"][0], CONFIG["hidden_dim"][1]), nn.ReLU(),
                nn.Linear(CONFIG["hidden_dim"][1], 1)
            )
        self.q1 = create_q()
        self.q2 = create_q()
        self.apply(weights_init_ortho)

    def forward(self, state, action):
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

class GaussianPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, action_space):
        super(GaussianPolicy, self).__init__()
        self.base = nn.Sequential(
            nn.Linear(num_inputs, CONFIG["hidden_dim"][0]), nn.ReLU(),
            nn.Linear(CONFIG["hidden_dim"][0], CONFIG["hidden_dim"][1]), nn.ReLU()
        )
        self.mu = nn.Linear(CONFIG["hidden_dim"][1], num_actions)
        self.log_std = nn.Linear(CONFIG["hidden_dim"][1], num_actions)
        self.apply(weights_init_ortho)

        self.action_scale = torch.tensor((action_space.high - action_space.low) / 2., dtype=torch.float32).to(device)
        self.action_bias = torch.tensor((action_space.high + action_space.low) / 2., dtype=torch.float32).to(device)

    def forward(self, state):
        x = self.base(state)
        mu = self.mu(x)
        log_std = torch.clamp(self.log_std(x), -20, 2)
        return mu, log_std

    def sample(self, state):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample() 
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        return action, log_prob.sum(1, keepdim=True), torch.tanh(mu) * self.action_scale + self.action_bias

# ==============================================================================
# 3. REPLAY BUFFER
# ==============================================================================
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, mask)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, mask = map(np.stack, zip(*batch))
        return state, action, reward, next_state, mask

    def __len__(self):
        return len(self.buffer)

# ==============================================================================
# 4. AGENTE SAC 
# ==============================================================================
class SACAgent:
    def __init__(self, num_inputs, action_space):
        self.critic = SoftQNetwork(num_inputs, action_space.shape[0]).to(device)
        self.critic_target = SoftQNetwork(num_inputs, action_space.shape[0]).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=CONFIG["lr"])

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], action_space).to(device)
        self.policy_opt = optim.Adam(self.policy.parameters(), lr=CONFIG["lr"])

        self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(device)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=CONFIG["lr"])

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)
        action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy().flatten()

    def update(self, memory):
        state, action, reward, next_state, mask = memory.sample(CONFIG["batch_size"])
        
        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(device)
        mask = torch.FloatTensor(mask).to(device).unsqueeze(1)

        with torch.no_grad():
            next_action, next_log_pi, _ = self.policy.sample(next_state)
            q1_target, q2_target = self.critic_target(next_state, next_action)
            min_target_q = torch.min(q1_target, q2_target) - torch.exp(self.log_alpha) * next_log_pi
            target_q = reward + mask * CONFIG["gamma"] * min_target_q

        curr_q1, curr_q2 = self.critic(state, action)
        q_loss = F.mse_loss(curr_q1, target_q) + F.mse_loss(curr_q2, target_q)
        self.critic_opt.zero_grad()
        q_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_opt.step()

        pi, log_pi, _ = self.policy.sample(state)
        q1_pi, q2_pi = self.critic(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (torch.exp(self.log_alpha).detach() * log_pi - min_q_pi).mean()
        
        self.policy_opt.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.policy_opt.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        for lp, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(tp.data * (1.0 - CONFIG["tau"]) + lp.data * CONFIG["tau"])

        return q_loss.item(), policy_loss.item(), torch.exp(self.log_alpha).item()

# ==============================================================================
# 5. ENTRENAMIENTO CON REINICIO (RESUME)
# ==============================================================================
if __name__ == "__main__":
    print(f"🚀 Iniciando/Reiniciando entrenamiento en {device}")
    
    # Entorno inicial
    raw_env = gym.make(CONFIG["env_name"])
    env = DummyVecEnv([lambda: raw_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    agent = SACAgent(env.observation_space.shape[0], env.action_space)
    memory = ReplayBuffer(CONFIG["buffer_size"])
    writer = SummaryWriter(CONFIG["log_dir"])
    
    # --- LÓGICA DE REINICIO ---
    start_step = 0
    # Buscamos el último checkpoint de política disponible
    policy_files = glob.glob(f"{CONFIG['model_dir']}/policy_*.pth")
    
    if policy_files:
        # Extraemos los números de los pasos de los nombres de archivos
        steps = [int(f.split('_')[-1].split('.')[0]) for f in policy_files]
        latest_step = max(steps)
        
        policy_path = f"{CONFIG['model_dir']}/policy_{latest_step}.pth"
        vec_path = f"{CONFIG['model_dir']}/vec_normalize_{latest_step}.pkl"
        
        if os.path.exists(policy_path):
            print(f"♻️ Cargando política desde paso {latest_step}...")
            agent.policy.load_state_dict(torch.load(policy_path, map_location=device))
            start_step = latest_step
            
        if os.path.exists(vec_path):
            print(f"👓 Cargando normalización del entorno...")
            env = VecNormalize.load(vec_path, env.venv)
            env.training = True 

    obs = env.reset()
    episode_reward = 0
    
    for step in tqdm(range(start_step, CONFIG["total_steps"]), desc="Entrenando Humanoid", initial=start_step, total=CONFIG["total_steps"]):
        
        if step < CONFIG["start_steps"] and start_step == 0:
            action = env.action_space.sample().flatten() 
        else:
            action = agent.select_action(obs[0])

        next_obs, reward, done, info = env.step([action])
        
        mask = 0 if done[0] and not info[0].get("TimeLimit.truncated") else 1
        memory.push(obs[0], action, reward[0], next_obs[0], mask)
        
        obs = next_obs
        episode_reward += info[0].get('episode', {}).get('r', reward[0])

        if len(memory) > CONFIG["batch_size"]:
            q_l, p_l, alpha = agent.update(memory)
            if step % 100 == 0:
                writer.add_scalar("Loss/Critic", q_l, step)
                writer.add_scalar("Loss/Policy", p_l, step)
                writer.add_scalar("Stats/Alpha", alpha, step)

        if done[0]:
            writer.add_scalar("Rollout/Episode_Reward", episode_reward, step)
            obs = env.reset()
            episode_reward = 0

        # Guardado periódico
        if step > 0 and step % CONFIG["save_freq"] == 0:
            save_path_policy = f"{CONFIG['model_dir']}/policy_{step}.pth"
            save_path_vec = f"{CONFIG['model_dir']}/vec_normalize_{step}.pkl"
            
            torch.save(agent.policy.state_dict(), save_path_policy)
            env.save(save_path_vec)
            print(f"\n💾 Checkpoint guardado en paso {step}")

    writer.close()
    print("✅ Entrenamiento completado.")