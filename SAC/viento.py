import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import time
import numpy as np
import mujoco

# ==============================================================================
# 🩹 PARCHE DE COMPATIBILIDAD
# ==============================================================================
try:
    if not hasattr(mujoco.MjData, 'solver_iter'):
        mujoco.MjData.solver_iter = property(lambda self: self.solver_niter)
    print("✅ Parche de MuJoCo aplicado correctamente.")
except Exception as e:
    print(f"⚠️ No se pudo aplicar el parche: {e}")

# ==============================================================================
# 🌪️ CONFIGURACIÓN DE VIENTO
# ==============================================================================
# [X, Y, Z] en Newtons. 
# X: Hacia adelante/atrás, Y: Lateral, Z: Arriba/Abajo
VIENTO_FUERZA = np.array([25.0, 0.0, 0.0]) # 40N es una ráfaga decente
APLICAR_VIENTO = True 

# --- CONFIGURACIÓN DE RUTAS ---
MODELS_DIR = "models_sac_humanoid_pro" 
MODEL_NAME = None 

def get_latest_model(directory):
    if not os.path.exists(directory): return None
    files = [f for f in os.listdir(directory) if f.endswith(".zip")]
    if not files: return None
    latest = max([os.path.join(directory, f) for f in files], key=os.path.getctime)
    return latest

def main():
    print("🚀 Iniciando Visualizador con VIENTO...")
    
    model_path = get_latest_model(MODELS_DIR) if not MODEL_NAME else os.path.join(MODELS_DIR, MODEL_NAME)
    stats_path = os.path.join(MODELS_DIR, "vec_normalize.pkl")

    if not model_path or not os.path.exists(model_path):
        print(f"❌ No encuentro el modelo en {MODELS_DIR}")
        return
    
    # 2. Crear Entorno
    env = gym.make("Humanoid-v4", render_mode="human")
    env = DummyVecEnv([lambda: env])

    # 3. Cargar Normalización
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
        print(f"   ✅ Estadísticas cargadas: {stats_path}")

    # 4. Cargar Agente
    model = SAC.load(model_path, env=env)
    
    # Obtener acceso directo a MuJoCo
    # env.envs[0] accede al entorno real dentro del DummyVecEnv
    unwrapped_env = env.envs[0].unwrapped
    model_mj = unwrapped_env.model
    data_mj = unwrapped_env.data
    
    # Buscar el ID del torso para aplicarle el viento
    torso_id = mujoco.mj_name2id(model_mj, mujoco.mjtObj.mjOBJ_BODY, 'torso')

    print(f"\n✅ Sistema listo. Viento activo: {APLICAR_VIENTO} (Fuerza: {VIENTO_FUERZA}N)")

    obs = env.reset()
    try:
        while True:
            # --- LÓGICA DE VIENTO ---
            if APLICAR_VIENTO:
                # xfrc_applied es un array de (nbody, 6) -> [FX, FY, FZ, TX, TY, TZ]
                data_mj.xfrc_applied[torso_id][:3] = VIENTO_FUERZA
            
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            
            # Nota: MuJoCo limpia xfrc_applied automáticamente después de mj_step, 
            # por eso lo reasignamos en cada iteración del bucle.

            time.sleep(0.016) # ~60 FPS
            
    except KeyboardInterrupt:
        print("\n👋 Cerrando.")
    finally:
        env.close()

if __name__ == "__main__":
    main()