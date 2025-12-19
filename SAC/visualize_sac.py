import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import os
import time
import numpy as np
import mujoco # <--- Importante para el parche

# ==============================================================================
# 🩹 PARCHE DE COMPATIBILIDAD (CRUCIAL PARA TU ERROR)
# ==============================================================================
# Esto engaña a Gymnasium para que crea que 'solver_iter' existe,
# redirigiéndolo al nuevo nombre 'solver_niter'.
try:
    # Si la clase MjData no tiene solver_iter...
    if not hasattr(mujoco.MjData, 'solver_iter'):
        # ... creamos una propiedad que apunte a solver_niter
        mujoco.MjData.solver_iter = property(lambda self: self.solver_niter)
    print("✅ Parche de MuJoCo aplicado correctamente.")
except Exception as e:
    print(f"⚠️ No se pudo aplicar el parche: {e}")
# ==============================================================================


# --- CONFIGURACIÓN ---
MODELS_DIR = "models_sac_humanoid_pro" 
MODEL_NAME = None # Dejar None para buscar automático

def get_latest_model(directory):
    if not os.path.exists(directory): return None
    files = [f for f in os.listdir(directory) if f.endswith(".zip")]
    if not files: return None
    latest = max([os.path.join(directory, f) for f in files], key=os.path.getctime)
    return latest

def main():
    print("🚀 Iniciando Visualizador PRO (Con Parche)...")
    
    # 1. Localizar archivos
    if MODEL_NAME:
        model_path = os.path.join(MODELS_DIR, MODEL_NAME)
    else:
        model_path = get_latest_model(MODELS_DIR)
        
    stats_path = os.path.join(MODELS_DIR, "vec_normalize.pkl")

    if not model_path or not os.path.exists(model_path):
        print(f"❌ No encuentro el modelo .zip en {MODELS_DIR}")
        return
    
    # Búsqueda de backup para el archivo de estadísticas
    if not os.path.exists(stats_path):
        print(f"⚠️ No encuentro '{stats_path}' en la raíz.")
        possible_path = os.path.join(MODELS_DIR, "best_model", "vec_normalize.pkl")
        if os.path.exists(possible_path):
            stats_path = possible_path
            print(f"   ✅ Encontrado en: {stats_path}")
        else:
            print("❌ Falta el archivo vec_normalize.pkl. El robot caminará mal sin él.")
            # Continuamos igual para ver si al menos no crashea
            
    print(f"🧠 Modelo: {model_path}")
    print(f"👓 Lentes: {stats_path}")

    # 2. Crear Entorno
    try:
        env = gym.make("Humanoid-v4", render_mode="human")
    except Exception as e:
        print(f"❌ Error creando entorno: {e}")
        return

    env = DummyVecEnv([lambda: env])

    # 3. Cargar Normalización (Si existe el archivo)
    if os.path.exists(stats_path):
        env = VecNormalize.load(stats_path, env)
        env.training = False
        env.norm_reward = False
    else:
        print("⚠️ CUIDADO: Cargando sin normalización (espera movimientos erráticos).")

    # 4. Cargar Agente
    try:
        model = SAC.load(model_path, env=env)
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return

    print("\n✅ Sistema listo. ¡Acción!")

    # 5. Bucle
    obs = env.reset()
    
    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            time.sleep(0.016) # 60 FPS
            
    except KeyboardInterrupt:
        print("\n👋 Cerrando.")
    finally:
        env.close()

if __name__ == "__main__":
    main()