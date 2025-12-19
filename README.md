# **PPO**

### EVALUACIÓN EN AMBIENTE MODIFICADO:
--------------------------------------
**Con visualización (recomendado para ver el efecto)**  
python ppo_funcional_vectorized.py --mode eval_modified --difficulty medium

### EVALUACIÓN NORMAL:
---------------------
python ppo_funcional_vectorized.py --mode eval

### ENTRENAMIENTO:
------------------
**Desde cero**  
python ppo_funcional_vectorized.py --mode train

**Continuar desde checkpoint**  
python ppo_funcional_vectorized.py --mode resume


# **SAC**

### ENTRENAMIENTO:
------------------
**Entrenar SAC (implementación propia)**  
python sac.py

### VISUALIZACIÓN:
------------------
**Ver ejecución del último modelo entrenado**  
python visualize_sac.py

### EVALUACIÓN CON PERTURBACIÓN (VIENTO):
----------------------------------------
**Ejecutar política con fuerza externa aplicada al torso**  
python viento.py


# **TD3**

### ENTRENAMIENTO:
------------------
**Entrenar desde cero (elige stage: balance o walk)**  
python td3_train.py --run_name TD3_stage2_walk --stage walk

**Entrenar balance**  
python td3_train.py --run_name TD3_stage1_balance --stage balance

### CONTINUAR DESDE CHECKPOINT (ejemplo):
----------------------------------------
python td3_train.py \
  --run_name TD3_stage2_walk \
  --stage walk \
  --resume runs_td3/TD3_stage2_walk/checkpoints/td3_step_5000000.pt \
  --total_steps 20000000

### VISUALIZACIÓN / EVALUACIÓN:
------------------------------
**Ver un checkpoint concreto (render)**  
python td3_watch.py \
  --run_dir runs_td3/TD3_stage2_walk \
  --target_steps 18000000 \
  --episodes 3
