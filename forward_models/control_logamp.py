import tensorflow as tf
import numpy as np
import time
import argparse

from Prac_Env import prac_env_v0
from Forward_Model import ForwardModel, Normalizer

# === Enable Eager Execution ===
tf.enable_eager_execution()

# === Parse Arguments ===
parser = argparse.ArgumentParser()
parser.add_argument('--test', action='store_true', help='Run in test mode without hardware')
parser.add_argument('--checkpoint-dir', type=str, default='./logs/', help='Checkpoint directory')
args = parser.parse_args()

# === Create Environment ===
env = prac_env_v0(test_mode=args.test)
env.seed(42)
state = env.reset()
target_state = env.current

# === Load Model & Normalizers ===
model = ForwardModel(output_units=env.observation_space.shape[-1])

state_normalizer = Normalizer(loc=np.zeros((1,), dtype=np.float32),
                            scale=np.ones((1,), dtype=np.float32))#loc=states_train.mean(axis=(0, 1), dtype=np.float64),
                            #scale=states_train.std(axis=(0, 1), dtype=np.float64))#loc=np.zeros((1,)), scale=np.ones((1,)))
delta_normalizer = Normalizer(loc=np.zeros((1,), dtype=np.float32),
                            scale=np.ones((1,), dtype=np.float32))#loc=states_train.mean(axis=(0, 1), dtype=np.float64),
                            #scale=states_train.std(axis=(0, 1), dtype=np.float64))#loc=np.zeros((1,)), scale=np.ones((1,)))
action_normalizer = Normalizer(loc=np.zeros((1,), dtype=np.float32),
                            scale=np.ones((1,), dtype=np.float32))#loc=states_train.mean(axis=(0, 1), dtype=np.float64),
                            #scale=states_train.std(axis=(0, 1), dtype=np.float64))#loc=np.zeros((env.action_space.n,)), scale=np.ones((env.action_space.n,)))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
global_step = tf.train.create_global_step()

checkpoint = tf.train.Checkpoint(
    model=model,
    state_normalizer=state_normalizer,
    delta_normalizer=delta_normalizer,
    action_normalizer=action_normalizer,
    optimizer=optimizer,
    global_step=global_step
)

# === Load Checkpoint ===
#args.checkpoint_dir}")

ckpt_path = tf.train.latest_checkpoint('.\data0')#args.checkpoint_dir)
if ckpt_path is None:
    raise RuntimeError(f'No checkpoint found in .\logs_1')
checkpoint.restore(ckpt_path)
print(f"[Checkpoint] Restored model from {ckpt_path}")

# === Deploy Control Loop ===
print(f"[Start] Running control loop with target: {target_state:.4f}\n")

for step in range(50):
    # Format state
    state_tensor = tf.convert_to_tensor(np.array([[[state]]], dtype=np.float32))
    state_norm = state_normalizer(state_tensor)

    best_action = None
    best_reward = -np.inf
    predicted_next_state = None

    for i in range(env.action_space.n):
        action_vec = np.zeros((1, 1, env.action_space.n), dtype=np.float32)
        action_vec[0, 0, i] = 1.0
        action_tensor = tf.convert_to_tensor(action_vec)
        action_norm = action_normalizer(action_tensor)

        # Predict normalized delta
        delta_pred_norm = model(state_norm, action_norm, training=False, reset_state=True)
        delta_pred = delta_normalizer.invert(delta_pred_norm)
        next_state_pred = state + delta_pred.numpy()[0, 0, 0]

        reward = -np.abs(next_state_pred - target_state)

        if reward > best_reward:
            best_reward = reward
            best_action = i
            predicted_next_state = next_state_pred

    # Take the action
    print(f"\n[Step {step + 1}]")
    print(f"Current State: {state:.4f}")
    print(f"Target State : {target_state:.4f}")
    print(f"Action Index : {best_action} ({env.action_list[best_action]:.4f})")
    print(f"Predicted NS : {predicted_next_state:.4f}")

    state, reward, done, _ = env.step(best_action)
    env.render()

    if done:
        print("[Exit] Environment signaled done.")
        break

    time.sleep(0.1)

env.close()
print("\n[Done] Control loop complete.")