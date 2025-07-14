import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

#This file is not really important at the moment
# from gym.envs.classic_control import PendulumEnv
from Prac_Env import prac_env_v0

from model import Normalizer, ForwardModel
from logamp_rollout import Rollout


# enable TF Eager
tf.enable_eager_execution()


# define job directory with saved checkpoints
job_dir = '.\logamp_logs'
max_episode_steps = 200
episodes = 4


# create an environment
env = prac_env_v0()
env.seed(42)

# create a rollout
rollout = Rollout(env, max_episode_steps=max_episode_steps)

# sample rollouts
states, actions, rewards, next_states, weights = rollout(
     lambda state: env.action_space.sample(),
     episodes=episodes)

# compute deltas between the next state and the current state
deltas = next_states - states

# create normalizers for the features and targets
# NOTE: it's important that the statistics match those used during training
#       they will be restored from the checkpoint
state_normalizer = Normalizer(
    loc=states.mean(axis=(0, 1)),
    scale=states.std(axis=(0, 1)))
delta_normalizer = Normalizer(
    loc=deltas.mean(axis=(0, 1)),
    scale=deltas.std(axis=(0, 1)))
action_normalizer = Normalizer(
    loc=actions.mean(axis=(0, 1)),
    scale=actions.std(axis=(0, 1)))

# create the forward model
model = ForwardModel(output_units=env.observation_space.shape[-1])

# create a checkpoint with references to all objects to restore
checkpoint = tf.train.Checkpoint(
    state_normalizer=state_normalizer,
    delta_normalizer=delta_normalizer,
    action_normalizer=action_normalizer,
    model=model)

# restore the latest checkpoint in job_dir
checkpoint_path = tf.train.latest_checkpoint(job_dir)
assert checkpoint_path is not None, 'job_dir must contain checkpoint'
checkpoint.restore(checkpoint_path)


# normalize features
states_norm = state_normalizer(states)
actions_norm = action_normalizer(actions)

# compute a forward pass while resetting the RNN state
deltas_norm_pred = model(states_norm, actions_norm, training=False, reset_state=True)

# de-normalize the predicted delta
deltas_pred = delta_normalizer.invert(deltas_norm_pred)

# add the prior states to the unnormalized deltas
next_states_pred = states + deltas_pred.numpy()

# plot the instantaneous predictions for each episode and state
state_size = env.observation_space.shape[-1]
fig, axes = plt.subplots(episodes, state_size, figsize=(12, 8))
for state_dim in range(state_size):
    for episode in range(episodes):
        ax = axes[episode, state_dim]
        ax.plot(next_states[episode, :, state_dim], label='Real')
        ax.plot(next_states_pred[episode, :, state_dim], label='Predicted')
        ax.legend(loc='lower right')
        ax.set_title('State: {}, Episode: {}'.format(state_dim, episode))
sns.despine()
plt.tight_layout()
plt.show()