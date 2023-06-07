import os
import sys
sys.path.append('./lm_envs')
import torch
import gym
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pickle
import time

WHITE_BOX = True  # set to False to do a black box attack

register(
    id='WBALMEnv-v0',
    entry_point='lm_envs:WBALMEnv',
    max_episode_steps=1,
)

TIMESTEPS = 250000

SD = int(str(time.time()).replace('.', '')) % 100000


if __name__ == '__main__':

    print(f'WBA experiment white_box={WHITE_BOX}')

    env = VecNormalize(DummyVecEnv([lambda: gym.make('WBALMEnv-v0', white_box=WHITE_BOX)]))

    model = PPO("MlpPolicy", env, learning_rate=3e-4, batch_size=2048, n_epochs=15, ent_coef=0.25,
                policy_kwargs={'log_std_init': -1.0,
                               'ortho_init': True,
                               'activation_fn': torch.nn.ReLU,
                               'net_arch': dict(pi=[64], vf=[64])})
    model.learn(total_timesteps=TIMESTEPS, progress_bar=True)

    all_rewards = env.venv.envs[0].mean_rewards
    sample_prompts = env.venv.envs[0].store_prompts
    sample_responses = env.venv.envs[0].store_responses
    with open(f'data/wba_latent_white_box={WHITE_BOX}_timesteps={TIMESTEPS}_id={SD}.pkl', 'wb') as f:
        pickle.dump({'all_rewards': all_rewards,
                     'sample_prompts': sample_prompts,
                     'sample_responses': sample_responses}, f)

    print('Done :)')
