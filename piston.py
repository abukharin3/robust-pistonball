import supersuit as ss
from ppo import PPO
from stable_baselines3.ppo import CnnPolicy
from pettingzoo.butterfly import cooperative_pong_v4 as pistonball_v6
import numpy as np
import pettingzoo.butterfly.pistonball_v6 as pistonball_v6

def train(model_path='models/policy', exp_name=''):
    env = pistonball_v6.parallel_env(
        n_pistons=20,
        time_penalty=-0.1,
        continuous=True,
        random_drop=True,
        random_rotate=True,
        ball_mass=0.75,
        ball_friction=0.3,
        ball_elasticity=1.5,
        max_cycles=125,
    )
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 2, num_cpus=1, base_class="stable_baselines3")
    model = PPO(
        CnnPolicy,
        env,
        verbose=3,
        gamma=0.95,
        n_steps=256,
        ent_coef=0.0905168,
        learning_rate=0.00062211,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        n_epochs=5,
        clip_range=0.3,
        batch_size=256,
    )
    _, rewards = model.learn(total_timesteps=2000000)
    print("Trained")
    model.save(model_path)
    np.save("results/train_{}.npy".format(exp_name), np.array(rewards))

def eval(model_path='models/policy', exp_name='', ball_mass=0.75, ball_friction=0.3):
    rewards = []
    for ball_mass in [0.1, 0.3, 0.6, 0.75, 1.0, 1.5]:
        env = pistonball_v6.env(
            n_pistons=20,
            time_penalty=-0.1,
            continuous=True,
            random_drop=True,
            random_rotate=True,
            ball_mass=ball_mass,
            ball_friction=0.3,
            ball_elasticity=0.1,
            max_cycles=125,
        )
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

        model = PPO.load(model_path)

        env.reset()
        envrewards = []
        for agent in env.agent_iter():
            obs, reward, done, info = env.last()
            venrewards.append(reward)
            act = model.predict(obs, deterministic=True)[0] if not done else None
            env.step(act)
            #env.render()
        rewards.append(envrewards)
    np.save("results/eval_{}.npy".format(exp_name), np.array(rewards))

if __name__ == "__main__":
    train()
    eval()