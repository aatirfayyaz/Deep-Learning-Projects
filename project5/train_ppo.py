import argparse
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from vec_env_utils import make_vec_env
from robot import Robot
from arm_dynamics import ArmDynamics
from arm_env import ArmEnv


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--timesteps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--nenv', type=int, default=8)
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--num_links', type=int, default=2)
    parser.add_argument('--link_mass', type=float, default=0.1)
    parser.add_argument('--friction', type=float, default=0.1)
    parser.add_argument('--link_length', type=float, default=1)
    parser.add_argument('--dt', type=float, default=0.01)
    args = parser.parse_args()
    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, args.timestr) if args.suffix is None \
        else os.path.join(args.save_dir, args.timestr + '_' + args.suffix)

    return args


def make_arm(args):
    arm = Robot(
        ArmDynamics(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.dt
        )
    )
    arm.reset()
    return arm


def train(args):
    set_random_seed(args.seed)

    # create arm
    arm = make_arm(args)

    # create parallel envs
    vec_env = make_vec_env(arm=arm, nenv=args.nenv, seed=args.seed)
    # check_env(env, warn=True)
    vec_env.reset()

    # ------ IMPLEMENT YOUR TRAINING CODE HERE ------------
    # ppo_model = PPO("MlpPolicy", env=vec_env, learning_rate=0.0003, batch_size=64, seed=args.seed,
    #                 ent_coef=0.05, gamma=0.99, max_grad_norm=0.5, verbose=1)
    ppo_model = PPO("MlpPolicy", env=vec_env, learning_rate=0.0003, batch_size=64, seed=args.seed,
                    ent_coef=0.1, gamma=0.99, max_grad_norm=0.5, verbose=1)

    if not os.path.exists(args.save_dir):
        os.makedirs(os.path.join(args.save_dir))
    ppo_model.learn(total_timesteps=500000)  # 786432, 933000
    save_path = os.path.join(args.save_dir, 'model')
    ppo_model.save(save_path)
    # test(vec_env, ppo_model)
    pass


def test(vec_env, ppo_model):
    # Testing the environment
    obs = vec_env.reset()
    for _ in range(100):
        action, _states = ppo_model.predict(obs)
        print(action)
        obs, rewards, dones, info = vec_env.step(action)
    pass


if __name__ == "__main__":
    train(get_args())
