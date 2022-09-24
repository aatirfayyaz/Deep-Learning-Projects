import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import math
from arm_gui import Renderer
from arm_dynamics import ArmDynamics
from robot import Robot
import time
import argparse


class ArmEnv(gym.Env):

    # ---------- IMPLEMENT YOUR ENVIRONMENT HERE ---------------------
    @staticmethod
    def cartesian_goal(radius, angle):
        return radius * np.array([np.cos(angle), np.sin(angle)]).reshape(-1, 1)

    @staticmethod
    def random_goal():
        radius_max = 2.0
        radius_min = 1.5
        angle_max = 0.5
        angle_min = -0.5
        radius = (radius_max - radius_min) * np.random.random_sample() + radius_min
        angle = (angle_max - angle_min) * np.random.random_sample() + angle_min
        angle -= np.pi / 2
        return ArmEnv.cartesian_goal(radius, angle)

    def __init__(self, arm):
        super(ArmEnv, self).__init__()
        self.arm = arm  # DO NOT modify
        self.goal = None  # Used for computing observation
        self.np_random = np.random  # Use this for random numbers, as it will be seeded appropriately

        obs_low = np.array([0, 0, -5 * math.pi, -5 * math.pi, -2.5, -2.5, -2.5, -2.5], dtype=np.float32)
        obs_high = np.array([2 * math.pi, 2 * math.pi, 5 * math.pi, 5 * math.pi, 2.5, 2.5, 2.5, 2.5], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        # State dim + EE position + Goal

        actions_low = np.array([-2, -2], dtype=np.float32)
        actions_high = -actions_low
        self.action_space = spaces.Box(low=actions_low, high=actions_high, dtype=np.float32)
        # Setting max and min torques as +/- 2 Nm for each link

        # Fill in the rest of this function as needed
        self.num_steps = 0
        # self.renderer = Renderer()

    # We will be calling this function to set the goal for your arm during testing.
    def set_goal(self, goal):
        self.goal = goal
        self.arm.goal = goal

    # For repeatable stochasticity
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Fill in any additional functions you might need
    def reset(self, goal=None):
        self.arm.reset()
        if goal is None:
            self.goal = ArmEnv.random_goal()
        else:
            self.goal = goal
        self.arm.goal = self.goal
        self.num_steps = 0
        pos_ee = self.arm.dynamics.compute_fk(self.arm.get_state())
        observation = np.append(self.arm.get_state(), self.goal).squeeze()
        observation = np.append(observation, pos_ee).squeeze().reshape(-1, ).astype(np.float32)
        return observation

    def step(self, action=None):
        self.num_steps += 1
        self.arm.set_action(action)
        self.arm.advance()
        new_state = self.arm.get_state()
        pos_ee = self.arm.dynamics.compute_fk(new_state)
        distance = np.linalg.norm(pos_ee - self.goal)
        vel_ee = np.linalg.norm(self.arm.dynamics.compute_vel_ee(new_state))
        reward = - distance ** 2  # - vel_ee ** 2

        done = False
        if self.num_steps >= 200:
            done = True
        info = dict(pos_ee=pos_ee, vel_ee=vel_ee, success=True)

        observation = np.append(new_state, self.goal).squeeze()
        observation = np.append(observation, pos_ee).squeeze()

        return observation, reward, done, info

    def render(self, mode="human"):
        pass
