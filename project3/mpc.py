import sys
import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from arm_dynamics_student import ArmDynamicsStudent
from robot import Robot
from arm_gui import ArmGUI, Renderer
import argparse
import time
import math
import torch

np.set_printoptions(suppress=True)
np.random.seed(0)
torch.manual_seed(0)

def get_cost(goal, states, dynamics):
    cost = 0

    for i in range(len(states)):
        state = states[i]
        ang_vel = np.linalg.norm(state[dynamics.num_links:])
        pos_ee = dynamics.compute_fk(state)
        cost += np.linalg.norm(goal - pos_ee) ** 2

        if cost < 0.685 and ang_vel > 1.0:
            cost *= 1.2
        elif cost < 0.03 and ang_vel > 0.285:
            cost *= 1.5
        elif ang_vel > 4.95:
            cost *= 5

    return cost


def get_trajectory(action, state, dynamics, cost_range):
    states = [state]

    for i in range(cost_range - 1):
        next_state = dynamics.advance(states[-1], action)
        states.append(next_state)

    return states


class MPC:

    def __init__(self, ):
        # Define other parameters here
        self.control_horizon = 10
        self.cost_range = int(self.control_horizon * 5.5)

    def compute_action(self, dynamics, state, goal, action):

        # if isinstance(dynamics, ArmDynamicsTeacher):
        #     raise NotImplementedError
        # code to skip teacher models
        # Put your code here. You must return an array of shape (num_links, 1)
        current_action = action
        current_state = state
        best_action = current_action
        delta_torque = 31.91
        delta_torque *= 0.01

        for k in range(self.control_horizon):
            for i in range(dynamics.num_links):
                delta_action = np.zeros((dynamics.num_links, 1))
                delta_action[i, 0] = delta_torque
                costs = []
                actions = [best_action + delta_action, best_action - delta_action, best_action]

                for j in range(len(actions)):
                    trajectory = get_trajectory(actions[j], current_state, dynamics, self.cost_range)
                    costs.append(get_cost(goal, trajectory, dynamics))

                best = costs.index(min(costs))
                best_action = np.array(actions[best]).reshape(dynamics.num_links, 1)

        return best_action


def main(args):
    # Arm
    arm = Robot(
        ArmDynamicsTeacher(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )
    )

    # Dynamics model used for control
    if args.model_path is not None:
        dynamics = ArmDynamicsStudent(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )
        dynamics.init_model(args.model_path, args.num_links, device=torch.device("cpu"))
    else:
        # Perfectly accurate model dynamics
        dynamics = ArmDynamicsTeacher(
            num_links=args.num_links,
            link_mass=args.link_mass,
            link_length=args.link_length,
            joint_viscous_friction=args.friction,
            dt=args.time_step,
        )

    # Controller
    controller = MPC()

    # Control loop
    arm.reset()
    action = np.zeros((arm.dynamics.get_action_dim(), 1))
    goal = np.zeros((2, 1))
    goal[0, 0] = args.xgoal
    goal[1, 0] = args.ygoal
    arm.goal = goal

    if args.gui:
        renderer = Renderer()
        time.sleep(0.25)

    dt = args.time_step
    k = 0
    while True:
        t = time.time()
        arm.advance()
        if args.gui:
            renderer.plot([(arm, "tab:blue")])
        k += 1
        time.sleep(max(0, dt - (time.time() - t)))
        if k == controller.control_horizon:
            state = arm.get_state()
            action = controller.compute_action(dynamics, state, goal, action)
            arm.set_action(action)
            k = 0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_links", type=int, default=3)
    parser.add_argument("--link_mass", type=float, default=0.1)
    parser.add_argument("--link_length", type=float, default=1)
    parser.add_argument("--friction", type=float, default=0.1)
    parser.add_argument("--time_step", type=float, default=0.01)
    parser.add_argument("--time_limit", type=float, default=5)
    parser.add_argument("--gui", action="store_const", const=True, default=False)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--xgoal", type=float, default=-1.4)
    parser.add_argument("--ygoal", type=float, default=-1.4)
    return parser.parse_args()


if __name__ == '__main__':
    main()