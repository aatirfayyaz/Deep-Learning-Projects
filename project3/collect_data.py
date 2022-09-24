import numpy as np
from arm_dynamics_teacher import ArmDynamicsTeacher
from robot import Robot
import argparse
import os
import time
import math

np.set_printoptions(suppress=True)
np.random.seed(0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_links', type=int, default=3)
    parser.add_argument('--link_mass', type=float, default=0.1)
    parser.add_argument('--link_length', type=float, default=1)
    parser.add_argument('--friction', type=float, default=0.1)
    parser.add_argument('--time_step', type=float, default=0.01)
    parser.add_argument('--time_limit', type=float, default=5)
    parser.add_argument('--save_dir', type=str, default='dataset')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # Teacher arm
    dynamics_teacher = ArmDynamicsTeacher(
        num_links=args.num_links,
        link_mass=args.link_mass,
        link_length=args.link_length,
        joint_viscous_friction=args.friction,
        dt=args.time_step
    )
    arm_teacher = Robot(dynamics_teacher)

    # ---
    # You code goes here. Replace the X, and Y by your collected data
    # Control the arm to collect a dataset for training the forward dynamics.
    X = []
    Y = []
    # ---

    for i in range(0, 1000000):
        if i < 500000:
            delta_torque = np.random.rand() * 0.5
        else:
            delta_torque = np.random.rand() * -0.5
        if i % 2 == 0:
            delta_torque2 = np.random.rand() * 0.5
        elif i % 5 == 0:
            delta_torque2 = 0
        else:
            delta_torque2 = np.random.rand() * -0.5

        # if torque == 0:
        #     continue

        start_up = np.zeros((arm_teacher.dynamics.get_state_dim(), 1))
        if i < 500000:
            start_up[0] = np.random.rand() * -math.pi / 2
        else:
            start_up[0] = np.random.rand() * -math.pi / 2 - math.pi / 2
        print('Collecting data @ ', start_up[0], 'with torques: ', delta_torque, delta_torque2)
        arm_teacher.set_state(start_up)
        action = start_up.copy()
        action[0] = delta_torque
        action[1] = delta_torque2
        arm_teacher.set_action(action)
        arm_teacher.set_t(0)

        #
        dt = args.time_step
        time_limit = args.time_limit

        while arm_teacher.get_t() < dt:
            t = time.time()

            # For X
            current_state = arm_teacher.get_state()
            current_action = np.zeros((arm_teacher.dynamics.get_action_dim(), 1))
            current_action[0] = delta_torque
            current_action[1] = delta_torque2
            X.append(np.concatenate((current_state, current_action), axis=0))

            # For Y
            arm_teacher.advance()
            new_state = arm_teacher.get_state()
            Y.append(new_state)
            time.sleep(max(0, dt - (time.time() - t)))
    # ---

    X = np.hstack(X)
    Y = np.hstack(Y)

    print('X shape:', X.shape, 'Y shape:', Y.shape)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    np.save(os.path.join(args.save_dir, 'X.npy'), X)
    np.save(os.path.join(args.save_dir, 'Y.npy'), Y)
