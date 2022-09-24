from base import Regressor
import numpy as np

class PositionRegressor(Regressor):

    """ Implement solution for Part 1 below  """
    def __init__(self):
        from sklearn import kernel_ridge
        self.reg = kernel_ridge.KernelRidge()

    def train(self, data):
        from sklearn.model_selection import train_test_split

        obs = data["obs"]
        info = data["info"]

        obs_flatten = obs.reshape(obs.shape[0], -1)
        obs_cleaned = obs_flatten / 255

        agent_pos = []

        for i in range(0, 500):
            agent_pos.append(info[i]["agent_pos"])

        train_obs, test_obs, train_agent_pos, test_agent_pos = train_test_split(obs_cleaned, agent_pos, test_size=0.01)

        self.reg.fit(train_obs, train_agent_pos)
        pass

    def predict(self, Xs):
        obs_flatten = Xs.reshape(Xs.shape[0], -1)
        obs_cleaned = obs_flatten / 255

        agent_pos_pred = self.reg.predict(obs_cleaned)

        return agent_pos_pred
