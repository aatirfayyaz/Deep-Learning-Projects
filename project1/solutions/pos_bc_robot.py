from base import RobotPolicy
import numpy as np

class POSBCRobot(RobotPolicy):
    
    """ Implement solution for Part 2 below """
    def __init__(self):
        from sklearn import discriminant_analysis
        self.act = discriminant_analysis.LinearDiscriminantAnalysis()

    def train(self, data):
        from sklearn.model_selection import train_test_split

        obs = data["obs"]
        actions = data["actions"]
        actions = actions.reshape(actions.shape[0], )
        obs_flatten = obs.reshape(obs.shape[0], -1)

        train_obs, test_obs, train_actions, test_actions = train_test_split(obs_flatten, actions, test_size=0.1)

        self.act.fit(train_obs, train_actions)

        pass

    def get_actions(self, observations):
        obs_flatten = observations.reshape(observations.shape[0], -1)

        actions_pred = self.act.predict(obs_flatten)

        return actions_pred
