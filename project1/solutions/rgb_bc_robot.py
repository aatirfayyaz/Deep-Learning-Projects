from base import RobotPolicy
import numpy as np

class RGBBCRobot(RobotPolicy):

    """ Implement solution for Part3 below """
    def __init__(self):
        from sklearn import discriminant_analysis
        from sklearn import decomposition
        from sklearn import neighbors
        from sklearn import preprocessing

        self.pca = decomposition.PCA(n_components=100, random_state=74)
        self.scale = preprocessing.StandardScaler()
        #self.act = discriminant_analysis.LinearDiscriminantAnalysis()
        self.act = neighbors.KNeighborsClassifier(n_neighbors=5)

    def train(self, data):
        from sklearn.model_selection import train_test_split

        obs = data["obs"]
        actions = data["actions"]
        actions = actions.reshape(actions.shape[0], )

        obs_flatten = obs.reshape(obs.shape[0], -1)
        obs_cleaned = obs_flatten / 255

        obs_transformed = self.pca.fit_transform(obs_cleaned, actions)
        obs_scaled = self.scale.fit_transform(obs_transformed)
        self.act.fit(obs_scaled, actions)

        pass

    def get_actions(self, observations):

        obs_flatten = observations.reshape(observations.shape[0], -1)
        obs_cleaned = obs_flatten / 255

        obs_transformed = self.pca.transform(obs_cleaned)
        obs_scaled = self.scale.transform(obs_transformed)
        actions_pred = self.act.predict(obs_scaled)

        return actions_pred
