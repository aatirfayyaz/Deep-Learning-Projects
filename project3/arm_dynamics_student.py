from arm_dynamics_base import ArmDynamicsBase
import numpy as np
import torch
from models import build_model

np.random.seed(0)
torch.manual_seed(0)

class ArmDynamicsStudent(ArmDynamicsBase):
    def init_model(self, model_path, num_links, time_step, device):
        # ---
        # Your code hoes here
        # Initialize the model loading the saved model from provided model_path
        self.model = build_model(num_links, time_step)
        self.model_loaded = True
        saved_model = torch.load(model_path, map_location=device)
        self.model.load_state_dict(saved_model)
        # ---

    def dynamics_step(self, state, action, dt):
        if self.model_loaded:
            # ---
            # Your code goes here
            # Use the loaded model to predict new state given the current state and action
            self.model.eval()
            current_state = np.concatenate((state, action)).T
            new_state = self.model.forward(torch.FloatTensor(current_state)).detach().numpy()
            return new_state.T
            # ---

        else:
            return state

