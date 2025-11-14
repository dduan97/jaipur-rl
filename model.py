from gymnasium import spaces
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

class MaskedPPOModel(TorchModelV2, nn.Module):
    """
    A custom model that handles Dict observations containing an action mask.
    It passes the observation vector to a standard FC network and then uses
    the action mask to filter the action logits. Currently kind of buggy so we ignore the mask.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        model_config = model_config or {
            "fcnet_hiddens": [256, 1024],  
            "fcnet_activation": "relu",  
        }

        print("Initializing with model config", model_config)

        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        orig_space = getattr(obs_space, "original_space", obs_space)
        assert (
            isinstance(orig_space, spaces.Dict)
            and "action_mask" in orig_space.spaces
            and "observation" in orig_space.spaces
        )
        self._feature_space = orig_space["observation"]

        # Create a standard fully connected network for feature extraction
        self.internal_model = TorchFC(
            self._feature_space,
            action_space,
            num_outputs,
            model_config,
            name + "_internal",
        )
        self.mask_inf = -1e10  # A large negative value to mask invalid actions
        self._last_obs = None

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs"]
        assert isinstance(obs, dict)
        assert "action_mask" in obs
        action_mask = obs["action_mask"]

        # Get the unmasked logits and the value function output
        logits, value_out = self.internal_model(
            {"obs": obs["observation"]}, state, seq_lens
        )
        inf_mask = torch.log(
            torch.maximum(
                action_mask,
                torch.tensor(1e-10, dtype=torch.float32, device=logits.device),
            )
        )
        masked_logits = logits + inf_mask
        return masked_logits, state

    def value_function(self):
        # The value function output is stored in the internal model
        return self.internal_model.value_function()

