import torch
import math

class RandomPolicy(torch.nn.Module):
    def __init__(self,
                 action_channel: int,
                 height: int, width: int,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.action_space = action_channel*height*width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[:-3]
        return torch.randint(0, self.action_space, batch_size)

class GreedyQMax(torch.nn.Module):
    """
    Bootstrap interface for a module that generates a qmap
    """
    def __init__(self,
                 map_dims: torch.Size,
                 qmap_module: torch.nn.Module,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.map_dims = map_dims
        self.q_mapper = qmap_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # query for qmap
        qmap: torch.Tensor = self.q_mapper(x)
        # select best action from map
        batch_shape = qmap.shape[:-len(self.map_dims)]
        qmap = qmap.view(*batch_shape, math.prod(self.map_dims))
        action = qmap.argmax(dim=-1)

        return action

