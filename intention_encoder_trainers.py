"""
Trainer for intention encoders
"""

import torch
from typing import Optional


class IntEncoderTrainer:
    def __init__(self,
                 intention_net: torch.nn.Module,
                 loss_fn: torch.nn.Module = torch.nn.BCEWithLogitsLoss(),
                 optim_type=torch.optim.SGD,
                 optim_params={'lr':1e-4}) -> None:
        self.intention_net = intention_net
        self.loss_fn = loss_fn
        self.optimizer = optim_type(
            self.intention_net.parameters(),
            **optim_params)

    def train(self, training_iter: int,
              raw_states: torch.Tensor, target_int: torch.Tensor) -> dict:
        """
        Want the network to guess the group intention
        - Inputs:
            - training_iter: int
            - raw_states: (...,c,h,w) tensor
            - target_int: (...,1,h,w) tensor
            - logger: optional tensorboard summary writer
        - Returns:
            - info: dict, any loss or metrics being tracked
        """
        pred_int = self.intention_net(raw_states)
        loss: torch.Tensor = self.loss_fn(pred_int, target_int)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {'int_loss':loss.item()}
