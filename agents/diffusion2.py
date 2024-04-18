"""
An alternative diffusion model using huggingface diffuser lib
"""
import torch
import copy
from diffusers.training_utils import EMAModel
from diffusers.schedulers import SchedulerMixin
from diffusers.optimization import get_scheduler
from typing import Optional

class QMapDiffusion(torch.nn.Module):
    def __init__(self,
                 inp_channel: int, out_channel: int,
                 height: int, width: int,
                 num_diffusion_iter: int,
                 noise_model: torch.nn.Module,
                 conditional_encoder: torch.nn.Module,
                 noise_scheduler: SchedulerMixin,
                 ema: Optional[EMAModel],
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inp_channel = inp_channel
        self.out_channel = out_channel
        self.height = height
        self.width = width
        self.num_diffusion_iter = num_diffusion_iter
        self.noise_model = noise_model
        self.conditional_encoder = conditional_encoder
        self.noise_scheduler = noise_scheduler
        self.noise_scheduler.set_timesteps(self.num_diffusion_iter)
        self.ema = ema
        self.tensor_kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        - Inputs:
            - x: (...,c_in,h,w) tensor
        - Returns:
            - out: (...,c_out,h,w) tensor
        """
        # get sizes
        batch_size = x.shape[:-3]

        # initialize image from Gaussian noise
        q_map = torch.randn(
            (*batch_size, self.out_channel, self.height, self.width),
            device=x.device, dtype=x.dtype)

        # conditional encoding
        cond = self.conditional_encoder(x) # (B, seq, feature_dim)

        # denoising
        for k in self.noise_scheduler.timesteps:

            # predict noise
            noise_pred = self.noise_model(
                sample=q_map,
                timestep=k,
                encoder_hidden_states=cond)

            # inverse diffusion step (remove noise)
            q_map = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=k,
                sample=q_map).prev_sample

        return q_map

    def step_ema(self):
        if self.ema is not None:
            self.ema.step(self.noise_model.parameters())


class QMapDiffTrainerBase:
    """
    Base Qmap trainer for QMapDiffusion
    """
    def __init__(self,
                 diff_model: QMapDiffusion,
                 discount_factor: float,
                 num_training_steps: int,
                 diff_loss: torch.nn.Module = torch.nn.MSELoss(),
                 optim_type=torch.optim.AdamW,
                 optim_params={'lr':1e-4, 'weight_decay':1e-6}
                 ) -> None:
        # policy and target
        self.policy = diff_model
        self.target = copy.deepcopy(diff_model)
        # training obj
        self.optimizer = optim_type(
            params=self.policy.parameters(),
            **optim_params)
        # lr scheduler
        self.lr_scheduler = get_scheduler(
            name='cosine',
            optimizer=self.optimizer,
            num_warmup_steps=500,
            num_training_steps=num_training_steps)
        # q value discount factor
        self.discount_factor = discount_factor
        # losses
        self.diff_loss = diff_loss

    def get_diffusion_loss(self, state_map: torch.Tensor, action_map: torch.Tensor) -> torch.Tensor:
        # get batch_shape and resize
        batch_shape = state_map.shape[:-3]
        state_map = state_map.view(-1, *state_map.shape[-3:])
        action_map = action_map.view(-1, *action_map.shape[-3:])

        # sample noise
        noise = torch.randn(state_map.shape,
                            device=state_map.device,
                            dtype=state_map.dtype)

        # random timesteps
        timesteps = torch.randint(
            0, self.policy.noise_scheduler.num_train_timesteps,
            (state_map.shape[0],), device=state_map.device, dtype=torch.int32)

        # add noise dep on timestep
        noisy_action_map = self.noise_scheduler.add_noise(
            action_map, noise, timesteps)

        # predict the noise residual
        noise_pred = self.policy.noise_model(
            noisy_action_map, noise, timesteps)

        # return loss
        return self.diff_loss(noise_pred, noise).view(*batch_shape)

class TDErrorQMapDiffTrainer(QMapDiffTrainerBase):
    """
    Treats output of policy as a qvalue map
    """
    def __init__(self,
                 diff_model: QMapDiffusion,
                 discount_factor: float,
                 num_training_steps: int,
                 target_transfer_period: int,
                 qmap_loss: torch.nn.Module= torch.nn.SmoothL1Loss(),
                 diff_loss: torch.nn.Module = torch.nn.MSELoss()) -> None:
        super().__init__(diff_model, discount_factor, num_training_steps, diff_loss)
        self.target_policy = copy.deepcopy(self.policy).eval()
        self.target_transfer_period = target_transfer_period
        self.qmap_loss = qmap_loss

    def train(self, training_iter: int,
              state_map: torch.Tensor, action_map: torch.Tensor,
              rewards: torch.Tensor, non_final_next_states: torch.Tensor,
              non_final_state_mask: torch.BoolTensor) -> dict:
        """
        - Inputs:
            - state_map: (B,C,H,W) tensor
            - actions: (B,) tensor
            - rewards: (B,) tensor
            - non_final_next_states: (<=B,C,H,W) tensor
            - non_final_state_mask: (B,) bool tensor
            - logger: optional SumaryWriter
        - Returns:
            - info: dict, any loss or metric info to be tracked
        """

        # standard dqn td error training
        with torch.no_grad():
            target_q_map = rewards + self.discount_factor * non_final_state_mask \
                * self.target_policy(non_final_next_states).detach()
        pred_q_map = self.policy(state_map)

        td_loss: torch.Tensor = self.qmap_loss(pred_q_map, target_q_map)

        # diffusion loss
        diff_loss = self.get_diffusion_loss(state_map, action_map)

        # backpropagation
        loss = td_loss + diff_loss
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        self.policy.step_ema()

        # transfer weights at interval
        if training_iter % self.target_transfer_period == 0:
            self.target_policy.load_state_dict(self.policy.state_dict())

        # losses
        return {'td_loss':td_loss.item(),
                'diff_loss':diff_loss.item(),
                'total_loss':loss.item()}