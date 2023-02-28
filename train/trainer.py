import torch
from torch import optim
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from datetime import datetime
import os
from typing import List, Tuple, Optional
from tqdm import tqdm

from networks.unet import UNet
from data.dataset import minmax_scale, chunk_and_cat
from diffusion import v_loss, v_sample
from train.learning_rate import LinearLR

class SimpleTrainer:
    """
    Manages the state of the network and its optimizer for training and sampling.
    For purely paired image-to-image training.
    """
    def __init__(
        self,
        train_dataset,
        valid_dataset,
        net_config: dict,
        optimizer_config: dict,
        lr_scheduler_config: dict,
        batch_size: int,
        val_batch_size = 4,
        max_train_steps = 100000,
        repeat_noise = 1,
        sample_eta = 0.0,
        loss_scales=1,
        checkpoint_every=5000,
        sample_steps=1000,
        x0_scale=1.,
        normalize_x=True,
        out_dir_name: str = 'training',
        dynamic_thresholding_quantile = 1.0,
        sample_freeze_dims: Optional[Tuple[int]] = None
    ):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.step = 0
        self.val_batch_size = val_batch_size
        self.max_train_steps = max_train_steps
        self.loss_scales = loss_scales
        self.validate_every = checkpoint_every
        self.sample_eta = sample_eta
        self.repeat_noise = repeat_noise
        self.sample_steps = sample_steps
        self.x0_scale = x0_scale
        self.normalize_x = normalize_x
        self.dynamic_thresholding_quantile = dynamic_thresholding_quantile
        self.sample_freeze_dims = sample_freeze_dims
        self.in_channels = train_dataset[0].size(0)
        self.image_size = train_dataset[0].size(1)

        self.net = UNet(**net_config).to(self.device)
        self.out_dir = out_dir=f'./runs/{out_dir_name}/train_at_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        os.makedirs(out_dir)

        self.writer = SummaryWriter(log_dir=out_dir)
        self.scaler = GradScaler()

        self.train_data = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.validation_data = DataLoader(valid_dataset, batch_size=val_batch_size, shuffle=True)
        self.optimizer = optim.Adam(**{'params': self.net.parameters()}, **optimizer_config)
        self.lr_scheduler = LinearLR(**{**{'optimizer': self.optimizer}, **lr_scheduler_config})


    def train_step(self, x, c):
        self.lr_scheduler.update(self.step)
        self.optimizer.zero_grad()
        loss = v_loss(self.net, x, c, self.loss_scales, self.x0_scale, self.normalize_x)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        clip_grad_value_(self.net.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.writer.add_scalar('Loss/train', loss, self.step)
        self.writer.add_scalar('lr/train', self.lr_scheduler.get(), self.step)


    def train(self):
        while True:
            for i, x in tqdm(enumerate(self.train_data)):
                self.train_step(x.to(self.device), None)

                if (self.step + 1) % self.validate_every == 0:
                    self.save_checkpoint()
                    self.unfrozen_sampling()

                self.step += 1
                if(self.step > self.max_train_steps):
                    print(f'Reached final step {self.step}, training ended.')
                    return


    @torch.no_grad()
    def save_checkpoint(self):
        checkpoint_dir = os.path.join(self.out_dir, f'checkpoint_{self.step}')
        os.makedirs(checkpoint_dir)

        torch.save([
            self.net.state_dict(),
            self.optimizer.state_dict(),
            self.lr_scheduler.state_dict()
        ], os.path.join(checkpoint_dir, "ckpt_{}.pth".format(self.step)))


    def save_samples(self, sample_intermediates: List[torch.Tensor], dir_title: str = 'samples'):
        save_dir=os.path.join(self.out_dir, f"{self.step}_{dir_title}")
        os.makedirs(save_dir)
        for step_i, sample_batch in enumerate(sample_intermediates):
            for sample_i, sample in enumerate(sample_batch):
                out_image = chunk_and_cat(minmax_scale(sample.cpu()))
                out_image.save(f"{save_dir}/{sample_i}_{step_i}.jpeg")


    @torch.no_grad()
    def unfrozen_sampling(self, intermediates_every=200, save_to_files=True):
        sample_intermediates = v_sample(
            self.net, 
            x0=torch.zeros(self.val_batch_size, self.in_channels, self.image_size, self.image_size).to(self.device),
            steps=self.sample_steps,
            normalize_x=self.normalize_x,
            eta=self.sample_eta,
            dynamic_thresholding_quantile=self.dynamic_thresholding_quantile,
            intermediates_every=intermediates_every)
        
        if save_to_files:
            self.save_samples(sample_intermediates, 'unfrozen')
        
        return sample_intermediates
    

    @torch.no_grad()
    def frozen_sampling(self, intermediates_every=200, save_to_files=True):
        x = next(self.validation_data)

        sample_intermediates = v_sample(
            self.net, 
            x0=x.to(self.device),
            steps=self.sample_steps,
            normalize_x=self.normalize_x,
            eta=self.sample_eta,
            dynamic_thresholding_quantile=self.dynamic_thresholding_quantile,
            intermediates_every=intermediates_every,
            freeze_dims=self.sample_freeze_dims)
        
        if save_to_files:
            self.save_samples(sample_intermediates, 'frozen')
        
        return sample_intermediates
    








        