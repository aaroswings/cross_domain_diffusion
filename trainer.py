import torch
from torch import optim
from torch.cuda.amp import GradScaler
from torch.nn.utils import clip_grad_value_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from datetime import datetime
import os
from typing import Tuple
from tqdm import tqdm

from networks.unet import UNet256
from data.dataset import PairedDataset
from vdiffusion import v_loss, v_paired_sample


class TrainPairedConcat:
    """
    Manages the state of the network and its optimizer for training and sampling.
    For purely paired image-to-image training.
    """
    def __init__(
        self,
        train_roots: Tuple[str, str],
        validation_roots: Tuple[str, str],
        net_config: dict,
        batch_size: int,
        image_size: int = 256,
        lr = 1e-4,
        max_train_steps = 100000,
        sample_eta = 0.0,
        loss_scales=1,
        validate_every=5000
    ):
        if image_size == 256:
            self.net = UNet256(**net_config).to(self.device)
        else:
            raise NotImplementedError("Image sizes of (256) are currently supported")
        
        self.step = 0
        self.image_size = image_size
        self.max_train_steps = max_train_steps
        self.loss_scales = loss_scales
        self.validate_every = validate_every

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.out_dir = out_dir=f'./runs/train_at_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        os.makedirs(out_dir)

        writer = SummaryWriter(log_dir=out_dir)
        self.scaler = GradScaler()

        self.train_data = DataLoader(PairedDataset(*train_roots).to(self.device), batch_size=batch_size, shuffle=True)
        self.validation_data = DataLoader(PairedDataset(*validation_roots).to(self.device), batch_size=batch_size, shuffle=True)
        self.opt = optim.Adam([*self.net.parameters()], lr=lr, betas=(0.9, 0.99))


    def train_step(self, x, c):
        self.opt.zero_grad()
        loss = v_loss(self.net, x, c, repeat_noise=2, loss_scales=self.loss_scales)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        clip_grad_value_(self.net.parameters(), 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        self._log_train_step(loss)


    def train(self):
        while True:
            for i, x in tqdm(enumerate(self.train_data)):
                self.train_step(x, None)

                if self.step % self.validate_every == 0:
                    v_paired_sample(self.net, size=self.image_size, save_dir=os.path.join(self.out_dir, f"{self.step}_samples_unguided"))

                self.step += 1
                if(self.step > self.max_train_steps):
                    print(f'Reached final step {self.step}, training ended.')
                    return


    @torch.no_grad()
    def validation_step(self):
        checkpoint_dir = os.path.join(self.out_dir, f'checkpoint_{self.step}')
        os.makedirs(checkpoint_dir)

        torch.save([
            self.net.state_dict(),
            self.opt.state_dict(),
        ], os.path.join(checkpoint_dir, "ckpt_{}.pth".format(self.step)))



    @torch.no_grad()
    def _log_train_step(self, loss):
        sqsum = 0.0
        for p in self.net.parameters():
            sqsum += (p.grad ** 2).sum().item()
        self.writer.add_scalar("grad_norm", np.sqrt(sqsum), self.step)
        self.writer.add_scalar('Loss/train', loss, self.step)